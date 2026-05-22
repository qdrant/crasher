use anyhow::Result;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::WriteOrdering;
use rand::Rng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::args::Args;
use crate::checker::{
    check_count_scroll_parity, check_filter_bool_index, check_filter_null_index,
    check_mandatory_payload_keys_present, check_optimizer_status,
    check_payload_indexes_consistency, check_points_consistency, check_search_result,
};
use crate::client::{
    create_collection, create_collection_snapshot, create_ephemeral_named_vector,
    create_payload_indexes, delete_collection, delete_collection_snapshot,
    delete_ephemeral_named_vector, delete_points, dump_segments_diagnostic, get_collection_info,
    get_exact_points_count, get_telemetry, insert_points_batch, list_collection_snapshots,
    query_batch_points, restore_collection_snapshot, set_payload,
};
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::{Cancelled, Client, Invariant};
use crate::generators::TestNamedVectors;

/// Throwaway named vector name used to exercise the dynamic vector-schema mutation
/// code path. The name is fixed (not unique per run) so any leftover from a crashed
/// previous run is wiped when the workload deletes & recreates the collection on
/// re-entry.
const EPHEMERAL_VECTOR_NAME: &str = "ephemeral-vec";

pub struct Workload {
    collection_name: String,
    test_named_vectors: TestNamedVectors,
    query_count: u32,
    points_count: u32,
    vec_dim: u32,
    payload_count: u32,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
    crash_lock: Arc<tokio::sync::Mutex<()>>, // take lock to prevent crash
    rng_seed: u64,
    // Largest point id written and confirmed by Qdrant
    // Confirmation means, that API responded with `Completed` status for upsert request
    max_confirmed_point_id: Arc<AtomicI64>,
}

impl Workload {
    pub fn new(
        collection_name: &str,
        stopped: Arc<AtomicBool>,
        crash_lock: Arc<tokio::sync::Mutex<()>>,
        duplication_factor: u32,
        points_count: u32,
        vec_dim: u32,
        rng_seed: u64,
    ) -> Self {
        let collection_name = collection_name.to_string();
        let payload_count = 5; // hardcoded
        let query_count = 2; //hardcoded
        let test_named_vectors = TestNamedVectors::new(duplication_factor, vec_dim);
        let write_ordering = None; // default
        Self {
            collection_name,
            test_named_vectors,
            query_count,
            points_count,
            vec_dim,
            payload_count,
            write_ordering,
            stopped,
            crash_lock,
            rng_seed,
            max_confirmed_point_id: Arc::new(AtomicI64::new(-1)),
        }
    }

    fn reset_max_confirmed_point_id(&self) {
        self.max_confirmed_point_id.store(-1, Ordering::Relaxed);
    }

    fn get_max_confirmed_point_id(&self) -> Option<u64> {
        let max_confirmed_id = self.max_confirmed_point_id.load(Ordering::Relaxed);
        if max_confirmed_id < 0 {
            None
        } else {
            Some(max_confirmed_id as u64)
        }
    }

    fn set_max_confirmed_point_id(&self, point_id: u64) {
        self.max_confirmed_point_id
            .store(point_id as i64, Ordering::Relaxed);
    }
}

impl Workload {
    pub async fn work(
        &self,
        client: &Qdrant,
        http_client: &reqwest::Client,
        args: Arc<Args>,
        rng: &mut impl Rng,
    ) {
        loop {
            if self.stopped.load(Ordering::Relaxed) {
                break;
            }
            let start = Instant::now();
            let run = self.run(client, http_client, args.clone(), rng).await;
            match run {
                Ok(()) => {
                    log::info!("Workload finished in {:?}", start.elapsed());
                }
                Err(Cancelled) => {
                    log::info!("Workload cancelled after {:?}", start.elapsed());
                    break;
                }
                Err(Invariant(msg)) => {
                    log::error!(
                        "Workload run failed due to an invariant violation! (rng_seed:{})\n{msg}",
                        self.rng_seed
                    );
                    // send stop signal to the main thread
                    self.stopped.store(true, Ordering::Relaxed);
                    log::info!("Stopping the workload...");
                    break;
                }
                Err(Client(error)) => {
                    // turn client error into hard error if it is a server error (a bit hacky)
                    let debug_error = format!("{error:?}");
                    if debug_error.contains("Service internal error") {
                        log::error!("Workload run failed due to a server error!\n{error:?}");
                        // send stop signal to the main thread
                        self.stopped.store(true, Ordering::Relaxed);
                        break;
                    }
                    log::warn!(
                        "Workload run failed due to client error - resuming soon\n{error:?}"
                    );
                    // no need to hammer the server while it restarts
                    sleep(Duration::from_secs(3)).await;
                }
            }
        }
    }

    pub async fn run(
        &self,
        client: &Qdrant,
        http_client: &reqwest::Client,
        args: Arc<Args>,
        rng: &mut impl Rng,
    ) -> Result<(), CrasherError> {
        log::info!("Starting workload...");
        // Do not crash during setup and validation phase
        let crash_guard = self.crash_lock.lock().await;
        if client.collection_exists(&self.collection_name).await? {
            // Validate and clean up existing data
            let current_count = get_exact_points_count(client, &self.collection_name).await?;
            let confirmed_point_count = match self.get_max_confirmed_point_id() {
                None => 0,
                Some(point_id) => point_id + 1,
            };
            let checkable_points =
                std::cmp::min(current_count as u64, confirmed_point_count) as usize;
            if checkable_points != 0 {
                log::info!(
                    "Run: previous data consistency check ({confirmed_point_count} / {current_count} points)"
                );
                self.check_data_consistency(
                    client,
                    http_client,
                    checkable_points,
                    args.only_sparse,
                    "post-crash-check",
                    args.http_port,
                )
                .await?;
            }

            // Always delete all points before checking snapshots
            log::info!("Run: delete all existing points");
            delete_points(client, &self.collection_name).await?;
            self.reset_max_confirmed_point_id();

            // Validate existing collection snapshots
            self.validate_snapshots(client, http_client, &args).await?;

            // Delete collection to not accumulate hidden state
            delete_collection(client, &self.collection_name).await?;
        }

        log::info!("Creating workload collection");
        self.reset_max_confirmed_point_id();
        create_collection(
            client,
            &self.collection_name,
            &self.test_named_vectors,
            args.clone(),
        )
        .await?;
        create_payload_indexes(client, &self.collection_name).await?;
        let _collection_info = get_collection_info(client, &self.collection_name).await?; // debug

        // Enable crashing to happen starting from here
        drop(crash_guard);

        log::info!("Run: insert points");
        insert_points_batch(
            client,
            &self.collection_name,
            self.points_count,
            self.vec_dim,
            0,    // no payload at first
            true, // add mandatory payload keys
            args.only_sparse,
            &self.test_named_vectors,
            None,
            self.stopped.clone(),
            rng,
            |inserted_id| self.set_max_confirmed_point_id(inserted_id),
        )
        .await?;

        // All written points should be confirmed here, as insert_points_batch waits for completion
        let points_count = get_exact_points_count(client, &self.collection_name).await?;
        log::info!("Run: post-point-insert data consistency check");
        self.check_data_consistency(
            client,
            http_client,
            points_count,
            args.only_sparse,
            "post-insert-check",
            args.http_port,
        )
        .await?;

        // Starts snapshotting process as the data was ingested properly
        log::info!("Run: trigger collection snapshot in the background");

        // Graceful shutdown of the snapshotting task
        let is_run_finished = Arc::new(AtomicBool::new(false));
        let finish = is_run_finished.clone();
        let snapshotting_handle = self.trigger_continuous_snapshotting(client, finish);

        // Mutate vector schema while the workload is running. The ephemeral vector is left
        // un-backfilled; existing points keep their existing named vectors, and snapshots
        // taken during this window include the extended schema. A crash anywhere up to
        // the matching delete below is recovered by the collection-exists branch on
        // re-entry (consistency check tolerates extra names, then delete_collection wipes).
        log::info!("Run: create ephemeral named vector '{EPHEMERAL_VECTOR_NAME}'");
        create_ephemeral_named_vector(
            client,
            &self.collection_name,
            EPHEMERAL_VECTOR_NAME,
            self.vec_dim,
        )
        .await?;

        log::info!("Run: set payload");
        for point_id in 1..self.points_count {
            if self.stopped.load(Ordering::Relaxed) {
                return Err(Cancelled);
            }
            if point_id % 2 == 0 {
                // skip half of the points
                continue;
            }
            set_payload(
                client,
                &self.collection_name,
                point_id as u64,
                self.payload_count,
                self.write_ordering,
                rng,
            )
            .await?;
        }

        // Set-payload should not remove any points, so count should remain the same
        log::info!("Run: post-payload-insert data consistency check");
        self.check_data_consistency(
            client,
            http_client,
            points_count,
            args.only_sparse,
            "post-set-payload-check",
            args.http_port,
        )
        .await?;

        log::info!("Run: query random vectors");
        for _i in 0..self.query_count {
            if self.stopped.load(Ordering::Relaxed) {
                return Err(Cancelled);
            }
            let results = query_batch_points(
                client,
                &self.collection_name,
                &self.test_named_vectors,
                args.only_sparse,
                self.vec_dim,
                self.payload_count,
                true,
                10,
                rng,
            )
            .await?;
            check_search_result(&results)?;
        }

        log::info!("Run: get full telemetry");
        let _telemetry = get_telemetry(http_client, args.http_port).await?;

        log::info!("Run: delete ephemeral named vector '{EPHEMERAL_VECTOR_NAME}'");
        delete_ephemeral_named_vector(client, &self.collection_name, EPHEMERAL_VECTOR_NAME).await?;

        // Stop ongoing snapshotting task
        is_run_finished.store(true, Ordering::Relaxed);
        match snapshotting_handle.await {
            Ok(Ok(())) => (),
            Err(_join_error) => (), // ignore JoinError
            Ok(Err(snapshot_err)) => return Err(snapshot_err), // capture failed snapshot
        }

        Ok(())
    }

    /// Validate data consistency from the client side
    /// - all point ids exist
    /// - all named vectors exist
    /// - all mandatory payload keys exist
    async fn check_data_consistency(
        &self,
        client: &Qdrant,
        http_client: &reqwest::Client,
        current_count: usize,
        only_sparse: bool,
        context: &str,
        http_port: u32,
    ) -> Result<(), CrasherError> {
        let mut errors = Vec::new();

        let expected_names = self.test_named_vectors.all_expected_names(only_sparse);
        // scroll-based traversal: well-formedness + missing / phantom / UUID / duplicate ids
        match check_points_consistency(
            &self.collection_name,
            client,
            current_count,
            self.points_count as u64,
            &expected_names,
        )
        .await
        {
            Err(Invariant(e)) => errors.push(format!("*Inconsistent storage*\n{e}")),
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        // check mandatory timestamp payload key via null index
        match check_filter_null_index(&self.collection_name, client, current_count).await {
            Err(Invariant(e)) => errors.push(format!("*Inconsistent Null Index*\n{e}")),
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        // check mandatory bool payload key via match query
        match check_filter_bool_index(&self.collection_name, client, current_count).await {
            Err(Invariant(e)) => errors.push(format!("*Inconsistent Bool Index*\n{e}")),
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        // check payload indexes consistency (all indexes agree on the expected distribution)
        match check_payload_indexes_consistency(&self.collection_name, client).await {
            Err(Invariant(e)) => errors.push(format!("*Inconsistent Payload Indexes*\n{e}")),
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        // count(filter) vs scroll(filter) parity on stable filters
        match check_count_scroll_parity(&self.collection_name, client).await {
            Err(Invariant(e)) => errors.push(format!("*Count/scroll filter parity*\n{e}")),
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        // every mandatory payload key must be set on every point (per-index oracle)
        match check_mandatory_payload_keys_present(&self.collection_name, client).await {
            Err(Invariant(e)) => errors.push(format!("*Mandatory payload keys*\n{e}")),
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        // check optimizer status is not red
        match check_optimizer_status(&self.collection_name, client).await {
            Err(Invariant(e)) => errors.push(format!("*Optimizer status error*\n{e}")),
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        if errors.is_empty() {
            return Ok(());
        }
        let full_report = errors.join("\n---\n");
        let segments_diag = dump_segments_diagnostic(http_client, &self.collection_name, http_port)
            .await
            .unwrap_or_else(|e| format!("(failed to collect per-segment diagnostic: {e})\n"));
        Err(Invariant(format!(
            "Data inconsistencies found out of {current_count} points ({context}):\n\
             {full_report}\n\
             {segments_diag}"
        )))
    }

    fn trigger_continuous_snapshotting(
        &self,
        client: &Qdrant,
        finish: Arc<AtomicBool>,
    ) -> JoinHandle<Result<(), CrasherError>> {
        let collection_name = self.collection_name.clone();
        let client_snapshot = client.clone();
        let stopped = self.stopped.clone();
        tokio::spawn(async move {
            while !stopped.load(Ordering::Relaxed) && !finish.load(Ordering::Relaxed) {
                if let Err(err) =
                    churn_collection_snapshot(&client_snapshot, &collection_name).await
                {
                    log::warn!("Snapshotting failed {err}");
                    // stop at first snapshot error silently
                    return Err(err);
                }
            }
            Ok(())
        })
    }

    /// Validate exiting snapshots by sequentially restoring and checking data consistency
    async fn validate_snapshots(
        &self,
        client: &Qdrant,
        http_client: &reqwest::Client,
        args: &Args,
    ) -> Result<(), CrasherError> {
        let snapshots = list_collection_snapshots(client, &self.collection_name).await?;
        if snapshots.is_empty() {
            return Ok(());
        }

        log::info!("Found {} local collection snapshots", snapshots.len());
        for snapshot in &snapshots {
            let snapshot_name = &snapshot.name;
            // Make sure everything was deleted before the snapshot restore
            let after_delete_count = get_exact_points_count(client, &self.collection_name).await?;
            if after_delete_count != 0 {
                return Err(Invariant(format!(
                    "Expected zero points before restoring snapshot but got {after_delete_count}"
                )));
            }

            if args.skip_snapshot_restore {
                log::info!("Run: skipping restoring snapshot '{snapshot_name}'");
                delete_collection_snapshot(client, &self.collection_name, snapshot_name).await?;
                continue;
            }

            log::info!("Run: restoring snapshot '{snapshot_name}'");
            match restore_collection_snapshot(
                &self.collection_name,
                snapshot_name,
                snapshot.checksum(),
                http_client,
                args.http_port,
            )
            .await
            {
                Ok(()) => {}
                // Snapshots captured during the ephemeral-vector window have a
                // schema strict-equality check against the current collection.
                // Drop them and move on instead of aborting the chaos test.
                Err(Invariant(msg)) if msg.contains("Snapshot is not compatible") => {
                    log::warn!(
                        "Skipping snapshot '{snapshot_name}' due to schema \
                         incompatibility: {msg}"
                    );
                    delete_collection_snapshot(client, &self.collection_name, snapshot_name)
                        .await?;
                    continue;
                }
                Err(e) => return Err(e),
            }

            let restored_count = get_exact_points_count(client, &self.collection_name).await?;

            if restored_count == 0 {
                log::info!("Snapshot `{snapshot_name}` does not contain any points");
                continue;
            }

            log::info!(
                "Run: data consistency check over {restored_count} points restored for snapshot '{snapshot_name}'"
            );
            self.check_data_consistency(
                client,
                http_client,
                restored_count,
                args.only_sparse,
                &format!("snapshot-restore-check-{snapshot_name}"),
                args.http_port,
            )
            .await?;
            // cleanup
            delete_collection_snapshot(client, &self.collection_name, snapshot_name).await?;
            delete_points(client, &self.collection_name).await?;
            self.reset_max_confirmed_point_id();
        }
        log::info!("All snapshots validated and deleted!");

        Ok(())
    }
}

async fn churn_collection_snapshot(
    client: &Qdrant,
    collection_name: &str,
) -> Result<(), CrasherError> {
    create_collection_snapshot(client, collection_name).await?;
    // keep at most the 2 latest snapshots for testing
    cleanup_old_snapshots(client, collection_name, 2).await?;
    Ok(())
}

async fn cleanup_old_snapshots(
    client: &Qdrant,
    collection_name: &str,
    keep: u32,
) -> Result<(), CrasherError> {
    let mut snapshots = list_collection_snapshots(client, collection_name).await?;
    // sort oldest first (smallest number is the oldest date)
    snapshots.sort_unstable_by_key(|snapshot| snapshot.creation_time.unwrap().seconds);

    // keep and delete rest
    let total_count = snapshots.len();
    if total_count <= keep as usize {
        return Ok(());
    }

    // Take the first N oldest snapshots and delete them
    let delete_count = total_count - keep as usize;
    for snapshot in snapshots.iter().take(delete_count) {
        delete_collection_snapshot(client, collection_name, &snapshot.name).await?;
    }

    Ok(())
}
