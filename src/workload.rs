use anyhow::Result;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    BoolIndexParamsBuilder, DatetimeIndexParamsBuilder, FieldType, FloatIndexParamsBuilder,
    GeoIndexParamsBuilder, IntegerIndexParamsBuilder, KeywordIndexParamsBuilder,
    TextIndexParamsBuilder, TokenizerType, UuidIndexParamsBuilder, WriteOrdering,
};
use rand::Rng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::args::Args;
use crate::checker::{
    check_filter_bool_index, check_filter_null_index, check_points_consistency, check_search_result,
};
use crate::client::{
    create_collection, create_collection_snapshot, create_field_index, delete_collection_snapshot,
    delete_points, get_collection_info, get_exact_points_count, get_telemetry, insert_points_batch,
    list_collection_snapshots, query_batch_points, restore_collection_snapshot, set_payload,
};
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::{Cancelled, Client, Invariant};
use crate::generators::{
    BOOL_PAYLOAD_KEY, DATETIME_PAYLOAD_KEY, FLOAT_PAYLOAD_KEY, GEO_PAYLOAD_KEY,
    INTEGER_PAYLOAD_KEY, KEYWORD_PAYLOAD_KEY, MANDATORY_PAYLOAD_BOOL_KEY,
    MANDATORY_PAYLOAD_TIMESTAMP_KEY, TEXT_PAYLOAD_KEY, TestNamedVectors, UUID_PAYLOAD_KEY,
};

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
        let payload_count = 5; // hardcoded
        let query_count = 2; //hardcoded
        let test_named_vectors = TestNamedVectors::new(duplication_factor, vec_dim);
        let write_ordering = None; // default
        Workload {
            collection_name: collection_name.to_string(),
            test_named_vectors,
            query_count,
            points_count,
            vec_dim,
            payload_count,
            write_ordering,
            stopped,
            crash_lock,
            rng_seed,
        }
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
            let run = self.run(client, http_client, args.clone(), rng).await;
            match run {
                Ok(()) => {
                    log::info!("Workload run finished");
                }
                Err(Cancelled) => {
                    log::info!("Workload run cancelled");
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
                    } else {
                        log::warn!(
                            "Workload run failed due to client error - resuming soon\n{error:?}"
                        );
                        // no need to hammer the server while it restarts
                        sleep(Duration::from_secs(3)).await;
                    }
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
        let start = Instant::now();
        log::info!("Starting workload...");
        // create and populate collection if it does not exist
        if !client.collection_exists(&self.collection_name).await? {
            log::info!("Creating workload collection");
            create_collection(
                client,
                &self.collection_name,
                &self.test_named_vectors,
                args.clone(),
            )
            .await?;

            // create keyword index for the payload
            create_field_index(
                client,
                &self.collection_name,
                KEYWORD_PAYLOAD_KEY,
                FieldType::Keyword,
                KeywordIndexParamsBuilder::default()
                    .is_tenant(true)
                    .on_disk(true),
            )
            .await?;

            // create integer index for the payload
            create_field_index(
                client,
                &self.collection_name,
                INTEGER_PAYLOAD_KEY,
                FieldType::Integer,
                IntegerIndexParamsBuilder::new(true, true)
                    .is_principal(true)
                    .on_disk(true),
            )
            .await?;

            // create float index for the payload
            create_field_index(
                client,
                &self.collection_name,
                FLOAT_PAYLOAD_KEY,
                FieldType::Float,
                FloatIndexParamsBuilder::default()
                    .is_principal(true)
                    .on_disk(true),
            )
            .await?;

            // create geo index for the payload
            create_field_index(
                client,
                &self.collection_name,
                GEO_PAYLOAD_KEY,
                FieldType::Geo,
                GeoIndexParamsBuilder::default().on_disk(true),
            )
            .await?;

            // create text payload index
            create_field_index(
                client,
                &self.collection_name,
                TEXT_PAYLOAD_KEY,
                FieldType::Text,
                TextIndexParamsBuilder::new(TokenizerType::Multilingual)
                    .ascii_folding(true)
                    .snowball_stemmer("english".to_string())
                    .stopwords_language("english".to_string())
                    .phrase_matching(true)
                    .on_disk(true),
            )
            .await?;

            // create boolean index for the payload
            create_field_index(
                client,
                &self.collection_name,
                BOOL_PAYLOAD_KEY,
                FieldType::Bool,
                BoolIndexParamsBuilder::new().on_disk(true),
            )
            .await?;

            // create timestamp index for payload
            create_field_index(
                client,
                &self.collection_name,
                DATETIME_PAYLOAD_KEY,
                FieldType::Datetime,
                DatetimeIndexParamsBuilder::default()
                    .is_principal(true)
                    .on_disk(true),
            )
            .await?;

            // create UUID index for the payload
            create_field_index(
                client,
                &self.collection_name,
                UUID_PAYLOAD_KEY,
                FieldType::Uuid,
                UuidIndexParamsBuilder::default()
                    .is_tenant(true)
                    .on_disk(true),
            )
            .await?;

            // mandatory payload field indices
            create_field_index(
                client,
                &self.collection_name,
                MANDATORY_PAYLOAD_TIMESTAMP_KEY,
                FieldType::Datetime,
                DatetimeIndexParamsBuilder::default()
                    .is_principal(true)
                    .on_disk(true),
            )
            .await?;
            create_field_index(
                client,
                &self.collection_name,
                MANDATORY_PAYLOAD_BOOL_KEY,
                FieldType::Bool,
                BoolIndexParamsBuilder::default().on_disk(true),
            )
            .await?;

            let collection_info = get_collection_info(client, &self.collection_name).await?;
            log::info!("Collection info: {collection_info:#?}");
        }

        // Validate and clean up existing data
        let current_count = get_exact_points_count(client, &self.collection_name).await?;
        if current_count != 0 {
            log::info!("Run: previous data consistency check ({current_count} points)");
            self.data_consistency_check(client, current_count).await?;

            log::info!("Run: delete existing points ({current_count} points)");
            delete_points(client, &self.collection_name, current_count).await?;

            let snapshots = list_collection_snapshots(client, &self.collection_name).await?;
            if !snapshots.is_empty() {
                log::info!("Found {} local collection snapshots", snapshots.len());
                // Do not crash during restore as it would leave a dummy shard behind & make sure we delete snapshots
                let _crash_lock_guard = self.crash_lock.lock().await;
                for snapshot_name in &snapshots {
                    log::info!("Run: restoring snapshot '{snapshot_name}'");
                    restore_collection_snapshot(&self.collection_name, snapshot_name, http_client)
                        .await?;

                    delete_collection_snapshot(client, &self.collection_name, snapshot_name)
                        .await?;
                    let restored_count =
                        get_exact_points_count(client, &self.collection_name).await?;

                    if restored_count == 0 {
                        log::info!("Snapshot `{snapshot_name}` does not contain any points");
                        continue;
                    }

                    log::info!(
                        "Run: data consistency check over {restored_count} points restored for snapshot '{snapshot_name}'"
                    );
                    self.data_consistency_check(client, restored_count).await?;

                    delete_points(client, &self.collection_name, restored_count).await?;
                }
                log::info!("All snapshots validated and deleted!");
            }
        }

        log::info!("Run: trigger collection snapshot in the background");
        let snapshotting_handle = self.trigger_continuous_snapshotting(client);

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
        )
        .await?;

        let points_count = get_exact_points_count(client, &self.collection_name).await?;
        log::info!("Run: post-point-insert data consistency check");
        self.data_consistency_check(client, points_count).await?;

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

        log::info!("Run: post-payload-insert data consistency check");
        self.data_consistency_check(client, points_count).await?;

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
        let _telemetry = get_telemetry(http_client).await?;

        // Stop ongoing snapshotting task
        snapshotting_handle.abort();
        match snapshotting_handle.await {
            Ok(Ok(())) => (),
            Err(_join_error) => (), // ignore JoinError
            Ok(Err(snapshot_err)) => return Err(snapshot_err), // capture failed snapshot
        }

        log::info!("Workload finished in {:?}", start.elapsed());
        Ok(())
    }

    /// Validate data consistency from the client side
    /// - all point ids exist
    /// - all named vectors exist
    /// - all mandatory payload keys exist
    async fn data_consistency_check(
        &self,
        client: &Qdrant,
        current_count: usize,
    ) -> Result<(), CrasherError> {
        let mut errors = Vec::new();

        // check all points and vector present in storage
        match check_points_consistency(&self.collection_name, client, current_count).await {
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

        if errors.is_empty() {
            return Ok(());
        }
        let full_report = errors.join("\n---\n");
        Err(Invariant(format!(
            "Data inconsistencies found out of {current_count} points:\n{full_report}"
        )))
    }

    fn trigger_continuous_snapshotting(
        &self,
        client: &Qdrant,
    ) -> JoinHandle<Result<(), CrasherError>> {
        let collection_name = self.collection_name.clone();
        let client_snapshot = client.clone();
        let stopped = self.stopped.clone();
        tokio::spawn(async move {
            while !stopped.load(Ordering::Relaxed) {
                match churn_collection_snapshot(&client_snapshot, &collection_name).await {
                    Ok(()) => tokio::time::sleep(Duration::from_millis(500)).await,
                    Err(err) => {
                        log::warn!("Snapshotting failed {err}");
                        // stop at first snapshot error silently
                        return Err(err);
                    }
                }
            }
            Ok(())
        })
    }
}

async fn churn_collection_snapshot(
    client: &Qdrant,
    collection_name: &str,
) -> Result<(), CrasherError> {
    let response = create_collection_snapshot(client, collection_name).await?;
    let snapshot_name = response
        .snapshot_description
        .expect("no snapshot description!")
        .name;
    let _response = delete_collection_snapshot(client, collection_name, &snapshot_name).await?;
    Ok(())
}
