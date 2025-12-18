use anyhow::Result;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::qdrant::{
    BoolIndexParamsBuilder, Condition, DatetimeIndexParamsBuilder, FieldType, Filter,
    FloatIndexParamsBuilder, GeoIndexParamsBuilder, IntegerIndexParamsBuilder,
    KeywordIndexParamsBuilder, QueryBatchResponse, ScrollPointsBuilder, TextIndexParamsBuilder,
    TokenizerType, UuidIndexParamsBuilder, VectorOutput, WriteOrdering, vector_output,
};
use rand::Rng;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::args::Args;
use crate::client::{
    create_collection, create_collection_snapshot, create_field_index, delete_collection_snapshot,
    delete_points, get_collection_info, get_exact_points_count, insert_points_batch,
    list_collection_snapshots, query_batch_points, restore_collection_snapshot, retrieve_points,
    set_payload,
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
        let payload_count = 1;
        let query_count = 2;
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
    pub async fn work(&self, client: &Qdrant, args: Arc<Args>, rng: &mut impl Rng) {
        loop {
            if self.stopped.load(Ordering::Relaxed) {
                break;
            }
            let run = self.run(client, args.clone(), rng).await;
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
                        "Workload run failed due to an invariant violation!(rng_seed:{}) \n{msg}",
                        self.rng_seed
                    );
                    // Uncomment to display telemetry for debugging
                    //let telemetry = get_telemetry().await.unwrap_or_default();
                    //log::error!("Telemetry:\n{}", telemetry);
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
                        sleep(std::time::Duration::from_secs(3)).await;
                    }
                }
            }
        }
    }

    pub async fn run(
        &self,
        client: &Qdrant,
        args: Arc<Args>,
        rng: &mut impl Rng,
    ) -> Result<(), CrasherError> {
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
                TextIndexParamsBuilder::new(TokenizerType::Word)
                    //.phrase_matching(true)
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
                    restore_collection_snapshot(&self.collection_name, snapshot_name).await?;

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
        let snapshotting_handle = self.trigger_continous_snapshotting(client);

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
            check_search_result(results)?;
        }

        // Stop on-going snapshotting task
        snapshotting_handle.abort();
        match snapshotting_handle.await {
            Ok(Ok(_)) => (),
            Err(_) => (),                                      // ignore JoinError
            Ok(Err(snapshot_err)) => return Err(snapshot_err), // capture failed snapshot
        }

        log::info!("Workload finished");
        Ok(())
    }

    /// Vector data consistency checker for id range
    #[allow(deprecated)]
    async fn check_points_consistency(
        &self,
        client: &Qdrant,
        current_count: usize,
    ) -> Result<(), CrasherError> {
        // fetch all existing points (rely on numeric ids!)
        let all_ids: Vec<_> = (1..current_count).collect();

        // track all errors
        let mut missing_points_errors: Vec<usize> = Vec::new();
        let mut malformed_points_errors: Vec<String> = Vec::new();

        // by batches to not overload the server
        for ids in all_ids.chunks(current_count.min(1000)) {
            let response = retrieve_points(client, &self.collection_name, ids).await?;
            // check if missing points
            if response.result.len() != ids.len() {
                let response_ids = response
                    .result
                    .iter()
                    .map(|point| point.id.clone().unwrap().point_id_options.unwrap())
                    .map(|id| match id {
                        PointIdOptions::Num(id) => id as usize,
                        PointIdOptions::Uuid(_) => panic!("UUID in the response"),
                    })
                    .collect::<HashSet<_>>();

                let missing_ids = ids
                    .iter()
                    .filter(|&id| !response_ids.contains(id))
                    .cloned()
                    .collect::<Vec<_>>();
                missing_points_errors.extend_from_slice(&missing_ids);
            }
            // check points are welformed
            for point in &response.result {
                let point_id = point.id.as_ref().expect("Point id should be present");
                // wellformed vectors
                if let Some(vectors) = &point.vectors {
                    if let Some(vector) = &vectors.vectors_options {
                        match vector {
                            VectorsOptions::Vector(anonymous) => {
                                malformed_points_errors.push(format!(
                                    "Vector {point_id:?} should be named: {anonymous:?}"
                                ));
                            }
                            VectorsOptions::Vectors(named_vectors) => {
                                if named_vectors.vectors.is_empty() {
                                    malformed_points_errors.push(format!(
                                        "Named vector {point_id:?} should not be empty"
                                    ));
                                }
                                for (name, vector) in &named_vectors.vectors {
                                    if vector.data.is_empty() {
                                        malformed_points_errors.push(format!(
                                            "Vector {name} with id {point_id:?} should not be empty"
                                        ));
                                    }
                                    if check_zeroed_vector(vector) {
                                        malformed_points_errors.push(format!(
                                            "Vector {name} with id {point_id:?} is zeroed"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                } else {
                    malformed_points_errors.push(format!(
                        "Vector {point_id:?} should be present in the response"
                    ));
                }
                // check mandatory payload keys
                if !point.payload.contains_key(MANDATORY_PAYLOAD_BOOL_KEY) {
                    malformed_points_errors.push(format!(
                        "Vector {point_id:?} is missing payload_key:{MANDATORY_PAYLOAD_BOOL_KEY} in storage"
                    ));
                }
                if !point.payload.contains_key(MANDATORY_PAYLOAD_TIMESTAMP_KEY) {
                    malformed_points_errors.push(format!(
                        "Vector {point_id:?} is missing payload_key:{MANDATORY_PAYLOAD_TIMESTAMP_KEY} in storage"
                    ));
                }
            }
        }

        // merge all violations found
        let mut errors_found = Vec::new();
        if !missing_points_errors.is_empty() {
            errors_found.push(format!(
                "{} missing points detected: {:?}",
                missing_points_errors.len(),
                missing_points_errors
            ));
        }

        if !malformed_points_errors.is_empty() {
            errors_found.push(format!(
                "{} malformed points detected: {:?}",
                malformed_points_errors.len(),
                malformed_points_errors
            ));
        }

        if errors_found.is_empty() {
            Ok(())
        } else {
            let errors_rendered = errors_found.join("/n  - ");
            Err(Invariant(format!(
                "data inconsistency found:\n  - {errors_rendered}",
            )))
        }
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
        // check all points and vector present in storage
        self.check_points_consistency(client, current_count).await?;
        // check mandatory timestamp payload key via null index
        self.check_filter_null_index(client, current_count).await?;
        // check mandatory bool payload key via match query
        self.check_filter_bool_index(client, current_count).await?;
        Ok(())
    }

    async fn check_filter_null_index(
        &self,
        client: &Qdrant,
        current_count: usize,
    ) -> Result<(), CrasherError> {
        let resp = client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(Filter::must([Condition::is_empty(
                        MANDATORY_PAYLOAD_TIMESTAMP_KEY,
                    )]))
                    .limit(current_count as u32),
            )
            .await?;
        let points: Vec<_> = resp
            .result
            .into_iter()
            .map(|point| point.id.and_then(|pid| pid.point_id_options))
            .collect();

        if points.is_empty() {
            Ok(())
        } else {
            Err(Invariant(format!(
                "Detected {} points missing the '{}' payload key when matching for null values!\n{:?}",
                points.len(),
                MANDATORY_PAYLOAD_TIMESTAMP_KEY,
                points,
            )))
        }
    }

    async fn check_filter_bool_index(
        &self,
        client: &Qdrant,
        current_count: usize,
    ) -> Result<(), CrasherError> {
        let resp = client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(Filter::must([Condition::matches(
                        MANDATORY_PAYLOAD_BOOL_KEY,
                        true,
                    )]))
                    .limit(current_count as u32),
            )
            .await?;
        let points: HashSet<_> = resp
            .result
            .into_iter()
            .filter_map(|point| {
                point.id.and_then(|pid| {
                    pid.point_id_options.and_then(|options| match options {
                        PointIdOptions::Num(id) => Some(id),
                        PointIdOptions::Uuid(_) => None,
                    })
                })
            })
            .collect();

        let missing_points: Vec<_> = (0..current_count as u64)
            .filter(|id| !points.contains(id))
            .collect();

        if missing_points.is_empty() {
            Ok(())
        } else {
            Err(Invariant(format!(
                "Out of {}, detected {} points missing the '{}: true' when matching payload!\n{:?}",
                current_count,
                missing_points.len(),
                MANDATORY_PAYLOAD_BOOL_KEY,
                missing_points,
            )))
        }
    }

    fn trigger_continous_snapshotting(
        &self,
        client: &Qdrant,
    ) -> JoinHandle<Result<(), CrasherError>> {
        let collection_name = self.collection_name.clone();
        let client_snapshot = client.clone();
        let stopped = self.stopped.clone();
        tokio::spawn(async move {
            while !stopped.load(Ordering::Relaxed) {
                match churn_collection_snapshot(&client_snapshot, &collection_name).await {
                    Ok(_) => tokio::time::sleep(Duration::from_millis(500)).await,
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

/// Checks if this is a zeroed vector.
#[allow(deprecated)]
fn check_zeroed_vector(vector: &VectorOutput) -> bool {
    vector
        .vector
        .as_ref()
        .map(|vector| match vector {
            vector_output::Vector::Dense(dense_vector) => {
                dense_vector.data.iter().all(|v| *v == 0.0)
            }
            vector_output::Vector::Sparse(sparse_vector) => sparse_vector.indices.is_empty(),
            vector_output::Vector::MultiDense(multi_dense_vector) => multi_dense_vector
                .vectors
                .iter()
                .all(|v| v.data.iter().all(|v| *v == 0.0)),
        })
        // else, check the deprecated field
        .unwrap_or_else(|| vector.data.iter().all(|v| *v == 0.0))
}

fn check_search_result(results: QueryBatchResponse) -> Result<(), CrasherError> {
    // assert no vector is only containing zeros
    for point in results.result.iter().flat_map(|result| &result.result) {
        if let Some(vectors) = point
            .vectors
            .as_ref()
            .and_then(|v| v.vectors_options.as_ref())
        {
            let zeroed_vector = match vectors {
                VectorsOptions::Vector(v) => check_zeroed_vector(v).then_some(("".to_string(), v)),
                VectorsOptions::Vectors(vectors) => vectors
                    .vectors
                    .iter()
                    .find_map(|(name, v)| check_zeroed_vector(v).then_some((name.to_string(), v))),
            };
            if let Some((name, vector)) = zeroed_vector {
                return Err(Invariant(format!(
                    "Query result contains zeroed vector: \npoint id: {:?}\nzeroed vector name: {}\nzeroed vector: {:?}\n\npoint: {:?}",
                    point.id, name, vector, point
                )));
            }
        }
    }
    Ok(())
}
