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
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::time::sleep;

use crate::args::Args;
use crate::client::{
    create_collection, create_field_index, delete_points, get_collection_info, get_points_count,
    insert_points_batch, query_batch_points, retrieve_points, set_payload,
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
    query_count: usize,
    points_count: usize,
    vec_dim: usize,
    payload_count: usize,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
}

impl Workload {
    pub fn new(
        collection_name: &str,
        stopped: Arc<AtomicBool>,
        duplication_factor: usize,
        points_count: usize,
        vec_dim: usize,
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
        }
    }
}

impl Workload {
    pub async fn work(&self, client: &Qdrant, args: Arc<Args>) {
        loop {
            if self.stopped.clone().load(Ordering::Relaxed) {
                break;
            }
            let run = self.run(client, args.clone()).await;
            match run {
                Ok(()) => {
                    log::info!("Workload run finished");
                }
                Err(Cancelled) => {
                    log::info!("Workload run cancelled");
                    break;
                }
                Err(Invariant(msg)) => {
                    log::error!("Workload run failed with violation!\n{msg}");
                    // send stop signal to the main thread
                    self.stopped.store(true, Ordering::Relaxed);
                    log::error!("Stopping the workload...");
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

    pub async fn run(&self, client: &Qdrant, args: Arc<Args>) -> Result<(), CrasherError> {
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

            if args.missing_payload_check {
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
            }

            let collection_info = get_collection_info(client, &self.collection_name).await?;
            log::info!("Collection info: {collection_info:#?}");
        }

        let current_count = get_points_count(client, &self.collection_name).await?;
        if current_count != 0 {
            if args.consistency_check {
                // can be disabled if qdrant is running internal data consistency check on the server side
                // `cargo run --features data-consistency-check`
                log::info!("Run: pre vector data consistency check ({current_count})");
                self.vector_data_consistency_check(client, current_count)
                    .await?;
            }

            if args.missing_payload_check {
                log::info!("Run: pre payload data consistency check");
                self.mandatory_payload_check(client).await?;
            }

            log::info!("Run: delete existing points ({current_count})");
            delete_points(client, &self.collection_name, current_count).await?;
        }

        log::info!("Run: insert points");
        insert_points_batch(
            client,
            &self.collection_name,
            self.points_count,
            self.vec_dim,
            0,                          // no payload at first
            args.missing_payload_check, // add timestamp payload for the missing payload check
            args.only_sparse,
            &self.test_named_vectors,
            None,
            self.stopped.clone(),
        )
        .await?;

        if args.missing_payload_check {
            log::info!("Run: post-point-insert payload data consistency check");
            self.mandatory_payload_check(client).await?;
        }

        log::info!("Run: point count");
        let points_count = get_points_count(client, &self.collection_name).await?;
        if points_count != self.points_count {
            return Err(Invariant(format!(
                "Collection has wrong number of points after insert {} vs {}",
                points_count, self.points_count
            )));
        }

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
            )
            .await?;
        }

        if args.missing_payload_check {
            log::info!("Run: post-payload-insert payload data consistency check");
            self.mandatory_payload_check(client).await?;
        }

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
            )
            .await?;
            check_search_result(results)?;
        }

        log::info!("Workload finished");
        Ok(())
    }

    /// Vector data consistency checker for id range
    async fn vector_data_consistency_check(
        &self,
        client: &Qdrant,
        points_count: usize,
    ) -> Result<(), CrasherError> {
        // fetch all existing points (rely on numeric ids!)
        let all_ids: Vec<_> = (1..points_count).collect();
        // by batches to not overload the server
        for ids in all_ids.chunks(100) {
            let response = retrieve_points(client, &self.collection_name, ids).await?;
            // assert all there empty
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
                    .collect::<Vec<_>>();
                return Err(Invariant(format!(
                    "Retrieve did not return all result {}/{}\nMissing ids: {:?}",
                    response.result.len(),
                    ids.len(),
                    missing_ids
                )));
            } else {
                for point in &response.result {
                    let point_id = point.id.as_ref().expect("Point id should be present");
                    if let Some(vectors) = &point.vectors {
                        if let Some(vector) = &vectors.vectors_options {
                            match vector {
                                VectorsOptions::Vector(anonymous) => {
                                    return Err(Invariant(format!(
                                        "Vector {point_id:?} should be named: {anonymous:?}"
                                    )));
                                }
                                VectorsOptions::Vectors(named_vectors) => {
                                    if named_vectors.vectors.is_empty() {
                                        return Err(Invariant(format!(
                                            "Named vector {point_id:?} should not be empty"
                                        )));
                                    }
                                    for (name, vector) in &named_vectors.vectors {
                                        if vector.data.is_empty() {
                                            return Err(Invariant(format!(
                                                "Vector {name} with id {point_id:?} should not be empty"
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        return Err(Invariant(format!(
                            "Vector {point_id:?} should be present in the response"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    async fn mandatory_payload_check(&self, client: &Qdrant) -> Result<(), CrasherError> {
        // TODO check present in storage with scroll by ids
        // TODO check present in typed index with match query

        self.check_null_index(client).await?;
        // self.check_bool_index(client).await?;

        Ok(())
    }

    async fn check_null_index(&self, client: &Qdrant) -> Result<(), CrasherError> {
        let resp = client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(Filter::must([Condition::is_empty(
                        MANDATORY_PAYLOAD_TIMESTAMP_KEY,
                    )]))
                    .limit(self.points_count.try_into().unwrap()),
            )
            .await?;
        let points: Vec<_> = resp
            .result
            .into_iter()
            .map(|point| point.id.and_then(|pid| pid.point_id_options))
            .collect();
        Ok(if !points.is_empty() {
            return Err(Invariant(format!(
                "Detected {} points missing the '{}' payload key!\n{:?}",
                points.len(),
                MANDATORY_PAYLOAD_TIMESTAMP_KEY,
                points,
            )));
        })
    }

    async fn check_bool_index(&self, client: &Qdrant) -> Result<(), CrasherError> {
        let current_count = get_points_count(client, &self.collection_name).await?;

        let resp = client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(Filter::must([Condition::matches(
                        MANDATORY_PAYLOAD_BOOL_KEY,
                        true,
                    )]))
                    .limit(self.points_count.try_into().unwrap()),
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

        Ok(if !missing_points.is_empty() {
            return Err(Invariant(format!(
                "Out of {}, detected {} points missing the '{}: true' payload!\n{:?}",
                current_count,
                missing_points.len(),
                MANDATORY_PAYLOAD_BOOL_KEY,
                missing_points,
            )));
        })
    }
}

/// Checks if this is a zeroed vector.
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
