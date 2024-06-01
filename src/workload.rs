use anyhow::Result;
use qdrant_client::client::QdrantClient;
use qdrant_client::prelude::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::{FieldType, WriteOrdering};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::sleep;

use crate::args::Args;
use crate::client::VectorType;
use crate::client::{
    create_collection, create_field_index, delete_points, get_collection_info, get_points_count,
    insert_points_batch, retrieve_points, search_points, set_payload,
};
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::{Cancelled, Client, Invariant};
use crate::generators::KEYWORD_PAYLOAD_KEY;

pub struct Workload {
    collection_name: String,
    search_count: usize,
    points_count: usize,
    vec_dim: usize,
    vec_sparsity: f64,
    payload_count: usize,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
}

impl Workload {
    pub fn new(stopped: Arc<AtomicBool>, points_count: usize) -> Self {
        let collection_name = "workload-crasher".to_string();
        let vec_dim = 1024;
        let vec_sparsity = 1f64;
        let payload_count = 1;
        let search_count = 10;
        let write_ordering = None; // default
        Workload {
            collection_name,
            search_count,
            points_count,
            vec_dim,
            vec_sparsity,
            payload_count,
            write_ordering,
            stopped,
        }
    }
}

impl Workload {
    pub async fn work(&self, client: &QdrantClient, args: Arc<Args>) {
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
                    log::error!("Workload run failed with violation!\n{}", msg);
                    // send stop signal to the main thread
                    self.stopped.store(true, Ordering::Relaxed);
                    break;
                }
                Err(Client(error)) => {
                    // turn client error into hard error if it is a server error (a bit hacky)
                    let debug_error = format!("{:?}", error);
                    if debug_error.contains("Service internal error") {
                        log::error!("Workload run failed due to a server error!\n{}", error);
                        // send stop signal to the main thread
                        self.stopped.store(true, Ordering::Relaxed);
                        break;
                    } else {
                        log::warn!(
                            "Workload run failed due to client error - resuming soon\n{}",
                            error
                        );
                        // no need to hammer the server while it restarts
                        sleep(std::time::Duration::from_secs(3)).await;
                    }
                }
            }
        }
    }

    pub async fn run(&self, client: &QdrantClient, args: Arc<Args>) -> Result<(), CrasherError> {
        log::info!("Starting workload...");
        // create and populate collection if it does not exist
        if !client.collection_exists(&self.collection_name).await? {
            log::info!("Creating workload collection");
            create_collection(client, &self.collection_name, self.vec_dim, args.clone()).await?;
            create_field_index(
                client,
                &self.collection_name,
                KEYWORD_PAYLOAD_KEY,
                FieldType::Keyword,
            )
            .await?;
            let collection_info = get_collection_info(client, &self.collection_name).await?;
            log::info!("Collection info: {:#?}", collection_info);
        }

        let current_count = get_points_count(client, &self.collection_name).await?;
        if current_count != 0 {
            log::info!(
                "Run: pre integrity check ({} existing points)",
                current_count
            );
            self.consistency_check(client, current_count).await?;

            log::info!("Run: pre search random vector");
            for i in 0..self.search_count {
                if self.stopped.load(Ordering::Relaxed) {
                    return Err(Cancelled);
                }
                let vec_type = if i % 2 == 0 {
                    VectorType::Dense { dim: self.vec_dim }
                } else {
                    VectorType::Sparse {
                        dim: self.vec_dim,
                        sparsity: self.vec_sparsity,
                    }
                };
                search_points(client, &self.collection_name, vec_type, self.payload_count).await?;
            }

            log::info!("Run: delete existing points");
            delete_points(client, &self.collection_name, current_count).await?;
        }

        log::info!("Run: insert points");
        insert_points_batch(
            client,
            &self.collection_name,
            self.points_count,
            self.vec_dim,
            0, // no payload at first
            args.only_sparse,
            None,
            self.stopped.clone(),
        )
        .await?;

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
                self.write_ordering.clone(),
            )
            .await?;
        }

        log::info!("Run: post consistency check");
        self.consistency_check(client, self.points_count).await?;

        log::info!("Run: post search random vector");
        for i in 0..self.search_count {
            if self.stopped.load(Ordering::Relaxed) {
                return Err(Cancelled);
            }
            let vec_type = if i % 2 == 0 {
                VectorType::Dense { dim: self.vec_dim }
            } else {
                VectorType::Sparse {
                    dim: self.vec_dim,
                    sparsity: self.vec_sparsity,
                }
            };
            search_points(client, &self.collection_name, vec_type, self.payload_count).await?;
        }

        log::info!("Workload finished");
        Ok(())
    }

    /// Consistency checker for id range
    async fn consistency_check(
        &self,
        client: &QdrantClient,
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
                        for vector in &vectors.vectors_options {
                            match vector {
                                VectorsOptions::Vector(anonymous) => {
                                    return Err(Invariant(format!(
                                        "Vector {:?} should be named: {:?}",
                                        point_id, anonymous
                                    )));
                                }
                                VectorsOptions::Vectors(named_vectors) => {
                                    if named_vectors.vectors.is_empty() {
                                        return Err(Invariant(format!(
                                            "Named vector {:?} should not be empty",
                                            point_id
                                        )));
                                    }
                                    for (name, vector) in &named_vectors.vectors {
                                        if vector.data.is_empty() {
                                            return Err(Invariant(format!(
                                                "Vector {} with id {:?} should not be empty",
                                                name, point_id
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        return Err(Invariant(format!(
                            "Vector {:?} should be present in the response",
                            point_id
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}
