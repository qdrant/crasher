use anyhow::Result;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::{FieldType, WriteOrdering};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::sleep;

use crate::args::Args;
use crate::client::{
    create_collection, create_field_index, get_collection_info, get_points_count,
    insert_points_batch, search_points, set_payload,
};
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::{Cancelled, Client, Invariant};
use crate::generators::KEYWORD_PAYLOAD_KEY;

pub struct Workload {
    collection_name: String,
    search_count: usize,
    points_count: usize,
    vec_dim: usize,
    payload_count: usize,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
}

impl Workload {
    pub fn new(stopped: Arc<AtomicBool>) -> Self {
        let collection_name = "workload-crasher".to_string();
        let vec_dim = 1024;
        let payload_count = 2;
        let search_count = 100;
        let points_count = 20_000;
        let write_ordering = None; // default
        Workload {
            collection_name,
            search_count,
            points_count,
            vec_dim,
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
                Ok(_) => {
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
                    // TODO disambiguate between server restarts and crashes
                    log::warn!(
                        "Workload run failed due to client error - restarting soon\n{}",
                        error
                    );
                    // no need to hammer the server while it restarts
                    sleep(std::time::Duration::from_secs(3)).await;
                }
            }
        }
    }

    pub async fn run(&self, client: &QdrantClient, args: Arc<Args>) -> Result<(), CrasherError> {
        log::info!("Starting workload...");
        // create and populate collection if it does not exist
        if !client.has_collection(&self.collection_name).await? {
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
            log::info!("Collection info: {:?}", collection_info);
        }

        log::info!("Insert points");
        insert_points_batch(
            client,
            &self.collection_name,
            self.points_count,
            self.vec_dim,
            0, // no payload at first
            None,
            self.stopped.clone(),
        )
        .await?;

        log::info!("Run set_payload");
        for point_id in 1..self.points_count {
            if self.stopped.load(Ordering::Relaxed) {
                return Err(Cancelled);
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

        log::info!("Run post-search");
        for _i in 0..self.search_count {
            if self.stopped.load(Ordering::Relaxed) {
                return Err(Cancelled);
            }
            search_points(
                client,
                &self.collection_name,
                self.vec_dim,
                self.payload_count,
            )
            .await?;
        }

        log::info!("Run post_count");
        let points_count = get_points_count(client, &self.collection_name).await?;
        if points_count != self.points_count {
            return Err(Invariant(format!(
                "Collection has wrong number of points after insert {} vs {}",
                points_count, self.points_count
            )));
        }

        log::info!("Workload finished");
        Ok(())
    }
}
