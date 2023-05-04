use anyhow::Result;
use qdrant_client::client::QdrantClient;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::sleep;

use crate::args::Args;
use crate::client::{create_collection, insert_points_batch, search_points};
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::{Cancelled, Client};

pub struct Workload {
    collection_name: String,
    search_count: usize,
    points_count: usize,
    vec_dim: usize,
    payload_count: usize,
    stopped: Arc<AtomicBool>,
}

impl Workload {
    pub fn new(stopped: Arc<AtomicBool>) -> Self {
        let collection_name = "workload-crasher".to_string();
        let vec_dim = 1024;
        let payload_count = 2;
        let search_count = 10;
        let points_count = 100000;
        Workload {
            collection_name,
            search_count,
            points_count,
            vec_dim,
            payload_count,
            stopped,
        }
    }
}

impl Workload {
    pub async fn work(&self, client: &QdrantClient, args: Arc<Args>) {
        loop {
            if self.stopped.clone().load(Ordering::SeqCst) {
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
                Err(Client(error)) => {
                    // TODO disambiguate between server restarts and crashes
                    log::error!("Workload run failed: {:?}", error);
                    // no need to hammer the server while it restarts
                    sleep(std::time::Duration::from_secs(3)).await;
                }
            }
        }
    }

    pub async fn run(&self, client: &QdrantClient, args: Arc<Args>) -> Result<(), CrasherError> {
        log::info!("Starting workload...");
        // create and populate collection if it does not exists
        if !client.has_collection(&self.collection_name).await? {
            log::info!("Creating workload collection");
            create_collection(client, &self.collection_name, self.vec_dim, args.clone()).await?;
        }

        log::info!("Run pre-search");
        // search `search_count` times
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

        log::info!("Insert points");
        // insert some points
        insert_points_batch(
            client,
            &self.collection_name,
            self.points_count,
            self.vec_dim,
            self.payload_count,
            None,
            self.stopped.clone(),
        )
        .await?;

        log::info!("Run post-search");
        // search `search_count` times
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

        log::info!("Workload finished");
        Ok(())
    }
}
