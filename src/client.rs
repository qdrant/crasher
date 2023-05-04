use crate::args::Args;
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::Cancelled;
use crate::generators::{random_filter, random_payload, random_vector};
use anyhow::Context;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollection, Distance, OptimizersConfigDiff, PointId, PointStruct, QuantizationConfig,
    ScalarQuantization, SearchPoints, SearchResponse, VectorParams, VectorsConfig, WriteOrdering,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

/// Wait for collection to be indexed
pub async fn wait_server_ready(
    client: &QdrantClient,
    stopped: Arc<AtomicBool>,
) -> Result<f64, CrasherError> {
    let start = std::time::Instant::now();
    loop {
        if stopped.load(Ordering::Relaxed) {
            return Err(Cancelled);
        }
        sleep(Duration::from_secs(1)).await;
        let healthcheck = client.health_check().await;
        match healthcheck {
            Ok(_) => break,
            Err(e) => {
                log::debug!("Healthcheck failed: {}", e)
            }
        }
    }
    Ok(start.elapsed().as_secs_f64())
}

/// Search points
pub async fn search_points(
    client: &QdrantClient,
    collection_name: &str,
    vec_dim: usize,
    payload_count: usize,
) -> Result<SearchResponse, anyhow::Error> {
    let query_vector = random_vector(vec_dim);
    let query_filter = random_filter(Some(payload_count));

    let response = client
        .search_points(&SearchPoints {
            collection_name: collection_name.to_string(),
            vector: query_vector,
            filter: query_filter,
            limit: 100,
            with_payload: Some(true.into()),
            params: None,
            score_threshold: None,
            offset: None,
            vector_name: None,
            with_vectors: None,
            read_consistency: None,
        })
        .await
        .context(format!("Failed to search points on {}", collection_name))?;

    Ok(response)
}

/// Create collection
pub async fn create_collection(
    client: &QdrantClient,
    collection_name: &str,
    vec_dim: usize,
    args: Arc<Args>,
) -> Result<(), anyhow::Error> {
    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Cosine.into(),
                    hnsw_config: None,
                    quantization_config: None,
                })),
            }),
            replication_factor: Some(args.replication_factor as u32),
            write_consistency_factor: Some(args.write_consistency_factor as u32),
            optimizers_config: Some(OptimizersConfigDiff {
                indexing_threshold: args.indexing_threshold.map(|i| i as u64),
                memmap_threshold: args.memmap_threshold.map(|i| i as u64),
                ..Default::default()
            }),
            quantization_config: if args.use_scalar_quantization {
                Some(QuantizationConfig {
                    quantization: Some(Quantization::Scalar(ScalarQuantization {
                        r#type: 1, //Int8
                        quantile: None,
                        always_ram: Some(true),
                    })),
                })
            } else {
                None
            },
            ..Default::default()
        })
        .await
        .context(format!("Failed to create collection {}", collection_name))?;
    Ok(())
}

/// insert points into collection (blocking)
pub async fn insert_points_batch(
    client: &QdrantClient,
    collection_name: &str,
    points_count: usize,
    vec_dim: usize,
    payload_count: usize,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
) -> Result<(), CrasherError> {
    let batch_size = 100;
    let num_batches = points_count / batch_size;

    for batch_id in 0..num_batches {
        let mut points = Vec::with_capacity(batch_size);
        let batch_base_id = batch_id as u64 * batch_size as u64;
        for i in 0..batch_size {
            let idx = batch_base_id + i as u64;

            let point_id = PointId {
                point_id_options: Some(PointIdOptions::Num(idx)),
            };

            points.push(PointStruct::new(
                point_id,
                random_vector(vec_dim),
                random_payload(Some(payload_count)),
            ));
        }
        if stopped.load(Ordering::Relaxed) {
            return Err(Cancelled);
        }

        // push batch blocking
        client
            .upsert_points_blocking(collection_name, points, write_ordering.clone())
            .await
            .context(format!(
                "Failed to insert {} points (batch {}/{}) into {}",
                batch_size, batch_id, num_batches, collection_name
            ))?;
    }
    Ok(())
}
