use crate::args::Args;
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::Cancelled;
use crate::generators::{
    random_dense_vector, random_filter, random_payload, random_sparse_vector, DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
};
use anyhow::Context;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::vectors_config::Config::ParamsMap;
use qdrant_client::qdrant::{
    CollectionInfo, CountPoints, CreateCollection, Distance, FieldType, GetResponse,
    HnswConfigDiff, OptimizersConfigDiff, PointId, PointStruct, PointsIdsList, PointsSelector,
    QuantizationConfig, ScalarQuantization, SearchPoints, SearchResponse, SparseIndexConfig,
    SparseVectorConfig, SparseVectorParams, Vector, VectorParams, VectorParamsMap, Vectors,
    VectorsConfig, WithPayloadSelector, WithVectorsSelector, WriteOrdering,
};
use rand::Rng;
use std::collections::HashMap;
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

/// Get points count
pub async fn get_collection_info(
    client: &QdrantClient,
    collection_name: &str,
) -> Result<CollectionInfo, anyhow::Error> {
    let collection_info = client
        .collection_info(collection_name)
        .await
        .context(format!(
            "Failed to fetch collection info for {}",
            collection_name
        ))?
        .result
        .unwrap();
    Ok(collection_info)
}

/// Get points count from collection info
#[allow(dead_code)]
pub async fn get_info_points_count(
    client: &QdrantClient,
    collection_name: &str,
) -> Result<usize, anyhow::Error> {
    let point_count = client
        .collection_info(collection_name)
        .await
        .context(format!(
            "Failed to fetch points count for {}",
            collection_name
        ))?
        .result
        .unwrap()
        .points_count;
    Ok(point_count.unwrap_or(0) as usize)
}

/// Get points count
pub async fn get_points_count(
    client: &QdrantClient,
    collection_name: &str,
) -> Result<usize, anyhow::Error> {
    let count_req = CountPoints {
        collection_name: collection_name.to_string(),
        filter: None,
        exact: Some(true),
        read_consistency: None,
        shard_key_selector: None,
    };
    let point_count = client
        .count(&count_req)
        .await
        .context(format!(
            "Failed to run points count for {}",
            collection_name
        ))?
        .result
        .unwrap()
        .count;
    Ok(point_count as usize)
}

/// Search points
pub async fn search_points(
    client: &QdrantClient,
    collection_name: &str,
    vec_dim: usize,
    payload_count: usize,
) -> Result<SearchResponse, anyhow::Error> {
    // TODO generate sparse search as well

    let query_vector = random_dense_vector(vec_dim);
    let query_filter = random_filter(Some(payload_count));

    let response = client
        .search_points(&SearchPoints {
            collection_name: collection_name.to_string(),
            vector: query_vector,
            vector_name: Some(DENSE_VECTOR_NAME.to_string()),
            filter: query_filter,
            limit: 100,
            with_payload: Some(true.into()),
            params: None,
            score_threshold: None,
            offset: None,
            with_vectors: None,
            read_consistency: None,
            timeout: None,
            shard_key_selector: None,
            sparse_indices: None,
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
    let sparse_vectors_config = {
        let params = vec![(
            SPARSE_VECTOR_NAME.to_string(),
            SparseVectorParams {
                index: Some(SparseIndexConfig {
                    full_scan_threshold: None,
                    on_disk: Some(true), // make configurable
                }),
            },
        )]
        .into_iter()
        .collect();

        Some(SparseVectorConfig { map: params })
    };

    let dense_vectors_config = {
        let params: HashMap<_, _> = vec![(
            DENSE_VECTOR_NAME.to_string(),
            VectorParams {
                size: vec_dim as u64,
                distance: Distance::Dot.into(), // make configurable
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
                hnsw_config: Some(HnswConfigDiff {
                    m: Some(32),
                    ef_construct: None,
                    full_scan_threshold: None,
                    max_indexing_threads: None,
                    on_disk: Some(true), // make configurable
                    payload_m: None,
                }),
                on_disk: Some(args.vectors_on_disk),
                datatype: None,
            },
        )]
        .into_iter()
        .collect();

        Some(VectorsConfig {
            config: Some(ParamsMap(VectorParamsMap { map: params })),
        })
    };

    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: dense_vectors_config,
            sparse_vectors_config,
            replication_factor: Some(args.replication_factor as u32),
            write_consistency_factor: Some(args.write_consistency_factor as u32),
            optimizers_config: Some(OptimizersConfigDiff {
                indexing_threshold: args.indexing_threshold.map(|i| i as u64),
                memmap_threshold: args.memmap_threshold.map(|i| i as u64),
                ..Default::default()
            }),
            on_disk_payload: Some(true), // make configurable
            ..Default::default()
        })
        .await
        .context(format!("Failed to create collection {}", collection_name))?;
    Ok(())
}

/// insert points into collection (blocking)
#[allow(clippy::too_many_arguments)]
pub async fn insert_points_batch(
    client: &QdrantClient,
    collection_name: &str,
    points_count: usize,
    vec_dim: usize,
    payload_count: usize,
    only_sparse_vectors: bool,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
) -> Result<(), CrasherError> {
    let max_batch_size = 64;
    // handle less than batch & spill over
    let (batch_size, num_batches, last_batch_size) = if points_count <= max_batch_size {
        (points_count, 1, points_count)
    } else {
        let remainder = points_count % max_batch_size;
        let div = points_count / max_batch_size;
        let num_batches = div + if remainder > 0 { 1 } else { 0 };
        let last_batch_size = if remainder > 0 {
            remainder
        } else {
            max_batch_size
        };
        (max_batch_size, num_batches, last_batch_size)
    };
    for batch_id in 0..num_batches {
        let batch_size = if batch_id == num_batches - 1 {
            last_batch_size
        } else {
            batch_size
        };
        let mut points = Vec::with_capacity(batch_size);
        let batch_base_id = batch_id as u64 * max_batch_size as u64;
        for i in 0..batch_size {
            let idx = batch_base_id + i as u64;
            let point_id = PointId {
                point_id_options: Some(PointIdOptions::Num(idx)),
            };
            let mut vectors_map: HashMap<String, Vector> = HashMap::new();
            let mut rng = rand::thread_rng();
            let add_dense = if only_sparse_vectors {
                false
            } else {
                rng.gen_bool(0.5)
            };
            if add_dense {
                vectors_map.insert(
                    DENSE_VECTOR_NAME.to_string(),
                    random_dense_vector(vec_dim).into(),
                );
            }

            let add_sparse = if only_sparse_vectors {
                true
            } else {
                rng.gen_bool(0.5)
            };
            if add_sparse {
                vectors_map.insert(
                    SPARSE_VECTOR_NAME.to_string(),
                    random_sparse_vector(vec_dim, 0.1).into(),
                );
            }

            let vectors: Vectors = vectors_map.into();

            points.push(PointStruct::new(
                point_id,
                vectors,
                random_payload(Some(payload_count)),
            ));
        }
        if stopped.load(Ordering::Relaxed) {
            return Err(Cancelled);
        }

        // push batch blocking
        client
            .upsert_points_blocking(collection_name, None, points, write_ordering.clone())
            .await
            .context(format!(
                "Failed to insert {} points (batch {}/{}) into {}",
                batch_size, batch_id, num_batches, collection_name
            ))?;
    }
    Ok(())
}

pub async fn create_field_index(
    client: &QdrantClient,
    collection_name: &str,
    field_name: &str,
    field_type: FieldType,
) -> Result<(), anyhow::Error> {
    client
        .create_field_index_blocking(
            collection_name.to_string(),
            field_name.to_string(),
            field_type,
            None,
            None,
        )
        .await
        .context(format!(
            "Failed to create field index {} for collection {}",
            field_name, collection_name
        ))?;
    Ok(())
}

/// Set payload (blocking)
pub async fn set_payload(
    client: &QdrantClient,
    collection_name: &str,
    point_id: u64,
    payload_count: usize,
    write_ordering: Option<WriteOrdering>,
) -> Result<(), anyhow::Error> {
    let payload = random_payload(Some(payload_count));

    let points_id_selector = vec![PointId {
        point_id_options: Some(PointIdOptions::Num(point_id)),
    }];

    let points_selector = &PointsSelector {
        points_selector_one_of: Some(PointsSelectorOneOf::Points(PointsIdsList {
            ids: points_id_selector,
        })),
    };

    let resp = client
        .set_payload_blocking(
            collection_name,
            None,
            points_selector,
            payload,
            None,
            write_ordering,
        )
        .await
        .context(format!(
            "Failed to set payload for {} with payload_count {}",
            point_id, payload_count
        ))?;

    if resp.result.unwrap().status != 2 {
        Err(anyhow::anyhow!(
            "Failed to set payload on point_id {} for {}",
            point_id,
            collection_name
        ))
    } else {
        Ok(())
    }
}

/// Retrieve points
pub async fn retrieve_points(
    client: &QdrantClient,
    collection_name: &str,
    ids: &[usize],
) -> Result<GetResponse, anyhow::Error> {
    // type inference issues forces to ascribe the types :shrug:
    let with_vectors: Option<WithVectorsSelector> = Some(true.into());
    let with_payload: Option<WithPayloadSelector> = Some(true.into());
    let response = client
        .get_points(
            collection_name,
            None,
            ids.iter()
                .map(|id| (*id as u64).into())
                .collect::<Vec<_>>()
                .as_slice(),
            with_vectors,
            with_payload,
            None,
        )
        .await
        .context(format!("Failed to retrieve points on {}", collection_name))?;

    Ok(response)
}
