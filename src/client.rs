use crate::args::Args;
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::Cancelled;
use crate::generators::{
    random_dense_vector, random_filter, random_payload, random_sparse_vector, TestNamedVectors,
};
use anyhow::Context;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::vectors_config::Config::ParamsMap;
use qdrant_client::qdrant::{
    CollectionInfo, CountPoints, CreateCollection, FieldType, GetResponse, OptimizersConfigDiff,
    PointId, PointStruct, PointsIdsList, PointsSelector, SearchBatchPoints, SearchBatchResponse,
    SearchPoints, SparseIndices, SparseVectorConfig, Vector, VectorParamsMap, Vectors,
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
                if start.elapsed().as_secs_f64() > 60.0 {
                    return Err(CrasherError::Invariant(
                        "Server did not start in time".to_string(),
                    ));
                } else {
                    log::debug!("Healthcheck failed: {}", e)
                }
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
pub async fn search_batch_points(
    client: &QdrantClient,
    collection_name: &str,
    test_named_vectors: &TestNamedVectors,
    only_sparse: bool,
    vec_dim: usize,
    payload_count: usize,
) -> Result<SearchBatchResponse, anyhow::Error> {
    let request_filter = random_filter(Some(payload_count));
    let mut requests = vec![];

    // sparse search requests
    for sparse_name in test_named_vectors.sparse_vector_names() {
        let sparse_vector = random_sparse_vector(vec_dim, 0.1);
        // split values & indices
        let data: Vec<_> = sparse_vector.iter().map(|(idx, _)| *idx).collect();
        let sparse_indices = Some(SparseIndices { data });
        let sparse_values = sparse_vector.iter().map(|(_, val)| *val).collect();

        let request = SearchPoints {
            collection_name: collection_name.to_string(),
            vector: sparse_values,
            vector_name: Some(sparse_name),
            filter: request_filter.clone(),
            limit: 100,
            with_payload: Some(true.into()),
            params: None,
            score_threshold: None,
            offset: None,
            with_vectors: None,
            read_consistency: None,
            timeout: None,
            shard_key_selector: None,
            sparse_indices,
        };
        requests.push(request);
    }

    // dense search requests
    if !only_sparse {
        // dense
        for dense_name in test_named_vectors.dense_vector_names() {
            let request = SearchPoints {
                collection_name: collection_name.to_string(),
                vector: random_dense_vector(vec_dim),
                vector_name: Some(dense_name.clone()),
                filter: request_filter.clone(),
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
            };
            requests.push(request);
        }
        // multi dense
        for multi_dense_name in test_named_vectors.multi_vector_names() {
            // TODO search with real multivectors when supported (relying on expansion dense for now)
            let request = SearchPoints {
                collection_name: collection_name.to_string(),
                vector: random_dense_vector(vec_dim),
                vector_name: Some(multi_dense_name.clone()),
                filter: request_filter.clone(),
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
            };
            requests.push(request);
        }
    }

    let search_batch_points = SearchBatchPoints {
        collection_name: collection_name.to_string(),
        search_points: requests,
        read_consistency: None,
        timeout: None,
    };

    let response = client
        .search_batch_points(&search_batch_points)
        .await
        .context(format!(
            "Failed to search batch points on {}",
            collection_name
        ))?;

    Ok(response)
}

/// Create collection
pub async fn create_collection(
    client: &QdrantClient,
    collection_name: &str,
    test_named_vectors: &TestNamedVectors,
    args: Arc<Args>,
) -> Result<(), anyhow::Error> {
    let sparse_vector_params = test_named_vectors.sparse_vectors();
    let sparse_vectors_config = Some(SparseVectorConfig {
        map: sparse_vector_params,
    });

    let dense_vectors_config = if !args.only_sparse {
        let mut all_dense_params = HashMap::new();

        // dense vectors
        let dense_vector_params = test_named_vectors.dense_vectors();
        all_dense_params.extend(dense_vector_params);

        // multi dense vectors
        let multi_dense_vector_params = test_named_vectors.multi_dense_vectors();
        all_dense_params.extend(multi_dense_vector_params);

        Some(VectorsConfig {
            config: Some(ParamsMap(VectorParamsMap {
                map: all_dense_params,
            })),
        })
    } else {
        None
    };

    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.to_string(),
            hnsw_config: None,
            vectors_config: dense_vectors_config,
            sparse_vectors_config,
            replication_factor: Some(args.replication_factor as u32),
            write_consistency_factor: Some(args.write_consistency_factor as u32),
            init_from_collection: None,
            quantization_config: None,
            optimizers_config: Some(OptimizersConfigDiff {
                deleted_threshold: None,
                vacuum_min_vector_number: None,
                default_segment_number: Some(args.segment_count as u64), // to force constant merges
                indexing_threshold: args.indexing_threshold.map(|i| i as u64),
                flush_interval_sec: Some(args.flush_interval_sec as u64),
                memmap_threshold: args.memmap_threshold.map(|i| i as u64),
                max_segment_size: None,
                max_optimization_threads: None,
            }),
            shard_number: None,
            on_disk_payload: Some(false), // a bit faster
            wal_config: None,
            timeout: None,
            sharding_method: None,
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
    test_named_vectors: &TestNamedVectors,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
) -> Result<(), CrasherError> {
    let max_batch_size = 32;
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
        let (wait, batch_size) = if batch_id == num_batches - 1 {
            (true, last_batch_size)
        } else {
            (false, batch_size)
        };
        let mut points = Vec::with_capacity(batch_size);
        let batch_base_id = batch_id as u64 * max_batch_size as u64;
        for i in 0..batch_size {
            let idx = batch_base_id + i as u64;
            let point_id = PointId {
                point_id_options: Some(PointIdOptions::Num(idx)),
            };
            let mut vectors_map: HashMap<String, Vector> = HashMap::new();

            if !only_sparse_vectors {
                // dense
                for name in test_named_vectors.dense_vector_names() {
                    vectors_map.insert(
                        name.clone(),
                        Vector::new_dense(random_dense_vector(vec_dim)),
                    );
                }
                // multi dense
                for name in test_named_vectors.multi_vector_names() {
                    let vec_count = rand::thread_rng().gen_range(1..5);
                    let multi_vector = (0..vec_count)
                        .map(|_| random_dense_vector(vec_dim))
                        .collect();
                    vectors_map.insert(name.clone(), Vector::new_multi(multi_vector));
                }
            }

            // always add sparse vectors
            for name in test_named_vectors.sparse_vector_names() {
                vectors_map.insert(name.clone(), random_sparse_vector(vec_dim, 0.1).into());
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

        if wait {
            // push batch blocking
            client
                .upsert_points_blocking(collection_name, None, points, write_ordering.clone())
                .await
                .context(format!(
                    "Failed to block insert {} points (batch {}/{}) into {}",
                    batch_size, batch_id, num_batches, collection_name
                ))?;
        } else {
            // push batch non-blocking
            client
                .upsert_points(collection_name, None, points, write_ordering.clone())
                .await
                .context(format!(
                    "Failed to insert {} points (batch {}/{}) into {}",
                    batch_size, batch_id, num_batches, collection_name
                ))?;
        }
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

/// delete points (blocking)
pub async fn delete_points(
    client: &QdrantClient,
    collection_name: &str,
    points_count: usize,
) -> Result<(), anyhow::Error> {
    let points_selector = (0..points_count as u64)
        .map(|id| PointId {
            point_id_options: Some(PointIdOptions::Num(id)),
        })
        .collect();

    // delete all points
    let resp = client
        .delete_points_blocking(
            collection_name,
            None,
            &PointsSelector {
                points_selector_one_of: Some(PointsSelectorOneOf::Points(PointsIdsList {
                    ids: points_selector,
                })),
            },
            None,
        )
        .await
        .context(format!(
            "Failed to delete {} points for {}",
            points_count, collection_name
        ))?;
    if resp.result.unwrap().status != 2 {
        Err(anyhow::anyhow!(
            "Failed to delete {} points for {}",
            points_count,
            collection_name
        ))
    } else {
        Ok(())
    }
}
