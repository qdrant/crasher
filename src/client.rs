use crate::args::Args;
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::Cancelled;
use crate::generators::{
    random_dense_vector, random_filter, random_payload, random_sparse_vector, DENSE_VECTOR_NAME_BQ,
    DENSE_VECTOR_NAME_ON_DISK, DENSE_VECTOR_NAME_PQ, DENSE_VECTOR_NAME_ROCKSDB,
    DENSE_VECTOR_NAME_SQ, DENSE_VECTOR_NAME_UINT8, SPARSE_VECTOR_NAME, SPARSE_VECTOR_NAME_EIGHT,
    SPARSE_VECTOR_NAME_FIVE, SPARSE_VECTOR_NAME_FOUR, SPARSE_VECTOR_NAME_INDEX_MMAP,
    SPARSE_VECTOR_NAME_NINE, SPARSE_VECTOR_NAME_ONE, SPARSE_VECTOR_NAME_SEVEN,
    SPARSE_VECTOR_NAME_SIX, SPARSE_VECTOR_NAME_TEN, SPARSE_VECTOR_NAME_THREE,
    SPARSE_VECTOR_NAME_TWO,
};
use anyhow::Context;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::vectors_config::Config::ParamsMap;
use qdrant_client::qdrant::{
    BinaryQuantization, CollectionInfo, CountPoints, CreateCollection, Distance, FieldType,
    GetResponse, HnswConfigDiff, OptimizersConfigDiff, PointId, PointStruct, PointsIdsList,
    PointsSelector, ProductQuantization, QuantizationConfig, ScalarQuantization, SearchPoints,
    SearchResponse, SparseIndexConfig, SparseIndices, SparseVectorConfig, SparseVectorParams,
    Vector, VectorParams, VectorParamsMap, Vectors, VectorsConfig, WithPayloadSelector,
    WithVectorsSelector, WriteOrdering,
};
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
pub async fn search_points(
    client: &QdrantClient,
    collection_name: &str,
    only_sparse: bool,
    vec_dim: usize,
    payload_count: usize,
) -> Result<SearchResponse, anyhow::Error> {
    // TODO use batch search to target all named vectors
    let (query_vector, sparse_indices, vector_name) = if only_sparse {
        let sparse_vector = random_sparse_vector(vec_dim, 0.1);
        // split values & indices
        let data: Vec<_> = sparse_vector.iter().map(|(idx, _)| *idx).collect();
        let sparse_indices = SparseIndices { data };
        let sparse_values = sparse_vector.iter().map(|(_, val)| *val).collect();
        (
            sparse_values,
            Some(sparse_indices),
            SPARSE_VECTOR_NAME.to_string(),
        )
    } else {
        (
            random_dense_vector(vec_dim),
            None,
            DENSE_VECTOR_NAME_SQ.to_string(),
        )
    };

    let filter = random_filter(Some(payload_count));

    let response = client
        .search_points(&SearchPoints {
            collection_name: collection_name.to_string(),
            vector: query_vector,
            vector_name: Some(vector_name),
            filter,
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
        let params = vec![
            (
                SPARSE_VECTOR_NAME.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(true), // in memory
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_INDEX_MMAP.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_ONE.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_TWO.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_THREE.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_FOUR.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_FIVE.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_SIX.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_SEVEN.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_EIGHT.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_NINE.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
            (
                SPARSE_VECTOR_NAME_TEN.to_string(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // mmap index
                    }),
                },
            ),
        ]
        .into_iter()
        .collect();

        Some(SparseVectorConfig { map: params })
    };

    let dense_vectors_config = if !args.only_sparse {
        let hnsw_config = Some(HnswConfigDiff {
            m: Some(32),
            ef_construct: None,
            full_scan_threshold: None,
            max_indexing_threads: None,
            on_disk: Some(false),
            payload_m: None,
        });

        let params: HashMap<_, _> = vec![
            (
                DENSE_VECTOR_NAME_ON_DISK.to_string(),
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config: hnsw_config.clone(),
                    on_disk: Some(true),
                    datatype: None,
                },
            ),
            (
                DENSE_VECTOR_NAME_ROCKSDB.to_string(),
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config: hnsw_config.clone(),
                    on_disk: Some(false), // in memory from rocksdb
                    datatype: None,
                },
            ),
            (
                DENSE_VECTOR_NAME_UINT8.to_string(),
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config: hnsw_config.clone(),
                    on_disk: Some(false),
                    datatype: Some(2), // UInt8
                },
            ),
            (
                DENSE_VECTOR_NAME_SQ.to_string(),
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Scalar(ScalarQuantization {
                            r#type: 1, // Int8
                            quantile: None,
                            always_ram: Some(false),
                        })),
                    }),
                    hnsw_config: hnsw_config.clone(),
                    on_disk: Some(false),
                    datatype: None,
                },
            ),
            (
                DENSE_VECTOR_NAME_PQ.to_string(),
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: 1, // x8
                            always_ram: Some(false),
                        })),
                    }),
                    hnsw_config: hnsw_config.clone(),
                    on_disk: Some(false),
                    datatype: None,
                },
            ),
            (
                DENSE_VECTOR_NAME_BQ.to_string(),
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(BinaryQuantization {
                            always_ram: Some(false),
                        })),
                    }),
                    hnsw_config: hnsw_config.clone(),
                    on_disk: Some(false),
                    datatype: None,
                },
            ),
        ]
        .into_iter()
        .collect();

        Some(VectorsConfig {
            config: Some(ParamsMap(VectorParamsMap { map: params })),
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
                // add all flavors of dense vectors
                vectors_map.insert(
                    DENSE_VECTOR_NAME_ON_DISK.to_string(),
                    random_dense_vector(vec_dim).into(),
                );
                vectors_map.insert(
                    DENSE_VECTOR_NAME_ROCKSDB.to_string(),
                    random_dense_vector(vec_dim).into(),
                );
                vectors_map.insert(
                    DENSE_VECTOR_NAME_UINT8.to_string(),
                    random_dense_vector(vec_dim).into(),
                );
                vectors_map.insert(
                    DENSE_VECTOR_NAME_SQ.to_string(),
                    random_dense_vector(vec_dim).into(),
                );
                vectors_map.insert(
                    DENSE_VECTOR_NAME_PQ.to_string(),
                    random_dense_vector(vec_dim).into(),
                );
                vectors_map.insert(
                    DENSE_VECTOR_NAME_BQ.to_string(),
                    random_dense_vector(vec_dim).into(),
                );
            }

            // always add sparse vectors
            vectors_map.insert(
                SPARSE_VECTOR_NAME.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_INDEX_MMAP.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_ONE.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_TWO.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_THREE.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_FOUR.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_FIVE.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_SIX.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_SEVEN.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_EIGHT.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_NINE.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

            vectors_map.insert(
                SPARSE_VECTOR_NAME_TEN.to_string(),
                random_sparse_vector(vec_dim, 0.1).into(),
            );

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
