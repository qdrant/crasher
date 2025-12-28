use crate::COLLECTION_NAME;
use crate::args::Args;
use crate::crasher_error::CrasherError;
use crate::crasher_error::CrasherError::Cancelled;
use crate::generators::{
    MANDATORY_PAYLOAD_BOOL_KEY, MANDATORY_PAYLOAD_TIMESTAMP_KEY, TestNamedVectors,
    random_dense_vector, random_filter, random_payload, random_sparse_vector,
};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::Filter;
use qdrant_client::qdrant::payload_index_params::IndexParams;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors_config::Config::ParamsMap;
use qdrant_client::qdrant::{
    CollectionInfo, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    CreateSnapshotResponse, DeletePointsBuilder, DeleteSnapshotRequestBuilder,
    DeleteSnapshotResponse, FieldType, GetPointsBuilder, GetResponse, OptimizersConfigDiff,
    PointId, PointStruct, Query, QueryBatchPointsBuilder, QueryBatchResponse, QueryPoints,
    ReplicaState, SetPayloadPointsBuilder, SparseVectorConfig, UpsertPointsBuilder, Vector,
    VectorInput, VectorParamsMap, Vectors, VectorsConfig, WriteOrdering,
};
use rand::Rng;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time::sleep;

/// Wait for collection to be indexed
pub async fn wait_server_ready(
    client: &Qdrant,
    stopped: Arc<AtomicBool>,
    wait_for_replicas: bool,
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
            Err(err) => {
                if start.elapsed().as_secs_f64() > 60.0 {
                    return Err(CrasherError::Invariant(
                        "Server did not start in time, /readyz not ready".to_string(),
                    ));
                } else {
                    log::debug!("Healthcheck failed: {err:?}")
                }
            }
        }
    }

    if wait_for_replicas {
        loop {
            if stopped.load(Ordering::Relaxed) {
                return Err(Cancelled);
            }

            let cluster_info = client.collection_cluster_info(COLLECTION_NAME).await?;

            let all_local_active = cluster_info
                .local_shards
                .iter()
                .all(|shard| shard.state == ReplicaState::Active as i32);
            let all_remote_active = cluster_info
                .remote_shards
                .iter()
                .all(|shard| shard.state == ReplicaState::Active as i32);

            if all_local_active && all_remote_active {
                break;
            }

            // Wait up to 3 minutes for recovery to finish
            if start.elapsed().as_secs_f64() > 180.0 {
                return Err(CrasherError::Invariant(
                    "Server did not start in time, not all replicas are active".to_string(),
                ));
            }

            sleep(Duration::from_secs(1)).await;
        }
    }

    Ok(start.elapsed().as_secs_f64())
}

/// Get points count
pub async fn get_collection_info(
    client: &Qdrant,
    collection_name: &str,
) -> Result<CollectionInfo, CrasherError> {
    let collection_info = client
        .collection_info(collection_name)
        .await?
        .result
        .unwrap();
    Ok(collection_info)
}

/// Get points count from collection info
#[allow(dead_code)]
pub async fn get_info_points_count(
    client: &Qdrant,
    collection_name: &str,
) -> Result<usize, CrasherError> {
    let point_count = client
        .collection_info(collection_name)
        .await?
        .result
        .unwrap()
        .points_count;
    Ok(point_count.unwrap_or(0) as usize)
}

/// Get points count
pub async fn get_exact_points_count(
    client: &Qdrant,
    collection_name: &str,
) -> Result<usize, CrasherError> {
    let point_count = client
        .count(CountPointsBuilder::new(collection_name).exact(true))
        .await?
        .result
        .unwrap()
        .count;
    Ok(point_count as usize)
}

/// Query points
#[allow(clippy::too_many_arguments)]
pub async fn query_batch_points(
    client: &Qdrant,
    collection_name: &str,
    test_named_vectors: &TestNamedVectors,
    only_sparse: bool,
    vec_dim: u32,
    payload_count: u32,
    with_vector: bool,
    limit: u64,
    rng: &mut impl Rng,
) -> Result<QueryBatchResponse, CrasherError> {
    let request_filter = random_filter(rng, Some(payload_count));
    let mut requests = vec![];
    // sparse search requests
    for sparse_name in test_named_vectors.sparse_names() {
        let sparse_vector = random_sparse_vector(rng, vec_dim, 0.1);
        // split values & indices
        let sparse_indices: Vec<_> = sparse_vector.iter().map(|(idx, _)| *idx).collect();
        let sparse_values: Vec<_> = sparse_vector.iter().map(|(_, val)| *val).collect();
        let query_vector = VectorInput::new_sparse(sparse_indices, sparse_values);
        let query_nearest = Query::new_nearest(query_vector);
        let request = QueryPoints {
            collection_name: collection_name.to_string(),
            prefetch: vec![],
            query: Some(query_nearest),
            using: Some(sparse_name),
            filter: request_filter.clone(),
            limit: Some(limit),
            with_payload: Some(true.into()),
            params: None,
            score_threshold: None,
            offset: None,
            with_vectors: Some(with_vector.into()),
            read_consistency: None,
            timeout: None,
            shard_key_selector: None,
            lookup_from: None,
        };
        requests.push(request);
    }

    // dense search requests
    if !only_sparse {
        // dense
        for dense_name in test_named_vectors.dense_names() {
            let query_vector =
                VectorInput::new_dense(random_dense_vector(rng, &dense_name, vec_dim));
            let query_nearest = Query::new_nearest(query_vector);
            let request = QueryPoints {
                collection_name: collection_name.to_string(),
                prefetch: vec![],
                query: Some(query_nearest),
                using: Some(dense_name.clone()),
                filter: request_filter.clone(),
                limit: Some(limit),
                with_payload: Some(true.into()),
                params: None,
                score_threshold: None,
                offset: None,
                with_vectors: Some(with_vector.into()),
                read_consistency: None,
                timeout: None,
                shard_key_selector: None,
                lookup_from: None,
            };
            requests.push(request);
        }
        // multi dense
        for multi_dense_name in test_named_vectors.multi_names() {
            let vec_count = rng.random_range(1..5);
            let multi_vector: Vec<_> = (0..vec_count)
                .map(|_| random_dense_vector(rng, &multi_dense_name, vec_dim))
                .collect();
            let query_vector = VectorInput::new_multi(multi_vector);
            let query_nearest = Query::new_nearest(query_vector);
            let request = QueryPoints {
                collection_name: collection_name.to_string(),
                prefetch: vec![],
                query: Some(query_nearest),
                using: Some(multi_dense_name.clone()),
                filter: request_filter.clone(),
                limit: Some(limit),
                with_payload: Some(true.into()),
                params: None,
                score_threshold: None,
                offset: None,
                with_vectors: Some(with_vector.into()),
                read_consistency: None,
                timeout: None,
                shard_key_selector: None,
                lookup_from: None,
            };
            requests.push(request);
        }
    }

    let response = client
        .query_batch(QueryBatchPointsBuilder::new(collection_name, requests))
        .await?;

    Ok(response)
}

/// Create collection
pub async fn create_collection(
    client: &Qdrant,
    collection_name: &str,
    test_named_vectors: &TestNamedVectors,
    args: Arc<Args>,
) -> Result<(), CrasherError> {
    let sparse_vector_params = test_named_vectors.sparse_params();
    let sparse_vectors_config = SparseVectorConfig {
        map: sparse_vector_params,
    };

    let dense_vectors_config = if !args.only_sparse {
        let mut all_dense_params = HashMap::new();

        // dense vectors
        let dense_vector_params = test_named_vectors.dense_params();
        all_dense_params.extend(dense_vector_params);

        // multi dense vectors
        let multi_dense_vector_params = test_named_vectors.multi_dense_params();
        all_dense_params.extend(multi_dense_vector_params);

        Some(VectorsConfig {
            config: Some(ParamsMap(VectorParamsMap {
                map: all_dense_params,
            })),
        })
    } else {
        None
    };

    let mut request = CreateCollectionBuilder::new(collection_name)
        .sparse_vectors_config(sparse_vectors_config)
        .replication_factor(args.replication_factor)
        .write_consistency_factor(args.write_consistency_factor)
        .shard_number(args.shard_count)
        .optimizers_config(OptimizersConfigDiff {
            deleted_threshold: None,
            vacuum_min_vector_number: None,
            default_segment_number: Some(args.segment_count as u64), // to force constant merges
            indexing_threshold: args.indexing_threshold.map(|i| i as u64),
            flush_interval_sec: Some(args.flush_interval_sec as u64),
            memmap_threshold: args.memmap_threshold.map(|i| i as u64),
            max_segment_size: None,
            max_optimization_threads: None,
            deprecated_max_optimization_threads: None,
        })
        .on_disk_payload(args.on_disk_payload);
    if let Some(dense_vectors_config) = dense_vectors_config {
        request = request.vectors_config(dense_vectors_config);
    }

    client.create_collection(request).await?;
    Ok(())
}

/// insert points into collection (blocking)
///
/// Split into upload into batches.
/// Only wait for the last batch to finish.
#[allow(clippy::too_many_arguments)]
pub async fn insert_points_batch(
    client: &Qdrant,
    collection_name: &str,
    points_count: u32,
    vec_dim: u32,
    payload_count: u32,
    mandatory_payload: bool,
    only_sparse_vectors: bool,
    test_named_vectors: &TestNamedVectors,
    write_ordering: Option<WriteOrdering>,
    stopped: Arc<AtomicBool>,
    rng: &mut impl Rng,
    on_batch_confirmation: impl Fn(u64) + Send + Sync,
) -> Result<(), CrasherError> {
    let max_batch_size = 10;
    // handle less than batch & spill over
    let (batch_size, num_batches, last_batch_size) = if points_count <= max_batch_size {
        (points_count, 1, points_count)
    } else {
        let remainder = points_count % max_batch_size;
        let div = points_count / max_batch_size;
        let num_batches = div + u32::from(remainder > 0);
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
        let mut points = Vec::with_capacity(batch_size as usize);
        let batch_base_id = batch_id as u64 * max_batch_size as u64;
        for i in 0..batch_size {
            let idx = batch_base_id + i as u64;
            let point_id = PointId {
                point_id_options: Some(PointIdOptions::Num(idx)),
            };
            let mut vectors_map: HashMap<String, Vector> = HashMap::new();

            if !only_sparse_vectors {
                // dense
                for name in test_named_vectors.dense_names() {
                    vectors_map.insert(
                        name.clone(),
                        Vector::new_dense(random_dense_vector(rng, &name, vec_dim)),
                    );
                }
                // multi dense
                for name in test_named_vectors.multi_names() {
                    let vec_count = rng.random_range(1..5);
                    let multi_vector: Vec<Vec<_>> = (0..vec_count)
                        .map(|_| random_dense_vector(rng, &name, vec_dim))
                        .collect();
                    vectors_map.insert(name.clone(), Vector::new_multi(multi_vector));
                }
            }

            // always add sparse vectors
            for name in test_named_vectors.sparse_names() {
                vectors_map.insert(name.clone(), random_sparse_vector(rng, vec_dim, 0.1).into());
            }

            let vectors: Vectors = vectors_map.into();

            let mut payload = random_payload(rng, Some(payload_count));

            if mandatory_payload {
                payload.insert(
                    MANDATORY_PAYLOAD_TIMESTAMP_KEY,
                    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                );
                payload.insert(MANDATORY_PAYLOAD_BOOL_KEY, true);
            }

            points.push(PointStruct::new(point_id, vectors, payload));
        }
        if stopped.load(Ordering::Relaxed) {
            return Err(Cancelled);
        }

        let inserting_points: Vec<_> = points
            .iter()
            .map(
                |p| match p.id.as_ref().unwrap().point_id_options.as_ref().unwrap() {
                    PointIdOptions::Num(id) => *id,
                    PointIdOptions::Uuid(_) => unreachable!("UUIDs are not supported"),
                },
            )
            .collect::<Vec<_>>();

        log::info!("Inserting points: {inserting_points:?}");

        let max_point_id = inserting_points.iter().max().unwrap();

        let _resp = client
            .upsert_points(
                UpsertPointsBuilder::new(collection_name, points)
                    .ordering(write_ordering.unwrap_or_default())
                    .wait(wait),
            )
            .await?;
        // Assume response is successful if we are here
        on_batch_confirmation(*max_point_id);
    }
    Ok(())
}

pub async fn create_field_index(
    client: &Qdrant,
    collection_name: &str,
    field_name: &str,
    field_type: FieldType,
    field_index_params: impl Into<IndexParams>,
) -> Result<(), CrasherError> {
    client
        .create_field_index(
            CreateFieldIndexCollectionBuilder::new(collection_name, field_name, field_type)
                .field_index_params(field_index_params.into())
                .wait(true),
        )
        .await?;
    Ok(())
}

/// Set payload (blocking)
///
/// Expects the `point_id` to exist.
pub async fn set_payload(
    client: &Qdrant,
    collection_name: &str,
    point_id: u64,
    payload_count: u32,
    write_ordering: Option<WriteOrdering>,
    rng: &mut impl Rng,
) -> Result<(), CrasherError> {
    let payload = random_payload(rng, Some(payload_count));

    let points_id_selector = vec![PointId {
        point_id_options: Some(PointIdOptions::Num(point_id)),
    }];

    let _resp = client
        .set_payload(
            SetPayloadPointsBuilder::new(collection_name, payload)
                .points_selector(points_id_selector)
                .ordering(write_ordering.unwrap_or_default())
                .wait(true),
        )
        .await?;

    Ok(())
}

/// Retrieve points with vectors and payload
pub async fn retrieve_points(
    client: &Qdrant,
    collection_name: &str,
    ids: &[usize],
) -> Result<GetResponse, CrasherError> {
    let response = client
        .get_points(
            GetPointsBuilder::new(
                collection_name,
                ids.iter().map(|id| (*id as u64).into()).collect::<Vec<_>>(),
            )
            .with_vectors(true)
            .with_payload(true),
        )
        .await?;

    Ok(response)
}

/// delete points (blocking)
pub async fn delete_points(client: &Qdrant, collection_name: &str) -> Result<(), CrasherError> {
    let points_selector = Filter::must([]);

    // delete all points
    client
        .delete_points(
            DeletePointsBuilder::new(collection_name)
                .points(points_selector)
                .wait(true),
        )
        .await?;
    Ok(())
}

/// Trigger collection snapshot
pub async fn create_collection_snapshot(
    client: &Qdrant,
    collection_name: &str,
) -> Result<CreateSnapshotResponse, CrasherError> {
    let collection_snapshot_info = client.create_snapshot(collection_name).await?;
    Ok(collection_snapshot_info)
}

/// Delete collection snapshot
pub async fn delete_collection_snapshot(
    client: &Qdrant,
    collection_name: &str,
    snapshot_name: &str,
) -> Result<DeleteSnapshotResponse, CrasherError> {
    let collection_snapshot_info = client
        .delete_snapshot(DeleteSnapshotRequestBuilder::new(
            collection_name,
            snapshot_name,
        ))
        .await?;
    Ok(collection_snapshot_info)
}

/// List collection snapshots
pub async fn list_collection_snapshots(
    client: &Qdrant,
    collection_name: &str,
) -> Result<Vec<String>, CrasherError> {
    let snapshots = client
        .list_snapshots(collection_name)
        .await?
        .snapshot_descriptions;
    Ok(snapshots
        .into_iter()
        .map(|s| s.name)
        .collect::<Vec<String>>())
}

// TODO add to config
const HTTP_PORT: u32 = 6333;

/// Restore local collection snapshot
pub async fn restore_collection_snapshot(
    collection_name: &str,
    snapshot_name: &str,
    client: &reqwest::Client,
) -> Result<(), CrasherError> {
    let url =
        format!("http://localhost:{HTTP_PORT}/collections/{collection_name}/snapshots/recover");

    // setup snapshot location
    let body = json!({
        "location": format!(
            "http://localhost:{HTTP_PORT}/collections/{collection_name}/snapshots/{snapshot_name}"
        )
    });

    let response = client.put(&url).json(&body).send().await?;
    let status = response.status();
    if !status.is_success() {
        let response_text = response.text().await?;
        return Err(CrasherError::Invariant(format!(
            "Invalid snapshot restore - status:{status} '{response_text}'"
        )));
    }
    Ok(())
}

pub async fn get_telemetry(client: &reqwest::Client) -> Result<String, CrasherError> {
    let url = format!("http://localhost:{HTTP_PORT}/telemetry?details_level=10");
    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(CrasherError::Invariant(
            "Failed to get telemetry".to_string(),
        ));
    }
    let response_body = response.text().await?;
    let json: Value = serde_json::from_str(&response_body)?;
    let pretty = serde_json::to_string_pretty(&json)?;
    Ok(pretty)
}
