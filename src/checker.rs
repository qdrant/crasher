use std::sync::Arc;

use crate::client::retrieve_points;
use crate::crasher_error::CrasherError::Invariant;
use ahash::AHashSet;
use futures::stream::{self, StreamExt, TryStreamExt};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::qdrant::{
    Condition, Filter, QueryBatchResponse, ScrollPointsBuilder, VectorOutput, vector_output,
};

use crate::crasher_error::CrasherError;
use crate::generators::{MANDATORY_PAYLOAD_BOOL_KEY, MANDATORY_PAYLOAD_TIMESTAMP_KEY};

/// Vector data consistency checker for id range
pub async fn check_points_consistency(
    collection_name: &str,
    client: &Qdrant,
    current_count: usize,
) -> Result<(), CrasherError> {
    // fetch all existing points (rely on numeric ids!)
    let all_ids: Vec<_> = (0..current_count).collect();

    // track all errors
    let mut missing_points_errors: Vec<usize> = Vec::new();
    let mut malformed_points_errors: Vec<String> = Vec::new();

    // Stream checks by batches to not overload the server
    let min_batch_size = 1_000;
    let ids_batch: Vec<Arc<[usize]>> = all_ids
        .chunks(current_count.min(min_batch_size))
        .map(Arc::from)
        .collect();
    let mut retrieve_points_f = stream::iter(ids_batch)
        .map(|ids| async {
            let resp = retrieve_points(client, collection_name, &ids).await?;
            Ok::<_, CrasherError>((ids, resp))
        })
        .buffered(1);

    while let Some((expected_ids, response)) = retrieve_points_f.try_next().await? {
        // check if missing points
        if response.result.len() != expected_ids.len() {
            let response_ids = response
                .result
                .iter()
                .map(|point| point.id.clone().unwrap().point_id_options.unwrap())
                .map(|id| match id {
                    PointIdOptions::Num(id) => id as usize,
                    PointIdOptions::Uuid(_) => panic!("UUID in the response"),
                })
                .collect::<AHashSet<_>>();

            let missing_ids = expected_ids
                .iter()
                .filter(|&id| !response_ids.contains(id))
                .copied()
                .collect::<Vec<_>>();
            missing_points_errors.extend_from_slice(&missing_ids);
        }
        // check points are well-formed
        for point in &response.result {
            let point_id = point.id.as_ref().expect("Point id should be present");
            // well-formed vectors
            if let Some(vectors) = &point.vectors {
                if let Some(vector) = &vectors.vectors_options {
                    match vector {
                        VectorsOptions::Vector(anonymous) => {
                            malformed_points_errors.push(format!(
                                "Vector {point_id:?} should be named: {anonymous:?}"
                            ));
                        }
                        VectorsOptions::Vectors(named_vectors) => {
                            // TODO check that all generated named vectors are present
                            if named_vectors.vectors.is_empty() {
                                malformed_points_errors
                                    .push(format!("Named vector {point_id:?} should not be empty"));
                            }
                            for (name, vector) in &named_vectors.vectors {
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
            "detected {} missing points:\n{:?}",
            missing_points_errors.len(),
            missing_points_errors
        ));
    }

    if !malformed_points_errors.is_empty() {
        errors_found.push(format!(
            "detected {} malformed points:\n{:?}",
            malformed_points_errors.len(),
            malformed_points_errors.join("\n")
        ));
    }

    if errors_found.is_empty() {
        Ok(())
    } else {
        let errors_rendered = errors_found.join("\n");
        Err(Invariant(errors_rendered))
    }
}

pub async fn check_filter_null_index(
    collection_name: &str,
    client: &Qdrant,
    current_count: usize,
) -> Result<(), CrasherError> {
    let resp = client
        .scroll(
            ScrollPointsBuilder::new(collection_name)
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
            "detected {} points missing the '{}' payload key when matching for null values:\n{:?}",
            points.len(),
            MANDATORY_PAYLOAD_TIMESTAMP_KEY,
            points,
        )))
    }
}

pub async fn check_filter_bool_index(
    collection_name: &str,
    client: &Qdrant,
    current_count: usize,
) -> Result<(), CrasherError> {
    let resp = client
        .scroll(
            ScrollPointsBuilder::new(collection_name)
                .filter(Filter::must([Condition::matches(
                    MANDATORY_PAYLOAD_BOOL_KEY,
                    true,
                )]))
                .limit(current_count as u32),
        )
        .await?;
    let points: AHashSet<_> = resp
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
            "detected {} points missing the '{}: true' when matching payload:\n{:?}",
            missing_points.len(),
            MANDATORY_PAYLOAD_BOOL_KEY,
            missing_points,
        )))
    }
}

/// Checks if this is a zeroed vector.
pub fn check_zeroed_vector(vector: &VectorOutput) -> bool {
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
        .unwrap_or_else(|| false)
}

pub fn check_search_result(results: &QueryBatchResponse) -> Result<(), CrasherError> {
    // assert no vector is only containing zeros
    for point in results.result.iter().flat_map(|result| &result.result) {
        if let Some(vectors) = point
            .vectors
            .as_ref()
            .and_then(|v| v.vectors_options.as_ref())
        {
            let zeroed_vector = match vectors {
                VectorsOptions::Vector(v) => check_zeroed_vector(v).then_some((String::new(), v)),
                VectorsOptions::Vectors(vectors) => vectors
                    .vectors
                    .iter()
                    .find_map(|(name, v)| check_zeroed_vector(v).then_some((name.clone(), v))),
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
