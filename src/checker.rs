use crate::client::{get_collection_info, get_exact_points_count, scroll_all_points};
use crate::crasher_error::CrasherError::Invariant;
use ahash::{AHashMap, AHashSet};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, Filter, QueryBatchResponse, ScrollPointsBuilder, VectorOutput,
    vector_output,
};

use crate::crasher_error::CrasherError;
use crate::generators::{
    BOOL_PAYLOAD_KEY, DATETIME_PAYLOAD_KEY, FLOAT_PAYLOAD_KEY, GEO_PAYLOAD_KEY,
    INTEGER_PAYLOAD_KEY, KEYWORD_PAYLOAD_KEY, MANDATORY_PAYLOAD_BOOL_KEY,
    MANDATORY_PAYLOAD_TIMESTAMP_KEY, TEXT_PAYLOAD_KEY, UUID_PAYLOAD_KEY,
};

/// Scroll-based consistency check over the whole collection.
///
/// Iterates every stored point and asserts:
/// - well-formedness of each point (vectors present, expected named vectors, no zeroed vectors,
///   mandatory payload keys);
/// - every id in `[0, expected_present_count)` is present in storage (missing-id check);
/// - no numeric id is `>= max_id_exclusive` (phantom-id check);
/// - no UUID-typed id is returned (the workload only inserts numeric ids);
/// - no id is returned more than once (duplicate check — weak guarantee, since the read
///   path generally dedupes by id, but cross-shard / replication-recovery edge cases can leak).
pub async fn check_points_consistency(
    collection_name: &str,
    client: &Qdrant,
    expected_present_count: usize,
    max_id_exclusive: u64,
    expected_vector_names: &AHashSet<String>,
) -> Result<(), CrasherError> {
    let mut seen: AHashMap<u64, u32> = AHashMap::default();
    let mut uuid_ids: Vec<String> = Vec::new();
    let mut malformed_points_errors: Vec<String> = Vec::new();

    scroll_all_points(client, collection_name, true, true, |point| {
        // Track id occurrences and bucket by type
        match point.id.as_ref().and_then(|p| p.point_id_options.as_ref()) {
            Some(PointIdOptions::Num(id)) => {
                *seen.entry(*id).or_insert(0) += 1;
            }
            Some(PointIdOptions::Uuid(s)) => {
                uuid_ids.push(s.clone());
            }
            None => {
                malformed_points_errors.push("Point with no id field".to_string());
                return;
            }
        }

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
                        if named_vectors.vectors.is_empty() {
                            malformed_points_errors
                                .push(format!("Named vector {point_id:?} should not be empty"));
                        }
                        let present_names: AHashSet<&String> =
                            named_vectors.vectors.keys().collect();
                        for expected_name in expected_vector_names {
                            if !present_names.contains(expected_name) {
                                malformed_points_errors.push(format!(
                                    "Point {point_id:?} is missing expected named vector '{expected_name}'"
                                ));
                            }
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
        // mandatory payload keys
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
    })
    .await?;

    // missing ids in [0, expected_present_count)
    let missing_ids: Vec<u64> = (0..expected_present_count as u64)
        .filter(|id| !seen.contains_key(id))
        .collect();

    // duplicate numeric ids (count > 1)
    let mut duplicates: Vec<(u64, u32)> = seen
        .iter()
        .filter(|(_, c)| **c > 1)
        .map(|(id, c)| (*id, *c))
        .collect();
    duplicates.sort_unstable_by_key(|(id, _)| *id);

    // phantom numeric ids (>= max_id_exclusive)
    let mut phantom_ids: Vec<u64> = seen
        .keys()
        .copied()
        .filter(|id| *id >= max_id_exclusive)
        .collect();
    phantom_ids.sort_unstable();

    let mut errors_found: Vec<String> = Vec::new();
    if !missing_ids.is_empty() {
        errors_found.push(format!(
            "detected {} missing points (expected ids [0, {expected_present_count})):\n{missing_ids:?}",
            missing_ids.len(),
        ));
    }
    if !phantom_ids.is_empty() {
        errors_found.push(format!(
            "detected {} phantom point ids (>= {max_id_exclusive}):\n{phantom_ids:?}",
            phantom_ids.len(),
        ));
    }
    if !uuid_ids.is_empty() {
        errors_found.push(format!(
            "detected {} UUID point ids (only numeric ids are inserted):\n{uuid_ids:?}",
            uuid_ids.len(),
        ));
    }
    if !duplicates.is_empty() {
        errors_found.push(format!(
            "detected {} duplicate point ids (id, occurrences):\n{duplicates:?}",
            duplicates.len(),
        ));
    }
    if !malformed_points_errors.is_empty() {
        errors_found.push(format!(
            "detected {} malformed points:\n{}",
            malformed_points_errors.len(),
            malformed_points_errors.join("\n"),
        ));
    }

    if errors_found.is_empty() {
        Ok(())
    } else {
        Err(Invariant(errors_found.join("\n")))
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

const INDEXED_PAYLOAD_KEYS: &[&str] = &[
    KEYWORD_PAYLOAD_KEY,
    INTEGER_PAYLOAD_KEY,
    FLOAT_PAYLOAD_KEY,
    GEO_PAYLOAD_KEY,
    TEXT_PAYLOAD_KEY,
    BOOL_PAYLOAD_KEY,
    DATETIME_PAYLOAD_KEY,
    UUID_PAYLOAD_KEY,
];

/// Check that every indexed payload field's total (non_empty + empty) equals the unfiltered
/// point count. A divergence indicates index corruption after a crash.
///
/// Stronger than a pairwise inter-index agreement check: if a bug drops the same record
/// from every index, all indexes still agree with each other but disagree with the oracle.
pub async fn check_payload_indexes_consistency(
    collection_name: &str,
    client: &Qdrant,
) -> Result<(), CrasherError> {
    let total_count = get_exact_points_count(client, collection_name).await? as u64;

    let mut index_totals: Vec<(&str, u64, u64)> = Vec::new();
    for &field_name in INDEXED_PAYLOAD_KEYS {
        let non_empty_count = client
            .count(
                CountPointsBuilder::new(collection_name)
                    .filter(Filter::must_not([Condition::is_empty(field_name)]))
                    .exact(true),
            )
            .await?
            .result
            .unwrap()
            .count;

        let empty_count = client
            .count(
                CountPointsBuilder::new(collection_name)
                    .filter(Filter::must([Condition::is_empty(field_name)]))
                    .exact(true),
            )
            .await?
            .result
            .unwrap()
            .count;

        index_totals.push((field_name, non_empty_count, empty_count));
    }

    let mismatches: Vec<&(&str, u64, u64)> = index_totals
        .iter()
        .filter(|(_, non_empty, empty)| non_empty + empty != total_count)
        .collect();

    if mismatches.is_empty() {
        Ok(())
    } else {
        let details: Vec<String> = mismatches
            .iter()
            .map(|(name, non_empty, empty)| {
                format!(
                    "'{name}': non_empty({non_empty}) + empty({empty}) = {} (expected {total_count})",
                    non_empty + empty
                )
            })
            .collect();
        Err(Invariant(format!(
            "Payload indexes disagree with total point count ({total_count}):\n{}",
            details.join("\n")
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

pub async fn check_optimizer_status(
    collection_name: &str,
    client: &Qdrant,
) -> Result<(), CrasherError> {
    let info = get_collection_info(client, collection_name).await?;
    if let Some(optimizer_status) = &info.optimizer_status
        && !optimizer_status.ok
    {
        return Err(Invariant(format!(
            "Optimizer status is red: {}",
            optimizer_status.error
        )));
    }
    Ok(())
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
