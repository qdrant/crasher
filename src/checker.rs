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
    MANDATORY_PAYLOAD_FLOAT_KEY, MANDATORY_PAYLOAD_GEO_KEY, MANDATORY_PAYLOAD_INTEGER_KEY,
    MANDATORY_PAYLOAD_KEYWORD_KEY, MANDATORY_PAYLOAD_TEXT_KEY, MANDATORY_PAYLOAD_TIMESTAMP_KEY,
    MANDATORY_PAYLOAD_UUID_KEY, TEXT_PAYLOAD_KEY, UUID_PAYLOAD_KEY,
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

    scroll_all_points(client, collection_name, None, true, true, |point| {
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

/// Check the indexed crasher payload fields for internal consistency. The workload sets all of
/// them together on the same points (see `set payload` in the workload), so the assertions are
/// crash-safe regardless of how many points have been set — they hold whether zero, some, or all
/// points carry the payload, which matters because the check also runs after a crash that may have
/// interrupted the set-payload phase. We deliberately do *not* assert the fully-set distribution
/// (e.g. "exactly half are non-empty"), since a partially-applied set-payload is a valid recovered
/// state. For every field we require:
///
/// 1. all fields report an identical `non_empty` count, and
/// 2. all fields report an identical `empty` count — they are set as a unit, so a per-field
///    divergence is real index corruption, not benign lag; and
/// 3. `non_empty + empty` is never *below* the point count — a deficit means points were lost from
///    both buckets of a field's index. We do not cap the overshoot: `is_empty` and
///    `must_not(is_empty)` are complements that should sum to the point count, but under
///    copy-on-write `set_payload` copies a point from a non-appendable segment into an appendable
///    one (new copy carries the value → `non_empty`) and retires the stale pre-set copy (empty →
///    `empty`); after a crash that retirement can lag, double-counting the point until the
///    optimizer's dedup pass scrubs it. That overshoot is benign and its size depends on how far
///    dedup has progressed, so bounding it is just flakiness with no corruption-detecting value —
///    the deficit half is the only sharp signal here. The exact cross-field agreement in (1)&(2)
///    is what actually catches index corruption.
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

    let mut errors: Vec<String> = Vec::new();

    // (1) & (2) All fields are set as a unit, so they must agree on both counts. A per-field
    // divergence is index corruption, not the uniform lag of copy-on-write deduplication.
    let (_, first_non_empty, first_empty) = index_totals[0];
    if index_totals.iter().any(|(_, ne, _)| *ne != first_non_empty) {
        let listing = index_totals
            .iter()
            .map(|(name, ne, _)| format!("'{name}': non_empty({ne})"))
            .collect::<Vec<_>>()
            .join(", ");
        errors.push(format!(
            "non_empty counts disagree across fields (must be identical): {listing}"
        ));
    }
    if index_totals.iter().any(|(_, _, e)| *e != first_empty) {
        let listing = index_totals
            .iter()
            .map(|(name, _, e)| format!("'{name}': empty({e})"))
            .collect::<Vec<_>>()
            .join(", ");
        errors.push(format!(
            "empty counts disagree across fields (must be identical): {listing}"
        ));
    }

    // (3) non_empty + empty must never fall below the point count (points lost from both buckets of
    // a field's index). The overshoot is left unbounded: it is benign copy-on-write residue whose
    // size only reflects how far the optimizer's dedup pass has progressed, so capping it adds
    // flakiness without catching corruption.
    for (name, non_empty, empty) in &index_totals {
        let sum = non_empty + empty;
        if sum < total_count {
            errors.push(format!(
                "'{name}': non_empty({non_empty}) + empty({empty}) = {sum} below point count {total_count} (points lost from both buckets)"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(Invariant(format!(
            "Payload indexes disagree with point count ({total_count}):\n{}",
            errors.join("\n")
        )))
    }
}

/// Mandatory indexed payload keys that the workload always sets via `add_mandatory_payload`.
/// Each one has a corresponding field index, so a divergence between "everyone has this key"
/// (the workload guarantee) and `count(is_empty(key))` (what the index reports) is a bug.
///
/// `MANDATORY_PAYLOAD_BOOL_KEY` and `MANDATORY_PAYLOAD_TIMESTAMP_KEY` already have dedicated
/// checks (`check_filter_bool_index`, `check_filter_null_index`) that go further than just
/// "non-empty"; they're not duplicated here.
const MANDATORY_INDEXED_PAYLOAD_KEYS: &[&str] = &[
    MANDATORY_PAYLOAD_KEYWORD_KEY,
    MANDATORY_PAYLOAD_INTEGER_KEY,
    MANDATORY_PAYLOAD_FLOAT_KEY,
    MANDATORY_PAYLOAD_GEO_KEY,
    MANDATORY_PAYLOAD_TEXT_KEY,
    MANDATORY_PAYLOAD_UUID_KEY,
];

/// For each mandatory indexed payload key, assert `count(is_empty(key)) == 0`.
///
/// The workload writes a deterministic value for every mandatory key on every point, so any
/// non-zero count from `is_empty` reflects an index that lost — or never indexed — the
/// underlying value. This is a tighter constraint than the sum-based check in
/// [`check_payload_indexes_consistency`], which can pass when records are miscategorised
/// between the empty/non-empty buckets without the sum changing.
pub async fn check_mandatory_payload_keys_present(
    collection_name: &str,
    client: &Qdrant,
) -> Result<(), CrasherError> {
    let mut errors: Vec<String> = Vec::new();

    for &key in MANDATORY_INDEXED_PAYLOAD_KEYS {
        let empty_count = client
            .count(
                CountPointsBuilder::new(collection_name)
                    .filter(Filter::must([Condition::is_empty(key)]))
                    .exact(true),
            )
            .await?
            .result
            .unwrap()
            .count;

        if empty_count != 0 {
            errors.push(format!(
                "'{key}': {empty_count} points report is_empty (expected 0)"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(Invariant(format!(
            "Mandatory payload keys missing on some points:\n{}",
            errors.join("\n"),
        )))
    }
}

/// Cross-check that `count(filter)` and `scroll(filter)` return the same number of points.
///
/// Catches divergence between the index-driven count path and the storage-iteration scroll
/// path: a payload-index bug that drops records from one but not the other would slip past
/// every other check (the indexes can still agree on totals — see
/// [`check_payload_indexes_consistency`] — yet count and scroll disagree).
///
/// Uses a small set of stable filters so it doesn't need an RNG. Both filters target the
/// mandatory payload keys and have well-known expected results across the collection.
pub async fn check_count_scroll_parity(
    collection_name: &str,
    client: &Qdrant,
) -> Result<(), CrasherError> {
    // Every inserted point sets MANDATORY_PAYLOAD_BOOL_KEY=true and a non-null timestamp,
    // so case (a) should match all points and case (b) should match none — but the value of
    // the parity check holds regardless of the absolute counts.
    let cases: [(&str, Filter); 2] = [
        (
            "match-mandatory-bool-true",
            Filter::must([Condition::matches(MANDATORY_PAYLOAD_BOOL_KEY, true)]),
        ),
        (
            "is-empty-mandatory-timestamp",
            Filter::must([Condition::is_empty(MANDATORY_PAYLOAD_TIMESTAMP_KEY)]),
        ),
    ];

    let mut errors: Vec<String> = Vec::new();
    for (name, filter) in cases {
        let count = client
            .count(
                CountPointsBuilder::new(collection_name)
                    .filter(filter.clone())
                    .exact(true),
            )
            .await?
            .result
            .unwrap()
            .count;

        let mut scroll_count: u64 = 0;
        scroll_all_points(
            client,
            collection_name,
            Some(filter),
            false,
            false,
            |_point| {
                scroll_count += 1;
            },
        )
        .await?;

        if count != scroll_count {
            errors.push(format!(
                "filter '{name}': count={count}, scroll={scroll_count}"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(Invariant(format!(
            "Count/scroll filter parity violations:\n{}",
            errors.join("\n"),
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

/// Sanity-check a batched query response.
///
/// Per scored point: score must be finite, returned vectors must not be all-zero.
/// Per per-query result: scores must be monotonic (either non-increasing or non-decreasing) —
/// the direction is not asserted because the workload mixes Cosine/Dot (higher = better)
/// and Euclid/Manhattan (lower = better) named vectors. Both directions are valid; what's
/// invalid is a zig-zag, which would indicate the result list isn't sorted at all.
pub fn check_search_result(results: &QueryBatchResponse) -> Result<(), CrasherError> {
    let mut errors: Vec<String> = Vec::new();

    for (query_idx, batch_result) in results.result.iter().enumerate() {
        for (rank, point) in batch_result.result.iter().enumerate() {
            // finite score
            if !point.score.is_finite() {
                errors.push(format!(
                    "query #{query_idx} rank #{rank}: non-finite score {} for point {:?}",
                    point.score, point.id,
                ));
            }
            // zeroed vectors
            if let Some(vectors) = point
                .vectors
                .as_ref()
                .and_then(|v| v.vectors_options.as_ref())
            {
                let zeroed_vector = match vectors {
                    VectorsOptions::Vector(v) => {
                        check_zeroed_vector(v).then_some((String::new(), v))
                    }
                    VectorsOptions::Vectors(vectors) => vectors
                        .vectors
                        .iter()
                        .find_map(|(name, v)| check_zeroed_vector(v).then_some((name.clone(), v))),
                };
                if let Some((name, vector)) = zeroed_vector {
                    errors.push(format!(
                        "query #{query_idx} rank #{rank}: zeroed vector '{name}' on point {:?}: {vector:?}",
                        point.id,
                    ));
                }
            }
        }

        // monotonicity (only when every score is finite — otherwise comparison is meaningless)
        let scores: Vec<f32> = batch_result.result.iter().map(|p| p.score).collect();
        if scores.len() >= 2 && scores.iter().all(|s| s.is_finite()) {
            let non_increasing = scores.windows(2).all(|w| w[0] >= w[1]);
            let non_decreasing = scores.windows(2).all(|w| w[0] <= w[1]);
            if !non_increasing && !non_decreasing {
                errors.push(format!(
                    "query #{query_idx}: scores not monotonic: {scores:?}"
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(Invariant(format!(
            "Search result violations:\n{}",
            errors.join("\n"),
        )))
    }
}
