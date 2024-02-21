use core::option::Option;
use core::option::Option::{None, Some};
use qdrant_client::client::Payload;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{FieldCondition, Filter, Match};
use rand::Rng;

pub const DENSE_VECTOR_NAME: &str = "dense-vector";
pub const SPARSE_VECTOR_NAME: &str = "sparse-vector";
pub const KEYWORD_PAYLOAD_KEY: &str = "a";

pub fn random_keyword(num_variants: usize) -> String {
    let mut rng = rand::thread_rng();
    let variant = rng.gen_range(0..num_variants);
    format!("keyword_{}", variant)
}

pub fn random_payload(keywords: Option<usize>) -> Payload {
    let mut payload = Payload::new();
    if let Some(keyword_variants) = keywords {
        if keyword_variants > 0 {
            payload.insert(KEYWORD_PAYLOAD_KEY, random_keyword(keyword_variants));
        }
    }
    payload
}

pub fn random_filter(keywords: Option<usize>) -> Option<Filter> {
    let mut filter = Filter {
        should: vec![],
        must: vec![],
        must_not: vec![],
    };
    let mut have_any = false;
    if let Some(keyword_variants) = keywords {
        have_any = true;
        filter.must.push(
            FieldCondition {
                key: KEYWORD_PAYLOAD_KEY.to_string(),
                r#match: Some(Match {
                    match_value: Some(MatchValue::Keyword(random_keyword(keyword_variants))),
                }),
                range: None,
                geo_bounding_box: None,
                geo_radius: None,
                values_count: None,
                geo_polygon: None,
            }
            .into(),
        )
    }
    if have_any {
        Some(filter)
    } else {
        None
    }
}

pub fn random_dense_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

pub fn random_sparse_vector(max_size: usize, sparsity: f64) -> Vec<(u32, f32)> {
    let mut rng = rand::thread_rng();
    let size = rng.gen_range(1..max_size);
    // (index, value)
    let mut pairs = Vec::with_capacity(size);
    for i in 1..=size {
        // probability of skipping a dimension to make the vectors sparse
        let skip = !rng.gen_bool(sparsity);
        if skip {
            continue;
        }
        // Only positive values are generated to make sure to hit the pruning path.
        pairs.push((i as u32, rng.gen_range(0.0..10.0) as f32));
    }
    pairs
}
