use core::option::Option;
use core::option::Option::{None, Some};
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    BinaryQuantization, Distance, FieldCondition, Filter, HnswConfigDiff, Match, MultiVectorConfig,
    ProductQuantization, QuantizationConfig, ScalarQuantization, SparseIndexConfig,
    SparseVectorParams, VectorParams,
};
use qdrant_client::Payload;
use rand::Rng;
use serde_json::json;
use std::collections::HashMap;
// TODO do not generate vector configuration manually but create all possibility exhaustively

/// Dense vectors base names
pub const DENSE_VECTOR_NAME_ON_DISK: &str = "dense-vector-on-disk";
pub const DENSE_VECTOR_NAME_ROCKSDB: &str = "dense-vector-rocksdb";
pub const DENSE_VECTOR_NAME_UINT8: &str = "dense-vector-uint8";
pub const DENSE_VECTOR_NAME_FLOAT16: &str = "dense-vector-float16";
pub const DENSE_VECTOR_NAME_SQ: &str = "dense-vector-sq";
pub const DENSE_VECTOR_NAME_PQ: &str = "dense-vector-pq";
pub const DENSE_VECTOR_NAME_BQ: &str = "dense-vector-bq";

/// Sparse vectors base names
pub const SPARSE_VECTOR_NAME_INDEX_DISK: &str = "sparse-vector-index-disk";
pub const SPARSE_VECTOR_NAME_INDEX_MEMORY: &str = "sparse-vector-index-memory";
pub const SPARSE_VECTOR_NAME_UINT8: &str = "sparse-vector-uint8";
pub const SPARSE_VECTOR_NAME_FLOAT16: &str = "sparse-vector-float16";

/// Multi vectors base names
pub const MULTI_VECTOR_NAME_ON_DISK: &str = "multi-dense-vector-on-disk";
pub const MULTI_VECTOR_NAME_ROCKSDB: &str = "multi-dense-vector-rocksdb";
pub const MULTI_VECTOR_NAME_UINT8: &str = "multi-dense-vector-uint8";
pub const MULTI_VECTOR_NAME_FLOAT16: &str = "multi-dense-vector-float16";
pub const MULTI_VECTOR_NAME_SQ: &str = "multi-dense-vector-sq";
pub const MULTI_VECTOR_NAME_PQ: &str = "multi-dense-vector-pq";
pub const MULTI_VECTOR_NAME_BQ: &str = "multi-dense-vector-bq";

/// Payload keys
pub const MISSING_PAYLOAD_TIMESTAMP_KEY: &str = "missing-payload-timestamp";
pub const KEYWORD_PAYLOAD_KEY: &str = "crasher-payload-keyword";
pub const INTEGER_PAYLOAD_KEY: &str = "crasher-payload-integer";
pub const FLOAT_PAYLOAD_KEY: &str = "crasher-payload-float";
pub const GEO_PAYLOAD_KEY: &str = "crasher-payload-geo";
pub const TEXT_PAYLOAD_KEY: &str = "crasher-payload-text";
pub const BOOL_PAYLOAD_KEY: &str = "crasher-payload-bool";
pub const DATETIME_PAYLOAD_KEY: &str = "crasher-payload-datetime";
pub const UUID_PAYLOAD_KEY: &str = "crasher-payload-uuid";

#[derive(Debug)]
pub struct TestNamedVectors {
    dense_vectors: HashMap<String, VectorParams>,
    sparse_vectors: HashMap<String, SparseVectorParams>,
    multi_vectors: HashMap<String, VectorParams>,
}

// TODO unit test names
impl TestNamedVectors {
    pub fn new(duplication_factor: usize, vec_dim: usize) -> Self {
        let mut sparse_vectors = HashMap::new();
        let mut dense_vectors = HashMap::new();
        let mut multi_vectors = HashMap::new();

        let hnsw_config = Some(HnswConfigDiff {
            m: Some(32),
            ef_construct: None,
            full_scan_threshold: None,
            max_indexing_threads: None,
            on_disk: Some(false),
            payload_m: None,
        });

        // dense vectors on disk
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", DENSE_VECTOR_NAME_ON_DISK, i);
            dense_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // dense vector rocksdb
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", DENSE_VECTOR_NAME_ROCKSDB, i);
            dense_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(false), // rocksdb
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // dense vectors uint8
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", DENSE_VECTOR_NAME_UINT8, i);
            dense_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: Some(2),   // UInt8
                    multivector_config: None,
                },
            );
        }

        // dense vectors float16
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", DENSE_VECTOR_NAME_FLOAT16, i);
            dense_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: Some(3),   // Float16
                    multivector_config: None,
                },
            );
        }

        // dense vectors SQ
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", DENSE_VECTOR_NAME_SQ, i);
            dense_vectors.insert(
                name,
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
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // dense vectors PQ
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", DENSE_VECTOR_NAME_PQ, i);
            dense_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: 1, // x8
                            always_ram: Some(false),
                        })),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // dense vectors BQ
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", DENSE_VECTOR_NAME_BQ, i);
            dense_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(BinaryQuantization {
                            always_ram: Some(false),
                        })),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // sparse vector index on disk
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", SPARSE_VECTOR_NAME_INDEX_DISK, i);
            sparse_vectors.insert(
                name.clone(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(true), // on disk
                        datatype: None,
                    }),
                    modifier: None,
                },
            );
        }

        // sparse vector index in memory
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", SPARSE_VECTOR_NAME_INDEX_MEMORY, i);
            sparse_vectors.insert(
                name.clone(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(false), // in memory
                        datatype: None,
                    }),
                    modifier: None,
                },
            );
        }

        // sparse vector uint8
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", SPARSE_VECTOR_NAME_UINT8, i);
            sparse_vectors.insert(
                name.clone(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(true), // on disk
                        datatype: Some(2),   // UInt8
                    }),
                    modifier: None,
                },
            );
        }

        // sparse vector float16
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", SPARSE_VECTOR_NAME_FLOAT16, i);
            sparse_vectors.insert(
                name.clone(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(true), // on disk
                        datatype: Some(3),   // Float16
                    }),
                    modifier: None,
                },
            );
        }

        let multivector_config = Some(MultiVectorConfig { comparator: 0 });
        // multi vector on disk
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", MULTI_VECTOR_NAME_ON_DISK, i);
            multi_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config,
                },
            );
        }

        // multi vector rocksdb
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", MULTI_VECTOR_NAME_ROCKSDB, i);
            multi_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(false), // rocksdb
                    datatype: None,
                    multivector_config,
                },
            );
        }

        // multi vector uint8
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", MULTI_VECTOR_NAME_UINT8, i);
            multi_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: Some(2),   // UInt8
                    multivector_config,
                },
            );
        }

        // multi vector float16
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", MULTI_VECTOR_NAME_FLOAT16, i);
            multi_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: None,
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: Some(3),   // Float16
                    multivector_config,
                },
            );
        }

        // multi vectors SQ
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", MULTI_VECTOR_NAME_SQ, i);
            multi_vectors.insert(
                name,
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
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config,
                },
            );
        }

        // multi vectors PQ
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", MULTI_VECTOR_NAME_PQ, i);
            multi_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Product(ProductQuantization {
                            compression: 1, // x8
                            always_ram: Some(false),
                        })),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config,
                },
            );
        }

        // multi vectors BQ
        for i in 1..=duplication_factor {
            let name = format!("{}-{}", MULTI_VECTOR_NAME_BQ, i);
            multi_vectors.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(BinaryQuantization {
                            always_ram: Some(false),
                        })),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config,
                },
            );
        }

        Self {
            dense_vectors,
            sparse_vectors,
            multi_vectors,
        }
    }

    pub fn sparse_vectors(&self) -> HashMap<String, SparseVectorParams> {
        self.sparse_vectors.clone()
    }

    pub fn dense_vectors(&self) -> HashMap<String, VectorParams> {
        self.dense_vectors.clone()
    }

    pub fn multi_dense_vectors(&self) -> HashMap<String, VectorParams> {
        self.multi_vectors.clone()
    }

    pub fn sparse_vector_names(&self) -> Vec<String> {
        self.sparse_vectors.keys().cloned().collect()
    }

    pub fn dense_vector_names(&self) -> Vec<String> {
        self.dense_vectors.keys().cloned().collect()
    }

    pub fn multi_vector_names(&self) -> Vec<String> {
        self.multi_vectors.keys().cloned().collect()
    }
}

pub fn random_keyword(rng: &mut impl Rng, num_variants: usize) -> String {
    let variant = rng.random_range(0..num_variants);
    format!("keyword_{}", variant)
}

pub fn random_payload(keywords: Option<usize>) -> Payload {
    let mut payload = Payload::new();
    let mut rng = rand::rng();
    if let Some(keyword_variants) = keywords {
        if keyword_variants > 0 {
            payload.insert(
                KEYWORD_PAYLOAD_KEY,
                random_keyword(&mut rng, keyword_variants),
            );
            payload.insert(INTEGER_PAYLOAD_KEY, rng.random_range(0..100));
            payload.insert(FLOAT_PAYLOAD_KEY, rng.random_range(0.0..100.0));
            // create geo payload with random coordinates
            let geo_value = json!({
                "lat": rng.random_range(-90.0..90.0),
                "lon": rng.random_range(-180.0..180.0),
            });
            payload.insert(GEO_PAYLOAD_KEY, geo_value);
            payload.insert(TEXT_PAYLOAD_KEY, random_keyword(&mut rng, keyword_variants));
            payload.insert(BOOL_PAYLOAD_KEY, rng.random_bool(0.5));
            payload.insert(DATETIME_PAYLOAD_KEY, chrono::Utc::now().to_rfc3339());
            payload.insert(UUID_PAYLOAD_KEY, uuid::Uuid::new_v4().to_string());
        }
    }
    payload
}

pub fn random_filter(keywords: Option<usize>) -> Option<Filter> {
    let mut rng = rand::rng();
    let mut filter = Filter {
        should: vec![],
        must: vec![],
        must_not: vec![],
        min_should: None,
    };
    let mut have_any = false;
    if let Some(keyword_variants) = keywords {
        have_any = true;
        filter.must.push(
            FieldCondition {
                key: KEYWORD_PAYLOAD_KEY.to_string(),
                r#match: Some(Match {
                    match_value: Some(MatchValue::Keyword(random_keyword(
                        &mut rng,
                        keyword_variants,
                    ))),
                }),
                range: None,
                geo_bounding_box: None,
                geo_radius: None,
                values_count: None,
                geo_polygon: None,
                datetime_range: None,
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
    let mut rng = rand::rng();
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

pub fn random_sparse_vector(max_size: usize, sparsity: f64) -> Vec<(u32, f32)> {
    let mut rng = rand::rng();
    let size = rng.random_range(1..=max_size);
    // (index, value)
    let mut pairs = Vec::with_capacity(size);
    for i in 1..=size {
        // probability of skipping a dimension to make the vectors sparse
        let skip = !rng.random_bool(sparsity);
        if skip {
            continue;
        }
        // Only positive values are generated to make sure to hit the pruning path.
        pairs.push((i as u32, rng.random_range(0.0..10.0) as f32));
    }
    if pairs.is_empty() {
        // make sure at least one dimension is present
        pairs.push((1, rng.random_range(0.0..10.0) as f32));
    }
    pairs
}
