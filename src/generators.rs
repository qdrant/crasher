use core::option::Option;
use core::option::Option::{None, Some};
use qdrant_client::Payload;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::{
    BinaryQuantizationBuilder, BinaryQuantizationEncoding, Condition, Distance, Filter,
    HnswConfigDiff, MultiVectorConfig, ProductQuantization, QuantizationConfig, ScalarQuantization,
    SparseIndexConfig, SparseVectorParams, VectorParams,
};
use rand::Rng;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
// TODO do not generate vector configuration manually but create all possibility exhaustively

/// Dense vectors base names
pub const DENSE_VECTOR_NAME_ON_DISK: &str = "dense-vector-on-disk";
pub const DENSE_VECTOR_NAME_ROCKSDB: &str = "dense-vector-rocksdb";
pub const DENSE_VECTOR_NAME_UINT8: &str = "dense-vector-uint8";
pub const DENSE_VECTOR_NAME_FLOAT16: &str = "dense-vector-float16";
pub const DENSE_VECTOR_NAME_SQ: &str = "dense-vector-sq";
pub const DENSE_VECTOR_NAME_PQ: &str = "dense-vector-pq";
pub const DENSE_VECTOR_NAME_BQ_1B: &str = "dense-vector-bq-1b";
pub const DENSE_VECTOR_NAME_BQ_1HB: &str = "dense-vector-bq-1Hb";
pub const DENSE_VECTOR_NAME_BQ_2B: &str = "dense-vector-bq-2b";
pub const DENSE_VECTOR_NAME_HNSW_INLINE: &str = "dense-vector-hnsw-inline";

/// Sparse vectors base names
pub const SPARSE_VECTOR_NAME_INDEX_DISK: &str = "sparse-vector-index-disk";
pub const SPARSE_VECTOR_NAME_INDEX_MEMORY: &str = "sparse-vector-index-memory";
pub const SPARSE_VECTOR_NAME_UINT8: &str = "sparse-vector-uint8";
pub const SPARSE_VECTOR_NAME_FLOAT16: &str = "sparse-vector-float16";
pub const SPARSE_VECTOR_NAME_IDF: &str = "sparse-vector-IDF";

/// Multi vectors base names
pub const MULTI_VECTOR_NAME_ON_DISK: &str = "multi-dense-vector-on-disk";
pub const MULTI_VECTOR_NAME_ROCKSDB: &str = "multi-dense-vector-rocksdb";
pub const MULTI_VECTOR_NAME_UINT8: &str = "multi-dense-vector-uint8";
pub const MULTI_VECTOR_NAME_FLOAT16: &str = "multi-dense-vector-float16";
pub const MULTI_VECTOR_NAME_SQ: &str = "multi-dense-vector-sq";
pub const MULTI_VECTOR_NAME_PQ: &str = "multi-dense-vector-pq";
pub const MULTI_VECTOR_NAME_BQ: &str = "multi-dense-vector-bq";

/// Mandatory payload keys (present on all points)
pub const MANDATORY_PAYLOAD_TIMESTAMP_KEY: &str = "mandatory-payload-timestamp";
pub const MANDATORY_PAYLOAD_BOOL_KEY: &str = "mandatory-payload-bool";
// TODO add one mandatory key per index type

// Indexed payload keys
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
    dense: BTreeMap<String, VectorParams>,
    sparse: BTreeMap<String, SparseVectorParams>,
    multi: BTreeMap<String, VectorParams>,
}

// TODO unit test names
impl TestNamedVectors {
    pub fn new(duplication_factor: u32, vec_dim: u32) -> Self {
        let mut sparse = BTreeMap::new();
        let mut dense = BTreeMap::new();
        let mut multi = BTreeMap::new();

        let hnsw_config = Some(HnswConfigDiff {
            m: Some(32),
            ef_construct: None,
            full_scan_threshold: None,
            max_indexing_threads: None,
            on_disk: Some(false),
            payload_m: None,
            inline_storage: None,
        });

        // warning: requires Quantization
        let hnsw_config_inline_storage = Some(HnswConfigDiff {
            m: Some(32),
            ef_construct: None,
            full_scan_threshold: None,
            max_indexing_threads: None,
            on_disk: Some(false),
            payload_m: None,
            inline_storage: Some(true),
        });

        // dense vectors on disk
        for i in 1..=duplication_factor {
            let name = format!("{DENSE_VECTOR_NAME_ON_DISK}-{i}");
            dense.insert(
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
            let name = format!("{DENSE_VECTOR_NAME_ROCKSDB}-{i}");
            dense.insert(
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
            let name = format!("{DENSE_VECTOR_NAME_UINT8}-{i}");
            dense.insert(
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
            let name = format!("{DENSE_VECTOR_NAME_FLOAT16}-{i}");
            dense.insert(
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
            let name = format!("{DENSE_VECTOR_NAME_SQ}-{i}");
            dense.insert(
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
            let name = format!("{DENSE_VECTOR_NAME_PQ}-{i}");
            dense.insert(
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

        // dense vectors BQ 1bit
        for i in 1..=duplication_factor {
            let name = format!("{DENSE_VECTOR_NAME_BQ_1B}-{i}");

            let bq_builder =
                BinaryQuantizationBuilder::new(false).encoding(BinaryQuantizationEncoding::OneBit);

            dense.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(bq_builder.build())),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // dense vectors BQ 1 and half bit
        for i in 1..=duplication_factor {
            let name = format!("{DENSE_VECTOR_NAME_BQ_1HB}-{i}");

            let bq_builder = BinaryQuantizationBuilder::new(false)
                .encoding(BinaryQuantizationEncoding::OneAndHalfBits);

            dense.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(bq_builder.build())),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // dense vectors BQ 2bits
        for i in 1..=duplication_factor {
            let name = format!("{DENSE_VECTOR_NAME_BQ_2B}-{i}");

            let bq_builder =
                BinaryQuantizationBuilder::new(false).encoding(BinaryQuantizationEncoding::TwoBits);

            dense.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(bq_builder.build())),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // dense vectors with HNSW inlined storage
        for i in 1..=duplication_factor {
            let name = format!("{DENSE_VECTOR_NAME_HNSW_INLINE}-{i}");

            let bq_builder =
                BinaryQuantizationBuilder::new(false).encoding(BinaryQuantizationEncoding::TwoBits);

            dense.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(bq_builder.build())),
                    }),
                    hnsw_config: hnsw_config_inline_storage,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config: None,
                },
            );
        }

        // sparse vector index on disk
        for i in 1..=duplication_factor {
            let name = format!("{SPARSE_VECTOR_NAME_INDEX_DISK}-{i}");
            sparse.insert(
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
            let name = format!("{SPARSE_VECTOR_NAME_INDEX_MEMORY}-{i}");
            sparse.insert(
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
            let name = format!("{SPARSE_VECTOR_NAME_UINT8}-{i}");
            sparse.insert(
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
            let name = format!("{SPARSE_VECTOR_NAME_FLOAT16}-{i}");
            sparse.insert(
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

        // sparse vector IDF
        for i in 1..=duplication_factor {
            let name = format!("{SPARSE_VECTOR_NAME_IDF}-{i}");
            sparse.insert(
                name.clone(),
                SparseVectorParams {
                    index: Some(SparseIndexConfig {
                        full_scan_threshold: None,
                        on_disk: Some(true), // on disk
                        datatype: Some(1),   // Float32
                    }),
                    modifier: Some(2), // IDF
                },
            );
        }

        let multivector_config = Some(MultiVectorConfig { comparator: 0 });
        // multi vector on disk
        for i in 1..=duplication_factor {
            let name = format!("{MULTI_VECTOR_NAME_ON_DISK}-{i}");
            multi.insert(
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
            let name = format!("{MULTI_VECTOR_NAME_ROCKSDB}-{i}");
            multi.insert(
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
            let name = format!("{MULTI_VECTOR_NAME_UINT8}-{i}");
            multi.insert(
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
            let name = format!("{MULTI_VECTOR_NAME_FLOAT16}-{i}");
            multi.insert(
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
            let name = format!("{MULTI_VECTOR_NAME_SQ}-{i}");
            multi.insert(
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
            let name = format!("{MULTI_VECTOR_NAME_PQ}-{i}");
            multi.insert(
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
            let name = format!("{MULTI_VECTOR_NAME_BQ}-{i}");
            let bq_builder =
                BinaryQuantizationBuilder::new(false).encoding(BinaryQuantizationEncoding::TwoBits);
            multi.insert(
                name,
                VectorParams {
                    size: vec_dim as u64,
                    distance: Distance::Dot.into(),
                    quantization_config: Some(QuantizationConfig {
                        quantization: Some(Quantization::Binary(bq_builder.build())),
                    }),
                    hnsw_config,
                    on_disk: Some(true), // on disk
                    datatype: None,
                    multivector_config,
                },
            );
        }

        Self {
            dense,
            sparse,
            multi,
        }
    }

    pub fn sparse_params(&self) -> HashMap<String, SparseVectorParams> {
        self.sparse.clone().into_iter().collect()
    }

    pub fn dense_params(&self) -> HashMap<String, VectorParams> {
        self.dense.clone().into_iter().collect()
    }

    pub fn multi_dense_params(&self) -> HashMap<String, VectorParams> {
        self.multi.clone().into_iter().collect()
    }

    pub fn sparse_names(&self) -> Vec<String> {
        self.sparse.keys().cloned().collect()
    }

    pub fn dense_names(&self) -> Vec<String> {
        self.dense.keys().cloned().collect()
    }

    pub fn multi_names(&self) -> Vec<String> {
        self.multi.keys().cloned().collect()
    }
}

pub fn random_keyword(rng: &mut impl Rng, num_variants: u32) -> String {
    let variant = rng.random_range(0..num_variants);
    format!("keyword_{variant}")
}

const WORDS: [&str; 8] = ["the", "quick", "fox", "jumps", "over", "the", "lazy", "dog"];

pub fn random_sentence(rng: &mut impl Rng, num_variants: u32) -> String {
    let variant = rng.random_range(0..num_variants);
    let words: Vec<_> = WORDS.iter().take(variant as usize).copied().collect();
    words.join(" ")
}

pub fn random_payload(rng: &mut impl Rng, keywords: Option<u32>) -> Payload {
    let mut payload = Payload::new();
    if let Some(keyword_variants) = keywords
        && keyword_variants > 0
    {
        payload.insert(KEYWORD_PAYLOAD_KEY, random_keyword(rng, keyword_variants));
        payload.insert(INTEGER_PAYLOAD_KEY, rng.random_range(0..100));
        payload.insert(FLOAT_PAYLOAD_KEY, rng.random_range(0.0..100.0));
        // create geo payload with random coordinates
        let geo_value = json!({
            "lat": rng.random_range(-90.0..90.0),
            "lon": rng.random_range(-180.0..180.0),
        });
        payload.insert(GEO_PAYLOAD_KEY, geo_value);
        payload.insert(TEXT_PAYLOAD_KEY, random_sentence(rng, keyword_variants));
        payload.insert(BOOL_PAYLOAD_KEY, rng.random_bool(0.5));
        payload.insert(DATETIME_PAYLOAD_KEY, chrono::Utc::now().to_rfc3339());
        payload.insert(UUID_PAYLOAD_KEY, uuid::Uuid::new_v4().to_string());
    }
    payload
}

pub fn random_filter(rng: &mut impl Rng, keywords: Option<u32>) -> Option<Filter> {
    if let Some(keyword_variants) = keywords {
        let mut filter = Filter {
            should: vec![],
            must: vec![],
            must_not: vec![],
            min_should: None,
        };

        filter.must.push(Condition::matches(
            KEYWORD_PAYLOAD_KEY,
            MatchValue::Keyword(random_keyword(rng, keyword_variants)),
        ));
        Some(filter)
    } else {
        None
    }
}

pub fn random_dense_vector(rng: &mut impl Rng, name: &str, dim: u32) -> Vec<f32> {
    let mut res = Vec::with_capacity(dim as usize);
    // uint8 vectors get mapped to integers. If we use regular range it is very possible we get a
    // zeroed vector, which we are asserting should not happen
    if name.contains("uint8") {
        for _ in 0..dim {
            res.push(rng.random_range(1.0..255.0));
        }
    } else {
        for _ in 0..dim {
            res.push(rng.random_range(-10.0..10.0));
        }
    }
    assert!(
        !res.iter().all(|&x| x == 0.0),
        "Zero vector generated {res:?}"
    );
    res
}

pub fn random_sparse_vector(rng: &mut impl Rng, max_size: u32, sparsity: f32) -> Vec<(u32, f32)> {
    let size = rng.random_range(1..=max_size);
    // (index, value)
    let mut pairs = Vec::with_capacity(size as usize);
    for i in 1..=size {
        // probability of skipping a dimension to make the vectors sparse
        let skip = !rng.random_bool(sparsity as f64);
        if skip {
            continue;
        }
        // Only positive values are generated to make sure to hit the pruning path.
        pairs.push((i, rng.random_range(0.0..10.0) as f32));
    }
    if pairs.is_empty() {
        // make sure at least one dimension is present
        pairs.push((1, rng.random_range(0.0..10.0) as f32));
    }
    pairs
}
