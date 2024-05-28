use clap::Parser;

/// Tool for crashing Qdrant instances
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
pub struct Args {
    /// Working directory for Qdrant data
    #[arg(long)]
    pub working_dir: String,
    /// Path to executable binary relative to `working_dir`
    #[arg(long)]
    pub exec_path: String,
    /// Probability to kill running instance
    #[arg(long, default_value_t = 0.1)]
    pub crash_probability: f64,
    /// The time in second to sleep between crashes attempt
    #[arg(long, default_value_t = 5)]
    pub sleep_duration_between_crash_sec: usize,
    /// Qdrant gRPC service URIs (can be used several times to specify several URIs)
    #[arg(long, default_value = "http://localhost:6334")]
    pub uris: Vec<String>,
    /// Replication factor for collections
    #[arg(long, default_value_t = 1)]
    pub replication_factor: usize,
    /// Writing consistency factor for collections
    #[arg(long, default_value_t = 1)]
    pub write_consistency_factor: usize,
    /// Optimizer indexing threshold
    #[arg(long)]
    pub indexing_threshold: Option<usize>,
    /// Maximum size (in KiloBytes) of vectors to store in-memory per segment.
    #[arg(long)]
    pub memmap_threshold: Option<usize>,
    /// Whether to use scalar quantization for vectors
    #[arg(long, default_value_t = true)]
    pub use_scalar_quantization: bool,
    /// If true - serve vectors from disk. If set to false, the vectors will be loaded in RAM.
    #[arg(long, default_value_t = true)]
    pub vectors_on_disk: bool,
    /// Timeout of gRPC client
    #[arg(long, default_value_t = 2000)]
    pub grpc_timeout_ms: usize,
    /// Whether to only upsert sparse vectors
    #[arg(long, default_value_t = false)]
    pub only_sparse: bool,
}
