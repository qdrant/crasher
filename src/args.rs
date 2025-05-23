use clap::Parser;

/// Tool for crashing Qdrant instances
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
pub struct Args {
    /// Working directory for Qdrant data
    #[arg(long)]
    pub working_dir: String,
    /// Backup working directory between Qdrant restarts (useful to debug storage recovery issues)
    #[arg(long)]
    pub backup_working_dir: Vec<String>,
    /// Path to executable binary relative to `working_dir`
    #[arg(long)]
    pub exec_path: String,
    /// Probability to kill running instance
    #[arg(long, default_value_t = 0.3)]
    pub crash_probability: f64,
    /// The time in second to sleep between crashes attempt
    #[arg(long, default_value_t = 5)]
    pub sleep_duration_between_crash_sec: usize,
    /// Qdrant gRPC service URIs (can be used several times to specify several URIs)
    #[arg(long, default_value = "http://localhost:6334")]
    pub uris: Vec<String>,
    /// Number of points to generate
    #[arg(long, default_value_t = 5_000)]
    pub points_count: usize,
    /// Dimension of generated vectors
    #[arg(long, default_value_t = 10)]
    pub vector_dimension: usize,
    /// Configure the flush interval for collections
    #[arg(long, default_value_t = 5)]
    pub flush_interval_sec: usize,
    /// Configure the number of segment
    #[arg(long, default_value_t = 2)]
    pub segment_count: usize,
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
    /// Timeout of gRPC client
    #[arg(long, default_value_t = 5_000)]
    pub grpc_timeout_ms: usize,
    /// Whether to use on-disk payload storage
    #[arg(long, default_value_t = true)]
    pub on_disk_payload: bool,
    /// Whether to only upsert sparse vectors
    #[arg(long, default_value_t = false)]
    pub only_sparse: bool,
    /// Duplication factor for generating additional named vectors
    #[arg(long, default_value_t = 2)]
    pub duplication_factor: usize,
    /// Whether to perform extra consistency check
    #[arg(long, default_value_t = false)]
    pub consistency_check: bool,
    /// Enable additional check that all points contain payload
    #[arg(long)]
    pub missing_payload_check: bool,
}
