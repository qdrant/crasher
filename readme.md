# Crasher

Tool for crashing local Qdrant instances for testing purposes.

## Status

Highly experimental, use at your own risk :fire:

## Usage

```commandline
./crasher -h
Tool for crashing Qdrant instances

Usage: crasher [OPTIONS] --working-dir <WORKING_DIR> --exec-path <EXEC_PATH>

Options:
      --working-dir <WORKING_DIR>
          Working directory for Qdrant data
      --storage-backup <STORAGE_BACKUP>
          Backup `storage` directory from `working_dir` between Qdrant restarts (useful to debug storage recovery issues)
      --exec-path <EXEC_PATH>
          Path to executable binary relative to `working_dir`
      --crash-probability <CRASH_PROBABILITY>
          Probability to kill running instance [default: 0.3]
      --sleep-duration-between-crash-sec <SLEEP_DURATION_BETWEEN_CRASH_SEC>
          The time in second to sleep between crashes attempt [default: 5]
      --uris <URIS>
          Qdrant gRPC service URIs (can be used several times to specify several URIs) [default: http://localhost:6334]
      --points-count <POINTS_COUNT>
          Number of points to generate [default: 5000]
      --vector-dimension <VECTOR_DIMENSION>
          Dimension of generated vectors [default: 10]
      --flush-interval-sec <FLUSH_INTERVAL_SEC>
          Configure the flush interval for collections [default: 5]
      --segment-count <SEGMENT_COUNT>
          Configure the number of segment [default: 2]
      --shard-count <SHARD_COUNT>
          Configure the number of shards [default: 3]
      --replication-factor <REPLICATION_FACTOR>
          Replication factor for collections [default: 1]
      --write-consistency-factor <WRITE_CONSISTENCY_FACTOR>
          Writing consistency factor for collections [default: 1]
      --indexing-threshold <INDEXING_THRESHOLD>
          Optimizer indexing threshold
      --memmap-threshold <MEMMAP_THRESHOLD>
          Maximum size (in `KiloBytes`) of vectors to store in-memory per segment
      --prevent-unindexed
          Whether to prevent the creation of large unindexed segments
      --grpc-timeout-ms <GRPC_TIMEOUT_MS>
          Timeout of gRPC client [default: 60000]
      --http-timeout-ms <HTTP_TIMEOUT_MS>
          Timeout of HTTP client [default: 60000]
      --on-disk-payload
          Whether to use on-disk payload storage
      --only-sparse
          Whether to only upsert sparse vectors
      --duplication-factor <DUPLICATION_FACTOR>
          Duplication factor for generating additional named vectors [default: 2]
      --skip-snapshot-restore
          Whether to perform snapshot restore
      --shutdown-on-error
          Whether to shut down qdrant on error (use false to investigate deadlocks)
      --cpu-quota <CPU_QUOTA>
          CPU quotas in percent for the Qdrant binary
      --rng-seed <RNG_SEED>
          Seed for internal RNG
  -h, --help
          Print help
  -V, --version
          Print version
```

## Vector configurations

Each base configuration is multiplied by the `--duplication-factor` (default: 2).

### Dense vectors

| Name | Distance | Storage | Datatype | Quantization | HNSW |
|------|----------|---------|----------|--------------|------|
| `dense-vector-mmap` | Dot | mmap | - | - | default |
| `dense-vector-memory` | Dot | memory | - | - | default |
| `dense-vector-uint8` | Dot | mmap | UInt8 | - | default |
| `dense-vector-float16` | Dot | mmap | Float16 | - | default |
| `dense-vector-sq` | Dot | mmap | - | SQ Int8 | default |
| `dense-vector-pq` | Dot | mmap | - | PQ x8 | default |
| `dense-vector-bq-1b` | Dot | mmap | - | BQ 1-bit | default |
| `dense-vector-bq-1Hb` | Dot | mmap | - | BQ 1.5-bit | default |
| `dense-vector-bq-2b` | Dot | mmap | - | BQ 2-bit | default |
| `dense-vector-hnsw-inline` | Dot | mmap | - | BQ 2-bit | inline storage |
| `dense-vector-cosine` | Cosine | mmap | - | - | default |
| `dense-vector-euclid` | Euclid | mmap | - | - | default |
| `dense-vector-manhattan` | Manhattan | mmap | - | - | default |
| `dense-vector-memory-sq` | Dot | memory | - | SQ Int8 | default |
| `dense-vector-memory-pq` | Dot | memory | - | PQ x8 | default |
| `dense-vector-memory-bq` | Dot | memory | - | BQ 1-bit | default |
| `dense-vector-memory-bq-1Hb` | Dot | memory | - | BQ 1.5-bit | default |
| `dense-vector-memory-bq-2b` | Dot | memory | - | BQ 2-bit | default |
| `dense-vector-memory-uint8` | Dot | memory | UInt8 | - | default |
| `dense-vector-memory-float16` | Dot | memory | Float16 | - | default |
| `dense-vector-memory-cosine` | Cosine | memory | - | - | default |
| `dense-vector-memory-euclid` | Euclid | memory | - | - | default |
| `dense-vector-memory-manhattan` | Manhattan | memory | - | - | default |
| `dense-vector-hnsw-on-disk` | Dot | mmap | - | - | on-disk graph |
| `dense-vector-pq-x16` | Dot | mmap | - | PQ x16 | default |
| `dense-vector-sq-ram` | Dot | mmap | - | SQ Int8 (quantile=0.95, always_ram) | default |

### Sparse vectors

| Name | Storage | Datatype | Modifier |
|------|---------|----------|----------|
| `sparse-vector-index-disk` | mmap | - | - |
| `sparse-vector-index-memory` | memory | - | - |
| `sparse-vector-uint8` | mmap | UInt8 | - |
| `sparse-vector-float16` | mmap | Float16 | - |
| `sparse-vector-IDF` | mmap | Float32 | IDF |
| `sparse-vector-memory-uint8` | memory | UInt8 | - |
| `sparse-vector-memory-float16` | memory | Float16 | - |
| `sparse-vector-memory-IDF` | memory | Float32 | IDF |

### Multi-dense vectors

| Name | Distance | Storage | Datatype | Quantization |
|------|----------|---------|----------|--------------|
| `multi-dense-vector-mmap` | Dot | mmap | - | - |
| `multi-dense-vector-memory` | Dot | memory | - | - |
| `multi-dense-vector-uint8` | Dot | mmap | UInt8 | - |
| `multi-dense-vector-float16` | Dot | mmap | Float16 | - |
| `multi-dense-vector-sq` | Dot | mmap | - | SQ Int8 |
| `multi-dense-vector-pq` | Dot | mmap | - | PQ x8 |
| `multi-dense-vector-bq` | Dot | mmap | - | BQ 2-bit |

Examples:

- using debug binary:

```bash
cargo run -r -- --working-dir "/home/agourlay/Workspace/qdrant/" --exec-path "target/debug/qdrant"
```

- using release binary:

```bash
cargo run -r -- --working-dir "/home/agourlay/Workspace/qdrant/" --exec-path "target/release/qdrant" --indexing-threshold 2000 --memmap-threshold 2000
```
