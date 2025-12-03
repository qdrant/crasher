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
      --backup-working-dir <BACKUP_WORKING_DIR>
          Backup working directory between Qdrant restarts (useful to debug storage recovery issues)
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
          Maximum size (in KiloBytes) of vectors to store in-memory per segment
      --grpc-timeout-ms <GRPC_TIMEOUT_MS>
          Timeout of gRPC client [default: 10000]
      --on-disk-payload
          Whether to use on-disk payload storage
      --only-sparse
          Whether to only upsert sparse vectors
      --duplication-factor <DUPLICATION_FACTOR>
          Duplication factor for generating additional named vectors [default: 2]
      --shutdown-on-error
          Whether to shutdown qdrant on error (use false to investigate deadlocks)
      --cpu-quota <CPU_QUOTA>
          CPU quotas in percent for the Qdrant binary
  -h, --help
          Print help
  -V, --version
          Print version
```

Examples:

- using debug binary:

```bash
cargo run -r -- --working-dir "/home/agourlay/Workspace/qdrant/" --exec-path "target/debug/qdrant"
```

- using release binary:

```bash
cargo run -r -- --working-dir "/home/agourlay/Workspace/qdrant/" --exec-path "target/release/qdrant" --indexing-threshold 2000 --memmap-threshold 2000
```
