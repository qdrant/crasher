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
      --exec-path <EXEC_PATH>
          Path to executable binary relative to `working_dir`
      --crash-probability <CRASH_PROBABILITY>
          Probability to kill running instance [default: 0.1]
      --uris <URIS>
          Qdrant gRPC service URIs (can be used several times to specify several URIs) [default: http://localhost:6334]
      --replication-factor <REPLICATION_FACTOR>
          Replication factor for collections [default: 1]
      --write-consistency-factor <WRITE_CONSISTENCY_FACTOR>
          Writing consistency factor for collections [default: 1]
      --indexing-threshold <INDEXING_THRESHOLD>
          Optimizer indexing threshold
      --memmap-threshold <MEMMAP_THRESHOLD>
          Maximum size (in KiloBytes) of vectors to store in-memory per segment
      --use-scalar-quantization
          Whether to use scalar quantization for vectors
      --grpc-timeout-ms <GRPC_TIMEOUT_MS>
          Timeout of gRPC client [default: 2000]
  -h, --help
          Print help
  -V, --version
          Print version
```

Examples:

```bash
cargo run -- --working-dir "/home/agourlay/Workspace/qdrant/" --exec-path "target/debug/qdrant"
```

```bash
cargo run -r -- --working-dir "/home/agourlay/Workspace/qdrant/" --exec-path "target/release/qdrant" --indexing-threshold 2000 --memmap-threshold 2000
```
