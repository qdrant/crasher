#!/usr/bin/env bash

set -e

QDRANT_DIR=${1:-./qdrant/}
QDRANT_EXEC=${2:-target/debug/qdrant}
CRASH_PROBABILITY=${3:-0.3}
RUN_TIME=${4:-300}
POINTS_COUNT=${5:-5000}
QDRANT_BACKUP_DIR=${6:?backup directory is required}

CRASHER_LOG=crasher.log
QDRANT_LOG=../qdrant.log

CRASHER_CMD=(
    cargo run --release
    --
    --working-dir "$QDRANT_DIR"
    --storage-backup $QDRANT_BACKUP_DIR
    --exec-path "$QDRANT_EXEC"
    --crash-probability "$CRASH_PROBABILITY"
    --points-count "$POINTS_COUNT"
    --async-io
)

echo "${CRASHER_CMD[*]}"

# Run the crasher in its own process group (job control monitor mode) so we can
# reap the WHOLE tree (cargo -> crasher -> qdrant) on exit. Otherwise an orphaned
# qdrant survives, keeps the CI runner's stdout/stderr pipe open and hangs the job
# until GitHub's hard timeout (this is what caused a 6h run).
set -m

QDRANT__LOGGER__ON_DISK__ENABLED=true \
QDRANT__LOGGER__ON_DISK__LOG_FILE=$QDRANT_LOG \
QDRANT__SERVICE__HARDWARE_REPORTING=true \
QDRANT__STORAGE__COLLECTION_STRICT_MODE=true \
QDRANT__FEATURE_FLAGS__ALL=true \
QDRANT__LOG_LEVEL="TRACE,raft::raft=info,actix_http=info,tonic=info,want=info,mio=info" \
"${CRASHER_CMD[@]}" &>"$CRASHER_LOG" &

pid=$!

set +m

echo "The PID is $pid"

function cleanup() {
    # SIGKILL the whole process group (-$pid). Stays valid as long as any member
    # is alive, so this reaps a surviving qdrant even if cargo already exited.
    kill -KILL -- -"$pid" 2>/dev/null || true
}

trap cleanup EXIT

trap 'exit $?' ERR
trap exit INT

started=$(date +%s)

while ps -p $pid >/dev/null && (( $(date +%s) - started < RUN_TIME ))
do
    sleep 10
done

if ps -p $pid >/dev/null
then
    echo "The process is still running. Stopping the process..."
    kill -- -"$pid" 2>/dev/null || true
    echo "OK"
else
    echo "The process has unexpectedly terminated on its own. Check the logs."
    # Reap any orphaned children (e.g. qdrant) still holding the runner pipe,
    # otherwise the job hangs even though the crasher itself already exited.
    kill -KILL -- -"$pid" 2>/dev/null || true
    exit 1
fi
