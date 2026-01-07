#!/usr/bin/env bash

set -e

QDRANT_DIR=${1:-./qdrant/}
QDRANT_EXEC=${2:-target/debug/qdrant}
CRASH_PROBABILITY=${3:-0.3}
RUN_TIME=${4:-300}
QDRANT_BACKUP_DIR=${5:?backup directory is required}

CRASHER_LOG=crasher.log
QDRANT_LOG=../qdrant.log

CRASHER_CMD=(
    cargo run --release
    --
    --working-dir "$QDRANT_DIR"
    --storage-backup $QDRANT_BACKUP_DIR
    --exec-path "$QDRANT_EXEC"
    --crash-probability "$CRASH_PROBABILITY"
)

echo "${CRASHER_CMD[*]}"

QDRANT__LOGGER__ON_DISK__ENABLED=true \
QDRANT__LOGGER__ON_DISK__LOG_FILE=$QDRANT_LOG \
QDRANT__SERVICE__HARDWARE_REPORTING=true \
QDRANT__STORAGE__COLLECTION_STRICT_MODE=true \
QDRANT__FEATURE_FLAGS__ALL=true \
QDRANT__LOG_LEVEL="TRACE,raft::raft=info,actix_http=info,tonic=info,want=info,mio=info" \
"${CRASHER_CMD[@]}" &>"$CRASHER_LOG" &

pid=$!

echo "The PID is $pid"

function cleanup() {
    if ps -p $pid >/dev/null
    then
        kill -KILL $pid
    fi
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
    kill $pid
    echo "OK"
else
    echo "The process has unexpectedly terminated on its own. Check the logs."
    exit 1
fi
