#!/usr/bin/env bash

set -e

QDRANT_DIR=${1:-./qdrant/}
QDRANT_EXEC=${2:-target/debug/qdrant}
CRASH_PROBABILITY=${3:-0.3}
RUN_TIME=${4:-300}
QDRANT_BACKUP_DIRS=( "${@:5}" )

CRASHER_LOG=crasher.log
QDRANT_LOG=../qdrant.log

CRASHER_CMD=(
    cargo run --release
    --
    --working-dir "$QDRANT_DIR"
    ${QDRANT_BACKUP_DIRS[@]/#/--backup-working-dir } # this does not handle spaces 😬
    --exec-path "$QDRANT_EXEC"
    --crash-probability "$CRASH_PROBABILITY"
    --missing-payload-check
)

echo "${CRASHER_CMD[*]}"

# TODO remove `*_USES_MMAP` once it can be done through the API
QDRANT__STORAGE__ON_DISK_PAYLOAD_USES_MMAP=true \
QDRANT__STORAGE__ON_DISK_SPARSE_VECTORS_USES_MMAP=true \
QDRANT__LOGGER__ON_DISK__ENABLED=true \
QDRANT__LOGGER__ON_DISK__LOG_FILE=$QDRANT_LOG \
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
