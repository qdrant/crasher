#!/bin/bash

set -e

QDRANT_DIR=${1:-"./qdrant/"}
QDRANT_EXEC=${2:-"target/debug/qdrant"}
CRASH_PROBABILITY=${3:-"0.2"}
LOG_FILE="crasher_output.log"
RUN_TIME=${4:-300}

CRASHER_CMD="cargo run -r -- --working-dir $QDRANT_DIR --exec-path $QDRANT_EXEC --crash-probability $CRASH_PROBABILITY"
echo "$CRASHER_CMD"
$CRASHER_CMD > $LOG_FILE 2>&1 &
pid=$!

echo "The PID is $pid"

function cleanup() {
    kill -9 $pid 2>/dev/null
}

trap cleanup EXIT

if ps -p $pid > /dev/null; then
    echo "The process is running. Wait for $RUN_TIME seconds."
    sleep $RUN_TIME

    if ps -p $pid > /dev/null; then
        echo "The process is still running. Stopping the process..."

        kill $pid

        echo "OK"
    else
        echo "The process has unexpectedly terminated on its own. Check the logs."
        exit 1
    fi
else
    echo "The process did not start correctly or has already terminated. Check the logs."
    exit 2
fi
