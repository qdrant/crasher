#!/bin/bash

set -e

CRASHER_DEFAULT_CMD="cargo run -r -- --working-dir ./qdrant/ --exec-path target/debug/qdrant --crash-probability 0.2"
CRASHER_CMD=${1:-"$CRASHER_DEFAULT_CMD"}
RUN_TIME=${2:-300}
LOG_FILE="crasher_output.log"

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
