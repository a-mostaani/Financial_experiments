#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/rshamostaani/miniconda3/envs/vllmFinance/bin/python"
STREAMLIT="/home/rshamostaani/miniconda3/envs/vllmFinance/bin/streamlit"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- Pre-flight checks ---

echo "[launcher] Checking Redis..."
if ! redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo "[launcher] ERROR: Redis is not running on localhost:6379. Please start it first." >&2
    exit 1
fi
echo "[launcher] Redis OK."

echo "[launcher] Checking PostgreSQL..."
if ! pg_isready -h localhost -p 5432 -U newsuser -d newsdb > /dev/null 2>&1; then
    echo "[launcher] ERROR: PostgreSQL is not reachable at localhost:5432 (db=newsdb, user=newsuser). Please start it first." >&2
    exit 1
fi
echo "[launcher] PostgreSQL OK."

# --- Graceful shutdown ---

STREAMER_PID=""
WORKER_PID=""
DASHBOARD_PID=""

cleanup() {
    echo ""
    echo "[launcher] Shutting down all services..."
    [ -n "$STREAMER_PID" ]  && kill "$STREAMER_PID"  2>/dev/null || true
    [ -n "$WORKER_PID" ]    && kill "$WORKER_PID"    2>/dev/null || true
    [ -n "$DASHBOARD_PID" ] && kill "$DASHBOARD_PID" 2>/dev/null || true
    wait "$STREAMER_PID" "$WORKER_PID" "$DASHBOARD_PID" 2>/dev/null || true
    echo "[launcher] All services stopped."
}
trap cleanup SIGINT SIGTERM EXIT

# --- Start services ---

echo "[launcher] Starting news_streamer.py → logs/streamer.log"
$PYTHON "$SCRIPT_DIR/news_streamer.py" >> "$LOG_DIR/streamer.log" 2>&1 &
STREAMER_PID=$!

sleep 2

echo "[launcher] Starting sentiment_worker.py → logs/worker.log"
$PYTHON "$SCRIPT_DIR/sentiment_worker.py" >> "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!

sleep 1

echo "[launcher] Starting dashboard.py → logs/dashboard.log"
$STREAMLIT run "$SCRIPT_DIR/dashboard.py" >> "$LOG_DIR/dashboard.log" 2>&1 &
DASHBOARD_PID=$!

echo "[launcher] All services started. Logs in $LOG_DIR/"
echo "[launcher] Press Ctrl+C to stop all."

wait $DASHBOARD_PID
