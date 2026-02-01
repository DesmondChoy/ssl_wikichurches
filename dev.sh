#!/bin/bash
# Development script to run both backend and frontend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Store PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

# Kill any existing processes on our ports
kill_existing() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo -e "${RED}[Cleanup]${NC} Killing existing processes on port $port (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

cleanup() {
    echo -e "\n${RED}Shutting down...${NC}"

    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "Stopping frontend (PID: $FRONTEND_PID)"
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi

    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Stopping backend (PID: $BACKEND_PID)"
        kill "$BACKEND_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null
    echo -e "${GREEN}Shutdown complete${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo -e "${BLUE}Starting SSL Attention development servers...${NC}\n"

# Kill any existing processes on our ports
kill_existing 8000
kill_existing 5173

# Start backend
echo -e "${GREEN}[Backend]${NC} Starting FastAPI on http://localhost:8000"
uvicorn app.backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start frontend
echo -e "${GREEN}[Frontend]${NC} Starting Vite on http://localhost:5173"
cd app/frontend
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}[Frontend]${NC} Installing npm dependencies..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!

echo -e "\n${BLUE}Both servers running. Press Ctrl+C to stop.${NC}\n"

# Wait for both processes
wait
