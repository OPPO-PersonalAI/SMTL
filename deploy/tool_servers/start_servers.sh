#!/usr/bin/env bash
# coding=utf-8
# Copyright 2026 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Unified server startup script
# Usage: ./start_all_servers.sh [start|stop|status|restart|test] [websearch_mode]
#       websearch_mode: dev (development mode), prod (production mode, 4 workers), perf (high performance mode, 8 workers)

# Load environment variables
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "✅ Loaded .env: $ENV_FILE"
else
    echo "⚠️  Warning: .env not found: $ENV_FILE"
fi

# Environment variable check
: "${CRAWL_PAGE_PORT:?Please set CRAWL_PAGE_PORT in .env}"
: "${WEBSEARCH_PORT:?Please set WEBSEARCH_PORT in .env}"

# Optional environment variables
if [[ -n "${NSJAILPATH:-}" ]]; then
    echo "ℹ️  NSJAILPATH: $NSJAILPATH"
fi

# Display configuration
echo "=========================================="
echo "Configuration:"
echo "  SERVER_HOST: $SERVER_HOST"
echo "  CRAWL_PAGE_PORT: $CRAWL_PAGE_PORT"
echo "  WEBSEARCH_PORT: $WEBSEARCH_PORT"
echo "  CODE_EXEC_PORT: $CODE_EXEC_PORT"
echo "=========================================="
echo ""

# Directory settings
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DIR/logs/$SERVER_HOST";   mkdir -p "$LOG_DIR"
PID_DIR="$DIR/pids/$SERVER_HOST";   mkdir -p "$PID_DIR"

# Service script paths
CRAWL_PAGE_SCRIPT="$DIR/scripts/crawl_page_server.py"
WEBSEARCH_SCRIPT="$DIR/scripts/cached_serper_server.py"

# Command and mode
cmd=${1:-}
websearch_mode=${2:-perf}

# ============================================
# Helper functions
# ============================================

# Check if process is running
is_running() {
    local pidf=$1
    if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if port is used
is_port_used() {
    local port=$1
    lsof -i:"$port" &>/dev/null
}

# Gracefully stop process
stop_process() {
    local service_name=$1
    local pidf=$2
    local port=$3
    
    local stopped=0
    
    echo "Stopping ${service_name}..."
    
    # Method 1: Stop via PID file
    if [[ -f "$pidf" ]]; then
        local pid=$(cat "$pidf" 2>/dev/null)
        if [[ -n "$pid" ]]; then
            if kill -0 "$pid" 2>/dev/null; then
                if kill "$pid" 2>/dev/null; then
                    # Wait for process to end
                    local count=0
                    while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                        sleep 1
                        count=$((count + 1))
                        echo -n "."
                    done
                    echo ""
                    
                    if kill -0 "$pid" 2>/dev/null; then
                        echo "⚠️  Process not responding, force stopping..."
                        kill -9 "$pid" 2>/dev/null
                    fi
                    
                    echo "✅ Stopped ${service_name} (PID $pid)"
                    stopped=1
                else
                    echo "⚠️  Cannot stop ${service_name} via PID $pid, trying other methods..."
                fi
            fi
        fi
        rm -f "$pidf"
    fi
    
    # Method 2: Stop via port
    local port_processes=($(lsof -t -i:"$port" 2>/dev/null))
    if [[ ${#port_processes[@]} -gt 0 ]]; then
        for p in "${port_processes[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                echo "⏳ Stopping process on port $port (PID $p)..."
                if kill "$p" 2>/dev/null; then
                    sleep 1
                    if kill -0 "$p" 2>/dev/null; then
                        kill -9 "$p" 2>/dev/null
                    fi
                    echo "✅ Stopped ${service_name} via port (PID $p)"
                    stopped=1
                fi
            fi
        done
    fi
    
    if [[ $stopped -eq 0 ]]; then
        echo "ℹ️  ${service_name} is not running and port $port is not in use"
    fi
}

# Show service status
show_status() {
    local service_name=$1
    local pidf=$2
    local port=$3
    
    echo "----------------------------------------"
    echo "Service: ${service_name}"
    echo "Port: ${port}"
    
    if is_running "$pidf"; then
        local pid=$(cat "$pidf")
        echo "Status: ✅ Running (PID $pid)"
        
        # Show process information
        ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,cmd 2>/dev/null | tail -n 1
    else
        echo "Status: ❌ Not running"
        if [[ -f "$pidf" ]]; then
            echo "⚠️  Warning: PID file exists but process does not"
        fi
    fi
    
    # Check port
    if is_port_used "$port"; then
        echo "Port: ✅ $port is in use"
    else
        echo "Port: ❌ $port is not in use"
    fi
}

# ============================================
# START command
# ============================================
if [[ "$cmd" == "start" ]]; then
    echo "=========================================="
    echo "Starting all services"
    echo "=========================================="
    echo ""
    
    # 1. Start CrawlPage
    echo "1️⃣  Starting CrawlPage service"
    pidf="$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid"
    logf="$LOG_DIR/CrawlPage_$CRAWL_PAGE_PORT.log"
    
    if is_running "$pidf"; then
        echo "⚠️  CrawlPage is already running (PID $(cat "$pidf"))"
    elif is_port_used "$CRAWL_PAGE_PORT"; then
        echo "❌ Port $CRAWL_PAGE_PORT is already in use"
        lsof -i:"$CRAWL_PAGE_PORT"
    else
        if [[ -f "$CRAWL_PAGE_SCRIPT" ]]; then
            echo "Start command: python $CRAWL_PAGE_SCRIPT"
            echo "Log file: $logf"
            nohup python -u "$CRAWL_PAGE_SCRIPT" > "$logf" 2>&1 &
            echo $! > "$pidf"
            sleep 2
            if is_running "$pidf"; then
                echo "✅ CrawlPage started successfully (PID $(cat "$pidf"))"
            else
                echo "❌ CrawlPage failed to start, check logs: tail -f $logf"
            fi
        else
            echo "❌ Script not found: $CRAWL_PAGE_SCRIPT"
        fi
    fi
    echo ""
    
    # 2. Start WebSearch (optimized)
    echo "2️⃣  Starting WebSearch service (optimized)"
    pidf="$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid"
    logf="$LOG_DIR/WebSearch_$WEBSEARCH_PORT.log"
    
    if is_running "$pidf"; then
        echo "⚠️  WebSearch is already running (PID $(cat "$pidf"))"
    elif is_port_used "$WEBSEARCH_PORT"; then
        echo "❌ Port $WEBSEARCH_PORT is already in use"
        lsof -i:"$WEBSEARCH_PORT"
    else
        if [[ -f "$WEBSEARCH_SCRIPT" ]]; then
            case "$websearch_mode" in
                dev)
                    echo "Mode: 🔧 Development mode (single process)"
                    echo "Start command: python $WEBSEARCH_SCRIPT"
                    echo "Log file: $logf"
                    nohup python -u "$WEBSEARCH_SCRIPT" > "$logf" 2>&1 &
                    echo $! > "$pidf"
                    ;;
                prod)
                    echo "Mode: 🚀 Production mode (4 workers)"
                    echo "Start command: uvicorn scripts.cached_serper_server:app --host $SERVER_HOST --port $WEBSEARCH_PORT --workers 4"
                    echo "Log file: $logf"
                    cd "$DIR" || exit 1
                    nohup uvicorn scripts.cached_serper_server:app \
                        --host "$SERVER_HOST" \
                        --port "$WEBSEARCH_PORT" \
                        --workers 4 \
                        --log-level info > "$logf" 2>&1 &
                    echo $! > "$pidf"
                    ;;
                perf)
                    echo "Mode: ⚡ High performance mode (8 workers)"
                    echo "Start command: uvicorn scripts.cached_serper_server:app --host $SERVER_HOST --port $WEBSEARCH_PORT --workers 8"
                    echo "Log file: $logf"
                    cd "$DIR" || exit 1
                    nohup uvicorn scripts.cached_serper_server:app \
                        --host "$SERVER_HOST" \
                        --port "$WEBSEARCH_PORT" \
                        --workers 8 \
                        --limit-concurrency 1000 \
                        --backlog 2048 \
                        --log-level info > "$logf" 2>&1 &
                    echo $! > "$pidf"
                    ;;
                *)
                    # echo "❌ Invalid mode: $websearch_mode"
                    # echo "Supported modes: dev, prod, perf"
                    # exit 1
                    # ;;
                    echo "Mode: ⚡ High performance mode (8 workers)"
                    echo "Start command: uvicorn scripts.cached_serper_server:app --host $SERVER_HOST --port $WEBSEARCH_PORT --workers 8"
                    echo "Log file: $logf"
                    cd "$DIR" || exit 1
                    nohup uvicorn scripts.cached_serper_server:app \
                        --host "$SERVER_HOST" \
                        --port "$WEBSEARCH_PORT" \
                        --workers 8 \
                        --limit-concurrency 1000 \
                        --backlog 2048 \
                        --log-level info > "$logf" 2>&1 &
                    echo $! > "$pidf"
                    ;;
            esac
            
            sleep 2
            if is_running "$pidf"; then
                echo "✅ WebSearch started successfully (PID $(cat "$pidf"))"
            else
                echo "❌ WebSearch failed to start, check logs: tail -f $logf"
            fi
        else
            echo "❌ Script not found: $WEBSEARCH_SCRIPT"
        fi
    fi
    echo ""
    
    echo "=========================================="
    echo "✅ All services started"
    echo "=========================================="
    echo ""
    echo "Check status: $0 status"
    echo "Check logs: tail -f $LOG_DIR/*.log"
    echo ""

# ============================================
# STOP command
# ============================================
elif [[ "$cmd" == "stop" ]]; then
    echo "=========================================="
    echo "Stopping all services"
    echo "=========================================="
    echo ""
    
    # Stop CrawlPage
    echo "1️⃣  Stopping CrawlPage"
    stop_process "CrawlPage" \
        "$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid" \
        "$CRAWL_PAGE_PORT"
    echo ""
    
    # Stop WebSearch
    echo "2️⃣  Stopping WebSearch"
    stop_process "WebSearch" \
        "$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid" \
        "$WEBSEARCH_PORT"
    echo ""
    
    echo "=========================================="
    echo "✅ All services stopped"
    echo "=========================================="
    echo ""

# ============================================
# STATUS command
# ============================================
elif [[ "$cmd" == "status" ]]; then
    echo "=========================================="
    echo "Service status"
    echo "=========================================="
    echo ""
    
    # CrawlPage status
    show_status "CrawlPage" \
        "$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid" \
        "$CRAWL_PAGE_PORT"
    echo ""
    
    # WebSearch status
    show_status "WebSearch" \
        "$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid" \
        "$WEBSEARCH_PORT"
    echo ""
    
    
    echo "=========================================="
    
    # Health check
    echo ""
    echo "🔍 Health check:"
    echo ""
    
    # WebSearch health check
    if curl -s "http://$SERVER_HOST:$WEBSEARCH_PORT/health" > /dev/null 2>&1; then
        echo "✅ WebSearch healthy: http://$SERVER_HOST:$WEBSEARCH_PORT/health"
    else
        echo "❌ WebSearch unhealthy or unreachable"
    fi
    echo ""

    # CrawlPage health check
    if curl -s "http://$SERVER_HOST:$CRAWL_PAGE_PORT/health" > /dev/null 2>&1; then
        echo "✅ CrawlPage healthy: http://$SERVER_HOST:$CRAWL_PAGE_PORT/health"
    else
        echo "❌ CrawlPage unhealthy or unreachable"
    fi
    echo ""

# ============================================
# RESTART command
# ============================================
elif [[ "$cmd" == "restart" ]]; then
    echo "🔄 Restarting all services..."
    echo ""
    
    # Stop first
    "$0" stop
    
    # Wait a bit
    echo "⏳ Waiting 3 seconds..."
    sleep 3
    echo ""
    
    # Start again
    "$0" start "$websearch_mode"

# ============================================
# TEST command
# ============================================
elif [[ "$cmd" == "test" ]]; then
    echo "=========================================="
    echo "Running tests"
    echo "=========================================="
    echo ""
    
    # Check test script paths
    TEST_ENV_SCRIPT="$DIR/v5/TestEnvironment.sh"
    TEST_WEBSEARCH_SCRIPT="$DIR/v5/test_cache_serper_server_v5.py"
    TEST_CRAWLPAGE_SCRIPT="$DIR/v5/test_crawl_page_with_cache.py"
    # TEST_CODEEXEC_SCRIPT="$DIR/v5/test_code_execute_server_v5.py"
    
    # Set test environment variables
    if [[ -f "$TEST_ENV_SCRIPT" ]]; then
        echo "--------------------Setting test environment variables------------------"
        source "$TEST_ENV_SCRIPT"
        echo "--------------------Setup complete------------------"
        echo ""
    fi
    
    # Test WebSearch
    if [[ -f "$TEST_WEBSEARCH_SCRIPT" ]]; then
        echo "--------------------Starting web search test ------------------"
        python -u "$TEST_WEBSEARCH_SCRIPT" "http://$SERVER_HOST:$WEBSEARCH_PORT/search"
        echo "-------------------------Test completed--------------------------"
        echo ""
    fi
    
    # Test CrawlPage
    if [[ -f "$TEST_CRAWLPAGE_SCRIPT" ]]; then
        echo "--------------------Starting crawl page test -------------------"
        python -u "$TEST_CRAWLPAGE_SCRIPT" "http://$SERVER_HOST:$CRAWL_PAGE_PORT/crawl_page"
        echo "-------------------------Test completed--------------------------"
        echo ""
    fi
    
# ============================================
# LOG command
# ============================================
elif [[ "$cmd" == "log" ]]; then
    service=${2:-all}
    
    if [[ "$service" == "all" ]]; then
        echo "📋 Viewing all logs in real-time (Press Ctrl+C to exit):"
        echo "=========================================="
        tail -f "$LOG_DIR"/*.log
    elif [[ "$service" == "crawlpage" ]]; then
        logf="$LOG_DIR/CrawlPage_$CRAWL_PAGE_PORT.log"
        echo "📋 CrawlPage logs (Press Ctrl+C to exit):"
        echo "=========================================="
        tail -f "$logf"
    elif [[ "$service" == "websearch" ]]; then
        logf="$LOG_DIR/WebSearch_$WEBSEARCH_PORT.log"
        echo "📋 WebSearch logs (Press Ctrl+C to exit):"
        echo "=========================================="
        tail -f "$logf"

    else
        echo "❌ Invalid service name: $service"
        echo "Supported: all, crawlpage, websearch"
        exit 1
    fi

# ============================================
# Help information
# ============================================
else
    echo "Usage: $0 <command> [websearch_mode] [service]"
    echo ""
    echo "Commands:"
    echo "  start   [mode]     - Start all services"
    echo "  stop               - Stop all services"
    echo "  status             - Check all service status"
    echo "  restart [mode]     - Restart all services"
    echo "  test               - Run tests"
    echo "  log     [service]  - View logs"
    echo ""
    echo "WebSearch mode (only for start/restart):"
    echo "  dev   - Development mode (single process, default)"
    echo "  prod  - Production mode (4 workers)"
    echo "  perf  - High performance mode (8 workers)"
    echo ""
    echo "Log service (only for log command):"
    echo "  all        - All services (default)"
    echo "  crawlpage  - CrawlPage service"
    echo "  websearch  - WebSearch service"
    echo ""
    echo "Examples:"
    echo "  $0 start dev           # Start all services in development mode"
    echo "  $0 start prod          # Start all services in production mode"
    echo "  $0 stop                # Stop all services"
    echo "  $0 status              # Check status"
    echo "  $0 restart prod        # Restart in production mode"
    echo "  $0 test                # Run tests"
    echo "  $0 log all             # View all logs"
    echo "  $0 log websearch       # View WebSearch logs only"
    echo ""
    echo "Service descriptions:"
    echo "  • CrawlPage: Page crawling service (port: $CRAWL_PAGE_PORT)"
    echo "  • WebSearch: Search service - optimized (port: $WEBSEARCH_PORT)"
    echo ""
    exit 1
fi