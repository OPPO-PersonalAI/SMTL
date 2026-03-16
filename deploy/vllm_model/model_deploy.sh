#!/bin/bash
set -euo pipefail

export TORCHDYNAMO_VERBOSE=1
export OMP_NUM_THREADS=16
export VLLM_USE_V1=1

model_path=./model/smtl
base_modelname=smtl
port=1
gpu_id=0
max_model_len=131072
LOG_DIR="./deploy/vllm_model/logs"
WAIT_TIMEOUT=180

mkdir -p "$LOG_DIR"

net0_ip=$(ifconfig net0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1 || true)
if [ -z "${net0_ip}" ]; then
    net0_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
if [ -z "${net0_ip}" ]; then
    net0_ip="0.0.0.0"
fi

ip_sanitized=$(echo "$net0_ip" | tr '.' '_')
log_file="${LOG_DIR}/${base_modelname}_${ip_sanitized}.log"
pid_file="${LOG_DIR}/${base_modelname}.pid"

echo "Deploying model: ${base_modelname} on GPU ${gpu_id}, port ${port}"

nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    vllm serve ${model_path} \
        --served-model-name ${base_modelname} \
        --max-model-len ${max_model_len} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --enable-chunked-prefill \
        --max-num-batched-tokens 6144 \
        --max-num-seqs 16 \
        --enable-prefix-caching \
        --swap-space 64 \
        --trust-remote-code \
        --uvicorn-log-level warning \
        --host 0.0.0.0 \
        --port ${port}
" > "$log_file" 2>&1 &

echo $! > "$pid_file"

echo "Waiting for server to start..."
start_time=$(date +%s)
server_started=0
while [ $(( $(date +%s) - start_time )) -lt $WAIT_TIMEOUT ]; do
    if grep -q "INFO:     Started server process" "$log_file"; then
        server_started=1
        break
    fi
    sleep 5
done

if [ $server_started -eq 1 ]; then
    echo "Server started successfully"
else
    echo "Warning: Server did not start within ${WAIT_TIMEOUT} seconds"
fi

echo -e "\n===== Deployment Complete ====="
echo "Server IP: $net0_ip"
echo "Access URL: http://$net0_ip:$port"
echo "Log file: $log_file"
echo "PID file: $pid_file"
echo "================================"