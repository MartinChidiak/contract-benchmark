#!/bin/bash
cd /home/martin/tesis_vllm_5090

docker run --rm --gpus all -p 8501:8501 \
  -v "$(pwd):/app" \
  -v "$HOME/.cache/huggingface:/hf_cache" \
  -e HF_HOME=/hf_cache \
  contract-benchmark \
  streamlit run /app/app.py --server.port 8501 --server.address 0.0.0.0 &

DOCKER_PID=$!
until curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; do sleep 0.5; done
cmd.exe /c start http://localhost:8501
wait $DOCKER_PID
