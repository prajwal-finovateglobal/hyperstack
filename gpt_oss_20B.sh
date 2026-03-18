#!/bin/bash
# Setup and run GPT-OSS-20B on Hyperstack A100 (Ubuntu)
# OpenAI-compatible API will be available at http://localhost:8000

set -e

echo "=== Disable unattended upgrades (avoid apt locks) ==="
sudo systemctl stop unattended-upgrades || true
sudo systemctl disable unattended-upgrades || true

echo "=== Step 1: Install Python 3.12 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

echo "=== Step 2: Create Fresh Virtual Environment ==="
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

echo "=== Step 3: Upgrade pip and install compatible setuptools ==="
pip install --upgrade pip
pip install "setuptools<81" wheel

echo "=== Step 4: Install vLLM (this will install correct torch stack) ==="
pip install vllm

echo "=== Step 5: Verify PyTorch + CUDA ==="
python3 - <<EOF
import torch
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "=== Step 6: Re-enable unattended upgrades ==="
sudo systemctl enable unattended-upgrades || true
sudo systemctl start unattended-upgrades || true

echo "=== Step 7: Start vLLM Server (GPT-OSS-20B) ==="
echo "Server will run at: http://localhost:8000"

vllm serve openai/gpt-oss-20b \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --async-scheduling
