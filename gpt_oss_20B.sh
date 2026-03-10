#!/bin/bash
# Setup and run GPT-OSS-20B on Hyperstack A100 (Ubuntu)
# OpenAI-compatible API will be available at http://localhost:8000

set -e

sudo systemctl stop unattended-upgrades
sudo systemctl disable unattended-upgrades


echo "=== Step 1: Install Python 3.12 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

echo "=== Step 2: Create Virtual Environment ==="
python3.12 -m venv .venv
source .venv/bin/activate

echo "=== Step 3: Upgrade pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Step 4: Install PyTorch (cu121) - known-good for Hyperstack A100 ==="
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 5: Verify PyTorch and CUDA ==="
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "=== Step 6: Install vLLM ==="
pip install vllm

sudo systemctl enable unattended-upgrades
sudo systemctl start unattended-upgrades

echo "=== Step 7: Run the vLLM Server (GPT-OSS-20B on A100) ==="
vllm serve openai/gpt-oss-20b \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --async-scheduling
