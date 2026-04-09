#!/bin/bash
# Setup and run GPT-OSS-20B on Hyperstack A100 (Ubuntu)

set -e

echo "=== Step 0: Disable unattended upgrades (avoid apt locks) ==="
sudo systemctl stop unattended-upgrades || true
sudo systemctl disable unattended-upgrades || true

echo "=== Step 1: Check for NVIDIA GPU ==="
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi found. Checking GPU..."
    if nvidia-smi; then
        echo "✅ GPU detected"
    else
        echo "⚠️ nvidia-smi exists but GPU not working. Reinstalling driver..."
        sudo apt update
        sudo apt install -y nvidia-driver-550
        sudo reboot
        exit 0
    fi
else
    echo "❌ nvidia-smi not found. Installing NVIDIA driver..."
    sudo apt update
    sudo apt install -y nvidia-driver-550
    echo "⚠️ Reboot required. Run script again after reboot."
    sudo reboot
    exit 0
fi

echo "=== Step 2: Install Python 3.12 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

echo "=== Step 3: Create Fresh Virtual Environment ==="
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

echo "=== Step 4: Upgrade pip and install compatible setuptools ==="
pip install --upgrade pip
pip install "setuptools<81" wheel

echo "=== Step 5: Install vLLM (auto installs correct torch) ==="
pip install vllm

echo "=== Step 6: Verify PyTorch + CUDA ==="
python3 - <<EOF
import torch
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available. GPU setup failed.")

print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "=== Step 7: Re-enable unattended upgrades ==="
sudo systemctl enable unattended-upgrades || true
sudo systemctl start unattended-upgrades || true

echo "=== Step 8: Start vLLM Server ==="
echo "🚀 Server running at: http://localhost:8000"

vllm serve openai/gpt-oss-20b \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --async-scheduling
