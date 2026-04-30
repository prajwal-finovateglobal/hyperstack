#!/bin/bash
set -e

echo "=== Step 0: Disable unattended upgrades ==="
sudo systemctl stop unattended-upgrades || true
sudo systemctl disable unattended-upgrades || true

echo "=== Step 1: Check for NVIDIA GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi && echo "✅ GPU detected" || { echo "❌ GPU not working"; exit 1; }
else
    echo "❌ nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

echo "=== Step 2: Install Python 3.12 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

echo "=== Step 3: Create Virtual Environment ==="
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

echo "=== Step 4: Upgrade pip ==="
pip install --upgrade pip
pip install "setuptools<81" wheel

echo "=== Step 5: Install PyTorch for CUDA 12.2 FIRST ==="
# Pin torch to CUDA 12.1 wheel (compatible with driver CUDA 12.2)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 6: Install vLLM (no-deps for torch, then install rest) ==="
# Install vLLM but tell it not to overwrite torch
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== Step 7: Verify PyTorch + CUDA ==="
python3 - <<EOF
import torch
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available. GPU setup failed.")
print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "=== Step 8: Start vLLM Server ==="
echo "🚀 Server running at: http://localhost:8000"
vllm serve openai/gpt-oss-20b \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 8192
