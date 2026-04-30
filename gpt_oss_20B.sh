#!/bin/bash
set -e

echo "=== Step 0: Disable unattended upgrades ==="
sudo systemctl stop unattended-upgrades || true
sudo systemctl disable unattended-upgrades || true

echo "=== Step 1: Verify GPU ==="
nvidia-smi || { echo "❌ nvidia-smi failed"; exit 1; }
echo "✅ GPU detected"

echo "=== Step 2: Install Python 3.12 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y
sudo apt install -y python3.12 python3.12-venv python3.12-dev

echo "=== Step 3: Create Virtual Environment ==="
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

echo "=== Step 4: Upgrade pip ==="
pip install --upgrade pip
pip install "setuptools<81" wheel

echo "=== Step 5: Install vLLM (latest, auto-installs torch for CUDA 12.8) ==="
pip install vllm

echo "=== Step 6: Fix pyairports (outlines dependency bug) ==="
mkdir -p .venv/lib/python3.12/site-packages/pyairports
echo "AIRPORT_LIST = []" > .venv/lib/python3.12/site-packages/pyairports/airports.py
touch .venv/lib/python3.12/site-packages/pyairports/__init__.py

echo "=== Step 7: Verify CUDA ==="
python3 - <<'EOF'
import torch
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available. GPU setup failed.")
print("GPU:", torch.cuda.get_device_name(0))
print("✅ CUDA OK")
EOF

echo "=== Step 8: Set HuggingFace token (required for gated model) ==="
# Get your token from https://huggingface.co/settings/tokens
# Then run: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set. Model download may fail if gated."
    echo "    Run: export HF_TOKEN=hf_your_token_here"
    echo "    Then re-run this script or just Step 9."
fi

echo "=== Step 9: Start vLLM Server ==="
echo "🚀 Server starting at: http://0.0.0.0:8000"
echo "   OpenAI-compatible API: http://0.0.0.0:8000/v1"

vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --served-model-name gpt-oss-20b
