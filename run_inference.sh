#!/bin/bash
set -e  
echo "Checking CUDA installation..."

# =====================[ 1. Detect CUDA ]=====================
# Try to find nvcc in PATH
if command -v nvcc &> /dev/null; then
    echo "nvcc not found in PATH, trying local user install..."

    export CUDA_HOME="/CUDA_PATH/cuda-12.4"

    if [ -d "$CUDA_HOME" ]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        echo "Found local CUDA installation at $CUDA_HOME"
    else
        echo "CUDA not found at $CUDA_HOME"
        echo "Please check if cuda-12.4.tar.gz has been extracted correctly."
        exit 1
    fi
else
    echo "System CUDA found at $(which nvcc)"
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
fi

# =====================[ 2. Verify CUDA works ]=====================
if command -v nvcc &> /dev/null; then
    echo "nvcc version:"
    nvcc --version
else
    echo "CUDA Toolkit not found (no nvcc). Using PyTorch's built-in CUDA runtime."
fi

# =====================[ 3. Activate conda env ]=====================
echo "Activating conda environment..."

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate /envs/rna-agent || {
        echo "Failed to activate conda environment!"
        exit 1
    }
else
    echo "conda command not found. Please load conda first."
    exit 1
fi

# =====================[ 4. GPU check via PyTorch ]=====================
echo "Checking GPU availability via PyTorch..."
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is NOT available. Check driver or environment.")
EOF

# =====================[ 5. Set environment variables ]=====================
echo "Setting CUDA environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# =====================[ 6. Run DeepSpeed ​​evaluation ]=====================
echo "Launching DeepSpeed evaluation..."
DEEPSPEED_PATH="/deepspeed_PATH"

$DEEPSPEED_PATH --num_gpus 1 inference_STARGen_BCD.py \
  --model_path /model \
  --train_data /test_data.jsonl \
  --output_dir /eval_result \
  --seed 42

echo "Evaluation script finished."

