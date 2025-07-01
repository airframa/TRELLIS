#!/bin/bash
# Universal launch script for TRELLIS decoder fine-tuning: testing different configurations
# Usage: ./launch_training.sh <config_name> [num_gpus]
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis

# Verify python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please ensure conda environment is activated."
    exit 1
fi

# Get arguments
CONFIG_NAME=${1:-"ultra_low"}  # Default to ultra_low if not specified
NUM_GPUS=${2:-1}               # Default to 1 GPU if not specified

# Set paths
TRELLIS_ROOT="/home/FrancMB/projects/prototyping/TRELLIS"
DATA_DIR="${TRELLIS_ROOT}/assets/3drealcar/HQ200-processed-trellis"
CONFIG_FILE="configs/3drealcar_${CONFIG_NAME}.json"
OUTPUT_DIR="${TRELLIS_ROOT}/outputs/3drealcar_decoder_${CONFIG_NAME}"

# Validate config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file ${CONFIG_FILE} not found!"
    echo "Available configs: ultra_low, stage1, stage2, original"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Save the config and trainer files for reproducibility
cp fine-tuning/threedrealcar_decoder_only_trainer.py ${OUTPUT_DIR}/
cp fine-tuning/threedrealcar_train_decoder_finetune.py ${OUTPUT_DIR}/
cp ${CONFIG_FILE} ${OUTPUT_DIR}/

# Clear all GPU caches
echo "Clearing GPU memory..."
python -c "
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('GPU memory cleared')
"

# Set memory optimization for configs that need it
if [[ "$CONFIG_NAME" == "ultra_low" ]] || [[ "$CONFIG_NAME" == "stage1" ]] || [[ "$CONFIG_NAME" == "stage2" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
    export CUDA_LAUNCH_BLOCKING=0  # Better performance
    export CUBLAS_WORKSPACE_CONFIG=:4096:8

    # Disable unnecessary features
    export WARP_DISABLE=1
    export TF_CPP_MIN_LOG_LEVEL=3
    echo "Memory optimization enabled"
fi

# Determine which GPUs to use (avoiding preprocessing GPUs)
if [ $NUM_GPUS -eq 1 ]; then
    # Use a single free GPU
    export CUDA_VISIBLE_DEVICES=1
    echo "Using GPU 1 (free GPU)"
elif [ $NUM_GPUS -le 5 ]; then
    # Use the free GPUs: 1,2,4,5,7
    export CUDA_VISIBLE_DEVICES=1,2,4,5,7
    echo "Using GPUs 1,2,4,5,7 (free GPUs)"
    NUM_GPUS=5  # Adjust to actual available
else
    echo "Warning: Only 5 GPUs available for training (0,3,6 are preprocessing)"
    export CUDA_VISIBLE_DEVICES=1,2,4,5,7
    NUM_GPUS=5
fi

# Display configuration info
echo "============================================"
echo "Training Configuration: ${CONFIG_NAME}"
echo "Config file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

# Run training
echo "Starting training..."
python -u ./fine-tuning/threedrealcar_train_decoder_finetune.py \
    --config ${CONFIG_FILE} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_gpus ${NUM_GPUS} \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "Training complete! Check ${OUTPUT_DIR} for results."