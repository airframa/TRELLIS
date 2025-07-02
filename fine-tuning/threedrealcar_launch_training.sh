#!/bin/bash
# Working multi-GPU launch script
# Usage: ./launch_working.sh <config_name>

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis

CONFIG_NAME=${1:-"stage2"}

# Set paths
TRELLIS_ROOT="/home/FrancMB/projects/prototyping/TRELLIS"
DATA_DIR="${TRELLIS_ROOT}/assets/3drealcar/HQ200-processed-trellis"
CONFIG_FILE="configs/3drealcar_${CONFIG_NAME}.json"
OUTPUT_DIR="/data/francMB/threedrealcar/outputs/3drealcar_decoder_${CONFIG_NAME}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Copy the fixed training script
cp ./fine-tuning/threedrealcar_train_decoder_finetune.py ${OUTPUT_DIR}/

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('GPU memory cleared')
"

# Set memory optimizations
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:64"

# Set NCCL environment (same as test script that worked)
export NCCL_DEBUG=WARN  # Change to INFO for debugging
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))

# Use GPUs 1-7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

echo "============================================"
echo "Configuration: ${CONFIG_NAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} total)"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

# Run with torchrun (same command that worked in test)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    ./fine-tuning/threedrealcar_train_decoder_finetune.py \
    --config ${CONFIG_FILE} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_gpus $NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "Training complete!"