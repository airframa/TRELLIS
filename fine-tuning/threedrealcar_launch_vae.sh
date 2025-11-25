#!/bin/bash
# Production training script for 3DRealCar VAE fine-tuning (decoder only)
# Usage: ./threedrealcar_launch_vae_training.sh [session_name]

SESSION_NAME=${1:-"3drealcar_vae_training"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis

# Set paths
TRELLIS_ROOT="/home/fmb/projects/TRELLIS"
# TRAIN_DIR="/data/fmb/threedrealcar/preprocessed/merged"
TRAIN_DIR="/data/fmb/threedrealcar/preprocessed/hybrid_tests/hybrid_25pct"
# VAL_DIR="/data/fmb/threedrealcar/preprocessed/merged"
VAL_DIR="/data/fmb/threedrealcar/preprocessed/hybrid_tests/hybrid_25pct"
CONFIG_FILE="configs/3drealcar_vae.json"  
OUTPUT_DIR="/data/fmb/threedrealcar/outputs/3drealcar_vae_${TIMESTAMP}"

# IMPORTANT: Pretrained models from TRELLIS
PRETRAINED_ENCODER="/data/fmb/threedrealcar/pretrained/trellis_encoder_slat_pretrained.pt"
PRETRAINED_DECODER="/data/fmb/threedrealcar/pretrained/trellis_decoder_gs_pretrained.pt"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs

# Copy configuration files for reproducibility
cp ${CONFIG_FILE} ${OUTPUT_DIR}/
cp ./fine-tuning/threedrealcar_train_vae.py ${OUTPUT_DIR}/

# Create run script that will be executed in tmux
cat > ${OUTPUT_DIR}/run_training.sh << EOF
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis

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

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=\$((29500 + RANDOM % 1000))
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
NUM_GPUS=10

echo "============================================"
echo "3DRealCar VAE Fine-tuning (Decoder Only)"
echo "Started at: \$(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training data: ${TRAIN_DIR}"
echo "Validation data: ${VAL_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Pretrained Encoder: ${PRETRAINED_ENCODER}"
echo "Pretrained Decoder: ${PRETRAINED_DECODER}"
echo "GPUs: \${CUDA_VISIBLE_DEVICES} (\${NUM_GPUS} total)"
echo "Master: \${MASTER_ADDR}:\${MASTER_PORT}"
echo "============================================"

echo -e "\n\nStarting VAE fine-tuning (decoder only)..."

# Run training
torchrun \
    --nproc_per_node=\$NUM_GPUS \
    --master_addr=\$MASTER_ADDR \
    --master_port=\$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    ./fine-tuning/threedrealcar_train_vae.py \
    --config ${CONFIG_FILE} \
    --data_dir ${TRAIN_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --pretrained_encoder ${PRETRAINED_ENCODER} \
    --pretrained_decoder ${PRETRAINED_DECODER} \
    --freeze_encoder \
    --num_gpus \$NUM_GPUS \
    --master_addr \$MASTER_ADDR \
    --master_port \$MASTER_PORT

echo "Training completed at: \$(date)"
EOF

chmod +x ${OUTPUT_DIR}/run_training.sh

# Create or attach to tmux session
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "Attaching to existing tmux session: ${SESSION_NAME}"
    tmux attach-session -t ${SESSION_NAME}
else
    echo "Creating new tmux session: ${SESSION_NAME}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Logs will be saved to: ${OUTPUT_DIR}/logs/training.log"
    
    # Create tmux session and run training
    tmux new-session -d -s ${SESSION_NAME} -n training \
        "cd ${TRELLIS_ROOT} && ${OUTPUT_DIR}/run_training.sh 2>&1 | tee ${OUTPUT_DIR}/logs/training.log"
    
    # Create monitoring window
    tmux new-window -t ${SESSION_NAME}:1 -n monitor \
        "cd ${OUTPUT_DIR} && watch -n 30 'tail -n 50 logs/training.log'"
    
    # Create GPU monitoring window
    tmux new-window -t ${SESSION_NAME}:2 -n gpu \
        "watch -n 2 nvidia-smi"
    
    # Create tensorboard window (optional but useful)
    tmux new-window -t ${SESSION_NAME}:3 -n tensorboard \
        "cd ${OUTPUT_DIR} && tensorboard --logdir=tb_logs --port=6006"
    
    echo ""
    echo "VAE fine-tuning started in tmux session: ${SESSION_NAME}"
    echo ""
    echo "Useful tmux commands:"
    echo "  - Attach to session:     tmux attach -t ${SESSION_NAME}"
    echo "  - Detach from session:   Ctrl+B, then D"
    echo "  - Switch windows:        Ctrl+B, then window number (0-3)"
    echo "  - Kill session:          tmux kill-session -t ${SESSION_NAME}"
    echo ""
    echo "Windows:"
    echo "  0: training   - Main training process"
    echo "  1: monitor    - Live log monitoring"
    echo "  2: gpu        - GPU usage monitoring"
    echo "  3: tensorboard - TensorBoard (http://localhost:6006)"
    echo ""
    echo "Log file: ${OUTPUT_DIR}/logs/training.log"
    echo ""
    echo "IMPORTANT: Encoder is FROZEN, only decoder is being fine-tuned!"
fi