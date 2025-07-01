#!/bin/bash
# Launch script for flexible batch processing

# Create configuration file
create_config() {
    cat > batch_processing_config.json << 'EOF'
{
    "extract_dir": "/data/francMB/threedrealcar/pipeline-workspace/extracted",
    "output_dir": "/data/francMB/threedrealcar",
    "script_path": "/home/FrancMB/projects/prototyping/TRELLIS/fine-tuning/preprocessing/threedrealcar_preprocess.py",
    "log_dir": "/data/francMB/threedrealcar/processing_logs",
    "max_parallel": 4
}
EOF
}

# Function to run batch processing
run_batch_processing() {
    local BATCHES=$1
    local CLEANUP=${2:-false}
    
    echo "=== 3DRealCar Batch Processing ==="
    echo "Batches: $BATCHES"
    echo "Cleanup after: $CLEANUP"
    echo
    
    # Activate conda environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate trellis
    
    # Run the processor
    if [ "$CLEANUP" = "true" ]; then
        python ./fine-tuning/preprocessing/threedrealcar_flexible_batch_processor.py \
            --config batch_processing_config.json \
            --batches "$BATCHES" \
            --cleanup
    else
        python ./fine-tuning/preprocessing/threedrealcar_flexible_batch_processor.py \
            --config batch_processing_config.json \
            --batches "$BATCHES"
    fi
}

# Function to monitor progress
monitor_progress() {
    local LOG_DIR="/data/francMB/threedrealcar/processing_logs"
    
    echo "=== Monitoring Options ==="
    echo "1. Main log: tail -f $LOG_DIR/pipeline_*.log"
    echo "2. Progress only: tail -f $LOG_DIR/progress.log"
    echo "3. Specific batch: tail -f $LOG_DIR/batch_logs/BATCH_NAME.log"
    echo "4. Processing state: cat $LOG_DIR/processing_state.json | python -m json.tool"
    echo
    
    # Show current state
    if [ -f "$LOG_DIR/processing_state.json" ]; then
        echo "Current State:"
        python3 -c "
import json
with open('$LOG_DIR/processing_state.json', 'r') as f:
    state = json.load(f)
    print(f'  Processed: {len(state[\"processed\"])} batches')
    print(f'  Failed: {len(state[\"failed\"])} batches')
    print(f'  Processing: {len(state[\"processing\"])} batches')
    if state['processed']:
        print(f'  Completed: {\", \".join(state[\"processed\"])}')
"
    fi
}

# Create config file at start
create_config

# Main menu
case "${1:-menu}" in
    "batch1")
        # Process first 4 batches
        tmux new-session -d -s batch_processing_1 \
            "bash -c 'source $(readlink -f $0); run_batch_processing \"0-200,1000-1200,1200-1400,1400-1600\" false; exec bash'"
        echo "Started batch processing in tmux session: batch_processing_1"
        echo "To attach: tmux attach -t batch_processing_1"
        ;;
        
    "batch2")
        # Process next 4 batches
        tmux new-session -d -s batch_processing_2 \
            "bash -c 'source $(readlink -f $0); run_batch_processing \"200-400,400-600,600-800,800-1000\" false; exec bash'"
        echo "Started batch processing in tmux session: batch_processing_2"
        echo "To attach: tmux attach -t batch_processing_2"
        ;;
        
    "batch3")
        # Process final batches
        tmux new-session -d -s batch_processing_3 \
            "bash -c 'source $(readlink -f $0); run_batch_processing \"HQ300\" false; exec bash'"
        echo "Started batch processing in tmux session: batch_processing_3"
        echo "To attach: tmux attach -t batch_processing_3"
        ;;
        
    "custom")
        # Custom batch selection
        if [ -z "$2" ]; then
            echo "Usage: $0 custom \"batch1,batch2,batch3\""
            exit 1
        fi
        tmux new-session -d -s batch_processing_custom \
            "bash -c 'source $(readlink -f $0); run_batch_processing \"$2\" false; exec bash'"
        echo "Started batch processing in tmux session: batch_processing_custom"
        echo "To attach: tmux attach -t batch_processing_custom"
        ;;
        
    "monitor")
        monitor_progress
        ;;
        
    "cleanup")
        # Manual cleanup of extracted directories
        if [ -z "$2" ]; then
            echo "Usage: $0 cleanup \"batch1,batch2,batch3\""
            exit 1
        fi
        echo "Cleaning up extracted directories for: $2"
        python3 -c "
import shutil
from pathlib import Path
batches = '$2'.split(',')
extract_dir = Path('/data/francMB/threedrealcar/pipeline-workspace/extracted')
total_freed = 0
for batch in batches:
    batch_path = extract_dir / batch.strip()
    if batch_path.exists():
        size_gb = sum(f.stat().st_size for f in batch_path.rglob('*') if f.is_file()) / 1e9
        print(f'Removing {batch} ({size_gb:.1f}GB)...')
        shutil.rmtree(batch_path)
        total_freed += size_gb
print(f'Total freed: {total_freed:.1f}GB')
"
        ;;
        
    *)
        echo "=== 3DRealCar Flexible Batch Processing ==="
        echo
        echo "Usage: $0 [command] [options]"
        echo
        echo "Commands:"
        echo "  batch1    - Process first 4 batches (0-200,1000-1200,1200-1400,1400-1600)"
        echo "  batch2    - Process next 4 batches (200-400,400-600,600-800,800-1000)"
        echo "  batch3    - Process final batches (1600-1800,1800-2045,HQ200,HQ300)"
        echo "  custom    - Process custom batches: $0 custom \"batch1,batch2,...\""
        echo "  monitor   - Monitor progress and view logs"
        echo "  cleanup   - Clean up extracted dirs: $0 cleanup \"batch1,batch2,...\""
        echo
        echo "Example workflow:"
        echo "  1. $0 batch1                    # Start first batch"
        echo "  2. $0 monitor                   # Check progress"
        echo "  3. $0 cleanup \"0-200,1000-1200,1200-1400,1400-1600\"  # Free space"
        echo "  4. $0 batch2                    # Continue with next batch"
        ;;
esac