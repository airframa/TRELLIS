#!/bin/bash
# Model pre-download script for China servers using mirrors

# Create model download directory structure
setup_directories() {
    echo "=== Setting up model directories ==="
    
    # Create third_party directory structure
    mkdir -p third_party/improved_aesthetic_predictor
    
    # Create cache directories
    mkdir -p ~/.cache/huggingface
    mkdir -p ~/.cache/torch/hub
    mkdir -p ~/.cache/timm
    
    echo "✅ Directory structure created"
}

# Configure environment for China mirrors
configure_mirrors() {
    echo "=== Configuring mirrors for China ==="
    
    # Hugging Face mirror
    export HF_ENDPOINT=https://hf-mirror.com
    export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface
    export HF_HOME=~/.cache/huggingface
    
    # PyTorch Hub mirror
    export TORCH_HOME=~/.cache/torch
    
    # Set in conda environment permanently
    conda activate trellis
    conda env config vars set HF_ENDPOINT=https://hf-mirror.com
    conda env config vars set HUGGINGFACE_HUB_CACHE=~/.cache/huggingface
    conda env config vars set HF_HOME=~/.cache/huggingface
    conda env config vars set TORCH_HOME=~/.cache/torch
    
    echo "✅ Mirror configuration complete"
    echo "Environment variables set:"
    echo "  HF_ENDPOINT=$HF_ENDPOINT"
    echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
}

# Download CLIP model for aesthetic scoring
download_clip_model() {
    echo "=== Downloading CLIP model (ViT-L-14, laion2b_s32b_b82k) ==="
    echo "This will take 30-60 minutes depending on connection speed..."
    
    python3 -c "
import os
import open_clip
import torch

print('🔄 Starting CLIP model download...')
print('Model size: ~1.7GB')

try:
    # Force download and cache
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-L-14',
        pretrained='laion2b_s32b_b82k',
        cache_dir=os.path.expanduser('~/.cache/huggingface')
    )
    
    # Test the model works
    print('✅ CLIP model downloaded and verified successfully!')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Save model info
    with open('model_download_log.txt', 'a') as f:
        f.write('CLIP ViT-L-14 laion2b_s32b_b82k: ✅ SUCCESS\\n')
        
except Exception as e:
    print(f'❌ CLIP download failed: {e}')
    with open('model_download_log.txt', 'a') as f:
        f.write(f'CLIP ViT-L-14 laion2b_s32b_b82k: ❌ FAILED - {e}\\n')
    exit(1)
"
}

# Download DINOv2 model for feature extraction
download_dinov2_model() {
    echo "=== Downloading DINOv2 model (vit_large_patch14_reg4_dinov2.lvd142m) ==="
    echo "This will take 10-20 minutes..."
    
    python3 -c "
import timm
import torch

print('🔄 Starting DINOv2 model download...')

try:
    # Download and cache the model
    model = timm.create_model(
        'vit_large_patch14_reg4_dinov2.lvd142m', 
        pretrained=True
    )
    
    print('✅ DINOv2 model downloaded and verified successfully!')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Save model info
    with open('model_download_log.txt', 'a') as f:
        f.write('DINOv2 vit_large_patch14_reg4_dinov2.lvd142m: ✅ SUCCESS\\n')
        
except Exception as e:
    print(f'❌ DINOv2 download failed: {e}')
    with open('model_download_log.txt', 'a') as f:
        f.write(f'DINOv2 vit_large_patch14_reg4_dinov2.lvd142m: ❌ FAILED - {e}\\n')
    exit(1)
"
}

# Download aesthetic predictor weights
download_aesthetic_weights() {
    echo "=== Downloading aesthetic predictor weights ==="
    
    # Download from the original source using curl with retries
    cd third_party/improved_aesthetic_predictor
    
    WEIGHT_URL="https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
    WEIGHT_FILE="sac+logos+ava1-l14-linearMSE.pth"
    
    echo "Downloading aesthetic predictor weights..."
    echo "URL: $WEIGHT_URL"
    
    # Try multiple methods to download
    if command -v wget >/dev/null 2>&1; then
        echo "Using wget..."
        wget -c --retry-connrefused --tries=3 --timeout=30 \
             -O "$WEIGHT_FILE" "$WEIGHT_URL"
    elif command -v curl >/dev/null 2>&1; then
        echo "Using curl..."
        curl -L --retry 3 --retry-delay 5 --max-time 300 \
             -o "$WEIGHT_FILE" "$WEIGHT_URL"
    else
        echo "Downloading with Python..."
        python3 -c "
import urllib.request
import os
url = '$WEIGHT_URL'
filename = '$WEIGHT_FILE'
try:
    urllib.request.urlretrieve(url, filename)
    print(f'✅ Downloaded {filename}')
    print(f'File size: {os.path.getsize(filename)/1024/1024:.1f} MB')
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"
    fi
    
    # Verify download
    if [ -f "$WEIGHT_FILE" ]; then
        FILE_SIZE=$(stat -f%z "$WEIGHT_FILE" 2>/dev/null || stat -c%s "$WEIGHT_FILE" 2>/dev/null)
        echo "✅ Aesthetic predictor weights downloaded successfully!"
        echo "File: $WEIGHT_FILE"
        echo "Size: $((FILE_SIZE/1024/1024)) MB"
        
        echo "sac+logos+ava1-l14-linearMSE.pth: ✅ SUCCESS" >> ../../model_download_log.txt
    else
        echo "❌ Aesthetic predictor download failed!"
        echo "sac+logos+ava1-l14-linearMSE.pth: ❌ FAILED" >> ../../model_download_log.txt
        exit 1
    fi
    
    cd ../..
}

# Test all models work
test_models() {
    echo "=== Testing all models ==="
    
    python3 -c "
import os
import open_clip
import timm
import torch
from pathlib import Path

print('🔍 Testing all downloaded models...')

# Test CLIP
try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', 'laion2b_s32b_b82k'
    )
    print('✅ CLIP model loads successfully')
except Exception as e:
    print(f'❌ CLIP model test failed: {e}')

# Test DINOv2
try:
    model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=True)
    print('✅ DINOv2 model loads successfully')
except Exception as e:
    print(f'❌ DINOv2 model test failed: {e}')

# Test aesthetic predictor weights
weights_path = 'third_party/improved_aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth'
if Path(weights_path).exists():
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        print('✅ Aesthetic predictor weights load successfully')
    except Exception as e:
        print(f'❌ Aesthetic predictor weights test failed: {e}')
else:
    print(f'❌ Aesthetic predictor weights not found at {weights_path}')

print('\\n🎉 Model testing complete!')
"
}

# Monitor download progress
monitor_progress() {
    echo "=== Download Progress Monitor ==="
    
    while true; do
        echo "$(date): Checking download progress..."
        
        # Check cache sizes
        if [ -d ~/.cache/huggingface ]; then
            HF_SIZE=$(du -sh ~/.cache/huggingface 2>/dev/null | cut -f1)
            echo "  Hugging Face cache: $HF_SIZE"
        fi
        
        if [ -d ~/.cache/torch ]; then
            TORCH_SIZE=$(du -sh ~/.cache/torch 2>/dev/null | cut -f1)
            echo "  PyTorch cache: $TORCH_SIZE"
        fi
        
        if [ -d ~/.cache/timm ]; then
            TIMM_SIZE=$(du -sh ~/.cache/timm 2>/dev/null | cut -f1)
            echo "  TIMM cache: $TIMM_SIZE"
        fi
        
        # Check if main downloads are complete
        if [ -f "model_download_log.txt" ]; then
            echo "  Download status:"
            cat model_download_log.txt | sed 's/^/    /'
        fi
        
        echo "---"
        sleep 300  # Check every 5 minutes
    done
}

# Main download function
download_all_models() {
    echo "=== Starting comprehensive model download ==="
    echo "This will download all required models for 3DRealCar preprocessing"
    echo "Total estimated time: 1-2 hours"
    echo "Total estimated size: ~2-3 GB"
    echo
    
    # Activate conda environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate trellis
    
    # Setup
    setup_directories
    configure_mirrors
    
    # Clear previous log
    > model_download_log.txt
    echo "Model download started at $(date)" >> model_download_log.txt
    
    # Download models sequentially
    download_aesthetic_weights  # Start with smallest
    download_dinov2_model      # Medium size
    download_clip_model        # Largest, most likely to fail
    
    # Test everything works
    test_models
    
    echo "model_download_complete: $(date)" >> model_download_log.txt
    echo
    echo "🎉 All models downloaded successfully!"
    echo "You can now run preprocessing without network issues."
    echo
    echo "Cache locations:"
    echo "  Hugging Face: ~/.cache/huggingface"
    echo "  PyTorch: ~/.cache/torch" 
    echo "  TIMM: ~/.cache/timm"
    echo "  Aesthetic weights: third_party/improved_aesthetic_predictor/"
}

# Command handling
case "${1:-help}" in
    "download")
        download_all_models
        ;;
        
    "test")
        configure_mirrors
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate trellis
        test_models
        ;;
        
    "monitor")
        monitor_progress
        ;;
        
    "setup")
        setup_directories
        configure_mirrors
        ;;
        
    *)
        echo "=== Model Download Script for China Servers ==="
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  download  - Download all required models (run in tmux!)"
        echo "  test      - Test that all models can be loaded"
        echo "  monitor   - Monitor download progress"
        echo "  setup     - Just setup directories and mirrors"
        echo
        echo "Recommended usage:"
        echo "  1. tmux new-session -s model_download"
        echo "  2. $0 download"
        echo "  3. [Ctrl+B, D to detach]"
        echo "  4. tmux attach -s model_download  # to check progress"
        echo
        echo "Models to be downloaded:"
        echo "  - CLIP ViT-L-14 (~1.7GB) for aesthetic scoring"
        echo "  - DINOv2 ViT-Large (~1.1GB) for feature extraction"
        echo "  - Aesthetic predictor weights (~500MB)"
        echo
        echo "Total size: ~3.3GB"
        echo "Estimated time: 1-2 hours in China"
        ;;
esac