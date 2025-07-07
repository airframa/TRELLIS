#!/usr/bin/env python3
"""
Extract the pretrained decoder from TRELLIS pipeline to use as starting point
"""
import os
import sys
import torch
# Add the TRELLIS root directory to sys.path
trellis_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, trellis_root)

from trellis.pipelines import TrellisImageTo3DPipeline

# Output directory
output_dir = "/data/francMB/threedrealcar/pretrained"
os.makedirs(output_dir, exist_ok=True)

print("Loading TRELLIS pipeline...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")

# Extract and save the Gaussian decoder
decoder_state = pipeline.models['slat_decoder_gs'].state_dict()
output_path = os.path.join(output_dir, "trellis_decoder_gs_pretrained.pt")
torch.save(decoder_state, output_path)

print(f"\nSaved pretrained decoder to: {output_path}")
print(f"Number of parameters: {len(decoder_state)}")
print(f"Total parameters: {sum(p.numel() for p in decoder_state.values()):,}")

# Verify by listing first few parameters
print("\nFirst 10 parameters:")
for i, (name, param) in enumerate(decoder_state.items()):
    if i >= 10:
        break
    print(f"  {name}: {list(param.shape)}")

print("\nDone! Use this checkpoint as --pretrained_decoder in your training script.")