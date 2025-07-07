import torch
import sys
import os
# Add the TRELLIS root directory to sys.path
trellis_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, trellis_root)
from trellis.pipelines import TrellisImageTo3DPipeline

# Load the base pipeline to check its structure
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")

# Check what models are in the pipeline
print("Models in TRELLIS pipeline:")
for name, model in pipeline.models.items():
    print(f"  - {name}: {type(model).__name__}")
    if hasattr(model, 'in_channels'):
        print(f"    Input channels: {model.in_channels}")
    if hasattr(model, 'out_channels'):
        print(f"    Output channels: {model.out_channels}")

# Load your checkpoint to check its structure
checkpoint_path = "/data/francMB/threedrealcar/outputs/3drealcar_final_20250703_222220/ckpts/decoder_ema0.9999_step0030000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint keys (first 10):")
for i, key in enumerate(list(checkpoint.keys())[:10]):
    print(f"  {key}")

# Check if the checkpoint matches any decoder
print("\nChecking compatibility...")
for decoder_name in ['slat_decoder_gs', 'slat_decoder_mesh', 'slat_decoder_rf']:
    if decoder_name in pipeline.models:
        decoder = pipeline.models[decoder_name]
        decoder_state = decoder.state_dict()
        
        # Check if keys match
        checkpoint_keys = set(checkpoint.keys())
        decoder_keys = set(decoder_state.keys())
        
        common_keys = checkpoint_keys.intersection(decoder_keys)
        print(f"\n{decoder_name}:")
        print(f"  Common keys: {len(common_keys)}/{len(decoder_keys)}")
        
        if len(common_keys) == len(decoder_keys):
            print(f"  ✓ Perfect match! Use this decoder.")
        elif len(common_keys) > 0:
            print(f"  ⚠ Partial match. May need adaptation.")
        else:
            print(f"  ✗ No match.")