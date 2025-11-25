import os
import torch
from PIL import Image
import imageio
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Configuration
CHECKPOINT_PATH = "/data/fmb/threedrealcar/outputs/3drealcar_vae_20251110_072659/ckpts/decoder_ema0.9999_step0180000.pt"
# CHECKPOINT_PATH = "/data/fmb/threedrealcar/pretrained/trellis_decoder_gs_pretrained.pt"

# Load the base pipeline
print("Loading TRELLIS pipeline...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Load your fine-tuned decoder weights
print(f"Loading fine-tuned decoder from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda')

# Replace the decoder weights with your fine-tuned ones
# The pipeline has multiple decoders for different outputs
if 'slat_decoder_gs' in pipeline.models:
    # This is the Gaussian decoder - most likely what you fine-tuned
    pipeline.models['slat_decoder_gs'].load_state_dict(checkpoint)
    print("✓ Loaded fine-tuned Gaussian decoder")

# If you also fine-tuned other decoders, load them too
# pipeline.models['slat_decoder_mesh'].load_state_dict(checkpoint)  # For mesh decoder
# pipeline.models['slat_decoder_rf'].load_state_dict(checkpoint)    # For radiance field decoder

# Now use the pipeline as normal
image = Image.open("/data/fmb/threedrealcar/preprocessed/merged/renders/0a2fc455054ec11c14821170410390fbc31337f26506cf717d4f9d5974e6da33/frame_00000.png")  # Use a car image for testing

print("Running inference with fine-tuned model...")
outputs = pipeline.run(
    image,
    seed=42,
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)

# Save outputs
print("Rendering outputs...")
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("car_finetuned_gs.mp4", video, fps=30)

# Save as PLY
outputs['gaussian'][0].save_ply("car_finetuned.ply")

print("Done! Check car_finetuned_gs.mp4 and car_finetuned.ply")