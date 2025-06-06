import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
import argparse

# Add the parent directory to the Python path to find trellis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preprocessing module
from preprocess_3drealcar import preprocess_3drealcar, RealCar3DProcessor

# Now import required TRELLIS components
from trellis.datasets import SparseStructure, SparseFeat2Render, SLat2Render


def validate_trellis_format(source_dir, output_dir):
    """
    Validate that 3DRealCar data was correctly converted to TRELLIS format.
    
    Args:
        source_dir: Source directory containing 3DRealCar data
        output_dir: Output directory where TRELLIS format data was saved
    """
    print(f"Validating conversion from 3DRealCar to TRELLIS format...")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Check if the preprocessing completed successfully
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.csv"
    
    if not metadata_path.exists():
        print(f"❌ Metadata file not found at {metadata_path}. Preprocessing may have failed.")
        return False
    
    # Step 2: Inspect metadata file
    import pandas as pd
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"✓ Successfully loaded metadata with {len(metadata)} entries")
        print(f"  Metadata columns: {', '.join(metadata.columns)}")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False
    
    if len(metadata) == 0:
        print(f"❌ Metadata is empty. No cars were processed successfully.")
        return False
    
    # Step 3: Check for a sample car ID
    sample_car_id = metadata['sha256'].iloc[0]
    print(f"Using sample car ID: {sample_car_id}")
    
    # Step 4: Check expected TRELLIS directory structure (FIXED)
    required_dirs = [
        output_path / "renders" / sample_car_id,
        output_path / "voxels",
        output_path / "features" / "dinov2_vitl14_reg",
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"❌ Required directory not found: {dir_path}")
            return False
        else:
            print(f"✓ Found directory: {dir_path}")
    
    # Step 5: Check for transforms.json (FIXED path)
    transforms_path = output_path / "renders" / sample_car_id / "transforms.json"
    if not transforms_path.exists():
        print(f"❌ transforms.json not found at {transforms_path}")
        return False
    
    try:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        print(f"✓ Successfully loaded transforms.json with {len(transforms['frames'])} frames")
    except Exception as e:
        print(f"❌ Failed to load transforms.json: {e}")
        return False
    
    # Step 6: Check for voxel data (FIXED path)
    voxel_ply_path = output_path / "voxels" / f"{sample_car_id}.ply"
    if not voxel_ply_path.exists():
        print(f"❌ Voxel PLY file not found at {voxel_ply_path}")
        return False
    else:
        print(f"✓ Found voxel PLY file: {voxel_ply_path}")
    
    # Step 7: Check for rendered images (FIXED path)
    first_frame = transforms['frames'][0]['file_path']
    image_path = output_path / "renders" / sample_car_id / first_frame.replace("./", "")
    
    if not image_path.exists():
        print(f"❌ Rendered image not found at {image_path}")
        return False
    else:
        print(f"✓ Found rendered image: {image_path}")
    
    # Step 8: Check mesh.ply (FIXED path)
    mesh_path = output_path / "renders" / sample_car_id / "mesh.ply"
    if not mesh_path.exists():
        print(f"❌ Mesh file not found at {mesh_path}")
        return False
    else:
        print(f"✓ Found mesh file: {mesh_path}")
    
    # Step 9: Check for DINOv2 features (FIXED path)
    features_path = output_path / "features" / "dinov2_vitl14_reg" / f"{sample_car_id}.npz"
    if not features_path.exists():
        print(f"❌ DINOv2 features not found at {features_path}")
    else:
        print(f"✓ Found DINOv2 features: {features_path}")
    
    # Step 10: Try loading the data with TRELLIS dataloaders
    try:
        # Test SparseStructure dataloader
        print("\nTesting TRELLIS dataloaders...")
        print("1. Testing SparseStructure dataloader")
        dataset_ss = SparseStructure(
            roots=str(output_dir),
            resolution=64,
            min_aesthetic_score=0.0  # Set to 0 to include all processed cars
        )
        print(f"✓ Successfully loaded SparseStructure dataset with {len(dataset_ss)} instances")
        print(f"  Dataset info:\n{dataset_ss}")
        
        # Try loading a sample
        try:
            sample = dataset_ss[0]
            print(f"✓ Successfully loaded a sample from SparseStructure dataset")
            print(f"  Sample contains keys: {list(sample.keys())}")
            print(f"  Sample SS shape: {sample['ss'].shape}")
        except Exception as e:
            print(f"❌ Failed to load a sample from SparseStructure dataset: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"❌ Failed to initialize SparseStructure dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test SparseFeat2Render dataloader if DinoV2 features were extracted
    try:
        print("\n2. Testing SparseFeat2Render dataloader")
        
        if features_path.exists():
            dataset_feat = SparseFeat2Render(
                roots=str(output_dir),
                image_size=224,
                model='dinov2_vitl14_reg',
                resolution=64,
                min_aesthetic_score=0.0
            )
            print(f"✓ Successfully loaded SparseFeat2Render dataset with {len(dataset_feat)} instances")
            
            # Try loading a sample
            try:
                sample = dataset_feat[0]
                print(f"✓ Successfully loaded a sample from SparseFeat2Render dataset")
                print(f"  Sample contains keys: {list(sample.keys())}")
                print(f"  Sample feature shape: {sample['feats'].shape}")
                print(f"  Sample coords shape: {sample['coords'].shape}")
            except Exception as e:
                print(f"❌ Failed to load a sample from SparseFeat2Render dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("ℹ️ Skipping SparseFeat2Render test - DinoV2 features not found")
    except Exception as e:
        print(f"❌ Failed to initialize SparseFeat2Render dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test SLat2Render dataloader if latents were extracted
    try:
        print("\n3. Testing SLat2Render dataloader")
        latents_path = output_path / "latents"
        
        if latents_path.exists() and any(latents_path.glob("*/*.npz")):
            latent_model = os.listdir(latents_path)[0]
            dataset_slat = SLat2Render(
                roots=str(output_dir),
                image_size=224,
                latent_model=latent_model,
                min_aesthetic_score=0.0
            )
            print(f"✓ Successfully loaded SLat2Render dataset with {len(dataset_slat)} instances")
            
            # Try loading a sample
            try:
                sample = dataset_slat[0]
                print(f"✓ Successfully loaded a sample from SLat2Render dataset")
                print(f"  Sample contains keys: {list(sample.keys())}")
            except Exception as e:
                print(f"❌ Failed to load a sample from SLat2Render dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("ℹ️ Skipping SLat2Render test - latent features not found")
    except Exception as e:
        print(f"❌ Failed to initialize SLat2Render dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Validation complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate 3DRealCar to TRELLIS format conversion")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing 3DRealCar data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory where TRELLIS format data was saved")
    parser.add_argument("--process", action="store_true", help="Run preprocessing before validation")
    
    args = parser.parse_args()
    
    if args.process:
        print("Running preprocessing...")
        preprocess_3drealcar(args.source_dir, args.output_dir)
    
    validate_trellis_format(args.source_dir, args.output_dir)