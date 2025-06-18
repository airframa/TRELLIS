# inspect_3drealcar_structure.py
import json
from pathlib import Path
import numpy as np

def inspect_complete_structure(sample_dir):
    sample_path = Path(sample_dir)
    
    print("=== DIRECTORY STRUCTURE ===")
    # List all subdirectories
    for subdir in sample_path.iterdir():
        if subdir.is_dir():
            print(f"\n{subdir.name}/")
            # List some files in each subdirectory
            files = list(subdir.iterdir())[:5]
            for f in files:
                print(f"  - {f.name}")
            if len(list(subdir.iterdir())) > 5:
                print(f"  ... and {len(list(subdir.iterdir())) - 5} more files")
    
    print("\n=== FRAME FILES ANALYSIS ===")
    # Check where images actually are
    image_locations = {
        "root": list(sample_path.glob("frame_*.jpg")),
        "colmap_processed/images": list((sample_path / "colmap_processed" / "images").glob("frame_*.jpg")) if (sample_path / "colmap_processed" / "images").exists() else [],
    }
    
    for location, images in image_locations.items():
        print(f"{location}: {len(images)} images")
        if images:
            print(f"  First few: {[img.name for img in images[:3]]}")
    
    print("\n=== JSON FORMAT ANALYSIS ===")
    # Analyze JSON structure
    json_files = sorted(sample_path.glob("frame_*.json"))
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            
        print(f"JSON keys: {list(data.keys())}")
        print(f"\nIntrinsics shape: {len(data['intrinsics'])} elements")
        print(f"Intrinsics: {data['intrinsics']}")
        print(f"\ncameraPoseARFrame shape: {len(data['cameraPoseARFrame'])} elements")
        
        # Check if intrinsics is a 3x3 matrix flattened
        intrinsics_array = np.array(data['intrinsics'])
        if len(intrinsics_array) == 9:
            intrinsics_matrix = intrinsics_array.reshape(3, 3)
            print(f"\nIntrinsics as 3x3 matrix:")
            print(intrinsics_matrix)

if __name__ == "__main__":
    inspect_complete_structure("./assets/3drealcar/sample_data/2024_04_22_10_35_34")