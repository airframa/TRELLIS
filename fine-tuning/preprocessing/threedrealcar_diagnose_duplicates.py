#!/usr/bin/env python3
"""
Diagnose duplicate issues in the dataset and create clean splits
"""
import os
import pandas as pd
from pathlib import Path
import json
from collections import Counter

BASE_DIR = Path("/data/francMB/threedrealcar")
OUTPUT_DIR = Path("/data/francMB/threedrealcar/splits")

FOLDERS = [
    "0-200", "200-400", "400-600", "600-800", "800-1000",
    "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2045",
    "HQ200", "HQ300-1", "HQ300-2"
]

def diagnose_source_duplicates():
    """Check for duplicates in source metadata files"""
    print("Diagnosing duplicates in source data...")
    print("=" * 50)
    
    all_sha256s = []
    folder_duplicates = {}
    
    for folder in FOLDERS:
        folder_path = BASE_DIR / folder
        metadata_path = folder_path / "metadata.csv"
        
        if not metadata_path.exists():
            continue
            
        df = pd.read_csv(metadata_path)
        
        # Check for duplicates within this folder
        duplicates = df[df.duplicated(subset=['sha256'], keep=False)]
        if len(duplicates) > 0:
            print(f"\n{folder}: Found {len(duplicates)} duplicate entries")
            folder_duplicates[folder] = duplicates['sha256'].unique().tolist()
            
            # Show first few duplicates
            for sha in duplicates['sha256'].unique()[:3]:
                dup_rows = df[df['sha256'] == sha]
                print(f"  - {sha}: appears {len(dup_rows)} times")
        else:
            print(f"\n{folder}: No duplicates")
        
        # Collect all sha256s
        all_sha256s.extend(df['sha256'].tolist())
    
    # Check for cross-folder duplicates
    sha256_counts = Counter(all_sha256s)
    cross_folder_dups = {sha: count for sha, count in sha256_counts.items() if count > 1}
    
    print(f"\n\nCross-folder duplicates: {len(cross_folder_dups)}")
    if cross_folder_dups:
        # Find which folders contain each duplicate
        for sha, count in list(cross_folder_dups.items())[:5]:  # Show first 5
            folders_with_sha = []
            for folder in FOLDERS:
                metadata_path = BASE_DIR / folder / "metadata.csv"
                if metadata_path.exists():
                    df = pd.read_csv(metadata_path)
                    if sha in df['sha256'].values:
                        folders_with_sha.append(folder)
            print(f"  - {sha}: appears in {folders_with_sha}")
    
    return folder_duplicates, cross_folder_dups

def check_split_overlaps():
    """Check which samples appear in multiple splits"""
    print("\n\nChecking split overlaps...")
    print("=" * 50)
    
    # Load split info
    with open(OUTPUT_DIR / 'split_info.json', 'r') as f:
        split_info = json.load(f)
    
    train_set = set(split_info['splits']['train']['samples'])
    val_set = set(split_info['splits']['val']['samples'])
    test_set = set(split_info['splits']['test']['samples'])
    
    # Check overlaps
    train_val = train_set.intersection(val_set)
    train_test = train_set.intersection(test_set)
    val_test = val_set.intersection(test_set)
    all_three = train_set.intersection(val_set).intersection(test_set)
    
    print(f"Train ∩ Val: {len(train_val)} samples")
    print(f"Train ∩ Test: {len(train_test)} samples")
    print(f"Val ∩ Test: {len(val_test)} samples")
    print(f"All three: {len(all_three)} samples")
    
    if train_val:
        print(f"\nSamples in both Train and Val:")
        for sha in list(train_val)[:5]:
            print(f"  - {sha}")
    
    return train_val, train_test, val_test

def create_clean_splits():
    """Create clean splits without duplicates"""
    print("\n\nCreating clean dataset...")
    print("=" * 50)
    
    # Collect all unique samples
    all_samples_dict = {}  # sha256 -> sample info
    
    for folder in FOLDERS:
        folder_path = BASE_DIR / folder
        metadata_path = folder_path / "metadata.csv"
        
        if not metadata_path.exists():
            continue
            
        df = pd.read_csv(metadata_path)
        
        # For each unique sha256, keep the first occurrence
        for _, row in df.iterrows():
            sha256 = row['sha256']
            if sha256 not in all_samples_dict:
                # Check if all required files exist
                features_exist = (folder_path / 'features' / 'dinov2_vitl14_reg' / f"{sha256}.npz").exists()
                latents_exist = (folder_path / 'latents' / 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16' / f"{sha256}.npz").exists()
                voxels_exist = (folder_path / 'voxels' / f"{sha256}.ply").exists()
                renders_exist = (folder_path / 'renders' / sha256).exists()
                
                if features_exist and latents_exist and voxels_exist and renders_exist:
                    all_samples_dict[sha256] = {
                        'sha256': sha256,
                        'folder': folder,
                        'aesthetic_score': row['aesthetic_score'],
                        'num_voxels': row['num_voxels'],
                        'num_views': row['num_views']
                    }
    
    print(f"Found {len(all_samples_dict)} unique samples with all files present")
    
    # Save clean dataset info
    clean_samples = list(all_samples_dict.values())
    clean_df = pd.DataFrame(clean_samples)
    
    # Show statistics
    print(f"\nClean dataset statistics:")
    print(f"  Total samples: {len(clean_samples)}")
    print(f"  Aesthetic score range: {clean_df['aesthetic_score'].min():.2f} - {clean_df['aesthetic_score'].max():.2f}")
    print(f"  Mean aesthetic score: {clean_df['aesthetic_score'].mean():.2f}")
    print(f"  Samples per folder:")
    for folder, count in clean_df['folder'].value_counts().items():
        print(f"    {folder}: {count}")
    
    return clean_samples

def main():
    # Diagnose issues
    folder_dups, cross_dups = diagnose_source_duplicates()
    train_val, train_test, val_test = check_split_overlaps()
    
    # Create clean dataset
    clean_samples = create_clean_splits()
    
    print("\n\nRecommendation:")
    print("=" * 50)
    print("Re-run the split creation script with the --clean flag to create splits")
    print("from the deduplicated dataset. This will ensure:")
    print("  1. No duplicates within splits")
    print("  2. No overlaps between splits")
    print("  3. All samples have all required files")
    
    # Save clean samples for use by the split script
    with open(OUTPUT_DIR / 'clean_samples.json', 'w') as f:
        json.dump(clean_samples, f, indent=2)
    print(f"\nSaved clean samples to {OUTPUT_DIR / 'clean_samples.json'}")

if __name__ == "__main__":
    main()