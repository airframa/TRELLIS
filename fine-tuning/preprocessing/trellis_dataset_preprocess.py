#!/usr/bin/env python3
"""
Direct Sketchfab download that GUARANTEES metadata matches files
Bypasses TRELLIS download.py which is broken
"""
import os
import sys
import pandas as pd
import objaverse.xl as oxl
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = "/data/fmb/trellis-dataset-subset"
NUM_SAMPLES = 2500
BATCH_SIZE = 1
NUM_VIEWS = 150
NUM_GPUS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = f"{OUTPUT_DIR}/preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("Direct Sketchfab Download - Guaranteed Matching")
logger.info("="*70)

# STEP 1: Get full annotations with ALL required fields
logger.info("\n[STEP 1/8] Loading Objaverse Annotations")
logger.info("="*70)

logger.info("Loading annotations (this includes fileIdentifier mapping)...")
annotations = oxl.get_annotations()
sketchfab = annotations[annotations['source'] == 'sketchfab'].copy()
logger.info(f"Total Sketchfab annotations: {len(sketchfab)}")
logger.info(f"Columns: {sketchfab.columns.tolist()}")

# Merge with TRELLIS aesthetic scores
logger.info("Loading TRELLIS quality scores...")
trellis_meta = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
logger.info(f"TRELLIS metadata: {len(trellis_meta)}")

# Merge on sha256
merged = sketchfab.merge(trellis_meta[['sha256', 'aesthetic_score', 'captions']], on='sha256', how='inner')
logger.info(f"Merged: {len(merged)} assets with quality scores")

# Filter high quality
high_quality = merged[merged['aesthetic_score'] >= 5.5].copy()
logger.info(f"High quality (>=5.5): {len(high_quality)}")

# Sample
selected = high_quality.sample(n=min(NUM_SAMPLES, len(high_quality)), random_state=42)
logger.info(f"Selected for download: {len(selected)}")

# Save initial metadata
metadata_file = Path(OUTPUT_DIR) / "metadata.csv"
selected.to_csv(metadata_file, index=False)
logger.info(f"✓ Saved metadata")

# STEP 2: Download with objaverse directly
logger.info("\n[STEP 2/8] Downloading with Objaverse")
logger.info("="*70)

download_dir = Path(OUTPUT_DIR) / "raw"
download_dir.mkdir(parents=True, exist_ok=True)

success_count = 0
failed_count = 0

num_batches = (len(selected) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(selected))
    batch = selected.iloc[start_idx:end_idx].copy()
    
    batch_num = batch_idx + 1
    logger.info(f"\n--- Batch {batch_num}/{num_batches} ---")
    logger.info(f"Downloading {len(batch)} assets")
    
    max_retries = 3
    batch_success = False
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt+1}/{max_retries}")
            
            # Download with objaverse
            file_paths = oxl.download_objects(
                objects=batch,
                download_dir=str(download_dir),
                save_repo_format="zip",
                processes=2  # Low to avoid SSL errors
            )
            
            logger.info(f"✓ Batch {batch_num} completed")
            logger.info(f"  Downloaded: {len(file_paths)} files")
            success_count += len(file_paths)
            batch_success = True
            break
            
        except Exception as e:
            logger.warning(f"✗ Attempt {attempt+1} failed: {str(e)[:200]}")
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 20
                logger.info(f"  Waiting {wait}s...")
                time.sleep(wait)
    
    if not batch_success:
        logger.error(f"✗ Batch {batch_num} completely failed")
        failed_count += len(batch)
    
    # Rate limit
    if batch_idx < num_batches - 1:
        time.sleep(2)

logger.info(f"\n✓ Download complete")
logger.info(f"  Success: {success_count}")
logger.info(f"  Failed: {failed_count}")

# STEP 3: Map downloaded files to metadata
logger.info("\n[STEP 3/8] Mapping Downloads to Metadata")
logger.info("="*70)

# Scan what was actually downloaded
raw_dir = download_dir / "hf-objaverse-v1" / "glbs"
downloaded_sha256s = set()

if raw_dir.exists():
    for folder in raw_dir.iterdir():
        if folder.is_dir():
            for glb in folder.glob("*.glb"):
                sha256_short = glb.stem  # 32-char hash
                downloaded_sha256s.add(sha256_short)

logger.info(f"Found {len(downloaded_sha256s)} GLB files")

# Map short SHA256 (32-char) to full SHA256 (64-char) using annotations
logger.info("Mapping short SHA to full SHA...")

sha_mapping = {}
for short_sha in downloaded_sha256s:
    # Check if short_sha is actually the full sha
    match = selected[selected['sha256'] == short_sha]
    if len(match) > 0:
        sha_mapping[short_sha] = short_sha
        continue
    
    # Try as fileIdentifier
    match = selected[selected['fileIdentifier'].str.contains(short_sha, na=False)]
    if len(match) > 0:
        sha_mapping[short_sha] = match.iloc[0]['sha256']
        continue
    
    # Try as prefix of sha256
    for _, row in selected.iterrows():
        if row['sha256'].startswith(short_sha):
            sha_mapping[short_sha] = row['sha256']
            break

logger.info(f"Mapped {len(sha_mapping)} files to original metadata")

# Update metadata with local_path
df_with_paths = selected.copy()
df_with_paths['local_path'] = None

for short_sha, full_sha in sha_mapping.items():
    # Find the file
    for folder in raw_dir.iterdir():
        if folder.is_dir():
            glb_file = folder / f"{short_sha}.glb"
            if glb_file.exists():
                rel_path = str(glb_file.relative_to(OUTPUT_DIR))
                df_with_paths.loc[df_with_paths['sha256'] == full_sha, 'local_path'] = rel_path
                break

df_with_paths.to_csv(metadata_file, index=False)
valid_count = df_with_paths['local_path'].notna().sum()
logger.info(f"✓ Updated metadata: {valid_count}/{len(df_with_paths)} have local_path")

if valid_count < 10:
    logger.error(f"Only {valid_count} valid. Too few to proceed.")
    sys.exit(1)

# Create instances file
instances_file = Path(OUTPUT_DIR) / "instances.txt"
df_valid = df_with_paths[df_with_paths['local_path'].notna()]
with open(instances_file, 'w') as f:
    for sha in df_valid['sha256']:
        f.write(f"{sha}\n")
logger.info(f"✓ Created instances.txt with {len(df_valid)} samples")

# STEP 4-8: Render, Voxelize, Extract (same as before)
logger.info(f"\n[STEP 4/8] Rendering {NUM_VIEWS} Views")
logger.info("="*70)

processes = []
for gpu in range(NUM_GPUS):
    cmd = f"""cd dataset_toolkits && \
CUDA_VISIBLE_DEVICES={gpu} python render.py ObjaverseXL \
    --source sketchfab \
    --output_dir {OUTPUT_DIR} \
    --instances {instances_file} \
    --num_views {NUM_VIEWS} \
    --rank {gpu} \
    --world_size {NUM_GPUS}"""
    
    proc = subprocess.Popen(cmd, shell=True)
    processes.append(proc)
    time.sleep(3)

for i, proc in enumerate(processes):
    proc.wait()
    logger.info(f"GPU {i} render done")

# Continue with voxelize and features...
logger.info(f"\n[STEP 5/8] Voxelizing")
processes = []
for gpu in range(NUM_GPUS):
    cmd = f"""cd dataset_toolkits && \
CUDA_VISIBLE_DEVICES={gpu} python voxelize.py ObjaverseXL \
    --source sketchfab \
    --output_dir {OUTPUT_DIR} \
    --instances {instances_file} \
    --rank {gpu} \
    --world_size {NUM_GPUS}"""
    proc = subprocess.Popen(cmd, shell=True)
    processes.append(proc)
    time.sleep(3)
for proc in processes:
    proc.wait()

logger.info(f"\n[STEP 6/8] Extracting Features")
processes = []
for gpu in range(NUM_GPUS):
    cmd = f"""cd dataset_toolkits && \
CUDA_VISIBLE_DEVICES={gpu} python extract_feature.py \
    --output_dir {OUTPUT_DIR} \
    --instances {instances_file} \
    --model dinov2_vitl14_reg \
    --rank {gpu} \
    --world_size {NUM_GPUS}"""
    proc = subprocess.Popen(cmd, shell=True)
    processes.append(proc)
    time.sleep(3)
for proc in processes:
    proc.wait()

# STEP 7: Use official build_metadata.py to consolidate everything
logger.info(f"\n[STEP 7/8] Consolidating Metadata (Official TRELLIS)")
logger.info("="*70)

cmd = f"""cd dataset_toolkits && \
python build_metadata.py ObjaverseXL \
    --source sketchfab \
    --output_dir {OUTPUT_DIR}"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
logger.info(result.stdout if result.stdout else "No output")

if result.returncode == 0:
    logger.info("✓ Metadata consolidated successfully")
else:
    logger.warning(f"Warning: {result.stderr[-300:]}")

# Read the statistics file that build_metadata.py creates
stats_file = Path(OUTPUT_DIR) / "statistics.txt"
if stats_file.exists():
    logger.info("\n--- Dataset Statistics ---")
    logger.info(stats_file.read_text())

# STEP 8: Final verification
logger.info(f"\n[STEP 8/8] Final Verification")
logger.info("="*70)

df_final = pd.read_csv(metadata_file)
logger.info(f"\nFinal metadata columns: {df_final.columns.tolist()}")
logger.info(f"Total samples: {len(df_final)}")
logger.info(f"With local_path: {df_final['local_path'].notna().sum()}")
logger.info(f"Rendered: {df_final['rendered'].sum() if 'rendered' in df_final.columns else 'N/A'}")
logger.info(f"Voxelized: {df_final['voxelized'].sum() if 'voxelized' in df_final.columns else 'N/A'}")

# Check for feature column (build_metadata creates feature_{model} columns)
feature_cols = [col for col in df_final.columns if col.startswith('feature_')]
if feature_cols:
    logger.info(f"Feature columns: {feature_cols}")
    for col in feature_cols:
        logger.info(f"  {col}: {df_final[col].sum()}")

logger.info("\n" + "="*70)
logger.info("✓ PREPROCESSING COMPLETE")
logger.info("="*70)
logger.info(f"Dataset: {OUTPUT_DIR}")
logger.info(f"Log: {LOG_FILE}")
logger.info("="*70)