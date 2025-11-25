#!/usr/bin/env python3
"""
Process a single batch of 3DRealCar data through the full pipeline.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_file):
    """Setup logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, logger, description):
    """Run a command and log output."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ Completed in {elapsed:.1f}s")
        
        if result.stdout:
            logger.debug(f"STDOUT:\n{result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"✗ Failed after {elapsed:.1f}s")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDERR:\n{e.stderr}")
        
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_name", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--log_file", required=True)
    args = parser.parse_args()
    
    # Setup logging FIRST (even before validation)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_file)
    
    try:
        source_dir = Path(args.source_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate source directory
        if not source_dir.exists():
            logger.error(f"Source directory does not exist: {source_dir}")
            return 1
        
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        
        logger.info(f"{'='*60}")
        logger.info(f"Processing batch: {args.batch_name}")
        logger.info(f"Source: {source_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"GPU: {args.gpu_id}")
        logger.info(f"{'='*60}\n")
        
        # Get TRELLIS root directory
        # Detect if we're in preprocessing/ or fine-tuning/preprocessing/
        script_path = Path(__file__).resolve()
        
        # Start from script directory and go up until we find TRELLIS root
        current = script_path.parent
        while current.name in ['preprocessing', 'fine-tuning']:
            current = current.parent
        
        trellis_root = current
        
        # Check if convert_format.py is in the same directory as this script (fine-tuning/preprocessing/)
        convert_format = script_path.parent / "convert_format.py"
        dataset_toolkits = trellis_root / "dataset_toolkits"
        
        # Validate paths
        if not convert_format.exists():
            logger.error(f"convert_format.py not found at: {convert_format}")
            logger.error(f"Script location: {script_path}")
            logger.error(f"Detected TRELLIS root: {trellis_root}")
            logger.error("Please ensure convert_format.py is in the same directory as this script")
            return 1
        
        if not dataset_toolkits.exists():
            logger.error(f"dataset_toolkits not found at: {dataset_toolkits}")
            logger.error(f"Detected TRELLIS root: {trellis_root}")
            return 1
        
        logger.info(f"TRELLIS root: {trellis_root}")
        logger.info(f"convert_format: {convert_format}")
        logger.info(f"dataset_toolkits: {dataset_toolkits}\n")
        
    except Exception as e:
        if 'logger' in locals():
            logger.exception(f"Error during setup: {e}")
        else:
            print(f"CRITICAL ERROR during setup: {e}", file=sys.stderr)
        return 1
    
    # Step 1: Format Conversion
    logger.info("\n[STEP 1/4] Format Conversion")
    cmd = [
        sys.executable,
        str(convert_format),
        "--source_dir", str(source_dir),
        "--output_dir", str(output_dir)
    ]
    if not run_command(cmd, logger, "Format conversion"):
        logger.error("Failed at Step 1: Format Conversion")
        return 1
    
    # Check if instances.txt was created
    instances_file = output_dir / "instances.txt"
    if not instances_file.exists():
        logger.error(f"instances.txt not created at {instances_file}")
        return 1
    
    # Count samples
    with open(instances_file) as f:
        num_samples = len(f.readlines())
    logger.info(f"Converted {num_samples} samples")
    
    # Step 2: Voxelization
    logger.info("\n[STEP 2/4] Voxelization")
    cmd = [
        sys.executable,
        str(dataset_toolkits / "voxelize.py"),
        "3DRealCar",
        "--output_dir", str(output_dir),
        "--instances", str(instances_file)
    ]
    if not run_command(cmd, logger, "Voxelization"):
        logger.error("Failed at Step 2: Voxelization")
        return 1
    
    # Step 3: Feature Extraction
    logger.info("\n[STEP 3/4] DINOv2 Feature Extraction")
    cmd = [
        sys.executable,
        str(dataset_toolkits / "extract_feature.py"),
        "--output_dir", str(output_dir),
        "--instances", str(instances_file),
        "--model", "dinov2_vitl14_reg"
    ]
    if not run_command(cmd, logger, "Feature extraction"):
        logger.error("Failed at Step 3: Feature Extraction")
        return 1
    
    # Step 4: Build Metadata
    logger.info("\n[STEP 4/4] Building Final Metadata")
    cmd = [
        sys.executable,
        str(dataset_toolkits / "build_metadata.py"),
        "3DRealCar",
        "--output_dir", str(output_dir)
    ]
    if not run_command(cmd, logger, "Build metadata"):
        logger.error("Failed at Step 4: Build Metadata")
        return 1
    
    # Success!
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Batch {args.batch_name} completed successfully!")
    logger.info(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())