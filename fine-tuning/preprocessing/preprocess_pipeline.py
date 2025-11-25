#!/usr/bin/env python3
"""
Main preprocessing pipeline for 3DRealCar dataset.
Orchestrates parallel batch processing across multiple GPUs.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Optional

# Configuration
BATCH_DIRS = [
    "0-200", "200-400", "400-600", "600-800", "800-1000",
    "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2045",
    "HQ200", "HQ339"
]

IGNORE_DIRS = ["logs", "3DRealCar_Segment_Dataset", "extraction_state.json"]

def get_available_gpus():
    """Get list of available GPU IDs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    return [int(x.strip()) for x in result.stdout.strip().split('\n')]

def get_batch_status(output_dir):
    """Check if a batch has been processed."""
    metadata_file = Path(output_dir) / "metadata.csv"
    if not metadata_file.exists():
        return "not_started"
    
    # Check if all steps completed
    import pandas as pd
    try:
        df = pd.read_csv(metadata_file)
        if 'feature_dinov2_vitl14_reg' in df.columns:
            if df['feature_dinov2_vitl14_reg'].all():
                return "completed"
            else:
                return "partial"
        return "partial"
    except:
        return "error"

def process_batch(batch_name: str, source_dir: Path, output_base_dir: Path, 
                  gpu_id: int, log_dir: Path, logger: logging.Logger) -> Optional[dict]:
    """Launch processing for a single batch."""
    output_dir = Path(output_base_dir) / batch_name
    log_file = Path(log_dir) / f"{batch_name}.log"
    
    # Check status
    status = get_batch_status(output_dir)
    if status == "completed":
        logger.info(f"[{batch_name}] Already completed, skipping")
        return None
    
    logger.info(f"[{batch_name}] Starting on GPU {gpu_id}")
    
    cmd = [
        sys.executable,
        "preprocess_single_batch.py",
        "--batch_name", batch_name,
        "--source_dir", str(source_dir),
        "--output_dir", str(output_dir),
        "--gpu_id", str(gpu_id),
        "--log_file", str(log_file)
    ]
    
    # Launch as subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    return {
        'batch': batch_name,
        'gpu': gpu_id,
        'process': process,
        'log_file': log_file,
        'start_time': time.time()
    }

def monitor_processes(active_jobs, logger):
    """Monitor and display status of active jobs."""
    while active_jobs:
        time.sleep(10)  # Check every 10 seconds
        
        completed = []
        for job in active_jobs:
            if job['process'].poll() is not None:
                # Process completed
                elapsed = time.time() - job['start_time']
                returncode = job['process'].returncode
                
                if returncode == 0:
                    msg = f"✓ [{job['batch']}] Completed successfully in {elapsed/60:.1f} minutes"
                    logger.info(msg)
                else:
                    msg = f"✗ [{job['batch']}] Failed with code {returncode}. Check {job['log_file']}"
                    logger.error(msg)
                
                completed.append(job)
        
        # Remove completed jobs
        for job in completed:
            active_jobs.remove(job)
        
        # Status update
        if active_jobs:
            status_msg = f"Active jobs: {len(active_jobs)}"
            logger.info(status_msg)
            for job in active_jobs:
                elapsed = time.time() - job['start_time']
                job_msg = f"  - {job['batch']:15s} on GPU {job['gpu']} ({elapsed/60:.1f} min)"
                logger.info(job_msg)

def setup_main_logging(output_base):
    """Setup main logging for the orchestrator."""
    log_dir = Path(output_base) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    main_log = log_dir / f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(main_log),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Main log file: {main_log}")
    return logger

def main():
    parser = argparse.ArgumentParser(description="Preprocess full 3DRealCar dataset")
    parser.add_argument("--source_base", required=True, 
                       help="Base directory containing batch folders (e.g., /data/fmb/threedrealcar/.../extracted)")
    parser.add_argument("--output_base", required=True,
                       help="Base output directory for processed data")
    parser.add_argument("--max_parallel", type=int, default=10,
                       help="Maximum number of parallel batch processes")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run (skip completed batches)")
    args = parser.parse_args()
    
    # Setup
    source_base = Path(args.source_base)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_base / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Setup main logging
    logger = setup_main_logging(output_base)
    
    # Validate paths
    if not source_base.exists():
        logger.error(f"Source directory does not exist: {source_base}")
        return 1
    
    logger.info(f"Source base: {source_base}")
    logger.info(f"Output base: {output_base}")
    logger.info(f"Max parallel: {args.max_parallel}")
    logger.info(f"Resume mode: {args.resume}\n")
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    logger.info(f"Found {len(available_gpus)} GPUs: {available_gpus}")
    
    # Find all batch directories
    batch_dirs = [d for d in os.listdir(source_base) 
                  if os.path.isdir(source_base / d) and d not in IGNORE_DIRS]
    batch_dirs.sort()
    
    logger.info(f"\nFound {len(batch_dirs)} batch directories:")
    for batch in batch_dirs:
        status = get_batch_status(output_base / batch)
        logger.info(f"  - {batch:15s} [{status}]")
    
    # Filter batches if resuming
    if args.resume:
        batch_dirs = [b for b in batch_dirs 
                     if get_batch_status(output_base / b) != "completed"]
        logger.info(f"\nResuming: {len(batch_dirs)} batches to process")
    
    if not batch_dirs:
        logger.info("\nAll batches completed!")
        return 0
    
    # Start processing
    logger.info(f"{'='*60}")
    logger.info(f"Starting parallel preprocessing")
    logger.info(f"Max parallel jobs: {args.max_parallel}")
    logger.info(f"Batches to process: {len(batch_dirs)}")
    logger.info(f"{'='*60}\n")
    
    start_time = time.time()
    active_jobs = []
    batch_queue = batch_dirs.copy()
    gpu_idx = 0
    
    # Track statistics
    completed_batches = []
    failed_batches = []
    
    # Main processing loop
    while batch_queue or active_jobs:
        # Start new jobs if slots available
        while len(active_jobs) < args.max_parallel and batch_queue:
            batch = batch_queue.pop(0)
            gpu_id = available_gpus[gpu_idx % len(available_gpus)]
            gpu_idx += 1
            
            job = process_batch(
                batch,
                source_base / batch,
                output_base,
                gpu_id,
                log_dir,
                logger
            )
            
            if job:  # None if already completed
                active_jobs.append(job)
                logger.info(f"Started [{job['batch']}] on GPU {gpu_id}")
                time.sleep(2)  # Brief delay between launches
        
        # Monitor progress and collect results
        if active_jobs:
            # Check for completed jobs
            for job in list(active_jobs):
                if job['process'].poll() is not None:
                    elapsed = time.time() - job['start_time']
                    if job['process'].returncode == 0:
                        completed_batches.append(job['batch'])
                        logger.info(f"✓ [{job['batch']}] Completed in {elapsed/60:.1f} min")
                    else:
                        failed_batches.append(job['batch'])
                        logger.error(f"✗ [{job['batch']}] Failed (code {job['process'].returncode})")
                    active_jobs.remove(job)
            
            # Log status every 10 seconds
            time.sleep(10)
            if active_jobs:
                logger.info(f"Progress: {len(completed_batches)} completed, {len(failed_batches)} failed, {len(active_jobs)} active, {len(batch_queue)} queued")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Preprocessing completed!")
    logger.info(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    logger.info(f"Successfully processed: {len(completed_batches)}/{len(batch_dirs)}")
    if failed_batches:
        logger.warning(f"Failed batches: {failed_batches}")
    logger.info(f"{'='*60}\n")
    
    # Merge datasets
    logger.info("Starting dataset merge...")
    merge_cmd = [
        sys.executable,
        "merge_datasets.py",
        "--input_base", str(output_base),
        "--output_dir", str(output_base / "merged")
    ]
    
    try:
        result = subprocess.run(merge_cmd, capture_output=True, text=True, check=True)
        logger.info("Merge completed successfully")
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Merge failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ ALL PROCESSING COMPLETE!")
    logger.info(f"Merged dataset location: {output_base / 'merged'}")
    main_log_pattern = f"main_{datetime.now().strftime('%Y%m%d')}_*.log"
    logger.info(f"Main log file: {log_dir / main_log_pattern}")
    logger.info(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    main()