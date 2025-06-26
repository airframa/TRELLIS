#!/usr/bin/env python3
"""
Flexible Batch Processing Pipeline for 3DRealCar
Processes selected batches with optimal GPU utilization and space management
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import signal
import time
import pandas as pd
import psutil
import shutil
from typing import List, Dict, Tuple

class MultiHandler:
    """Custom logging handler that writes to multiple files"""
    def __init__(self, base_log_dir: Path):
        self.base_log_dir = base_log_dir
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        self.main_log = base_log_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Setup handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all log handlers"""
        # Main file handler (everything)
        main_handler = logging.FileHandler(self.main_log)
        main_handler.setFormatter(self.detailed_formatter)
        main_handler.setLevel(logging.DEBUG)
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.simple_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Progress log (just progress updates)
        progress_handler = logging.FileHandler(self.base_log_dir / 'progress.log')
        progress_handler.setFormatter(self.simple_formatter)
        progress_handler.setLevel(logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[main_handler, console_handler, progress_handler]
        )
        
        # Also create batch-specific log directory
        self.batch_log_dir = self.base_log_dir / 'batch_logs'
        self.batch_log_dir.mkdir(exist_ok=True)
    
    def get_batch_logger(self, batch_name: str):
        """Get a logger for a specific batch"""
        batch_logger = logging.getLogger(f'batch.{batch_name}')
        batch_handler = logging.FileHandler(self.batch_log_dir / f'{batch_name}.log')
        batch_handler.setFormatter(self.detailed_formatter)
        batch_logger.addHandler(batch_handler)
        batch_logger.setLevel(logging.DEBUG)
        return batch_logger


def get_gpu_info():
    """Get GPU information and availability"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.free,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            gpus.append({
                'id': int(parts[0]),
                'name': parts[1],
                'memory_free': int(parts[2]),
                'memory_total': int(parts[3])
            })
        return gpus
    except:
        return []


def get_disk_usage(path: Path) -> Dict:
    """Get disk usage statistics"""
    usage = shutil.disk_usage(path)
    return {
        'total_gb': usage.total / 1e9,
        'used_gb': usage.used / 1e9,
        'free_gb': usage.free / 1e9,
        'percent': (usage.used / usage.total) * 100
    }


def estimate_output_size(extract_path: Path) -> float:
    """Estimate output size based on extracted data"""
    # Rough estimate: output is ~1.5x the extracted size
    total_size = 0
    for item in extract_path.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    return (total_size / 1e9) * 1.5


def process_single_batch(args) -> Dict:
    """Process a single batch with dedicated logging"""
    batch_name, source_dir, output_dir, script_path, gpu_list, batch_logger = args
    
    start_time = time.time()
    
    # Set GPU visibility
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        '--source_dir', str(source_dir),
        '--output_dir', str(output_dir)
    ]
    
    batch_logger.info(f"Starting processing of {batch_name}")
    batch_logger.info(f"Source: {source_dir}")
    batch_logger.info(f"Output: {output_dir}")
    batch_logger.info(f"GPUs: {gpu_list}")
    batch_logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run preprocessing with real-time output capture
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Capture output in real-time
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            batch_logger.debug(line)
            
            # Look for progress indicator
            # if "Converting to TRELLIS format" in line:
            #     batch_logger.info("Starting TRELLIS conversion...")
            # elif "Extracting DINOv2 features" in line:
            #     batch_logger.info("Extracting DINOv2 features...")
            # elif "Encoding latent features" in line:
            #     batch_logger.info("Encoding latent features...")
            if "Validation complete" in line:
                batch_logger.info("Validation complete!")
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            # Analyze results
            metadata_path = output_dir / 'metadata.csv'
            stats = {'processing_time_min': elapsed / 60}
            
            if metadata_path.exists():
                try:
                    df = pd.read_csv(metadata_path)
                    
                    # Convert pandas types to native Python types for JSON serialization
                    stats.update({
                        'total_cars': int(len(df)),
                        'with_features': int(df.get('feature_dinov2_vitl14_reg', pd.Series()).sum()),
                        'with_latents': int(df.get('latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16', pd.Series()).sum()),
                        'avg_aesthetic_score': float(df.get('aesthetic_score', pd.Series()).mean()) if not df.get('aesthetic_score', pd.Series()).isna().all() else 0.0
                    })
                    
                    # Calculate output size
                    output_size = 0
                    for item in output_dir.rglob('*'):
                        if item.is_file():
                            output_size += item.stat().st_size
                    stats['output_size_gb'] = float(output_size / 1e9)
                    
                except Exception as e:
                    batch_logger.warning(f"Error reading metadata for stats: {e}")
                    # Continue without detailed stats
            
            batch_logger.info(f"SUCCESS! Processed in {elapsed/60:.1f} minutes")
            batch_logger.info(f"Stats: {json.dumps(stats, indent=2)}")
            
            return {
                'batch': batch_name,
                'status': 'success',
                'stats': stats,
                'message': f"Processed in {elapsed/60:.1f} minutes"
            }
        else:
            batch_logger.error(f"FAILED! Exit code: {process.returncode}")
            batch_logger.error(f"Last output lines: {output_lines[-50:]}")
            
            return {
                'batch': batch_name,
                'status': 'failed',
                'error': '\n'.join(output_lines[-100:]),
                'message': f"Failed after {elapsed/60:.1f} minutes"
            }
            
    except Exception as e:
        batch_logger.error(f"Exception: {str(e)}")
        return {
            'batch': batch_name,
            'status': 'error',
            'message': str(e)
        }


class FlexibleBatchProcessor:
    """Flexible batch processor with optimal GPU utilization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.extract_dir = Path(config['extract_dir'])
        self.output_dir = Path(config['output_dir'])
        self.script_path = Path(config['script_path'])
        
        # Setup logging
        self.log_handler = MultiHandler(Path(config['log_dir']))
        self.logger = logging.getLogger('main')
        
        # State management
        self.state_file = Path(config['log_dir']) / 'processing_state.json'
        self.state = self._load_state()
        
        # GPU allocation strategy
        self.gpus = get_gpu_info()
        self.logger.info(f"Found {len(self.gpus)} GPUs: {[g['name'] for g in self.gpus]}")
        
        # Signal handling
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_state(self) -> Dict:
        """Load processing state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'processed': [],
            'failed': [],
            'processing': [],
            'start_time': datetime.now().isoformat(),
            'stats': {}
        }
    
    def _save_state(self):
        """Save processing state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.warning("Received shutdown signal! Waiting for current batches to complete...")
        self.running = False
    
    def allocate_gpus_to_batches(self, batches: List[str]) -> List[Tuple[str, List[int]]]:
        """Intelligently allocate GPUs to batches"""
        num_batches = len(batches)
        num_gpus = len(self.gpus)
        
        if num_batches == 0:
            return []
        
        # Strategy: Distribute GPUs as evenly as possible
        allocations = []
        
        if num_batches <= num_gpus:
            # More GPUs than batches: give each batch multiple GPUs
            gpus_per_batch = num_gpus // num_batches
            extra_gpus = num_gpus % num_batches
            
            gpu_idx = 0
            for i, batch in enumerate(batches):
                num_gpus_for_batch = gpus_per_batch + (1 if i < extra_gpus else 0)
                batch_gpus = []
                for _ in range(num_gpus_for_batch):
                    batch_gpus.append(self.gpus[gpu_idx]['id'])
                    gpu_idx += 1
                allocations.append((batch, batch_gpus))
        else:
            # More batches than GPUs: some batches share GPUs (process sequentially)
            for i, batch in enumerate(batches):
                gpu_id = self.gpus[i % num_gpus]['id']
                allocations.append((batch, [gpu_id]))
        
        return allocations
    
    def check_space_requirements(self, batches: List[str]) -> bool:
        """Check if there's enough space for processing"""
        disk_info = get_disk_usage(self.output_dir)
        
        self.logger.info(f"Disk space: {disk_info['free_gb']:.1f}GB free ({disk_info['percent']:.1f}% used)")
        
        # Estimate space needed
        total_needed = 0
        for batch in batches:
            extract_path = self.extract_dir / batch
            if extract_path.exists():
                estimated = estimate_output_size(extract_path)
                total_needed += estimated
                self.logger.info(f"  {batch}: ~{estimated:.1f}GB estimated output")
        
        # Add 20% safety margin
        total_needed *= 1.2
        
        self.logger.info(f"Total space needed: ~{total_needed:.1f}GB (with safety margin)")
        
        if total_needed > disk_info['free_gb']:
            self.logger.error(f"Insufficient space! Need {total_needed:.1f}GB, have {disk_info['free_gb']:.1f}GB")
            return False
        
        return True
    
    def process_batches(self, selected_batches: List[str]):
        """Process selected batches with optimal parallelization"""
        # Filter out already processed batches
        batches_to_process = [b for b in selected_batches 
                             if b not in self.state['processed'] and b not in self.state['failed']]
        
        if not batches_to_process:
            self.logger.info("No batches to process (all already completed)")
            return
        
        self.logger.info(f"Processing batches: {batches_to_process}")
        
        # Check space
        if not self.check_space_requirements(batches_to_process):
            self.logger.error("Aborting due to insufficient disk space")
            return
        
        # Allocate GPUs
        gpu_allocations = self.allocate_gpus_to_batches(batches_to_process)
        
        self.logger.info("\nGPU Allocations:")
        for batch, gpus in gpu_allocations:
            self.logger.info(f"  {batch}: GPUs {gpus}")
        
        # Process batches
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting batch processing...")
        self.logger.info("="*60)
        
        # Mark batches as processing
        for batch in batches_to_process:
            if batch not in self.state['processing']:
                self.state['processing'].append(batch)
        self._save_state()
        
        # Prepare tasks
        tasks = []
        for batch, gpu_list in gpu_allocations:
            source_dir = self.extract_dir / batch
            output_dir = self.output_dir / batch
            batch_logger = self.log_handler.get_batch_logger(batch)
            
            tasks.append((
                batch,
                source_dir,
                output_dir,
                self.script_path,
                gpu_list,
                batch_logger
            ))
        
        # Process in parallel
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=min(len(tasks), self.config.get('max_parallel', 4))) as executor:
            future_to_batch = {
                executor.submit(process_single_batch, task): task[0]
                for task in tasks
            }
            
            for future in as_completed(future_to_batch):
                if not self.running:
                    self.logger.warning("Shutdown requested, waiting for running batches...")
                    executor.shutdown(wait=True)
                    break
                
                result = future.result()
                results.append(result)
                batch_name = result['batch']
                
                # Update state
                self.state['processing'].remove(batch_name)
                
                if result['status'] == 'success':
                    self.logger.info(f"✅ {batch_name}: {result['message']}")
                    if 'stats' in result:
                        self.logger.info(f"   Stats: {json.dumps(result['stats'], indent=2)}")
                    
                    self.state['processed'].append(batch_name)
                    self.state['stats'][batch_name] = result.get('stats', {})
                else:
                    self.logger.error(f"❌ {batch_name}: {result['message']}")
                    self.state['failed'].append(batch_name)
                
                self._save_state()
                
                # Progress update
                completed = len(self.state['processed']) + len(self.state['failed'])
                total = len(selected_batches)
                self.logger.info(f"\nProgress: {completed}/{total} batches completed")
        
        # Summary
        elapsed = time.time() - start_time
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Batch processing completed in {elapsed/60:.1f} minutes")
        self.logger.info("="*60)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print processing summary"""
        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"  Processed: {len(self.state['processed'])} batches")
        self.logger.info(f"  Failed: {len(self.state['failed'])} batches")
        
        if self.state['processed']:
            self.logger.info("\nSuccessfully processed:")
            total_cars = 0
            total_size = 0
            
            for batch in self.state['processed']:
                stats = self.state['stats'].get(batch, {})
                self.logger.info(f"  {batch}: {stats.get('total_cars', 0)} cars, "
                               f"{stats.get('output_size_gb', 0):.1f}GB")
                total_cars += stats.get('total_cars', 0)
                total_size += stats.get('output_size_gb', 0)
            
            self.logger.info(f"\nTotal: {total_cars} cars, {total_size:.1f}GB output")
        
        if self.state['failed']:
            self.logger.info(f"\nFailed batches: {self.state['failed']}")
        
        # Disk usage after processing
        disk_info = get_disk_usage(self.output_dir)
        self.logger.info(f"\nDisk usage after processing: {disk_info['used_gb']:.1f}GB used, "
                        f"{disk_info['free_gb']:.1f}GB free ({disk_info['percent']:.1f}% full)")
    
    def cleanup_extracted(self, batches: List[str]):
        """Clean up extracted directories for completed batches"""
        self.logger.info("\nCleanup phase...")
        
        cleaned_size = 0
        for batch in batches:
            if batch in self.state['processed']:
                extract_path = self.extract_dir / batch
                if extract_path.exists():
                    size = sum(f.stat().st_size for f in extract_path.rglob('*') if f.is_file()) / 1e9
                    self.logger.info(f"  Removing {batch} ({size:.1f}GB)...")
                    shutil.rmtree(extract_path)
                    cleaned_size += size
        
        if cleaned_size > 0:
            self.logger.info(f"✅ Freed {cleaned_size:.1f}GB of disk space")


def main():
    parser = argparse.ArgumentParser(description="Flexible Batch Processing Pipeline")
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration JSON file')
    parser.add_argument('--batches', type=str, required=True,
                        help='Comma-separated list of batches to process')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up extracted directories after processing')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Parse batch list
    selected_batches = [b.strip() for b in args.batches.split(',')]
    
    # Create processor
    processor = FlexibleBatchProcessor(config)
    
    # Process batches
    processor.process_batches(selected_batches)
    
    # Optional cleanup
    if args.cleanup:
        processor.cleanup_extracted(selected_batches)


if __name__ == "__main__":
    main()