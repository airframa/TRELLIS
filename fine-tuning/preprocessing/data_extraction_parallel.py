#!/usr/bin/env python3
"""
Parallel ZIP Extraction Script for 3DRealCar Dataset
Extracts all zip files in parallel with progress tracking
"""

import os
import sys
import json
import zipfile
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import signal
import time

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_directory_size(path: Path) -> float:
    """Get directory size in GB"""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / 1e9


def check_disk_space(path: Path, required_gb: float = 100) -> tuple:
    """Check available disk space"""
    stat = shutil.disk_usage(path)
    available_gb = stat.free / 1e9
    return available_gb > required_gb, available_gb


def extract_single_zip(args):
    """Extract a single zip file (for parallel processing)"""
    zip_path, extract_base_dir, batch_name = args
    
    extract_path = extract_base_dir / batch_name
    
    # Skip if already extracted
    if extract_path.exists():
        car_count = len([d for d in extract_path.iterdir() if d.is_dir()])
        if car_count > 0:
            return {
                'batch': batch_name,
                'status': 'skipped',
                'message': f'Already extracted with {car_count} cars',
                'car_count': car_count
            }
    
    try:
        # Create temporary extraction path
        temp_path = extract_path.with_suffix('.tmp')
        temp_path.mkdir(parents=True, exist_ok=True)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total_files = len(zf.namelist())
            zf.extractall(temp_path)
        
        # Move to final location
        if extract_path.exists():
            shutil.rmtree(extract_path)
        temp_path.rename(extract_path)
        
        # Count extracted cars
        car_count = len([d for d in extract_path.iterdir() if d.is_dir()])
        
        return {
            'batch': batch_name,
            'status': 'success',
            'message': f'Extracted {total_files} files, {car_count} cars',
            'car_count': car_count,
            'size_gb': zip_path.stat().st_size / 1e9
        }
        
    except Exception as e:
        # Clean up on failure
        if temp_path.exists():
            shutil.rmtree(temp_path)
        
        return {
            'batch': batch_name,
            'status': 'failed',
            'message': str(e),
            'car_count': 0
        }


class ParallelExtractor:
    """Manages parallel extraction of all zip files"""
    
    def __init__(self, download_dir: Path, extract_dir: Path, max_workers: int = None):
        self.download_dir = download_dir
        self.extract_dir = extract_dir
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers or min(cpu_count() // 2, 8)  # Use half CPUs, max 8
        self.logger = setup_logging(self.extract_dir / 'logs')
        
        self.state_file = self.extract_dir / 'extraction_state.json'
        self.state = self._load_state()
        
        # Signal handling for graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_state(self):
        """Load extraction state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'extracted': [],
            'failed': [],
            'start_time': datetime.now().isoformat()
        }
    
    def _save_state(self):
        """Save extraction state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal, waiting for current extractions to complete...")
        self.running = False
    
    def get_zip_files(self):
        """Get all zip files to process"""
        zip_files = []
        
        for zip_path in sorted(self.download_dir.glob("*.zip")):
            batch_name = zip_path.stem
            
            # Skip if already extracted
            if batch_name in self.state['extracted']:
                continue
            
            # Verify it's a valid zip
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    _ = zf.namelist()[:1]  # Quick check
                zip_files.append((zip_path, batch_name))
            except:
                self.logger.warning(f"Invalid zip file: {zip_path}")
                continue
        
        return zip_files
    
    def extract_all(self):
        """Extract all zip files in parallel"""
        zip_files = self.get_zip_files()
        
        if not zip_files:
            self.logger.info("No zip files to extract")
            return
        
        self.logger.info(f"Found {len(zip_files)} zip files to extract")
        self.logger.info(f"Using {self.max_workers} parallel workers")
        
        # Check total disk space needed (rough estimate: 1.1x zip size)
        total_zip_size = sum(zp[0].stat().st_size for zp in zip_files) / 1e9
        required_space = total_zip_size * 1.1
        
        has_space, available_gb = check_disk_space(self.extract_dir, required_space)
        
        if not has_space:
            self.logger.error(f"Insufficient disk space! Need ~{required_space:.1f}GB, have {available_gb:.1f}GB")
            return
        
        self.logger.info(f"Disk space check: Need ~{required_space:.1f}GB, have {available_gb:.1f}GB")
        
        # Prepare extraction tasks
        tasks = [
            (zip_path, self.extract_dir, batch_name)
            for zip_path, batch_name in zip_files
        ]
        
        # Extract in parallel
        total_cars = 0
        successful = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(extract_single_zip, task): task 
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                if not self.running:
                    self.logger.info("Shutting down...")
                    executor.shutdown(wait=False)
                    break
                
                result = future.result()
                
                if result['status'] == 'success':
                    self.logger.info(f"✅ {result['batch']}: {result['message']}")
                    self.state['extracted'].append(result['batch'])
                    total_cars += result['car_count']
                    successful += 1
                elif result['status'] == 'skipped':
                    self.logger.info(f"⏭️  {result['batch']}: {result['message']}")
                    total_cars += result['car_count']
                else:
                    self.logger.error(f"❌ {result['batch']}: {result['message']}")
                    self.state['failed'].append(result['batch'])
                    failed += 1
                
                self._save_state()
                
                # Progress update
                completed = successful + failed + len([r for r in future_to_task if future_to_task[r] == 'skipped'])
                self.logger.info(f"Progress: {completed}/{len(tasks)} batches")
        
        # Final summary
        self.logger.info("\n" + "="*60)
        self.logger.info("Extraction Complete!")
        self.logger.info(f"Successfully extracted: {successful} batches")
        self.logger.info(f"Already extracted: {len(tasks) - successful - failed} batches")
        self.logger.info(f"Failed: {failed} batches")
        self.logger.info(f"Total cars: {total_cars}")
        
        if self.state['failed']:
            self.logger.info(f"Failed batches: {', '.join(self.state['failed'])}")
    
    def verify_extraction(self):
        """Verify all extractions are complete"""
        self.logger.info("\nVerifying extractions...")
        
        all_good = True
        total_cars = 0
        
        for batch in sorted(self.state['extracted']):
            extract_path = self.extract_dir / batch
            
            if not extract_path.exists():
                self.logger.error(f"❌ {batch}: Directory missing!")
                all_good = False
                continue
            
            car_dirs = [d for d in extract_path.iterdir() if d.is_dir()]
            car_count = len(car_dirs)
            total_cars += car_count
            
            if car_count == 0:
                self.logger.error(f"❌ {batch}: No cars found!")
                all_good = False
            else:
                size_gb = get_directory_size(extract_path)
                self.logger.info(f"✅ {batch}: {car_count} cars, {size_gb:.1f}GB")
        
        self.logger.info(f"\nTotal extracted cars: {total_cars}")
        
        return all_good
    
    def cleanup_downloads(self, dry_run=True):
        """Clean up downloaded zip files after successful extraction"""
        self.logger.info("\nCleanup phase...")
        
        if not self.verify_extraction():
            self.logger.error("Extraction verification failed! Not cleaning up.")
            return
        
        total_size = 0
        files_to_remove = []
        
        for batch in self.state['extracted']:
            zip_path = self.download_dir / f"{batch}.zip"
            if zip_path.exists():
                size_gb = zip_path.stat().st_size / 1e9
                total_size += size_gb
                files_to_remove.append((zip_path, size_gb))
        
        self.logger.info(f"Would free {total_size:.1f}GB by removing {len(files_to_remove)} zip files")
        
        if dry_run:
            self.logger.info("DRY RUN - No files deleted. Run with --cleanup to actually delete.")
            for zip_path, size_gb in files_to_remove:
                self.logger.info(f"  Would delete: {zip_path.name} ({size_gb:.1f}GB)")
        else:
            self.logger.info("Deleting zip files...")
            for zip_path, size_gb in files_to_remove:
                self.logger.info(f"  Deleting: {zip_path.name} ({size_gb:.1f}GB)")
                zip_path.unlink()
            self.logger.info(f"✅ Freed {total_size:.1f}GB of disk space")


def main():
    parser = argparse.ArgumentParser(description="Parallel ZIP Extraction for 3DRealCar")
    parser.add_argument('--download-dir', type=str, required=True,
                        help='Directory containing downloaded zip files')
    parser.add_argument('--extract-dir', type=str, required=True,
                        help='Directory to extract files to')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Actually delete zip files after successful extraction')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing extractions')
    
    args = parser.parse_args()
    
    download_dir = Path(args.download_dir)
    extract_dir = Path(args.extract_dir)
    
    if not download_dir.exists():
        print(f"Error: Download directory not found: {download_dir}")
        return
    
    extractor = ParallelExtractor(download_dir, extract_dir, args.workers)
    
    if args.verify_only:
        extractor.verify_extraction()
    else:
        # Extract all
        extractor.extract_all()
        
        # Verify
        if extractor.verify_extraction():
            # Cleanup if requested
            extractor.cleanup_downloads(dry_run=not args.cleanup)


if __name__ == "__main__":
    main()