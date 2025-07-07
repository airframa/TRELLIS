#!/usr/bin/env python3
"""
Monitor training progress from log file
"""
import re
import sys
import time
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path):
    """Parse training log and extract metrics"""
    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return None
    
    metrics = {
        'step': 0,
        'loss': 0,
        'val_loss': None,
        'speed': 0,
        'eta': 'Unknown',
        'elapsed': 'Unknown'
    }
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Parse from end of file for most recent info
    for line in reversed(lines[-1000:]):  # Check last 1000 lines
        # Parse training step
        if "Step:" in line and "ETA:" in line:
            match = re.search(r'Step: (\d+)/(\d+).*Speed: ([\d.]+) steps/h.*ETA: ([\d.]+) h.*Elapsed: ([\d.]+) h', line)
            if match:
                metrics['step'] = int(match.group(1))
                metrics['total_steps'] = int(match.group(2))
                metrics['speed'] = float(match.group(3))
                metrics['eta'] = f"{float(match.group(4)):.1f}h"
                metrics['elapsed'] = f"{float(match.group(5)):.1f}h"
        
        # Parse training loss
        if "loss:" in line and "rec:" in line:
            loss_match = re.search(r'loss: ([\d.]+)', line)
            if loss_match:
                metrics['loss'] = float(loss_match.group(1))
        
        # Parse validation loss
        if "[Validation]" in line:
            val_match = re.search(r'Loss = ([\d.]+)', line)
            if val_match:
                metrics['val_loss'] = float(val_match.group(1))
    
    return metrics

def monitor_training(log_path, interval=30):
    """Monitor training progress"""
    print("="*60)
    print("3DRealCar Training Monitor")
    print("="*60)
    print(f"Log file: {log_path}")
    print(f"Update interval: {interval}s")
    print("Press Ctrl+C to exit")
    print("="*60)
    
    while True:
        try:
            metrics = parse_log_file(log_path)
            
            if metrics:
                # Clear screen
                print("\033[H\033[J", end='')
                
                # Header
                print("="*60)
                print(f"3DRealCar Training Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*60)
                
                # Progress bar
                if 'total_steps' in metrics:
                    progress = metrics['step'] / metrics['total_steps']
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"Progress: [{bar}] {progress*100:.1f}%")
                
                # Metrics
                print(f"\nStep:          {metrics['step']:,}/{metrics.get('total_steps', 'Unknown'):,}")
                print(f"Training Loss: {metrics['loss']:.6f}")
                if metrics['val_loss'] is not None:
                    print(f"Val Loss:      {metrics['val_loss']:.6f}")
                print(f"Speed:         {metrics['speed']:.1f} steps/hour")
                print(f"Elapsed:       {metrics['elapsed']}")
                print(f"ETA:           {metrics['eta']}")
                
                # Estimates
                if metrics['speed'] > 0 and 'total_steps' in metrics:
                    steps_per_day = metrics['speed'] * 24
                    print(f"\nEstimated steps/day: {steps_per_day:,.0f}")
                    
                    remaining_steps = metrics['total_steps'] - metrics['step']
                    days_remaining = remaining_steps / steps_per_day
                    print(f"Estimated days remaining: {days_remaining:.1f}")
                
                print("\n" + "="*60)
            else:
                print("Waiting for log data...")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <log_path> [interval]")
        sys.exit(1)
    
    log_path = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    monitor_training(log_path, interval)