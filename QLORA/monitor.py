#!/usr/bin/env python3
"""
Training Monitor - Real-time monitoring of QLoRA training

Usage:
    python monitor.py [log_file]
    
If no log file is specified, monitors the default training.log
"""

import sys
import time
import re
from pathlib import Path
import subprocess

def get_gpu_stats():
    """Get current GPU memory usage and utilization."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            mem_used, mem_total, gpu_util = result.stdout.strip().split(',')
            return {
                'mem_used': float(mem_used),
                'mem_total': float(mem_total),
                'gpu_util': float(gpu_util)
            }
    except Exception as e:
        pass
    return None

def parse_training_log(log_file):
    """Parse training log file and extract key metrics."""
    if not Path(log_file).exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {
        'current_loss': None,
        'steps_completed': 0,
        'epoch': None,
        'learning_rate': None,
        'samples_processed': 0,
        'time_elapsed': None,
    }
    
    # Find latest loss value
    loss_matches = re.findall(r'loss[=:]\s*([\d.]+)', content)
    if loss_matches:
        metrics['current_loss'] = float(loss_matches[-1])
    
    # Find step number
    step_matches = re.findall(r'Step\s+(\d+)/', content)
    if step_matches:
        metrics['steps_completed'] = int(step_matches[-1])
    
    # Find epoch
    epoch_matches = re.findall(r'Epoch\s+(\d+)/', content)
    if epoch_matches:
        metrics['epoch'] = int(epoch_matches[-1])
    
    # Find learning rate
    lr_matches = re.findall(r'lr[=:]\s*([\d.e-]+)', content)
    if lr_matches:
        metrics['learning_rate'] = float(lr_matches[-1])
    
    return metrics

def print_dashboard(gpu_stats, training_metrics):
    """Print a nice dashboard with current stats."""
    # Clear screen
    print("\033[2J\033[H", end='')
    
    print("=" * 70)
    print(" QLoRA Training Monitor ".center(70))
    print("=" * 70)
    print()
    
    # GPU Stats
    if gpu_stats:
        mem_pct = (gpu_stats['mem_used'] / gpu_stats['mem_total']) * 100
        print("GPU Status:")
        print(f"  Memory: {gpu_stats['mem_used']:.0f} MB / {gpu_stats['mem_total']:.0f} MB ({mem_pct:.1f}%)")
        print(f"  Utilization: {gpu_stats['gpu_util']:.0f}%")
        
        # Draw progress bar for memory
        bar_width = 40
        filled = int(bar_width * mem_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  [{bar}]")
    else:
        print("GPU Status: Unable to query")
    
    print()
    
    # Training Stats
    if training_metrics:
        print("Training Progress:")
        if training_metrics['epoch']:
            print(f"  Epoch: {training_metrics['epoch']}")
        if training_metrics['steps_completed']:
            print(f"  Steps Completed: {training_metrics['steps_completed']}")
        if training_metrics['current_loss']:
            print(f"  Current Loss: {training_metrics['current_loss']:.4f}")
        if training_metrics['learning_rate']:
            print(f"  Learning Rate: {training_metrics['learning_rate']:.2e}")
    else:
        print("Training Progress: Waiting for training to start...")
    
    print()
    print("=" * 70)
    print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to exit")
    print("=" * 70)

def monitor_training(log_file="training.log", refresh_interval=5):
    """Monitor training progress in real-time."""
    try:
        while True:
            gpu_stats = get_gpu_stats()
            training_metrics = parse_training_log(log_file)
            print_dashboard(gpu_stats, training_metrics)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def print_summary(log_file="training.log"):
    """Print a summary of the training run."""
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    print("\n" + "=" * 70)
    print(" Training Summary ".center(70))
    print("=" * 70 + "\n")
    
    # Extract all loss values
    losses = [float(x) for x in re.findall(r'loss[=:]\s*([\d.]+)', content)]
    if losses:
        print(f"Loss Progression:")
        print(f"  Initial: {losses[0]:.4f}")
        print(f"  Final: {losses[-1]:.4f}")
        print(f"  Improvement: {losses[0] - losses[-1]:.4f}")
        print(f"  Min: {min(losses):.4f}")
        print(f"  Max: {max(losses):.4f}")
        print()
    
    # Check if training completed
    if "Training complete" in content:
        print("Status: ✓ Training completed successfully")
    elif "Error" in content or "exception" in content.lower():
        print("Status: ✗ Training encountered errors")
    else:
        print("Status: ⋯ Training in progress")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--summary":
            log_file = sys.argv[2] if len(sys.argv) > 2 else "training.log"
            print_summary(log_file)
        else:
            log_file = sys.argv[1]
            monitor_training(log_file)
    else:
        # Default: monitor training.log
        print("Monitoring training.log (press Ctrl+C to exit)")
        print("To monitor a different file: python monitor.py <log_file>")
        print("To see summary: python monitor.py --summary [log_file]")
        print()
        time.sleep(2)
        monitor_training()