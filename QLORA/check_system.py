#!/usr/bin/env python3
"""
System Check - Verify that your system is ready for QLoRA training

Run this before starting training to check:
- GPU availability and memory
- CUDA version compatibility
- Required dependencies
- Disk space
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(f" {text} ".center(70))
    print("=" * 70)

def check_gpu():
    """Check GPU availability and specs."""
    print_header("GPU Check")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', 
                               '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                name, mem, cap = line.split(',')
                mem_gb = float(mem.strip().split()[0]) / 1024
                print(f"✓ GPU Found: {name.strip()}")
                print(f"  Memory: {mem_gb:.1f} GB")
                print(f"  Compute Capability: {cap.strip()}")
                
                # Check if memory is sufficient
                if mem_gb < 15:
                    print(f"  ⚠ Warning: {mem_gb:.1f} GB may not be enough (recommend 16+ GB)")
                else:
                    print(f"  ✓ Memory sufficient for QLoRA training")
            return True
        else:
            print("✗ No NVIDIA GPU found")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False

def check_cuda():
    """Check CUDA availability in PyTorch."""
    print_header("CUDA Check")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: Yes")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
            
            # Test GPU
            print("\n  Testing GPU computation...")
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("  ✓ GPU computation successful")
            return True
        else:
            print("✗ CUDA not available in PyTorch")
            print("  This may mean:")
            print("  - PyTorch was installed without CUDA support")
            print("  - CUDA drivers are not properly installed")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    print_header("Dependencies Check")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT (LoRA)',
        'bitsandbytes': 'BitsAndBytes',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
    }
    
    all_installed = True
    for module, name in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: Not installed")
            all_installed = False
    
    return all_installed

def check_disk_space():
    """Check available disk space."""
    print_header("Disk Space Check")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_pct = (used / total) * 100
        
        print(f"Disk Space:")
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Free: {free_gb:.1f} GB")
        print(f"  Used: {used_pct:.1f}%")
        
        # Model + dataset + checkpoints need ~50GB
        if free_gb < 50:
            print(f"  ⚠ Warning: {free_gb:.1f} GB free may not be enough")
            print(f"    Recommend: 50+ GB free space")
            return False
        else:
            print(f"  ✓ Sufficient space available")
            return True
    except Exception as e:
        print(f"✗ Could not check disk space: {e}")
        return False

def check_python_version():
    """Check Python version."""
    print_header("Python Version Check")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version compatible")
        return True
    else:
        print("✗ Python 3.8 or higher required")
        return False

def estimate_training_time():
    """Estimate training time based on GPU."""
    print_header("Training Time Estimate")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', 
                               '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_name = result.stdout.strip().lower()
            
            estimates = {
                'a100': '6-8 hours',
                'v100': '10-14 hours',
                '4090': '8-12 hours',
                '3090': '10-14 hours',
                'a6000': '8-10 hours',
            }
            
            estimate = None
            for key, time in estimates.items():
                if key in gpu_name:
                    estimate = time
                    break
            
            if estimate:
                print(f"Estimated training time (1 epoch): {estimate}")
            else:
                print("Estimated training time: 8-16 hours (varies by GPU)")
                print(f"Your GPU: {result.stdout.strip()}")
        
        print("\nFactors affecting training time:")
        print("  - GPU model and memory")
        print("  - Batch size and gradient accumulation")
        print("  - Sequence length")
        print("  - System load")
    except:
        print("Could not estimate training time")

def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print(" QLoRA Training System Check ".center(70))
    print("=" * 70)
    
    checks = {
        'Python Version': check_python_version(),
        'GPU': check_gpu(),
        'CUDA': check_cuda(),
        'Dependencies': check_dependencies(),
        'Disk Space': check_disk_space(),
    }
    
    estimate_training_time()
    
    # Summary
    print_header("Summary")
    
    all_passed = all(checks.values())
    
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("\n✓ All checks passed! Your system is ready for training.")
        print("\nNext steps:")
        print("  1. python train_qlora.py --debug    # Quick test")
        print("  2. python train_qlora.py            # Full training")
    else:
        print("\n✗ Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - CUDA issues: Reinstall PyTorch with CUDA support")
        print("  - Disk space: Free up space or use a different directory")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())