#!/bin/bash
# Setup script for QLoRA training environment

set -e  # Exit on error

echo "=========================================="
echo "QLoRA Training Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check for GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "⚠ Warning: nvidia-smi not found. GPU may not be available."
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support)
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."

python3 << EOF
import torch
import transformers
import peft
import bitsandbytes

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"✓ Transformers version: {transformers.__version__}")
print(f"✓ PEFT version: {peft.__version__}")
print(f"✓ BitsAndBytes version: {bitsandbytes.__version__}")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Test with debug mode: python train_qlora.py --debug"
echo "3. Run full training: python train_qlora.py"
echo ""
echo "For long training sessions, use tmux:"
echo "  tmux new -s training"
echo "  python train_qlora.py"
echo "  # Detach: Ctrl+B then D"
echo "  # Reattach: tmux attach -t training"
echo ""