#!/bin/bash
# Quick installation script for the banana picking task
# Run with: bash quick_install.sh

echo "=========================================="
echo "Installing PickBananaFromOpenDrawer dependencies"
echo "=========================================="
echo ""

# Check if conda environment is active
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠ No conda environment detected."
    echo "Please activate your conda environment first:"
    echo "  conda activate ms4"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Using conda environment: ${CONDA_DEFAULT_ENV}"
fi

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
echo "✓ Detected Python ${PYTHON_VERSION}"
echo ""

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements_banana_task.txt
if [ $? -eq 0 ]; then
    echo "✓ Core dependencies installed"
else
    echo "✗ Failed to install core dependencies"
    exit 1
fi
echo ""

# Check OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - SAPIEN needs manual installation"
    echo ""
    echo "Please install SAPIEN manually:"
    echo "  For Python 3.${PYTHON_MINOR}:"
    echo "  pip install 'sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp3${PYTHON_MINOR}-cp3${PYTHON_MINOR}-macosx_12_0_universal2.whl'"
    echo ""
    echo "Check the latest release at:"
    echo "  https://github.com/haosulab/SAPIEN/releases"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux - Installing SAPIEN..."
    pip install 'sapien>=3.0.0'
    if [ $? -eq 0 ]; then
        echo "✓ SAPIEN installed"
    else
        echo "✗ Failed to install SAPIEN"
        exit 1
    fi
else
    echo "⚠ Unknown OS: $OSTYPE"
    echo "Please install SAPIEN manually"
fi
echo ""

# Download assets
echo "Downloading required assets..."
echo "This may take several minutes..."
echo ""

python3 -m mani_skill.utils.download_asset partnet_mobility_cabinet
if [ $? -eq 0 ]; then
    echo "✓ PartNet Mobility cabinet dataset downloaded"
else
    echo "✗ Failed to download cabinet dataset"
fi

python3 -m mani_skill.utils.download_asset ycb
if [ $? -eq 0 ]; then
    echo "✓ YCB dataset downloaded"
else
    echo "✗ Failed to download YCB dataset"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To test the installation, run:"
echo "  python3 -c 'import gymnasium; import sapien; import mani_skill; print(\"✓ Success!\")'"
echo ""
echo "To run the banana picking demo:"
echo "  python3 run_pick_banana_scripted.py"
echo ""
