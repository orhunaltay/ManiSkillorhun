# Installation Guide for LightBulb-in-Socket Task

## Prerequisites

- Python 3.9 or higher
- pip

## Installation Steps

### Step 1: Install ManiSkill

Since you're working with the ManiSkill repository, you need to install it in development mode:

```bash
cd /Users/orhunaltay/ManiSkill  # or wherever your ManiSkill directory is

# Install ManiSkill in editable/development mode
pip install -e .
```

This will install all required dependencies including:
- gymnasium==0.29.1
- transforms3d
- sapien (physics engine)
- and many others

### Step 2: Install SAPIEN (macOS specific)

**IMPORTANT**: If you're on macOS, SAPIEN needs to be installed manually since it's not on PyPI yet.

Check your Python version:
```bash
python --version  # e.g., Python 3.10.x
```

Then install the appropriate SAPIEN wheel:

**For Python 3.9:**
```bash
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp39-cp39-macosx_12_0_universal2.whl"
```

**For Python 3.10:**
```bash
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp310-cp310-macosx_12_0_universal2.whl"
```

**For Python 3.11:**
```bash
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp311-cp311-macosx_12_0_universal2.whl"
```

**For Python 3.12:**
```bash
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp312-cp312-macosx_12_0_universal2.whl"
```

### Step 3: Verify Installation

```bash
python -c "import gymnasium; import mani_skill; print('Success!')"
```

If this runs without errors, you're ready to go!

### Step 4: Run the LightBulb Task

```bash
cd /Users/orhunaltay/ManiSkill
python lightbulb_motion_planning.py
```

## Quick Installation (One-liner for Python 3.10 on macOS)

```bash
cd /Users/orhunaltay/ManiSkill && \
pip install -e . && \
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp310-cp310-macosx_12_0_universal2.whl"
```

## Troubleshooting

### "No module named 'gymnasium'"
- Make sure you ran `pip install -e .` in the ManiSkill directory
- Verify with: `pip show gymnasium`

### "No module named 'sapien'"
- Install SAPIEN manually using the appropriate wheel for your Python version
- Check available versions at: https://github.com/haosulab/SAPIEN/releases

### "No module named 'mani_skill'"
- Run `pip install -e .` in the ManiSkill root directory
- Make sure you're in the correct directory

### ImportError with transforms3d
- Run: `pip install transforms3d`

### Rendering Issues on macOS
- ManiSkill uses GPU rendering which may have limited support on macOS
- Try running with `render_mode=None` or `render_mode="rgb_array"` instead of `render_mode="human"`

## Alternative: Virtual Environment (Recommended)

Create a clean environment for ManiSkill:

```bash
# Create virtual environment
python3 -m venv maniskill_env

# Activate it
source maniskill_env/bin/activate  # macOS/Linux
# or
maniskill_env\Scripts\activate  # Windows

# Install ManiSkill
cd /Users/orhunaltay/ManiSkill
pip install -e .

# Install SAPIEN (macOS - adjust for your Python version)
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp310-cp310-macosx_12_0_universal2.whl"

# Run the task
python lightbulb_motion_planning.py
```

## Linux Installation

If you're on Linux, installation is simpler:

```bash
cd /path/to/ManiSkill
pip install -e .
# SAPIEN will install automatically from setup.py
```

## Windows Installation

```bash
cd C:\path\to\ManiSkill
pip install -e .
# SAPIEN will install automatically from setup.py
```

## Dependencies List

Key dependencies that will be installed:
- gymnasium==0.29.1 (RL environment interface)
- transforms3d (rotation math)
- sapien>=3.0.0 (physics engine)
- numpy>=1.22
- scipy
- torch (for GPU acceleration)
- h5py (data storage)
- trimesh (mesh processing)

For a complete list, see `setup.py`
