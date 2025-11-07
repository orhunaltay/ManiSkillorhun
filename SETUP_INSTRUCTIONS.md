# Setup Instructions for PickBananaFromOpenDrawer Task

## Prerequisites

You're currently in the `ms4` conda environment but missing some dependencies. Here's how to set it up:

## Installation Steps

### 1. Activate your conda environment (if not already active)
```bash
conda activate ms4
```

### 2. Install ManiSkill in development mode

From the ManiSkill directory:
```bash
cd /Users/orhunaltay/ManiSkill  # or wherever your ManiSkill directory is
pip install -e .
```

This will install all required dependencies including:
- `gymnasium==0.29.1`
- `numpy`, `scipy`, `trimesh`, `transforms3d`
- `sapien` (for Mac, needs manual installation - see below)
- Other dependencies

### 3. Install SAPIEN for Mac (Special Instructions)

Since you're on Mac, SAPIEN needs to be installed manually. Check your Python version first:
```bash
python --version  # Should show Python 3.9, 3.10, 3.11, etc.
```

Then install the appropriate SAPIEN wheel from the SAPIEN releases:
```bash
# For Python 3.11 (adjust cp311 based on your Python version):
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp311-cp311-macosx_12_0_universal2.whl"
```

Available versions:
- Python 3.9: `cp39-cp39`
- Python 3.10: `cp310-cp310`
- Python 3.11: `cp311-cp311`
- Python 3.12: `cp312-cp312`

Check the [SAPIEN releases page](https://github.com/haosulab/SAPIEN/releases) for the latest nightly build.

### 4. Download Required Assets

ManiSkill needs PartNet Mobility and YCB datasets:
```bash
# Download PartNet Mobility cabinet dataset
python -m mani_skill.utils.download_asset partnet_mobility_cabinet

# Download YCB dataset (for banana object)
python -m mani_skill.utils.download_asset ycb
```

## Running the Script

After installation, you can run:
```bash
python3 run_pick_banana_scripted.py
```

## Quick Install (Alternative)

If you just want to test quickly with minimal setup:

```bash
# Install core dependencies
pip install gymnasium==0.29.1 numpy scipy transforms3d trimesh imageio pyyaml tqdm

# Install SAPIEN for your Python version (see step 3 above)
```

Then run the script from the ManiSkill directory.

## Troubleshooting

### "ModuleNotFoundError: No module named 'gymnasium'"
- Make sure you've activated the conda environment: `conda activate ms4`
- Install gymnasium: `pip install gymnasium==0.29.1`

### "ModuleNotFoundError: No module named 'mani_skill'"
- Install ManiSkill: `pip install -e .` from the ManiSkill directory

### "ModuleNotFoundError: No module named 'sapien'"
- Install SAPIEN manually (see step 3 above)
- Make sure you use the correct Python version in the wheel URL

### "Asset not found" errors
- Download the required assets (see step 4 above)

### Import errors for the task
- Make sure you're running from the ManiSkill root directory
- Or add the ManiSkill directory to your PYTHONPATH:
  ```bash
  export PYTHONPATH=/Users/orhunaltay/ManiSkill:$PYTHONPATH
  ```

## Verifying Installation

Test that everything is installed correctly:
```python
import gymnasium
import sapien
import mani_skill
print("✓ All imports successful!")
```

## Minimal Test

If you want to test without rendering:
```python
import gymnasium as gym
import pick_banana_from_open_drawer_simple

env = gym.make(
    "PickBananaFromOpenDrawerSimple-v1",
    obs_mode="state_dict",
    render_mode="rgb_array",  # No window
    control_mode="pd_ee_delta_pos",
    num_envs=1,
)

obs, info = env.reset(seed=0)
print("Environment created successfully!")
env.close()
```

## System Requirements

- **OS**: macOS 12.0 or later
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **RAM**: At least 4GB recommended
- **GPU**: Not required for CPU simulation (SAPIEN CPU backend)

## Getting Help

If you encounter issues:
1. Check the [ManiSkill documentation](https://maniskill.readthedocs.io/)
2. Verify your Python version matches the SAPIEN wheel
3. Make sure you're in the correct conda environment
4. Check that assets are downloaded to `~/.maniskill/data/`
