# Fix "ModuleNotFoundError: No module named 'gymnasium'" Issue

This error means Python can't find the gymnasium package. This is almost always an **environment mismatch** issue.

## Step 1: Check Your Environment

Run this diagnostic script:
```bash
python3 check_environment.py
```

This will show you:
- Which Python you're using
- Whether you're in the correct conda environment
- Which packages are installed

## Step 2: Common Problem - Using System Python Instead of Conda Python

### The Issue
When you type `python3`, your system might be using the **system Python** (e.g., `/usr/local/bin/python3`) instead of your **conda environment Python** (e.g., `~/miniconda3/envs/ms4/bin/python`).

### The Solution

**Option A: Use `python` instead of `python3`**

When inside a conda environment, use `python` (not `python3`):

```bash
conda activate ms4
python run_pick_banana_scripted.py  # Use 'python', not 'python3'
```

**Option B: Make sure you're actually in the conda environment**

```bash
# First, activate the environment
conda activate ms4

# Verify you're in it
which python  # Should show something like /Users/orhunaltay/miniconda3/envs/ms4/bin/python

# Then run the script with 'python'
python run_pick_banana_scripted.py
```

## Step 3: Install Packages in the Correct Environment

If packages are still missing, install them **while the conda environment is active**:

```bash
# Make sure you're in the environment
conda activate ms4

# Verify which pip you're using
which pip  # Should be in your conda env, not system

# Install packages
pip install gymnasium==0.29.1
pip install numpy scipy transforms3d trimesh imageio pyyaml tqdm h5py
```

## Step 4: Install SAPIEN for Mac

```bash
# Check your Python version first
python --version

# For Python 3.11 (most common):
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp311-cp311-macosx_12_0_universal2.whl"

# For Python 3.10, use cp310-cp310
# For Python 3.12, use cp312-cp312
```

## Step 5: Install ManiSkill

```bash
conda activate ms4
cd /Users/orhunaltay/ManiSkill
pip install -e .
```

## Complete Fresh Start (If Nothing Else Works)

If you're still having issues, start completely fresh:

```bash
# Deactivate current environment
conda deactivate

# Remove and recreate the environment
conda env remove -n ms4
conda create -n ms4 python=3.11 -y

# Activate it
conda activate ms4

# Verify you're using the right Python
which python
which pip

# Install ManiSkill
cd /Users/orhunaltay/ManiSkill
pip install -e .

# Install SAPIEN for Mac
pip install "sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-cp311-cp311-macosx_12_0_universal2.whl"

# Download assets
python -m mani_skill.utils.download_asset partnet_mobility_cabinet
python -m mani_skill.utils.download_asset ycb

# Test
python -c "import gymnasium; print('✓ Works!')"

# Run the script
python run_pick_banana_scripted.py
```

## Quick Checklist

Before running the script, verify:

- [ ] `conda activate ms4` has been run
- [ ] `which python` shows a path inside your conda environment
- [ ] `python -c "import gymnasium"` doesn't give an error
- [ ] You're using `python` (not `python3`) when inside conda
- [ ] You're in the correct directory: `/Users/orhunaltay/ManiSkill`

## Still Not Working?

Show me the output of these commands:

```bash
conda activate ms4
which python
which pip
python --version
python -c "import sys; print(sys.executable)"
python -c "import gymnasium; print('success')"
```

Copy and paste the output and I'll help you diagnose further.

## Why This Happens

On Mac, you often have multiple Python installations:
1. **System Python**: `/usr/bin/python3` (came with macOS)
2. **Homebrew Python**: `/usr/local/bin/python3` (if you use Homebrew)
3. **Conda Python**: `~/miniconda3/envs/ms4/bin/python` (your conda environment)

When you run `python3`, your shell might pick the **wrong one**. Inside a conda environment, always use `python` (without the 3), which automatically picks the right one.
