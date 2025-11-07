#!/usr/bin/env python3
"""
Quick environment diagnostic script.
Run this to check your Python environment setup.
"""

import sys
import subprocess

print("=" * 80)
print("Python Environment Diagnostic")
print("=" * 80)
print()

# Check Python executable
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[0]}")
print()

# Check if in conda environment
import os
conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
print(f"Conda environment: {conda_env}")
print()

# Try to import key packages
print("Checking package imports:")
print("-" * 40)

packages = [
    'gymnasium',
    'gym',
    'numpy',
    'sapien',
    'mani_skill',
    'torch',
    'trimesh',
]

for package in packages:
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package:20s} version: {version}")
    except ImportError as e:
        print(f"✗ {package:20s} NOT FOUND")
    except Exception as e:
        print(f"? {package:20s} ERROR: {e}")

print()
print("=" * 80)
print("Diagnosis:")
print("=" * 80)

if conda_env == 'Not in conda environment':
    print("⚠ WARNING: You are not in a conda environment!")
    print("  Solution: Run 'conda activate ms4' first")
elif conda_env != 'ms4':
    print(f"⚠ WARNING: You are in '{conda_env}' but should be in 'ms4'")
    print("  Solution: Run 'conda activate ms4'")

# Check if gymnasium is installed
try:
    import gymnasium
    print("✓ gymnasium is installed and importable")
except ImportError:
    print("✗ gymnasium is NOT installed")
    print("  Solution: Run 'pip install gymnasium==0.29.1'")

print()
