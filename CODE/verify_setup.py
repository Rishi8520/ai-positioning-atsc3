#!/usr/bin/env python3
"""
Verify that all required packages are installed correctly.
"""

import sys


def check_import(module_name, package_name=None, attr_check=None):
    """Try to import a module and report status."""
    if package_name is None:
        package_name = module_name
    
    try:
        mod = __import__(module_name)
        # Optional attribute check for packages with submodules
        if attr_check:
            for attr in attr_check.split('.'):
                mod = getattr(mod, attr)
        print(f"✓ {package_name:20s} installed")
        return True
    except (ImportError, AttributeError) as e:
        print(f"✗ {package_name:20s} MISSING")
        return False


def check_commpy():
    """Special check for CommPy (scikit-commpy)."""
    try:
        import commpy
        # Test that we can actually import the channelcoding module
        from commpy.channelcoding import ldpc_bp_decode
        print(f"✓ {'CommPy':20s} installed")
        return True
    except ImportError as e:
        print(f"✗ {'CommPy':20s} MISSING ({e})")
        return False


def main():
    print("=" * 60)
    print("AI-Positioning PoC Environment Verification")
    print("=" * 60)
    
    all_ok = True
    
    print("\n[BROADCAST TEAM]")
    all_ok &= check_commpy()  # Use special check for CommPy
    all_ok &= check_import("reedsolo", "Reed-Solomon")
    all_ok &= check_import("numpy", "NumPy")
    all_ok &= check_import("scipy", "SciPy")
    
    print("\n[GNSS TEAM]")
    all_ok &= check_import("pyproj", "PyProj")
    all_ok &= check_import("georinex", "GeoRINEX")
    all_ok &= check_import("PIL", "Pillow")
    
    print("\n[AI/ML TEAM]")
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("onnx", "ONNX")
    all_ok &= check_import("onnxruntime", "ONNX Runtime")
    all_ok &= check_import("sklearn", "scikit-learn")
    all_ok &= check_import("pandas", "Pandas")
    
    print("\n[VISUALIZATION]")
    all_ok &= check_import("matplotlib", "Matplotlib")
    all_ok &= check_import("seaborn", "Seaborn")
    
    print("\n[DEVELOPMENT]")
    all_ok &= check_import("pytest", "pytest")
    all_ok &= check_import("flask", "Flask")
    all_ok &= check_import("jupyter", "Jupyter")
    
    print("\n[DATA FORMATS]")
    all_ok &= check_import("h5py", "HDF5")
    all_ok &= check_import("yaml", "PyYAML")
    
    # Check PyTorch device
    print("\n[PYTORCH CONFIGURATION]")
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CPU device: ✓")
    
    # Additional detailed checks
    print("\n[DETAILED PACKAGE VERSIONS]")
    try:
        import commpy
        print(f"  CommPy: {commpy.__version__}")
    except:
        print(f"  CommPy: (version unavailable)")
    
    import reedsolo
    print(f"  Reed-Solomon: {reedsolo.__version__ if hasattr(reedsolo, '__version__') else 'installed'}")
    
    import numpy
    print(f"  NumPy: {numpy.__version__}")
    
    import scipy
    print(f"  SciPy: {scipy.__version__}")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All packages installed successfully!")
        print("\nYou're ready to start development!")
    else:
        print("✗ Some packages are missing. Please check above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()