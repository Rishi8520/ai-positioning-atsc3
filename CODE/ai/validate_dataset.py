"""
Dataset Validation Script
Checks quality and integrity of generated dataset
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Dataset directory
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "DATA"

def validate_dataset():
    """Validate dataset files and check for issues"""
    
    print("="*80)
    print("DATASET VALIDATION")
    print("="*80)
    
    # Check if directory exists
    if not DATA_DIR.exists():
        print(f"❌ ERROR: Data directory not found: {DATA_DIR}")
        return False
    
    print(f"✓ Data directory found: {DATA_DIR}\n")
    
    # Check required files
    required_files = [
        'X_train.npy', 'y_train.npy',
        'X_val.npy', 'y_val.npy',
        'X_test.npy', 'y_test.npy',
        'dataset_metadata.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not (DATA_DIR / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ ERROR: Missing files: {missing_files}")
        return False
    
    print("✓ All required files present\n")
    
    # Load datasets
    print("Loading datasets...")
    X_train = np.load(DATA_DIR / 'X_train.npy')
    y_train = np.load(DATA_DIR / 'y_train.npy')
    X_val = np.load(DATA_DIR / 'X_val.npy')
    y_val = np.load(DATA_DIR / 'y_val.npy')
    X_test = np.load(DATA_DIR / 'X_test.npy')
    y_test = np.load(DATA_DIR / 'y_test.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}\n")
    
    # Check shapes
    all_checks_passed = True
    
    if X_train.shape[1] != 50:
        print(f"❌ ERROR: X_train should have 50 features, got {X_train.shape[1]}")
        all_checks_passed = False
    else:
        print("✓ X_train has correct feature dimension (50)")
    
    if y_train.shape[1] != 5:
        print(f"❌ ERROR: y_train should have 5 targets, got {y_train.shape[1]}")
        all_checks_passed = False
    else:
        print("✓ y_train has correct target dimension (5)")
    
    # Check for NaN and Inf
    datasets = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    print("\n--- Data Quality Checks ---")
    for name, data in datasets.items():
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count > 0:
            print(f"❌ {name}: Found {nan_count} NaN values")
            all_checks_passed = False
        else:
            print(f"✓ {name}: No NaN values")
        
        if inf_count > 0:
            print(f"❌ {name}: Found {inf_count} Inf values")
            all_checks_passed = False
        else:
            print(f"✓ {name}: No Inf values")
    
    # Check value ranges
    print("\n--- Value Range Checks ---")
    
    # Features should be in [0, 1]
    if X_train.min() < -0.1 or X_train.max() > 1.1:
        print(f"⚠ WARNING: X_train range [{X_train.min():.3f}, {X_train.max():.3f}] outside [0, 1]")
    else:
        print(f"✓ X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    # Targets should be in expected ranges
    target_names = ['redundancy_ratio', 'spectrum_mbps', 'availability_pct', 
                    'convergence_time_sec', 'accuracy_hpe_cm']
    expected_ranges = [(1.0, 5.0), (0.1, 2.0), (0.80, 0.99), (10.0, 60.0), (1.0, 50.0)]
    
    for i, (name, (min_val, max_val)) in enumerate(zip(target_names, expected_ranges)):
        actual_min = y_train[:, i].min()
        actual_max = y_train[:, i].max()
        
        if actual_min < min_val - 0.1 or actual_max > max_val + 0.1:
            print(f"⚠ {name}: range [{actual_min:.2f}, {actual_max:.2f}] outside expected [{min_val}, {max_val}]")
        else:
            print(f"✓ {name}: [{actual_min:.2f}, {actual_max:.2f}]")
    
    # Load and display metadata
    print("\n--- Metadata ---")
    with open(DATA_DIR / 'dataset_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Generation date: {metadata['generation_date']}")
    print(f"Generator version: {metadata['generator_version']}")
    print(f"Total samples: {metadata['num_samples_total']}")
    print(f"Total size: {sum(metadata['file_sizes_mb'].values()):.2f} MB")
    
    # Final verdict
    print("\n" + "="*80)
    if all_checks_passed:
        print("✅ VALIDATION PASSED - Dataset is ready for training!")
    else:
        print("❌ VALIDATION FAILED - Please regenerate dataset")
    print("="*80)
    
    return all_checks_passed

if __name__ == "__main__":
    validate_dataset()