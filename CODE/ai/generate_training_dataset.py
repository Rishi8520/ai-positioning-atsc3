"""
Physics-Based Synthetic Dataset Generator for PPaaS AI System
Generates realistic GNSS telemetry data with proper correlations

Output: Saves dataset to parent directory (../data/)
"""

import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dataset size
NUM_SAMPLES = 15000
TRAIN_RATIO = 0.70  # 10,500 samples
VAL_RATIO = 0.15    # 2,250 samples
TEST_RATIO = 0.15   # 2,250 samples

# Feature dimensions
INPUT_DIM = 50
OUTPUT_DIM = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Output directory (parent directory of CODE/)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "DATA"  # Goes to BUILD-A-THON_4/DATA/

# Output ranges for targets
OUTPUT_RANGES = {
    'redundancy_ratio': (1.0, 5.0),
    'spectrum_mbps': (0.1, 2.0),
    'availability_pct': (0.80, 0.99),
    'convergence_time_sec': (10.0, 60.0),
    'accuracy_hpe_cm': (1.0, 50.0)
}

# ==============================================================================
# PHYSICS-BASED GNSS DATA GENERATOR
# ==============================================================================

class GNSSDataGenerator:
    """Generate synthetic GNSS data with realistic physics-based correlations"""
    
    def __init__(self, seed=RANDOM_SEED):
        np.random.seed(seed)
        self.seed = seed
    
    def _generate_signal_strength(self, num_samples: int, scenario: np.ndarray) -> np.ndarray:
        """
        Generate signal strength (4 channels)
        Urban scenarios have lower signal strength
        """
        # Base signal strength: -130 to -90 dBm, normalized to [0, 1]
        base_signal = np.random.uniform(0.3, 0.8, (num_samples, 4))
        
        # Urban scenarios reduce signal strength
        urban_factor = scenario[:, 0].reshape(-1, 1)  # urban_density
        signal_degradation = urban_factor * 0.3
        
        signal_strength = base_signal - signal_degradation
        signal_strength = np.clip(signal_strength, 0.0, 1.0)
        
        return signal_strength
    
    def _generate_carrier_phase(self, num_samples: int, multipath: np.ndarray) -> np.ndarray:
        """
        Generate carrier phase measurements (4 channels)
        Affected by multipath
        """
        # Base carrier phase: 0 to 2π, normalized to [0, 1]
        carrier_phase = np.random.uniform(0.0, 1.0, (num_samples, 4))
        
        # Multipath causes phase errors
        multipath_factor = multipath[:, 0].reshape(-1, 1)
        phase_error = np.random.normal(0, 0.1, (num_samples, 4)) * multipath_factor
        
        carrier_phase += phase_error
        carrier_phase = np.clip(carrier_phase, 0.0, 1.0)
        
        return carrier_phase
    
    def _generate_pseudorange_error(self, num_samples: int, nlos_prob: np.ndarray) -> np.ndarray:
        """
        Generate pseudorange error (4 channels)
        NLOS increases error
        """
        # Base error: 0 to 100m, normalized to [0, 1]
        base_error = np.random.uniform(0.01, 0.15, (num_samples, 4))
        
        # NLOS dramatically increases error
        nlos_factor = nlos_prob.reshape(-1, 1)
        error_increase = nlos_factor * 0.5
        
        pseudorange_error = base_error + error_increase
        pseudorange_error = np.clip(pseudorange_error, 0.0, 1.0)
        
        return pseudorange_error
    
    def _generate_doppler_shift(self, num_samples: int, vehicle_speed: np.ndarray) -> np.ndarray:
        """
        Generate Doppler shift (4 channels)
        Correlated with vehicle speed
        """
        # Doppler shift: -5000 to 5000 Hz, normalized to [0, 1]
        speed_factor = vehicle_speed.reshape(-1, 1)
        
        # Create base doppler correlated with speed
        doppler_base = 0.5 + (speed_factor - 0.5) * 0.3  # Speed correlation
        
        # Expand to 4 channels with slight variations
        doppler = np.tile(doppler_base, (1, 4))  # Replicate to 4 columns
        
        # Add channel-specific noise
        doppler += np.random.normal(0, 0.05, (num_samples, 4))
        doppler = np.clip(doppler, 0.0, 1.0)
        
        return doppler
    
    def _generate_snr_cnr(self, signal_strength: np.ndarray, noise_power: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate SNR and CNR (4 channels each)
        Derived from signal strength and noise
        """
        # SNR: Signal to Noise Ratio
        noise_expanded = np.tile(noise_power, (1, 2))  # Expand from 2 to 4
        snr = signal_strength - noise_expanded * 0.3
        snr = np.clip(snr, 0.0, 1.0)
        
        # CNR: Carrier to Noise Ratio (slightly higher than SNR)
        cnr = snr + np.random.uniform(0.05, 0.15, snr.shape)
        cnr = np.clip(cnr, 0.0, 1.0)
        
        return snr, cnr
    
    def _generate_environmental_context(self, num_samples: int) -> np.ndarray:
        """
        Generate environmental context (10 features)
        Includes urban density, blockage, multipath, NLOS, etc.
        """
        env = np.zeros((num_samples, 10))
        
        # Urban density (0=rural, 1=dense urban)
        env[:, 0] = np.random.beta(2, 5, num_samples)  # Skewed towards lower values
        
        # Blockage probability (correlated with urban density)
        env[:, 1] = env[:, 0] * 0.6 + np.random.uniform(0, 0.2, num_samples)
        env[:, 1] = np.clip(env[:, 1], 0, 1)
        
        # Multipath likelihood (high in urban areas)
        env[:, 2] = env[:, 0] * 0.7 + np.random.uniform(0, 0.15, num_samples)
        env[:, 2] = np.clip(env[:, 2], 0, 1)
        
        # Shadow fading
        env[:, 3] = np.random.beta(2, 3, num_samples)
        
        # NLOS probability (correlated with blockage)
        env[:, 4] = env[:, 1] * 0.8 + np.random.uniform(0, 0.1, num_samples)
        env[:, 4] = np.clip(env[:, 4], 0, 1)
        
        # Tunnel probability (rare events)
        env[:, 5] = np.random.exponential(0.05, num_samples)
        env[:, 5] = np.clip(env[:, 5], 0, 1)
        
        # Time of day (uniform distribution)
        env[:, 6] = np.random.uniform(0, 1, num_samples)
        
        # Vehicle speed (0-200 kmh, normalized)
        env[:, 7] = np.random.gamma(2, 0.15, num_samples)  # Most vehicles 30-80 kmh
        env[:, 7] = np.clip(env[:, 7], 0, 1)
        
        # Heading (uniform distribution)
        env[:, 8] = np.random.uniform(0, 1, num_samples)
        
        # GNSS availability (high in rural, lower in urban)
        env[:, 9] = 0.95 - env[:, 0] * 0.25 + np.random.uniform(-0.05, 0.05, num_samples)
        env[:, 9] = np.clip(env[:, 9], 0.7, 1.0)
        
        return env
    
    def generate_features(self, num_samples: int) -> np.ndarray:
        """
        Generate complete 50D feature vectors
        
        Returns:
            X: (num_samples, 50) feature array
        """
        print(f"Generating {num_samples} feature vectors (50D)...")
        
        # First generate environmental context (needed for other features)
        env_context = self._generate_environmental_context(num_samples)
        
        # Extract key environmental factors
        urban_density = env_context[:, 0:1]
        multipath = env_context[:, 2:3]
        nlos_prob = env_context[:, 4]
        vehicle_speed = env_context[:, 7]
        
        # Generate signal features
        signal_strength = self._generate_signal_strength(num_samples, env_context)  # 4D
        carrier_phase = self._generate_carrier_phase(num_samples, multipath)  # 4D
        pseudorange_error = self._generate_pseudorange_error(num_samples, nlos_prob)  # 4D
        doppler_shift = self._generate_doppler_shift(num_samples, vehicle_speed)  # 4D
        
        # Tracking lock (degrades with poor signal)
        tracking_lock = signal_strength * 0.9 + np.random.uniform(0, 0.1, (num_samples, 4))  # 4D
        tracking_lock = np.clip(tracking_lock, 0.5, 1.0)
        
        # Power measurements
        received_power = signal_strength * 0.95 + np.random.normal(0, 0.05, (num_samples, 4))  # 4D
        received_power = np.clip(received_power, 0, 1)
        
        carrier_power = received_power * 0.9  # 4D
        
        noise_power = np.random.uniform(0.1, 0.3, (num_samples, 2))  # 2D
        
        # SNR and CNR
        snr, cnr = self._generate_snr_cnr(signal_strength, noise_power)  # 4D each
        
        # Multipath indicator
        multipath_indicator = multipath[:, 0:1] + np.random.uniform(-0.1, 0.1, (num_samples, 1))
        multipath_indicator = np.clip(multipath_indicator, 0, 1)
        multipath_indicator = np.tile(multipath_indicator, (1, 2))  # 2D
        
        # Concatenate all features (should sum to 50)
        X = np.concatenate([
            signal_strength,      # 4
            carrier_phase,        # 4
            pseudorange_error,    # 4
            doppler_shift,        # 4
            tracking_lock,        # 4
            received_power,       # 4
            carrier_power,        # 4
            noise_power,          # 2
            snr,                  # 4
            cnr,                  # 4
            multipath_indicator,  # 2
            env_context          # 10
        ], axis=1)
        
        assert X.shape == (num_samples, 50), f"Expected (n, 50), got {X.shape}"
        
        print(f"✓ Generated features: {X.shape}")
        return X
    
    def generate_targets(self, X: np.ndarray) -> np.ndarray:
        """
        Generate 5D target outputs based on physics-based correlations
        
        Args:
            X: (num_samples, 50) feature array
        
        Returns:
            y: (num_samples, 5) target array
        """
        num_samples = X.shape[0]
        print(f"Generating {num_samples} target vectors (5D)...")
        
        y = np.zeros((num_samples, 5))
        
        # Extract key features for correlations
        signal_strength_avg = np.mean(X[:, 0:4], axis=1)      # Average signal strength
        snr_avg = np.mean(X[:, 30:34], axis=1)                # Average SNR
        multipath_avg = np.mean(X[:, 38:40], axis=1)          # Multipath indicator
        urban_density = X[:, 40]                              # Urban density
        nlos_prob = X[:, 44]                                  # NLOS probability
        
        # 1. Redundancy Ratio (1.0 - 5.0)
        # Higher redundancy needed when signal is poor
        redundancy = 2.5 - (signal_strength_avg - 0.5) * 3.0  # Inverse correlation
        redundancy += nlos_prob * 1.5                         # NLOS increases redundancy
        redundancy += np.random.normal(0, 0.3, num_samples)   # Noise
        y[:, 0] = np.clip(redundancy, 1.0, 5.0)
        
        # 2. Spectrum (Mbps) (0.1 - 2.0)
        # More spectrum needed for higher redundancy and urban areas
        spectrum = 0.5 + (redundancy - 2.5) * 0.3
        spectrum += urban_density * 0.4                       # Urban needs more spectrum
        spectrum += np.random.normal(0, 0.15, num_samples)
        y[:, 1] = np.clip(spectrum, 0.1, 2.0)
        
        # 3. Availability (%) (0.80 - 0.99)
        # Better signal = higher availability
        availability = 0.85 + snr_avg * 0.14
        availability -= nlos_prob * 0.1                       # NLOS reduces availability
        availability += np.random.normal(0, 0.02, num_samples)
        y[:, 2] = np.clip(availability, 0.80, 0.99)
        
        # 4. Convergence Time (sec) (10 - 60)
        # Poor signal = longer convergence
        convergence = 35.0 - (snr_avg - 0.5) * 40.0          # Inverse correlation
        convergence += multipath_avg * 15.0                   # Multipath delays convergence
        convergence += nlos_prob * 20.0                       # NLOS significantly delays
        convergence += np.random.normal(0, 5, num_samples)
        y[:, 3] = np.clip(convergence, 10.0, 60.0)
        
        # 5. Accuracy HPE (cm) (1.0 - 50.0)
        # Better signal = better accuracy
        accuracy = 25.0 - (signal_strength_avg - 0.5) * 30.0  # Inverse correlation
        accuracy += multipath_avg * 15.0                      # Multipath degrades accuracy
        accuracy += nlos_prob * 20.0                          # NLOS degrades accuracy
        accuracy += np.random.normal(0, 5, num_samples)
        y[:, 4] = np.clip(accuracy, 1.0, 50.0)
        
        print(f"✓ Generated targets: {y.shape}")
        return y
    
    def generate_dataset(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete dataset
        
        Returns:
            X: (num_samples, 50) features
            y: (num_samples, 5) targets
        """
        X = self.generate_features(num_samples)
        y = self.generate_targets(X)
        return X, y

# ==============================================================================
# DATASET STATISTICS & VALIDATION
# ==============================================================================

def compute_statistics(X: np.ndarray, y: np.ndarray) -> Dict:
    """Compute dataset statistics"""
    
    stats = {
        'num_samples': int(X.shape[0]),
        'num_features': int(X.shape[1]),
        'num_targets': int(y.shape[1]),
        'features': {
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist(),
            'min': X.min(axis=0).tolist(),
            'max': X.max(axis=0).tolist(),
            'nan_count': int(np.isnan(X).sum()),
            'inf_count': int(np.isinf(X).sum())
        },
        'targets': {
            'mean': y.mean(axis=0).tolist(),
            'std': y.std(axis=0).tolist(),
            'min': y.min(axis=0).tolist(),
            'max': y.max(axis=0).tolist(),
            'nan_count': int(np.isnan(y).sum()),
            'inf_count': int(np.isinf(y).sum())
        },
        'target_names': [
            'redundancy_ratio',
            'spectrum_mbps',
            'availability_pct',
            'convergence_time_sec',
            'accuracy_hpe_cm'
        ],
        'target_ranges': OUTPUT_RANGES
    }
    
    return stats

def print_statistics(stats: Dict):
    """Pretty print dataset statistics"""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Samples: {stats['num_samples']}")
    print(f"Features: {stats['num_features']}")
    print(f"Targets: {stats['num_targets']}")
    
    print("\n--- Target Statistics ---")
    for i, name in enumerate(stats['target_names']):
        print(f"{name:25s}: mean={stats['targets']['mean'][i]:6.2f}, "
              f"std={stats['targets']['std'][i]:6.2f}, "
              f"min={stats['targets']['min'][i]:6.2f}, "
              f"max={stats['targets']['max'][i]:6.2f}")
    
    print("\n--- Data Quality ---")
    print(f"Features - NaN: {stats['features']['nan_count']}, Inf: {stats['features']['inf_count']}")
    print(f"Targets  - NaN: {stats['targets']['nan_count']}, Inf: {stats['targets']['inf_count']}")
    print("="*80 + "\n")

# ==============================================================================
# MAIN GENERATION PIPELINE
# ==============================================================================

def main():
    """Main dataset generation pipeline"""
    
    print("="*80)
    print("PPaaS AI SYSTEM - PHYSICS-BASED SYNTHETIC DATASET GENERATOR")
    print("="*80)
    print(f"Configuration:")
    print(f"  Total samples: {NUM_SAMPLES}")
    print(f"  Train: {int(NUM_SAMPLES * TRAIN_RATIO)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {int(NUM_SAMPLES * VAL_RATIO)} ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {int(NUM_SAMPLES * TEST_RATIO)} ({TEST_RATIO*100:.0f}%)")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Output directory: {DATA_DIR}")
    print("="*80 + "\n")
    
    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created output directory: {DATA_DIR}\n")
    
    # Initialize generator
    generator = GNSSDataGenerator(seed=RANDOM_SEED)
    
    # Generate complete dataset
    print("Generating complete dataset...")
    X, y = generator.generate_dataset(NUM_SAMPLES)
    
    # Compute and print statistics
    stats = compute_statistics(X, y)
    print_statistics(stats)
    
    # Split into train/val/test
    train_size = int(NUM_SAMPLES * TRAIN_RATIO)
    val_size = int(NUM_SAMPLES * VAL_RATIO)
    
    indices = np.random.permutation(NUM_SAMPLES)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Split dataset:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}\n")
    
    # Save datasets
    print("Saving datasets...")
    
    np.save(DATA_DIR / 'X_train.npy', X_train)
    np.save(DATA_DIR / 'y_train.npy', y_train)
    np.save(DATA_DIR / 'X_val.npy', X_val)
    np.save(DATA_DIR / 'y_val.npy', y_val)
    np.save(DATA_DIR / 'X_test.npy', X_test)
    np.save(DATA_DIR / 'y_test.npy', y_test)
    
    # Save full dataset (optional)
    np.save(DATA_DIR / 'X_full.npy', X)
    np.save(DATA_DIR / 'y_full.npy', y)
    
    print(f"✓ Saved: X_train.npy, y_train.npy")
    print(f"✓ Saved: X_val.npy, y_val.npy")
    print(f"✓ Saved: X_test.npy, y_test.npy")
    print(f"✓ Saved: X_full.npy, y_full.npy\n")
    
    # Save metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'generator_version': '2.0_physics_based',
        'random_seed': RANDOM_SEED,
        'num_samples_total': NUM_SAMPLES,
        'num_samples_train': train_size,
        'num_samples_val': val_size,
        'num_samples_test': len(test_idx),
        'input_dim': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
        'statistics': stats,
        'file_sizes_mb': {
            'X_train': X_train.nbytes / (1024**2),
            'y_train': y_train.nbytes / (1024**2),
            'X_val': X_val.nbytes / (1024**2),
            'y_val': y_val.nbytes / (1024**2),
            'X_test': X_test.nbytes / (1024**2),
            'y_test': y_test.nbytes / (1024**2),
        }
    }
    
    with open(DATA_DIR / 'dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved: dataset_metadata.json\n")
    
    # Print file sizes
    total_size_mb = sum(metadata['file_sizes_mb'].values())
    print(f"Total dataset size: {total_size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"\nDataset location: {DATA_DIR}")
    print("\nFiles created:")
    print("  - X_train.npy, y_train.npy  (training set)")
    print("  - X_val.npy, y_val.npy      (validation set)")
    print("  - X_test.npy, y_test.npy    (test set)")
    print("  - X_full.npy, y_full.npy    (complete dataset)")
    print("  - dataset_metadata.json     (metadata)")
    print("\nNext steps:")
    print("  1. Verify dataset: python validate_dataset.py")
    print("  2. Train model with new data")
    print("="*80)

if __name__ == "__main__":
    main()