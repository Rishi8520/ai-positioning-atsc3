# ===============================================================================
# FILE: ai_data_preprocessor.py
# MODULE: GNSS Telemetry Data Preprocessing Pipeline
# AUTHOR: Tarunika D (AI/ML Systems)
# DATE: January 2026
# PURPOSE: Data loading, normalization, augmentation for model training
# PRODUCTION: Phase 3 - Ready for Deployment
# V2 UPDATES: Config integration, enhanced validation, consistency with v2 architecture
# ===============================================================================

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# V2 Configuration
try:
    from config_v2 import cfg
    USE_V2_CONFIG = True
except ImportError:
    USE_V2_CONFIG = False
    cfg = None

# ===============================================================================
# LOGGING
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# CONSTANTS (compatible with config_v2)
# ===============================================================================

# Get from v2 config if available
if USE_V2_CONFIG and cfg:
    INPUT_FEATURE_DIM = cfg.data.input_dim
    OUTPUT_FEATURE_DIM = cfg.data.output_dim
    OUTLIER_THRESHOLD = cfg.data.outlier_threshold
    NORMALIZE_METHOD = cfg.data.input_scaler
    OUTPUT_SCALER = cfg.data.output_scaler
else:
    INPUT_FEATURE_DIM = 50
    OUTPUT_FEATURE_DIM = 5
    OUTLIER_THRESHOLD = 5.0
    NORMALIZE_METHOD = "standard"
    OUTPUT_SCALER = "minmax"
FEATURE_NAMES = [
    # Subchannel state (20D)
    "signal_strength_0", "signal_strength_1", "signal_strength_2", "signal_strength_3",
    "carrier_phase_0", "carrier_phase_1", "carrier_phase_2", "carrier_phase_3",
    "pseudorange_error_0", "pseudorange_error_1", "pseudorange_error_2", "pseudorange_error_3",
    "doppler_shift_0", "doppler_shift_1", "doppler_shift_2", "doppler_shift_3",
    "tracking_lock_0", "tracking_lock_1", "tracking_lock_2", "tracking_lock_3",
    # Power measurements (10D)
    "received_power_0", "received_power_1", "received_power_2", "received_power_3",
    "carrier_power_0", "carrier_power_1", "carrier_power_2", "carrier_power_3",
    "noise_power_0", "noise_power_1",
    # SNR estimates (10D)
    "snr_db_0", "snr_db_1", "snr_db_2", "snr_db_3",
    "cnr_db_hz_0", "cnr_db_hz_1", "cnr_db_hz_2", "cnr_db_hz_3",
    "multipath_indicator_0", "multipath_indicator_1",
    # Environmental (10D)
    "urban_density_estimate", "blockage_probability", "multipath_likelihood",
    "shadow_fading_estimate", "nlos_probability", "tunnel_probability",
    "time_of_day_normalized", "vehicle_speed_kmh", "heading_deg", "gnss_availability_pct"
]

OUTPUT_FEATURE_NAMES = [
    "redundancy_ratio",
    "spectrum_mbps",
    "availability_pct",
    "convergence_time_sec",
    "accuracy_hpe_cm"
]

# ===============================================================================
# DATA LOADER
# ===============================================================================

class TelemetryLoader:
    """Load telemetry data from files or generate synthetic data"""
    
    @staticmethod
    def load_from_csv(csv_path: str, n_rows: Optional[int] = None) -> np.ndarray:
        """
        Load telemetry data from CSV file
        
        Args:
            csv_path: Path to CSV file
            n_rows: Number of rows to load (None = all)
        
        Returns:
            (N, 50) array of telemetry features
        """
        logger.info(f"Loading telemetry from {csv_path}")
        
        try:
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1, max_rows=n_rows)
            logger.info(f"Loaded {len(data)} samples with shape {data.shape}")
            return data
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {csv_path}")
            raise FileNotFoundError(f"Missing dataset: {csv_path}") from e
        except ValueError as e:
            logger.error(f"Invalid data in CSV: {e}")
            raise ValueError(f"CSV format error: {e}") from e
        except Exception as e:
            logger.critical(f"Unexpected error loading CSV: {e}")
            raise RuntimeError(f"Failed to load CSV: {e}") from e
    
    @staticmethod
    def load_from_numpy(npy_path: str) -> np.ndarray:
        """Load from numpy binary format"""
        logger.info(f"Loading telemetry from {npy_path}")
        data = np.load(npy_path)
        logger.info(f"Loaded {len(data)} samples with shape {data.shape}")
        return data
    
    @staticmethod
    def generate_synthetic(num_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic telemetry data for testing
        
        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
        
        Returns:
            (num_samples, 50) array of synthetic telemetry
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Generating {num_samples} synthetic telemetry samples")
        
        data = np.random.randn(num_samples, INPUT_FEATURE_DIM)
        
        # Add realistic correlations
        # Signal strength correlates negatively with noise
        data[:, :4] = data[:, :4] - data[:, 28:29]
        
        # Multipath correlates with urban density
        data[:, 38] = data[:, 38] + data[:, 30] * 0.5
        
        # GNSS availability inversely correlates with tunnel probability
        data[:, 49] = data[:, 49] - data[:, 35] * 0.7
        
        return data


# ===============================================================================
# FEATURE NORMALIZATION
# ===============================================================================

class FeatureNormalizer:
    """Normalize features for neural network training (V2 compatible)"""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize normalizer
        
        Args:
            method: "standard" (z-score), "minmax" (0-1), or "robust" (IQR-based)
                    - Input features use 'standard' for v2 compatibility
                    - Output targets use 'minmax' for sigmoid output compatibility
        """
        self.method = method
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.is_fitted = False
        self.feature_stats = None  # NEW: Track statistics for validation
        logger.info(f"FeatureNormalizer initialized with method: {method}")
    
    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        """
        Fit normalizer to data
        
        Args:
            X: (N, D) array of features
        
        Returns:
            self
        """
        # V2 ENHANCEMENT: Validate data before fitting
        self._validate_data(X, "input")
        
        self.scaler.fit(X)
        self.is_fitted = True
        
        # NEW: Track statistics for later reference
        self.feature_stats = {
            "mean": X.mean(axis=0),
            "std": X.std(axis=0),
            "min": X.min(axis=0),
            "max": X.max(axis=0),
            "shape": X.shape
        }
        
        logger.info(f"Normalizer fitted on {len(X)} samples (shape {X.shape})")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        return self.scaler.transform(X)
    
    def inverse_transform(self, X_normalized: np.ndarray) -> np.ndarray:
        """Reverse normalization"""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        return self.scaler.inverse_transform(X_normalized)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def _validate_data(self, X: np.ndarray, data_type: str = "input") -> None:
        """
        NEW: Validate data integrity according to v2 standards
        
        Args:
            X: Data array to validate
            data_type: "input" or "output" for context-specific checks
        
        Raises:
            ValueError if validation fails
        """
        if X is None or len(X) == 0:
            raise ValueError(f"{data_type} array is empty")
        
        # Check NaN/Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            raise ValueError(f"Found {nan_count} NaNs and {inf_count} Infs in {data_type}")
        
        # Check dimensions
        expected_dim = OUTPUT_FEATURE_DIM if data_type == "output" else INPUT_FEATURE_DIM
        if X.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {X.shape[1]}")
        
        logger.debug(f"Data validation passed for {data_type}: shape={X.shape}")


# ===============================================================================
# DATA AUGMENTATION
# ===============================================================================

class DataAugmenter:
    """Augment training data with realistic variations"""
    
    @staticmethod
    def add_gaussian_noise(X: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to features
        
        Args:
            X: (N, D) array
            noise_std: Standard deviation of noise
        
        Returns:
            Augmented array
        """
        noise = np.random.randn(*X.shape) * noise_std
        return X + noise
    
    @staticmethod
    def add_sensor_drift(X: np.ndarray, drift_range: float = 0.05) -> np.ndarray:
        """
        Simulate sensor drift
        
        Args:
            X: (N, D) array
            drift_range: Maximum drift range
        
        Returns:
            Data with simulated sensor drift
        """
        # Random drift per feature
        drift = np.random.uniform(-drift_range, drift_range, size=X.shape[1])
        return X + drift
    
    @staticmethod
    def mixup(X1: np.ndarray, X2: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: interpolate between samples
        
        Args:
            X1, X2: Two data arrays of same shape
            alpha: Interpolation weight
        
        Returns:
            Interpolated (X_mixed, indices)
        """
        assert len(X1) == len(X2), "Arrays must have same length"
        
        lam = np.random.beta(alpha, alpha, size=len(X1))
        X_mixed = lam[:, np.newaxis] * X1 + (1 - lam[:, np.newaxis]) * X2
        
        return X_mixed, np.arange(len(X1))
    
    @staticmethod
    def temporal_smoothing(X: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply temporal smoothing to simulate real sensor data
        
        Args:
            X: (N, D) array
            kernel_size: Size of smoothing kernel
        
        Returns:
            Smoothed array
        """
        if kernel_size == 1:
            return X
        
        X_smoothed = X.copy()
        
        for t in range(kernel_size, len(X)):
            X_smoothed[t] = np.mean(X[t-kernel_size:t+1], axis=0)
        
        return X_smoothed
    
    @staticmethod
    def augment_batch(X: np.ndarray, augmentation_factor: int = 2) -> np.ndarray:
        """
        Generate augmented versions of data
        
        Args:
            X: (N, D) array
            augmentation_factor: Number of augmented versions per sample
        
        Returns:
            (N * augmentation_factor, D) augmented array
        """
        augmented_list = [X]
        
        for _ in range(augmentation_factor - 1):
            # Mix different augmentations randomly
            choice = np.random.choice([1, 2, 3])
            
            if choice == 1:
                aug = DataAugmenter.add_gaussian_noise(X, noise_std=0.05)
            elif choice == 2:
                aug = DataAugmenter.add_sensor_drift(X, drift_range=0.02)
            else:
                aug = DataAugmenter.temporal_smoothing(X, kernel_size=2)
            
            augmented_list.append(aug)
        
        X_augmented = np.vstack(augmented_list)
        return X_augmented


# ===============================================================================
# DATA SPLITTER
# ===============================================================================

class DataSplitter:
    """Split data into train/val/test sets"""
    
    @staticmethod
    def split(X: np.ndarray, y: np.ndarray,
              train_ratio: float = 0.7,
              val_ratio: float = 0.15,
              test_ratio: float = 0.15,
              random_state: Optional[int] = None) -> dict:
        """
        Split data into train/val/test
        
        Args:
            X, y: Input and output arrays
            train_ratio, val_ratio, test_ratio: Split ratios
            random_state: Random seed
        
        Returns:
            Dictionary with train/val/test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
        
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Second split: val vs test
        test_size_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size_ratio,
            random_state=random_state
        )
        
        logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }


# ===============================================================================
# COMPLETE PIPELINE
# ===============================================================================

class DataPreprocessingPipeline:
    """Complete data preprocessing pipeline (V2 compatible)"""
    
    def __init__(self, normalization_method: str = "standard"):
        # Use v2 config if available
        if USE_V2_CONFIG and cfg:
            normalization_method = cfg.data.input_scaler
        
        self.normalizer_X = FeatureNormalizer(method=normalization_method)
        self.normalizer_y = FeatureNormalizer(method=OUTPUT_SCALER)  # Always minmax for sigmoid
        self.augmenter = DataAugmenter()
        
        # NEW: Track pipeline statistics
        self.pipeline_stats = {}
        
        logger.info("DataPreprocessingPipeline initialized")
        logger.info(f"Input scaler: {normalization_method}, Output scaler: {OUTPUT_SCALER}")
    
    def process(self, X: np.ndarray, y: np.ndarray,
                augment: bool = True,
                augmentation_factor: int = 2) -> dict:
        """
        Complete preprocessing pipeline (V2 compatible)
        
        Args:
            X, y: Raw input and output arrays
            augment: Whether to apply augmentation
            augmentation_factor: Augmentation multiplier
        
        Returns:
            Dictionary with processed train/val/test splits
        """
        logger.info("Starting data preprocessing pipeline")
        
        # V2 ENHANCEMENT: Enhanced validation
        try:
            self.normalizer_X._validate_data(X, "input")
            self.normalizer_y._validate_data(y, "output")
        except ValueError as e:
            logger.error(f"Data validation failed: {e}")
            raise
        
        # Remove outliers (configurable threshold from v2)
        threshold = OUTLIER_THRESHOLD if USE_V2_CONFIG else 3.0
        initial_count = len(X)
        
        valid_mask = (np.abs(X) < threshold).all(axis=1) & (np.abs(y) < threshold).all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        outlier_count = initial_count - len(X)
        
        logger.info(f"Removed {outlier_count} outliers using {threshold}-sigma rule")
        
        # NEW: Track stats
        self.pipeline_stats["initial_samples"] = initial_count
        self.pipeline_stats["outliers_removed"] = outlier_count
        
        # Augment if requested
        if augment:
            X_orig_len = len(X)
            X = self.augmenter.augment_batch(X, augmentation_factor=augmentation_factor)
            y = np.repeat(y, augmentation_factor, axis=0)
            logger.info(f"Augmented: {X_orig_len} → {len(X)} samples")
            self.pipeline_stats["augmented_samples"] = len(X)
        
        # Normalize inputs and outputs
        X_normalized = self.normalizer_X.fit_transform(X)
        y_normalized = self.normalizer_y.fit_transform(y)
        
        # Split into train/val/test
        splits = DataSplitter.split(X_normalized, y_normalized)
        
        self.pipeline_stats["final_train_size"] = len(splits['X_train'])
        self.pipeline_stats["final_val_size"] = len(splits['X_val'])
        self.pipeline_stats["final_test_size"] = len(splits['X_test'])
        
        logger.info("Data preprocessing complete")
        logger.info(f"Final split: train={len(splits['X_train'])}, val={len(splits['X_val'])}, test={len(splits['X_test'])}")
        
        return splits


# ===============================================================================
# MAIN FOR TESTING
# ===============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("DATA PREPROCESSING - STANDALONE TEST")
    logger.info("=" * 80)
    
    # Generate synthetic data
    loader = TelemetryLoader()
    X = loader.generate_synthetic(num_samples=1000, seed=42)
    
    # Generate synthetic outputs
    y = np.random.rand(1000, 5)
    y[:, 0] = y[:, 0] * 4.0 + 1.0  # redundancy
    y[:, 1] = y[:, 1] * 1.9 + 0.1  # spectrum
    y[:, 2] = y[:, 2] * 0.19 + 0.8  # availability
    y[:, 3] = y[:, 3] * 50.0 + 10.0  # convergence
    y[:, 4] = y[:, 4] * 49.0 + 1.0  # accuracy
    
    logger.info(f"Generated X={X.shape}, y={y.shape}")
    
    # Process data
    pipeline = DataPreprocessingPipeline(normalization_method="standard")
    splits = pipeline.process(X, y, augment=True, augmentation_factor=2)
    
    logger.info(f"Train: X={splits['X_train'].shape}, y={splits['y_train'].shape}")
    logger.info(f"Val:   X={splits['X_val'].shape}, y={splits['y_val'].shape}")
    logger.info(f"Test:  X={splits['X_test'].shape}, y={splits['y_test'].shape}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
