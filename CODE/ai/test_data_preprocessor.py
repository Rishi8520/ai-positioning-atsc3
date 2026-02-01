"""Unit tests for data preprocessing module"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_data_preprocessor import (
    TelemetryLoader,
    DataPreprocessingPipeline,
    FeatureNormalizer
)


class TestTelemetryLoader:
    """Test data loading functionality"""
    
    def test_generate_synthetic(self):
        """Test synthetic data generation"""
        loader = TelemetryLoader()
        X = loader.generate_synthetic(num_samples=100, seed=42)
        
        assert X.shape == (100, 50), f"Expected shape (100, 50), got {X.shape}"
        assert not np.isnan(X).any(), "Generated data contains NaN"
        assert not np.isinf(X).any(), "Generated data contains Inf"
    
    def test_deterministic_generation(self):
        """Test that same seed produces same data"""
        loader = TelemetryLoader()
        X1 = loader.generate_synthetic(num_samples=100, seed=42)
        X2 = loader.generate_synthetic(num_samples=100, seed=42)
        
        np.testing.assert_array_equal(X1, X2, "Same seed should produce identical data")
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data"""
        loader = TelemetryLoader()
        X1 = loader.generate_synthetic(num_samples=100, seed=42)
        X2 = loader.generate_synthetic(num_samples=100, seed=43)
        
        assert not np.allclose(X1, X2), "Different seeds should produce different data"


class TestFeatureNormalizer:
    """Test feature normalization"""
    
    def test_standard_scaler(self):
        """Test StandardScaler normalization"""
        X = np.random.randn(100, 50)
        normalizer = FeatureNormalizer(method="standard")
        X_normalized = normalizer.fit_transform(X)
        
        # Check normalized mean is close to 0 and std is close to 1
        assert np.allclose(X_normalized.mean(axis=0), 0, atol=1e-10), "StandardScaler mean not 0"
        assert np.allclose(X_normalized.std(axis=0), 1, atol=1e-10), "StandardScaler std not 1"
    
    def test_minmax_scaler(self):
        """Test MinMaxScaler normalization"""
        X = np.random.rand(100, 50) * 100
        normalizer = FeatureNormalizer(method="minmax")
        X_normalized = normalizer.fit_transform(X)
        
        # Check normalized range is [0, 1]
        assert X_normalized.min() >= -0.01, "MinMax scaler output below 0"
        assert X_normalized.max() <= 1.01, "MinMax scaler output above 1"
    
    def test_scaler_inverse_transform(self):
        """Test inverse transformation"""
        X = np.random.rand(100, 50) * 100
        normalizer = FeatureNormalizer(method="standard")
        X_normalized = normalizer.fit_transform(X)
        X_recovered = normalizer.inverse_transform(X_normalized)
        
        np.testing.assert_array_almost_equal(X, X_recovered, decimal=5, 
                                             err_msg="Inverse transform failed")


class TestDataPreprocessingPipeline:
    """Test complete preprocessing pipeline"""
    
    def test_pipeline_output_shape(self):
        """Test pipeline produces correct output shapes"""
        X = np.random.randn(1000, 50)
        y = np.random.rand(1000, 5)
        
        pipeline = DataPreprocessingPipeline()
        splits = pipeline.process(X, y, augment=False)
        
        # Check split shapes
        assert splits['X_train'].shape[1] == 50, "X_train should have 50 features"
        assert splits['y_train'].shape[1] == 5, "y_train should have 5 targets"
        
        total_samples = len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test'])
        assert total_samples == 1000, "Total samples should equal original"
    
    def test_pipeline_with_augmentation(self):
        """Test pipeline with data augmentation"""
        X = np.random.randn(100, 50)
        y = np.random.rand(100, 5)
        
        pipeline = DataPreprocessingPipeline()
        splits = pipeline.process(X, y, augment=True, augmentation_factor=2)
        
        # Should have augmented data
        total_samples = len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test'])
        assert total_samples > 100, "Augmentation should increase sample count"
        logger.info(f"Original: 100 samples, Augmented: {total_samples} samples")
    
    def test_pipeline_handles_nan(self):
        """Test pipeline handles NaN values"""
        X = np.random.randn(100, 50)
        y = np.random.rand(100, 5)
        
        # Add NaN to test validation
        X[0, 0] = np.nan
        X[5, 10] = np.nan
        
        pipeline = DataPreprocessingPipeline()
        splits = pipeline.process(X, y, augment=False)
        
        # Check no NaN in output
        assert not np.isnan(splits['X_train']).any(), "Output contains NaN values"
        assert not np.isnan(splits['y_train']).any(), "Output targets contain NaN"
    
    def test_pipeline_output_normalization(self):
        """Test that outputs are properly normalized"""
        X = np.random.rand(100, 50) * 10
        y = np.random.rand(100, 5)
        
        pipeline = DataPreprocessingPipeline()
        splits = pipeline.process(X, y, augment=False)
        
        # Check X is normalized (mean ~0, std ~1)
        X_train_mean = np.mean(splits['X_train'])
        X_train_std = np.std(splits['X_train'])
        
        assert np.abs(X_train_mean) < 0.5, f"X_train mean {X_train_mean} not close to 0"
        assert np.abs(X_train_std - 1.0) < 0.5, f"X_train std {X_train_std} not close to 1"


class TestDataPreprocessingEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_small_dataset(self):
        """Test with very small dataset"""
        X = np.random.randn(10, 50)
        y = np.random.rand(10, 5)
        
        pipeline = DataPreprocessingPipeline()
        splits = pipeline.process(X, y, augment=False)
        
        # Should still work, may have small splits
        assert len(splits['X_train']) > 0, "Train set is empty"
        assert len(splits['X_val']) > 0, "Val set is empty"
        assert len(splits['X_test']) > 0, "Test set is empty"
    
    def test_constant_features(self):
        """Test with constant features"""
        X = np.ones((100, 50)) * 5.0  # All same value
        y = np.random.rand(100, 5)
        
        pipeline = DataPreprocessingPipeline()
        splits = pipeline.process(X, y, augment=False)
        
        # Should handle constant features gracefully
        assert splits['X_train'].shape[0] > 0, "Failed with constant features"
    
    def test_inf_values_detected(self):
        """Test that Inf values are handled"""
        X = np.random.randn(100, 50)
        y = np.random.rand(100, 5)
        
        # Add Inf value
        X[0, 0] = np.inf
        
        pipeline = DataPreprocessingPipeline()
        splits = pipeline.process(X, y, augment=False)
        
        # Should be handled (either removed or converted)
        assert not np.isinf(splits['X_train']).any(), "Output contains Inf"


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    pytest.main([__file__, "-v", "--tb=short"])
