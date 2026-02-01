"""Unit tests for inference engine"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_inference_engine_v2 import InferenceEngineV2, InferenceBackend, InferenceMetrics
from ai_broadcast_decision_model_v2 import BroadcastDecision


class TestInferenceEngineV2:
    """Test inference engine V2 functionality"""
    
    @pytest.fixture
    def test_input(self):
        """Create test input (50D feature vector)"""
        np.random.seed(42)
        return np.random.uniform(0, 1, size=50).astype(np.float32)
    
    def test_engine_initialization(self):
        """Test engine can be initialized"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            assert engine is not None, "Engine initialization failed"
            assert engine.backend == InferenceBackend.PYTORCH, "Backend not set correctly"
        except FileNotFoundError:
            # Model not trained yet, skip
            pytest.skip("Model not found - train model first with ai_broadcast_decision_model_v2.py")
    
    def test_inference_output_structure(self, test_input):
        """Test inference returns correct structure"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            result = engine.infer(test_input)
            
            # Check result has required fields
            assert hasattr(result, 'broadcast_decision'), "Missing broadcast_decision"
            assert hasattr(result, 'metrics'), "Missing metrics"
            assert hasattr(result, 'timestamp'), "Missing timestamp"
            
            # Check broadcast decision
            assert isinstance(result.broadcast_decision, BroadcastDecision)
            assert isinstance(result.metrics, InferenceMetrics)
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_output_ranges(self, test_input):
        """Test inference outputs are within valid ranges"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            result = engine.infer(test_input)
            decision = result.broadcast_decision
            
            # Check output ranges match specification
            assert 1.0 <= decision.redundancy_ratio <= 5.0, \
                f"Redundancy {decision.redundancy_ratio} out of range [1.0, 5.0]"
            assert 0.1 <= decision.spectrum_mbps <= 2.0, \
                f"Spectrum {decision.spectrum_mbps} out of range [0.1, 2.0]"
            assert 0.80 <= decision.availability_pct <= 0.99, \
                f"Availability {decision.availability_pct} out of range [0.80, 0.99]"
            assert 10.0 <= decision.convergence_time_sec <= 60.0, \
                f"Convergence {decision.convergence_time_sec} out of range [10.0, 60.0]"
            assert 1.0 <= decision.accuracy_hpe_cm <= 50.0, \
                f"Accuracy {decision.accuracy_hpe_cm} out of range [1.0, 50.0]"
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_confidence_score(self, test_input):
        """Test confidence score is valid"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            result = engine.infer(test_input)
            
            assert 0 <= result.metrics.confidence <= 1, \
                f"Confidence {result.metrics.confidence} out of range [0, 1]"
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_uncertainty_score(self, test_input):
        """Test uncertainty score is valid"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH, mc_samples=5)
            result = engine.infer(test_input)
            
            assert 0 <= result.metrics.uncertainty <= 1, \
                f"Uncertainty {result.metrics.uncertainty} out of range [0, 1]"
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_inference_latency(self, test_input):
        """Test inference completes within reasonable time"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            result = engine.infer(test_input)
            
            # Should complete in < 5 seconds
            assert result.metrics.inference_time_ms < 5000, \
                f"Inference took {result.metrics.inference_time_ms}ms (expected < 5000ms)"
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_batch_inference(self):
        """Test batch inference with multiple inputs"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            
            # Create batch of inputs
            batch = np.random.uniform(0, 1, size=(5, 50)).astype(np.float32)
            
            results = []
            for i in range(batch.shape[0]):
                result = engine.infer(batch[i])
                results.append(result)
            
            # All should be valid
            assert len(results) == 5, "Batch inference failed"
            for result in results:
                assert result.broadcast_decision is not None
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_consistency(self, test_input):
        """Test inference is consistent for same input"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            
            # Same input should produce similar outputs (MC Dropout adds variance)
            result1 = engine.infer(test_input)
            result2 = engine.infer(test_input)
            
            # Mean should be close (within 10% due to MC Dropout)
            assert np.abs(result1.broadcast_decision.accuracy_hpe_cm - 
                          result2.broadcast_decision.accuracy_hpe_cm) < 5, \
                "Results not consistent for same input"
            
        except FileNotFoundError:
            pytest.skip("Model not found")


class TestInferenceMetrics:
    """Test inference metrics"""
    
    def test_metrics_to_dict(self):
        """Test metrics can be converted to dict"""
        metrics = InferenceMetrics(
            inference_time_ms=25.5,
            confidence=0.95,
            uncertainty=0.05,
            policy_applied="model",
            fallback_reason=None,
            backend="pytorch"
        )
        
        metrics_dict = metrics.__dict__
        assert metrics_dict['confidence'] == 0.95
        assert metrics_dict['backend'] == 'pytorch'


class TestInferenceEdgeCases:
    """Test edge cases"""
    
    def test_zero_input(self):
        """Test with all-zero input"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            zero_input = np.zeros(50, dtype=np.float32)
            
            result = engine.infer(zero_input)
            assert result.broadcast_decision is not None
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_max_input(self):
        """Test with maximum values"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            max_input = np.ones(50, dtype=np.float32)
            
            result = engine.infer(max_input)
            assert result.broadcast_decision is not None
            
        except FileNotFoundError:
            pytest.skip("Model not found")
    
    def test_mixed_input(self):
        """Test with mixed extreme values"""
        try:
            engine = InferenceEngineV2(backend=InferenceBackend.PYTORCH)
            mixed_input = np.random.choice([0.0, 1.0], size=50).astype(np.float32)
            
            result = engine.infer(mixed_input)
            assert result.broadcast_decision is not None
            
        except FileNotFoundError:
            pytest.skip("Model not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
