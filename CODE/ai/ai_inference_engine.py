# ===============================================================================
# FILE: ai_inference_engine.py
# MODULE: Real-Time Inference Engine for PPaaS
# AUTHOR: Tarunika D (AI/ML Systems)
# DATE: January 2026
# PURPOSE: Real-time broadcast decision engine with fallback policies
# PRODUCTION: Phase 3 - Ready for Deployment
# ===============================================================================

import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from enum import Enum
from collections import deque

import numpy as np
import torch

from ai_broadcast_decision_model import DecisionInferenceEngine, BroadcastDecision

# ===============================================================================
# LOGGING
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# CONSTANTS
# ===============================================================================

INFERENCE_LATENCY_TARGET_MS = 50
CONFIDENCE_THRESHOLD = 0.6
FALLBACK_WINDOW_SIZE = 10
MAX_INFERENCE_QUEUE = 100

# ===============================================================================
# ENUMS
# ===============================================================================

class FallbackPolicy(Enum):
    """Fallback policies when model confidence is low"""
    CONSERVATIVE = "conservative"    # Increase robustness, reduce spectrum
    AGGRESSIVE = "aggressive"         # Maximize accuracy, increase spectrum
    BALANCED = "balanced"             # Middle ground


# ===============================================================================
# DATA CLASSES
# ===============================================================================

@dataclass
class InferenceMetrics:
    """Metrics from inference execution"""
    inference_time_ms: float
    confidence: float
    policy_applied: str  # "model" or "fallback"
    fallback_reason: Optional[str]


@dataclass
class InferenceResult:
    """Complete inference result"""
    broadcast_decision: BroadcastDecision
    metrics: InferenceMetrics
    timestamp: float


# ===============================================================================
# INFERENCE ENGINE
# ===============================================================================

class InferenceEngine:
    """Real-time inference engine for broadcast decisions"""
    
    def __init__(self, model_path: str, confidence_threshold=CONFIDENCE_THRESHOLD):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model directory
            confidence_threshold: Minimum confidence to accept model output
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load model
        try:
            self.decision_engine = DecisionInferenceEngine(model_path)
            logger.info(f"Loaded inference model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Metrics tracking
        self.inference_history = deque(maxlen=FALLBACK_WINDOW_SIZE)
        self.latency_history = deque(maxlen=100)
        
        logger.info("InferenceEngine initialized")
    
    def _get_fallback_decision(self, policy: FallbackPolicy) -> BroadcastDecision:
        """
        Get fallback decision when model confidence is low
        
        Args:
            policy: Fallback policy to apply
        
        Returns:
            BroadcastDecision with predefined conservative/balanced/aggressive parameters
        """
        fallback_decisions = {
            FallbackPolicy.CONSERVATIVE: BroadcastDecision(
                redundancy_ratio=3.0,
                spectrum_mbps=1.5,
                availability_pct=0.95,
                convergence_time_sec=45.0,
                accuracy_hpe_cm=10.0,
                confidence=0.5
            ),
            FallbackPolicy.BALANCED: BroadcastDecision(
                redundancy_ratio=2.0,
                spectrum_mbps=1.5,
                availability_pct=0.90,
                convergence_time_sec=35.0,
                accuracy_hpe_cm=8.0,
                confidence=0.5
            ),
            FallbackPolicy.AGGRESSIVE: BroadcastDecision(
                redundancy_ratio=1.5,
                spectrum_mbps=2.0,
                availability_pct=0.85,
                convergence_time_sec=25.0,
                accuracy_hpe_cm=3.0,
                confidence=0.5
            )
        }
        
        return fallback_decisions[policy]
    
    def _select_fallback_policy(self, recent_confidence_scores: List[float]) -> FallbackPolicy:
        """
        Select fallback policy based on recent confidence history
        
        Args:
            recent_confidence_scores: List of recent confidence scores
        
        Returns:
            Selected FallbackPolicy
        """
        avg_confidence = np.mean(recent_confidence_scores) if recent_confidence_scores else 0.5
        
        if avg_confidence < 0.4:
            return FallbackPolicy.CONSERVATIVE
        elif avg_confidence > 0.7:
            return FallbackPolicy.AGGRESSIVE
        else:
            return FallbackPolicy.BALANCED
    
    def infer(self, telemetry_features: np.ndarray) -> InferenceResult:
        """
        Perform real-time inference
        
        Args:
            telemetry_features: (50,) array of telemetry + channel state
        
        Returns:
            InferenceResult with decision, metrics, timestamp
        """
        start_time = time.time()
        timestamp = start_time
        
        # Input validation
        if telemetry_features.shape != (50,):
            logger.error(f"Invalid input shape: {telemetry_features.shape}")
            raise ValueError(f"Expected shape (50,), got {telemetry_features.shape}")
        
        try:
            # Model inference
            decision = self.decision_engine.infer(telemetry_features)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Record metrics
            self.inference_history.append(decision.confidence)
            self.latency_history.append(inference_time)
            
            # Check confidence threshold
            if decision.confidence >= self.confidence_threshold:
                # Use model output
                policy_applied = "model"
                fallback_reason = None
                logger.debug(f"Model inference accepted (confidence={decision.confidence:.3f})")
            else:
                # Use fallback policy
                fallback_policy = self._select_fallback_policy(list(self.inference_history))
                decision = self._get_fallback_decision(fallback_policy)
                policy_applied = "fallback"
                fallback_reason = f"{fallback_policy.value} (low confidence)"
                logger.info(f"Fallback policy applied: {fallback_reason}")
            
            # Ensure latency target
            if inference_time > INFERENCE_LATENCY_TARGET_MS:
                logger.warning(f"Inference latency exceeds target: {inference_time:.2f}ms > {INFERENCE_LATENCY_TARGET_MS}ms")
            
            metrics = InferenceMetrics(
                inference_time_ms=inference_time,
                confidence=decision.confidence,
                policy_applied=policy_applied,
                fallback_reason=fallback_reason
            )
            
            result = InferenceResult(
                broadcast_decision=decision,
                metrics=metrics,
                timestamp=timestamp
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Return conservative fallback on error
            decision = self._get_fallback_decision(FallbackPolicy.CONSERVATIVE)
            metrics = InferenceMetrics(
                inference_time_ms=(time.time() - start_time) * 1000,
                confidence=0.0,
                policy_applied="fallback",
                fallback_reason=f"error: {str(e)}"
            )
            result = InferenceResult(
                broadcast_decision=decision,
                metrics=metrics,
                timestamp=timestamp
            )
            return result
    
    def batch_infer(self, telemetry_batch: np.ndarray) -> List[InferenceResult]:
        """
        Batch inference for multiple samples
        
        Args:
            telemetry_batch: (N, 50) array of telemetry
        
        Returns:
            List of InferenceResult objects
        """
        results = []
        
        for i in range(len(telemetry_batch)):
            result = self.infer(telemetry_batch[i])
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        if not self.latency_history:
            return {"error": "No inference history"}
        
        latencies = np.array(list(self.latency_history))
        confidences = np.array(list(self.inference_history))
        
        return {
            "num_inferences": len(self.latency_history),
            "avg_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences))
        }


# ===============================================================================
# BATCH INFERENCE PROCESSOR
# ===============================================================================

class BatchInferenceProcessor:
    """Process batches of telemetry and return broadcast decisions"""
    
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.queue = deque(maxlen=MAX_INFERENCE_QUEUE)
        logger.info("BatchInferenceProcessor initialized")
    
    def add_telemetry(self, vehicle_id: str, telemetry: np.ndarray) -> Optional[InferenceResult]:
        """
        Add telemetry and get inference result
        
        Args:
            vehicle_id: Vehicle identifier
            telemetry: (50,) telemetry array
        
        Returns:
            InferenceResult or None if queue is full
        """
        if len(self.queue) >= MAX_INFERENCE_QUEUE:
            logger.warning("Inference queue full, dropping oldest entry")
            self.queue.popleft()
        
        result = self.engine.infer(telemetry)
        self.queue.append({
            "vehicle_id": vehicle_id,
            "result": result
        })
        
        return result
    
    def get_aggregated_decision(self) -> Optional[BroadcastDecision]:
        """
        Get aggregated decision from all recent inferences
        
        Returns:
            Aggregated BroadcastDecision based on fleet telemetry
        """
        if not self.queue:
            return None
        
        decisions = [item["result"].broadcast_decision for item in self.queue]
        
        # Aggregate by averaging
        aggregated = BroadcastDecision(
            redundancy_ratio=np.mean([d.redundancy_ratio for d in decisions]),
            spectrum_mbps=np.mean([d.spectrum_mbps for d in decisions]),
            availability_pct=np.mean([d.availability_pct for d in decisions]),
            convergence_time_sec=np.mean([d.convergence_time_sec for d in decisions]),
            accuracy_hpe_cm=np.mean([d.accuracy_hpe_cm for d in decisions]),
            confidence=np.mean([d.confidence for d in decisions])
        )
        
        logger.debug(f"Aggregated decision from {len(self.queue)} samples")
        return aggregated
    
    def get_queue_statistics(self) -> Dict:
        """Get statistics about queued inferences"""
        if not self.queue:
            return {"queue_size": 0}
        
        confidences = [item["result"].metrics.confidence for item in self.queue]
        latencies = [item["result"].metrics.inference_time_ms for item in self.queue]
        
        return {
            "queue_size": len(self.queue),
            "avg_confidence": float(np.mean(confidences)),
            "avg_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95))
        }


# ===============================================================================
# MAIN FOR TESTING
# ===============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("INFERENCE ENGINE - STANDALONE TEST")
    logger.info("=" * 80)
    
    # Initialize engine
    model_path = "./models/broadcast_decision_model"
    engine = InferenceEngine(model_path, confidence_threshold=0.6)
    
    # Test single inference
    logger.info("\nTesting single inference...")
    test_telemetry = np.random.randn(50)
    result = engine.infer(test_telemetry)
    logger.info(f"Decision: {result.broadcast_decision.to_dict()}")
    logger.info(f"Metrics: {asdict(result.metrics)}")
    
    # Test batch inference
    logger.info("\nTesting batch inference...")
    batch_processor = BatchInferenceProcessor(engine)
    
    for i in range(10):
        vehicle_id = f"vehicle_{i:03d}"
        telemetry = np.random.randn(50)
        result = batch_processor.add_telemetry(vehicle_id, telemetry)
    
    aggregated = batch_processor.get_aggregated_decision()
    logger.info(f"Aggregated decision: {aggregated.to_dict()}")
    
    # Print statistics
    logger.info("\nEngine Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
