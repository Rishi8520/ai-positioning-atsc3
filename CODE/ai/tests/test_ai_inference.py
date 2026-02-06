#!/usr/bin/env python3
"""
Quick test of AI inference - make sure model works before pipeline integration
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add parent dir to path
from ai_inference_engine_v2 import InferenceEngineV2, InferenceBackend

# Initialize inference engine
print("=" * 70)
print("  AI INFERENCE TEST")
print("=" * 70)

model_path = "../models/broadcast_decision_model_v2"
print(f"Loading model from: {model_path}")

engine = InferenceEngineV2(
    model_path=model_path,
    confidence_threshold=0.6,
    backend=InferenceBackend.PYTORCH,
    mc_samples=20  # Monte Carlo Dropout for uncertainty
)

print("✓ Model loaded successfully\n")

# Create a test telemetry sample (50D vector)
# Simulating rural scenario with good conditions
print("Creating test telemetry (rural scenario with good signal)...")
test_telemetry = np.random.randn(50) * 0.5  # Random but realistic

# Set some specific values for rural/good conditions
test_telemetry[40] = 0.15  # Low urban density (rural)
test_telemetry[38:40] = 0.2  # Low multipath
test_telemetry[49] = 0.95  # High GNSS availability

print(f"✓ Telemetry shape: {test_telemetry.shape}\n")

# AI Inference
print("Running AI inference...")
result = engine.infer(test_telemetry)

# Extract decision
decision = result.broadcast_decision

print("\n" + "=" * 70)
print("  AI DECISION OUTPUT")
print("=" * 70)
print(f"Redundancy Ratio:     {decision.redundancy_ratio:.2f}  (1.0-5.0)")
print(f"Spectrum (Mbps):      {decision.spectrum_mbps:.2f}  (0.1-2.0)")
print(f"Availability:         {decision.availability_pct:.2%}  (80-99%)")
print(f"Convergence Time (s): {decision.convergence_time_sec:.1f}  (10-60s)")
print(f"Accuracy HPE (cm):    {decision.accuracy_hpe_cm:.1f}  (1-50cm)")
print(f"\nConfidence:           {decision.confidence:.3f}")
print(f"Uncertainty:          {decision.uncertainty:.3f}")

# Metrics
print("\n" + "=" * 70)
print("  INFERENCE METRICS")
print("=" * 70)
print(f"Inference Time:       {result.metrics.inference_time_ms:.2f} ms")
print(f"Policy Applied:       {result.metrics.policy_applied}")
print(f"Backend:              {result.metrics.backend}")

print("\n" + "=" * 70)
print("  TEST COMPLETE ✓")
print("=" * 70)
print("\nNext step: Integrate into scenario pipeline")