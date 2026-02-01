import numpy as np
from pathlib import Path
from .ai_broadcast_decision_model_v2 import DecisionInferenceEngineV2

# Load model
model_path = Path(__file__).parent / "models" / "broadcast_decision_model_v2"
engine = DecisionInferenceEngineV2(str(model_path))

# Load test data
data_dir = Path(__file__).parent.parent.parent / "DATA"
X_test = np.load(data_dir / 'X_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

# Test on 10 samples
print("="*80)
print("MODEL INFERENCE TEST - 10 SAMPLES")
print("="*80)

for i in range(10):
    decision = engine.infer(X_test[i], mc_samples=20)
    
    print(f"\nSample {i+1}:")
    print(f"  Predicted: redundancy={decision.redundancy_ratio:.2f}, "
          f"spectrum={decision.spectrum_mbps:.2f} Mbps, "
          f"availability={decision.availability_pct*100:.1f}%")
    print(f"  True:      redundancy={y_test[i,0]:.2f}, "
          f"spectrum={y_test[i,1]:.2f} Mbps, "
          f"availability={y_test[i,2]*100:.1f}%")
    print(f"  Confidence: {decision.confidence:.3f}, Uncertainty: {decision.uncertainty:.3f}")

print("\n" + "="*80)