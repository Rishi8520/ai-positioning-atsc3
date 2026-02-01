# PPaaS AI System - Complete Execution & Reference Guide (V2)

**Version:** 2.0 (Updated February 2026)  
**Status:** Production Ready with V2 Enhancements



---

##  QUICK START (Pick One)

### 1. PowerShell (RECOMMENDED)
```powershell
.\run.ps1
```

### 2. Batch Script
```bash
.\run.bat
```

### 3. Manual
```powershell
.venv\Scripts\Activate.ps1
python main.py
```

---

## 📊 5-Stage System Overview (V2)

Your system executes 5 stages automatically with V2 enhancements:

| Stage | Module | Time | Purpose | V2 Features |
|-------|--------|------|---------|------------|
| **1** | Data Preprocessing | 2-5s | Load, normalize, augment, split | ✓ Config-driven, validation |
| **2** | Model Training | 60-120s | Train with residual blocks | ✓ Residual connections, MC Dropout |
| **3** | Inference Engine | 5-10s | Real-time with uncertainty | ✓ Uncertainty quantification, ONNX |
| **4** | Feedback Loop | 10-20s | Detect drift with uncertainty | ✓ Uncertainty-aware decisions |
| **5** | Intent Parser | 5-15s | NLP with constraint validation | ✓ Constraint validation |
| | **TOTAL** | **80-170s** | **2-3 minutes** | ✓ **All V2 Compatible** |

---

## 📋 FILES CREATED

### Execution Framework
- **main.py** - Orchestrator (V2 compatible, ~436 lines)
- **run.ps1** - PowerShell launcher ⭐ USE THIS
- **run.bat** - Batch launcher
- **config_v2.py** - V2 Configuration ⭐ RECOMMENDED

### Core Modules (V2 Compatible)
- **ai_data_preprocessor.py** - Stage 1 (✓ V2 updated)
- **ai_broadcast_decision_model_v2.py** - Stage 2 (✓ V2 native)
- **ai_inference_engine_v2.py** - Stage 3 (✓ V2 native)
- **ai_feedback_loop.py** - Stage 4 (✓ V2 updated)
- **ai_intent_parser.py** - Stage 5 (✓ V2 updated)

---

## 🎯 STAGE-BY-STAGE BREAKDOWN

### STAGE 1: Data Preprocessing & Augmentation (2-5 seconds)

**File:** `ai_data_preprocessor.py` (✓ V2 Updated)

**What It Does (V2 Enhanced):**
- Loads/generates 10,000 telemetry samples (50D features)
- **Validates data integrity** (NaN/Inf checking) ← V2 NEW
- Normalizes features using config method (standard/minmax/robust)
- Augments data (Gaussian noise, sensor drift, mixup, smoothing)
- Removes outliers (configurable threshold, default 5σ) ← V2 Enhanced
- Splits into train/val/test (70/15/15)
- **Tracks pipeline statistics** ← V2 NEW

**Key Output:**
- `X_train`, `X_val`, `X_test` - Normalized feature arrays (50D)
- `y_train`, `y_val`, `y_test` - Normalized output targets (5D with minmax)
- Pipeline statistics with sample counts

**Configuration Options** (in `config_v2.py`):
```python
cfg.data.input_scaler = "standard"        # "minmax" or "robust"
cfg.data.output_scaler = "minmax"         # Always minmax for sigmoid
cfg.data.outlier_threshold = 5.0          # Configurable threshold
cfg.data.normalize_inputs = True
cfg.data.normalize_outputs = True
```

**Run Individually:**
```bash
python ai_data_preprocessor.py
```

---

### STAGE 2: Neural Network Training (60-120 seconds)

**File:** `ai_broadcast_decision_model_v2.py` (✓ V2 Native)

**What It Does (V2 Enhanced):**
- Initializes BroadcastDecisionNetV2 with residual connections
- Trains with **AdamW** optimizer and **cosine annealing** scheduler (V2) ← Enhanced
- Implements **multi-task weighted loss** (V2) ← NEW
- Uses **Monte Carlo Dropout** for uncertainty quantification (V2) ← NEW
- Implements **gradient clipping** for stability (V2) ← NEW
- Monitors validation loss with early stopping
- Saves trained model + uncertainty estimates

**Network Architecture (V2):**
```
Input (50D)
  ↓
Dense Layer: 50 → 128 neurons + BatchNorm + ReLU
  ↓
Residual Blocks: 128 → 128 (V2) ← Enhanced architecture
  ↓
Dense Layer: 128 → 64 neurons + BatchNorm + ReLU
  ↓
Dense Layer: 64 → 32 neurons + BatchNorm + ReLU
  ↓
Multi-head Output: 5 separate output heads (V2) ← NEW
  ↓
Output (5D + Sigmoid): [redundancy, spectrum, availability, convergence, accuracy]
```

**Output Features (5D):**
- Redundancy Ratio: 1.0-5.0
- Spectrum (Mbps): 0.1-2.0
- Availability (%): 80-99%
- Convergence Time (sec): 10-60s
- Accuracy HPE (cm): 1-50cm

**Key Output:**
- Trained model: `models/broadcast_decision_model_v2/`
- Model weights: `model_weights.pt`
- Scalers: `scaler_X.pkl`, `scaler_y.pkl`
- Training history with uncertainty

**Configuration Options** (in `config_v2.py`):
```python
cfg.training.num_epochs = 200
cfg.training.batch_size = 32
cfg.training.learning_rate = 0.001
cfg.training.optimizer = "adamw"                    # V2: AdamW
cfg.training.scheduler = "cosine"                   # V2: Cosine annealing
cfg.training.gradient_clip_max_norm = 1.0           # V2: NEW
cfg.training.early_stopping_patience = 50
cfg.training.device = "cpu"                         # or "cuda"
cfg.model.use_residual = True                       # V2: Residual blocks
cfg.inference.mc_samples = 20                       # V2: MC Dropout samples
```

**Run Individually:**
```bash
python ai_broadcast_decision_model_v2.py
```

---

### STAGE 3: Real-Time Inference Engine (5-10 seconds)

**File:** `ai_inference_engine_v2.py` (✓ V2 Native)

**What It Does (V2 Enhanced):**
- Loads trained model from Stage 2
- Performs **Monte Carlo Dropout inference** for uncertainty (V2) ← NEW
- **ONNX backend support** for 3-5x faster inference (V2) ← NEW
- Implements **uncertainty-aware fallback policies** (V2) ← Enhanced
- Batch inference for multiple vehicles
- **Advanced health monitoring** (V2) ← NEW
- Tracks latency, confidence, AND uncertainty metrics (V2) ← NEW

**Key Classes:**
- `InferenceEngineV2` - Core inference engine with uncertainty
- `InferenceResult` - Complete result with metrics

**Key Output:**
- Broadcast decisions with confidence AND uncertainty (V2)
- Inference latency metrics (avg, p95, p99)
- Uncertainty estimates from MC Dropout
- Aggregated fleet decisions

**Configuration Options** (in `config_v2.py`):
```python
cfg.inference.model_dir = "models/broadcast_decision_model_v2"
cfg.inference.mc_samples = 20                       # V2: MC Dropout samples
cfg.inference.confidence_threshold = 0.7
cfg.inference.use_onnx = False                      # V2: ONNX backend
cfg.inference.device = "cpu"
cfg.inference.enable_fallback = True
cfg.inference.fallback_policy = "conservative"      # V2: Uncertainty-aware
cfg.inference.target_latency_ms = 50.0
```

**Example Usage (V2):**
```python
from ai_inference_engine_v2 import InferenceEngineV2
import numpy as np

# Initialize engine (with V2 features)
engine = InferenceEngineV2(
    mc_samples=20  # V2: Monte Carlo Dropout
)

# Single inference with uncertainty
telemetry = np.random.randn(50)
result = engine.infer(telemetry)

# Access V2 results with uncertainty
decision = result.broadcast_decision
print(f"Decision: {decision.to_dict()}")
print(f"Uncertainty: {decision.uncertainty:.4f}")  # V2 NEW
print(f"Confidence: {decision.confidence:.4f}")
print(f"Metrics: {result.metrics}")
```

**V2 Output:**
```json
{
  "broadcast_decision": {
    "redundancy_ratio": 2.5,
    "spectrum_mbps": 1.8,
    "availability_pct": 0.92,
    "convergence_time_sec": 35.0,
    "accuracy_hpe_cm": 5.0,
    "confidence": 0.88,
    "uncertainty": 0.12
  },
  "metrics": {
    "inference_time_ms": 12.5,
    "confidence": 0.88,
    "uncertainty": 0.12,
    "policy_applied": "model",
    "backend": "pytorch"
  },
  "timestamp": 1707177600.0
}
#   'confidence': 0.75
# }

# Get statistics
stats = engine.get_statistics()
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
print(f"Avg confidence: {stats['avg_confidence']:.3f}")
```

**Run Individually:**
```bash
python ai_inference_engine.py
```

---

### STAGE 4: Feedback Loop & Drift Detection (10-20 seconds)

**File:** `ai_feedback_loop.py` (✓ V2 Updated)

**What It Does (V2 Enhanced):**
- Collects real-time field telemetry from vehicles
- **Tracks model uncertainty** from inference engine (V2) ← NEW
- Buffers telemetry (up to 10,000 samples, configurable)
- Detects statistical drift with **uncertainty consideration** (V2) ← Enhanced
- **Uncertainty-aware drift recommendations** (V2) ← NEW
- Monitors KPIs (HPE, availability, convergence time)
- Triggers model retraining with better confidence

**Key Classes:**
- `TelemetryAggregator` - Buffers and aggregates field data
- `DriftDetector` - Statistical drift detection with uncertainty (V2)
- `PerformanceMonitor` - Tracks system KPIs
- `FeedbackLoop` - Orchestrates monitoring pipeline

**Drift Detection (V2 Enhanced):**
```python
cfg.feedback.buffer_size = 10000
cfg.feedback.drift_method = "zscore"                # V2: Configurable
cfg.feedback.drift_zscore_threshold = 3.0
cfg.feedback.drift_window_size = 1000
cfg.feedback.kpi_degradation_threshold = 0.1        # V2: NEW
cfg.feedback.drift_detection_enabled = True
```

**Key Output:**
- Drift detection with uncertainty estimates (V2)
- Recommendation: "retrain" if drift + high uncertainty
- Aggregated statistics with uncertainty tracking
- Performance history for alerting

**Example Usage (V2):**
```python
from ai_feedback_loop import FeedbackLoop, FieldTelemetry
import time

# Initialize feedback loop
loop = FeedbackLoop()

# Process field telemetry with V2 uncertainty
telemetry = FieldTelemetry(
    timestamp=time.time(),
    vehicle_id="vehicle_001",
    rtk_mode="FIX",
    actual_hpe_cm=4.5,
    actual_availability_pct=95.0,
    convergence_time_sec=32.0,
    num_satellites=15,
    signal_strength_avg_db=28.0,
    multipath_indicator=0.2,
    model_uncertainty=0.08,                # V2 NEW
    inference_confidence=0.92               # V2 NEW
)

# Process and get enhanced drift detection
result = loop.process_telemetry(telemetry)
if result and result.drift_detected:
    print(f"Drift in {result.metric_affected}")
    print(f"Uncertainty: {result.uncertainty:.4f}")  # V2 NEW
    print(f"Recommendation: {result.recommendation}")

# Get aggregated statistics with uncertainty
stats = loop.get_aggregated_statistics()
print(f"Avg HPE: {stats['telemetry']['hpe_mean_cm']:.2f}cm")
print(f"Availability: {stats['telemetry']['availability_mean_pct']:.1f}%")
```

**Run Individually:**
```bash
python ai_feedback_loop.py
```

### STAGE 5: Natural Language Intent Parser (5-15 seconds)

**File:** `ai_intent_parser.py` (✓ V2 Updated)

**What It Does (V2 Enhanced):**
- Parses operator intent from natural language
- Detects canonical intent type (3 types)
- **Validates extracted constraints** against ranges (V2) ← NEW
- Extracts numeric constraints from text
- Generates intent embeddings (32D)
- Confidence scoring based on keywords/phrases
- **Provides validation feedback** for invalid constraints (V2) ← NEW

**Supported Intent Types:**
```python
IntentType.MAXIMIZE_ACCURACY       # Sub-3cm HPE target
IntentType.MAXIMIZE_RELIABILITY    # 98%+ FIX availability
IntentType.OPTIMIZE_SPECTRUM       # Minimal bandwidth usage
```

**Constraint Validation Ranges (V2):**
```python
HPE:         [0.1, 100.0] cm
Availability: [50, 99.99] %
Spectrum:     [0.1, 10.0] Mbps
Convergence:  [1, 300] seconds
```

**Key Output:**
- Parsed intent type
- **Validated** constraints (V2) with status
- Intent embeddings (32D vectors)
- Confidence scores (0-1)
- Validation notes for failed constraints (V2)

**Example Usage (V2):**
```python
from ai_intent_parser import IntentParser

# Initialize parser
parser = IntentParser()

# Parse natural language intent
text = "I need sub-3cm accuracy for drone inspection at 95% availability"
result = parser.parse(text)

# Access V2 results with validation
print(result.intent_type.value)      # "maximize_accuracy"
print(result.confidence)              # 0.85
print(result.constraints.target_hpe_cm)        # 3.0
print(result.constraints.min_availability_pct) # 95.0
print(result.constraints.is_valid)             # True (V2)
print(result.constraints.validation_notes)     # "Valid" (V2)
print(result.intent_embedding.shape)  # (32,) - 32D embedding

# Serialized output includes validation (V2)
output = parser.to_dict(result)
print(output['constraints']['is_valid'])
print(output['constraints']['validation_notes'])
```

**Run Individually:**
```bash
python ai_intent_parser.py
```

## ⚙️ CONFIGURATION GUIDE

### Default Configuration (Production Quality, 2-3 min)
```python
# In config.py - TrainingConfig
device = "cpu"
num_epochs = 200
batch_size = 32
learning_rate = 0.001
early_stopping_patience = 50
```

### Fast Testing (30-50 sec)
```python
# For quick runs during development
device = "cpu"
num_epochs = 50
batch_size = 16
```

### GPU Optimized (30-60 sec, requires NVIDIA)
```python
# For fast training with GPU acceleration
device = "cuda"
num_epochs = 200
batch_size = 64
learning_rate = 0.001
```

### Production Extended (3-5 min, maximum accuracy)
```python
# For best accuracy in production
device = "cpu"
num_epochs = 500
learning_rate = 0.0005
early_stopping_patience = 100
```

**How to Customize:**
1. Edit `config.py` in any text editor
2. Modify the settings in TrainingConfig section
3. Save the file
4. Run `.\run.ps1`

---

## 📊 EXPECTED OUTPUTS

### Console Output
```
============================================================================
PPaaS AI SYSTEM - COMPLETE WORKFLOW ORCHESTRATION
Start Time: 2026-01-25 14:32:10
============================================================================

STAGE 1: DATA PREPROCESSING & AUGMENTATION
[INFO] Generated telemetry: shape=(10000, 50)
[INFO] Train: X=(14000, 50), y=(14000, 5)
[INFO] STAGE 1 COMPLETE

STAGE 2: NEURAL NETWORK TRAINING
[INFO] Training model for 200 epochs...
[INFO] Final training loss: 0.008234
[INFO] Final validation loss: 0.012567
[INFO] STAGE 2 COMPLETE

STAGE 3: INFERENCE ENGINE
[INFO] Inference time: 15.23ms
[INFO] Confidence: 0.756
[INFO] STAGE 3 COMPLETE

STAGE 4: FEEDBACK LOOP & DRIFT DETECTION
[INFO] Telemetry statistics:
[INFO]   Avg HPE: 4.56cm
[INFO]   Avg availability: 92.1%
[INFO] STAGE 4 COMPLETE

STAGE 5: INTENT PARSER
[INFO] Input: sub-3cm accuracy for drone...
[INFO] Intent Type: maximize_accuracy
[INFO] STAGE 5 COMPLETE

============================================================================
ALL STAGES COMPLETE - SYSTEM OPERATIONAL
Total execution time: 145.23s
============================================================================
```

### Files Created After Execution
- **ppaas_system.log** - Complete execution log
- **results/models/modelv1.pth** - Trained neural network model

---

## 🔧 RUNNING INDIVIDUAL STAGES

If you need to run stages separately:

```bash
# Stage 1 Only
python ai_data_preprocessor.py

# Stage 2 Only
python ai_broadcast_decision_model.py

# Stage 3 Only
python ai_inference_engine.py

# Stage 4 Only
python ai_feedback_loop.py

# Stage 5 Only
python ai_intent_parser.py

# All Together (RECOMMENDED)
python main.py
```

---

## 🐛 TROUBLESHOOTING

### Virtual Environment Issues
```
Error: Virtual environment not found
Solution:
  $ python -m venv .venv
  $ .venv\Scripts\Activate.ps1
```

### Missing Dependencies
```
Error: ModuleNotFoundError: No module named 'torch'
Solution:
  $ pip install numpy torch scikit-learn transformers
```

### Out of Memory
```
Error: CUDA out of memory (GPU)
Solution: Edit config.py
  device = "cpu"
  batch_size = 16
```

### GPU Not Detected
```
Error: CUDA device not found
Solution: Edit config.py
  device = "cpu"
```

### Slow Execution
```
Issue: Training takes too long
Solution: Edit config.py
  num_epochs = 50  (instead of 200)
  batch_size = 16
```

### Model Not Found
```
Error: Model file not found in Stage 3
Solution: Run complete workflow first
  $ python main.py
```

### Python Not Found
```
Error: 'python' is not recognized
Solution: Activate virtual environment
  $ .venv\Scripts\Activate.ps1
```


---

## 🎓 QUICK REFERENCE

| Need | Action |
|------|--------|
| Run system | `.\run.ps1` |
| Customize settings | Edit `config.py` |
| Run Stage 1 only | `python ai_data_preprocessor.py` |
| Run Stage 2 only | `python ai_broadcast_decision_model.py` |
| Run Stage 3 only | `python ai_inference_engine.py` |
| Run Stage 4 only | `python ai_feedback_loop.py` |
| Run Stage 5 only | `python ai_intent_parser.py` |
| Check logs | `Get-Content ppaas_system.log` |
| View model | `results/models/modelv1.pth` |

---

## 🚀 RECOMMENDED WORKFLOW

1. **Read** (2 min)
   - Review this file (sections you need)

2. **Customize** (Optional, 1 min)
   - Edit `config.py` if needed

3. **Run** (3 min)
   ```powershell
   .\run.ps1
   ```

4. **Monitor** (2-3 min)
   - Watch console output
   - Verify all stages show [OK]

5. **Verify** (1 min)
   ```powershell
   Get-Content ppaas_system.log | tail -20
   ```

6. **Use** (As needed)
   - Reference API examples above

---

## 📋 SUCCESS CRITERIA

After execution, verify:

- [ ] Console shows "ALL STAGES COMPLETE"
- [ ] All 5 stages marked [OK]
- [ ] Exit code = 0 (success)
- [ ] `ppaas_system.log` created
- [ ] `results/models/modelv1.pth` created
- [ ] Training loss < 0.02
- [ ] Validation loss < 0.02
- [ ] No error messages

---

## 🎯 NEXT STEPS

1. **Run Now:**
   ```powershell
   .\run.ps1
   ```

2. **Expected Result:**
   - System executes all 5 stages
   - Total time: 2-3 minutes
   - All stages marked [OK]

3. **Use Trained Model:**
   ```python
   from ai_inference_engine import InferenceEngine
   engine = InferenceEngine("results/models/modelv1.pth")
   ```

---

## 📞 SUPPORT

| Type | File |
|------|------|
| Architecture | ARCHITECTURE.md |
| Pre/Post Checklist | DEPLOYMENT_CHECKLIST.md |
| API Examples | QUICK_START.md |
| Detailed Specs | FINAL_REPORT.md |

---

