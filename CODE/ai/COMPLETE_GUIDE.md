# PPaaS AI System - Complete Execution & Reference Guide



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

## 📊 5-Stage System Overview

Your system executes 5 stages automatically:

| Stage | Module | Time | Purpose |
|-------|--------|------|---------|
| **1** | Data Preprocessing | 2-5s | Load, normalize, augment, split data |
| **2** | Model Training | 60-120s | Train neural network, save model |
| **3** | Inference Engine | 5-10s | Test real-time inference |
| **4** | Feedback Loop | 10-20s | Detect drift, monitor KPIs |
| **5** | Intent Parser | 5-15s | NLP intent parsing |
| | **TOTAL** | **80-170s** | **2-3 minutes** |

---

## 📋 FILES CREATED

### Execution Framework (NEW)
- **main.py** - Orchestrator (300+ lines, runs all stages)
- **run.ps1** - PowerShell launcher ⭐ USE THIS
- **run.bat** - Batch launcher
- **config.py** - Configuration (CORRECTED)

### Core Modules (Validated)
- **ai_data_preprocessor.py** - Stage 1
- **ai_broadcast_decision_model.py** - Stage 2
- **ai_inference_engine.py** - Stage 3
- **ai_feedback_loop.py** - Stage 4
- **ai_intent_parser.py** - Stage 5

---

## 🎯 STAGE-BY-STAGE BREAKDOWN

### STAGE 1: Data Preprocessing & Augmentation (2-5 seconds)

**File:** `ai_data_preprocessor.py`

**What It Does:**
- Loads/generates 10,000 telemetry samples (50D features)
- Normalizes features using standard/minmax/robust scaling
- Augments data (Gaussian noise, sensor drift, mixup, smoothing)
- Removes outliers (3-sigma rule)
- Splits into train/val/test (70/15/15)

**Key Output:**
- `X_train`, `X_val`, `X_test` - Normalized feature arrays (50D)
- `y_train`, `y_val`, `y_test` - Normalized output targets (5D)

**Configuration Options** (in `config.py`):
```python
cfg.data.normalize_method = "standard"  # "minmax" or "robust"
cfg.data.train_ratio = 0.7
cfg.data.val_ratio = 0.15
cfg.data.test_ratio = 0.1
```

**Run Individually:**
```bash
python ai_data_preprocessor.py
```

---

### STAGE 2: Neural Network Training (60-120 seconds)

**File:** `ai_broadcast_decision_model.py`

**What It Does:**
- Initializes BroadcastDecisionNet neural network
- Trains with Adam optimizer and MSE loss
- Implements early stopping (monitors validation loss)
- Saves trained model to `results/models/modelv1.pth`
- Evaluates on test set

**Network Architecture:**
```
Input (50D)
  ↓
Dense Layer: 50 → 128 neurons + ReLU
  ↓
Dense Layer: 128 → 64 neurons + ReLU
  ↓
Dense Layer: 64 → 32 neurons + ReLU
  ↓
Output Layer: 32 → 5 neurons + Sigmoid
  ↓
Output (5D): [redundancy_ratio, spectrum_mbps, availability_pct, 
              convergence_time_sec, accuracy_hpe_cm]
```

**Output Features (5D):**
- Redundancy Ratio: 1.0-5.0
- Spectrum (Mbps): 0.1-2.0
- Availability (%): 80-99%
- Convergence Time (sec): 10-60s
- Accuracy HPE (cm): 1-50cm

**Key Output:**
- Trained model: `results/models/modelv1.pth`
- Training loss history
- Validation metrics

**Configuration Options** (in `config.py`):
```python
cfg.training.num_epochs = 200           # Reduce for faster testing
cfg.training.batch_size = 32            # Reduce if out of memory
cfg.training.learning_rate = 0.001
cfg.training.early_stopping_patience = 50
cfg.training.device = "cpu"             # or "cuda" for GPU
```

**Run Individually:**
```bash
python ai_broadcast_decision_model.py
```

---

### STAGE 3: Real-Time Inference Engine (5-10 seconds)

**File:** `ai_inference_engine.py`

**What It Does:**
- Loads trained model from Stage 2
- Performs single-sample real-time inference
- Implements fallback policies (conservative/balanced/aggressive)
- Batch inference for multiple vehicles
- Tracks latency and confidence metrics

**Key Classes:**
- `InferenceEngine` - Core inference engine
- `BatchInferenceProcessor` - Queue-based batch processing

**Key Output:**
- Broadcast decisions with confidence scores
- Inference latency metrics (avg, p95, p99)
- Aggregated fleet decisions

**Configuration Options** (in `config.py`):
```python
cfg.inference.model_path = "results/models/modelv1.pth"
cfg.inference.confidence_threshold = 0.6
cfg.inference.target_latency_ms = 50    # p95 target
cfg.inference.fallback_policy = "conservative_defaults"
```

**Example Usage:**
```python
from ai_inference_engine import InferenceEngine
import numpy as np

# Initialize engine
engine = InferenceEngine("results/models/modelv1.pth")

# Single inference
telemetry = np.random.randn(50)  # 50D feature vector
result = engine.infer(telemetry)

# Access results
print(result.broadcast_decision.to_dict())
# Output: {
#   'redundancy_ratio': 2.5,
#   'spectrum_mbps': 1.8,
#   'availability_pct': 0.92,
#   'convergence_time_sec': 35.0,
#   'accuracy_hpe_cm': 5.0,
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

**File:** `ai_feedback_loop.py`

**What It Does:**
- Collects real-time field telemetry from vehicles
- Buffers telemetry (up to 10,000 samples)
- Detects statistical drift in positioning performance
- Monitors KPIs (HPE, availability, convergence time)
- Triggers model retraining recommendations

**Key Classes:**
- `TelemetryAggregator` - Buffers and aggregates field data
- `DriftDetector` - Statistical drift detection (z-score based)
- `PerformanceMonitor` - Tracks system KPIs
- `FeedbackLoop` - Orchestrates monitoring pipeline

**Drift Detection Thresholds:**
```python
cfg.feedback.drift_zscore_threshold = 3.0
cfg.feedback.drift_window_size = 1000
cfg.feedback.drift_detection_enabled = True
```

**Key Output:**
- Drift detection results (metric affected, magnitude, recommendation)
- Aggregated statistics (HPE mean/std, availability, convergence)
- Performance history for alerting

**Example Usage:**
```python
from ai_feedback_loop import FeedbackLoop, FieldTelemetry
import time

# Initialize feedback loop
loop = FeedbackLoop()

# Process field telemetry
telemetry = FieldTelemetry(
    timestamp=time.time(),
    vehicle_id="vehicle_001",
    rtk_mode="FIX",
    actual_hpe_cm=4.5,
    actual_availability_pct=95.0,
    convergence_time_sec=32.0,
    num_satellites=15,
    signal_strength_avg_db=28.0,
    multipath_indicator=0.2
)

# Process and get drift detection
result = loop.process_telemetry(telemetry)
if result and result.drift_detected:
    print(f"Drift in {result.metric_affected}: {result.recommendation}")

# Get aggregated statistics
stats = loop.get_aggregated_statistics()
print(f"Avg HPE: {stats['telemetry']['hpe_mean_cm']:.2f}cm")
print(f"Availability: {stats['telemetry']['availability_mean_pct']:.1f}%")
```

**Run Individually:**
```bash
python ai_feedback_loop.py
```

---

### STAGE 5: Natural Language Intent Parser (5-15 seconds)

**File:** `ai_intent_parser.py`

**What It Does:**
- Parses operator intent from natural language
- Detects canonical intent type (3 types)
- Extracts numeric constraints from text
- Generates intent embeddings (32D)
- Confidence scoring based on keywords/phrases

**Supported Intent Types:**
```python
IntentType.MAXIMIZE_ACCURACY       # Sub-3cm HPE target
IntentType.MAXIMIZE_RELIABILITY    # 98%+ FIX availability
IntentType.OPTIMIZE_SPECTRUM       # Minimal bandwidth usage
```

**Key Output:**
- Parsed intent type
- Extracted constraints (HPE, availability, spectrum, convergence)
- Intent embeddings (32D vectors)
- Confidence scores (0-1)

**Example Usage:**
```python
from ai_intent_parser import IntentParser

# Initialize parser
parser = IntentParser()

# Parse natural language intent
text = "I need sub-3cm accuracy for drone inspection at 95% availability"
result = parser.parse(text)

# Access results
print(result.intent_type.value)      # "maximize_accuracy"
print(result.confidence)              # 0.85
print(result.constraints.target_hpe_cm)        # 3.0
print(result.constraints.min_availability_pct) # 95.0
print(result.intent_embedding.shape)  # (32,) - 32D embedding
```

**Run Individually:**
```bash
python ai_intent_parser.py
```

---

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

