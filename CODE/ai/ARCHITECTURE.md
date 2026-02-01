# System Architecture & Data Flow (V2)

**Version:** 2.0 (Updated February 2026)  
**Status:** Production Ready with Enhanced Features

## Key V2 Enhancements

- **Config-Driven Design**: All parameters configurable via `config_v2.cfg`
- **Uncertainty Quantification**: Monte Carlo Dropout provides uncertainty estimates
- **Enhanced Validation**: Data & constraint validation at each stage
- **Improved Monitoring**: Better observability with statistics tracking
- **Better Error Handling**: Graceful fallbacks and detailed error messages

## Complete Execution Flow

```
START
  │
  ├─────────────────────────────────────────────┐
  │                                             │
  ▼                                             │
┌─────────────────────────────────────────────┐ │
│  STAGE 1: DATA PREPROCESSING (2-5s)         │ │
│                                             │ │
│  Input: Raw Telemetry (10,000 samples)     │ │
│    ↓                                        │ │
│  • Load/Generate data (50D features)        │ │
│  • Normalize (standard/minmax/robust)       │ │
│  • Augment (noise, drift, mixup)            │ │
│  • Remove outliers (3-sigma)                │ │
│  • Split (train:70%, val:15%, test:15%)     │ │
│    ↓                                        │ │
│  Output: Preprocessed Data                  │ │
│    - X_train (10,500 × 50)                  │ │
│    - X_val (2,250 × 50)                     │ │
│    - X_test (2,250 × 50)                    │ │
└────────────┬────────────────────────────────┘ │
             │                                  │
             ▼                                  │
┌─────────────────────────────────────────────┐ │
│  STAGE 2: NEURAL NETWORK TRAINING (60-120s)│ │
│                                             │ │
│  Input: Preprocessed Data                   │ │
│    ↓                                        │ │
│  BroadcastDecisionNet                       │ │
│  ├─ Input Layer:    50D                     │ │
│  ├─ Hidden 1:     128 neurons (ReLU)        │ │
│  ├─ Residual Blocks: Skip connections      │ │ (V2: Enhanced)
│  ├─ Hidden 2:      64 neurons (ReLU)        │ │
│  ├─ Hidden 3:      32 neurons (ReLU)        │ │
│  └─ Output Layer:    5D (Sigmoid)           │ │
│    ↓                                        │ │
│  Training Loop (200 epochs)                 │ │
│  ├─ Optimizer: AdamW (V2: Improved)        │ │
│  ├─ Loss: Multi-task weighted MSE (V2)     │ │ (V2: Enhanced)
│  ├─ Early Stopping: patience=50 epochs      │ │
│  ├─ Gradient Clipping (V2: Stability)      │ │ (V2: New)
│  └─ Batch Size: 32 samples                  │ │
│    ↓                                        │ │
│  Output: Trained Model (V2: With Uncertainty) │ │ (V2: Enhanced)
│    - File: models/broadcast_decision_model_v2/│ │
│    - Includes: Uncertainty estimates       │ │
│    - Loss: Lower with residual blocks      │ │
└────────────┬────────────────────────────────┘ │
             │                                  │
             ├───────────────────────────────┐  │
             │                               │  │
             ▼                               ▼  │
    ┌──────────────────────────┐   ┌──────────────────┐
    │  STAGE 3: INFERENCE       │   │ STAGE 4: FEEDBACK│
    │  (5-10 seconds)           │   │ (10-20 seconds)  │
    │                           │   │                  │
    │ InferenceEngineV2         │   │ FeedbackLoop(V2) │
    │ ├─ Load Model (V2)        │   │ ├─ Collect       │
    │ ├─ MC Dropout Sampling    │   │ │  Telemetry    │
    │ ├─ Uncertainty Quantif.   │   │ ├─ Uncertainty   │
    │ ├─ ONNX Backend Support   │   │ │  Tracking (V2) │
    │ ├─ Batch Inference        │   │ ├─ Enhanced Drift│
    │ ├─ Fallback Policies      │   │ │  Detection (V2)│
    │ └─ Confidence Scoring     │   │ ├─ KPI Monitor   │
    │                           │   │ └─ Statistics    │
    │ Output:                   │   │                  │
    │ - Decisions + Uncertainty │   │ Output:          │
    │ - Latency (ms)            │   │ - Drift Results  │
    │ - Confidence (0-1)        │   │ - Uncertainties  │
    │ - Uncertainty (V2)        │   │ - Aggregates     │
    └──────────┬────────────────┘   └────────┬─────────┘
               │                             │
               └─────────────┬───────────────┘
                             │
                             ▼
            ┌────────────────────────────┐
            │ STAGE 5: INTENT PARSER     │
            │ (5-15 seconds)             │
            │                            │
            │ IntentParser (V2)          │
            │ ├─ Load Transformer        │
            │ ├─ Parse NLP Input         │
            │ ├─ Detect Intent Type      │
            │ ├─ Validate Constraints(V2)│ (V2: New)
            │ ├─ Extract Constraints     │
            │ └─ Generate Embeddings(32D)│
            │                            │
            │ Output:                    │
            │ - Intent Type              │
            │ - Constraints (Validated)  │
            │ - Confidence               │
            │ - Embeddings (32D)         │
            │ - Validation Status (V2)   │
            └────────────┬───────────────┘
                         │
                         ▼
                    ┌─────────────┐
                    │   SUCCESS   │
                    │ (STAGE 1-5) │
                    └─────────────┘
                         │
                         ↓
                    END (OK)
```

---

## Data Dimensions

```
STAGE 1: Data Preprocessing
├─ Input:  Telemetry CSV → 10,000 samples
├─ Process: Normalize + Augment → 14,000 samples
└─ Output: 
   ├─ X_train: (10,500, 50)   ← 50D feature vectors
   ├─ y_train: (10,500, 5)    ← 5D targets
   ├─ X_val:   (2,250, 50)
   ├─ y_val:   (2,250, 5)
   ├─ X_test:  (2,250, 50)
   └─ y_test:  (2,250, 5)

STAGE 2: Neural Network Training
├─ Model Inputs:  (batch_size=32, 50)
├─ Layer 1:       (batch_size=32, 128)
├─ Layer 2:       (batch_size=32, 64)
├─ Layer 3:       (batch_size=32, 32)
└─ Model Output:  (batch_size=32, 5)

STAGE 3: Inference
├─ Input:  Single telemetry sample (50,)
├─ Process: Model forward pass
├─ Output: 
│  ├─ Decision: 5D vector (redundancy, spectrum, availability, convergence, accuracy)
│  ├─ Confidence: scalar (0.0-1.0)
│  └─ Latency: scalar (milliseconds)
└─ Batch Output: List of decisions

STAGE 4: Feedback Loop
├─ Input: Field telemetry (500 samples)
├─ Aggregation: Deque buffer (max 10,000)
└─ Output:
   ├─ Drift detection: Boolean + magnitude
   ├─ Statistics: Dict of aggregated metrics
   └─ Alerts: Text recommendations

STAGE 5: Intent Parser
├─ Input: Natural language text
├─ Process: NLP pipeline
└─ Output:
   ├─ Intent type: Enum (MAXIMIZE_ACCURACY, etc.)
   ├─ Confidence: scalar (0.0-1.0)
   ├─ Constraints: Dict with thresholds
   └─ Embeddings: 32D vector
```

---

## Configuration Flow

```
User edits config.py
        │
        ▼
┌──────────────────────────────────────┐
│   Config.from_environment()          │
│                                      │
│   Loads:                             │
│   ├─ ModelConfig                     │
│   ├─ TrainingConfig                  │
│   ├─ DataConfig                      │
│   ├─ InferenceConfig                 │
│   ├─ FeedbackConfig                  │
│   └─ APIConfig                       │
│                                      │
│   Device: cpu / cuda                 │
│   Epochs: 200 / 50 / 500             │
│   Batch: 32 / 16 / 64                │
│   Learning Rate: 0.001               │
│   Early Stop: patience=50             │
└──────────┬───────────────────────────┘
           │
           ├─→ Stage 1: DataConfig
           ├─→ Stage 2: TrainingConfig
           ├─→ Stage 3: InferenceConfig
           ├─→ Stage 4: FeedbackConfig
           └─→ Stage 5: (uses transformer model)
```

---

## Feature Dimensions

### Input Features (50D)
```
Telemetry Features:
├─ Signal Strength (4D):     [ss0, ss1, ss2, ss3]
├─ Carrier Phase (4D):       [cp0, cp1, cp2, cp3]
├─ Pseudorange Error (4D):   [pr0, pr1, pr2, pr3]
├─ Doppler Shift (4D):       [ds0, ds1, ds2, ds3]
├─ Tracking Lock (4D):       [tl0, tl1, tl2, tl3]
├─ Received Power (4D):      [rp0, rp1, rp2, rp3]
├─ Carrier Power (4D):       [cp0, cp1, cp2, cp3]
├─ Noise Power (2D):         [np0, np1]
├─ SNR (4D):                 [snr0, snr1, snr2, snr3]
├─ CNR (4D):                 [cnr0, cnr1, cnr2, cnr3]
├─ Multipath (2D):           [mp0, mp1]
└─ Environment (10D):        [urban_density, blockage, multipath, shadow_fading, 
                              nlos_prob, tunnel_prob, time_of_day, 
                              vehicle_speed, heading, gnss_availability]

Total: 4+4+4+4+4+4+4+2+4+4+2+10 = 50D
```

### Output Features (5D)
```
Broadcast Decision Parameters:
├─ Redundancy Ratio:         (1.0 - 5.0)    [0.2 - 1.0 normalized]
├─ Spectrum (Mbps):          (0.1 - 2.0)    [0.05 - 1.0 normalized]
├─ Availability (%):         (80% - 99%)    [0.8 - 0.99]
├─ Convergence Time (sec):   (10 - 60s)     [0.167 - 1.0 normalized]
└─ Accuracy HPE (cm):        (1 - 50cm)     [0.02 - 1.0 normalized]

Total: 5D (after sigmoid activation)
```

---

## Execution Timeline

```
Timeline:  0s ─────────────────────────────────────── 170s
           │                                           │
Stage 1:   ├─(2-5s)─┤                                  │
           │                                           │
Stage 2:   │        ├────────(60-120s)────────┤       │
           │                                   │       │
Stage 3:   │                                   ├─(5-10s)┤
           │                                   │       │
Stage 4:   │                                   │   ├──(10-20s)┤
           │                                   │   │   │
Stage 5:   │                                   │   ├─(5-15s)┤
           │                                   │   │   │
Logging:   ├──────────────────────────────────┼───┼───┤
           │                                   │   │   │
END:       │                                   │   │   ├─ COMPLETE
           0s────────────────────────────────80s──170s
           
CPU:       [IDLE]  [ACTIVE] [MODERATE] [LOW] [ACTIVE]
GPU:       [IDLE/OFF unless device="cuda"]
Memory:    500MB   1GB      700MB      400MB 600MB
```

---

## System Dependencies

```
┌─────────────────────────────┐
│   Python 3.12+              │
└────────────┬────────────────┘
             │
    ┌────────┴────────┬─────────────────┬──────────────────┐
    │                 │                 │                  │
    ▼                 ▼                 ▼                  ▼
┌─────────┐  ┌──────────────┐  ┌──────────────────┐  ┌───────────┐
│ NumPy   │  │ PyTorch      │  │ scikit-learn     │  │ Transformers
│ 1.x+    │  │ 2.x+         │  │ 1.x+             │  │ 4.x+ 
│         │  │              │  │                  │  │
│ Arrays  │  │ Neural Nets  │  │ Preprocessing    │  │ HuggingFace
│ Linear  │  │ Training     │  │ Model Selection  │  │ NLP Models
│ Algebra │  │ GPU/CPU      │  │ Scaling          │  │ Embeddings
└─────────┘  └──────────────┘  └──────────────────┘  └───────────┘
    │              │                    │                  │
    └──────────────┼────────────────────┼──────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  main.py         │
         │ (Orchestrator)   │
         └──────────────────┘
              │
    ┌─────────┼─────────────────┬─────────────────┬─────────────┐
    │         │                 │                 │             │
    ▼         ▼                 ▼                 ▼             ▼
 Stage 1   Stage 2           Stage 3          Stage 4       Stage 5
```

---

## Error Handling Flow

```
                        ┌──────────────┐
                        │  main.py     │
                        │  (runs all   │
                        │   stages)    │
                        └──────┬───────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
          ┌───────────┐  ┌───────────┐  ┌──────────┐
          │ Try Block │  │ Try Block │  │ Try Blk  │
          │ (Stage 1) │  │ (Stage 2) │  │(Stage N) │
          └───────────┘  └───────────┘  └──────────┘
                │              │              │
         ┌──────┴──────┐       │              │
         │             │       │              │
         │        ┌────┴───────┴──────────────┤
         │        │                          │
         ▼        ▼                          ▼
       SUCCESS  EXCEPTION                NO ERROR
         │        │                         │
         │        └─→ Log Error             │
         │            Print Error           │
         │            Raise Exception  ┌────┴────────┐
         │                             │  CONTINUE   │
         │                             │ to next     │
         ▼                             │ stage or    │
      [CONTINUE]                       │ EXIT with   │
                                       │ code=1      │
                                       └─────────────┘
```

---

## Monitoring Points

```
During Execution:

Stage 1:  "Generating 10000 synthetic telemetry samples"
          "Train: X=(14000, 50), y=(14000, 5)"
          "STAGE 1 COMPLETE"

Stage 2:  "Training model for 200 epochs..."
          "Epoch 10/200 - Loss: 0.234"
          "Epoch 50/200 - Loss: 0.098"
          "Epoch 200/200 - Loss: 0.012"
          "STAGE 2 COMPLETE"

Stage 3:  "Inference time: 15.23ms"
          "Confidence: 0.756"
          "STAGE 3 COMPLETE"

Stage 4:  "Telemetry statistics:"
          "  Avg HPE: 4.56cm"
          "  Avg availability: 92.1%"
          "STAGE 4 COMPLETE"

Stage 5:  "Input: sub-3cm accuracy for drone..."
          "Intent Type: maximize_accuracy"
          "Confidence: 0.85"
          "STAGE 5 COMPLETE"

Final:    "ALL STAGES COMPLETE - SYSTEM OPERATIONAL"
          "Total execution time: 145.23s"
```

---

## Success Criteria

```
✓ Stage 1: Data splits created correctly
✓ Stage 2: Model trained with loss < 0.02
✓ Stage 3: Inference latency < 50ms (p95)
✓ Stage 4: Drift detection operational
✓ Stage 5: Intent parsing working
✓ Exit Code: 0 (success)
✓ Log File: ppaas_system.log created
✓ Model File: results/models/modelv1.pth created
✓ Console Output: "ALL STAGES COMPLETE"
```

---

**System Architecture Diagram Complete**
All stages validated and documented.
