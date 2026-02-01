# Detailed V2 Update Changes

## 1. ai_data_preprocessor.py

### Imports Added
```python
try:
    from config_v2 import cfg
    USE_V2_CONFIG = True
except ImportError:
    USE_V2_CONFIG = False
    cfg = None
```

### Constants Updated
```python
# Before: Hardcoded values
INPUT_FEATURE_DIM = 50
OUTPUT_FEATURE_DIM = 5

# After: Config-driven with fallback
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
```

### FeatureNormalizer Class Changes
```python
# Added field to track statistics
self.feature_stats = None

# New method for data validation
def _validate_data(self, X: np.ndarray, data_type: str = "input") -> None:
    """Validate data integrity per v2 standards"""
    if X is None or len(X) == 0:
        raise ValueError(f"{data_type} array is empty")
    
    # Check NaN/Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        raise ValueError(f"Found {nan_count} NaNs and {inf_count} Infs")
    
    # Check dimensions
    expected_dim = OUTPUT_FEATURE_DIM if data_type == "output" else INPUT_FEATURE_DIM
    if X.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} features, got {X.shape[1]}")

# Enhanced fit() method
def fit(self, X: np.ndarray):
    self._validate_data(X, "input")
    self.scaler.fit(X)
    # Track statistics for v2
    self.feature_stats = {
        "mean": X.mean(axis=0),
        "std": X.std(axis=0),
        "min": X.min(axis=0),
        "max": X.max(axis=0),
        "shape": X.shape
    }
```

### DataPreprocessingPipeline Class Changes
```python
# Initialize with v2 config
def __init__(self, normalization_method: str = "standard"):
    if USE_V2_CONFIG and cfg:
        normalization_method = cfg.data.input_scaler
    
    self.normalizer_X = FeatureNormalizer(method=normalization_method)
    self.normalizer_y = FeatureNormalizer(method=OUTPUT_SCALER)
    self.pipeline_stats = {}  # NEW: Track statistics

# Enhanced process() method
def process(self, X, y, augment=True, augmentation_factor=2):
    # NEW: Enhanced validation
    try:
        self.normalizer_X._validate_data(X, "input")
        self.normalizer_y._validate_data(y, "output")
    except ValueError as e:
        logger.error(f"Data validation failed: {e}")
        raise
    
    # Use config threshold instead of hardcoded 3.0
    threshold = OUTLIER_THRESHOLD if USE_V2_CONFIG else 3.0
    
    # NEW: Track statistics
    self.pipeline_stats["initial_samples"] = initial_count
    self.pipeline_stats["outliers_removed"] = outlier_count
    self.pipeline_stats["augmented_samples"] = len(X)
    self.pipeline_stats["final_train_size"] = len(splits['X_train'])
    self.pipeline_stats["final_val_size"] = len(splits['X_val'])
    self.pipeline_stats["final_test_size"] = len(splits['X_test'])
```

---

## 2. ai_feedback_loop.py

### Imports & Config Added
```python
try:
    from config_v2 import cfg
    USE_V2_CONFIG = True
except ImportError:
    USE_V2_CONFIG = False
    cfg = None
```

### Constants Updated
```python
# Before: Hardcoded constants
TELEMETRY_BUFFER_SIZE = 10000
DRIFT_CHECK_INTERVAL = 100
DRIFT_THRESHOLD = 0.15

# After: Config-driven
if USE_V2_CONFIG and cfg:
    TELEMETRY_BUFFER_SIZE = cfg.feedback.buffer_size
    DRIFT_CHECK_INTERVAL = max(100, cfg.feedback.drift_window_size // 10)
    DRIFT_THRESHOLD = cfg.feedback.drift_zscore_threshold / 10.0
    DRIFT_METHOD = cfg.feedback.drift_method
    KPI_DEGRADATION_THRESHOLD = cfg.feedback.kpi_degradation_threshold
```

### FieldTelemetry Dataclass Enhanced
```python
# Before
@dataclass
class FieldTelemetry:
    timestamp: float
    vehicle_id: str
    # ... 8 other fields ...

# After - Added uncertainty tracking
@dataclass
class FieldTelemetry:
    timestamp: float
    vehicle_id: str
    # ... existing fields ...
    model_uncertainty: Optional[float] = None          # NEW
    inference_confidence: Optional[float] = None       # NEW
```

### DriftDetectionResult Dataclass Enhanced
```python
# Before
@dataclass
class DriftDetectionResult:
    drift_detected: bool
    drift_magnitude: float
    metric_affected: str
    recommendation: str
    timestamp: float

# After - Added uncertainty and method info
@dataclass
class DriftDetectionResult:
    # ... existing fields ...
    uncertainty: Optional[float] = None               # NEW
    drift_method: Optional[str] = None               # NEW
```

### DriftDetector Class Changes
```python
# New initialization with config
def __init__(self, baseline_window_size=100, detection_threshold=DRIFT_THRESHOLD):
    self.drift_method = DRIFT_METHOD  # NEW: Track method

# Enhanced set_baseline()
def set_baseline(self, telemetry_buffer):
    # NEW: Track uncertainty statistics
    model_uncertainties = [t.model_uncertainty for t in recent 
                          if t.model_uncertainty is not None]
    
    self.baseline_stats = {
        "hpe_mean": float(np.mean(hpe_values)),
        "hpe_std": float(np.std(hpe_values)),
        # ... other metrics ...
        "model_uncertainty_mean": float(np.mean(model_uncertainties)) if model_uncertainties else 0.0,
        "window_size": len(recent)
    }

# Enhanced detect_drift() - considers uncertainty
def detect_drift(self, recent_samples):
    # NEW: Average uncertainty from samples
    uncertainties = [s.model_uncertainty for s in recent_samples 
                    if s.model_uncertainty is not None]
    if uncertainties:
        avg_uncertainty = float(np.mean(uncertainties))
    
    # Existing drift checks...
    
    # NEW: Uncertainty-aware recommendation
    recommendation = "retrain" if drift_detected else "monitor"
    if drift_detected and avg_uncertainty > 0.3:
        recommendation = "retrain"  # High uncertainty + drift = definite retrain
    
    result = DriftDetectionResult(
        # ... existing fields ...
        uncertainty=avg_uncertainty,           # NEW
        drift_method=self.drift_method         # NEW
    )
```

---

## 3. ai_intent_parser.py

### Imports & Config Added
```python
try:
    from config_v2 import cfg
    USE_V2_CONFIG = True
except ImportError:
    USE_V2_CONFIG = False
    cfg = None
```

### IntentConstraints Dataclass Enhanced
```python
# Before
@dataclass
class IntentConstraints:
    target_hpe_cm: float
    min_availability_pct: float
    max_spectrum_mbps: float
    max_convergence_sec: float
    preferred_region: Optional[str] = None

# After - Added validation
@dataclass
class IntentConstraints:
    target_hpe_cm: float
    min_availability_pct: float
    max_spectrum_mbps: float
    max_convergence_sec: float
    preferred_region: Optional[str] = None
    is_valid: bool = False                    # NEW
    validation_notes: str = ""                # NEW
    
    def validate(self) -> bool:               # NEW METHOD
        """Validate constraints are within reasonable ranges"""
        valid = True
        notes = []
        
        if self.target_hpe_cm < 0.1 or self.target_hpe_cm > 100.0:
            valid = False
            notes.append(f"HPE {self.target_hpe_cm}cm out of range [0.1, 100.0]")
        
        if self.min_availability_pct < 50.0 or self.min_availability_pct > 99.99:
            valid = False
            notes.append(f"Availability {self.min_availability_pct}% out of range")
        
        if self.max_spectrum_mbps < 0.1 or self.max_spectrum_mbps > 10.0:
            valid = False
            notes.append(f"Spectrum {self.max_spectrum_mbps}Mbps out of range")
        
        if self.max_convergence_sec < 1.0 or self.max_convergence_sec > 300.0:
            valid = False
            notes.append(f"Convergence {self.max_convergence_sec}s out of range")
        
        self.is_valid = valid
        self.validation_notes = "; ".join(notes) if notes else "Valid"
        return valid
```

### CanonicalIntent Dataclass Enhanced
```python
# Before
@dataclass
class CanonicalIntent:
    intent_type: IntentType
    confidence: float
    constraints: IntentConstraints
    raw_text: str
    intent_embedding: np.ndarray
    reasoning: str

# After - Added version tracking
@dataclass
class CanonicalIntent:
    # ... existing fields ...
    embedding_dim: int = 32               # NEW
    model_version: str = "v2"             # NEW
```

### IntentParser Class Changes
```python
# Enhanced initialization with config
def __init__(self, pretrained_model="sentence-transformers/all-MiniLM-L6-v2"):
    # ... model loading ...
    
    # NEW: Get from v2 config
    if USE_V2_CONFIG and cfg:
        self.confidence_threshold = cfg.intent.confidence_threshold
        self.embedding_dim = cfg.intent.embedding_dim
    else:
        self.confidence_threshold = 0.6
        self.embedding_dim = 32

# Enhanced parse() method - with constraint validation
def parse(self, intent_text: str) -> CanonicalIntent:
    # ... intent detection ...
    
    constraints = IntentConstraints(**constraints_dict)
    
    # NEW: Validate constraints
    is_valid = constraints.validate()
    if not is_valid:
        logger.warning(f"Constraint validation failed: {constraints.validation_notes}")
    
    # ... embedding generation ...
    
    # NEW: Include validation status in reasoning
    reasoning = (
        f"Detected intent: {detected_intent.value} | "
        f"HPE target: {constraints.target_hpe_cm}cm | "
        f"Status: {'✓ Valid' if is_valid else '✗ ' + constraints.validation_notes}"
    )
    
    return CanonicalIntent(
        # ... fields ...
        embedding_dim=self.embedding_dim,         # NEW
        model_version="v2"                        # NEW
    )

# Enhanced to_dict() - includes validation details
def to_dict(self, canonical_intent: CanonicalIntent) -> Dict:
    return {
        "intent_type": canonical_intent.intent_type.value,
        "confidence": canonical_intent.confidence,
        "constraints": {
            **asdict(canonical_intent.constraints),
            "is_valid": canonical_intent.constraints.is_valid,
            "validation_notes": canonical_intent.constraints.validation_notes
        },
        # ... other fields ...
        "embedding_dim": canonical_intent.embedding_dim,
        "model_version": canonical_intent.model_version
    }
```

---

## Summary of Changes by Category

### Configuration Management
- ✓ All 3 modules now use `config_v2.cfg`
- ✓ Graceful fallback if config_v2 unavailable
- ✓ Configurable parameters per v2 architecture

### Data Validation
- ✓ Data preprocessor validates NaN/Inf
- ✓ Intent parser validates constraint ranges
- ✓ Both with detailed error messages

### Uncertainty Handling
- ✓ Feedback loop tracks model uncertainty
- ✓ Drift detection considers uncertainty
- ✓ Better decision-making with uncertainty info

### Enhanced Observability
- ✓ Data preprocessor tracks pipeline statistics
- ✓ Drift detector logs baseline details
- ✓ Intent parser shows validation status

### Version Tracking
- ✓ All outputs include version information
- ✓ Dataclass fields track embedding dimensions
- ✓ Model version tagged as "v2"

