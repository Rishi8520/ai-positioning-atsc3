"""
Configuration Module V2 for PPaaS AI System
Integrates with new model architecture and dataset
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Literal
from pathlib import Path
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
@dataclass
class DataConfig:
    """Data paths and preprocessing configuration"""
    
    # Dataset paths (relative to project root)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "DATA")
    
    # Dataset files
    train_features: str = "X_train.npy"
    train_targets: str = "y_train.npy"
    val_features: str = "X_val.npy"
    val_targets: str = "y_val.npy"
    test_features: str = "X_test.npy"
    test_targets: str = "y_test.npy"
    
    # Feature dimensions
    input_dim: int = 50
    output_dim: int = 5
    
    # Normalization
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    input_scaler: str = "standard"  # or "minmax"
    output_scaler: str = "minmax"   # MUST be minmax for sigmoid output
    
    # Data validation
    check_nan: bool = True
    check_range: bool = True
    outlier_threshold: float = 5.0

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    
    input_dim: int = 50
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    output_dim: int = 5
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    use_residual: bool = True
    activation: str = "relu"
    
    # Output ranges (real-world values)
    output_ranges: Dict = field(default_factory=lambda: {
        0: (1.0, 5.0),      # redundancy_ratio
        1: (0.1, 2.0),      # spectrum_mbps
        2: (0.80, 0.99),    # availability_pct
        3: (10.0, 60.0),    # convergence_time_sec
        4: (1.0, 50.0)      # accuracy_hpe_cm
    })
    
    # Target names
    output_names: List[str] = field(default_factory=lambda: [
        'redundancy_ratio',
        'spectrum_mbps',
        'availability_pct',
        'convergence_time_sec',
        'accuracy_hpe_cm'
    ])

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
@dataclass
class TrainingConfig:
    """Training loop parameters"""
    
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Optimizer
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_t_max: int = 200
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.0001
    
    # Gradient
    gradient_clip_max_norm: float = 1.0
    
    # Device
    device: str = "cpu"  # or "cuda"
    
    # Multi-task loss weights
    loss_weights: Dict = field(default_factory=lambda: {
        'redundancy': 1.0,
        'spectrum': 1.5,
        'availability': 2.0,
        'convergence': 1.0,
        'accuracy': 1.5
    })
    
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models")
    save_best_only: bool = True
    checkpoint_frequency: int = 10

# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================
@dataclass
class InferenceConfig:
    """Real-time inference parameters"""
    
    # Model path
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models" / "broadcast_decision_model_v2")
    
    # Backend
    use_onnx: bool = False
    onnx_model_path: str = "model.onnx"
    
    # Device
    device: str = "cpu"
    
    # Uncertainty quantification
    mc_samples: int = 20  # For Monte Carlo Dropout
    confidence_threshold: float = 0.7
    
    # Performance
    target_latency_ms: float = 50.0
    batch_size: int = 32
    
    # Fallback
    enable_fallback: bool = True
    fallback_policy: str = "conservative"  # or "balanced", "aggressive"

# ==============================================================================
# FEEDBACK LOOP CONFIGURATION
# ==============================================================================
@dataclass
class FeedbackConfig:
    """Feedback loop and drift detection parameters"""
    
    # Telemetry buffer
    buffer_size: int = 10000
    flush_interval_sec: int = 300
    storage_path: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "DATA" / "telemetry_archive")
    
    # Drift detection
    drift_detection_enabled: bool = True
    drift_method: str = "zscore"  # or "ks_test", "kl_divergence"
    drift_zscore_threshold: float = 3.0
    drift_window_size: int = 1000
    
    # KPI monitoring
    kpi_monitoring_enabled: bool = True
    kpi_degradation_threshold: float = 0.1  # 10% degradation
    
    # Retraining
    auto_retraining_enabled: bool = False  # Manual for now
    min_samples_for_retraining: int = 5000
    retraining_trigger_threshold: float = 0.15  # 15% performance drop

# ==============================================================================
# INTENT PARSER CONFIGURATION
# ==============================================================================
@dataclass
class IntentConfig:
    """Intent parsing configuration"""
    
    # Model
    transformer_model: str = "distilbert-base-uncased"
    embedding_dim: int = 32
    
    # Intent types
    supported_intents: List[str] = field(default_factory=lambda: [
        "maximize_accuracy",
        "maximize_reliability",
        "optimize_spectrum"
    ])
    
    # Confidence
    confidence_threshold: float = 0.6

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
@dataclass
class LoggingConfig:
    """Logging configuration"""
    
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path(__file__).parent / "logs" / "ppaas_ai.log")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance logging
    log_inference_time: bool = True
    log_predictions: bool = False  # Set to True for debugging

# ==============================================================================
# MASTER CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    """Master configuration class"""
    
    environment: Environment = Environment.DEVELOPMENT
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_environment(cls) -> "Config":
        """Load configuration from environment variables"""
        env = os.getenv("PPAAS_ENV", "development").lower()
        
        config = cls(environment=Environment(env))
        
        # Override from environment variables
        if os.getenv("LEARNING_RATE"):
            config.training.learning_rate = float(os.getenv("LEARNING_RATE"))
        
        if os.getenv("BATCH_SIZE"):
            config.training.batch_size = int(os.getenv("BATCH_SIZE"))
        
        if os.getenv("DEVICE"):
            config.training.device = os.getenv("DEVICE")
            config.inference.device = os.getenv("DEVICE")
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)
    
    def save(self, path: Path):
        """Save configuration to JSON"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from JSON"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# ==============================================================================
# GLOBAL CONFIG INSTANCE
# ==============================================================================
cfg = Config.from_environment()

if __name__ == "__main__":
    print("="*80)
    print("PPaaS AI System Configuration V2")
    print("="*80)
    print(f"Environment: {cfg.environment}")
    print(f"Data directory: {cfg.data.data_dir}")
    print(f"Model directory: {cfg.inference.model_dir}")
    print(f"Input dim: {cfg.model.input_dim}")
    print(f"Output dim: {cfg.model.output_dim}")
    print(f"Training epochs: {cfg.training.num_epochs}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Device: {cfg.training.device}")
    print(f"MC Dropout samples: {cfg.inference.mc_samples}")
    print("="*80)