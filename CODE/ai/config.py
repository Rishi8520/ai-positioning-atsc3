"""
Configuration module for Intent-Driven Broadcast Decision AI System
All parameters externalized, environment-aware, zero hardcoding
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    input_dim: int = 50  # 50-dimensional features
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    output_dim: int = 5  # 5 broadcast control outputs
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"
    weight_init: str = "kaiming_uniform"
    bias_init: float = 0.0


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training loop parameters"""
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    loss_fn: str = "mse"
    
    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.0001
    
    # Scheduler
    scheduler_type: str = "cosine"
    scheduler_t_max: int = 200
    
    # Gradient
    gradient_clip_max_norm: float = 1.0
    
    # Validation
    validation_split: float = 0.2
    
    # Checkpointing
    checkpoint_frequency: int = 10
    keep_best_checkpoints: int = 3
    
    # Device
    device: str = "cpu"  # Use "cuda" for GPU, "cpu" for CPU
    
    # Targets
    target_train_loss: float = 0.01
    target_val_loss: float = 0.015


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Data preprocessing parameters"""
    # File paths
    telemetry_path: str = "data/telemetryphase1.jsonl"
    metrics_path: str = "data/channelmetrics.csv"
    
    # Processing
    feature_dim: int = 50
    normalize_method: str = "minmax"  # or "standard"
    outlier_threshold: float = 5.0  # 5-sigma
    
    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_seed: int = 42
    
    # Storage
    storage_format: str = "hdf5"  # or "numpy", "pickle"
    storage_path: str = "results/data_cache.h5"
    
    # Missing value handling
    missing_value_strategy: str = "forward_fill"  # or "interpolate", "drop"
    
    # Expected sample counts
    expected_telemetry_samples: int = 10000
    expected_metrics_samples: int = 1000


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfig:
    """Real-time inference parameters"""
    # Backend
    use_onnx: bool = False  # False = PyTorch, True = ONNX Runtime
    model_path: str = "results/models/modelv1.pth"
    onnx_model_path: str = "results/models/modelv1.onnx"
    
    # Device
    device: str = "cpu"  # Use "cuda" for GPU, "cpu" for CPU
    
    # Confidence
    confidence_method: str = "heuristic"  # or "bayesian", "ensemble"
    confidence_threshold: float = 0.5
    
    # Performance targets
    target_latency_ms: int = 50  # p95 latency target
    target_throughput_inferences_per_sec: int = 100
    
    # Fallback policies: "conservative_defaults", "aggressive_robust", "spectrum_efficient"
    fallback_policy: str = "conservative_defaults"
    
    # Quantization
    use_quantization: bool = True
    quantization_type: str = "int8"
    
    # Batch processing
    max_batch_size: int = 128
    auto_batch: bool = True


# ============================================================================
# FEEDBACK & ONLINE LEARNING CONFIGURATION
# ============================================================================

@dataclass
class FeedbackConfig:
    """Feedback loop and online learning parameters"""
    # Telemetry
    telemetry_buffer_size: int = 10000
    telemetry_flush_interval_sec: int = 300
    telemetry_storage_path: str = "results/telemetry_archive.h5"
    
    # Drift detection
    drift_detection_enabled: bool = True
    drift_detection_method: str = "zscore"  # or "ks_test", "covariate_shift"
    drift_zscore_threshold: float = 3.0
    drift_window_size: int = 1000
    
    # Performance monitoring
    kpi_monitoring_enabled: bool = True
    kpi_degradation_threshold: float = 0.1  # 10% degradation triggers alert
    
    # Model versioning
    version_storage_path: str = "results/models/versions"
    max_versions_to_keep: int = 10
    
    # Online learning
    online_learning_enabled: bool = True
    retraining_schedule: str = "daily"  # or "weekly", "manual"
    min_samples_for_retraining: int = 1000
    retraining_batch_size: int = 32
    retraining_num_epochs: int = 20
    
    # Safety
    safe_rollback_enabled: bool = True
    pre_deployment_validation: bool = True


# ============================================================================
# API CONFIGURATION
# ============================================================================

@dataclass
class APIConfig:
    """FastAPI server parameters"""
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Workers
    num_workers: int = 4
    worker_timeout_sec: int = 300
    
    # Performance
    rate_limit_requests_per_sec: int = 1000
    max_request_size_mb: int = 10
    request_timeout_sec: int = 60
    
    # Features
    enable_cors: bool = True
    enable_prometheus: bool = True
    enable_health_checks: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file_path: str = "results/logs/api.log"


# ============================================================================
# SCENARIO CONFIGURATION
# ============================================================================

@dataclass
class ScenarioConfig:
    """3 deployment scenarios with specific targets"""
    
    @dataclass
    class Scenario:
        name: str
        canonical_intent: str
        target_hpe_cm: float
        target_convergence_time_sec: int
        target_fix_availability_pct: float
        target_spectrum_mbps: float
        broadcast_redundancy: float
        broadcast_update_freq_hz: float
        broadcast_tile_resolution_deg: float
        broadcast_fec_overhead_pct: float
    
    scenarios: List[Scenario] = field(default_factory=lambda: [
        ScenarioConfig.Scenario(
            name="high_accuracy_rural_drone",
            canonical_intent="provide_sub_3cm_accuracy",
            target_hpe_cm=3.0,
            target_convergence_time_sec=120,
            target_fix_availability_pct=95.0,
            target_spectrum_mbps=5.0,
            broadcast_redundancy=2.0,
            broadcast_update_freq_hz=5.0,
            broadcast_tile_resolution_deg=1.0,
            broadcast_fec_overhead_pct=40
        ),
        ScenarioConfig.Scenario(
            name="low_bandwidth_spectrum_optimized",
            canonical_intent="optimize_atsc_spectrum_usage",
            target_hpe_cm=15.0,
            target_convergence_time_sec=300,
            target_fix_availability_pct=80.0,
            target_spectrum_mbps=2.5,
            broadcast_redundancy=1.0,
            broadcast_update_freq_hz=0.5,
            broadcast_tile_resolution_deg=0.25,
            broadcast_fec_overhead_pct=10
        ),
        ScenarioConfig.Scenario(
            name="urban_canyon_robust_continuity",
            canonical_intent="guarantee_service_continuity",
            target_hpe_cm=10.0,
            target_convergence_time_sec=180,
            target_fix_availability_pct=85.0,
            target_spectrum_mbps=3.5,
            broadcast_redundancy=1.8,
            broadcast_update_freq_hz=2.0,
            broadcast_tile_resolution_deg=0.9,
            broadcast_fec_overhead_pct=30
        )
    ])


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

@dataclass
class DeploymentConfig:
    """Model export and deployment parameters"""
    # Export targets
    export_pytorch: bool = True
    export_onnx: bool = True
    export_quantized: bool = True
    
    # Paths
    pytorch_export_path: str = "results/models/modelv1.pth"
    onnx_export_path: str = "results/models/modelv1.onnx"
    quantized_export_path: str = "results/models/modelv1_int8.onnx"
    
    # ONNX settings
    onnx_opset_version: int = 14
    onnx_optimize: bool = True
    
    # Quantization
    quantization_type: str = "int8"
    quantization_calibration_samples: int = 100
    
    # Docker
    docker_image_name: str = "ai-positioning:1.0"
    docker_registry: str = "localhost"


# ============================================================================
# COMPOSITE CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config:
    """Master configuration class"""
    environment: Environment = Environment.PRODUCTION
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    api: APIConfig = field(default_factory=APIConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    @classmethod
    def from_environment(cls) -> "Config":
        """Load configuration from environment"""
        env = os.getenv("ENV", "production").lower()
        cfg = cls(environment=Environment(env))
        
        # Override from environment variables
        if os.getenv("LEARNING_RATE"):
            cfg.training.learning_rate = float(os.getenv("LEARNING_RATE"))
        if os.getenv("BATCH_SIZE"):
            cfg.training.batch_size = int(os.getenv("BATCH_SIZE"))
        if os.getenv("USE_ONNX"):
            cfg.inference.use_onnx = os.getenv("USE_ONNX").lower() == "true"
        
        return cfg
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================

cfg = Config.from_environment()

if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"Environment: {cfg.environment}")
    print(f"Model input dim: {cfg.model.input_dim}")
    print(f"Training epochs: {cfg.training.num_epochs}")
    print(f"Inference latency target: {cfg.inference.target_latency_ms}ms")
    print(f"API workers: {cfg.api.num_workers}")
    print(f"Scenarios: {len(cfg.scenario.scenarios)}")
