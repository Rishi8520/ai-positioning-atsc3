# ===============================================================================
# FILE: ai_feedback_loop.py
# MODULE: Online Learning & Feedback Loop for PPaaS
# AUTHOR: Tarunika D (AI/ML Systems)
# DATE: January 2026
# PURPOSE: Collect fleet telemetry, detect drift, trigger model retraining
# PRODUCTION: Phase 3 - Ready for Deployment
# V2 UPDATES: Config integration, uncertainty tracking, enhanced drift detection
# ===============================================================================

import logging
import time
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum

import numpy as np
from datetime import datetime

# V2 Configuration
try:
    from config_v2 import cfg
    USE_V2_CONFIG = True
except ImportError:
    USE_V2_CONFIG = False
    cfg = None

# ===============================================================================
# LOGGING
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# CONSTANTS (V2 compatible)
# ===============================================================================

# Get from v2 config if available
if USE_V2_CONFIG and cfg:
    TELEMETRY_BUFFER_SIZE = cfg.feedback.buffer_size
    DRIFT_CHECK_INTERVAL = max(100, cfg.feedback.drift_window_size // 10)
    DRIFT_THRESHOLD = cfg.feedback.drift_zscore_threshold / 10.0  # Convert z-score to normalized
    MODEL_VERSION_HISTORY_SIZE = 10
    DRIFT_METHOD = cfg.feedback.drift_method
    KPI_DEGRADATION_THRESHOLD = cfg.feedback.kpi_degradation_threshold
else:
    TELEMETRY_BUFFER_SIZE = 10000
    DRIFT_CHECK_INTERVAL = 100
    DRIFT_THRESHOLD = 0.15
    MODEL_VERSION_HISTORY_SIZE = 10
    DRIFT_METHOD = "zscore"
    KPI_DEGRADATION_THRESHOLD = 0.1

# ===============================================================================
# DATA CLASSES
# ===============================================================================

@dataclass
class FieldTelemetry:
    """Actual positioning performance from field (V2 enhanced)"""
    timestamp: float
    vehicle_id: str
    rtk_mode: str  # "STAND-ALONE", "FLOAT", "FIX"
    actual_hpe_cm: float  # Actual horizontal positioning error
    actual_vpe_cm: float  # Actual vertical positioning error
    actual_availability_pct: float
    convergence_time_sec: float
    num_satellites: int
    signal_strength_avg_db: float
    multipath_indicator: float
    model_uncertainty: Optional[float] = None  # NEW: Uncertainty from inference engine
    inference_confidence: Optional[float] = None  # NEW: Confidence from model


@dataclass
class DriftDetectionResult:
    """Result of drift detection (V2 enhanced)"""
    drift_detected: bool
    drift_magnitude: float  # Absolute drift amount
    metric_affected: str  # Which metric drifted
    recommendation: str  # "retrain" or "monitor"
    timestamp: float
    uncertainty: Optional[float] = None  # NEW: Uncertainty from MC Dropout
    drift_method: Optional[str] = None  # NEW: Which method detected drift


@dataclass
class ModelVersion:
    """Track model versions over time"""
    version_id: int
    timestamp: float
    training_samples: int
    val_loss: float
    test_accuracy: float
    performance_delta: float  # vs previous version


# ===============================================================================
# TELEMETRY AGGREGATOR
# ===============================================================================

class TelemetryAggregator:
    """Aggregate field telemetry from vehicles"""
    
    def __init__(self, buffer_size: int = TELEMETRY_BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)
        self.aggregation_windows = {}
        self.validation_stats = {
            'received': 0,
            'valid': 0,
            'invalid': 0
        }
        logger.info(f"TelemetryAggregator initialized with buffer size {buffer_size}")
    
    def add_telemetry(self, telemetry: FieldTelemetry) -> Optional[int]:
        """
        Add field telemetry to buffer with validation
        
        Args:
            telemetry: FieldTelemetry object
        
        Returns:
            Current buffer size if valid, None if rejected
        """
        self.validation_stats['received'] += 1
        
        if not self._validate_telemetry(telemetry):
            self.validation_stats['invalid'] += 1
            logger.warning(
                f"Rejected telemetry from {telemetry.vehicle_id}: "
                f"HPE={telemetry.actual_hpe_cm}cm, "
                f"Availability={telemetry.actual_availability_pct}%"
            )
            return None
        
        self.buffer.append(telemetry)
        self.validation_stats['valid'] += 1
        return len(self.buffer)
    
    def _validate_telemetry(self, telemetry: FieldTelemetry) -> bool:
        """
        Validate telemetry values are within reasonable ranges
        
        Args:
            telemetry: FieldTelemetry object
        
        Returns:
            True if valid, False otherwise
        """
        # Check HPE (horizontal positioning error)
        if telemetry.actual_hpe_cm < 0 or telemetry.actual_hpe_cm > 100:
            return False
        
        # Check VPE (vertical positioning error)
        if telemetry.actual_vpe_cm < 0 or telemetry.actual_vpe_cm > 100:
            return False
        
        # Check availability percentage
        if telemetry.actual_availability_pct < 0 or telemetry.actual_availability_pct > 100:
            return False
        
        # Check convergence time
        if telemetry.convergence_time_sec < 0 or telemetry.convergence_time_sec > 300:
            return False
        
        # Check satellite count
        if telemetry.num_satellites < 0 or telemetry.num_satellites > 40:
            return False
        
        # Check signal strength (dB)
        if telemetry.signal_strength_avg_db < -200 or telemetry.signal_strength_avg_db > 0:
            return False
        
        return True
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get telemetry validation statistics
        
        Returns:
            Dictionary with 'received', 'valid', 'invalid' counts
        """
        return self.validation_stats.copy()
    
    def get_window_statistics(self, window_size: int = 100) -> Dict:
        """
        Compute statistics over recent window
        
        Args:
            window_size: Number of recent samples
        
        Returns:
            Dictionary of aggregated statistics
        """
        if len(self.buffer) == 0:
            return {}
        
        recent = list(self.buffer)[-window_size:]
        
        hpe_values = [t.actual_hpe_cm for t in recent]
        availability_values = [t.actual_availability_pct for t in recent]
        convergence_values = [t.convergence_time_sec for t in recent]
        
        fix_count = sum(1 for t in recent if t.rtk_mode == "FIX")
        float_count = sum(1 for t in recent if t.rtk_mode == "FLOAT")
        
        return {
            "window_size": len(recent),
            "hpe_mean_cm": float(np.mean(hpe_values)),
            "hpe_std_cm": float(np.std(hpe_values)),
            "hpe_p95_cm": float(np.percentile(hpe_values, 95)),
            "availability_mean_pct": float(np.mean(availability_values)),
            "convergence_mean_sec": float(np.mean(convergence_values)),
            "fix_ratio": float(fix_count / len(recent)),
            "float_ratio": float(float_count / len(recent)),
            "num_vehicles": len(set(t.vehicle_id for t in recent)),
            "timestamp": time.time()
        }
    
    def export_buffer(self, output_path: str, num_samples: Optional[int] = None) -> int:
        """
        Export buffer to file for model retraining
        
        Args:
            output_path: Path to save JSON/CSV
            num_samples: Number of most recent samples to export
        
        Returns:
            Number of samples exported
        """
        if len(self.buffer) == 0:
            logger.warning("Buffer is empty, nothing to export")
            return 0
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get samples
        samples_to_export = list(self.buffer)
        if num_samples is not None:
            samples_to_export = samples_to_export[-num_samples:]
        
        # Export to JSON
        data_dicts = [asdict(t) for t in samples_to_export]
        
        with open(path, 'w') as f:
            json.dump(data_dicts, f, indent=2)
        
        logger.info(f"Exported {len(samples_to_export)} telemetry samples to {path}")
        return len(samples_to_export)


# ===============================================================================
# DRIFT DETECTION
# ===============================================================================

class DriftDetector:
    """Detect data/concept drift in positioning performance (V2 enhanced)"""
    
    def __init__(self, baseline_window_size: int = 100, detection_threshold: float = DRIFT_THRESHOLD):
        self.baseline_window_size = baseline_window_size
        self.detection_threshold = detection_threshold
        self.baseline_stats = {}
        self.drift_method = DRIFT_METHOD  # V2: configurable drift detection method
        logger.info(f"DriftDetector initialized with method={self.drift_method}")
    
    def set_baseline(self, telemetry_buffer: List[FieldTelemetry]) -> Dict:
        """
        Set baseline statistics from recent history
        
        Args:
            telemetry_buffer: List of recent telemetry samples
        
        Returns:
            Baseline statistics
        """
        recent = telemetry_buffer[-self.baseline_window_size:]
        
        if len(recent) == 0:
            logger.warning("Empty telemetry buffer, cannot set baseline")
            return {}
        
        hpe_values = np.array([t.actual_hpe_cm for t in recent])
        availability_values = np.array([t.actual_availability_pct for t in recent])
        convergence_values = np.array([t.convergence_time_sec for t in recent])
        
        # NEW: Include uncertainty statistics
        model_uncertainties = [t.model_uncertainty for t in recent if t.model_uncertainty is not None]
        
        self.baseline_stats = {
            "hpe_mean": float(np.mean(hpe_values)),
            "hpe_std": float(np.std(hpe_values)),
            "availability_mean": float(np.mean(availability_values)),
            "availability_std": float(np.std(availability_values)),
            "convergence_mean": float(np.mean(convergence_values)),
            "convergence_std": float(np.std(convergence_values)),
            "model_uncertainty_mean": float(np.mean(model_uncertainties)) if model_uncertainties else 0.0,
            "window_size": len(recent)
        }
        
        logger.info(f"Baseline statistics set from {len(recent)} samples")
        logger.info(f"  HPE: {self.baseline_stats['hpe_mean']:.2f} ± {self.baseline_stats['hpe_std']:.2f} cm")
        logger.info(f"  Availability: {self.baseline_stats['availability_mean']:.1f} ± {self.baseline_stats['availability_std']:.1f} %")
        
        return self.baseline_stats
    
    def detect_drift(self, recent_samples: List[FieldTelemetry]) -> DriftDetectionResult:
        """
        Detect drift in recent samples (V2 enhanced with uncertainty)
        
        Args:
            recent_samples: Recent telemetry samples to check
        
        Returns:
            DriftDetectionResult with uncertainty and method info
        """
        if not self.baseline_stats or len(recent_samples) < 10:
            return DriftDetectionResult(
                drift_detected=False,
                drift_magnitude=0.0,
                metric_affected="none",
                recommendation="monitor",
                timestamp=time.time(),
                uncertainty=0.0,
                drift_method=self.drift_method
            )
        
        # Check each metric
        drift_detected = False
        max_drift = 0.0
        affected_metric = "none"
        avg_uncertainty = 0.0
        
        # NEW: Average model uncertainty from recent samples
        uncertainties = [s.model_uncertainty for s in recent_samples if s.model_uncertainty is not None]
        if uncertainties:
            avg_uncertainty = float(np.mean(uncertainties))
        
        # HPE drift check
        recent_hpe = np.array([t.actual_hpe_cm for t in recent_samples])
        hpe_mean = np.mean(recent_hpe)
        hpe_std_baseline = self.baseline_stats["hpe_std"] + 1e-6
        hpe_drift = abs(hpe_mean - self.baseline_stats["hpe_mean"]) / hpe_std_baseline
        
        if hpe_drift > self.detection_threshold:
            drift_detected = True
            if hpe_drift > max_drift:
                max_drift = hpe_drift
                affected_metric = "hpe"
        
        # Availability drift check
        recent_avail = np.array([t.actual_availability_pct for t in recent_samples])
        avail_mean = np.mean(recent_avail)
        avail_std_baseline = self.baseline_stats["availability_std"] + 1e-6
        avail_drift = abs(avail_mean - self.baseline_stats["availability_mean"]) / avail_std_baseline
        
        if avail_drift > self.detection_threshold:
            drift_detected = True
            if avail_drift > max_drift:
                max_drift = avail_drift
                affected_metric = "availability"
        
        # Convergence drift check
        recent_conv = np.array([t.convergence_time_sec for t in recent_samples])
        conv_mean = np.mean(recent_conv)
        conv_std_baseline = self.baseline_stats["convergence_std"] + 1e-6
        conv_drift = abs(conv_mean - self.baseline_stats["convergence_mean"]) / conv_std_baseline
        
        if conv_drift > self.detection_threshold:
            drift_detected = True
            if conv_drift > max_drift:
                max_drift = conv_drift
                affected_metric = "convergence"
        
        # NEW: Consider uncertainty in recommendation
        recommendation = "retrain" if drift_detected else "monitor"
        if drift_detected and avg_uncertainty > 0.3:
            recommendation = "retrain"  # High uncertainty + drift = definite retrain
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_magnitude=max_drift,
            metric_affected=affected_metric,
            recommendation=recommendation,
            timestamp=time.time(),
            uncertainty=avg_uncertainty,
            drift_method=self.drift_method
        )
        
        if drift_detected:
            logger.warning(
                f"Drift detected in {affected_metric}: magnitude={max_drift:.3f}, "
                f"uncertainty={avg_uncertainty:.4f}, recommendation={recommendation}"
            )
        
        return result


# ===============================================================================
# PERFORMANCE MONITOR
# ===============================================================================

class PerformanceMonitor:
    """Monitor system performance over time"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.model_versions = deque(maxlen=MODEL_VERSION_HISTORY_SIZE)
        logger.info("PerformanceMonitor initialized")
    
    def record_inference(self, 
                        decision_accuracy: float,
                        positioning_error_cm: float,
                        inference_latency_ms: float) -> None:
        """
        Record inference performance
        
        Args:
            decision_accuracy: How well the decision predicted actual outcomes
            positioning_error_cm: Resulting positioning error
            inference_latency_ms: Latency of decision inference
        """
        record = {
            "timestamp": time.time(),
            "decision_accuracy": decision_accuracy,
            "positioning_error_cm": positioning_error_cm,
            "inference_latency_ms": inference_latency_ms
        }
        self.performance_history.append(record)
    
    def record_model_version(self,
                            version_id: int,
                            training_samples: int,
                            val_loss: float,
                            test_accuracy: float) -> None:
        """
        Record model version performance
        
        Args:
            version_id: Model version identifier
            training_samples: Number of training samples
            val_loss: Validation loss
            test_accuracy: Test set accuracy
        """
        # Calculate performance delta vs previous version
        performance_delta = 0.0
        if len(self.model_versions) > 0:
            prev_version = list(self.model_versions)[-1]
            performance_delta = test_accuracy - prev_version.test_accuracy
        
        version = ModelVersion(
            version_id=version_id,
            timestamp=time.time(),
            training_samples=training_samples,
            val_loss=val_loss,
            test_accuracy=test_accuracy,
            performance_delta=performance_delta
        )
        
        self.model_versions.append(version)
        logger.info(f"Recorded model version {version_id}: accuracy={test_accuracy:.4f}, delta={performance_delta:.4f}")
    
    def get_statistics(self) -> Dict:
        """Get overall performance statistics"""
        if len(self.performance_history) == 0:
            return {"error": "No performance history"}
        
        histories = list(self.performance_history)
        accuracies = np.array([h["decision_accuracy"] for h in histories])
        errors = np.array([h["positioning_error_cm"] for h in histories])
        latencies = np.array([h["inference_latency_ms"] for h in histories])
        
        return {
            "num_inferences": len(histories),
            "avg_decision_accuracy": float(np.mean(accuracies)),
            "avg_positioning_error_cm": float(np.mean(errors)),
            "avg_inference_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "model_versions_trained": len(self.model_versions)
        }


# ===============================================================================
# FEEDBACK LOOP ORCHESTRATOR
# ===============================================================================

class FeedbackLoop:
    """Orchestrate complete feedback loop"""
    
    def __init__(self, check_interval: int = DRIFT_CHECK_INTERVAL):
        self.aggregator = TelemetryAggregator()
        self.drift_detector = DriftDetector()
        self.performance_monitor = PerformanceMonitor()
        self.check_interval = check_interval
        self.sample_count = 0
        logger.info("FeedbackLoop orchestrator initialized")
    
    def process_telemetry(self, telemetry: FieldTelemetry) -> Optional[DriftDetectionResult]:
        """
        Process incoming field telemetry
        
        Args:
            telemetry: FieldTelemetry from vehicle
        
        Returns:
            DriftDetectionResult if drift check triggered, else None
        """
        self.aggregator.add_telemetry(telemetry)
        self.sample_count += 1
        
        # Periodic drift check
        if self.sample_count % self.check_interval == 0:
            logger.info(f"Periodic drift check at sample {self.sample_count}")
            
            # Set baseline on first check
            if not self.drift_detector.baseline_stats:
                self.drift_detector.set_baseline(list(self.aggregator.buffer))
                return None
            
            # Check for drift
            recent_samples = list(self.aggregator.buffer)[-self.check_interval:]
            drift_result = self.drift_detector.detect_drift(recent_samples)
            
            return drift_result
        
        return None
    
    def get_aggregated_statistics(self) -> Dict:
        """Get all aggregated statistics"""
        return {
            "telemetry": self.aggregator.get_window_statistics(window_size=100),
            "performance": self.performance_monitor.get_statistics(),
            "drift_detector_baseline": self.drift_detector.baseline_stats
        }
    
    def export_for_retraining(self, output_path: str) -> int:
        """Export telemetry buffer for model retraining"""
        return self.aggregator.export_buffer(output_path)


# ===============================================================================
# MAIN FOR TESTING
# ===============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("FEEDBACK LOOP - STANDALONE TEST")
    logger.info("=" * 80)
    
    # Initialize feedback loop
    loop = FeedbackLoop()
    
    # Simulate telemetry collection
    logger.info("\nSimulating telemetry collection...")
    for i in range(1000):
        telemetry = FieldTelemetry(
            timestamp=time.time(),
            vehicle_id=f"vehicle_{i % 10:02d}",
            rtk_mode=np.random.choice(["STAND-ALONE", "FLOAT", "FIX"]),
            actual_hpe_cm=np.random.normal(5.0, 2.0),
            actual_vpe_cm=np.random.normal(8.0, 3.0),
            actual_availability_pct=np.random.normal(92.0, 5.0),
            convergence_time_sec=np.random.normal(35.0, 10.0),
            num_satellites=np.random.randint(8, 20),
            signal_strength_avg_db=np.random.normal(25.0, 5.0),
            multipath_indicator=np.random.uniform(0.0, 1.0)
        )
        
        drift_result = loop.process_telemetry(telemetry)
        
        if drift_result:
            logger.info(f"Drift check result: {asdict(drift_result)}")
    
    # Get statistics
    logger.info("\nFinal Statistics:")
    stats = loop.get_aggregated_statistics()
    logger.info(json.dumps(stats, indent=2))
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
