#!/usr/bin/env python3
# filepath: /media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE/broadcast/broadcast_controller.py
"""
ATSC 3.0 Broadcast Controller

PURPOSE:
Central controller that manages broadcast pipeline configuration based on
AI Orchestrator commands or traditional static settings. Acts as the bridge
between the AI decision layer and the physical broadcast pipeline.

KEY RESPONSIBILITIES:
1. Receive and validate AI broadcast commands (intent-driven)
2. Translate high-level intents to concrete broadcast parameters
3. Apply configurations to the broadcast pipeline
4. Collect and report metrics for AI feedback loop
5. Support dynamic reconfiguration during operation
6. Maintain configuration history for analysis

INTEGRATION:
- Receives commands from: ai/orchestrator.py
- Controls: broadcast/pipeline.py
- Reports metrics to: AI Orchestrator, logging system
- Used by: gnss/rtcm_to_broadcast_handoff.py

AI BROADCAST COMMAND FORMAT (from technical doc):
{
    "command_id": "uuid",
    "timestamp": "ISO-8601",
    "intent": "maximize_reliability | minimize_bandwidth | maximize_accuracy",
    "broadcast_config": {
        "fec_ldpc_rate": "RATE_6_15",
        "fec_rs_symbols": 24,
        "fec_overhead_pct": 40.0,
        "modulation": "QPSK",
        "fft_size": "FFT_8K",
        "guard_interval": "GI_1_4",
        "update_frequency_hz": 5.0,
        "tile_resolution": "high",
        "plp_mode": "mobile"
    },
    "constraints": {
        "max_latency_ms": 100,
        "min_reliability_pct": 99.0,
        "max_bandwidth_hz": 6000000
    },
    "priority": "high | medium | low"
}

USAGE:
    from broadcast.broadcast_controller import BroadcastController
    
    # Initialize controller
    controller = BroadcastController()
    
    # Apply AI command
    result = controller.apply_command(ai_command)
    
    # Process data with current config
    broadcast_result = controller.process(rtcm_data)
    
    # Get metrics for AI feedback
    metrics = controller.get_metrics()

MODES:
    - TRADITIONAL: Static configuration, no dynamic adaptation
    - AI_CONTROLLED: Dynamic configuration from AI Orchestrator
    - HYBRID: AI suggestions with operator override capability
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import broadcast modules
try:
    from broadcast.pipeline import BroadcastPipeline, BroadcastConfig, BroadcastResult
    from broadcast.config import (
        FECCodeRate,
        ModulationScheme,
        FFTSize,
        GuardInterval,
        PilotPattern
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = str(e)

# Optional: Import decoder for verification
try:
    from broadcast.decoder import BroadcastDecoder, DecoderConfig
    DECODER_AVAILABLE = True
except ImportError:
    DECODER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ControllerMode(Enum):
    """Operating mode for the broadcast controller."""
    TRADITIONAL = auto()   # Static configuration
    AI_CONTROLLED = auto() # Dynamic AI-driven configuration
    HYBRID = auto()        # AI suggestions with operator override


class Intent(Enum):
    """High-level operational intents from AI Orchestrator."""
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    MINIMIZE_BANDWIDTH = "minimize_bandwidth"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    BALANCE_ALL = "balance_all"
    LOW_LATENCY = "low_latency"
    URBAN_ROBUST = "urban_robust"
    RURAL_EFFICIENT = "rural_efficient"


class CommandPriority(Enum):
    """Priority levels for AI commands."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class ConfigurationStatus(Enum):
    """Status of configuration application."""
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"


class PLPMode(Enum):
    """Physical Layer Pipe mode."""
    FIXED = "fixed"       # Static allocation
    MOBILE = "mobile"     # Optimized for mobile reception
    INDOOR = "indoor"     # Optimized for indoor reception
    ROBUST = "robust"     # Maximum robustness


class TileResolution(Enum):
    """Resolution for correction tiles/bitmaps."""
    LOW = "low"           # Coarse grid, less bandwidth
    MEDIUM = "medium"     # Balanced
    HIGH = "high"         # Fine grid, more bandwidth
    ADAPTIVE = "adaptive" # Varies based on region


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class BroadcastConstraints:
    """Constraints for broadcast configuration."""
    
    # Latency constraints
    max_latency_ms: float = 1000.0
    target_latency_ms: float = 100.0
    
    # Reliability constraints
    min_reliability_pct: float = 95.0
    target_reliability_pct: float = 99.0
    
    # Bandwidth constraints
    max_bandwidth_hz: float = 6_000_000  # 6 MHz
    target_bandwidth_hz: float = 3_000_000
    
    # Power constraints
    max_power_dbm: float = 30.0
    
    # FEC constraints
    max_fec_overhead_pct: float = 70.0
    min_fec_overhead_pct: float = 10.0
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate constraints are reasonable."""
        errors = []
        
        if self.max_latency_ms < self.target_latency_ms:
            errors.append("max_latency_ms must be >= target_latency_ms")
        
        if self.min_reliability_pct > self.target_reliability_pct:
            errors.append("min_reliability_pct must be <= target_reliability_pct")
        
        if self.max_bandwidth_hz < self.target_bandwidth_hz:
            errors.append("max_bandwidth_hz must be >= target_bandwidth_hz")
        
        if not (0 <= self.min_reliability_pct <= 100):
            errors.append("reliability must be between 0 and 100")
        
        if self.max_fec_overhead_pct < self.min_fec_overhead_pct:
            errors.append("max_fec_overhead must be >= min_fec_overhead")
        
        return len(errors) == 0, errors


@dataclass
class AIBroadcastCommand:
    """Command from AI Orchestrator to configure broadcast."""
    
    # Command identification
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Intent and priority
    intent: Intent = Intent.BALANCE_ALL
    priority: CommandPriority = CommandPriority.MEDIUM
    
    # Broadcast configuration
    fec_ldpc_rate: str = "RATE_8_15"
    fec_rs_symbols: int = 16
    fec_overhead_pct: float = 15.0
    modulation: str = "QPSK"
    fft_size: str = "FFT_8K"
    guard_interval: str = "GI_1_8"
    
    # Additional parameters
    update_frequency_hz: float = 1.0
    tile_resolution: TileResolution = TileResolution.MEDIUM
    plp_mode: PLPMode = PLPMode.FIXED
    
    # Constraints
    constraints: BroadcastConstraints = field(default_factory=BroadcastConstraints)
    
    # Metadata
    source: str = "ai_orchestrator"
    scenario: str = ""
    reason: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIBroadcastCommand':
        """Create command from dictionary."""
        cmd = cls()
        
        # Basic fields
        cmd.command_id = data.get("command_id", cmd.command_id)
        cmd.timestamp = data.get("timestamp", cmd.timestamp)
        cmd.source = data.get("source", cmd.source)
        cmd.scenario = data.get("scenario", cmd.scenario)
        cmd.reason = data.get("reason", cmd.reason)
        
        # Intent
        intent_str = data.get("intent", "balance_all")
        try:
            cmd.intent = Intent(intent_str.lower())
        except ValueError:
            cmd.intent = Intent.BALANCE_ALL
        
        # Priority
        priority_str = data.get("priority", "medium")
        priority_map = {
            "critical": CommandPriority.CRITICAL,
            "high": CommandPriority.HIGH,
            "medium": CommandPriority.MEDIUM,
            "low": CommandPriority.LOW
        }
        cmd.priority = priority_map.get(priority_str.lower(), CommandPriority.MEDIUM)
        
        # Broadcast config
        bc = data.get("broadcast_config", data)
        cmd.fec_ldpc_rate = bc.get("fec_ldpc_rate", cmd.fec_ldpc_rate)
        cmd.fec_rs_symbols = bc.get("fec_rs_symbols", cmd.fec_rs_symbols)
        cmd.fec_overhead_pct = bc.get("fec_overhead_pct", cmd.fec_overhead_pct)
        cmd.modulation = bc.get("modulation", cmd.modulation)
        cmd.fft_size = bc.get("fft_size", cmd.fft_size)
        cmd.guard_interval = bc.get("guard_interval", cmd.guard_interval)
        cmd.update_frequency_hz = bc.get("update_frequency_hz", cmd.update_frequency_hz)
        
        # Tile resolution
        tile_str = bc.get("tile_resolution", "medium")
        try:
            cmd.tile_resolution = TileResolution(tile_str.lower())
        except ValueError:
            cmd.tile_resolution = TileResolution.MEDIUM
        
        # PLP mode
        plp_str = bc.get("plp_mode", "fixed")
        try:
            cmd.plp_mode = PLPMode(plp_str.lower())
        except ValueError:
            cmd.plp_mode = PLPMode.FIXED
        
        # Constraints
        constraints_data = data.get("constraints", {})
        cmd.constraints = BroadcastConstraints(
            max_latency_ms=constraints_data.get("max_latency_ms", 1000.0),
            target_latency_ms=constraints_data.get("target_latency_ms", 100.0),
            min_reliability_pct=constraints_data.get("min_reliability_pct", 95.0),
            target_reliability_pct=constraints_data.get("target_reliability_pct", 99.0),
            max_bandwidth_hz=constraints_data.get("max_bandwidth_hz", 6_000_000),
            max_fec_overhead_pct=constraints_data.get("max_fec_overhead_pct", 70.0),
            min_fec_overhead_pct=constraints_data.get("min_fec_overhead_pct", 10.0)
        )
        
        return cmd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert command to dictionary."""
        return {
            "command_id": self.command_id,
            "timestamp": self.timestamp,
            "intent": self.intent.value,
            "priority": self.priority.name.lower(),
            "broadcast_config": {
                "fec_ldpc_rate": self.fec_ldpc_rate,
                "fec_rs_symbols": self.fec_rs_symbols,
                "fec_overhead_pct": self.fec_overhead_pct,
                "modulation": self.modulation,
                "fft_size": self.fft_size,
                "guard_interval": self.guard_interval,
                "update_frequency_hz": self.update_frequency_hz,
                "tile_resolution": self.tile_resolution.value,
                "plp_mode": self.plp_mode.value
            },
            "constraints": asdict(self.constraints),
            "source": self.source,
            "scenario": self.scenario,
            "reason": self.reason
        }


@dataclass
class CommandResult:
    """Result of applying a command."""
    
    command_id: str
    status: ConfigurationStatus
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Applied configuration
    applied_config: Optional[Dict[str, Any]] = None
    
    # Validation results
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # Modifications made
    parameters_modified: List[str] = field(default_factory=list)
    overrides_applied: List[str] = field(default_factory=list)
    
    # Timing
    application_time_ms: float = 0.0
    
    # Error info
    error_message: str = ""


@dataclass
class ControllerMetrics:
    """Metrics collected by the broadcast controller."""
    
    # Timing
    collection_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    uptime_seconds: float = 0.0
    
    # Command statistics
    commands_received: int = 0
    commands_applied: int = 0
    commands_rejected: int = 0
    commands_failed: int = 0
    
    # Current configuration
    current_mode: str = "traditional"
    current_intent: str = "balance_all"
    current_fec_rate: str = "RATE_8_15"
    current_modulation: str = "QPSK"
    current_fec_overhead_pct: float = 15.0
    
    # Processing statistics
    data_processed_bytes: int = 0
    frames_transmitted: int = 0
    
    # Performance metrics
    avg_processing_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    estimated_spectral_efficiency: float = 0.0
    
    # Quality metrics
    estimated_ber: float = 0.0
    estimated_reliability_pct: float = 99.0
    
    # Resource usage
    cpu_usage_pct: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_ai_feedback(self) -> Dict[str, Any]:
        """Format metrics as AI feedback."""
        return {
            "timestamp": self.collection_timestamp,
            "performance": {
                "latency_ms": self.avg_latency_ms,
                "processing_time_ms": self.avg_processing_time_ms,
                "spectral_efficiency": self.estimated_spectral_efficiency
            },
            "quality": {
                "ber": self.estimated_ber,
                "reliability_pct": self.estimated_reliability_pct
            },
            "throughput": {
                "bytes_processed": self.data_processed_bytes,
                "frames_transmitted": self.frames_transmitted
            },
            "configuration": {
                "mode": self.current_mode,
                "intent": self.current_intent,
                "fec_rate": self.current_fec_rate,
                "modulation": self.current_modulation,
                "fec_overhead_pct": self.current_fec_overhead_pct
            }
        }


@dataclass
class ConfigurationHistoryEntry:
    """Entry in configuration history."""
    
    timestamp: str
    command_id: str
    intent: str
    config: Dict[str, Any]
    status: str
    metrics_snapshot: Optional[Dict[str, Any]] = None


# ============================================================================
# CONFIGURATION TRANSLATOR
# ============================================================================


class ConfigTranslator:
    """
    Translates between different configuration formats.
    
    Handles conversion between:
    - AI command format (strings, high-level)
    - BroadcastConfig format (enums, concrete)
    - JSON/dict format (serializable)
    """
    
    # String to enum mappings
    FEC_RATE_MAP = {
        "RATE_2_15": FECCodeRate.RATE_2_15 if PIPELINE_AVAILABLE else None,
        "RATE_3_15": FECCodeRate.RATE_3_15 if PIPELINE_AVAILABLE else None,
        "RATE_4_15": FECCodeRate.RATE_4_15 if PIPELINE_AVAILABLE else None,
        "RATE_5_15": FECCodeRate.RATE_5_15 if PIPELINE_AVAILABLE else None,
        "RATE_6_15": FECCodeRate.RATE_6_15 if PIPELINE_AVAILABLE else None,
        "RATE_7_15": FECCodeRate.RATE_7_15 if PIPELINE_AVAILABLE else None,
        "RATE_8_15": FECCodeRate.RATE_8_15 if PIPELINE_AVAILABLE else None,
        "RATE_9_15": FECCodeRate.RATE_9_15 if PIPELINE_AVAILABLE else None,
        "RATE_10_15": FECCodeRate.RATE_10_15 if PIPELINE_AVAILABLE else None,
        "RATE_11_15": FECCodeRate.RATE_11_15 if PIPELINE_AVAILABLE else None,
        "RATE_12_15": FECCodeRate.RATE_12_15 if PIPELINE_AVAILABLE else None,
        "RATE_13_15": FECCodeRate.RATE_13_15 if PIPELINE_AVAILABLE else None,
        "2/15": FECCodeRate.RATE_2_15 if PIPELINE_AVAILABLE else None,
        "6/15": FECCodeRate.RATE_6_15 if PIPELINE_AVAILABLE else None,
        "8/15": FECCodeRate.RATE_8_15 if PIPELINE_AVAILABLE else None,
    }
    
    MODULATION_MAP = {
        "QPSK": ModulationScheme.QPSK if PIPELINE_AVAILABLE else None,
        "QAM16": ModulationScheme.QAM16 if PIPELINE_AVAILABLE else None,
        "QAM64": ModulationScheme.QAM64 if PIPELINE_AVAILABLE else None,
        "QAM256": ModulationScheme.QAM256 if PIPELINE_AVAILABLE else None,
        "16QAM": ModulationScheme.QAM16 if PIPELINE_AVAILABLE else None,
        "64QAM": ModulationScheme.QAM64 if PIPELINE_AVAILABLE else None,
        "256QAM": ModulationScheme.QAM256 if PIPELINE_AVAILABLE else None,
    }
    
    FFT_SIZE_MAP = {
        "FFT_8K": FFTSize.FFT_8K if PIPELINE_AVAILABLE else None,
        "FFT_16K": FFTSize.FFT_16K if PIPELINE_AVAILABLE else None,
        "FFT_32K": FFTSize.FFT_32K if PIPELINE_AVAILABLE else None,
        "8K": FFTSize.FFT_8K if PIPELINE_AVAILABLE else None,
        "16K": FFTSize.FFT_16K if PIPELINE_AVAILABLE else None,
        "32K": FFTSize.FFT_32K if PIPELINE_AVAILABLE else None,
    }
    
    GUARD_INTERVAL_MAP = {
        "GI_1_4": GuardInterval.GI_1_4 if PIPELINE_AVAILABLE else None,
        "GI_1_8": GuardInterval.GI_1_8 if PIPELINE_AVAILABLE else None,
        "GI_1_16": GuardInterval.GI_1_16 if PIPELINE_AVAILABLE else None,
        "GI_1_32": GuardInterval.GI_1_32 if PIPELINE_AVAILABLE else None,
        "1/4": GuardInterval.GI_1_4 if PIPELINE_AVAILABLE else None,
        "1/8": GuardInterval.GI_1_8 if PIPELINE_AVAILABLE else None,
        "1/16": GuardInterval.GI_1_16 if PIPELINE_AVAILABLE else None,
        "1/32": GuardInterval.GI_1_32 if PIPELINE_AVAILABLE else None,
    }
    
    @classmethod
    def command_to_broadcast_config(
        cls,
        command: AIBroadcastCommand
    ) -> Optional['BroadcastConfig']:
        """Convert AI command to BroadcastConfig."""
        if not PIPELINE_AVAILABLE:
            logger.error("Pipeline not available for config translation")
            return None
        
        # Translate string parameters to enums
        fec_rate = cls.FEC_RATE_MAP.get(
            command.fec_ldpc_rate.upper().replace("/", "_"),
            FECCodeRate.RATE_8_15
        )
        
        modulation = cls.MODULATION_MAP.get(
            command.modulation.upper().replace("-", "").replace("_", ""),
            ModulationScheme.QPSK
        )
        
        fft_size = cls.FFT_SIZE_MAP.get(
            command.fft_size.upper().replace("-", "_"),
            FFTSize.FFT_8K
        )
        
        guard_interval = cls.GUARD_INTERVAL_MAP.get(
            command.guard_interval.upper().replace("/", "_"),
            GuardInterval.GI_1_8
        )
        
        return BroadcastConfig(
            fec_ldpc_rate=fec_rate,
            fec_rs_symbols=command.fec_rs_symbols,
            modulation=modulation,
            fft_size=fft_size,
            guard_interval=guard_interval,
            fec_overhead_pct=command.fec_overhead_pct
        )
    
    @classmethod
    def broadcast_config_to_dict(cls, config: 'BroadcastConfig') -> Dict[str, Any]:
        """Convert BroadcastConfig to dictionary."""
        return {
            "fec_ldpc_rate": config.fec_ldpc_rate.name if hasattr(config.fec_ldpc_rate, 'name') else str(config.fec_ldpc_rate),
            "fec_rs_symbols": config.fec_rs_symbols,
            "modulation": config.modulation.name if hasattr(config.modulation, 'name') else str(config.modulation),
            "fft_size": config.fft_size.name if hasattr(config.fft_size, 'name') else str(config.fft_size),
            "guard_interval": config.guard_interval.name if hasattr(config.guard_interval, 'name') else str(config.guard_interval),
            "fec_overhead_pct": config.fec_overhead_pct
        }
    
    @classmethod
    def intent_to_config_hints(cls, intent: Intent) -> Dict[str, Any]:
        """
        Get configuration hints based on intent.
        
        These are suggestions that can be used when AI doesn't specify
        all parameters, or for validation.
        """
        hints = {
            Intent.MAXIMIZE_RELIABILITY: {
                "fec_ldpc_rate": "RATE_4_15",  # Low rate = more FEC
                "fec_rs_symbols": 32,
                "fec_overhead_pct": 60.0,
                "modulation": "QPSK",  # Most robust
                "guard_interval": "GI_1_4",  # More guard time
                "plp_mode": "robust"
            },
            Intent.MINIMIZE_BANDWIDTH: {
                "fec_ldpc_rate": "RATE_10_15",  # High rate = less FEC
                "fec_rs_symbols": 12,
                "fec_overhead_pct": 15.0,
                "modulation": "QAM64",  # Higher spectral efficiency
                "guard_interval": "GI_1_32",  # Less guard time
                "plp_mode": "fixed"
            },
            Intent.MAXIMIZE_ACCURACY: {
                "fec_ldpc_rate": "RATE_6_15",
                "fec_rs_symbols": 24,
                "fec_overhead_pct": 40.0,
                "modulation": "QPSK",
                "guard_interval": "GI_1_8",
                "update_frequency_hz": 10.0,  # Faster updates
                "tile_resolution": "high"
            },
            Intent.LOW_LATENCY: {
                "fec_ldpc_rate": "RATE_8_15",
                "fec_rs_symbols": 16,
                "fec_overhead_pct": 20.0,
                "modulation": "QAM16",
                "guard_interval": "GI_1_16",
                "update_frequency_hz": 20.0
            },
            Intent.URBAN_ROBUST: {
                "fec_ldpc_rate": "RATE_4_15",
                "fec_rs_symbols": 32,
                "fec_overhead_pct": 65.0,
                "modulation": "QPSK",
                "guard_interval": "GI_1_4",
                "plp_mode": "robust",
                "update_frequency_hz": 10.0
            },
            Intent.RURAL_EFFICIENT: {
                "fec_ldpc_rate": "RATE_8_15",
                "fec_rs_symbols": 16,
                "fec_overhead_pct": 25.0,
                "modulation": "QAM16",
                "guard_interval": "GI_1_8",
                "plp_mode": "fixed"
            },
            Intent.BALANCE_ALL: {
                "fec_ldpc_rate": "RATE_6_15",
                "fec_rs_symbols": 20,
                "fec_overhead_pct": 35.0,
                "modulation": "QPSK",
                "guard_interval": "GI_1_8",
                "plp_mode": "mobile"
            }
        }
        
        return hints.get(intent, hints[Intent.BALANCE_ALL])


# ============================================================================
# COMMAND VALIDATOR
# ============================================================================


class CommandValidator:
    """Validates AI commands before application."""
    
    # Valid ranges for parameters
    VALID_FEC_RATES = [
        "RATE_2_15", "RATE_3_15", "RATE_4_15", "RATE_5_15",
        "RATE_6_15", "RATE_7_15", "RATE_8_15", "RATE_9_15",
        "RATE_10_15", "RATE_11_15", "RATE_12_15", "RATE_13_15"
    ]
    
    VALID_MODULATIONS = ["QPSK", "QAM16", "QAM64", "QAM256"]
    
    VALID_FFT_SIZES = ["FFT_8K", "FFT_16K", "FFT_32K"]
    
    VALID_GUARD_INTERVALS = ["GI_1_4", "GI_1_8", "GI_1_16", "GI_1_32"]
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, reject commands with any issues.
                        If False, allow with warnings and corrections.
        """
        self.strict_mode = strict_mode
    
    def validate(
        self,
        command: AIBroadcastCommand,
        constraints: Optional[BroadcastConstraints] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate an AI command.
        
        Args:
            command: Command to validate
            constraints: Optional additional constraints
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Validate FEC rate
        fec_rate_normalized = command.fec_ldpc_rate.upper().replace("/", "_")
        if not fec_rate_normalized.startswith("RATE_"):
            fec_rate_normalized = f"RATE_{fec_rate_normalized}"
        
        if fec_rate_normalized not in self.VALID_FEC_RATES:
            errors.append(f"Invalid FEC rate: {command.fec_ldpc_rate}")
        
        # Validate RS symbols
        if not (4 <= command.fec_rs_symbols <= 64):
            errors.append(f"RS symbols must be 4-64, got: {command.fec_rs_symbols}")
        
        # Validate FEC overhead
        if not (5.0 <= command.fec_overhead_pct <= 80.0):
            if command.fec_overhead_pct < 5.0:
                warnings.append(f"FEC overhead {command.fec_overhead_pct}% is very low")
            elif command.fec_overhead_pct > 80.0:
                errors.append(f"FEC overhead {command.fec_overhead_pct}% exceeds 80% limit")
        
        # Validate modulation
        mod_normalized = command.modulation.upper().replace("-", "").replace("_", "")
        if mod_normalized not in self.VALID_MODULATIONS:
            errors.append(f"Invalid modulation: {command.modulation}")
        
        # Validate FFT size
        fft_normalized = command.fft_size.upper().replace("-", "_")
        if not fft_normalized.startswith("FFT_"):
            fft_normalized = f"FFT_{fft_normalized}"
        
        if fft_normalized not in self.VALID_FFT_SIZES:
            errors.append(f"Invalid FFT size: {command.fft_size}")
        
        # Validate guard interval
        gi_normalized = command.guard_interval.upper().replace("/", "_")
        if not gi_normalized.startswith("GI_"):
            gi_normalized = f"GI_{gi_normalized}"
        
        if gi_normalized not in self.VALID_GUARD_INTERVALS:
            errors.append(f"Invalid guard interval: {command.guard_interval}")
        
        # Validate update frequency
        if not (0.1 <= command.update_frequency_hz <= 100.0):
            warnings.append(f"Update frequency {command.update_frequency_hz} Hz is unusual")
        
        # Validate constraints if provided
        if constraints:
            if command.fec_overhead_pct > constraints.max_fec_overhead_pct:
                errors.append(
                    f"FEC overhead {command.fec_overhead_pct}% exceeds max {constraints.max_fec_overhead_pct}%"
                )
            
            if command.fec_overhead_pct < constraints.min_fec_overhead_pct:
                errors.append(
                    f"FEC overhead {command.fec_overhead_pct}% below min {constraints.min_fec_overhead_pct}%"
                )
        
        # Validate command's own constraints
        constraint_valid, constraint_errors = command.constraints.validate()
        if not constraint_valid:
            errors.extend(constraint_errors)
        
        # Determine overall validity
        is_valid = len(errors) == 0 if self.strict_mode else True
        
        return is_valid, errors, warnings


# ============================================================================
# METRICS COLLECTOR
# ============================================================================


class MetricsCollector:
    """Collects and aggregates metrics from broadcast operations."""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            history_size: Maximum number of processing times to keep for averaging
        """
        self.history_size = history_size
        self.processing_times: deque = deque(maxlen=history_size)
        self.latencies: deque = deque(maxlen=history_size)
        
        # Counters
        self.start_time = time.time()
        self.commands_received = 0
        self.commands_applied = 0
        self.commands_rejected = 0
        self.commands_failed = 0
        self.data_processed_bytes = 0
        self.frames_transmitted = 0
        
        # Current state
        self.current_config: Optional[Dict[str, Any]] = None
        self.current_mode = "traditional"
        self.current_intent = "balance_all"
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def record_processing(
        self,
        processing_time_ms: float,
        bytes_processed: int,
        frames: int = 1
    ) -> None:
        """Record a processing operation."""
        with self._lock:
            self.processing_times.append(processing_time_ms)
            self.data_processed_bytes += bytes_processed
            self.frames_transmitted += frames
    
    def record_latency(self, latency_ms: float) -> None:
        """Record end-to-end latency."""
        with self._lock:
            self.latencies.append(latency_ms)
    
    def record_command(self, applied: bool, rejected: bool = False) -> None:
        """Record a command reception."""
        with self._lock:
            self.commands_received += 1
            if applied:
                self.commands_applied += 1
            elif rejected:
                self.commands_rejected += 1
            else:
                self.commands_failed += 1
    
    def update_config(
        self,
        config: Dict[str, Any],
        mode: str,
        intent: str
    ) -> None:
        """Update current configuration state."""
        with self._lock:
            self.current_config = config
            self.current_mode = mode
            self.current_intent = intent
    
    def get_metrics(self) -> ControllerMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            metrics = ControllerMetrics()
            
            # Timing
            metrics.uptime_seconds = time.time() - self.start_time
            
            # Command stats
            metrics.commands_received = self.commands_received
            metrics.commands_applied = self.commands_applied
            metrics.commands_rejected = self.commands_rejected
            metrics.commands_failed = self.commands_failed
            
            # Current config
            metrics.current_mode = self.current_mode
            metrics.current_intent = self.current_intent
            
            if self.current_config:
                metrics.current_fec_rate = self.current_config.get("fec_ldpc_rate", "RATE_8_15")
                metrics.current_modulation = self.current_config.get("modulation", "QPSK")
                metrics.current_fec_overhead_pct = self.current_config.get("fec_overhead_pct", 15.0)
            
            # Processing stats
            metrics.data_processed_bytes = self.data_processed_bytes
            metrics.frames_transmitted = self.frames_transmitted
            
            # Averages
            if self.processing_times:
                metrics.avg_processing_time_ms = sum(self.processing_times) / len(self.processing_times)
            
            if self.latencies:
                metrics.avg_latency_ms = sum(self.latencies) / len(self.latencies)
            
            # Estimate spectral efficiency based on modulation and FEC
            metrics.estimated_spectral_efficiency = self._estimate_spectral_efficiency()
            
            return metrics
    
    def _estimate_spectral_efficiency(self) -> float:
        """Estimate spectral efficiency from current config."""
        if not self.current_config:
            return 1.0
        
        # Bits per symbol based on modulation
        mod = self.current_config.get("modulation", "QPSK")
        bits_per_symbol = {
            "QPSK": 2,
            "QAM16": 4,
            "QAM64": 6,
            "QAM256": 8
        }.get(mod, 2)
        
        # Code rate from FEC
        fec_rate = self.current_config.get("fec_ldpc_rate", "RATE_8_15")
        rate_parts = fec_rate.replace("RATE_", "").split("_")
        if len(rate_parts) == 2:
            try:
                code_rate = int(rate_parts[0]) / int(rate_parts[1])
            except ValueError:
                code_rate = 0.5
        else:
            code_rate = 0.5
        
        return bits_per_symbol * code_rate
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.processing_times.clear()
            self.latencies.clear()
            self.start_time = time.time()
            self.commands_received = 0
            self.commands_applied = 0
            self.commands_rejected = 0
            self.commands_failed = 0
            self.data_processed_bytes = 0
            self.frames_transmitted = 0


# ============================================================================
# BROADCAST CONTROLLER
# ============================================================================


class BroadcastController:
    """
    Central controller for ATSC 3.0 broadcast operations.
    
    Manages configuration, processes data through the pipeline,
    and collects metrics for AI feedback.
    """
    
    def __init__(
        self,
        mode: ControllerMode = ControllerMode.TRADITIONAL,
        config: Optional[Union[BroadcastConfig, Dict[str, Any]]] = None,
        history_size: int = 100,
        strict_validation: bool = False
    ):
        """
        Initialize broadcast controller.
        
        Args:
            mode: Operating mode (TRADITIONAL, AI_CONTROLLED, HYBRID)
            config: Initial broadcast configuration
            history_size: Number of configuration changes to keep in history
            strict_validation: If True, reject invalid commands entirely
        """
        self.mode = mode
        self.strict_validation = strict_validation
        
        # Initialize components
        self.validator = CommandValidator(strict_mode=strict_validation)
        self.metrics_collector = MetricsCollector()
        
        # Configuration history
        self.history_size = history_size
        self.config_history: deque = deque(maxlen=history_size)
        
        # Current state
        self.current_config: Optional[BroadcastConfig] = None
        self.current_command: Optional[AIBroadcastCommand] = None
        self.current_intent: Intent = Intent.BALANCE_ALL
        
        # Pipeline (lazy initialization)
        self._pipeline: Optional[BroadcastPipeline] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Operator overrides (for HYBRID mode)
        self.operator_overrides: Dict[str, Any] = {}
        
        # Event callbacks
        self._on_config_change: List[Callable[[Dict[str, Any]], None]] = []
        self._on_command_applied: List[Callable[[AIBroadcastCommand, CommandResult], None]] = []
        
        # Apply initial config if provided
        if config:
            self._apply_initial_config(config)
        
        logger.info(f"BroadcastController initialized in {mode.name} mode")
    
    def _apply_initial_config(
        self,
        config: Union[BroadcastConfig, Dict[str, Any]]
    ) -> None:
        """Apply initial configuration."""
        if isinstance(config, dict):
            # Create command from dict
            cmd = AIBroadcastCommand.from_dict(config)
            self.current_config = ConfigTranslator.command_to_broadcast_config(cmd)
        else:
            self.current_config = config
        
        if self.current_config:
            config_dict = ConfigTranslator.broadcast_config_to_dict(self.current_config)
            self.metrics_collector.update_config(
                config_dict,
                self.mode.name.lower(),
                self.current_intent.value
            )
    
    @property
    def pipeline(self) -> Optional[BroadcastPipeline]:
        """Get or create broadcast pipeline."""
        if self._pipeline is None and PIPELINE_AVAILABLE:
            self._pipeline = BroadcastPipeline(config=self.current_config)
        return self._pipeline
    
    def set_mode(self, mode: ControllerMode) -> None:
        """Change operating mode."""
        with self._lock:
            old_mode = self.mode
            self.mode = mode
            logger.info(f"Controller mode changed: {old_mode.name} -> {mode.name}")
            
            self.metrics_collector.update_config(
                ConfigTranslator.broadcast_config_to_dict(self.current_config) if self.current_config else {},
                mode.name.lower(),
                self.current_intent.value
            )
    
    def set_operator_override(self, parameter: str, value: Any) -> None:
        """
        Set an operator override (for HYBRID mode).
        
        Args:
            parameter: Parameter name to override
            value: Override value
        """
        with self._lock:
            self.operator_overrides[parameter] = value
            logger.info(f"Operator override set: {parameter} = {value}")
    
    def clear_operator_overrides(self) -> None:
        """Clear all operator overrides."""
        with self._lock:
            self.operator_overrides.clear()
            logger.info("Operator overrides cleared")
    
    def apply_command(
        self,
        command: Union[AIBroadcastCommand, Dict[str, Any]],
        force: bool = False
    ) -> CommandResult:
        """
        Apply an AI broadcast command.
        
        Args:
            command: Command to apply (object or dict)
            force: If True, skip validation and apply anyway
        
        Returns:
            CommandResult indicating success/failure
        """
        start_time = time.time()
        
        # Convert dict to command if needed
        if isinstance(command, dict):
            command = AIBroadcastCommand.from_dict(command)
        
        result = CommandResult(
            command_id=command.command_id,
            status=ConfigurationStatus.PENDING
        )
        
        with self._lock:
            try:
                # Check if we should accept commands
                if self.mode == ControllerMode.TRADITIONAL and not force:
                    result.status = ConfigurationStatus.REJECTED
                    result.error_message = "Controller is in TRADITIONAL mode, not accepting AI commands"
                    self.metrics_collector.record_command(applied=False, rejected=True)
                    return result
                
                # Validate command
                if not force:
                    is_valid, errors, warnings = self.validator.validate(command)
                    result.validation_passed = is_valid
                    result.validation_errors = errors
                    
                    if not is_valid and self.strict_validation:
                        result.status = ConfigurationStatus.REJECTED
                        result.error_message = f"Validation failed: {'; '.join(errors)}"
                        self.metrics_collector.record_command(applied=False, rejected=True)
                        return result
                    
                    # Log warnings
                    for warning in warnings:
                        logger.warning(f"Command {command.command_id}: {warning}")
                
                # Apply operator overrides in HYBRID mode
                if self.mode == ControllerMode.HYBRID and self.operator_overrides:
                    command = self._apply_overrides(command)
                    result.overrides_applied = list(self.operator_overrides.keys())
                
                # Translate to BroadcastConfig
                new_config = ConfigTranslator.command_to_broadcast_config(command)
                
                if new_config is None:
                    result.status = ConfigurationStatus.FAILED
                    result.error_message = "Failed to translate command to broadcast config"
                    self.metrics_collector.record_command(applied=False)
                    return result
                
                # Track what changed
                if self.current_config:
                    result.parameters_modified = self._get_modified_params(
                        self.current_config,
                        new_config
                    )
                
                # Update configuration
                old_config = self.current_config
                self.current_config = new_config
                self.current_command = command
                self.current_intent = command.intent
                
                # Update pipeline if exists
                if self._pipeline is not None:
                    self._pipeline = BroadcastPipeline(config=new_config)
                
                # Record in history
                config_dict = ConfigTranslator.broadcast_config_to_dict(new_config)
                result.applied_config = config_dict
                
                history_entry = ConfigurationHistoryEntry(
                    timestamp=datetime.now().isoformat(),
                    command_id=command.command_id,
                    intent=command.intent.value,
                    config=config_dict,
                    status="applied"
                )
                self.config_history.append(history_entry)
                
                # Update metrics collector
                self.metrics_collector.update_config(
                    config_dict,
                    self.mode.name.lower(),
                    command.intent.value
                )
                self.metrics_collector.record_command(applied=True)
                
                # Trigger callbacks
                for callback in self._on_config_change:
                    try:
                        callback(config_dict)
                    except Exception as e:
                        logger.error(f"Config change callback error: {e}")
                
                result.status = ConfigurationStatus.APPLIED
                result.application_time_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Applied command {command.command_id}: "
                    f"intent={command.intent.value}, "
                    f"fec={command.fec_ldpc_rate}, "
                    f"mod={command.modulation}"
                )
                
                # Trigger command applied callbacks
                for callback in self._on_command_applied:
                    try:
                        callback(command, result)
                    except Exception as e:
                        logger.error(f"Command applied callback error: {e}")
                
            except Exception as e:
                result.status = ConfigurationStatus.FAILED
                result.error_message = str(e)
                self.metrics_collector.record_command(applied=False)
                logger.exception(f"Failed to apply command {command.command_id}")
        
        return result
    
    def _apply_overrides(self, command: AIBroadcastCommand) -> AIBroadcastCommand:
        """Apply operator overrides to command."""
        # Create a copy with overrides applied
        overridden = AIBroadcastCommand.from_dict(command.to_dict())
        
        for param, value in self.operator_overrides.items():
            if hasattr(overridden, param):
                setattr(overridden, param, value)
        
        return overridden
    
    def _get_modified_params(
        self,
        old_config: BroadcastConfig,
        new_config: BroadcastConfig
    ) -> List[str]:
        """Get list of parameters that changed."""
        modified = []
        
        old_dict = ConfigTranslator.broadcast_config_to_dict(old_config)
        new_dict = ConfigTranslator.broadcast_config_to_dict(new_config)
        
        for key in new_dict:
            if key in old_dict and old_dict[key] != new_dict[key]:
                modified.append(key)
        
        return modified
    
    def process(
        self,
        data: bytes,
        config_override: Optional[BroadcastConfig] = None
    ) -> Optional[BroadcastResult]:
        """
        Process data through the broadcast pipeline.
        
        Args:
            data: Raw data to broadcast (e.g., RTCM)
            config_override: Optional config override for this processing
        
        Returns:
            BroadcastResult or None if pipeline unavailable
        """
        if not PIPELINE_AVAILABLE:
            logger.error("Broadcast pipeline not available")
            return None
        
        start_time = time.time()
        
        with self._lock:
            config = config_override or self.current_config
            
            # Get or create pipeline
            pipeline = self.pipeline
            if pipeline is None:
                pipeline = BroadcastPipeline(config=config)
                self._pipeline = pipeline
        
        # Process (outside lock to allow concurrent processing)
        try:
            result = pipeline.process(data, config=config)
            
            # Record metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_processing(
                processing_time_ms,
                len(data),
                frames=1
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            return None
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get current configuration as dictionary."""
        with self._lock:
            if self.current_config:
                return ConfigTranslator.broadcast_config_to_dict(self.current_config)
            return None
    
    def get_metrics(self) -> ControllerMetrics:
        """Get current metrics snapshot."""
        return self.metrics_collector.get_metrics()
    
    def get_ai_feedback(self) -> Dict[str, Any]:
        """
        Get metrics formatted for AI Orchestrator feedback.
        
        Returns:
            Dictionary with performance, quality, and configuration data
        """
        metrics = self.get_metrics()
        return metrics.to_ai_feedback()
    
    def get_config_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get configuration history.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of configuration history entries
        """
        with self._lock:
            history = list(self.config_history)
            if limit:
                history = history[-limit:]
            return [
                {
                    "timestamp": entry.timestamp,
                    "command_id": entry.command_id,
                    "intent": entry.intent,
                    "config": entry.config,
                    "status": entry.status
                }
                for entry in history
            ]
    
    def on_config_change(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register callback for configuration changes."""
        self._on_config_change.append(callback)
    
    def on_command_applied(
        self,
        callback: Callable[[AIBroadcastCommand, CommandResult], None]
    ) -> None:
        """Register callback for when commands are applied."""
        self._on_command_applied.append(callback)
    
    def reset(self) -> None:
        """Reset controller to initial state."""
        with self._lock:
            self.current_config = None
            self.current_command = None
            self.current_intent = Intent.BALANCE_ALL
            self._pipeline = None
            self.config_history.clear()
            self.operator_overrides.clear()
            self.metrics_collector.reset()
            logger.info("Controller reset to initial state")
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status summary."""
        metrics = self.get_metrics()
        
        return {
            "mode": self.mode.name,
            "intent": self.current_intent.value,
            "config": self.get_current_config(),
            "uptime_seconds": metrics.uptime_seconds,
            "commands": {
                "received": metrics.commands_received,
                "applied": metrics.commands_applied,
                "rejected": metrics.commands_rejected,
                "failed": metrics.commands_failed
            },
            "processing": {
                "bytes_processed": metrics.data_processed_bytes,
                "frames_transmitted": metrics.frames_transmitted,
                "avg_time_ms": metrics.avg_processing_time_ms
            },
            "overrides_active": len(self.operator_overrides) > 0,
            "pipeline_available": PIPELINE_AVAILABLE,
            "decoder_available": DECODER_AVAILABLE
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_traditional_controller(
    config: Optional[Dict[str, Any]] = None
) -> BroadcastController:
    """
    Create a controller in TRADITIONAL mode with default/provided config.
    
    Args:
        config: Optional initial configuration
    
    Returns:
        BroadcastController in TRADITIONAL mode
    """
    default_config = {
        "fec_ldpc_rate": "RATE_8_15",
        "fec_rs_symbols": 16,
        "fec_overhead_pct": 15.0,
        "modulation": "QPSK",
        "fft_size": "FFT_8K",
        "guard_interval": "GI_1_8"
    }
    
    if config:
        default_config.update(config)
    
    return BroadcastController(
        mode=ControllerMode.TRADITIONAL,
        config=default_config
    )


def create_ai_controller(
    initial_intent: Intent = Intent.BALANCE_ALL,
    strict_validation: bool = False
) -> BroadcastController:
    """
    Create a controller in AI_CONTROLLED mode.
    
    Args:
        initial_intent: Initial intent to apply
        strict_validation: If True, reject invalid commands
    
    Returns:
        BroadcastController in AI_CONTROLLED mode
    """
    controller = BroadcastController(
        mode=ControllerMode.AI_CONTROLLED,
        strict_validation=strict_validation
    )
    
    # Apply initial intent-based config
    hints = ConfigTranslator.intent_to_config_hints(initial_intent)
    initial_command = AIBroadcastCommand.from_dict({
        "intent": initial_intent.value,
        "broadcast_config": hints
    })
    
    controller.apply_command(initial_command, force=True)
    
    return controller


def create_hybrid_controller(
    base_config: Optional[Dict[str, Any]] = None,
    operator_overrides: Optional[Dict[str, Any]] = None
) -> BroadcastController:
    """
    Create a controller in HYBRID mode.
    
    Args:
        base_config: Base configuration
        operator_overrides: Parameters that operator controls
    
    Returns:
        BroadcastController in HYBRID mode
    """
    controller = BroadcastController(
        mode=ControllerMode.HYBRID,
        config=base_config
    )
    
    if operator_overrides:
        for param, value in operator_overrides.items():
            controller.set_operator_override(param, value)
    
    return controller


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """CLI for testing broadcast controller."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ATSC 3.0 Broadcast Controller - Test and Demo"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show controller status")
    status_parser.add_argument(
        "--mode",
        choices=["traditional", "ai", "hybrid"],
        default="traditional",
        help="Controller mode"
    )
    
    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a configuration")
    apply_parser.add_argument(
        "--intent",
        choices=[i.value for i in Intent],
        default="balance_all",
        help="Intent to apply"
    )
    apply_parser.add_argument(
        "--fec-rate",
        default="RATE_8_15",
        help="FEC LDPC rate"
    )
    apply_parser.add_argument(
        "--modulation",
        default="QPSK",
        help="Modulation scheme"
    )
    apply_parser.add_argument(
        "--output",
        "-o",
        help="Output file for result JSON"
    )
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process data through pipeline")
    process_parser.add_argument(
        "input_file",
        help="Input data file"
    )
    process_parser.add_argument(
        "--output",
        "-o",
        help="Output file for OFDM signal"
    )
    process_parser.add_argument(
        "--intent",
        choices=[i.value for i in Intent],
        default="balance_all",
        help="Intent to use"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo sequence")
    demo_parser.add_argument(
        "--scenario",
        choices=["traditional_vs_ai", "intent_switching", "hybrid_override"],
        default="traditional_vs_ai",
        help="Demo scenario"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    if args.command == "status":
        # Create controller in specified mode
        mode_map = {
            "traditional": ControllerMode.TRADITIONAL,
            "ai": ControllerMode.AI_CONTROLLED,
            "hybrid": ControllerMode.HYBRID
        }
        controller = BroadcastController(mode=mode_map[args.mode])
        
        status = controller.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "apply":
        # Create AI controller and apply command
        controller = create_ai_controller()
        
        command = AIBroadcastCommand(
            intent=Intent(args.intent),
            fec_ldpc_rate=args.fec_rate,
            modulation=args.modulation
        )
        
        result = controller.apply_command(command)
        
        result_dict = {
            "command_id": result.command_id,
            "status": result.status.value,
            "applied_config": result.applied_config,
            "validation_passed": result.validation_passed,
            "validation_errors": result.validation_errors,
            "parameters_modified": result.parameters_modified,
            "application_time_ms": result.application_time_ms
        }
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result_dict, f, indent=2)
            print(f"Result saved to: {args.output}")
        else:
            print(json.dumps(result_dict, indent=2))
    
    elif args.command == "process":
        if not PIPELINE_AVAILABLE:
            print("ERROR: Broadcast pipeline not available")
            return 1
        
        # Read input file
        with open(args.input_file, "rb") as f:
            data = f.read()
        
        print(f"Read {len(data)} bytes from {args.input_file}")
        
        # Create controller with intent
        controller = create_ai_controller(Intent(args.intent))
        
        # Process
        result = controller.process(data)
        
        if result:
            print(f"Processing successful")
            print(f"  OFDM samples: {len(result.signal.samples) if hasattr(result, 'signal') else 'N/A'}")
            
            if args.output and hasattr(result, 'signal'):
                with open(args.output, "wb") as f:
                    result.signal.samples.tofile(f)
                print(f"Saved OFDM signal to: {args.output}")
            
            # Show metrics
            metrics = controller.get_metrics()
            print(f"\nMetrics:")
            print(f"  Processing time: {metrics.avg_processing_time_ms:.2f} ms")
            print(f"  Spectral efficiency: {metrics.estimated_spectral_efficiency:.2f} bits/s/Hz")
        else:
            print("Processing failed")
            return 1
    
    elif args.command == "demo":
        print(f"Running demo: {args.scenario}")
        print("=" * 60)
        
        if args.scenario == "traditional_vs_ai":
            # Demo: Compare traditional and AI configurations
            print("\n1. Creating TRADITIONAL controller...")
            trad_controller = create_traditional_controller()
            print(f"   Config: {trad_controller.get_current_config()}")
            
            print("\n2. Creating AI controller with MAXIMIZE_RELIABILITY intent...")
            ai_controller = create_ai_controller(Intent.MAXIMIZE_RELIABILITY)
            print(f"   Config: {ai_controller.get_current_config()}")
            
            print("\n3. Comparison:")
            trad_config = trad_controller.get_current_config()
            ai_config = ai_controller.get_current_config()
            
            print(f"   FEC Rate: {trad_config['fec_ldpc_rate']} vs {ai_config['fec_ldpc_rate']}")
            print(f"   FEC Overhead: {trad_config['fec_overhead_pct']}% vs {ai_config['fec_overhead_pct']}%")
            print(f"   Guard Interval: {trad_config['guard_interval']} vs {ai_config['guard_interval']}")
        
        elif args.scenario == "intent_switching":
            # Demo: Switch between intents
            controller = create_ai_controller()
            
            intents = [
                Intent.MAXIMIZE_RELIABILITY,
                Intent.MINIMIZE_BANDWIDTH,
                Intent.MAXIMIZE_ACCURACY,
                Intent.URBAN_ROBUST
            ]
            
            for intent in intents:
                print(f"\nSwitching to intent: {intent.value}")
                hints = ConfigTranslator.intent_to_config_hints(intent)
                command = AIBroadcastCommand.from_dict({
                    "intent": intent.value,
                    "broadcast_config": hints
                })
                result = controller.apply_command(command)
                print(f"   Status: {result.status.value}")
                print(f"   Config: FEC={hints['fec_ldpc_rate']}, Overhead={hints['fec_overhead_pct']}%")
            
            print("\nConfiguration history:")
            for entry in controller.get_config_history():
                print(f"   {entry['timestamp']}: {entry['intent']}")
        
        elif args.scenario == "hybrid_override":
            # Demo: Hybrid mode with operator override
            print("\n1. Creating HYBRID controller...")
            controller = create_hybrid_controller()
            
            print("\n2. Setting operator override: modulation=QAM16")
            controller.set_operator_override("modulation", "QAM16")
            
            print("\n3. AI suggests QPSK for reliability...")
            command = AIBroadcastCommand(
                intent=Intent.MAXIMIZE_RELIABILITY,
                modulation="QPSK",
                fec_ldpc_rate="RATE_4_15"
            )
            result = controller.apply_command(command)
            
            print(f"   Applied config: {result.applied_config}")
            print(f"   Overrides applied: {result.overrides_applied}")
            print(f"   Note: Modulation remains QAM16 due to operator override")
        
        print("\n" + "=" * 60)
        print("Demo complete")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())