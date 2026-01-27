#!/usr/bin/env python3
# filepath: /media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE/gnss/rtcm_to_broadcast_handoff.py
"""
RTCM to Broadcast Pipeline Handoff Test

PURPOSE:
Integration test that validates the handoff between GNSS corrections (RTCM)
and the ATSC 3.0 broadcast pipeline. Demonstrates Traditional vs AI-Native
broadcast configuration differences.

This script sits at the BOUNDARY between GNSS and Broadcast teams:
- Uses RTCM data from GNSS team (rtcm_generator.py)
- Calls broadcast pipeline from Broadcast team (broadcast/pipeline.py)
- Produces proof artifacts for demo and validation

MODIFICATIONS FROM ORIGINAL:
1. ✅ Traditional vs AI mode support (--mode flag)
2. ✅ Scenario profile integration (same schema as gnss_baseline_runner.py)
3. ✅ Intent-aware configuration logging
4. ✅ Enhanced metrics for AI feedback (spectral efficiency, FEC overhead %)
5. ✅ Comparison report generation (Traditional vs AI side-by-side)
6. ✅ Optional round-trip verification (if decoder available)
7. ✅ Organized output structure matching gnss_baseline_runner.py

USAGE:
  # Run traditional baseline for scenario 1
  python rtcm_to_broadcast_handoff.py --scenario scenario1 --mode traditional

  # Run AI-optimized configuration for scenario 1
  python rtcm_to_broadcast_handoff.py --scenario scenario1 --mode ai

  # Run both modes and generate comparison
  python rtcm_to_broadcast_handoff.py --scenario scenario1 --mode both

  # Run with explicit RTCM file
  python rtcm_to_broadcast_handoff.py --rtcm path/to/corrections.rtcm --mode traditional

SCENARIO PROFILE BROADCAST SCHEMA:
  The scenario_profile.json should contain broadcast settings in:
  - traditional_config.broadcast: FEC, modulation for baseline
  - ai_config.broadcast: FEC, modulation for AI-optimized mode

OUTPUT STRUCTURE:
  OUTPUTs/<scenario>/handoff/<mode>/run_<timestamp>/
    - ofdm_signal.bin: Raw OFDM signal (first 1MB)
    - broadcast_config.json: Configuration used
    - metrics.json: Broadcast metrics (expansion ratio, spectral efficiency)
    - handoff_log.txt: Detailed processing log
    - manifest.json: Run metadata

INTEGRATION NOTES:
- Does NOT reimplement broadcast logic - calls broadcast/pipeline.py
- Does NOT auto-generate RTCM - expects files from rtcm_generator.py
- Does NOT simulate channel - that's broadcast/channel_simulator.py
- Optional decoder verification via --verify-roundtrip flag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import broadcast modules (from broadcast team's code)
try:
    from broadcast.pipeline import BroadcastPipeline, BroadcastConfig, BroadcastResult
    from broadcast.config import FECCodeRate, ModulationScheme, FFTSize, GuardInterval, PilotPattern
    BROADCAST_AVAILABLE = True
except ImportError as e:
    BROADCAST_AVAILABLE = False
    BROADCAST_IMPORT_ERROR = str(e)

# Optional: Import decoder for round-trip verification
try:
    from broadcast.decoder import BroadcastDecoder
    DECODER_AVAILABLE = True
except ImportError:
    DECODER_AVAILABLE = False

# ----------------------------
# Constants / Defaults
# ----------------------------

DEFAULT_SCENARIO_ROOT = Path(os.getenv("SCENARIO_ROOT", "DATA/scenarios"))
DEFAULT_RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "OUTPUTs"))

# Mode types for Traditional vs AI comparison
MODE_TRADITIONAL = "traditional"
MODE_AI = "ai"
MODE_BOTH = "both"

# RTCM constants
RTCM_PREAMBLE = 0xD3

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ----------------------------
# Data Structures
# ----------------------------


@dataclass
class BroadcastScenarioConfig:
    """Broadcast configuration loaded from scenario profile."""
    
    name: str
    description: str = ""
    intent: str = ""
    scenario_type: str = ""
    
    # RTCM file path
    rtcm_file: Optional[Path] = None
    
    # Traditional broadcast settings
    traditional_broadcast: Dict[str, Any] = field(default_factory=dict)
    
    # AI-optimized broadcast settings
    ai_broadcast: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation targets
    target_spectral_efficiency: float = 1.0  # bits/s/Hz
    target_latency_ms: float = 100.0
    max_fec_overhead_pct: float = 50.0


@dataclass
class HandoffMetrics:
    """Metrics from broadcast pipeline processing."""
    
    # Input metrics
    rtcm_bytes: int = 0
    rtcm_frames: int = 0
    
    # Pipeline metrics
    alp_bytes: int = 0
    fec_bytes: int = 0
    ofdm_samples: int = 0
    
    # Ratios
    expansion_ratio: float = 0.0  # ofdm_bytes / rtcm_bytes
    fec_overhead_pct: float = 0.0  # (fec_bytes - rtcm_bytes) / rtcm_bytes * 100
    
    # Timing
    processing_time_ms: float = 0.0
    estimated_tx_time_ms: float = 0.0
    
    # Spectral efficiency (approximate)
    spectral_efficiency_bps_hz: float = 0.0
    
    # Verification (if decoder available)
    roundtrip_verified: bool = False
    roundtrip_match: bool = False
    bytes_recovered: int = 0


@dataclass
class HandoffManifest:
    """Metadata for a single handoff test run."""
    
    run_id: str
    scenario_name: str
    mode: str  # "traditional" or "ai"
    intent: str
    timestamp: str
    
    # Configuration used
    broadcast_config: Dict[str, Any] = field(default_factory=dict)
    
    # Input file
    rtcm_file: str = ""
    rtcm_size_bytes: int = 0
    
    # Status
    success: bool = False
    error_message: str = ""
    duration_sec: float = 0.0
    
    # Decoder status
    decoder_available: bool = False
    roundtrip_verified: bool = False


# ----------------------------
# Utility Functions
# ----------------------------


def _now_run_id(prefix: str = "run") -> str:
    """Generate timestamped run ID."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _die(msg: str, code: int = 2) -> None:
    """Print error message and exit."""
    logger.error(msg)
    sys.exit(code)


def _ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    """Write object to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def format_bytes(n: int) -> str:
    """Format byte count for human readability."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.2f} KB"
    else:
        return f"{n / (1024 * 1024):.2f} MB"


def format_duration(ms: float) -> str:
    """Format duration for human readability."""
    if ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


# ----------------------------
# RTCM File Handling
# ----------------------------


def validate_rtcm_file(rtcm_path: Path) -> Tuple[bool, int, int]:
    """
    Validate RTCM file and count frames.
    
    Returns:
        Tuple of (is_valid, file_size, frame_count)
    """
    if not rtcm_path.exists():
        return False, 0, 0
    
    file_size = rtcm_path.stat().st_size
    if file_size == 0:
        return False, 0, 0
    
    # Count RTCM frames (look for 0xD3 preamble)
    frame_count = 0
    with open(rtcm_path, "rb") as f:
        data = f.read()
        i = 0
        while i < len(data):
            if data[i] == RTCM_PREAMBLE:
                # Check if we have enough bytes for header
                if i + 3 <= len(data):
                    # Extract length from header (10 bits)
                    length = ((data[i + 1] & 0x03) << 8) | data[i + 2]
                    if length <= 1023:  # Valid RTCM length
                        frame_count += 1
                        i += 3 + length + 3  # header + payload + CRC
                        continue
                i += 1
            else:
                i += 1
    
    return True, file_size, frame_count


def read_rtcm_file(rtcm_path: Path) -> bytes:
    """Read RTCM file contents."""
    with open(rtcm_path, "rb") as f:
        return f.read()


# ----------------------------
# Scenario Profile Loading
# ----------------------------


def load_scenario_config(
    scenario_name: str,
    scenario_root: Path = DEFAULT_SCENARIO_ROOT,
) -> BroadcastScenarioConfig:
    """
    Load scenario configuration from profile JSON.
    
    Args:
        scenario_name: Name of the scenario directory
        scenario_root: Root directory containing scenarios
    
    Returns:
        BroadcastScenarioConfig with all settings loaded
    """
    scenario_dir = scenario_root / scenario_name
    profile_path = scenario_dir / "scenario_profile.json"
    
    if not profile_path.exists():
        _die(f"Scenario profile not found: {profile_path}")
    
    profile = _load_json(profile_path)
    
    # Build configuration
    config = BroadcastScenarioConfig(
        name=profile.get("name", scenario_name),
        description=profile.get("description", ""),
        intent=profile.get("intent", ""),
        scenario_type=profile.get("scenario_type", ""),
    )
    
    # Load RTCM file path
    files = profile.get("files", profile)
    if files.get("rtcm_file"):
        config.rtcm_file = scenario_dir / files["rtcm_file"]
    elif files.get("corrections"):
        config.rtcm_file = scenario_dir / files["corrections"]
    
    # Load broadcast configurations
    trad_config = profile.get("traditional_config", {})
    ai_config = profile.get("ai_config", {})
    
    config.traditional_broadcast = trad_config.get("broadcast", {})
    config.ai_broadcast = ai_config.get("broadcast", {})
    
    # Load evaluation targets
    evaluation = profile.get("evaluation", {})
    config.target_spectral_efficiency = evaluation.get("target_spectral_efficiency", 1.0)
    config.target_latency_ms = evaluation.get("target_latency_ms", 100.0)
    config.max_fec_overhead_pct = evaluation.get("max_fec_overhead_pct", 50.0)
    
    return config


# ----------------------------
# Broadcast Config Translation
# ----------------------------


def string_to_fec_code_rate(s: str) -> FECCodeRate:
    """Convert string to FECCodeRate enum."""
    mapping = {
        "RATE_2_15": FECCodeRate.RATE_2_15,
        "RATE_3_15": FECCodeRate.RATE_3_15,
        "RATE_4_15": FECCodeRate.RATE_4_15,
        "RATE_5_15": FECCodeRate.RATE_5_15,
        "RATE_6_15": FECCodeRate.RATE_6_15,
        "RATE_7_15": FECCodeRate.RATE_7_15,
        "RATE_8_15": FECCodeRate.RATE_8_15,
        "RATE_9_15": FECCodeRate.RATE_9_15,
        "RATE_10_15": FECCodeRate.RATE_10_15,
        "RATE_11_15": FECCodeRate.RATE_11_15,
        "RATE_12_15": FECCodeRate.RATE_12_15,
        "RATE_13_15": FECCodeRate.RATE_13_15,
        # Also support fraction format
        "2/15": FECCodeRate.RATE_2_15,
        "6/15": FECCodeRate.RATE_6_15,
        "8/15": FECCodeRate.RATE_8_15,
    }
    return mapping.get(s.upper().replace("/", "_").replace("RATE_", "RATE_"), FECCodeRate.RATE_8_15)


def string_to_modulation(s: str) -> ModulationScheme:
    """Convert string to ModulationScheme enum."""
    mapping = {
        "QPSK": ModulationScheme.QPSK,
        "QAM16": ModulationScheme.QAM16,
        "QAM64": ModulationScheme.QAM64,
        "QAM256": ModulationScheme.QAM256,
        "16QAM": ModulationScheme.QAM16,
        "64QAM": ModulationScheme.QAM64,
        "256QAM": ModulationScheme.QAM256,
    }
    return mapping.get(s.upper().replace("-", "").replace("_", ""), ModulationScheme.QPSK)


def string_to_fft_size(s: str) -> FFTSize:
    """Convert string to FFTSize enum."""
    mapping = {
        "FFT_8K": FFTSize.FFT_8K,
        "FFT_16K": FFTSize.FFT_16K,
        "FFT_32K": FFTSize.FFT_32K,
        "8K": FFTSize.FFT_8K,
        "16K": FFTSize.FFT_16K,
        "32K": FFTSize.FFT_32K,
    }
    return mapping.get(s.upper().replace("-", "_"), FFTSize.FFT_8K)


def string_to_guard_interval(s: str) -> GuardInterval:
    """Convert string to GuardInterval enum."""
    mapping = {
        "GI_1_4": GuardInterval.GI_1_4,
        "GI_1_8": GuardInterval.GI_1_8,
        "GI_1_16": GuardInterval.GI_1_16,
        "GI_1_32": GuardInterval.GI_1_32,
        "1/4": GuardInterval.GI_1_4,
        "1/8": GuardInterval.GI_1_8,
        "1/16": GuardInterval.GI_1_16,
        "1/32": GuardInterval.GI_1_32,
    }
    return mapping.get(s.upper().replace("/", "_").replace("GI_", "GI_"), GuardInterval.GI_1_8)


def profile_to_broadcast_config(
    profile_broadcast: Dict[str, Any],
    mode: str
) -> BroadcastConfig:
    """
    Convert scenario profile broadcast settings to BroadcastConfig.
    
    Args:
        profile_broadcast: Dictionary from scenario profile
        mode: "traditional" or "ai" (for logging)
    
    Returns:
        BroadcastConfig ready for pipeline
    """
    # Extract settings with defaults
    fec_ldpc_rate = profile_broadcast.get("fec_ldpc_rate", "RATE_8_15")
    fec_rs_symbols = profile_broadcast.get("fec_rs_symbols", 16)
    modulation = profile_broadcast.get("modulation", "QPSK")
    fft_size = profile_broadcast.get("fft_size", "FFT_8K")
    guard_interval = profile_broadcast.get("guard_interval", "GI_1_8")
    fec_overhead_pct = profile_broadcast.get("fec_overhead_pct", 15.0)
    
    # Convert strings to enums
    config = BroadcastConfig(
        fec_ldpc_rate=string_to_fec_code_rate(str(fec_ldpc_rate)),
        fec_rs_symbols=int(fec_rs_symbols),
        modulation=string_to_modulation(str(modulation)),
        fft_size=string_to_fft_size(str(fft_size)),
        guard_interval=string_to_guard_interval(str(guard_interval)),
        fec_overhead_pct=float(fec_overhead_pct),
    )
    
    logger.info(f"[{mode.upper()}] Broadcast config:")
    logger.info(f"  FEC LDPC Rate: {config.fec_ldpc_rate.name}")
    logger.info(f"  FEC RS Symbols: {config.fec_rs_symbols}")
    logger.info(f"  Modulation: {config.modulation.name}")
    logger.info(f"  FFT Size: {config.fft_size.name}")
    logger.info(f"  Guard Interval: {config.guard_interval.name}")
    logger.info(f"  FEC Overhead: {config.fec_overhead_pct:.1f}%")
    
    return config


def get_default_broadcast_config(mode: str) -> BroadcastConfig:
    """
    Get default broadcast config when no scenario profile available.
    
    Traditional: Conservative, lower overhead
    AI: Optimized for reliability, higher overhead
    """
    if mode == MODE_AI:
        # AI mode: More robust settings
        return BroadcastConfig(
            fec_ldpc_rate=FECCodeRate.RATE_6_15,  # Lower rate = more FEC
            fec_rs_symbols=24,
            modulation=ModulationScheme.QPSK,  # Most robust
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_4,  # More guard time
            fec_overhead_pct=40.0,
        )
    else:
        # Traditional mode: Standard settings
        return BroadcastConfig(
            fec_ldpc_rate=FECCodeRate.RATE_8_15,  # Higher rate = less FEC
            fec_rs_symbols=16,
            modulation=ModulationScheme.QPSK,
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            fec_overhead_pct=15.0,
        )


# ----------------------------
# Metrics Computation
# ----------------------------


def compute_handoff_metrics(
    rtcm_data: bytes,
    result: BroadcastResult,
    processing_time_ms: float,
    broadcast_config: BroadcastConfig,
) -> HandoffMetrics:
    """
    Compute comprehensive metrics from broadcast pipeline result.
    
    Args:
        rtcm_data: Original RTCM data
        result: BroadcastResult from pipeline
        processing_time_ms: Time taken for processing
        broadcast_config: Configuration used
    
    Returns:
        HandoffMetrics object
    """
    metrics = HandoffMetrics()
    
    # Input metrics
    metrics.rtcm_bytes = len(rtcm_data)
    
    # Count RTCM frames
    frame_count = 0
    i = 0
    while i < len(rtcm_data):
        if rtcm_data[i] == RTCM_PREAMBLE:
            frame_count += 1
            if i + 3 <= len(rtcm_data):
                length = ((rtcm_data[i + 1] & 0x03) << 8) | rtcm_data[i + 2]
                i += 3 + length + 3
            else:
                i += 1
        else:
            i += 1
    metrics.rtcm_frames = frame_count
    
    # Pipeline metrics (extract from result)
    # Note: Actual field names depend on BroadcastResult structure
    if hasattr(result, 'alp_size'):
        metrics.alp_bytes = result.alp_size
    if hasattr(result, 'fec_size'):
        metrics.fec_bytes = result.fec_size
    elif hasattr(result, 'encoded_size'):
        metrics.fec_bytes = result.encoded_size
    
    # OFDM signal size
    if hasattr(result, 'signal') and result.signal is not None:
        if hasattr(result.signal, 'samples'):
            metrics.ofdm_samples = len(result.signal.samples)
        elif hasattr(result.signal, '__len__'):
            metrics.ofdm_samples = len(result.signal)
    
    # Calculate expansion ratio
    ofdm_bytes = metrics.ofdm_samples * 8  # Assuming complex64 (8 bytes per sample)
    if metrics.rtcm_bytes > 0:
        metrics.expansion_ratio = ofdm_bytes / metrics.rtcm_bytes
    
    # FEC overhead percentage
    if metrics.fec_bytes > 0 and metrics.rtcm_bytes > 0:
        metrics.fec_overhead_pct = (metrics.fec_bytes - metrics.rtcm_bytes) / metrics.rtcm_bytes * 100
    else:
        # Use configured overhead as estimate
        metrics.fec_overhead_pct = broadcast_config.fec_overhead_pct
    
    # Timing
    metrics.processing_time_ms = processing_time_ms
    
    # Estimate TX time (approximate based on OFDM parameters)
    # ATSC 3.0 symbol rate is approximately 6 MHz bandwidth
    symbol_rate = 6e6  # Hz (approximate)
    fft_size_val = int(broadcast_config.fft_size.name.replace("FFT_", "").replace("K", "")) * 1024
    symbols_per_second = symbol_rate / fft_size_val
    if symbols_per_second > 0 and metrics.ofdm_samples > 0:
        num_symbols = metrics.ofdm_samples / fft_size_val
        metrics.estimated_tx_time_ms = (num_symbols / symbols_per_second) * 1000
    
    # Spectral efficiency (bits per second per Hz)
    # Approximate based on modulation and code rate
    bits_per_symbol = {
        ModulationScheme.QPSK: 2,
        ModulationScheme.QAM16: 4,
        ModulationScheme.QAM64: 6,
        ModulationScheme.QAM256: 8,
    }.get(broadcast_config.modulation, 2)
    
    # Code rate approximation
    code_rate_str = broadcast_config.fec_ldpc_rate.name
    if "_" in code_rate_str:
        parts = code_rate_str.replace("RATE_", "").split("_")
        if len(parts) == 2:
            try:
                code_rate = int(parts[0]) / int(parts[1])
            except ValueError:
                code_rate = 0.5
        else:
            code_rate = 0.5
    else:
        code_rate = 0.5
    
    metrics.spectral_efficiency_bps_hz = bits_per_symbol * code_rate
    
    return metrics


# ----------------------------
# Round-Trip Verification
# ----------------------------


def verify_roundtrip(
    original_data: bytes,
    result: BroadcastResult,
    metrics: HandoffMetrics,
) -> Tuple[bool, int]:
    """
    Verify round-trip integrity using decoder (if available).
    
    Args:
        original_data: Original RTCM data
        result: BroadcastResult containing encoded signal
        metrics: Metrics object to update
    
    Returns:
        Tuple of (match, bytes_recovered)
    """
    if not DECODER_AVAILABLE:
        logger.warning("Decoder not available - skipping round-trip verification")
        return False, 0
    
    try:
        # Initialize decoder
        decoder = BroadcastDecoder()
        
        # Decode the OFDM signal back to data
        if hasattr(result, 'signal') and result.signal is not None:
            recovered_data = decoder.decode(result.signal)
            
            if recovered_data:
                metrics.bytes_recovered = len(recovered_data)
                metrics.roundtrip_verified = True
                metrics.roundtrip_match = (recovered_data == original_data)
                
                if metrics.roundtrip_match:
                    logger.info("✓ Round-trip verification PASSED - data matches")
                else:
                    logger.warning("✗ Round-trip verification FAILED - data mismatch")
                    logger.warning(f"  Original: {len(original_data)} bytes, Recovered: {len(recovered_data)} bytes")
                
                return metrics.roundtrip_match, metrics.bytes_recovered
    
    except Exception as e:
        logger.error(f"Round-trip verification error: {e}")
    
    return False, 0


# ----------------------------
# Main Handoff Test
# ----------------------------


def run_handoff_test(
    rtcm_path: Path,
    broadcast_config: BroadcastConfig,
    output_dir: Path,
    mode: str,
    scenario_name: str = "unknown",
    intent: str = "",
    verify_roundtrip_flag: bool = False,
) -> Tuple[bool, HandoffManifest, HandoffMetrics]:
    """
    Run the RTCM to broadcast handoff test.
    
    Args:
        rtcm_path: Path to RTCM file
        broadcast_config: Broadcast configuration to use
        output_dir: Directory for output artifacts
        mode: "traditional" or "ai"
        scenario_name: Name of scenario for logging
        intent: Intent string for AI mode
        verify_roundtrip_flag: Whether to verify round-trip
    
    Returns:
        Tuple of (success, manifest, metrics)
    """
    run_id = _now_run_id()
    run_dir = output_dir / mode / run_id
    _ensure_dir(run_dir)
    
    # Initialize manifest
    manifest = HandoffManifest(
        run_id=run_id,
        scenario_name=scenario_name,
        mode=mode,
        intent=intent,
        timestamp=datetime.now().isoformat(),
        decoder_available=DECODER_AVAILABLE,
    )
    
    # Setup logging to file
    log_path = run_dir / "handoff_log.txt"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        # ========================================
        # STAGE 1: Validate Input
        # ========================================
        logger.info("=" * 60)
        logger.info(f"RTCM to Broadcast Handoff Test")
        logger.info(f"Scenario: {scenario_name}")
        logger.info(f"Mode: {mode.upper()}")
        if intent:
            logger.info(f"Intent: {intent}")
        logger.info("=" * 60)
        
        logger.info("\n[STAGE 1/4] Validating RTCM input...")
        
        is_valid, file_size, frame_count = validate_rtcm_file(rtcm_path)
        if not is_valid:
            manifest.error_message = f"Invalid RTCM file: {rtcm_path}"
            logger.error(manifest.error_message)
            return False, manifest, HandoffMetrics()
        
        manifest.rtcm_file = str(rtcm_path)
        manifest.rtcm_size_bytes = file_size
        
        logger.info(f"  ✓ RTCM file: {rtcm_path.name}")
        logger.info(f"  ✓ Size: {format_bytes(file_size)}")
        logger.info(f"  ✓ Frames: {frame_count}")
        
        # Read RTCM data
        rtcm_data = read_rtcm_file(rtcm_path)
        
        # ========================================
        # STAGE 2: Initialize Pipeline
        # ========================================
        logger.info("\n[STAGE 2/4] Initializing broadcast pipeline...")
        
        if not BROADCAST_AVAILABLE:
            manifest.error_message = f"Broadcast module not available: {BROADCAST_IMPORT_ERROR}"
            logger.error(manifest.error_message)
            return False, manifest, HandoffMetrics()
        
        # Save broadcast config
        config_dict = {
            "fec_ldpc_rate": broadcast_config.fec_ldpc_rate.name,
            "fec_rs_symbols": broadcast_config.fec_rs_symbols,
            "modulation": broadcast_config.modulation.name,
            "fft_size": broadcast_config.fft_size.name,
            "guard_interval": broadcast_config.guard_interval.name,
            "fec_overhead_pct": broadcast_config.fec_overhead_pct,
        }
        manifest.broadcast_config = config_dict
        _write_json(run_dir / "broadcast_config.json", config_dict)
        
        # Initialize pipeline
        pipeline = BroadcastPipeline(config=broadcast_config)
        logger.info("  ✓ Pipeline initialized")
        
        # ========================================
        # STAGE 3: Process Through Pipeline
        # ========================================
        logger.info("\n[STAGE 3/4] Processing RTCM through broadcast pipeline...")
        
        start_time = time.time()
        result = pipeline.process(rtcm_data, config=broadcast_config)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        manifest.duration_sec = end_time - start_time
        
        logger.info(f"  ✓ Processing complete in {format_duration(processing_time_ms)}")
        
        # Compute metrics
        metrics = compute_handoff_metrics(rtcm_data, result, processing_time_ms, broadcast_config)
        
        logger.info(f"  ✓ ALP bytes: {format_bytes(metrics.alp_bytes)}")
        logger.info(f"  ✓ FEC bytes: {format_bytes(metrics.fec_bytes)}")
        logger.info(f"  ✓ OFDM samples: {metrics.ofdm_samples:,}")
        logger.info(f"  ✓ Expansion ratio: {metrics.expansion_ratio:.2f}x")
        logger.info(f"  ✓ FEC overhead: {metrics.fec_overhead_pct:.1f}%")
        logger.info(f"  ✓ Spectral efficiency: {metrics.spectral_efficiency_bps_hz:.2f} bits/s/Hz")
        
        # ========================================
        # STAGE 4: Save Artifacts & Verify
        # ========================================
        logger.info("\n[STAGE 4/4] Saving artifacts...")
        
        # Save OFDM signal preview (first 1MB)
        if hasattr(result, 'signal') and result.signal is not None:
            signal_data = None
            if hasattr(result.signal, 'samples'):
                signal_data = result.signal.samples
            elif hasattr(result.signal, 'tobytes'):
                signal_data = result.signal.tobytes()
            elif isinstance(result.signal, bytes):
                signal_data = result.signal
            
            if signal_data:
                preview_size = min(len(signal_data) if isinstance(signal_data, bytes) else len(signal_data) * 8, 1024 * 1024)
                with open(run_dir / "ofdm_signal.bin", "wb") as f:
                    if isinstance(signal_data, bytes):
                        f.write(signal_data[:preview_size])
                    else:
                        import numpy as np
                        np.array(signal_data[:preview_size // 8]).tofile(f)
                logger.info(f"  ✓ Saved OFDM signal preview ({format_bytes(preview_size)})")
        
        # Save metrics
        metrics_dict = asdict(metrics)
        _write_json(run_dir / "metrics.json", metrics_dict)
        logger.info("  ✓ Saved metrics.json")
        
        # Optional: Round-trip verification
        if verify_roundtrip_flag and DECODER_AVAILABLE:
            logger.info("\n[OPTIONAL] Verifying round-trip integrity...")
            match, recovered = verify_roundtrip(rtcm_data, result, metrics)
            manifest.roundtrip_verified = True
            metrics.roundtrip_verified = True
            metrics.roundtrip_match = match
            metrics.bytes_recovered = recovered
        
        # Save manifest
        manifest.success = True
        _write_json(run_dir / "manifest.json", asdict(manifest))
        logger.info("  ✓ Saved manifest.json")
        
        # ========================================
        # Summary
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("HANDOFF TEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Status: SUCCESS")
        logger.info(f"  Output: {run_dir}")
        logger.info(f"  Mode: {mode.upper()}")
        logger.info(f"  FEC Overhead: {metrics.fec_overhead_pct:.1f}%")
        logger.info(f"  Spectral Efficiency: {metrics.spectral_efficiency_bps_hz:.2f} bits/s/Hz")
        logger.info("=" * 60)
        
        return True, manifest, metrics
    
    except Exception as e:
        manifest.error_message = str(e)
        manifest.success = False
        logger.exception(f"Handoff test failed: {e}")
        _write_json(run_dir / "manifest.json", asdict(manifest))
        return False, manifest, HandoffMetrics()
    
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


# ----------------------------
# Comparison Report
# ----------------------------


def generate_comparison_report(
    scenario_name: str,
    traditional_metrics: Optional[HandoffMetrics],
    ai_metrics: Optional[HandoffMetrics],
    traditional_manifest: Optional[HandoffManifest],
    ai_manifest: Optional[HandoffManifest],
    output_dir: Path,
) -> Dict:
    """
    Generate comparison report between Traditional and AI modes.
    
    Returns:
        Dictionary containing comparison data
    """
    report = {
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "traditional": None,
        "ai": None,
        "comparison": {},
    }
    
    if traditional_metrics:
        report["traditional"] = {
            "metrics": asdict(traditional_metrics),
            "config": traditional_manifest.broadcast_config if traditional_manifest else {},
        }
    
    if ai_metrics:
        report["ai"] = {
            "metrics": asdict(ai_metrics),
            "config": ai_manifest.broadcast_config if ai_manifest else {},
        }
    
    # Compute comparison metrics
    if traditional_metrics and ai_metrics:
        comparison = report["comparison"]
        
        # FEC overhead comparison
        comparison["fec_overhead_traditional_pct"] = traditional_metrics.fec_overhead_pct
        comparison["fec_overhead_ai_pct"] = ai_metrics.fec_overhead_pct
        comparison["fec_overhead_increase_pct"] = (
            ai_metrics.fec_overhead_pct - traditional_metrics.fec_overhead_pct
        )
        
        # Spectral efficiency comparison
        comparison["spectral_efficiency_traditional"] = traditional_metrics.spectral_efficiency_bps_hz
        comparison["spectral_efficiency_ai"] = ai_metrics.spectral_efficiency_bps_hz
        
        # Expansion ratio comparison
        comparison["expansion_ratio_traditional"] = traditional_metrics.expansion_ratio
        comparison["expansion_ratio_ai"] = ai_metrics.expansion_ratio
        
        # Processing time comparison
        comparison["processing_time_traditional_ms"] = traditional_metrics.processing_time_ms
        comparison["processing_time_ai_ms"] = ai_metrics.processing_time_ms
        
        # Summary
        trade_offs = []
        
        if ai_metrics.fec_overhead_pct > traditional_metrics.fec_overhead_pct:
            trade_offs.append(f"AI uses {comparison['fec_overhead_increase_pct']:.1f}% more FEC overhead for robustness")
        
        if ai_metrics.spectral_efficiency_bps_hz < traditional_metrics.spectral_efficiency_bps_hz:
            trade_offs.append("AI trades spectral efficiency for reliability")
        elif ai_metrics.spectral_efficiency_bps_hz > traditional_metrics.spectral_efficiency_bps_hz:
            trade_offs.append("AI achieves higher spectral efficiency")
        
        comparison["trade_offs"] = trade_offs
        comparison["summary"] = (
            "AI mode applies intent-driven configuration with enhanced robustness. "
            "Traditional mode uses static baseline configuration."
        )
    
    # Write report
    report_path = output_dir / "comparison_report.json"
    _write_json(report_path, report)
    logger.info(f"Wrote comparison report: {report_path}")
    
    return report


# ----------------------------
# Argument Parsing
# ----------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RTCM to Broadcast Pipeline Handoff Test - Traditional vs AI-Native",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run traditional baseline for scenario 1
  python rtcm_to_broadcast_handoff.py --scenario scenario1 --mode traditional

  # Run AI-optimized configuration for scenario 1  
  python rtcm_to_broadcast_handoff.py --scenario scenario1 --mode ai

  # Run both modes and generate comparison
  python rtcm_to_broadcast_handoff.py --scenario scenario1 --mode both

  # Run with explicit RTCM file
  python rtcm_to_broadcast_handoff.py --rtcm path/to/corrections.rtcm --mode traditional

  # Verify round-trip integrity (requires decoder)
  python rtcm_to_broadcast_handoff.py --scenario scenario1 --mode both --verify-roundtrip
        """,
    )
    
    # Scenario-based input
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name (loads from DATA/scenarios/<name>/scenario_profile.json)",
    )
    
    # Explicit file input
    parser.add_argument(
        "--rtcm",
        type=str,
        help="Path to RTCM corrections file",
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MODE_TRADITIONAL, MODE_AI, MODE_BOTH],
        default=MODE_TRADITIONAL,
        help="Run mode: traditional (baseline), ai (optimized), or both (comparison)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Output directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    
    # Verification options
    parser.add_argument(
        "--verify-roundtrip",
        action="store_true",
        help="Verify round-trip integrity using decoder (if available)",
    )
    
    # Path configuration
    parser.add_argument(
        "--scenario-root",
        type=str,
        default=str(DEFAULT_SCENARIO_ROOT),
        help=f"Root directory for scenarios (default: {DEFAULT_SCENARIO_ROOT})",
    )
    
    return parser.parse_args()


# ----------------------------
# Main Entry Point
# ----------------------------


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Check broadcast module availability
    if not BROADCAST_AVAILABLE:
        _die(f"Broadcast module not available: {BROADCAST_IMPORT_ERROR}")
        return 1
    
    # Determine configuration source
    scenario_config = None
    rtcm_path = None
    scenario_name = "explicit"
    intent = ""
    
    if args.scenario:
        # Load from scenario profile
        scenario_root = Path(args.scenario_root).expanduser().resolve()
        scenario_config = load_scenario_config(args.scenario, scenario_root)
        scenario_name = scenario_config.name
        intent = scenario_config.intent
        rtcm_path = scenario_config.rtcm_file
        
        logger.info(f"Loaded scenario: {scenario_name}")
        logger.info(f"  Intent: {intent}")
        logger.info(f"  Description: {scenario_config.description}")
    
    # Override RTCM path if explicitly provided
    if args.rtcm:
        rtcm_path = Path(args.rtcm).expanduser().resolve()
    
    # Validate RTCM file
    if not rtcm_path:
        _die(
            "No RTCM file specified. Either:\n"
            "  1. Use --scenario with a scenario_profile.json that has rtcm_file, or\n"
            "  2. Use --rtcm to specify the file directly.\n\n"
            "To generate RTCM files, run: python gnss/rtcm_generator_v2.py --help"
        )
        return 1
    
    if not rtcm_path.exists():
        _die(
            f"RTCM file not found: {rtcm_path}\n\n"
            "To generate RTCM files, run: python gnss/rtcm_generator_v2.py --help"
        )
        return 1
    
    # Setup output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.scenario:
        output_dir = output_dir / args.scenario / "handoff"
    else:
        output_dir = output_dir / "handoff"
    _ensure_dir(output_dir)
    
    # Run based on mode
    traditional_metrics = None
    traditional_manifest = None
    ai_metrics = None
    ai_manifest = None
    
    if args.mode in [MODE_TRADITIONAL, MODE_BOTH]:
        logger.info("=" * 60)
        logger.info("Running TRADITIONAL mode (baseline)")
        logger.info("=" * 60)
        
        # Get broadcast config
        if scenario_config and scenario_config.traditional_broadcast:
            broadcast_config = profile_to_broadcast_config(
                scenario_config.traditional_broadcast, MODE_TRADITIONAL
            )
        else:
            broadcast_config = get_default_broadcast_config(MODE_TRADITIONAL)
            logger.info("[TRADITIONAL] Using default broadcast config")
        
        success, traditional_manifest, traditional_metrics = run_handoff_test(
            rtcm_path=rtcm_path,
            broadcast_config=broadcast_config,
            output_dir=output_dir,
            mode=MODE_TRADITIONAL,
            scenario_name=scenario_name,
            intent="baseline",
            verify_roundtrip_flag=args.verify_roundtrip,
        )
        
        if not success:
            logger.warning(f"Traditional run failed: {traditional_manifest.error_message}")
    
    if args.mode in [MODE_AI, MODE_BOTH]:
        logger.info("=" * 60)
        logger.info("Running AI mode (intent-driven)")
        logger.info("=" * 60)
        
        # Get broadcast config
        if scenario_config and scenario_config.ai_broadcast:
            broadcast_config = profile_to_broadcast_config(
                scenario_config.ai_broadcast, MODE_AI
            )
        else:
            broadcast_config = get_default_broadcast_config(MODE_AI)
            logger.info("[AI] Using default broadcast config")
        
        success, ai_manifest, ai_metrics = run_handoff_test(
            rtcm_path=rtcm_path,
            broadcast_config=broadcast_config,
            output_dir=output_dir,
            mode=MODE_AI,
            scenario_name=scenario_name,
            intent=intent,
            verify_roundtrip_flag=args.verify_roundtrip,
        )
        
        if not success:
            logger.warning(f"AI run failed: {ai_manifest.error_message}")
    
    # Generate comparison report if both modes were run
    if args.mode == MODE_BOTH and traditional_metrics and ai_metrics:
        logger.info("=" * 60)
        logger.info("Generating comparison report")
        logger.info("=" * 60)
        
        report = generate_comparison_report(
            scenario_name,
            traditional_metrics,
            ai_metrics,
            traditional_manifest,
            ai_manifest,
            output_dir,
        )
        
        # Print summary
        if "comparison" in report and report["comparison"]:
            comp = report["comparison"]
            print("\n" + "=" * 60)
            print("COMPARISON SUMMARY: Traditional vs AI-Native Broadcast")
            print("=" * 60)
            print(f"  FEC Overhead:")
            print(f"    Traditional: {comp.get('fec_overhead_traditional_pct', 0):.1f}%")
            print(f"    AI-Native:   {comp.get('fec_overhead_ai_pct', 0):.1f}%")
            print(f"    Difference:  {comp.get('fec_overhead_increase_pct', 0):+.1f}%")
            print(f"\n  Spectral Efficiency:")
            print(f"    Traditional: {comp.get('spectral_efficiency_traditional', 0):.2f} bits/s/Hz")
            print(f"    AI-Native:   {comp.get('spectral_efficiency_ai', 0):.2f} bits/s/Hz")
            print(f"\n  Trade-offs:")
            for trade_off in comp.get("trade_offs", []):
                print(f"    • {trade_off}")
            print("=" * 60)
    
    logger.info(f"\nResults written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())