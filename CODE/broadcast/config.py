"""
Broadcast module configuration constants and data structures.
Based on ATSC 3.0 specifications (A322, A331, A300).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ATSC 3.0 Physical Layer Parameters
class ModulationScheme(Enum):
    """ATSC 3.0 modulation schemes."""
    QPSK = "QPSK"
    QAM_16 = "16-QAM"
    QAM_64 = "64-QAM"
    QAM_256 = "256-QAM"
    QAM_1024 = "1024-QAM"
    QAM_4096 = "4096-QAM"

class FECCodeRate(Enum):
    """Forward Error Correction code rates."""
    RATE_2_15 = 2/15
    RATE_3_15 = 3/15
    RATE_4_15 = 4/15
    RATE_5_15 = 5/15
    RATE_6_15 = 6/15
    RATE_7_15 = 7/15
    RATE_8_15 = 8/15
    RATE_9_15 = 9/15
    RATE_10_15 = 10/15
    RATE_11_15 = 11/15
    RATE_12_15 = 12/15
    RATE_13_15 = 13/15

class PLPMode(Enum):
    """PLP (Physical Layer Pipe) modes."""
    STANDARD = "standard"
    MOBILE = "mobile"
    HIGH_CAPACITY = "high_capacity"

@dataclass
class BroadcastConfig:
    """Traditional static broadcast configuration."""
    fec_overhead_pct: float = 15.0
    redundancy: float = 1.0
    update_frequency_hz: float = 1.0
    tile_resolution: str = "medium"
    plp_mode: PLPMode = PLPMode.STANDARD
    modulation: ModulationScheme = ModulationScheme.QPSK
    code_rate: FECCodeRate = FECCodeRate.RATE_8_15

@dataclass
class AIBroadcastConfig:
    """AI-driven adaptive broadcast configuration."""
    timestamp: int
    intent: str
    fec_overhead_pct: float
    redundancy: float
    update_frequency_hz: float
    tile_resolution: str
    plp_mode: PLPMode
    modulation: ModulationScheme
    code_rate: FECCodeRate
    confidence: float = 0.9
    reasoning: Optional[str] = None

# ATSC 3.0 Frame Parameters
ATSC_FRAME_DURATION_MS = 50  # Target 50ms per frame
ATSC_SYMBOL_DURATION_US = 1000  # Microseconds
ATSC_SUBCARRIERS = 8192  # FFT size for OFDM

# Correction Data Parameters
RTCM_MAX_FRAME_SIZE = 1024  # bytes
BITMAP_TILE_SIZE = 100  # 100x100 pixels
MAX_CORRECTION_AGE_SEC = 10  # Maximum acceptable correction age

# PLP Configuration
PLP_A_ID = 0  # RTK correction stream
PLP_B_ID = 1  # Bitmap/error maps
PLP_C_ID = 2  # Optional extended services

# Frequency ranges for adaptive control
MIN_UPDATE_FREQUENCY_HZ = 0.5
MAX_UPDATE_FREQUENCY_HZ = 10.0
MIN_FEC_OVERHEAD_PCT = 10
MAX_FEC_OVERHEAD_PCT = 40