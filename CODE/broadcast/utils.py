"""
Utility functions for broadcast processing.
"""

import struct
import time
from typing import List, Tuple
import numpy as np

def crc24q(data: bytes) -> int:
    """
    Calculate CRC-24Q for RTCM frames.
    Polynomial: 0x1864CFB (CRC-24Q used in RTCM 3.x)
    """
    crc = 0
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
    return crc & 0xFFFFFF

def validate_rtcm_frame(frame: bytes) -> Tuple[bool, str]:
    """
    Validate RTCM 3.x frame structure.
    
    Returns:
        (is_valid, error_message)
    """
    if len(frame) < 6:
        return False, "Frame too short (minimum 6 bytes)"
    
    # Check preamble (0xD3)
    if frame[0] != 0xD3:
        return False, f"Invalid preamble: 0x{frame[0]:02X} (expected 0xD3)"
    
    # Extract length (10 bits after 6 reserved bits)
    length = ((frame[1] & 0x03) << 8) | frame[2]
    
    if len(frame) != length + 6:
        return False, f"Length mismatch: header says {length + 6}, got {len(frame)}"
    
    # Verify CRC
    payload_end = 3 + length
    calculated_crc = crc24q(frame[:payload_end])
    frame_crc = (frame[payload_end] << 16) | (frame[payload_end + 1] << 8) | frame[payload_end + 2]
    
    if calculated_crc != frame_crc:
        return False, f"CRC mismatch: calculated 0x{calculated_crc:06X}, got 0x{frame_crc:06X}"
    
    return True, "Valid"

def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)

def bytes_to_hex_string(data: bytes, max_len: int = 16) -> str:
    """Convert bytes to hex string for logging."""
    hex_str = ' '.join(f'{b:02X}' for b in data[:max_len])
    if len(data) > max_len:
        hex_str += '...'
    return hex_str

def calculate_spectrum_efficiency(
    data_bits: int,
    total_symbols: int,
    subcarriers: int,
    bits_per_symbol: int
) -> float:
    """
    Calculate spectrum efficiency in bits/s/Hz.
    
    Args:
        data_bits: Number of data bits transmitted
        total_symbols: Number of OFDM symbols used
        subcarriers: Number of subcarriers
        bits_per_symbol: Bits per modulation symbol
    
    Returns:
        Spectrum efficiency in bits/s/Hz
    """
    total_capacity = total_symbols * subcarriers * bits_per_symbol
    if total_capacity == 0:
        return 0.0
    return data_bits / total_capacity

def db_to_linear(db: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (db / 10)

def linear_to_db(linear: float) -> float:
    """Convert linear scale to dB."""
    return 10 * np.log10(linear) if linear > 0 else -np.inf