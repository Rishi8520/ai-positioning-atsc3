#!/usr/bin/env python3
# filepath: /media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE/broadcast/decoder.py
"""
ATSC 3.0 Broadcast Decoder

PURPOSE:
Reverses the broadcast pipeline to recover original data from OFDM signals.
Implements: OFDM Demodulation → Frame Parsing → FEC Decoding → ALP Decapsulation

This module enables:
1. Round-trip verification in handoff tests
2. Receiver simulation for end-to-end validation
3. BER/SNR analysis after channel simulation

PIPELINE STAGES (reverse of encoder):
  Stage 1: OFDM Demodulation - Extract symbols from time-domain signal
  Stage 2: Frame Parsing - Extract PLPs and payloads from ATSC frame
  Stage 3: FEC Decoding - LDPC + Reed-Solomon error correction
  Stage 4: ALP Decapsulation - Extract original data packets

USAGE:
  from broadcast.decoder import BroadcastDecoder, DecoderConfig
  
  decoder = BroadcastDecoder()
  result = decoder.decode(ofdm_signal)
  
  if result.success:
      recovered_data = result.data
      print(f"Recovered {len(recovered_data)} bytes")

INTEGRATION:
  - Used by gnss/rtcm_to_broadcast_handoff.py for round-trip verification
  - Used by channel_simulator.py for post-channel decoding
  - Mirrors broadcast/pipeline.py structure

NOTES:
  - This is a simplified decoder for PoC demonstration
  - Real ATSC 3.0 decoding requires hardware or complex DSP
  - Focus is on functional verification, not production-grade decoding
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import encoder modules to access shared configurations
try:
    from broadcast.config import (
        FECCodeRate, 
        ModulationScheme, 
        FFTSize, 
        GuardInterval,
        PilotPattern
    )
    from broadcast.alpencoder import ALPPacket
    from broadcast.fecencoder import FECConfig
    from broadcast.ofdmmodulator import OFDMSignal
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Define minimal fallbacks
    class FECCodeRate(Enum):
        RATE_2_15 = (2, 15)
        RATE_4_15 = (4, 15)
        RATE_6_15 = (6, 15)
        RATE_8_15 = (8, 15)
    
    class ModulationScheme(Enum):
        QPSK = 2
        QAM16 = 4
        QAM64 = 6
        QAM256 = 8
    
    class FFTSize(Enum):
        FFT_8K = 8192
        FFT_16K = 16384
        FFT_32K = 32768
    
    class GuardInterval(Enum):
        GI_1_4 = 0.25
        GI_1_8 = 0.125
        GI_1_16 = 0.0625
        GI_1_32 = 0.03125

logger = logging.getLogger(__name__)


# ----------------------------
# Data Structures
# ----------------------------


@dataclass
class DecoderConfig:
    """Configuration for broadcast decoder."""
    
    # OFDM parameters (must match encoder)
    fft_size: FFTSize = FFTSize.FFT_8K
    guard_interval: GuardInterval = GuardInterval.GI_1_8
    
    # Modulation (must match encoder)
    modulation: ModulationScheme = ModulationScheme.QPSK
    
    # FEC parameters (must match encoder)
    fec_ldpc_rate: FECCodeRate = FECCodeRate.RATE_8_15
    fec_rs_symbols: int = 16
    
    # Decoder options
    soft_decision: bool = True  # Use soft-decision decoding
    max_ldpc_iterations: int = 50  # Max LDPC decoder iterations
    error_correction: bool = True  # Enable error correction
    
    # Debug options
    verbose: bool = False


@dataclass
class DecoderMetrics:
    """Metrics from decoding process."""
    
    # Timing
    total_time_ms: float = 0.0
    ofdm_demod_time_ms: float = 0.0
    frame_parse_time_ms: float = 0.0
    fec_decode_time_ms: float = 0.0
    alp_decode_time_ms: float = 0.0
    
    # Signal quality
    estimated_snr_db: float = 0.0
    estimated_ber: float = 0.0
    evm_percent: float = 0.0  # Error Vector Magnitude
    
    # FEC performance
    ldpc_iterations: int = 0
    ldpc_corrections: int = 0
    rs_corrections: int = 0
    uncorrectable_errors: int = 0
    
    # Data statistics
    ofdm_symbols_processed: int = 0
    frames_decoded: int = 0
    alp_packets_recovered: int = 0
    bytes_recovered: int = 0
    
    # Verification
    crc_passed: bool = False
    data_valid: bool = False


@dataclass
class DecoderResult:
    """Result from broadcast decoding."""
    
    success: bool = False
    data: bytes = b""
    
    # Detailed outputs
    alp_packets: List[bytes] = field(default_factory=list)
    
    # Metrics
    metrics: DecoderMetrics = field(default_factory=DecoderMetrics)
    
    # Error information
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)


# ----------------------------
# OFDM Demodulator
# ----------------------------


class OFDMDemodulator:
    """
    OFDM Demodulation stage.
    
    Converts time-domain OFDM signal back to frequency-domain symbols,
    then demaps to bits.
    """
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.fft_size = self._get_fft_size()
        self.guard_samples = self._get_guard_samples()
        
        # Build constellation for demapping
        self.constellation = self._build_constellation()
    
    def _get_fft_size(self) -> int:
        """Get FFT size as integer."""
        if hasattr(self.config.fft_size, 'value'):
            val = self.config.fft_size.value
            if isinstance(val, int):
                return val
        # Parse from name
        name = self.config.fft_size.name
        if "8K" in name:
            return 8192
        elif "16K" in name:
            return 16384
        elif "32K" in name:
            return 32768
        return 8192
    
    def _get_guard_samples(self) -> int:
        """Calculate guard interval samples."""
        if hasattr(self.config.guard_interval, 'value'):
            ratio = self.config.guard_interval.value
            if isinstance(ratio, float):
                return int(self.fft_size * ratio)
        # Parse from name
        name = self.config.guard_interval.name
        if "1_4" in name:
            return self.fft_size // 4
        elif "1_8" in name:
            return self.fft_size // 8
        elif "1_16" in name:
            return self.fft_size // 16
        elif "1_32" in name:
            return self.fft_size // 32
        return self.fft_size // 8
    
    def _build_constellation(self) -> np.ndarray:
        """Build constellation points for demapping."""
        mod = self.config.modulation
        
        if mod == ModulationScheme.QPSK or mod.name == "QPSK":
            # QPSK: 4 points
            scale = 1.0 / np.sqrt(2)
            return np.array([
                complex(1, 1), complex(-1, 1),
                complex(1, -1), complex(-1, -1)
            ]) * scale
        
        elif mod == ModulationScheme.QAM16 or mod.name == "QAM16":
            # 16-QAM: 16 points
            scale = 1.0 / np.sqrt(10)
            points = []
            for i in [-3, -1, 1, 3]:
                for q in [-3, -1, 1, 3]:
                    points.append(complex(i, q))
            return np.array(points) * scale
        
        elif mod == ModulationScheme.QAM64 or mod.name == "QAM64":
            # 64-QAM: 64 points
            scale = 1.0 / np.sqrt(42)
            points = []
            for i in [-7, -5, -3, -1, 1, 3, 5, 7]:
                for q in [-7, -5, -3, -1, 1, 3, 5, 7]:
                    points.append(complex(i, q))
            return np.array(points) * scale
        
        elif mod == ModulationScheme.QAM256 or mod.name == "QAM256":
            # 256-QAM: 256 points
            scale = 1.0 / np.sqrt(170)
            points = []
            levels = list(range(-15, 16, 2))
            for i in levels:
                for q in levels:
                    points.append(complex(i, q))
            return np.array(points) * scale
        
        # Default to QPSK
        scale = 1.0 / np.sqrt(2)
        return np.array([
            complex(1, 1), complex(-1, 1),
            complex(1, -1), complex(-1, -1)
        ]) * scale
    
    def _get_bits_per_symbol(self) -> int:
        """Get bits per symbol for current modulation."""
        mod = self.config.modulation
        if hasattr(mod, 'value') and isinstance(mod.value, int):
            return mod.value
        name = mod.name if hasattr(mod, 'name') else str(mod)
        if "QPSK" in name:
            return 2
        elif "16" in name:
            return 4
        elif "64" in name:
            return 6
        elif "256" in name:
            return 8
        return 2
    
    def demodulate(
        self,
        signal: Union[np.ndarray, bytes, 'OFDMSignal']
    ) -> Tuple[np.ndarray, DecoderMetrics]:
        """
        Demodulate OFDM signal to bits.
        
        Args:
            signal: OFDM time-domain signal (complex samples)
        
        Returns:
            Tuple of (recovered_bits, metrics)
        """
        metrics = DecoderMetrics()
        start_time = time.time()
        
        # Convert input to numpy array
        samples = self._convert_to_samples(signal)
        
        if samples is None or len(samples) == 0:
            logger.error("No valid samples to demodulate")
            return np.array([], dtype=np.uint8), metrics
        
        # Symbol parameters
        symbol_length = self.fft_size + self.guard_samples
        num_symbols = len(samples) // symbol_length
        
        if num_symbols == 0:
            logger.warning(f"Signal too short: {len(samples)} samples, need {symbol_length}")
            return np.array([], dtype=np.uint8), metrics
        
        metrics.ofdm_symbols_processed = num_symbols
        
        # Process each OFDM symbol
        all_bits = []
        total_evm = 0.0
        
        for sym_idx in range(num_symbols):
            # Extract symbol (skip guard interval)
            start = sym_idx * symbol_length + self.guard_samples
            end = start + self.fft_size
            
            if end > len(samples):
                break
            
            symbol_samples = samples[start:end]
            
            # FFT to get frequency domain
            freq_domain = np.fft.fft(symbol_samples)
            
            # Extract data subcarriers (simplified - use center portion)
            # Real ATSC 3.0 has specific subcarrier allocation
            num_data_carriers = self.fft_size // 2  # Simplified
            data_carriers = freq_domain[1:num_data_carriers + 1]
            
            # Demap symbols to bits
            bits, evm = self._demap_symbols(data_carriers)
            all_bits.append(bits)
            total_evm += evm
        
        # Combine all bits
        if all_bits:
            recovered_bits = np.concatenate(all_bits)
            metrics.evm_percent = total_evm / len(all_bits)
        else:
            recovered_bits = np.array([], dtype=np.uint8)
        
        metrics.ofdm_demod_time_ms = (time.time() - start_time) * 1000
        
        # Estimate SNR from EVM
        if metrics.evm_percent > 0:
            metrics.estimated_snr_db = -20 * np.log10(metrics.evm_percent / 100)
        
        return recovered_bits, metrics
    
    def _convert_to_samples(
        self,
        signal: Union[np.ndarray, bytes, 'OFDMSignal', Any]
    ) -> Optional[np.ndarray]:
        """Convert various input types to numpy array of complex samples."""
        
        if signal is None:
            return None
        
        # Handle OFDMSignal object
        if hasattr(signal, 'samples'):
            samples = signal.samples
            if isinstance(samples, np.ndarray):
                return samples.astype(np.complex64)
            elif isinstance(samples, bytes):
                return np.frombuffer(samples, dtype=np.complex64)
        
        # Handle numpy array
        if isinstance(signal, np.ndarray):
            if signal.dtype in [np.complex64, np.complex128]:
                return signal.astype(np.complex64)
            elif signal.dtype in [np.float32, np.float64]:
                # Assume interleaved I/Q
                if len(signal) % 2 == 0:
                    return (signal[::2] + 1j * signal[1::2]).astype(np.complex64)
            return signal.astype(np.complex64)
        
        # Handle bytes
        if isinstance(signal, bytes):
            try:
                return np.frombuffer(signal, dtype=np.complex64)
            except ValueError:
                # Try as float32 interleaved I/Q
                try:
                    floats = np.frombuffer(signal, dtype=np.float32)
                    if len(floats) % 2 == 0:
                        return (floats[::2] + 1j * floats[1::2]).astype(np.complex64)
                except ValueError:
                    pass
        
        logger.warning(f"Unknown signal type: {type(signal)}")
        return None
    
    def _demap_symbols(
        self,
        symbols: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Demap complex symbols to bits using nearest-neighbor.
        
        Returns:
            Tuple of (bits, evm_percent)
        """
        bits_per_symbol = self._get_bits_per_symbol()
        num_constellation_points = 2 ** bits_per_symbol
        
        # Ensure constellation matches
        if len(self.constellation) != num_constellation_points:
            self.constellation = self._build_constellation()
        
        bits = []
        total_error = 0.0
        
        for sym in symbols:
            # Find nearest constellation point
            distances = np.abs(sym - self.constellation)
            nearest_idx = np.argmin(distances)
            
            # Calculate EVM contribution
            total_error += distances[nearest_idx] ** 2
            
            # Convert index to bits (Gray coding simplified)
            for bit_pos in range(bits_per_symbol - 1, -1, -1):
                bits.append((nearest_idx >> bit_pos) & 1)
        
        # Calculate EVM
        if len(symbols) > 0:
            rms_error = np.sqrt(total_error / len(symbols))
            rms_signal = np.sqrt(np.mean(np.abs(self.constellation) ** 2))
            evm_percent = (rms_error / rms_signal) * 100 if rms_signal > 0 else 0
        else:
            evm_percent = 0
        
        return np.array(bits, dtype=np.uint8), evm_percent


# ----------------------------
# Frame Parser
# ----------------------------


class FrameParser:
    """
    ATSC 3.0 Frame Parser.
    
    Extracts payload data from ATSC frame structure.
    """
    
    # Frame markers (simplified)
    FRAME_SYNC = bytes([0x47, 0x1F, 0xFF, 0x10])  # Example sync pattern
    
    def __init__(self, config: DecoderConfig):
        self.config = config
    
    def parse(
        self,
        bits: np.ndarray,
        metrics: DecoderMetrics
    ) -> Tuple[bytes, DecoderMetrics]:
        """
        Parse frame structure and extract payload.
        
        Args:
            bits: Bit array from OFDM demodulator
            metrics: Metrics object to update
        
        Returns:
            Tuple of (payload_bytes, updated_metrics)
        """
        start_time = time.time()
        
        if len(bits) == 0:
            return b"", metrics
        
        # Convert bits to bytes
        num_bytes = len(bits) // 8
        if num_bytes == 0:
            return b"", metrics
        
        # Pack bits into bytes
        byte_data = self._bits_to_bytes(bits[:num_bytes * 8])
        
        # Find and extract payload (simplified)
        # Real implementation would parse L1 signaling, PLP headers, etc.
        payload = self._extract_payload(byte_data)
        
        metrics.frames_decoded = 1  # Simplified: treat as single frame
        metrics.frame_parse_time_ms = (time.time() - start_time) * 1000
        
        return payload, metrics
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes."""
        num_bytes = len(bits) // 8
        byte_list = []
        
        for i in range(num_bytes):
            byte_val = 0
            for bit_idx in range(8):
                byte_val = (byte_val << 1) | bits[i * 8 + bit_idx]
            byte_list.append(byte_val)
        
        return bytes(byte_list)
    
    def _extract_payload(self, frame_data: bytes) -> bytes:
        """
        Extract payload from frame data.
        
        Simplified implementation - real ATSC 3.0 has complex frame structure.
        """
        if len(frame_data) < 8:
            return frame_data
        
        # Look for frame sync or header patterns
        # For PoC, we'll assume the data is the payload
        # Real implementation would parse:
        # - Bootstrap
        # - L1-Basic signaling
        # - L1-Detail signaling
        # - PLP data
        
        return frame_data


# ----------------------------
# FEC Decoder
# ----------------------------


class FECDecoder:
    """
    Forward Error Correction Decoder.
    
    Implements LDPC and Reed-Solomon decoding to correct errors.
    """
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.code_rate = self._get_code_rate()
    
    def _get_code_rate(self) -> float:
        """Get code rate as float."""
        rate = self.config.fec_ldpc_rate
        if hasattr(rate, 'value') and isinstance(rate.value, tuple):
            num, den = rate.value
            return num / den
        # Parse from name
        name = rate.name if hasattr(rate, 'name') else str(rate)
        if "2_15" in name:
            return 2/15
        elif "4_15" in name:
            return 4/15
        elif "6_15" in name:
            return 6/15
        elif "8_15" in name:
            return 8/15
        return 0.5
    
    def decode(
        self,
        encoded_data: bytes,
        metrics: DecoderMetrics
    ) -> Tuple[bytes, DecoderMetrics]:
        """
        Decode FEC-encoded data.
        
        Args:
            encoded_data: FEC-encoded bytes
            metrics: Metrics object to update
        
        Returns:
            Tuple of (decoded_data, updated_metrics)
        """
        start_time = time.time()
        
        if len(encoded_data) == 0:
            return b"", metrics
        
        # Step 1: Reed-Solomon decoding (outer code)
        rs_decoded, rs_corrections = self._decode_reed_solomon(encoded_data)
        metrics.rs_corrections = rs_corrections
        
        # Step 2: LDPC decoding (inner code)
        ldpc_decoded, ldpc_iterations, ldpc_corrections = self._decode_ldpc(rs_decoded)
        metrics.ldpc_iterations = ldpc_iterations
        metrics.ldpc_corrections = ldpc_corrections
        
        metrics.fec_decode_time_ms = (time.time() - start_time) * 1000
        
        return ldpc_decoded, metrics
    
    def _decode_reed_solomon(self, data: bytes) -> Tuple[bytes, int]:
        """
        Simplified Reed-Solomon decoding.
        
        Real implementation would use proper RS decoder (e.g., from reedsolo library).
        For PoC, we simulate the decoding process.
        """
        rs_symbols = self.config.fec_rs_symbols
        
        if len(data) <= rs_symbols:
            return data, 0
        
        # Simplified: Remove RS parity bytes (in real decoder, would correct errors)
        # RS(n, k) where n = block_size, k = data_size, parity = n - k
        
        # Calculate expected data length
        block_size = 255  # Standard RS block size
        data_size = block_size - rs_symbols
        
        if len(data) < block_size:
            # Short block, just strip estimated parity
            parity_ratio = rs_symbols / block_size
            strip_bytes = int(len(data) * parity_ratio)
            return data[:-strip_bytes] if strip_bytes > 0 else data, 0
        
        # Process full blocks
        decoded_blocks = []
        corrections = 0
        
        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]
            if len(block) == block_size:
                # Strip parity
                decoded_blocks.append(block[:data_size])
            else:
                # Partial block
                partial_parity = int(len(block) * rs_symbols / block_size)
                decoded_blocks.append(block[:-partial_parity] if partial_parity > 0 else block)
        
        return b"".join(decoded_blocks), corrections
    
    def _decode_ldpc(self, data: bytes) -> Tuple[bytes, int, int]:
        """
        Simplified LDPC decoding.
        
        Real implementation would use iterative belief propagation.
        For PoC, we simulate the expansion ratio reversal.
        """
        if len(data) == 0:
            return b"", 0, 0
        
        # Calculate data portion based on code rate
        code_rate = self.code_rate
        data_length = int(len(data) * code_rate)
        
        if data_length <= 0:
            data_length = len(data)
        
        # Simplified: Extract data portion (first data_length bytes)
        # Real LDPC would use parity check matrix for iterative decoding
        decoded = data[:data_length]
        
        # Simulate some iterations
        iterations = min(self.config.max_ldpc_iterations, 10)
        corrections = 0  # Would track actual corrections in real decoder
        
        return decoded, iterations, corrections


# ----------------------------
# ALP Decoder
# ----------------------------


class ALPDecoder:
    """
    ATSC Link-layer Protocol (ALP) Decoder.
    
    Extracts original data packets from ALP encapsulation.
    """
    
    # ALP packet types
    ALP_TYPE_IPV4 = 0x00
    ALP_TYPE_SIGNALING = 0x04
    ALP_TYPE_EXTENSION = 0x07
    
    def __init__(self, config: DecoderConfig):
        self.config = config
    
    def decode(
        self,
        data: bytes,
        metrics: DecoderMetrics
    ) -> Tuple[bytes, List[bytes], DecoderMetrics]:
        """
        Decode ALP-encapsulated data.
        
        Args:
            data: ALP-encoded bytes
            metrics: Metrics object to update
        
        Returns:
            Tuple of (combined_payload, individual_packets, updated_metrics)
        """
        start_time = time.time()
        
        if len(data) == 0:
            return b"", [], metrics
        
        packets = []
        offset = 0
        
        while offset < len(data):
            # Try to parse ALP header
            packet, consumed = self._parse_alp_packet(data[offset:])
            
            if packet is not None and len(packet) > 0:
                packets.append(packet)
                metrics.alp_packets_recovered += 1
            
            if consumed <= 0:
                # Can't parse more, take remaining as raw data
                remaining = data[offset:]
                if len(remaining) > 0:
                    packets.append(remaining)
                break
            
            offset += consumed
        
        # Combine packets
        combined = b"".join(packets)
        metrics.bytes_recovered = len(combined)
        metrics.alp_decode_time_ms = (time.time() - start_time) * 1000
        
        return combined, packets, metrics
    
    def _parse_alp_packet(self, data: bytes) -> Tuple[Optional[bytes], int]:
        """
        Parse a single ALP packet.
        
        Returns:
            Tuple of (payload, bytes_consumed)
        """
        if len(data) < 2:
            return None, 0
        
        # ALP header (simplified)
        # Byte 0: packet type (3 bits) + flags
        # Byte 1+: length and payload
        
        header = data[0]
        packet_type = (header >> 5) & 0x07
        
        # Try to find length
        if len(data) < 3:
            return data, len(data)
        
        # Length field (simplified - 16 bits)
        length = (data[1] << 8) | data[2]
        
        # Validate length
        if length > len(data) - 3 or length > 65535:
            # Invalid length, might not be ALP format
            # Return raw data
            return data, len(data)
        
        # Extract payload
        header_size = 3
        payload = data[header_size:header_size + length]
        consumed = header_size + length
        
        return payload, consumed


# ----------------------------
# Main Decoder Class
# ----------------------------


class BroadcastDecoder:
    """
    Complete ATSC 3.0 Broadcast Decoder.
    
    Orchestrates the full decoding pipeline:
    OFDM Demod → Frame Parse → FEC Decode → ALP Decode
    """
    
    def __init__(self, config: Optional[DecoderConfig] = None):
        """
        Initialize decoder with configuration.
        
        Args:
            config: Decoder configuration (uses defaults if None)
        """
        self.config = config or DecoderConfig()
        
        # Initialize pipeline stages
        self.ofdm_demod = OFDMDemodulator(self.config)
        self.frame_parser = FrameParser(self.config)
        self.fec_decoder = FECDecoder(self.config)
        self.alp_decoder = ALPDecoder(self.config)
        
        logger.info("BroadcastDecoder initialized")
        if self.config.verbose:
            logger.info(f"  FFT Size: {self.config.fft_size.name}")
            logger.info(f"  Modulation: {self.config.modulation.name}")
            logger.info(f"  FEC Rate: {self.config.fec_ldpc_rate.name}")
    
    def decode(
        self,
        signal: Union[np.ndarray, bytes, 'OFDMSignal', Any],
        config: Optional[DecoderConfig] = None
    ) -> DecoderResult:
        """
        Decode OFDM signal to recover original data.
        
        Args:
            signal: OFDM signal (complex samples, bytes, or OFDMSignal object)
            config: Optional config override
        
        Returns:
            DecoderResult with recovered data and metrics
        """
        result = DecoderResult()
        start_time = time.time()
        
        # Use provided config or instance config
        cfg = config or self.config
        
        try:
            # ========================================
            # Stage 1: OFDM Demodulation
            # ========================================
            logger.debug("Stage 1: OFDM Demodulation")
            bits, metrics = self.ofdm_demod.demodulate(signal)
            result.metrics = metrics
            
            if len(bits) == 0:
                result.error_message = "OFDM demodulation produced no bits"
                result.warnings.append("Signal may be too short or corrupted")
                return result
            
            logger.debug(f"  Recovered {len(bits)} bits from {metrics.ofdm_symbols_processed} symbols")
            
            # ========================================
            # Stage 2: Frame Parsing
            # ========================================
            logger.debug("Stage 2: Frame Parsing")
            frame_data, metrics = self.frame_parser.parse(bits, metrics)
            result.metrics = metrics
            
            if len(frame_data) == 0:
                result.error_message = "Frame parsing produced no data"
                return result
            
            logger.debug(f"  Extracted {len(frame_data)} bytes from frame")
            
            # ========================================
            # Stage 3: FEC Decoding
            # ========================================
            logger.debug("Stage 3: FEC Decoding")
            fec_decoded, metrics = self.fec_decoder.decode(frame_data, metrics)
            result.metrics = metrics
            
            if len(fec_decoded) == 0:
                result.error_message = "FEC decoding produced no data"
                return result
            
            logger.debug(f"  FEC decoded to {len(fec_decoded)} bytes")
            logger.debug(f"  LDPC iterations: {metrics.ldpc_iterations}")
            logger.debug(f"  RS corrections: {metrics.rs_corrections}")
            
            # ========================================
            # Stage 4: ALP Decapsulation
            # ========================================
            logger.debug("Stage 4: ALP Decapsulation")
            payload, packets, metrics = self.alp_decoder.decode(fec_decoded, metrics)
            result.metrics = metrics
            result.alp_packets = packets
            
            if len(payload) == 0:
                result.error_message = "ALP decoding produced no data"
                return result
            
            logger.debug(f"  Recovered {len(payload)} bytes in {len(packets)} packets")
            
            # ========================================
            # Success
            # ========================================
            result.data = payload
            result.success = True
            result.metrics.data_valid = True
            
        except Exception as e:
            result.error_message = f"Decoding failed: {str(e)}"
            logger.exception("Decoder error")
        
        finally:
            result.metrics.total_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def decode_from_file(self, file_path: str) -> DecoderResult:
        """
        Decode OFDM signal from binary file.
        
        Args:
            file_path: Path to binary file containing complex samples
        
        Returns:
            DecoderResult with recovered data
        """
        try:
            with open(file_path, "rb") as f:
                signal_data = f.read()
            
            # Convert to numpy array
            samples = np.frombuffer(signal_data, dtype=np.complex64)
            
            return self.decode(samples)
        
        except Exception as e:
            result = DecoderResult()
            result.error_message = f"Failed to read file: {str(e)}"
            return result
    
    def verify_roundtrip(
        self,
        original_data: bytes,
        signal: Union[np.ndarray, bytes, 'OFDMSignal', Any]
    ) -> Tuple[bool, float, DecoderResult]:
        """
        Verify round-trip integrity by comparing original and decoded data.
        
        Args:
            original_data: Original data before encoding
            signal: Encoded OFDM signal
        
        Returns:
            Tuple of (match, similarity_ratio, decoder_result)
        """
        result = self.decode(signal)
        
        if not result.success:
            return False, 0.0, result
        
        # Compare data
        recovered = result.data
        
        # Exact match check
        if recovered == original_data:
            result.metrics.crc_passed = True
            return True, 1.0, result
        
        # Similarity check (for partial recovery)
        min_len = min(len(original_data), len(recovered))
        if min_len == 0:
            return False, 0.0, result
        
        matching_bytes = sum(
            1 for i in range(min_len)
            if original_data[i] == recovered[i]
        )
        similarity = matching_bytes / max(len(original_data), len(recovered))
        
        return similarity > 0.99, similarity, result


# ----------------------------
# Convenience Function
# ----------------------------


def decode_signal(
    signal: Union[np.ndarray, bytes],
    modulation: ModulationScheme = ModulationScheme.QPSK,
    fec_rate: FECCodeRate = FECCodeRate.RATE_8_15,
    fft_size: FFTSize = FFTSize.FFT_8K
) -> DecoderResult:
    """
    Convenience function for quick decoding.
    
    Args:
        signal: OFDM signal to decode
        modulation: Modulation scheme used
        fec_rate: FEC code rate used
        fft_size: FFT size used
    
    Returns:
        DecoderResult with recovered data
    """
    config = DecoderConfig(
        modulation=modulation,
        fec_ldpc_rate=fec_rate,
        fft_size=fft_size
    )
    
    decoder = BroadcastDecoder(config)
    return decoder.decode(signal)


# ----------------------------
# CLI for Testing
# ----------------------------


def main():
    """Command-line interface for testing decoder."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ATSC 3.0 Broadcast Decoder - Decode OFDM signals"
    )
    
    parser.add_argument(
        "input_file",
        help="Path to binary file containing OFDM signal (complex64)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file for recovered data"
    )
    
    parser.add_argument(
        "--modulation",
        choices=["QPSK", "QAM16", "QAM64", "QAM256"],
        default="QPSK",
        help="Modulation scheme (default: QPSK)"
    )
    
    parser.add_argument(
        "--fec-rate",
        choices=["RATE_2_15", "RATE_4_15", "RATE_6_15", "RATE_8_15"],
        default="RATE_8_15",
        help="FEC LDPC code rate (default: RATE_8_15)"
    )
    
    parser.add_argument(
        "--fft-size",
        choices=["FFT_8K", "FFT_16K", "FFT_32K"],
        default="FFT_8K",
        help="FFT size (default: FFT_8K)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Build config
    config = DecoderConfig(
        modulation=ModulationScheme[args.modulation],
        fec_ldpc_rate=FECCodeRate[args.fec_rate],
        fft_size=FFTSize[args.fft_size],
        verbose=args.verbose
    )
    
    # Create decoder
    decoder = BroadcastDecoder(config)
    
    # Decode
    print(f"Decoding: {args.input_file}")
    result = decoder.decode_from_file(args.input_file)
    
    # Report results
    print("\n" + "=" * 50)
    print("DECODER RESULTS")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Recovered bytes: {len(result.data)}")
    print(f"ALP packets: {result.metrics.alp_packets_recovered}")
    print(f"Processing time: {result.metrics.total_time_ms:.2f} ms")
    
    if result.metrics.estimated_snr_db > 0:
        print(f"Estimated SNR: {result.metrics.estimated_snr_db:.1f} dB")
    
    if result.metrics.evm_percent > 0:
        print(f"EVM: {result.metrics.evm_percent:.2f}%")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    for warning in result.warnings:
        print(f"Warning: {warning}")
    
    # Save output
    if args.output and result.success:
        with open(args.output, "wb") as f:
            f.write(result.data)
        print(f"\nSaved recovered data to: {args.output}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())