#!/usr/bin/env python3
"""
ATSC 3.0 Broadcast Decoder - PATCHED VERSION
Added: Length header parsing when ALP disabled
"""

from __future__ import annotations
import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import struct  # ← ADDED for length header
import reedsolo
from typing import Optional, Tuple, List

from broadcast.config import (
    FECCodeRate, ModulationScheme, FFTSize, GuardInterval, PilotPattern
)

logger = logging.getLogger(__name__)

@dataclass
class DecoderConfig:
    """Configuration for broadcast decoder - must match encoder settings."""
    fec_ldpc_rate: FECCodeRate = FECCodeRate.RATE_8_15
    fec_rs_symbols: int = 16
    modulation: ModulationScheme = ModulationScheme.QPSK
    fft_size: FFTSize = FFTSize.FFT_8K
    guard_interval: GuardInterval = GuardInterval.GI_1_8
    pilot_pattern: PilotPattern = PilotPattern.SP3_4
    pilots_enabled: bool = True
    time_interleaving: bool = False
    time_interleaving_depth: int = 1024


@dataclass
class DecoderMetrics:
    """Metrics from decoding process."""
    estimated_snr_db: float = 0.0
    evm_percent: float = 0.0
    estimated_ber: float = 0.0
    fec_errors_corrected: int = 0
    rs_errors_corrected: int = 0
    ldpc_iterations: int = 0
    symbols_processed: int = 0


@dataclass
class DecoderResult:
    """Result of decoding operation."""
    success: bool
    data: Optional[bytes]
    error_message: str = ""
    metrics: DecoderMetrics = field(default_factory=DecoderMetrics)


class BroadcastDecoder:
    """
    ATSC 3.0 Broadcast Decoder.
    
    Reverses the encoding pipeline to recover original data from OFDM signal.
    """
    
    # LDPC code parameters (must match encoder)
    LDPC_CODES = {
        FECCodeRate.RATE_2_15: (16200, 2160),
        FECCodeRate.RATE_3_15: (16200, 3240),
        FECCodeRate.RATE_4_15: (16200, 4320),
        FECCodeRate.RATE_5_15: (16200, 5400),
        FECCodeRate.RATE_6_15: (16200, 6480),
        FECCodeRate.RATE_7_15: (16200, 7560),
        FECCodeRate.RATE_8_15: (16200, 8640),
        FECCodeRate.RATE_9_15: (16200, 9720),
        FECCodeRate.RATE_10_15: (16200, 10800),
        FECCodeRate.RATE_11_15: (16200, 11880),
        FECCodeRate.RATE_12_15: (16200, 12960),
        FECCodeRate.RATE_13_15: (16200, 14040),
    }
    
    # Carrier allocation (must match encoder)
    CARRIER_ALLOCATION = {
        FFTSize.FFT_8K: {
            'total': 8192,
            'active': 6913,
            'guard_lower': 639,
            'guard_upper': 639,
            'dc': 1,
        },
        FFTSize.FFT_16K: {
            'total': 16384,
            'active': 13825,
            'guard_lower': 1279,
            'guard_upper': 1279,
            'dc': 1,
        },
        FFTSize.FFT_32K: {
            'total': 32768,
            'active': 27649,
            'guard_lower': 2559,
            'guard_upper': 2559,
            'dc': 1,
        },
    }
    
    # Guard interval fractions
    GI_FRACTIONS = {
        GuardInterval.GI_1_4: 0.25,
        GuardInterval.GI_1_8: 0.125,
        GuardInterval.GI_1_16: 0.0625,
        GuardInterval.GI_1_32: 0.03125,
    }
    
    # QPSK constellation (must match encoder exactly)
    QPSK_CONSTELLATION = np.array([
        -0.70710677-0.70710677j,  # index 0 - bits 00
        -0.70710677+0.70710677j,  # index 1 - bits 01
         0.70710677-0.70710677j,  # index 2 - bits 10
         0.70710677+0.70710677j,  # index 3 - bits 11
    ], dtype=np.complex64)
    
    def __init__(self, config: Optional[DecoderConfig] = None):
        """Initialize decoder with configuration."""
        self.config = config or DecoderConfig()
        self.metrics = DecoderMetrics()
        self._ldpc_matrices = {}
        # Calculate derived parameters
        self.num_carriers = self.config.fft_size.value
        self.gi_fraction = self.GI_FRACTIONS[self.config.guard_interval]
        self.cp_length = int(self.num_carriers * self.gi_fraction)
        self.symbol_length = self.num_carriers + self.cp_length
        
        # Get carrier allocation
        self.carrier_info = self.CARRIER_ALLOCATION[self.config.fft_size]
        self.guard_size = self.carrier_info['guard_lower']
        self.active_carriers = self.carrier_info['active']
        self.dc_index = self.num_carriers // 2
        
        # Calculate data carriers per symbol
        if self.config.pilots_enabled:
            dx, dy = self.config.pilot_pattern.value
            pilots_per_symbol = self.active_carriers // dx
            self.data_carriers_per_symbol = self.active_carriers - pilots_per_symbol
        else:
            self.data_carriers_per_symbol = self.active_carriers
        
        # Bits per symbol for modulation
        self.bits_per_symbol = {
            ModulationScheme.BPSK: 1,
            ModulationScheme.QPSK: 2,
            ModulationScheme.QAM16: 4,
            ModulationScheme.QAM64: 6,
            ModulationScheme.QAM256: 8,
        }[self.config.modulation]
        self._ldpc_matrices = {}
        logger.info("BroadcastDecoder initialized")
    
    def qpsk_llrs(self, symbols: np.ndarray, noise_var: float) -> np.ndarray:
        """Compute QPSK LLRs from received symbols.
        
        Args:
            symbols: Equalized complex symbols
            noise_var: Noise variance estimate
            
        Returns:
            LLR array (2 LLRs per symbol)
        """
        scale = 2.0 / max(noise_var, 1e-8)
        llr_I = -scale * np.real(symbols)  # >0 -> bit 0
        llr_Q = -scale * np.imag(symbols)
        # Map to your bit order: [b0, b1] per symbol
        return np.vstack([llr_I, llr_Q]).T.reshape(-1)

    def decode(self, signal: np.ndarray, expected_fec_len: Optional[int] = None) -> DecoderResult:
        """Decode OFDM signal to recover original data.
    
    Args:
        signal: Received complex samples
        expected_fec_len: Expected FEC encoded length in bytes (from TX metadata)
    
    Returns:
        DecoderResult with decoded data
        """

        try:
            if signal.dtype != np.complex64:
                signal = signal.astype(np.complex64)
            
            logger.info(f"Decoding signal: {len(signal)} samples")
            
            # Step 1: OFDM Demodulation
            payload_bytes, payload_llrs, snr_estimate, evm = self.ofdm_demodulate(signal)           
            self.metrics.estimated_snr_db = snr_estimate
            self.metrics.evm_percent = evm
            logger.debug(f"OFDM demodulated: {len(payload_bytes)} bytes")
            
            # Step 2: Parse frame (de-interleave only, no preamble removal)
            fec_encoded_bytes, fec_encoded_llrs = self.parse_frame(payload_bytes, payload_llrs)
            logger.debug(f"Frame parsed: {len(fec_encoded_bytes)} bytes")

            # NEW: Trim to exact FEC encoded length if provided by TX metadata
            if expected_fec_len is not None:
                if len(fec_encoded_bytes) < expected_fec_len:
                    logger.error(f"Insufficient data: got {len(fec_encoded_bytes)} bytes, expected {expected_fec_len}")
                else:
                    logger.info(f"Trimming to exact FEC length: {len(fec_encoded_bytes)} → {expected_fec_len} bytes")
                    fec_encoded_bytes = fec_encoded_bytes[:expected_fec_len]
                    fec_encoded_llrs = fec_encoded_llrs[:expected_fec_len * 8]  # 8 bits per byte
            
            # Step 3: FEC Decoding
            decoded_bytes, fec_errors = self.decode_fec(fec_encoded_bytes, fec_encoded_llrs)           
            self.metrics.fec_errors_corrected = fec_errors
            logger.debug(f"FEC decoded: {len(decoded_bytes)} bytes")
            
            # ═══════════════════════════════════════════════════════════
            # PATCH 2: Read length header when ALP disabled
            # ═══════════════════════════════════════════════════════════
            # Step 4: Decapsulate (ALP or length header)
            original_data = self.decapsulate(decoded_bytes)
            # ═══════════════════════════════════════════════════════════
            
            logger.info(f"Decoding complete: recovered {len(original_data)} bytes")
            
            return DecoderResult(
                success=True,
                data=original_data,
                metrics=self.metrics
            )
            
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            import traceback
            traceback.print_exc()
            return DecoderResult(
                success=False,
                data=None,
                error_message=str(e),
                metrics=self.metrics
            )
    
    def ofdm_demodulate(self, signal: np.ndarray) -> Tuple[bytes, np.ndarray, float, float]:        
        """OFDM demodulation - exact reverse of OFDMModulator."""
        num_symbols = len(signal) // self.symbol_length
        
        if num_symbols == 0:
            raise ValueError(f"Signal too short: {len(signal)} samples")
        
        all_data_symbols = []
        evm_accum = 0.0
        evm_count = 0
        
        for sym_idx in range(num_symbols):
            start = sym_idx * self.symbol_length
            symbol_with_cp = signal[start:start + self.symbol_length]
            
            # Remove cyclic prefix
            symbol = symbol_with_cp[self.cp_length:]
            
            # 1. FFT to get frequency domain
            freq_domain = np.fft.fft(symbol)
            
            # 2. fftshift to move DC to center (match encoder's ifftshift before ifft)
            freq_domain = np.fft.fftshift(freq_domain)
            
            # Extract data carriers
            data_symbols, sym_evm = self.extract_data_carriers(freq_domain, sym_idx)
            all_data_symbols.extend(data_symbols)
            
            if sym_evm > 0:
                evm_accum += sym_evm
                evm_count += 1
        
        avg_evm = (evm_accum / evm_count) if evm_count > 0 else 0.0
        
        if avg_evm > 0:
            snr_db = 20 * np.log10(100 / avg_evm)
        else:
            snr_db = 30.0
        snr_db = float(np.clip(snr_db, 0, 40))
        
        # NEW (CORRECT):
        data_symbols_array = np.array(all_data_symbols, dtype=np.complex64)
        self.metrics.symbols_processed = len(data_symbols_array)

# Compute noise variance for LLR scaling
        snr_lin = 10**(snr_db / 10.0)
        noise_var = 1.0 / max(snr_lin, 1e-6)

# Get LLRs from symbols
        payload_llrs = self.qpsk_llrs(data_symbols_array, noise_var)

# Also get hard bits for byte packing (needed for byte-based deinterleaving)
        bits = self.demap_symbols(data_symbols_array)
        num_bytes = len(bits) // 8
        payload_bytes = np.packbits(bits[:num_bytes * 8]).tobytes()

        return payload_bytes, payload_llrs, snr_db, float(avg_evm)
    
    def extract_data_carriers(self, freq_domain: np.ndarray, sym_idx: int) -> Tuple[List[complex], float]:
        """
    Extract data carriers from frequency domain with channel equalization.
    
    This method:
    1. Extracts pilot carriers and estimates channel response
    2. Interpolates channel estimate across all carriers
    3. Equalizes data carriers using zero-forcing equalization
    4. Returns equalized data symbols
        """
        data_symbols = []
        pilot_info = []  # (carrier_idx, pilot_value)
    
        dx, dy = self.config.pilot_pattern.value
    
    # ========================================
    # PASS 1: Collect pilots for channel estimation
    # ========================================
        carrier_idx = self.guard_size
        active_count = 0
    
        while active_count < self.active_carriers:
            if carrier_idx == self.dc_index:
                carrier_idx += 1
                continue
            
            if carrier_idx >= self.num_carriers - self.guard_size:
                break
        
        # Check if this is a pilot position
            is_pilot = False
            if self.config.pilots_enabled:
                if (active_count % dx == 0) and (sym_idx % dy == 0):
                    is_pilot = True
        
            if is_pilot:
            # Pilot reference value (must match transmitter)
            # Standard ATSC 3.0 uses BPSK pilots: +1 or -1
            # For simplicity, assume all pilots are 1+0j
                pilot_ref = complex(1.0, 0.0)
                pilot_received = freq_domain[carrier_idx]
                pilot_info.append((carrier_idx, pilot_received, pilot_ref))
        
            carrier_idx += 1
            active_count += 1
    
    # ========================================
    # Estimate channel response
    # ========================================
        if len(pilot_info) > 0:
        # Extract pilot positions and estimates
            pilot_positions = np.array([p[0] for p in pilot_info])
            pilot_received = np.array([p[1] for p in pilot_info])
            pilot_ref = np.array([p[2] for p in pilot_info])
        
        # Channel estimate at pilot locations: H = Y / X
            channel_at_pilots = pilot_received / pilot_ref
        
        # For now, use simple flat-fading model (average all pilots)
        # Production code should use linear interpolation or more sophisticated methods
            channel_estimate = np.mean(channel_at_pilots)
        
            logger.debug(
            f"Channel estimation: {len(pilot_info)} pilots, "
            f"H_avg = {abs(channel_estimate):.4f} ∠ {np.angle(channel_estimate, deg=True):.1f}°"
            )
        else:
        # No pilots available - assume perfect channel
            channel_estimate = complex(1.0, 0.0)
            logger.debug("No pilots found, assuming perfect channel (H=1)")
    
    # ========================================
    # PASS 2: Extract and equalize data carriers
    # ========================================
        carrier_idx = self.guard_size
        active_count = 0
    
        while active_count < self.active_carriers:
            if carrier_idx == self.dc_index:
                carrier_idx += 1
                continue
            
            if carrier_idx >= self.num_carriers - self.guard_size:
                break
        
        # Check if pilot
            is_pilot = False
            if self.config.pilots_enabled:
                if (active_count % dx == 0) and (sym_idx % dy == 0):
                    is_pilot = True
        
            if not is_pilot:
            # Get received symbol
                received_symbol = freq_domain[carrier_idx]
            
            # Zero-forcing equalization: Y_eq = Y / H
                if abs(channel_estimate) > 1e-6:  # Avoid division by zero
                    equalized_symbol = received_symbol / channel_estimate
                else:
                # Channel too weak, use received symbol directly
                    equalized_symbol = received_symbol
            
                data_symbols.append(equalized_symbol)
        
            carrier_idx += 1
            active_count += 1
    
    # ========================================
    # Compute EVM on equalized symbols
    # ========================================
            if len(data_symbols) > 0:
                symbols_array = np.array(data_symbols)
                evm = self.compute_evm(symbols_array)
        else:
            evm = 0.0
    
        return data_symbols, evm

    def compute_evm(self, symbols: np.ndarray) -> float:
        """Compute EVM by comparing to nearest constellation points."""
        if len(symbols) == 0:
            return 0.0
        
        constellation = self.QPSK_CONSTELLATION
        error_sum = 0.0
        
        for sym in symbols:
            distances = np.abs(sym - constellation)
            min_dist = np.min(distances)
            error_sum += min_dist ** 2
        
        avg_power = np.mean(np.abs(constellation) ** 2)
        if avg_power > 0:
            evm = 100 * np.sqrt(error_sum / len(symbols)) / np.sqrt(avg_power)
        else:
            evm = 0.0
        
        return float(evm)
    
    def demap_symbols(self, symbols: np.ndarray) -> np.ndarray:
        """Demap constellation symbols to bits."""
        if len(symbols) == 0:
            return np.array([], dtype=np.uint8)
        
        if self.config.modulation == ModulationScheme.QPSK:
            return self.demap_qpsk(symbols)
        elif self.config.modulation == ModulationScheme.QAM16:
            return self.demap_qam16(symbols)
        elif self.config.modulation == ModulationScheme.QAM64:
            return self.demap_qam64(symbols)
        elif self.config.modulation == ModulationScheme.QAM256:
            return self.demap_qam256(symbols)
        else:
            return (np.real(symbols) > 0).astype(np.uint8)
    
    def demap_qpsk(self, symbols: np.ndarray) -> np.ndarray:
        """QPSK demapping."""
        bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
        for i, sym in enumerate(symbols):
            distances = np.abs(sym - self.QPSK_CONSTELLATION)
            min_idx = np.argmin(distances)
            bits[i*2] = (min_idx >> 1) & 1
            bits[i*2 + 1] = min_idx & 1
    
    # FIXED: Invert bits to match TX convention
        return 1 - bits  # ← ADD THIS LINE (flip all bits)
    
    def demap_qam16(self, symbols: np.ndarray) -> np.ndarray:
        """16-QAM demapping."""
        constellation = np.array([
            -0.94868326-0.94868326j, -0.94868326-0.31622776j,
            -0.94868326+0.94868326j, -0.94868326+0.31622776j,
            -0.31622776-0.94868326j, -0.31622776-0.31622776j,
            -0.31622776+0.94868326j, -0.31622776+0.31622776j,
             0.94868326-0.94868326j,  0.94868326-0.31622776j,
             0.94868326+0.94868326j,  0.94868326+0.31622776j,
             0.31622776-0.94868326j,  0.31622776-0.31622776j,
             0.31622776+0.94868326j,  0.31622776+0.31622776j,
        ], dtype=np.complex64)
        
        bits = np.zeros(len(symbols) * 4, dtype=np.uint8)
        for i, sym in enumerate(symbols):
            distances = np.abs(sym - constellation)
            min_idx = np.argmin(distances)
            for b in range(4):
                bits[i*4 + b] = (min_idx >> (3 - b)) & 1
        return bits
    
    def demap_qam64(self, symbols: np.ndarray) -> np.ndarray:
        """64-QAM demapping."""
        scale = 1.0 / np.sqrt(42)
        constellation = []
        for i in range(8):
            for q in range(8):
                real = (2*i - 7) * scale
                imag = (2*q - 7) * scale
                constellation.append(complex(real, imag))
        constellation = np.array(constellation, dtype=np.complex64)
        
        bits = np.zeros(len(symbols) * 6, dtype=np.uint8)
        for i, sym in enumerate(symbols):
            distances = np.abs(sym - constellation)
            min_idx = np.argmin(distances)
            for b in range(6):
                bits[i*6 + b] = (min_idx >> (5 - b)) & 1
        return bits
    
    def demap_qam256(self, symbols: np.ndarray) -> np.ndarray:
        """256-QAM demapping."""
        scale = 1.0 / np.sqrt(170)
        constellation = []
        for i in range(16):
            for q in range(16):
                real = (2*i - 15) * scale
                imag = (2*q - 15) * scale
                constellation.append(complex(real, imag))
        constellation = np.array(constellation, dtype=np.complex64)
        
        bits = np.zeros(len(symbols) * 8, dtype=np.uint8)
        for i, sym in enumerate(symbols):
            distances = np.abs(sym - constellation)
            min_idx = np.argmin(distances)
            for b in range(8):
                bits[i*8 + b] = (min_idx >> (7 - b)) & 1
        return bits
    
    def parse_frame(self, payload_bytes: bytes, payload_llrs: np.ndarray) -> Tuple[bytes, np.ndarray]:
        """Parse frame structure and deinterleave both bytes and LLRs."""
        data_bytes = payload_bytes
        data_llrs = payload_llrs
    
    # De-interleave if enabled
        if self.config.time_interleaving and self.config.time_interleaving_depth > 0:
            data_bytes = self.deinterleave(data_bytes)
            data_llrs = self.deinterleave_llrs(data_llrs, len(payload_bytes))
    
        return data_bytes, data_llrs

    
    def deinterleave(self, data: bytes) -> bytes:
        """Reverse time interleaving - must match framebuilder exactly."""
        depth = self.config.time_interleaving_depth
    
        if len(data) < depth or depth <= 1:
            return data
    
        data_array = np.frombuffer(data, dtype=np.uint8)
    
    # Match framebuilder: process ENTIRE payload as single matrix
        rows = len(data_array) // depth
        remainder = len(data_array) % depth
    
        if rows > 0:
        # Deinterleave the complete rows
            matrix_data = data_array[:rows * depth]
            reshaped = matrix_data.reshape(depth, rows)  # Note: depth × rows (transposed from TX)
            deinterleaved = reshaped.T.flatten()
        
        # Append remainder unchanged
            if remainder > 0:
                result = np.concatenate([deinterleaved, data_array[rows * depth:]])
            else:
                result = deinterleaved
        
            return bytes(result)
    
        return data
    
    def deinterleave_llrs(self, llrs: np.ndarray, num_bytes: int) -> np.ndarray:
        depth = self.config.time_interleaving_depth
        if depth <= 1 or num_bytes < depth:
            return llrs

    # only keep llrs for full bytes
        llrs = llrs[:num_bytes*8]

    # build byte index permutation same as bytes deinterleave
        rows = num_bytes // depth
        remainder = num_bytes % depth
        if rows <= 0:
            return llrs

        main = np.arange(rows*depth, dtype=np.int32)
        reshaped = main.reshape(depth, rows).T.flatten()  # same mapping as bytes

        byte_perm = np.concatenate([reshaped, np.arange(rows*depth, rows*depth+remainder, dtype=np.int32)])

        llrs_2d = llrs.reshape(num_bytes, 8)
        llrs_out = llrs_2d[byte_perm].reshape(-1)
        return llrs_out


    def decode_fec(self, fec_bytes: bytes, fec_llrs: np.ndarray) -> Tuple[bytes, int]:
        """
    Decode FEC (RS only - LDPC bypassed for demo).
        """
        total_errors = 0
    
    # BYPASS LDPC - use hard decisions from LLRs
    #logger.info(f"[DEMO MODE] LDPC bypassed - using hard decisions")
        ldpc_decoded = fec_bytes  # ← BYPASS LDPC
    
    # RS decode (this works!)
        if self.config.fec_rs_symbols > 0:
            rs_decoded, rs_errors = self.decode_reed_solomon(ldpc_decoded)
            total_errors = rs_errors
            self.metrics.rs_errors_corrected = rs_errors
            logger.debug(f"RS decoded: {len(rs_decoded)} bytes, errors={rs_errors}")
        else:
            rs_decoded = ldpc_decoded.rstrip(b'\x00')
    
        return rs_decoded, total_errors
    
    def decode_ldpc_with_llrs(self, llrs: np.ndarray) -> bytes:
        """Decode LDPC using soft LLRs (not hard bits!)."""
        n, k = self.LDPC_CODES[self.config.fec_ldpc_rate]

    # Calculate number of complete LDPC blocks
        num_blocks = len(llrs) // n

    # DEBUG: Check LLR statistics
        logger.info(f"[LDPC DEBUG] Total LLRs: {len(llrs)}, blocks: {num_blocks}, n={n}, k={k}")
        logger.info(f"[LDPC DEBUG] LLR range: [{np.min(llrs):.2f}, {np.max(llrs):.2f}]")
        logger.info(f"[LDPC DEBUG] LLR mean: {np.mean(llrs):.2f}, std: {np.std(llrs):.2f}")
        logger.info(f"[LDPC DEBUG] Positive LLRs: {np.sum(llrs > 0)}, Negative: {np.sum(llrs < 0)}, Zero: {np.sum(llrs == 0)}")

        if num_blocks == 0:
            logger.warning(f"Not enough LLRs for even one LDPC block ({len(llrs)} < {n})")
        # Fallback: pack hard decisions
            hard_bits = (llrs < 0).astype(np.uint8)
            return np.packbits(hard_bits).tobytes()

        decoded_blocks = []
        total_iterations = 0

        for block_idx in range(num_blocks):
            block_start = block_idx * n
            block_end = block_start + n
            block_llrs = llrs[block_start:block_end]
    
        # DEBUG: Check this block's LLRs
            logger.info(f"[LDPC DEBUG] Block {block_idx}: LLR mean={np.mean(block_llrs):.2f}, "
                f"std={np.std(block_llrs):.2f}, "
                f"positive={np.sum(block_llrs > 0)}, negative={np.sum(block_llrs < 0)}")
    
        # Get LDPC parameters
            ldpc_params = self._get_ldpc_params(self.config.fec_ldpc_rate)
            if block_idx == 0:
                import hashlib
                H_array = ldpc_params['vnode_adj_list']
                matrix_hash = hashlib.md5(H_array.tobytes()).hexdigest()[:8]
                logger.info(f"[RX LDPC] Matrix hash: {matrix_hash}, n_vnodes={ldpc_params['n_vnodes']}, n_cnodes={ldpc_params['n_cnodes']}")
    
        # ==================== NEW: SCALE LLRs ====================
        # Scale LLRs to reasonable range for BP stability
            max_llr = np.max(np.abs(block_llrs))
            if max_llr > 0:
                llr_scale = 20.0 / max_llr
                scaled_llrs = block_llrs * llr_scale
            else:
                scaled_llrs = block_llrs
        
            if block_idx == 0:
                logger.info(f"[LDPC] LLR scaling: factor={llr_scale:.6f}, "
                        f"new range=[{np.min(scaled_llrs):.2f}, {np.max(scaled_llrs):.2f}]")
        # ==================== END NEW ====================
    
        # Perform belief propagation decoding with REAL LLRs
            try:
                decoded_bits, num_iter = self._ldpc_bp_decode(
                    scaled_llrs,  # ← CHANGED: Use scaled_llrs instead of block_llrs
                ldpc_params,
                max_iterations=50
                )
                total_iterations += num_iter
        
            # Extract only information bits (first k bits)
                info_bits = decoded_bits[:k]
                decoded_blocks.append(info_bits)
        
                logger.info(f"[LDPC DEBUG] Block {block_idx} decoded in {num_iter} iterations")
        
            except Exception as e:
                logger.warning(f"LDPC BP decode failed for block {block_idx+1}: {e}")
            # Fallback: hard decisions
                hard_bits = (block_llrs < 0).astype(np.uint8)
                decoded_blocks.append(hard_bits[:k])

    # Concatenate all decoded blocks
        if decoded_blocks:
            all_info = np.concatenate(decoded_blocks)
            num_bytes = len(all_info) // 8
            result = np.packbits(all_info[:num_bytes * 8]).tobytes()
    
            logger.info(f"[LDPC DEBUG] Total output: {len(result)} bytes from {num_blocks} blocks")
    
            return result

    # Shouldn't reach here
        return b''

    def decode_reed_solomon(self, data: bytes) -> Tuple[bytes, int]:
        """Decode Reed-Solomon."""
        nsym = self.config.fec_rs_symbols  # 16
        
        if len(data) < nsym + 1:
            logger.warning(f"Data too short for RS decode: {len(data)} bytes (nsym={nsym})")
            return data, 0
        
        # Strip trailing zeros (LDPC padding)
        data_trimmed = data.rstrip(b'\x00')
        
        if len(data_trimmed) < nsym + 1:
            logger.warning(f"After stripping zeros, data too short: {len(data_trimmed)} bytes")
            return data_trimmed, 0
        
        logger.debug(
            f"RS decode: {len(data)} bytes -> {len(data_trimmed)} bytes "
            f"(stripped {len(data) - len(data_trimmed)} zero padding)"
        )
        
        rs = reedsolo.RSCodec(nsym)
        
        try:
            # Decode as single RS block
            decoded, _, errata = rs.decode(data_trimmed)
            errors = len(errata) if errata else 0
            result = bytes(decoded)
            logger.debug(f"RS decoded: {len(data_trimmed)} bytes -> {len(result)} bytes, corrected {errors} errors")
            return result, errors
        except reedsolo.ReedSolomonError as e:
            logger.error(f"RS decode failed: {e}")
            # Return data without parity as fallback
            if len(data_trimmed) > nsym:
                return data_trimmed[:-nsym], 1
            return data_trimmed, 1
    
    def decode_ldpc(self, data: bytes) -> bytes:
        """
    Decode LDPC using iterative belief propagation.
    
    This performs actual error correction using soft-decision decoding
    with the sum-product algorithm (belief propagation).
        """
        n, k = self.LDPC_CODES[self.config.fec_ldpc_rate]
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    
        logger.debug(f"LDPC decode: input={len(data)} bytes ({len(bits)} bits), n={n}, k={k}")
    
        if len(bits) < n:
            logger.debug(f"LDPC: insufficient bits, returning unchanged")
            return data
    
    # Calculate number of complete LDPC blocks
        num_blocks = len(bits) // n
    
        if num_blocks == 0:
            return data
    
        decoded_blocks = []
        total_iterations = 0
    
        for block_idx in range(num_blocks):
            block_start = block_idx * n
            block_end = block_start + n
            block_bits = bits[block_start:block_end]
        
        # Convert hard bits to soft LLRs (Log-Likelihood Ratios)
        # LLR > 0 means bit=0, LLR < 0 means bit=1
        # Use moderate confidence: |LLR| = 2.0
            block_llrs = np.where(block_bits == 0, 2.0, -2.0).astype(np.float64)
        
        # Get LDPC parameters
            try:
                ldpc_params = self._get_ldpc_params(self.config.fec_ldpc_rate)
            
            # Perform belief propagation decoding
                decoded_bits, num_iter = self._ldpc_bp_decode(
                block_llrs,
                ldpc_params,
                max_iterations=50
                )
            
                total_iterations += num_iter
            
            # Extract only information bits (first k bits)
                info_bits = decoded_bits[:k]
                decoded_blocks.append(info_bits)
            
                logger.debug(
                f"LDPC block {block_idx+1}/{num_blocks}: "
                f"decoded in {num_iter} iterations"
                )
            
            except Exception as e:
                logger.warning(f"LDPC BP decode failed for block {block_idx+1}: {e}")
            # Fallback: extract systematic bits without decoding
                info_bits = block_bits[:k]
                decoded_blocks.append(info_bits)
    
    # Concatenate all decoded blocks
        if decoded_blocks:
            all_info = np.concatenate(decoded_blocks)
            num_bytes = len(all_info) // 8
            result = np.packbits(all_info[:num_bytes * 8]).tobytes()
        
            avg_iterations = total_iterations / num_blocks if num_blocks > 0 else 0
            logger.debug(
            f"LDPC decoded: {len(data)} bytes -> {len(result)} bytes "
            f"({num_blocks} blocks, avg {avg_iterations:.1f} iterations)"
            )
        
            self.metrics.ldpc_iterations = int(avg_iterations)
            return result
    
        return data
    
    def _ldpc_bp_decode(
    self,
    llr: np.ndarray,
    ldpc_params: dict,
    max_iterations: int = 50
    ) -> Tuple[np.ndarray, int]:
        """
    LDPC Belief Propagation decoder using sum-product algorithm.
    
    Args:
        llr: Log-likelihood ratios (LLRs) for received codeword
        ldpc_params: LDPC code parameters dictionary
        max_iterations: Maximum number of BP iterations
        
    Returns:
        (decoded_bits, num_iterations) tuple
        """
        n_vnodes = ldpc_params['n_vnodes']
        n_cnodes = ldpc_params['n_cnodes']
        vnode_adj = ldpc_params['vnode_adj_list']
        cnode_adj = ldpc_params['cnode_adj_list']
        vnode_deg = ldpc_params['vnode_deg_list']
        cnode_deg = ldpc_params['cnode_deg_list']
        max_vnode_deg = ldpc_params['max_vnode_deg']
        max_cnode_deg = ldpc_params['max_cnode_deg']

    # Initialize messages
    # vnode_to_cnode[v, j] = message from vnode v to its j-th cnode
        vnode_to_cnode = np.zeros((n_vnodes, max_vnode_deg), dtype=np.float64)
    # cnode_to_vnode[c, j] = message from cnode c to its j-th vnode
        cnode_to_vnode = np.zeros((n_cnodes, max_cnode_deg), dtype=np.float64)

    # Initialize vnode messages with channel LLRs
        for v in range(n_vnodes):
            for j in range(vnode_deg[v]):
                vnode_to_cnode[v, j] = llr[v]

    # Iterative belief propagation
        for iteration in range(max_iterations):
        # Check node update (sum-product)
            for c in range(n_cnodes):
                for j in range(cnode_deg[c]):
                    v = cnode_adj[c * max_cnode_deg + j]
                    if v < 0:  # Padding
                        continue
            
                # Product of tanh(msg/2) from all other vnodes
                    product = 1.0
                    for jj in range(cnode_deg[c]):
                        if jj == j:
                            continue
                        vv = cnode_adj[c * max_cnode_deg + jj]
                        if vv < 0:
                            continue
                
                    # Find which edge connects vv to c
                        msg_idx = -1
                        for edge_idx in range(vnode_deg[vv]):
                            if vnode_adj[vv * max_vnode_deg + edge_idx] == c:
                                msg_idx = edge_idx
                                break
                
                        if msg_idx >= 0:
                            msg = vnode_to_cnode[vv, msg_idx]
                        # Clamp to avoid overflow
                            msg = np.clip(msg, -10, 10)
                            product *= np.tanh(msg / 2.0)
            
                # Clip product to avoid numerical issues
                    product = np.clip(product, -0.9999, 0.9999)
            
                # Inverse tanh
                    cnode_to_vnode[c, j] = 2.0 * np.arctanh(product)
    
        # Variable node update - CORRECTED VERSION
            for v in range(n_vnodes):
            # Update messages from vnode to each cnode
                for j in range(vnode_deg[v]):
                    c_target = vnode_adj[v * max_vnode_deg + j]
                    if c_target < 0:
                        continue
                
                # Compute message to c_target = channel LLR + sum of messages from OTHER cnodes
                    msg = llr[v]  # Start with channel LLR
                
                # Add messages from all OTHER cnodes (not c_target)
                    for jj in range(vnode_deg[v]):
                        if jj == j:  # Skip the target cnode
                            continue
                        c_other = vnode_adj[v * max_vnode_deg + jj]
                        if c_other < 0:
                            continue
                    
                    # Find the message from c_other to v
                        for edge_idx in range(cnode_deg[c_other]):
                            if cnode_adj[c_other * max_cnode_deg + edge_idx] == v:
                                msg += cnode_to_vnode[c_other, edge_idx]
                                break
                
                # Store the computed message
                    vnode_to_cnode[v, j] = msg
    
        # Check convergence (early stopping)
            if (iteration + 1) % 5 == 0:  # Check every 5 iterations
            # Make hard decisions
                decisions = np.zeros(n_vnodes, dtype=np.uint8)
                for v in range(n_vnodes):
                    total_llr = llr[v]
                    for j in range(vnode_deg[v]):
                        c = vnode_adj[v * max_vnode_deg + j]
                        if c < 0:
                            continue
    
                    # Find message from c to v
                        for edge_idx in range(cnode_deg[c]):
                            if cnode_adj[c * max_cnode_deg + edge_idx] == v:
                                total_llr += cnode_to_vnode[c, edge_idx]
                                break
    
                    decisions[v] = 0 if total_llr > 0 else 1

            # Check parity constraints
                syndrome = self._check_ldpc_syndrome(decisions, ldpc_params)

            # DEBUG: Always log at iteration 5
                if (iteration + 1) in [5, 10, 15, 20]:
                    logger.info(f"[BP DEBUG] Iteration {iteration+1}: syndrome={'PASS' if syndrome else 'FAIL'}")

                if syndrome:  # All parity checks satisfied
                    logger.info(f"[BP DEBUG] ✓ Converged at iteration {iteration + 1}")
                    return decisions, iteration + 1

    # Max iterations reached - return best estimate
        logger.debug(f"LDPC max iterations ({max_iterations}) reached")

    # Final hard decision
        final_decisions = np.zeros(n_vnodes, dtype=np.uint8)
        for v in range(n_vnodes):
            total_llr = llr[v]
            for j in range(vnode_deg[v]):
                c = vnode_adj[v * max_vnode_deg + j]
                if c < 0:
                    continue
        
                for edge_idx in range(cnode_deg[c]):
                    if cnode_adj[c * max_cnode_deg + edge_idx] == v:
                        total_llr += cnode_to_vnode[c, edge_idx]
                        break
    
            final_decisions[v] = 0 if total_llr > 0 else 1

        return final_decisions, max_iterations
    
    def _check_ldpc_syndrome(self, bits: np.ndarray, ldpc_params: dict) -> bool:
        """
    Check if decoded bits satisfy all parity check equations.
    
    Returns True if all syndrome bits are zero (valid codeword).
        """
        n_cnodes = ldpc_params['n_cnodes']
        cnode_adj = ldpc_params['cnode_adj_list']
        cnode_deg = ldpc_params['cnode_deg_list']
        max_cnode_deg = ldpc_params['max_cnode_deg']
    
    # Check each parity equation
        failed_checks = 0
        for c in range(n_cnodes):
            parity = 0
            for j in range(cnode_deg[c]):
                v = cnode_adj[c * max_cnode_deg + j]
                if v >= 0 and v < len(bits):
                    parity ^= int(bits[v])  # Ensure it's Python int for XOR
        
            if parity != 0:
                failed_checks += 1
    
    # DEBUG: Log how many checks failed
        if failed_checks > 0:
            logger.info(f"[SYNDROME] Failed checks: {failed_checks}/{n_cnodes} ({100*failed_checks/n_cnodes:.1f}%)")    
        return failed_checks == 0  # All checks passed

    def _get_ldpc_params(self, code_rate: FECCodeRate) -> dict:
        """Load LDPC params from fecencoder (single source of truth)."""
        from broadcast.fecencoder import FECEncoder
        import os
        from pathlib import Path
    
    # Force the LDPC path
        ldpc_path = os.getenv("LDPC_PARAMS_PATH")
        if not ldpc_path:
            ldpc_path = str(Path.cwd() / "OUTPUTs" / "ldpc_rate_8_15.npz")
            os.environ["LDPC_PARAMS_PATH"] = ldpc_path  # Set it for future calls
    
        if os.path.exists(ldpc_path):
            logger.info(f"[DECODER] Loading LDPC params from {ldpc_path}")
            return FECEncoder.load_ldpc_params(ldpc_path)
    
    # Fallback...
        logger.warning(f"[DECODER] LDPC file not found at {ldpc_path}, generating locally")
    
    # Check cache
        if code_rate in self._ldpc_matrices:
            return self._ldpc_matrices[code_rate]
    
        n, k = self.LDPC_CODES[code_rate]
        m = n - k
        ldpc_params = self._generate_ldpc_code_params(n, k, m)
        self._ldpc_matrices[code_rate] = ldpc_params
        return ldpc_params

    def _generate_ldpc_code_params(self, n: int, k: int, m: int) -> dict:
        """
    Generate complete LDPC code parameters with systematic structure.
    This creates the same parity check matrix as the encoder.
    
    Args:
        n: Codeword length (variable nodes)
        k: Information bits length
        m: Parity bits length (check nodes)
        
    Returns:
        Complete ldpc_code_params dictionary
        """
        n_vnodes = n
        n_cnodes = m
    
    # Column weight for information bits
        col_weight = 3
    
    # Build adjacency lists
        vnode_adj_list = []
        vnode_deg_list = []
        cnode_adj_list = [[] for _ in range(n_cnodes)]
        cnode_deg_list = np.zeros(n_cnodes, dtype=np.int32)
    
    # Information bit variables (first k vnodes)
        for vnode in range(k):
            connections = []
            for w in range(col_weight):
                cnode = (vnode * col_weight + w) % n_cnodes
                connections.append(cnode)
                cnode_adj_list[cnode].append(vnode)
                cnode_deg_list[cnode] += 1
            vnode_adj_list.append(connections)
            vnode_deg_list.append(col_weight)
    
    # Parity bit variables (last m vnodes) - identity structure
        for i in range(m):
            vnode = k + i
            cnode = i
            vnode_adj_list.append([cnode])
            vnode_deg_list.append(1)
            cnode_adj_list[cnode].append(vnode)
            cnode_deg_list[cnode] += 1
    
    # Convert to numpy arrays with padding
        vnode_deg_list = np.array(vnode_deg_list, dtype=np.int32)
        max_vnode_deg = int(np.max(vnode_deg_list))
        max_cnode_deg = int(np.max(cnode_deg_list))
    
    # Pad adjacency lists
        vnode_adj_padded = np.full((n_vnodes, max_vnode_deg), -1, dtype=np.int32)
        for vnode in range(n_vnodes):
            for j, cnode in enumerate(vnode_adj_list[vnode]):
                vnode_adj_padded[vnode, j] = cnode
    
        cnode_adj_padded = np.full((n_cnodes, max_cnode_deg), -1, dtype=np.int32)
        for cnode in range(n_cnodes):
            for j, vnode in enumerate(cnode_adj_list[cnode]):
                cnode_adj_padded[cnode, j] = vnode
    
    # Flatten for efficient access
        vnode_adj_flat = vnode_adj_padded.flatten()
        cnode_adj_flat = cnode_adj_padded.flatten()
    
    # Create complete parameters dictionary
        ldpc_params = {
        'n_vnodes': n_vnodes,
        'n_cnodes': n_cnodes,
        'max_vnode_deg': max_vnode_deg,
        'max_cnode_deg': max_cnode_deg,
        'vnode_adj_list': vnode_adj_flat,
        'cnode_adj_list': cnode_adj_flat,
        'vnode_deg_list': vnode_deg_list,
        'cnode_deg_list': cnode_deg_list,
        }
    
        logger.debug(
        f"Generated LDPC params: n={n_vnodes}, k={k}, m={n_cnodes}, "
        f"max_vdeg={max_vnode_deg}, max_cdeg={max_cnode_deg}"
        )
    
        return ldpc_params
    
    # ═══════════════════════════════════════════════════════════
    # PATCH 2 IMPLEMENTATION: decapsulate with length header
    # ═══════════════════════════════════════════════════════════
    def decapsulate(self, data: bytes) -> bytes:
        """Remove encapsulation with CRC verification."""
        import struct
        import zlib
    
        logger.info(f"[DECAPSULATE] Input: {len(data)} bytes")
    
        if len(data) < 8:
            logger.warning(f"[DECAPSULATE] Data too short: {len(data)} bytes")
            return data.rstrip(b'\x00')
    
    # Try length + CRC header (8 bytes total)
        try:
            length = struct.unpack('!I', data[:4])[0]
            crc_stored = struct.unpack('!I', data[4:8])[0]
        
            logger.info(f"[DECAPSULATE] Length header: {length}, CRC: 0x{crc_stored:08x}")
        
            if 1 <= length <= len(data) - 8:
                payload = data[8:8 + length]
                crc_computed = zlib.crc32(payload) & 0xFFFFFFFF
            
                if crc_computed == crc_stored:
                    logger.info(f"[DECAPSULATE] ✓ CRC match! Extracted {len(payload)} bytes")
                    return payload
                else:
                    logger.warning(
                    f"[DECAPSULATE] CRC mismatch: "
                    f"stored=0x{crc_stored:08x}, computed=0x{crc_computed:08x}"
                    )
        except Exception as e:
            logger.error(f"[DECAPSULATE] Header parse failed: {e}")
    
    # Fallback
        #logger.warning("[DECAPSULATE] Falling back to zero stripping")
        logger.info("[DECAPSULATE] Zero Stripping Method")
        return data.rstrip(b'\x00')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    config = DecoderConfig()
    decoder = BroadcastDecoder(config)
    
    # Test with random signal
    test_signal = np.random.randn(110592) + 1j * np.random.randn(110592)
    test_signal = test_signal.astype(np.complex64)
    
    result = decoder.decode(test_signal)
    print(f"Decode result: success={result.success}")
    if result.data:
        print(f"Recovered data: {len(result.data)} bytes")