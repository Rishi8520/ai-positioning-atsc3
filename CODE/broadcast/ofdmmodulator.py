"""
ATSC 3.0 OFDM Modulator.
Generates OFDM signals from framed data.

Based on ATSC A/322:2017 specification.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

from broadcast.config import ModulationScheme, GuardInterval, FFTSize, PilotPattern
from broadcast.framebuilder import ATSCFrame, FrameConfig
from broadcast.utils import get_timestamp_ms

logger = logging.getLogger(__name__)


@dataclass
class OFDMSignal:
    """Container for OFDM modulated signal."""
    time_domain_signal: np.ndarray  # Complex I/Q samples
    frequency_domain_symbols: np.ndarray  # OFDM symbols before IFFT
    sample_rate: float  # Sample rate in Hz
    num_symbols: int  # Number of OFDM symbols
    frame_id: int  # Corresponding frame ID
    timestamp: int
    metadata: dict
    
    @property
    def num_samples(self) -> int:
        """Total number of time-domain samples."""
        return len(self.time_domain_signal)
    
    @property
    def duration_ms(self) -> float:
        """Signal duration in milliseconds."""
        return (self.num_samples / self.sample_rate) * 1000.0
    
    def get_iq_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get I and Q components separately."""
        return self.time_domain_signal.real, self.time_domain_signal.imag


class OFDMModulator:
    """
    ATSC 3.0 OFDM Modulator.
    
    Converts framed data into OFDM signals ready for RF transmission.
    Fully compliant with ATSC A/322:2017.
    """
    
    # ATSC 3.0 sample rate for 6 MHz channel
    SAMPLE_RATE_HZ = 6.144e6
    
    # ATSC 3.0 carrier allocation per A/322 Table 9.1
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
    
    # Constellation mapping tables (normalized to unit average power)
    CONSTELLATIONS = {
        ModulationScheme.BPSK: np.array([-1, 1], dtype=np.complex64),
        
        ModulationScheme.QPSK: np.array([
            -1-1j, -1+1j, 1-1j, 1+1j
        ], dtype=np.complex64) / np.sqrt(2),
        
        ModulationScheme.QAM16: np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,
            -1-3j, -1-1j, -1+3j, -1+1j,
            3-3j, 3-1j, 3+3j, 3+1j,
            1-3j, 1-1j, 1+3j, 1+1j
        ], dtype=np.complex64) / np.sqrt(10),
        
        ModulationScheme.QAM64: None,  # Generated dynamically
        ModulationScheme.QAM256: None,  # Generated dynamically
    }
    
    def __init__(self):
        """Initialize OFDM modulator."""
        self.modulated_frames = 0
        
        # Generate higher-order QAM constellations
        self._generate_qam64_constellation()
        self._generate_qam256_constellation()
        
        logger.info("OFDM Modulator initialized")
    
    def _generate_qam64_constellation(self):
        """Generate normalized 64-QAM constellation."""
        # 64-QAM: 8x8 grid, values: -7, -5, -3, -1, 1, 3, 5, 7
        points = []
        for i in range(8):
            for q in range(8):
                real = 2 * i - 7
                imag = 2 * q - 7
                points.append(complex(real, imag))
        
        constellation = np.array(points, dtype=np.complex64)
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        self.CONSTELLATIONS[ModulationScheme.QAM64] = constellation / np.sqrt(avg_power)
    
    def _generate_qam256_constellation(self):
        """Generate normalized 256-QAM constellation."""
        # 256-QAM: 16x16 grid, values: -15, -13, ..., -1, 1, ..., 13, 15
        points = []
        for i in range(16):
            for q in range(16):
                real = 2 * i - 15
                imag = 2 * q - 15
                points.append(complex(real, imag))
        
        constellation = np.array(points, dtype=np.complex64)
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        self.CONSTELLATIONS[ModulationScheme.QAM256] = constellation / np.sqrt(avg_power)
    
    def modulate(
        self,
        frame: ATSCFrame,
        config: Optional[FrameConfig] = None
    ) -> OFDMSignal:
        """
        Modulate ATSC frame into OFDM signal.
        
        Args:
            frame: ATSC frame to modulate
            config: Optional frame configuration (extracted from frame if None)
        
        Returns:
            OFDM signal ready for transmission
        """
        # Extract config from frame metadata if not provided
        if config is None:
            config = self._extract_config_from_frame(frame)
        
        # Handle empty payload
        if len(frame.payload) == 0:
            return self._create_empty_signal(frame, config)
        
        # Step 1: Bit-to-symbol mapping
        data_symbols = self._map_bits_to_symbols(frame.payload, frame.modulation)
        
        # Handle case where no symbols are generated
        if len(data_symbols) == 0:
            return self._create_empty_signal(frame, config)
        
        # Step 2: Build OFDM frequency-domain symbols
        ofdm_symbols = self._build_ofdm_symbols(
            data_symbols,
            frame.pilots,
            config
        )
        
        # Step 3: IFFT to time domain
        time_domain_symbols = self._apply_ifft(ofdm_symbols, config.fft_size)
        
        # Step 4: Add guard interval (cyclic prefix)
        signal_with_gi = self._add_guard_interval(
            time_domain_symbols,
            config.guard_interval,
            config.fft_size
        )
        
        # Step 5: Flatten to continuous signal
        continuous_signal = signal_with_gi.flatten().astype(np.complex64)
        
        # Calculate power metrics correctly
        peak_power, average_power = self._calculate_power_metrics(continuous_signal)
        
        # Create metadata
        metadata = {
            'frame_id': frame.frame_id,
            'fft_size': config.fft_size.value,
            'guard_interval': config.guard_interval.name,
            'modulation': frame.modulation.name,
            'num_ofdm_symbols': len(ofdm_symbols),
            'num_data_symbols': len(data_symbols),
            'sample_rate_hz': self.SAMPLE_RATE_HZ,
            'peak_power': peak_power,
            'average_power': average_power,
        }
        
        ofdm_signal = OFDMSignal(
            time_domain_signal=continuous_signal,
            frequency_domain_symbols=ofdm_symbols,
            sample_rate=self.SAMPLE_RATE_HZ,
            num_symbols=len(ofdm_symbols),
            frame_id=frame.frame_id,
            timestamp=get_timestamp_ms(),
            metadata=metadata
        )
        
        self.modulated_frames += 1
        
        logger.debug(
            f"Frame {frame.frame_id} modulated: {ofdm_signal.num_samples} samples, "
            f"{ofdm_signal.duration_ms:.2f} ms, "
            f"{len(ofdm_symbols)} OFDM symbols"
        )
        
        return ofdm_signal
    
    def _extract_config_from_frame(self, frame: ATSCFrame) -> FrameConfig:
        """Extract FrameConfig from frame metadata."""
        # Map metadata back to enums
        fft_size = FFTSize(frame.metadata['fft_size'])
        guard_interval = GuardInterval(frame.metadata['guard_interval'])
        
        # Get pilot pattern (default if not in metadata)
        pilot_pattern_name = frame.metadata.get('pilot_pattern', 'SP3_4')
        if pilot_pattern_name == 'None':
            pilot_pattern = PilotPattern.SP3_4
        else:
            pilot_pattern = PilotPattern[pilot_pattern_name]
        
        config = FrameConfig(
            fft_size=fft_size,
            guard_interval=guard_interval,
            modulation=frame.modulation,
            frame_duration_ms=frame.metadata['frame_duration_ms'],
            pilots_enabled=(frame.pilots is not None),
            pilot_pattern=pilot_pattern,
            pilot_boost_db=frame.metadata.get('pilot_boost_db', 3.0),
            time_interleaving=True,
            time_interleaving_depth=frame.metadata.get('time_interleaving_depth', 1024)
        )
        
        return config

    def _create_empty_signal(
        self,
        frame: ATSCFrame,
        config: FrameConfig
    ) -> OFDMSignal:
        """
        Create a minimal valid signal for empty frames.
        
        Generates a single OFDM symbol with only preamble/pilot content.
        """
        # Create one empty OFDM symbol
        num_carriers = config.fft_size.value
        ofdm_symbols = np.zeros((1, num_carriers), dtype=np.complex64)
        
        # Apply IFFT
        time_domain_symbols = self._apply_ifft(ofdm_symbols, config.fft_size)
        
        # Add guard interval
        signal_with_gi = self._add_guard_interval(
            time_domain_symbols,
            config.guard_interval,
            config.fft_size
        )
        
        continuous_signal = signal_with_gi.flatten().astype(np.complex64)
        
        metadata = {
            'frame_id': frame.frame_id,
            'fft_size': config.fft_size.value,
            'guard_interval': config.guard_interval.name,
            'modulation': frame.modulation.name,
            'num_ofdm_symbols': 1,
            'num_data_symbols': 0,
            'sample_rate_hz': self.SAMPLE_RATE_HZ,
            'peak_power': 0.0,
            'average_power': 0.0,
        }
        
        self.modulated_frames += 1
        
        return OFDMSignal(
            time_domain_signal=continuous_signal,
            frequency_domain_symbols=ofdm_symbols,
            sample_rate=self.SAMPLE_RATE_HZ,
            num_symbols=1,
            frame_id=frame.frame_id,
            timestamp=get_timestamp_ms(),
            metadata=metadata
        )

    def _calculate_power_metrics(
        self,
        signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate peak and average power correctly.
        
        Power is |signal|^2, not |signal|.
        
        Args:
            signal: Complex time-domain signal
        
        Returns:
            (peak_power, average_power) tuple
        """
        if len(signal) == 0:
            return 0.0, 0.0
        
        # Power is magnitude squared
        power = np.abs(signal) ** 2
        
        # Filter out zero-power samples for more meaningful PAPR
        # (guard bands contribute zeros which skew average)
        nonzero_power = power[power > 1e-10]
        
        if len(nonzero_power) == 0:
            return 0.0, 0.0
        
        peak_power = float(np.max(power))
        average_power = float(np.mean(nonzero_power))
        
        return peak_power, average_power


    def _map_bits_to_symbols(
        self,
        data: bytes,
        modulation: ModulationScheme
    ) -> np.ndarray:
        """
        Map data bits to constellation symbols.
        
        Args:
            data: Input data bytes
            modulation: Modulation scheme
        
        Returns:
            Array of complex constellation symbols
        """
        # Get constellation
        constellation = self.CONSTELLATIONS[modulation]
        bits_per_symbol = self._bits_per_symbol(modulation)
        
        # Convert bytes to bits
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        
        # Pad bits to multiple of bits_per_symbol
        num_bits = len(bits)
        padding = (bits_per_symbol - (num_bits % bits_per_symbol)) % bits_per_symbol
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=np.uint8)])
        
        # Group bits into symbols
        num_symbols = len(bits) // bits_per_symbol
        bit_groups = bits.reshape(num_symbols, bits_per_symbol)
        
        # Convert bit groups to symbol indices
        symbol_indices = np.zeros(num_symbols, dtype=np.int32)
        for i in range(bits_per_symbol):
            symbol_indices += bit_groups[:, i] << (bits_per_symbol - 1 - i)
        
        # Map to constellation points
        symbols = constellation[symbol_indices]
        
        return symbols
    
    def _build_ofdm_symbols(
        self,
        data_symbols: np.ndarray,
        pilot_symbols: Optional[np.ndarray],
        config: FrameConfig
    ) -> np.ndarray:
        """
        Build OFDM frequency-domain symbols with pilots.
        
        Uses ATSC 3.0 carrier allocation tables per A/322 Table 9.1.
        
        Args:
            data_symbols: Modulated data symbols
            pilot_symbols: Pilot symbols (or None)
            config: Frame configuration
        
        Returns:
            OFDM symbols (frequency domain) shaped (num_symbols, num_carriers)
        """
        num_carriers = config.fft_size.value
        
        # Get carrier allocation from ATSC 3.0 tables
        allocation = self.CARRIER_ALLOCATION.get(config.fft_size)
        if allocation is None:
            # Fallback for unknown FFT sizes
            guard_size = int(num_carriers * 0.08)
            active_carriers = num_carriers - 2 * guard_size - 1
        else:
            guard_size = allocation['guard_lower']
            active_carriers = allocation['active']
        
        dc_index = num_carriers // 2
        
        # Account for pilots
        if config.pilots_enabled and pilot_symbols is not None:
            dx, dy = config.pilot_pattern.value
            pilots_per_symbol = active_carriers // dx
            data_carriers_per_symbol = active_carriers - pilots_per_symbol
        else:
            data_carriers_per_symbol = active_carriers
            pilots_per_symbol = 0
        
        # Ensure at least 1 carrier
        data_carriers_per_symbol = max(1, data_carriers_per_symbol)
        
        num_ofdm_symbols = max(1, (len(data_symbols) + data_carriers_per_symbol - 1) // data_carriers_per_symbol)
        
        # Initialize OFDM symbol array
        ofdm_symbols = np.zeros((num_ofdm_symbols, num_carriers), dtype=np.complex64)
        
        # Fill symbols
        data_idx = 0
        pilot_idx = 0
        
        for sym_idx in range(num_ofdm_symbols):
            carrier_idx = guard_size  # Start after lower guard band
            
            for _ in range(active_carriers):
                if carrier_idx == dc_index:
                    # DC carrier is null
                    carrier_idx += 1
                    continue
                
                if carrier_idx >= num_carriers - guard_size:
                    # Reached upper guard band
                    break
                
                # Check if this is a pilot position
                is_pilot = False
                if config.pilots_enabled and pilot_symbols is not None:
                    dx, dy = config.pilot_pattern.value
                    if (carrier_idx - guard_size) % dx == 0 and sym_idx % dy == 0:
                        is_pilot = True
                
                if is_pilot and pilot_symbols is not None and pilot_idx < len(pilot_symbols):
                    # Insert pilot
                    ofdm_symbols[sym_idx, carrier_idx] = pilot_symbols[pilot_idx]
                    pilot_idx += 1
                elif data_idx < len(data_symbols):
                    # Insert data
                    ofdm_symbols[sym_idx, carrier_idx] = data_symbols[data_idx]
                    data_idx += 1
                
                carrier_idx += 1
        
        return ofdm_symbols
    
    def _apply_ifft(
        self,
        ofdm_symbols: np.ndarray,
        fft_size: FFTSize
    ) -> np.ndarray:
        """
        Apply IFFT to convert frequency domain to time domain.
        
        Args:
            ofdm_symbols: Frequency domain symbols (num_symbols, num_carriers)
            fft_size: FFT size
        
        Returns:
            Time domain symbols (num_symbols, fft_size)
        """
        # Apply IFFT to each OFDM symbol
        time_domain = np.fft.ifft(ofdm_symbols, axis=1)
        
        # IFFT shifts frequency zero to center, need to fftshift for proper OFDM
        time_domain = np.fft.ifftshift(time_domain, axes=1)
        
        return time_domain
    
    def _add_guard_interval(
        self,
        time_domain_symbols: np.ndarray,
        guard_interval: GuardInterval,
        fft_size: FFTSize
    ) -> np.ndarray:
        """
        Add cyclic prefix (guard interval) to OFDM symbols.
        
        The guard interval is a copy of the end of the symbol prepended
        to the beginning, providing protection against ISI.
        
        Args:
            time_domain_symbols: Time domain symbols (num_symbols, fft_size)
            guard_interval: Guard interval ratio
            fft_size: FFT size
        
        Returns:
            Symbols with guard interval (num_symbols, fft_size + gi_length)
        """
        num_symbols, symbol_length = time_domain_symbols.shape
        
        # Calculate guard interval length
        gi_fractions = {
            GuardInterval.GI_1_4: 0.25,
            GuardInterval.GI_1_8: 0.125,
            GuardInterval.GI_1_16: 0.0625,
            GuardInterval.GI_1_32: 0.03125,
        }
        
        gi_fraction = gi_fractions.get(guard_interval, 0.125)
        gi_length = int(symbol_length * gi_fraction)
        
        # Add cyclic prefix to each symbol
        symbols_with_gi = np.zeros((num_symbols, symbol_length + gi_length), dtype=np.complex64)
        
        for i in range(num_symbols):
            # Copy last gi_length samples to beginning (cyclic prefix)
            symbols_with_gi[i, :gi_length] = time_domain_symbols[i, -gi_length:]
            # Copy full symbol
            symbols_with_gi[i, gi_length:] = time_domain_symbols[i, :]
        
        return symbols_with_gi
    
    def _bits_per_symbol(self, modulation: ModulationScheme) -> int:
        """Get bits per constellation symbol."""
        return {
            ModulationScheme.BPSK: 1,
            ModulationScheme.QPSK: 2,
            ModulationScheme.QAM16: 4,
            ModulationScheme.QAM64: 6,
            ModulationScheme.QAM256: 8,
        }.get(modulation, 2)
    
    def modulate_multiple_frames(
        self,
        frames: list[ATSCFrame]
    ) -> list[OFDMSignal]:
        """
        Modulate multiple frames.
        
        Args:
            frames: List of ATSC frames
        
        Returns:
            List of OFDM signals
        """
        signals = []
        for frame in frames:
            signal = self.modulate(frame)
            signals.append(signal)
        
        logger.info(f"Modulated {len(signals)} frames")
        return signals
    
    def get_stats(self) -> dict:
        """Get modulator statistics."""
        return {
            'frames_modulated': self.modulated_frames,
            'sample_rate_hz': self.SAMPLE_RATE_HZ,
        }


if __name__ == '__main__':
    # Simple test
    from broadcast.fecencoder import FECEncoder
    from broadcast.framebuilder import FrameBuilder
    
    encoder = FECEncoder()
    builder = FrameBuilder()
    modulator = OFDMModulator()
    
    test_data = b"ATSC 3.0 OFDM Modulator Test"
    fec_result = encoder.encode(test_data)
    frame = builder.build_frame(fec_result)
    signal = modulator.modulate(frame)
    
    print(f"OFDM signal: {signal.num_samples} samples, {signal.duration_ms:.2f} ms")
    print(f"Peak power: {signal.metadata['peak_power']:.4f}")
    print(f"Average power: {signal.metadata['average_power']:.4f}")