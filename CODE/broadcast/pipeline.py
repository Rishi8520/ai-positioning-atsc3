"""
ATSC 3.0 Broadcast Pipeline.
End-to-end processing from raw data to OFDM signals.

Orchestrates: FEC Encoder → Frame Builder → OFDM Modulator
"""

import logging
import numpy as np 
from typing import List, Optional
from dataclasses import dataclass

from broadcast.fecencoder import FECEncoder, FECConfig, FECEncodedData
from broadcast.framebuilder import FrameBuilder, FrameConfig, ATSCFrame
from broadcast.ofdmmodulator import OFDMModulator, OFDMSignal
from broadcast.alpencoder import ALPEncoder, ALPPacket
from broadcast.config import FECCodeRate, ModulationScheme, GuardInterval, FFTSize, PilotPattern
from broadcast.utils import get_timestamp_ms

logger = logging.getLogger(__name__)


@dataclass
class BroadcastConfig:
    """Complete broadcast pipeline configuration."""
    # ALP parameters
    use_alp: bool = True  # Enable/disable ALP layer

    # FEC parameters
    fec_ldpc_rate: FECCodeRate = FECCodeRate.RATE_8_15
    fec_rs_symbols: int = 16
    fec_overhead_pct: float = 15.0
    
    # Frame parameters
    fft_size: FFTSize = FFTSize.FFT_8K
    guard_interval: GuardInterval = GuardInterval.GI_1_8
    modulation: ModulationScheme = ModulationScheme.QPSK
    frame_duration_ms: float = 50.0
    pilots_enabled: bool = True
    pilot_pattern: PilotPattern = PilotPattern.SP3_4
    pilot_boost_db: float = 3.0
    time_interleaving: bool = True
    time_interleaving_depth: int = 1024
    
    def get_fec_config(self) -> FECConfig:
        """Extract FEC configuration."""
        return FECConfig(
            ldpc_rate=self.fec_ldpc_rate,
            reed_solomon_nsym=self.fec_rs_symbols,
            overhead_pct=self.fec_overhead_pct
        )
    
    def get_frame_config(self) -> FrameConfig:
        """Extract frame configuration."""
        return FrameConfig(
            fft_size=self.fft_size,
            guard_interval=self.guard_interval,
            modulation=self.modulation,
            frame_duration_ms=self.frame_duration_ms,
            pilots_enabled=self.pilots_enabled,
            pilot_pattern=self.pilot_pattern,
            pilot_boost_db=self.pilot_boost_db,
            time_interleaving=self.time_interleaving,
            time_interleaving_depth=self.time_interleaving_depth
        )


@dataclass
class BroadcastResult:
    """Result of complete broadcast pipeline."""
    original_data: bytes
    alp_packet: Optional[ALPPacket]  
    fec_result: FECEncodedData
    frame: ATSCFrame
    signal: OFDMSignal
    timestamp: int
    processing_time_ms: float
    
    @property
    def total_overhead_bytes(self) -> int:
        """Total overhead added (FEC + framing + OFDM)."""
        return len(self.signal.time_domain_signal.tobytes()) - len(self.original_data)
    
    @property
    def compression_ratio(self) -> float:
        """
        Expansion ratio of output to input size.
        
        Note: This is actually an EXPANSION ratio (output > input)
        because broadcast adds FEC redundancy and OFDM overhead.
        
        Returns:
            Ratio of output bytes to input bytes (typically > 1.0)
        """
        if len(self.original_data) == 0:
            return 0.0
        return len(self.signal.time_domain_signal.tobytes()) / len(self.original_data)
    
    @property
    def expansion_ratio(self) -> float:
        """
        Expansion ratio of output to input size.
        
        Alias for compression_ratio with clearer naming.
        """
        return self.compression_ratio
    
    @property
    def spectral_efficiency(self) -> float:
        """
        Bits of original data per OFDM sample.
        
        Higher is better - indicates how efficiently the 
        broadcast bandwidth is used for actual data.
        """
        if self.signal.num_samples == 0:
            return 0.0
        return (len(self.original_data) * 8) / self.signal.num_samples

class BroadcastPipeline:
    """
    Complete ATSC 3.0 broadcast pipeline.
    
    Processes raw data through FEC encoding, frame building,
    and OFDM modulation in a single streamlined workflow.
    """
    
    def __init__(self, config: Optional[BroadcastConfig] = None):
        """
        Initialize broadcast pipeline.
        
        Args:
            config: Broadcast configuration (uses defaults if None)
        """
        self.config = config or BroadcastConfig()
        
        # Initialize components
        self.alp_encoder = ALPEncoder() if self.config.use_alp else None

        self.fec_encoder = FECEncoder(
            default_ldpc_rate=self.config.fec_ldpc_rate,
            default_rs_symbols=self.config.fec_rs_symbols
        )
        
        self.frame_builder = FrameBuilder(
            default_config=self.config.get_frame_config()
        )
        
        self.ofdm_modulator = OFDMModulator()
        
        # Statistics
        self.packets_processed = 0
        self.total_input_bytes = 0
        self.total_output_bytes = 0
        
        logger.info(
            f"Broadcast Pipeline initialized: "
            f"FEC={self.config.fec_ldpc_rate.name}, "
            f"Modulation={self.config.modulation.name}, "
            f"FFT={self.config.fft_size.name}"
        )
    
    def process(
        self,
        data: bytes,
        config: Optional[BroadcastConfig] = None
    ) -> BroadcastResult:
        """
        Process data through complete broadcast pipeline.
        
        Args:
            data: Raw input data (e.g., RTCM corrections)
            config: Optional custom configuration (uses pipeline default if None)
        
        Returns:
            BroadcastResult with all intermediate and final outputs
        """
        start_time = get_timestamp_ms()
        
        # Use custom config or default
        if config is None:
            config = self.config

        # Step 0: ALP Encapsulation (if enabled)
        alp_packet = None
        if config.use_alp and self.alp_encoder:
            logger.debug(f"Step 0: ALP encapsulation of {len(data)} bytes")
            alp_packet = self.alp_encoder.encode_rtcm(data)
            data_to_encode = alp_packet.to_bytes()
        else:
            data_to_encode = data
        
        # Step 1: FEC Encoding
        logger.debug(f"Step 1: FEC encoding {len(data)} bytes")
        fec_result = self.fec_encoder.encode(data, config.get_fec_config())
        
        # Step 2: Frame Building
        logger.debug(f"Step 2: Building ATSC 3.0 frame")
        frame = self.frame_builder.build_frame(fec_result, config.get_frame_config())
        
        # Step 3: OFDM Modulation
        logger.debug(f"Step 3: OFDM modulation")
        signal = self.ofdm_modulator.modulate(frame, config.get_frame_config())
        
        # Calculate processing time
        end_time = get_timestamp_ms()
        processing_time = end_time - start_time
        
        # Update statistics
        self.packets_processed += 1
        self.total_input_bytes += len(data)
        self.total_output_bytes += signal.num_samples * 8  # Complex64 = 8 bytes
        
        # Create result
        result = BroadcastResult(
            original_data=data,
            alp_packet=alp_packet,
            fec_result=fec_result,
            frame=frame,
            signal=signal,
            timestamp=start_time,
            processing_time_ms=processing_time
        )
        
        logger.info(
            f"Pipeline processed packet: "
            f"{len(data)} bytes → {signal.num_samples} samples "
            f"in {processing_time:.2f} ms"
            f"{' (with ALP)' if alp_packet else ''}"
        )
        
        return result
    
    def process_batch(
        self,
        data_list: List[bytes],
        config: Optional[BroadcastConfig] = None
    ) -> List[BroadcastResult]:
        """
        Process multiple data packets through pipeline.
        
        Args:
            data_list: List of raw data packets
            config: Optional custom configuration
        
        Returns:
            List of BroadcastResult objects
        """
        results = []
        
        logger.info(f"Processing batch of {len(data_list)} packets")
        
        for i, data in enumerate(data_list):
            try:
                result = self.process(data, config)
                results.append(result)
                logger.debug(f"Processed packet {i+1}/{len(data_list)}")
            except Exception as e:
                logger.error(f"Error processing packet {i+1}: {e}")
                # Continue with next packet
                continue
        
        logger.info(f"Batch processing complete: {len(results)}/{len(data_list)} succeeded")
        
        return results
    
    def process_with_adaptive_fec(
        self,
        data: bytes,
        overhead_pct: float
    ) -> BroadcastResult:
        """
        Process data with AI-controlled adaptive FEC.
        
        This method allows dynamic FEC adjustment based on
        channel conditions or AI recommendations.
        
        Args:
            data: Raw input data
            overhead_pct: Desired FEC overhead percentage (0-100)
        
        Returns:
            BroadcastResult
        """
        # Create custom config with adaptive FEC
        config = BroadcastConfig(
            fec_ldpc_rate=self.config.fec_ldpc_rate,
            fec_rs_symbols=self.config.fec_rs_symbols,
            fec_overhead_pct=overhead_pct,
            fft_size=self.config.fft_size,
            guard_interval=self.config.guard_interval,
            modulation=self.config.modulation,
            frame_duration_ms=self.config.frame_duration_ms,
            pilots_enabled=self.config.pilots_enabled,
            pilot_pattern=self.config.pilot_pattern,
            pilot_boost_db=self.config.pilot_boost_db,
            time_interleaving=self.config.time_interleaving,
            time_interleaving_depth=self.config.time_interleaving_depth
        )
        
        logger.info(f"Adaptive FEC: {overhead_pct}% overhead requested")
        
        return self.process(data, config)
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        avg_compression = (
            self.total_output_bytes / self.total_input_bytes 
            if self.total_input_bytes > 0 
            else 0.0
        )
        
        stats = {
            'packets_processed': self.packets_processed,
            'total_input_bytes': self.total_input_bytes,
            'total_output_bytes': self.total_output_bytes,
            'average_compression_ratio': avg_compression,
            'fec_encoder_stats': self.fec_encoder.get_stats(),
            'frame_builder_stats': self.frame_builder.get_stats(),
            'modulator_stats': self.ofdm_modulator.get_stats(),
        }
        
        # Add ALP stats if enabled
        if self.alp_encoder:
            stats['alp_encoder_stats'] = self.alp_encoder.get_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics."""
        self.packets_processed = 0
        self.total_input_bytes = 0
        self.total_output_bytes = 0
        logger.info("Pipeline statistics reset")


# Convenience function for quick processing
def broadcast_data(
    data: bytes,
    fec_overhead_pct: float = 15.0,
    modulation: ModulationScheme = ModulationScheme.QPSK,
    fft_size: FFTSize = FFTSize.FFT_8K
) -> BroadcastResult:
    """
    Quick one-shot broadcast processing.
    
    Convenience function for simple use cases.
    
    Args:
        data: Raw input data
        fec_overhead_pct: FEC overhead percentage
        modulation: Modulation scheme
        fft_size: FFT size
    
    Returns:
        BroadcastResult
    """
    config = BroadcastConfig(
        fec_overhead_pct=fec_overhead_pct,
        modulation=modulation,
        fft_size=fft_size
    )
    
    pipeline = BroadcastPipeline(config)
    return pipeline.process(data)


if __name__ == '__main__':
    
    # Simple test
    pipeline = BroadcastPipeline()
    
    test_data = b"ATSC 3.0 Broadcast Pipeline Test - RTK Corrections"
    result = pipeline.process(test_data)
    
    print(f"\n=== Broadcast Pipeline Test ===")
    print(f"Input: {len(result.original_data)} bytes")
    print(f"FEC encoded: {len(result.fec_result.encoded_data)} bytes")
    print(f"Frame: {result.frame.total_bytes} bytes")
    print(f"OFDM signal: {result.signal.num_samples} samples ({result.signal.duration_ms:.2f} ms)")
    print(f"Processing time: {result.processing_time_ms:.2f} ms")
    print(f"Expansion ratio: {result.expansion_ratio:.2f}x")
    print(f"Spectral efficiency: {result.spectral_efficiency:.4f} bits/sample")
    print(f"FEC code rate: {result.fec_result.code_rate:.3f}")
    
    # Safe PAPR calculation
    if result.signal.metadata['average_power'] > 0:
        papr_db = 10 * np.log10(
            result.signal.metadata['peak_power'] / result.signal.metadata['average_power']
        )
        print(f"PAPR: {papr_db:.1f} dB")