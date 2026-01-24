"""
Unit tests for ATSC 3.0 Broadcast Pipeline.
"""

import pytest
import numpy as np
from broadcast.pipeline import (
    BroadcastPipeline,
    BroadcastConfig,
    BroadcastResult,
    broadcast_data
)
from broadcast.config import (
    FECCodeRate,
    ModulationScheme,
    GuardInterval,
    FFTSize,
    PilotPattern
)
from broadcast.alpencoder import ALPPacket


class TestBroadcastConfig:
    """Test suite for BroadcastConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BroadcastConfig()
        
        assert config.fec_ldpc_rate == FECCodeRate.RATE_8_15
        assert config.fec_rs_symbols == 16
        assert config.fec_overhead_pct == 15.0
        assert config.fft_size == FFTSize.FFT_8K
        assert config.guard_interval == GuardInterval.GI_1_8
        assert config.modulation == ModulationScheme.QPSK
        assert config.frame_duration_ms == 50.0
        assert config.pilots_enabled is True
        assert config.pilot_pattern == PilotPattern.SP3_4
        assert config.pilot_boost_db == 3.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BroadcastConfig(
            fec_ldpc_rate=FECCodeRate.RATE_10_15,
            fec_rs_symbols=20,
            fec_overhead_pct=25.0,
            fft_size=FFTSize.FFT_16K,
            modulation=ModulationScheme.QAM16,
            pilots_enabled=False
        )
        
        assert config.fec_ldpc_rate == FECCodeRate.RATE_10_15
        assert config.fec_rs_symbols == 20
        assert config.fft_size == FFTSize.FFT_16K
        assert config.modulation == ModulationScheme.QAM16
        assert config.pilots_enabled is False
    
    def test_get_fec_config(self):
        """Test FEC config extraction."""
        config = BroadcastConfig(
            fec_ldpc_rate=FECCodeRate.RATE_6_15,
            fec_rs_symbols=24,
            fec_overhead_pct=30.0
        )
        
        fec_config = config.get_fec_config()
        
        assert fec_config.ldpc_rate == FECCodeRate.RATE_6_15
        assert fec_config.reed_solomon_nsym == 24
        assert fec_config.overhead_pct == 30.0
    
    def test_get_frame_config(self):
        """Test frame config extraction."""
        config = BroadcastConfig(
            fft_size=FFTSize.FFT_32K,
            guard_interval=GuardInterval.GI_1_4,
            modulation=ModulationScheme.QAM64
        )
        
        frame_config = config.get_frame_config()
        
        assert frame_config.fft_size == FFTSize.FFT_32K
        assert frame_config.guard_interval == GuardInterval.GI_1_4
        assert frame_config.modulation == ModulationScheme.QAM64


class TestBroadcastResult:
    """Test suite for BroadcastResult."""
    
    def test_result_properties(self):
        """Test result property calculations."""
        pipeline = BroadcastPipeline()
        test_data = b"Test data for result properties"
        
        result = pipeline.process(test_data)
        
        # Verify result structure
        assert isinstance(result, BroadcastResult)
        assert result.original_data == test_data
        assert result.fec_result is not None
        assert result.frame is not None
        assert result.signal is not None
        assert result.timestamp > 0
        assert result.processing_time_ms >= 0
        
        # Verify properties
        assert result.total_overhead_bytes > 0
        assert result.compression_ratio > 1.0


class TestBroadcastPipeline:
    """Test suite for Broadcast Pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = BroadcastPipeline()
        
        assert pipeline.config is not None
        assert pipeline.fec_encoder is not None
        assert pipeline.frame_builder is not None
        assert pipeline.ofdm_modulator is not None
        assert pipeline.packets_processed == 0
        assert pipeline.total_input_bytes == 0
        assert pipeline.total_output_bytes == 0
    
    def test_pipeline_custom_initialization(self):
        """Test pipeline with custom config."""
        config = BroadcastConfig(
            fec_ldpc_rate=FECCodeRate.RATE_12_15,
            modulation=ModulationScheme.QAM256,
            fft_size=FFTSize.FFT_16K
        )
        
        pipeline = BroadcastPipeline(config)
        
        assert pipeline.config.fec_ldpc_rate == FECCodeRate.RATE_12_15
        assert pipeline.config.modulation == ModulationScheme.QAM256
        assert pipeline.config.fft_size == FFTSize.FFT_16K
    
    def test_process_basic(self):
        """Test basic data processing."""
        pipeline = BroadcastPipeline()
        test_data = b"Basic pipeline test data"
        
        result = pipeline.process(test_data)
        
        # Verify result
        assert isinstance(result, BroadcastResult)
        assert result.original_data == test_data
        assert len(result.fec_result.encoded_data) > len(test_data)
        assert result.frame.total_bytes > 0
        assert result.signal.num_samples > 0
        
        # Verify statistics updated
        assert pipeline.packets_processed == 1
        assert pipeline.total_input_bytes == len(test_data)
        assert pipeline.total_output_bytes > 0
    
    def test_process_rtcm_data(self):
        """Test processing realistic RTCM correction data."""
        pipeline = BroadcastPipeline()
        
        # Simulate RTCM frame (typical size 50-200 bytes)
        rtcm_data = b"\xd3\x00\x13\x3e\xd0\x00\x03" + b"\x00" * 100
        
        result = pipeline.process(rtcm_data)
        
        assert isinstance(result, BroadcastResult)
        assert result.signal.num_samples > 0
        assert result.processing_time_ms > 0
    
    def test_process_empty_data(self):
        """Test processing empty data."""
        pipeline = BroadcastPipeline()
        
        result = pipeline.process(b"")
        
        # Should still produce valid output
        assert isinstance(result, BroadcastResult)
        assert result.signal is not None
    
    def test_process_large_data(self):
        """Test processing large data packet."""
        pipeline = BroadcastPipeline()
        
        # Large RTCM frame (1KB)
        large_data = b"\x00" * 1024
        
        result = pipeline.process(large_data)
        
        assert isinstance(result, BroadcastResult)
        assert result.signal.num_samples > 0
        assert result.signal.num_symbols > 1
    
    def test_process_batch(self):
        """Test batch processing."""
        pipeline = BroadcastPipeline()
        
        data_list = [
            b"Packet 1 data",
            b"Packet 2 data",
            b"Packet 3 data",
            b"Packet 4 data"
        ]
        
        results = pipeline.process_batch(data_list)
        
        assert len(results) == 4
        assert all(isinstance(r, BroadcastResult) for r in results)
        assert pipeline.packets_processed == 4
        
        # Verify each packet processed correctly
        for i, result in enumerate(results):
            assert result.original_data == data_list[i]
    
    def test_process_with_custom_config(self):
        """Test processing with custom configuration."""
        pipeline = BroadcastPipeline()
        
        custom_config = BroadcastConfig(
            fec_ldpc_rate=FECCodeRate.RATE_6_15,
            modulation=ModulationScheme.QAM16,
            fft_size=FFTSize.FFT_16K
        )
        
        test_data = b"Custom config test"
        result = pipeline.process(test_data, custom_config)
        
        assert result.signal.metadata['modulation'] == 'QAM16'
        assert result.signal.metadata['fft_size'] == 16384
    
    def test_process_with_adaptive_fec(self):
        """Test adaptive FEC processing."""
        pipeline = BroadcastPipeline()
        test_data = b"Adaptive FEC test data"
        
        # Test different overhead levels
        overhead_levels = [10.0, 20.0, 30.0]
        results = []
        
        for overhead_pct in overhead_levels:
            result = pipeline.process_with_adaptive_fec(test_data, overhead_pct)
            results.append(result)
            assert isinstance(result, BroadcastResult)
        
        # Higher overhead should produce more redundancy
        # (though exact relationship depends on FEC mapping)
        assert all(r.fec_result.overhead_bytes > 0 for r in results)
    
    def test_process_different_modulations(self):
        """Test processing with different modulation schemes."""
        test_data = b"Modulation test data"
        
        modulations = [
            ModulationScheme.BPSK,
            ModulationScheme.QPSK,
            ModulationScheme.QAM16,
            ModulationScheme.QAM64
        ]
        
        for modulation in modulations:
            config = BroadcastConfig(modulation=modulation)
            pipeline = BroadcastPipeline(config)
            result = pipeline.process(test_data)
            
            assert result.signal.metadata['modulation'] == modulation.name
    
    def test_process_different_fft_sizes(self):
        """Test processing with different FFT sizes."""
        test_data = b"FFT size test data"
        
        fft_sizes = [FFTSize.FFT_8K, FFTSize.FFT_16K, FFTSize.FFT_32K]
        
        for fft_size in fft_sizes:
            config = BroadcastConfig(fft_size=fft_size)
            pipeline = BroadcastPipeline(config)
            result = pipeline.process(test_data)
            
            assert result.signal.metadata['fft_size'] == fft_size.value
    
    def test_get_stats(self):
        """Test statistics reporting."""
        pipeline = BroadcastPipeline()
        
        # Process some data
        for i in range(5):
            test_data = f"Packet {i}".encode()
            pipeline.process(test_data)
        
        stats = pipeline.get_stats()
        
        # Verify stats structure
        assert 'packets_processed' in stats
        assert 'total_input_bytes' in stats
        assert 'total_output_bytes' in stats
        assert 'average_compression_ratio' in stats
        assert 'fec_encoder_stats' in stats
        assert 'frame_builder_stats' in stats
        assert 'modulator_stats' in stats
        
        # Verify values
        assert stats['packets_processed'] == 5
        assert stats['total_input_bytes'] > 0
        assert stats['total_output_bytes'] > 0
        assert stats['average_compression_ratio'] > 0
    
    def test_reset_stats(self):
        """Test statistics reset."""
        pipeline = BroadcastPipeline()
        
        # Process some data
        pipeline.process(b"Test data")
        assert pipeline.packets_processed == 1
        
        # Reset stats
        pipeline.reset_stats()
        
        assert pipeline.packets_processed == 0
        assert pipeline.total_input_bytes == 0
        assert pipeline.total_output_bytes == 0
    
    def test_processing_time(self):
        """Test processing time measurement."""
        pipeline = BroadcastPipeline()
        test_data = b"Processing time test"
        
        result = pipeline.process(test_data)
        
        # Processing time should be positive and reasonable
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 10000  # Less than 10 seconds
    
    def test_compression_ratio(self):
        """Test compression/expansion ratio calculation."""
        pipeline = BroadcastPipeline()
        
        # Use larger test data to reduce minimum frame overhead impact
        test_data = b"Compression ratio test data" * 20  # 540 bytes
        
        result = pipeline.process(test_data)
        
        # Output should be larger than input (adding redundancy + OFDM overhead)
        # This is actually an "expansion ratio" not compression
        assert result.compression_ratio > 1.0
        
        # With larger input, ratio should be more reasonable
        # OFDM adds significant overhead due to:
        # - FEC redundancy (~1.5-3x)
        # - Guard intervals
        # - Pilot symbols
        # - Complex samples (8 bytes each)
        # Typical expansion is 100-500x for small packets
        assert result.compression_ratio < 2000.0, \
            f"Expansion ratio {result.compression_ratio:.1f}x too high"
        
        # Test new properties
        assert result.expansion_ratio == result.compression_ratio
        assert result.spectral_efficiency > 0
    
    def test_end_to_end_signal_quality(self):
        """Test end-to-end signal quality metrics."""
        pipeline = BroadcastPipeline()
        test_data = b"Signal quality test"
        
        result = pipeline.process(test_data)
        
        # Verify signal properties
        signal = result.signal
        assert signal.metadata['peak_power'] >= 0
        assert signal.metadata['average_power'] >= 0
        
        # Check I/Q data
        i_data, q_data = signal.get_iq_data()
        assert len(i_data) == signal.num_samples
        assert len(q_data) == signal.num_samples
    
    def test_multiple_pipelines(self):
        """Test multiple independent pipelines."""
        config1 = BroadcastConfig(modulation=ModulationScheme.QPSK)
        config2 = BroadcastConfig(modulation=ModulationScheme.QAM16)
        
        pipeline1 = BroadcastPipeline(config1)
        pipeline2 = BroadcastPipeline(config2)
        
        test_data = b"Multi-pipeline test"
        
        result1 = pipeline1.process(test_data)
        result2 = pipeline2.process(test_data)
        
        # Both should succeed independently
        assert result1.signal.metadata['modulation'] == 'QPSK'
        assert result2.signal.metadata['modulation'] == 'QAM16'
        
        # Statistics should be independent
        assert pipeline1.packets_processed == 1
        assert pipeline2.packets_processed == 1


class TestConvenienceFunction:
    """Test suite for convenience broadcast_data function."""
    
    def test_broadcast_data_basic(self):
        """Test basic broadcast_data function."""
        test_data = b"Convenience function test"
        
        result = broadcast_data(test_data)
        
        assert isinstance(result, BroadcastResult)
        assert result.original_data == test_data
        assert result.signal.num_samples > 0
    
    def test_broadcast_data_with_params(self):
        """Test broadcast_data with custom parameters."""
        test_data = b"Custom params test"
        
        result = broadcast_data(
            test_data,
            fec_overhead_pct=25.0,
            modulation=ModulationScheme.QAM64,
            fft_size=FFTSize.FFT_16K
        )
        
        assert result.signal.metadata['modulation'] == 'QAM64'
        assert result.signal.metadata['fft_size'] == 16384
    
    def test_broadcast_data_different_overhead(self):
        """Test broadcast_data with different FEC overhead."""
        test_data = b"Overhead test"
        
        overhead_levels = [10.0, 20.0, 30.0]
        
        for overhead_pct in overhead_levels:
            result = broadcast_data(test_data, fec_overhead_pct=overhead_pct)
            assert isinstance(result, BroadcastResult)
            assert result.fec_result.overhead_bytes > 0

    def test_pipeline_with_alp(self):
        """Test pipeline with ALP layer enabled."""
        config = BroadcastConfig(use_alp=True)
        pipeline = BroadcastPipeline(config)
        
        test_data = b"ALP integration test"
        result = pipeline.process(test_data)
        
        # Verify ALP packet was created
        assert result.alp_packet is not None
        assert isinstance(result.alp_packet, ALPPacket)
        assert result.alp_packet.payload == test_data
        
    def test_pipeline_without_alp(self):
        """Test pipeline with ALP layer disabled."""
        config = BroadcastConfig(use_alp=False)
        pipeline = BroadcastPipeline(config)
        
        test_data = b"No ALP test"
        result = pipeline.process(test_data)
        
        # Verify no ALP packet
        assert result.alp_packet is None

class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_rtcm_to_ofdm_complete(self):
        """Test complete RTCM to OFDM conversion."""
        pipeline = BroadcastPipeline()
        
        # Realistic RTCM message (GPS corrections)
        rtcm_header = b"\xd3\x00\x13"  # RTCM3 header
        rtcm_payload = b"\x3e\xd0\x00\x03" + b"\x12\x34\x56\x78" * 10
        rtcm_crc = b"\xaa\xbb\xcc"
        rtcm_message = rtcm_header + rtcm_payload + rtcm_crc
        
        result = pipeline.process(rtcm_message)
        
        # Verify complete chain
        assert result.original_data == rtcm_message
        assert result.fec_result.ldpc_applied is True
        assert result.fec_result.rs_applied is True
        assert result.frame.frame_type.value == 1  # SUBFRAME
        assert result.signal.num_samples > 0
        
        # Verify signal is ready for transmission
        i_data, q_data = result.signal.get_iq_data()
        assert len(i_data) > 0
        assert len(q_data) > 0
    
    def test_continuous_stream_processing(self):
        """Test processing continuous stream of packets."""
        pipeline = BroadcastPipeline()
        
        # Simulate continuous RTCM stream
        num_packets = 20
        packets = [f"RTCM packet {i}".encode() for i in range(num_packets)]
        
        results = pipeline.process_batch(packets)
        
        assert len(results) == num_packets
        assert pipeline.packets_processed == num_packets
        
        # All packets should have sequential frame IDs
        for i, result in enumerate(results):
            assert result.frame.frame_id == i
    
    def test_performance_benchmark(self):
        """Test pipeline performance."""
        pipeline = BroadcastPipeline()
        
        # Process 100 packets
        packets = [b"Performance test data" * 5 for _ in range(100)]
        
        results = pipeline.process_batch(packets)
        
        # Calculate average processing time
        avg_time = sum(r.processing_time_ms for r in results) / len(results)
        
        # Should process reasonably fast (< 100ms per packet on average)
        assert avg_time < 100.0
        
        # Get throughput
        stats = pipeline.get_stats()
        total_input_kb = stats['total_input_bytes'] / 1024
        total_time_sec = sum(r.processing_time_ms for r in results) / 1000
        throughput_kbps = total_input_kb / total_time_sec if total_time_sec > 0 else 0
        
        # Should have reasonable throughput
        assert throughput_kbps > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])