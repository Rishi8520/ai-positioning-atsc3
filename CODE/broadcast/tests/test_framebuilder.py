"""
Unit tests for ATSC 3.0 Frame Builder.
"""

import pytest
import numpy as np
from broadcast.framebuilder import (
    FrameBuilder,
    FrameConfig,
    ATSCFrame,
    FrameType
)
from broadcast.fecencoder import FECEncoder
from broadcast.config import ModulationScheme, GuardInterval, FFTSize


class TestFrameConfig:
    """Test suite for FrameConfig dataclass."""
    
    def test_valid_config(self):
        """Test creating valid frame configuration."""
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0
        )
        
        assert config.fft_size == FFTSize.FFT_8K
        assert config.guard_interval == GuardInterval.GI_1_8
        assert config.modulation == ModulationScheme.QPSK
        assert config.frame_duration_ms == 50.0
        assert config.pilots_enabled is True  # Default
        assert config.time_interleaving is True  # Default
    
    def test_custom_config(self):
        """Test configuration with custom parameters."""
        config = FrameConfig(
            fft_size=FFTSize.FFT_16K,
            guard_interval=GuardInterval.GI_1_4,
            modulation=ModulationScheme.QAM16,
            frame_duration_ms=100.0,
            pilots_enabled=False,
            time_interleaving=False
        )
        
        assert config.pilots_enabled is False
        assert config.time_interleaving is False
    
    def test_invalid_frame_duration(self):
        """Test that invalid frame duration raises error."""
        with pytest.raises(ValueError, match="Frame duration must be positive"):
            FrameConfig(
                fft_size=FFTSize.FFT_8K,
                guard_interval=GuardInterval.GI_1_8,
                modulation=ModulationScheme.QPSK,
                frame_duration_ms=-10.0  # Invalid: negative
            )


class TestATSCFrame:
    """Test suite for ATSCFrame dataclass."""
    
    def test_frame_properties(self):
        """Test frame property calculations."""
        frame = ATSCFrame(
            frame_id=0,
            frame_type=FrameType.SUBFRAME,
            preamble=b'\x00' * 200,
            payload=b'\x00' * 2000,
            pilots=None,
            total_symbols=100,
            modulation=ModulationScheme.QPSK,
            timestamp=1234567890,
            metadata={}
        )
        
        assert frame.total_bytes == 2200  # 200 + 2000
        assert frame.payload_bytes == 2000
        assert frame.frame_id == 0
        assert frame.total_symbols == 100


class TestFrameBuilder:
    """Test suite for Frame Builder."""
    
    def test_builder_initialization(self):
        """Test builder initializes with defaults."""
        builder = FrameBuilder()
        
        assert builder.default_config.fft_size == FFTSize.FFT_8K
        assert builder.default_config.guard_interval == GuardInterval.GI_1_8
        assert builder.default_config.modulation == ModulationScheme.QPSK
        assert builder.default_config.frame_duration_ms == 50.0
        assert builder.frame_counter == 0
        assert builder.frames_built == 0
    
    def test_builder_custom_initialization(self):
        """Test builder with custom default configuration."""
        custom_config = FrameConfig(
            fft_size=FFTSize.FFT_16K,
            guard_interval=GuardInterval.GI_1_4,
            modulation=ModulationScheme.QAM16,
            frame_duration_ms=100.0
        )
        builder = FrameBuilder(default_config=custom_config)
        
        assert builder.default_config.fft_size == FFTSize.FFT_16K
        assert builder.default_config.modulation == ModulationScheme.QAM16
    
    def test_build_basic_frame(self):
        """Test building basic frame with default configuration."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        test_data = b"Test data for ATSC 3.0 frame"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        
        # Verify frame structure
        assert isinstance(frame, ATSCFrame)
        assert frame.frame_id == 0
        assert frame.frame_type == FrameType.SUBFRAME
        assert len(frame.preamble) > 0
        assert len(frame.payload) > 0
        assert frame.total_symbols > 0
        assert frame.timestamp > 0
        
        # Verify builder state updated
        assert builder.frames_built == 1
        assert builder.frame_counter == 1
    
    def test_frame_has_preamble(self):
        """Test that frame contains valid preamble."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        test_data = b"Preamble test data"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        
        # Verify preamble has ATSC sync pattern
        assert len(frame.preamble) >= 4
        assert frame.preamble[0] == 0xB8  # ATSC 3.0 sync byte
        assert frame.preamble[1] == 0x47
    
    def test_frame_with_different_fft_sizes(self):
        """Test frame building with different FFT sizes."""
        encoder = FECEncoder()
        test_data = b"FFT size test"
        fec_result = encoder.encode(test_data)
        
        fft_sizes = [FFTSize.FFT_8K, FFTSize.FFT_16K, FFTSize.FFT_32K]
        
        for fft_size in fft_sizes:
            config = FrameConfig(
                fft_size=fft_size,
                guard_interval=GuardInterval.GI_1_8,
                modulation=ModulationScheme.QPSK,
                frame_duration_ms=50.0
            )
            builder = FrameBuilder(default_config=config)
            frame = builder.build_frame(fec_result)
            
            assert isinstance(frame, ATSCFrame)
            assert frame.metadata['fft_size'] == fft_size.value
            # Larger FFT should have more active carriers
            assert frame.total_symbols > 0
    
    def test_frame_with_different_modulations(self):
        """Test frame building with different modulation schemes."""
        encoder = FECEncoder()
        test_data = b"Modulation test data"
        fec_result = encoder.encode(test_data)
        
        modulations = [
            ModulationScheme.BPSK,
            ModulationScheme.QPSK,
            ModulationScheme.QAM16,
            ModulationScheme.QAM64,
            ModulationScheme.QAM256
        ]
        
        for modulation in modulations:
            config = FrameConfig(
                fft_size=FFTSize.FFT_8K,
                guard_interval=GuardInterval.GI_1_8,
                modulation=modulation,
                frame_duration_ms=50.0
            )
            builder = FrameBuilder(default_config=config)
            frame = builder.build_frame(fec_result)
            
            assert frame.modulation == modulation
            assert frame.metadata['modulation'] == modulation.name
    
    def test_frame_with_different_guard_intervals(self):
        """Test frame building with different guard intervals."""
        encoder = FECEncoder()
        test_data = b"Guard interval test"
        fec_result = encoder.encode(test_data)
        
        guard_intervals = [
            GuardInterval.GI_1_4,
            GuardInterval.GI_1_8,
            GuardInterval.GI_1_16,
            GuardInterval.GI_1_32
        ]
        
        for gi in guard_intervals:
            config = FrameConfig(
                fft_size=FFTSize.FFT_8K,
                guard_interval=gi,
                modulation=ModulationScheme.QPSK,
                frame_duration_ms=50.0
            )
            builder = FrameBuilder(default_config=config)
            frame = builder.build_frame(fec_result)
            
            assert isinstance(frame, ATSCFrame)
            assert frame.metadata['guard_interval'] == gi.value
    
    def test_frame_with_pilots_enabled(self):
        """Test frame with pilot symbols enabled."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0,
            pilots_enabled=True
        )
        
        test_data = b"Pilot test data"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result, config)
        
        # Verify pilots are generated
        assert frame.pilots is not None
        assert isinstance(frame.pilots, np.ndarray)
        assert len(frame.pilots) > 0
        assert frame.metadata['pilot_symbols'] > 0
    
    def test_frame_with_pilots_disabled(self):
        """Test frame without pilot symbols."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0,
            pilots_enabled=False
        )
        
        test_data = b"No pilots test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result, config)
        
        # Verify no pilots
        assert frame.pilots is None
        assert frame.metadata['pilot_symbols'] == 0
    
    def test_frame_metadata(self):
        """Test that frame contains complete metadata."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        test_data = b"Metadata test data"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        
        # Verify metadata keys
        assert 'fft_size' in frame.metadata
        assert 'guard_interval' in frame.metadata
        assert 'modulation' in frame.metadata
        assert 'frame_duration_ms' in frame.metadata
        assert 'data_symbols' in frame.metadata
        assert 'pilot_symbols' in frame.metadata
        assert 'original_data_size' in frame.metadata
        assert 'fec_overhead_bytes' in frame.metadata
        assert 'fec_code_rate' in frame.metadata
        
        # Verify metadata values
        assert frame.metadata['original_data_size'] == len(test_data)
        assert frame.metadata['fec_overhead_bytes'] > 0
        assert 0 < frame.metadata['fec_code_rate'] <= 1.0
    
    def test_frame_with_time_interleaving(self):
        """Test frame with time interleaving enabled."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0,
            time_interleaving=True
        )
        
        test_data = b"Time interleaving test data"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result, config)
        
        # Payload should be interleaved (different from input)
        assert isinstance(frame, ATSCFrame)
        assert len(frame.payload) > 0
    
    def test_frame_without_time_interleaving(self):
        """Test frame without time interleaving."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0,
            time_interleaving=False
        )
        
        test_data = b"No interleaving test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result, config)
        
        assert isinstance(frame, ATSCFrame)
    
    def test_build_multiple_frames(self):
        """Test building multiple frames from list."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        # Create multiple FEC-encoded data
        test_data_list = [
            b"Frame 1 data",
            b"Frame 2 data",
            b"Frame 3 data",
            b"Frame 4 data"
        ]
        
        fec_results = [encoder.encode(data) for data in test_data_list]
        frames = builder.build_multiple_frames(fec_results)
        
        # Verify all frames built
        assert len(frames) == 4
        assert all(isinstance(f, ATSCFrame) for f in frames)
        
        # Verify frame IDs are sequential
        for i, frame in enumerate(frames):
            assert frame.frame_id == i
        
        assert builder.frames_built == 4
    
    def test_frame_counter_increments(self):
        """Test that frame counter increments correctly."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        test_data = b"Counter test"
        fec_result = encoder.encode(test_data)
        
        # Build multiple frames
        frame1 = builder.build_frame(fec_result)
        frame2 = builder.build_frame(fec_result)
        frame3 = builder.build_frame(fec_result)
        
        assert frame1.frame_id == 0
        assert frame2.frame_id == 1
        assert frame3.frame_id == 2
        assert builder.frame_counter == 3
    
    def test_symbol_count_calculation(self):
        """Test OFDM symbol count calculation."""
        encoder = FECEncoder()
        test_data = b"Symbol count test"
        fec_result = encoder.encode(test_data)
        
        # Different frame durations should produce different symbol counts
        config_50ms = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0
        )
        
        config_100ms = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=100.0
        )
        
        builder = FrameBuilder()
        frame_50 = builder.build_frame(fec_result, config_50ms)
        frame_100 = builder.build_frame(fec_result, config_100ms)
        
        # Longer frame should have more symbols
        assert frame_100.total_symbols > frame_50.total_symbols
    
    def test_bits_per_symbol_calculation(self):
        """Test bits per modulation symbol calculation."""
        builder = FrameBuilder()
        
        config_qpsk = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0
        )
        
        config_qam16 = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QAM16,
            frame_duration_ms=50.0
        )
        
        # Verify internal calculation
        assert builder._bits_per_symbol(config_qpsk) == 2
        assert builder._bits_per_symbol(config_qam16) == 4
    
    def test_active_carriers_calculation(self):
        """Test active carrier calculation."""
        builder = FrameBuilder()
        
        config_8k = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0
        )
        
        config_16k = FrameConfig(
            fft_size=FFTSize.FFT_16K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0
        )
        
        carriers_8k = builder._active_carriers(config_8k)
        carriers_16k = builder._active_carriers(config_16k)
        
        # Larger FFT should have more active carriers
        assert carriers_16k > carriers_8k
        # Should be less than total FFT size (due to guards and pilots)
        assert carriers_8k < FFTSize.FFT_8K.value
        assert carriers_16k < FFTSize.FFT_16K.value
    
    def test_frame_with_large_payload(self):
        """Test frame building with large payload."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        # Large RTCM frame (500 bytes)
        test_data = b"\x00" * 500
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        
        assert isinstance(frame, ATSCFrame)
        assert frame.payload_bytes > len(test_data)
        assert frame.total_symbols > builder.BOOTSTRAP_SYMBOLS  # Fixed: was PREAMBLE_SYMBOLS

    def test_frame_with_empty_data(self):
        """Test frame building with empty data."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        fec_result = encoder.encode(b"")
        frame = builder.build_frame(fec_result)
        
        # Should still produce valid frame with preamble
        assert isinstance(frame, ATSCFrame)
        assert len(frame.preamble) > 0
    
    def test_get_stats(self):
        """Test statistics reporting."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        
        test_data = b"Stats test"
        fec_result = encoder.encode(test_data)
        
        # Build some frames
        builder.build_frame(fec_result)
        builder.build_frame(fec_result)
        builder.build_frame(fec_result)
        
        stats = builder.get_stats()
        
        assert stats['frames_built'] == 3
        assert stats['current_frame_id'] == 3
        assert stats['default_fft_size'] == 'FFT_8K'
        assert stats['default_modulation'] == 'QPSK'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])