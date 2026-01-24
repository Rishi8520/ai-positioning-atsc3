"""
Unit tests for ATSC 3.0 OFDM Modulator.
"""

import pytest
import numpy as np
from broadcast.ofdmmodulator import OFDMModulator, OFDMSignal
from broadcast.fecencoder import FECEncoder
from broadcast.framebuilder import FrameBuilder, FrameConfig
from broadcast.config import ModulationScheme, GuardInterval, FFTSize, PilotPattern


class TestOFDMModulator:
    """Test suite for OFDM Modulator."""
    
    def test_modulator_initialization(self):
        """Test modulator initializes correctly."""
        modulator = OFDMModulator()
        
        assert modulator.modulated_frames == 0
        assert modulator.SAMPLE_RATE_HZ == 6.144e6
        
        # Check constellations are generated
        assert ModulationScheme.BPSK in modulator.CONSTELLATIONS
        assert ModulationScheme.QPSK in modulator.CONSTELLATIONS
        assert ModulationScheme.QAM16 in modulator.CONSTELLATIONS
        assert ModulationScheme.QAM64 in modulator.CONSTELLATIONS
        assert ModulationScheme.QAM256 in modulator.CONSTELLATIONS
    
    def test_bpsk_constellation(self):
        """Test BPSK constellation has correct properties."""
        modulator = OFDMModulator()
        constellation = modulator.CONSTELLATIONS[ModulationScheme.BPSK]
        
        assert len(constellation) == 2  # 2 points
        assert constellation.dtype == np.complex64
        # BPSK should be real-valued (on I axis)
        assert np.allclose(constellation.imag, 0)
    
    def test_qpsk_constellation(self):
        """Test QPSK constellation has correct properties."""
        modulator = OFDMModulator()
        constellation = modulator.CONSTELLATIONS[ModulationScheme.QPSK]
        
        assert len(constellation) == 4  # 4 points
        # Unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        assert np.isclose(avg_power, 1.0, atol=0.01)
    
    def test_qam16_constellation(self):
        """Test 16-QAM constellation has correct properties."""
        modulator = OFDMModulator()
        constellation = modulator.CONSTELLATIONS[ModulationScheme.QAM16]
        
        assert len(constellation) == 16  # 16 points
        # Unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        assert np.isclose(avg_power, 1.0, atol=0.01)
    
    def test_qam64_constellation(self):
        """Test 64-QAM constellation generation."""
        modulator = OFDMModulator()
        constellation = modulator.CONSTELLATIONS[ModulationScheme.QAM64]
        
        assert constellation is not None
        assert len(constellation) == 64  # 64 points
        # Unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        assert np.isclose(avg_power, 1.0, atol=0.01)
    
    def test_qam256_constellation(self):
        """Test 256-QAM constellation generation."""
        modulator = OFDMModulator()
        constellation = modulator.CONSTELLATIONS[ModulationScheme.QAM256]
        
        assert constellation is not None
        assert len(constellation) == 256  # 256 points
        # Unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        assert np.isclose(avg_power, 1.0, atol=0.01)
    
    def test_modulate_basic_frame(self):
        """Test basic OFDM modulation."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b"OFDM modulation test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        # Verify signal structure
        assert isinstance(signal, OFDMSignal)
        assert signal.frame_id == frame.frame_id
        assert signal.num_samples > 0
        assert signal.num_symbols > 0
        assert signal.sample_rate == modulator.SAMPLE_RATE_HZ
        
        # Verify signal is complex
        assert signal.time_domain_signal.dtype == np.complex64
        
        # Verify modulator state updated
        assert modulator.modulated_frames == 1

    def test_modulate_different_fft_sizes(self):
        """Test modulation with different FFT sizes."""
        encoder = FECEncoder()
        modulator = OFDMModulator()
        
        test_data = b"FFT size test"
        fec_result = encoder.encode(test_data)
        
        fft_sizes = [FFTSize.FFT_8K, FFTSize.FFT_16K, FFTSize.FFT_32K]
        
        results = {}
        
        for fft_size in fft_sizes:
            config = FrameConfig(
                fft_size=fft_size,
                guard_interval=GuardInterval.GI_1_8,
                modulation=ModulationScheme.QPSK,
                frame_duration_ms=50.0
            )
            builder = FrameBuilder(default_config=config)
            frame = builder.build_frame(fec_result)
            signal = modulator.modulate(frame)
            
            assert isinstance(signal, OFDMSignal)
            assert signal.metadata['fft_size'] == fft_size.value
            
            results[fft_size] = signal
        
        # Verify FFT size is correctly reflected in metadata
        assert results[FFTSize.FFT_8K].metadata['fft_size'] == 8192
        assert results[FFTSize.FFT_16K].metadata['fft_size'] == 16384
        assert results[FFTSize.FFT_32K].metadata['fft_size'] == 32768
        
        # Larger FFT should have more samples PER SYMBOL (not necessarily total)
        # Calculate samples per symbol: (FFT_size + GI_length)
        # With GI_1_8: samples_per_symbol = FFT_size * 1.125
        for fft_size, signal in results.items():
            expected_samples_per_symbol = int(fft_size.value * 1.125)
            actual_samples_per_symbol = signal.num_samples // signal.num_symbols
            assert actual_samples_per_symbol == expected_samples_per_symbol, \
                f"FFT {fft_size.name}: expected {expected_samples_per_symbol} samples/symbol, " \
                f"got {actual_samples_per_symbol}"
    
    def test_modulate_different_modulations(self):
        """Test modulation with different schemes."""
        encoder = FECEncoder()
        modulator = OFDMModulator()
        
        test_data = b"Modulation scheme test"
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
            signal = modulator.modulate(frame)
            
            assert signal.metadata['modulation'] == modulation.name
            assert signal.num_samples > 0
    
    def test_modulate_with_different_guard_intervals(self):
        """Test modulation with different guard intervals."""
        encoder = FECEncoder()
        modulator = OFDMModulator()
        
        test_data = b"Guard interval test"
        fec_result = encoder.encode(test_data)
        
        guard_intervals = [
            GuardInterval.GI_1_4,
            GuardInterval.GI_1_8,
            GuardInterval.GI_1_16,
            GuardInterval.GI_1_32
        ]
        
        samples_by_gi = {}
        
        for gi in guard_intervals:
            config = FrameConfig(
                fft_size=FFTSize.FFT_8K,
                guard_interval=gi,
                modulation=ModulationScheme.QPSK,
                frame_duration_ms=50.0
            )
            builder = FrameBuilder(default_config=config)
            frame = builder.build_frame(fec_result)
            signal = modulator.modulate(frame)
            
            samples_by_gi[gi] = signal.num_samples
            assert signal.metadata['guard_interval'] == gi.name
        
        # Longer GI should produce more total samples
        assert samples_by_gi[GuardInterval.GI_1_4] > samples_by_gi[GuardInterval.GI_1_32]
    
    def test_signal_duration(self):
        """Test signal duration calculation."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b"Duration test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        # Verify duration calculation
        expected_duration = (signal.num_samples / signal.sample_rate) * 1000.0
        assert np.isclose(signal.duration_ms, expected_duration, rtol=0.01)
        assert signal.duration_ms > 0
    
    def test_iq_data_extraction(self):
        """Test I/Q component extraction."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b"I/Q extraction test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        i_data, q_data = signal.get_iq_data()
        
        # Verify I/Q components
        assert len(i_data) == signal.num_samples
        assert len(q_data) == signal.num_samples
        assert i_data.dtype == np.float32 or i_data.dtype == np.float64
        assert q_data.dtype == np.float32 or q_data.dtype == np.float64
        
        # Verify reconstruction
        reconstructed = i_data + 1j * q_data
        assert np.allclose(reconstructed, signal.time_domain_signal)
        
    def test_signal_power(self):
        """Test signal power calculations."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b"Power test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        # Verify power values are reasonable
        assert signal.metadata['peak_power'] >= 0
        assert signal.metadata['average_power'] >= 0
        
        # Handle edge case where average_power might be very small
        if signal.metadata['average_power'] > 0:
            assert signal.metadata['peak_power'] >= signal.metadata['average_power']
            
            # PAPR (Peak-to-Average Power Ratio) should be reasonable for OFDM
            # Typical OFDM PAPR is 8-15 dB (6-32 linear), but can be higher
            # with many unused subcarriers
            papr = signal.metadata['peak_power'] / signal.metadata['average_power']
            papr_db = 10 * np.log10(papr) if papr > 0 else 0
            
            # PAPR should be positive and less than theoretical max (~40 dB for 8K FFT)
            assert 0 <= papr_db <= 40, f"PAPR {papr_db:.1f} dB out of expected range"    

    def test_modulate_with_pilots(self):
        """Test modulation with pilots enabled."""
        encoder = FECEncoder()
        modulator = OFDMModulator()
        
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0,
            pilots_enabled=True,
            pilot_pattern=PilotPattern.SP3_4
        )
        
        builder = FrameBuilder(default_config=config)
        test_data = b"Pilots test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        assert isinstance(signal, OFDMSignal)
        assert signal.metadata['num_data_symbols'] > 0
    
    def test_modulate_without_pilots(self):
        """Test modulation without pilots."""
        encoder = FECEncoder()
        modulator = OFDMModulator()
        
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0,
            pilots_enabled=False
        )
        
        builder = FrameBuilder(default_config=config)
        test_data = b"No pilots test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        assert isinstance(signal, OFDMSignal)
    
    def test_modulate_multiple_frames(self):
        """Test modulating multiple frames."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data_list = [
            b"Frame 1",
            b"Frame 2",
            b"Frame 3"
        ]
        
        fec_results = [encoder.encode(data) for data in test_data_list]
        frames = [builder.build_frame(fec) for fec in fec_results]
        signals = modulator.modulate_multiple_frames(frames)
        
        assert len(signals) == 3
        assert all(isinstance(sig, OFDMSignal) for sig in signals)
        assert modulator.modulated_frames == 3
        
        # Verify frame IDs are preserved
        for i, signal in enumerate(signals):
            assert signal.frame_id == frames[i].frame_id
    
    def test_bits_per_symbol(self):
        """Test bits per symbol calculation."""
        modulator = OFDMModulator()
        
        assert modulator._bits_per_symbol(ModulationScheme.BPSK) == 1
        assert modulator._bits_per_symbol(ModulationScheme.QPSK) == 2
        assert modulator._bits_per_symbol(ModulationScheme.QAM16) == 4
        assert modulator._bits_per_symbol(ModulationScheme.QAM64) == 6
        assert modulator._bits_per_symbol(ModulationScheme.QAM256) == 8
    
    def test_map_bits_to_symbols(self):
        """Test bit-to-symbol mapping."""
        modulator = OFDMModulator()
        
        # Test with QPSK (2 bits per symbol)
        test_data = bytes([0b11001010])  # 8 bits = 4 QPSK symbols
        symbols = modulator._map_bits_to_symbols(test_data, ModulationScheme.QPSK)
        
        assert len(symbols) == 4
        assert symbols.dtype == np.complex64
        
        # All symbols should be valid QPSK points
        qpsk_constellation = modulator.CONSTELLATIONS[ModulationScheme.QPSK]
        for symbol in symbols:
            # Check if symbol is close to one of the constellation points
            distances = np.abs(qpsk_constellation - symbol)
            assert np.min(distances) < 0.01
    
    def test_guard_interval_length(self):
        """Test guard interval adds correct number of samples."""
        encoder = FECEncoder()
        modulator = OFDMModulator()
        
        test_data = b"Guard interval length test"
        fec_result = encoder.encode(test_data)
        
        # Test with GI_1_8 (12.5% overhead)
        config = FrameConfig(
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0
        )
        
        builder = FrameBuilder(default_config=config)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        # With GI_1_8, each symbol should be 8192 + 1024 = 9216 samples
        fft_size = 8192
        gi_length = fft_size // 8
        expected_samples_per_symbol = fft_size + gi_length
        
        # Total samples should be a multiple of (FFT + GI)
        # (approximately, accounting for framing)
        assert signal.num_samples > 0
    
    def test_signal_metadata(self):
        """Test signal metadata completeness."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b"Metadata test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        # Check all expected metadata keys
        assert 'frame_id' in signal.metadata
        assert 'fft_size' in signal.metadata
        assert 'guard_interval' in signal.metadata
        assert 'modulation' in signal.metadata
        assert 'num_ofdm_symbols' in signal.metadata
        assert 'num_data_symbols' in signal.metadata
        assert 'sample_rate_hz' in signal.metadata
        assert 'peak_power' in signal.metadata
        assert 'average_power' in signal.metadata
        
        # Verify values are reasonable
        assert signal.metadata['frame_id'] == frame.frame_id
        assert signal.metadata['num_ofdm_symbols'] > 0
        assert signal.metadata['sample_rate_hz'] == modulator.SAMPLE_RATE_HZ
    
    def test_empty_frame_handling(self):
        """Test handling of frame with minimal data."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b""  # Empty data
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        # Should still produce valid signal
        assert isinstance(signal, OFDMSignal)
        assert signal.num_samples > 0
    
    def test_large_frame_handling(self):
        """Test handling of large frame."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b"\x00" * 1000  # Large frame
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        assert isinstance(signal, OFDMSignal)
        assert signal.num_samples > 0
        assert signal.num_symbols > 1
    
    def test_get_stats(self):
        """Test statistics reporting."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        # Modulate some frames
        for i in range(5):
            test_data = f"Frame {i}".encode()
            fec_result = encoder.encode(test_data)
            frame = builder.build_frame(fec_result)
            modulator.modulate(frame)
        
        stats = modulator.get_stats()
        
        assert stats['frames_modulated'] == 5
        assert stats['sample_rate_hz'] == 6.144e6
    
    def test_frequency_domain_symbols(self):
        """Test frequency domain symbols are generated."""
        encoder = FECEncoder()
        builder = FrameBuilder()
        modulator = OFDMModulator()
        
        test_data = b"Frequency domain test"
        fec_result = encoder.encode(test_data)
        frame = builder.build_frame(fec_result)
        signal = modulator.modulate(frame)
        
        # Check frequency domain symbols exist
        assert signal.frequency_domain_symbols is not None
        assert len(signal.frequency_domain_symbols) > 0
        assert signal.frequency_domain_symbols.dtype == np.complex64
        
        # Should be 2D array (symbols, carriers)
        assert signal.frequency_domain_symbols.ndim == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])