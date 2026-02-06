#!/usr/bin/env python3
"""Test full OFDM modulation round-trip."""

import numpy as np
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder
from broadcast.framebuilder import FrameBuilder, FrameConfig
from broadcast.fecencoder import FECEncoder
from broadcast.config import ModulationScheme, FFTSize, GuardInterval, PilotPattern

def test_ofdm_roundtrip():
    """Test if OFDM modulation + demodulation preserves bit order."""
    
    # Known test pattern
    test_data = bytes([0x55, 0xAA, 0x55, 0xAA, 0xFF, 0x00, 0xFF, 0x00] * 32)  # 256 bytes
    
    print(f"Original data: {len(test_data)} bytes")
    print(f"First 8 bytes (hex): {test_data[:8].hex()}")
    
    # TX: Encode
    encoder = FECEncoder()
    fec_result = encoder.encode(test_data)
    print(f"\nAfter FEC: {len(fec_result.encoded_data)} bytes")
    
    # TX: Frame
    config = FrameConfig(
        fft_size=FFTSize.FFT_8K,
        guard_interval=GuardInterval.GI_1_8,
        modulation=ModulationScheme.QPSK,
        frame_duration_ms=50.0,
        pilots_enabled=True,
        pilot_pattern=PilotPattern.SP3_4,
        time_interleaving=False  # Disabled for test
    )
    
    builder = FrameBuilder()
    frame = builder.build_frame(fec_result, config)
    print(f"Frame payload: {len(frame.payload)} bytes")
    
    # TX: Modulate
    modulator = OFDMModulator()
    signal = modulator.modulate(frame, config)
    print(f"OFDM signal: {signal.num_samples} samples")
    
    # RX: Demodulate
    decoder = BroadcastDecoder()
    payload_bytes, payload_llrs, snr, evm = decoder.ofdm_demodulate(signal.time_domain_signal)
    
    print(f"\nRX demodulated: {len(payload_bytes)} bytes")
    print(f"First 8 bytes (hex): {payload_bytes[:8].hex()}")
    
    # Compare
    min_len = min(len(frame.payload), len(payload_bytes))
    matches = sum(1 for i in range(min_len) if frame.payload[i] == payload_bytes[i])
    
    print(f"\nByte match: {matches}/{min_len} ({100*matches/min_len:.1f}%)")
    
    if matches == min_len:
        print("✅ OFDM round-trip is BYTE-PERFECT!")
    else:
        print(f"❌ {min_len - matches} byte errors detected!")
        
        # Show first few mismatches
        print("\nFirst 10 mismatches:")
        print("Pos | TX (hex) | RX (hex) | TX (bin)     | RX (bin)")
        print("----|----------|----------|--------------|-------------")
        
        count = 0
        for i in range(min(min_len, 100)):
            if frame.payload[i] != payload_bytes[i]:
                tx_byte = frame.payload[i]
                rx_byte = payload_bytes[i]
                print(f"{i:3d} |   0x{tx_byte:02x}   |   0x{rx_byte:02x}   | {tx_byte:08b} | {rx_byte:08b}")
                count += 1
                if count >= 10:
                    break

if __name__ == "__main__":
    test_ofdm_roundtrip()