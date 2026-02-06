# test_end_to_end_simple.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.framebuilder import FrameBuilder, FrameConfig
from broadcast.config import ModulationScheme, FFTSize, GuardInterval, FECCodeRate

print("=" * 70)
print("END-TO-END TEST (FEC ON, Time Interleaving OFF)")
print("=" * 70)

# Test data
test_data = b"HELLO" * 20  # 100 bytes

print(f"\n1. ENCODING")
print("-" * 70)
print(f"Input: {len(test_data)} bytes")

# Encode with FEC
from broadcast.fecencoder import FECEncoder
fec_encoder = FECEncoder()
fec_result = fec_encoder.encode(test_data)
print(f"After FEC: {len(fec_result.encoded_data)} bytes")

# Build frame WITH time_interleaving=False
frame_config = FrameConfig(
    fft_size=FFTSize.FFT_8K,
    guard_interval=GuardInterval.GI_1_8,
    modulation=ModulationScheme.QPSK,
    frame_duration_ms=50.0,
    pilots_enabled=False,  # NO PILOTS
    time_interleaving=False  # NO TIME INTERLEAVING
)

frame_builder = FrameBuilder(default_config=frame_config)
frame = frame_builder.build_frame(fec_result, frame_config)
print(f"After framing: {len(frame.payload)} bytes")

# Modulate
modulator = OFDMModulator()
signal = modulator.modulate(frame, frame_config)
print(f"After modulation: {signal.num_samples} samples")

# Decode
print(f"\n2. DECODING")
print("-" * 70)

decoder_config = DecoderConfig(
    pilots_enabled=False,
    time_interleaving=False,  # NO TIME INTERLEAVING
    fec_rs_symbols=16
)

decoder = BroadcastDecoder(decoder_config)
result = decoder.decode(signal.time_domain_signal)

print(f"Recovered: {len(result.data)} bytes")
print(f"EVM: {result.metrics.evm_percent:.2f}%")
print(f"SNR: {result.metrics.estimated_snr_db:.2f} dB")

# Compare
if result.data == test_data:
    print("\n✓ DATA MATCH: 100%")
else:
    match = sum(a == b for a, b in zip(test_data[:len(result.data)], result.data))
    pct = (match / len(test_data)) * 100
    print(f"\n✗ DATA MISMATCH: {pct:.1f}% match")
    print(f"   Expected: {len(test_data)} bytes")
    print(f"   Got: {len(result.data)} bytes")
    
    # Show first few bytes
    print(f"\n   First 20 bytes:")
    print(f"   TX: {test_data[:20].hex()}")
    print(f"   RX: {result.data[:20].hex() if len(result.data) >= 20 else result.data.hex()}")

print("=" * 70)