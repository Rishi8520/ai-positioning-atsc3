# test_ofdm_chain.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.framebuilder import FrameBuilder, FrameConfig
from broadcast.fecencoder import FECEncoder
from broadcast.config import ModulationScheme, FFTSize, GuardInterval, FECCodeRate

print("=" * 60)
print("OFDM Chain Test")
print("=" * 60)

# Test data
test_data = b"HELLO WORLD! " * 200  # 2600 bytes

# Encode
print("\n1. Encoding...")
fec_encoder = FECEncoder()
frame_builder = FrameBuilder()
modulator = OFDMModulator()

fec_result = fec_encoder.encode(test_data)
frame = frame_builder.build_frame(fec_result)
signal = modulator.modulate(frame)

print(f"   TX: {len(test_data)} bytes → {signal.num_samples} samples")

# Decode
print("\n2. Decoding...")
decoder = BroadcastDecoder(DecoderConfig())
result = decoder.decode(signal.time_domain_signal)

print(f"   RX: {signal.num_samples} samples → {len(result.data)} bytes")
print(f"   EVM: {result.metrics.evm_percent:.2f}%")
print(f"   SNR: {result.metrics.estimated_snr_db:.2f} dB")

# Compare
print("\n3. Verification...")
if result.data == test_data:
    print("   ✓ DATA MATCH: 100%")
else:
    match = sum(a == b for a, b in zip(test_data[:len(result.data)], result.data))
    print(f"   ✗ DATA MISMATCH: {match/len(test_data)*100:.2f}% match")
    print(f"   Original: {len(test_data)} bytes")
    print(f"   Recovered: {len(result.data)} bytes")

print("\n" + "=" * 60)