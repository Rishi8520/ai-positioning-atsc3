# test_full_diagnostic.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.config import FFTSize, PilotPattern, ModulationScheme
from broadcast.framebuilder import FrameBuilder
from broadcast.fecencoder import FECEncoder

print("=" * 70)
print("COMPLETE DIAGNOSTIC")
print("=" * 70)

# Test data
test_data = b"HELLO WORLD! " * 200

# Expected values per ATSC 3.0
fft_size = FFTSize.FFT_8K
pilot_pattern = PilotPattern.SP3_4
modulation = ModulationScheme.QPSK

print("\n1. ATSC 3.0 SPEC VALUES")
print("-" * 70)
print(f"FFT Size: {fft_size.value}")
print(f"Active carriers (spec): 6913")
print(f"Pilot pattern: SP3_4 (dx=3, dy=4)")
print(f"Pilots per symbol: 6913 // 3 = 2304")
print(f"Data carriers per symbol: 6913 - 2304 = 4609")

# Encode
print("\n2. FRAMEBUILDER OUTPUT")
print("-" * 70)
fec_encoder = FECEncoder()
frame_builder = FrameBuilder()

fec_result = fec_encoder.encode(test_data)
frame = frame_builder.build_frame(fec_result)

print(f"Active carriers (framebuilder): {frame.metadata['active_carriers']}")
print(f"Data symbols: {frame.metadata['data_symbols']}")
print(f"Pilot symbols: {frame.metadata['pilot_symbols']}")

if frame.pilots is not None:
    print(f"Pilots generated: {len(frame.pilots)}")
else:
    print("⚠️ NO PILOTS GENERATED!")

print(f"Payload size: {len(frame.payload)} bytes")

# Calculate what decoder expects
dx, dy = 3, 4
spec_active = 6913
spec_pilots_per_sym = spec_active // dx
spec_data_per_sym = spec_active - spec_pilots_per_sym

print("\n3. DECODER EXPECTATIONS")
print("-" * 70)
print(f"Active carriers: {spec_active}")
print(f"Pilots per symbol: {spec_pilots_per_sym}")
print(f"Data carriers per symbol: {spec_data_per_sym}")

print("\n4. MISMATCH ANALYSIS")
print("-" * 70)
encoder_data_per_sym = frame.metadata['active_carriers']
if encoder_data_per_sym == spec_data_per_sym:
    print(f"✓ Encoder/Decoder MATCH: {encoder_data_per_sym} data carriers/symbol")
else:
    print(f"✗ MISMATCH!")
    print(f"  Encoder: {encoder_data_per_sym} data carriers/symbol")
    print(f"  Decoder expects: {spec_data_per_sym} data carriers/symbol")
    print(f"  Difference: {abs(encoder_data_per_sym - spec_data_per_sym)}")

print("\n" + "=" * 70)