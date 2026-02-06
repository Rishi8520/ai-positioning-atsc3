# test_frame_structure.py
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.framebuilder import FrameBuilder, FrameConfig
from broadcast.fecencoder import FECEncoder
from broadcast.config import ModulationScheme, FFTSize, GuardInterval

print("=" * 70)
print("FRAME STRUCTURE TEST")
print("=" * 70)

test_data = b"HELLO" * 20  # 100 bytes

# Encode
fec_encoder = FECEncoder()
fec_result = fec_encoder.encode(test_data)

# Build frame
frame_config = FrameConfig(
    fft_size=FFTSize.FFT_8K,
    guard_interval=GuardInterval.GI_1_8,
    modulation=ModulationScheme.QPSK,
    frame_duration_ms=50.0,
    pilots_enabled=False,
    time_interleaving=False
)

frame_builder = FrameBuilder(default_config=frame_config)
frame = frame_builder.build_frame(fec_result, frame_config)

print(f"\n1. FRAME COMPONENTS")
print("-" * 70)
print(f"Preamble size: {len(frame.preamble)} bytes")
print(f"Payload size: {len(frame.payload)} bytes")
print(f"Total frame: {frame.total_bytes} bytes")

print(f"\n2. PREAMBLE CONTENT")
print("-" * 70)
if len(frame.preamble) > 0:
    print(f"First 20 bytes: {frame.preamble[:20].hex()}")
else:
    print("No preamble")

print(f"\n3. PAYLOAD START")
print("-" * 70)
print(f"First 20 bytes: {frame.payload[:20].hex()}")

print(f"\n4. FEC ENCODED DATA")
print("-" * 70)
print(f"FEC data size: {len(fec_result.encoded_data)} bytes")
print(f"First 20 bytes: {fec_result.encoded_data[:20].hex()}")

print(f"\n5. ORIGINAL DATA IN FEC")
print("-" * 70)
print(f"Original: {test_data[:20].hex()}")

print("=" * 70)