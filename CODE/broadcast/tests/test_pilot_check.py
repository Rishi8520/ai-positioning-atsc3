# test_pilot_check.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.framebuilder import FrameBuilder
from broadcast.fecencoder import FECEncoder

print("=" * 60)
print("Pilot Generation Check")
print("=" * 60)

# Test data
test_data = b"HELLO WORLD! " * 200

# Encode
fec_encoder = FECEncoder()
frame_builder = FrameBuilder()

fec_result = fec_encoder.encode(test_data)
frame = frame_builder.build_frame(fec_result)

print(f"\nFrame info:")
print(f"  Payload size: {len(frame.payload)} bytes")
print(f"  Modulation: {frame.modulation}")
print(f"  Pilots enabled: {frame.pilots is not None}")

if frame.pilots is not None:
    print(f"  Number of pilots: {len(frame.pilots)}")
    print(f"  Pilot values (first 10): {frame.pilots[:10]}")
else:
    print(f"  ⚠️  WARNING: No pilots generated!")

print(f"\nMetadata:")
for key, value in frame.metadata.items():
    print(f"  {key}: {value}")

print("=" * 60)