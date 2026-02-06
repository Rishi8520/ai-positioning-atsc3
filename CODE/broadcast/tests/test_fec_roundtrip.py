# test_fec_roundtrip.py
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.fecencoder import FECEncoder
from broadcast.decoder import BroadcastDecoder, DecoderConfig
import numpy as np

print("=" * 70)
print("FEC ENCODE/DECODE ROUNDTRIP TEST")
print("=" * 70)

test_data = b"HELLO" * 20  # 100 bytes

print(f"\n1. ORIGINAL DATA")
print("-" * 70)
print(f"Size: {len(test_data)} bytes")
print(f"First 20 bytes: {test_data[:20].hex()}")
print(f"Content: {test_data[:20]}")

# Encode
encoder = FECEncoder()
fec_result = encoder.encode(test_data)

print(f"\n2. FEC ENCODED")
print("-" * 70)
print(f"Encoded size: {len(fec_result.encoded_data)} bytes")
print(f"First 20 bytes: {fec_result.encoded_data[:20].hex()}")

# Decode (directly, no OFDM)
decoder = BroadcastDecoder(DecoderConfig())
decoded_bytes, fec_errors = decoder._decode_fec(fec_result.encoded_data)

print(f"\n3. FEC DECODED")
print("-" * 70)
print(f"Decoded size: {len(decoded_bytes)} bytes")
print(f"First 20 bytes: {decoded_bytes[:20].hex()}")
print(f"FEC errors corrected: {fec_errors}")

# De-capsulate ALP
final_data = decoder._decapsulate_alp(decoded_bytes)

print(f"\n4. AFTER ALP DECAPSULATION")
print("-" * 70)
print(f"Final size: {len(final_data)} bytes")
print(f"First 20 bytes: {final_data[:20].hex()}")

# Compare
if final_data == test_data:
    print("\n✓ PERFECT MATCH!")
else:
    print(f"\n✗ MISMATCH")
    print(f"   Expected: {len(test_data)} bytes")
    print(f"   Got: {len(final_data)} bytes")
    
    # Check for offset
    for offset in range(-10, 10):
        if offset < 0:
            match = final_data[:len(test_data)+offset] == test_data[-offset:]
        else:
            match = final_data[offset:offset+len(test_data)] == test_data
        if match:
            print(f"   ⚠️ Data matches with offset={offset}")

print("=" * 70)