# test_alp_roundtrip.py
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.alpencoder import ALPEncoder
from broadcast.decoder import BroadcastDecoder, DecoderConfig

print("=" * 70)
print("ALP ENCODE/DECODE ROUNDTRIP TEST")
print("=" * 70)

test_data = b"HELLO" * 20  # 100 bytes

print(f"\n1. ORIGINAL DATA")
print("-" * 70)
print(f"Size: {len(test_data)} bytes")
print(f"First 20 bytes: {test_data[:20].hex()}")

# ALP Encode
alp_encoder = ALPEncoder()
alp_encoded = alp_encoder.encapsulate(test_data)

print(f"\n2. ALP ENCODED")
print("-" * 70)
print(f"Size: {len(alp_encoded)} bytes")
print(f"First 20 bytes: {alp_encoded[:20].hex()}")
print(f"Header bytes: {alp_encoded[:10].hex()}")

# ALP Decode
decoder = BroadcastDecoder(DecoderConfig())
alp_decoded = decoder._decapsulate_alp(alp_encoded)

print(f"\n3. ALP DECODED")
print("-" * 70)
print(f"Size: {len(alp_decoded)} bytes")
print(f"First 20 bytes: {alp_decoded[:20].hex()}")

# Compare
if alp_decoded == test_data:
    print("\n✓ ALP ROUNDTRIP SUCCESSFUL")
else:
    print(f"\n✗ ALP ROUNDTRIP FAILED")
    print(f"   Expected: {len(test_data)} bytes")
    print(f"   Got: {len(alp_decoded)} bytes")

# Also test without ALP
print(f"\n4. TEST WITHOUT ALP (Plain Data)")
print("-" * 70)
plain_decoded = decoder._decapsulate_alp(test_data)
print(f"Decoded size: {len(plain_decoded)} bytes")
print(f"First 20 bytes: {plain_decoded[:20].hex()}")

if plain_decoded == test_data:
    print("✓ Plain data passes through correctly")
else:
    print("✗ Decoder corrupts plain data!")

print("=" * 70)