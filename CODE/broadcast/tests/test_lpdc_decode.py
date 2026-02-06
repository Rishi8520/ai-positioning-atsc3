# test_ldpc_decode.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.fecencoder import FECEncoder
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.config import FECCodeRate

print("=" * 70)
print("LDPC ENCODE/DECODE TEST")
print("=" * 70)

test_data = b"HELLO" * 20  # 100 bytes

# Encode
encoder = FECEncoder()
fec_result = encoder.encode(test_data)

print(f"\n1. FEC ENCODING")
print("-" * 70)
print(f"Original: {len(test_data)} bytes")
print(f"FEC encoded: {len(fec_result.encoded_data)} bytes")

# Check LDPC parameters
rate = FECCodeRate.RATE_8_15
n, k = 16200, 8640  # bits
n_bytes, k_bytes = n//8, k//8  # 2025, 1080 bytes

print(f"\nLDPC parameters:")
print(f"  n = {n} bits ({n_bytes} bytes)")
print(f"  k = {k} bits ({k_bytes} bytes)")
print(f"  Rate = {k/n} = 8/15")

# Decode LDPC directly
decoder = BroadcastDecoder(DecoderConfig())
ldpc_decoded = decoder._decode_ldpc(fec_result.encoded_data)

print(f"\n2. LDPC DECODING")
print("-" * 70)
print(f"Input: {len(fec_result.encoded_data)} bytes")
print(f"Output: {len(ldpc_decoded)} bytes")
print(f"Expected: {k_bytes} bytes (k bits)")

if len(ldpc_decoded) == k_bytes:
    print("✓ LDPC decoding produced correct size")
else:
    print(f"✗ LDPC decoding size mismatch")
    print(f"   Expected: {k_bytes} bytes")
    print(f"   Got: {len(ldpc_decoded)} bytes")
    print(f"   Difference: {len(ldpc_decoded) - k_bytes} bytes")

# Check first few bytes
print(f"\n3. DATA CHECK")
print("-" * 70)
print(f"First 20 bytes of LDPC output: {ldpc_decoded[:20].hex()}")
print(f"First 20 bytes of FEC input:   {fec_result.encoded_data[:20].hex()}")

if ldpc_decoded[:20] == fec_result.encoded_data[:20]:
    print("✓ First 20 bytes match (systematic code)")
else:
    print("✗ First 20 bytes don't match")

print("=" * 70)
