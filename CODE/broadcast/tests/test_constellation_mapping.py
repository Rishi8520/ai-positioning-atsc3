# test_constellation_mapping.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.config import ModulationScheme

print("=" * 70)
print("CONSTELLATION MAPPING TEST")
print("=" * 70)

modulator = OFDMModulator()
decoder = BroadcastDecoder(DecoderConfig())

# QPSK: 2 bits per symbol, 4 constellation points
print("\n1. ENCODER CONSTELLATION (from OFDMModulator)")
print("-" * 70)
qpsk_tx = modulator.CONSTELLATIONS[ModulationScheme.QPSK]
print(f"QPSK constellation: {qpsk_tx}")

for i in range(4):
    bits = [(i >> 1) & 1, i & 1]
    print(f"  Bits {bits} → Symbol {qpsk_tx[i]}")

# Check encoder mapping
print("\n2. ENCODER BIT MAPPING")
print("-" * 70)

# Test all 4 bit patterns
test_bits = np.array([0,0, 0,1, 1,0, 1,1], dtype=np.uint8)  # 4 symbols
test_bytes = np.packbits(test_bits).tobytes()

print(f"Test bits: {test_bits}")
print(f"Test bytes: {test_bytes.hex()}")

symbols = modulator._map_bits_to_symbols(test_bytes, ModulationScheme.QPSK)
print(f"Mapped symbols: {symbols}")

# Check decoder constellation
print("\n3. DECODER CONSTELLATION")
print("-" * 70)
qpsk_rx = decoder.QPSK_CONSTELLATION
print(f"QPSK constellation: {qpsk_rx}")

for i in range(4):
    print(f"  Index {i} → Symbol {qpsk_rx[i]} (bits: ?)")

# Test decoder demapping
print("\n4. DECODER BIT DEMAPPING")
print("-" * 70)

# Use the encoder's symbols
decoded_bits = decoder._demap_qpsk(symbols)
print(f"Decoded bits: {decoded_bits}")

# Compare
print("\n5. ROUNDTRIP CHECK")
print("-" * 70)
if np.array_equal(test_bits, decoded_bits):
    print("✓ Encoder/Decoder MATCH - Roundtrip successful")
else:
    print("✗ MISMATCH!")
    print(f"  Original: {test_bits}")
    print(f"  Decoded:  {decoded_bits}")
    
    # Show bit-by-bit
    for i in range(0, len(test_bits), 2):
        tx_bits = test_bits[i:i+2]
        rx_bits = decoded_bits[i:i+2] if i+1 < len(decoded_bits) else []
        tx_sym = symbols[i//2]
        print(f"  {tx_bits} → {tx_sym} → {rx_bits}")

print("=" * 70)