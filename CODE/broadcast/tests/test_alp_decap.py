# test_alp_decap.py
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.decoder import BroadcastDecoder, DecoderConfig

print("=" * 70)
print("ALP DECAPSULATION TEST")
print("=" * 70)

test_data = b"HELLO" * 20  # 100 bytes

print(f"\n1. ORIGINAL DATA (No ALP header)")
print("-" * 70)
print(f"Size: {len(test_data)} bytes")
print(f"First 10 bytes (hex): {test_data[:10].hex()}")
print(f"First 10 bytes (ascii): {test_data[:10]}")

# Test decoder's ALP decapsulation on plain data
decoder = BroadcastDecoder(DecoderConfig())
decoded = decoder._decapsulate_alp(test_data)

print(f"\n2. AFTER ALP DECAPSULATION")
print("-" * 70)
print(f"Size: {len(decoded)} bytes")
print(f"First 10 bytes (hex): {decoded[:10].hex()}")

# Check byte 0
byte0 = test_data[0]
print(f"\n3. ANALYSIS")
print("-" * 70)
print(f"First byte: 0x{byte0:02x} = {byte0} = '{chr(byte0)}'")
print(f"Bit 7 (header mode): {(byte0 >> 7) & 1}")
print(f"Lower 7 bits: {byte0 & 0x7F}")

# The decoder thinks this is ALP if byte0 bit 7 is set a certain way
# Let's see what it's detecting

if decoded == test_data:
    print("\n✓ Data passes through unchanged")
else:
    print(f"\n✗ Data was modified!")
    print(f"   Original: {len(test_data)} bytes")
    print(f"   Decoded: {len(decoded)} bytes")
    print(f"   Lost: {len(test_data) - len(decoded)} bytes")
    
    # Show offset
    for i in range(min(20, len(decoded))):
        print(f"   [{i:2d}] TX: 0x{test_data[i]:02x} ({chr(test_data[i])})  RX: 0x{decoded[i]:02x} ({chr(decoded[i])})")

print("=" * 70)