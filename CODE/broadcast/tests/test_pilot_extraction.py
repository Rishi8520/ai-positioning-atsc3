# test_pilot_extraction.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.decoder import BroadcastDecoder, DecoderConfig

print("=" * 70)
print("DECODER PILOT EXTRACTION TEST")
print("=" * 70)

# Create fake signal with known structure
decoder = BroadcastDecoder(DecoderConfig())

# Build a test frequency domain symbol
fft_size = 8192
guard_size = 639
dc_index = 4096
test_freq = np.zeros(fft_size, dtype=np.complex64)

# Mark active carriers with different values
# Pilots = 2.0, Data = 1.0
dx, dy = 3, 4
sym_idx = 0  # First symbol (dy=0, so pilots active)

carrier_idx = guard_size
active_count = 0

pilot_count_should_be = 0
data_count_should_be = 0

while active_count < 6913:
    if carrier_idx == dc_index:
        carrier_idx += 1
        continue
    if carrier_idx >= fft_size - guard_size:
        break
    
    # Check pilot
    is_pilot = (active_count % dx == 0) and (sym_idx % dy == 0)
    
    if is_pilot:
        test_freq[carrier_idx] = 2.0  # Mark as pilot
        pilot_count_should_be += 1
    else:
        test_freq[carrier_idx] = 1.0  # Mark as data
        data_count_should_be += 1
    
    carrier_idx += 1
    active_count += 1

print(f"Test signal created:")
print(f"  Pilots marked: {pilot_count_should_be}")
print(f"  Data marked: {data_count_should_be}")
print(f"  Total: {pilot_count_should_be + data_count_should_be}")

# Now extract using decoder method
data_symbols, evm = decoder._extract_data_carriers(test_freq, sym_idx)

print(f"\nDecoder extracted:")
print(f"  Data symbols: {len(data_symbols)}")

# Check how many are pilots (value=2.0) vs data (value=1.0)
pilots_extracted_as_data = sum(1 for s in data_symbols if abs(s - 2.0) < 0.01)
real_data_extracted = sum(1 for s in data_symbols if abs(s - 1.0) < 0.01)

print(f"  Pilots mistakenly extracted as data: {pilots_extracted_as_data}")
print(f"  Real data extracted: {real_data_extracted}")

if pilots_extracted_as_data > 0:
    print(f"\n✗ DECODER IS EXTRACTING PILOTS AS DATA!")
else:
    print(f"\n✓ Decoder correctly skipping pilots")

print("=" * 70)