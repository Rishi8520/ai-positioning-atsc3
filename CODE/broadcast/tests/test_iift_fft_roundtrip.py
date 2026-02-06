# test_ifft_fft_roundtrip.py (FIXED)
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.config import FFTSize, GuardInterval

print("=" * 70)
print("IFFT/FFT ROUNDTRIP TEST")
print("=" * 70)

modulator = OFDMModulator()

# Create known frequency-domain symbol
fft_size = FFTSize.FFT_8K
num_carriers = fft_size.value
guard_size = 639
dc_index = num_carriers // 2

freq_domain = np.zeros((1, num_carriers), dtype=np.complex64)

# Place 12 known symbols in active carriers
test_pattern = [-0.707-0.707j, -0.707+0.707j, 0.707-0.707j, 0.707+0.707j] * 3
carrier_idx = guard_size
for i, sym in enumerate(test_pattern):
    if carrier_idx == dc_index:
        carrier_idx += 1
    freq_domain[0, carrier_idx] = sym
    carrier_idx += 1

print(f"\n1. FREQUENCY DOMAIN INPUT")
print("-" * 70)
print(f"First 12 symbols: {test_pattern}")

# Apply IFFT
time_domain = modulator._apply_ifft(freq_domain, fft_size)
print(f"\n2. AFTER IFFT")
print(f"   Shape: {time_domain.shape}")

# Add guard interval
gi = GuardInterval.GI_1_8
with_gi = modulator._add_guard_interval(time_domain, gi, fft_size)
print(f"   With GI shape: {with_gi.shape}")

# Flatten
signal = with_gi.flatten()
print(f"   Signal length: {len(signal)}")

# Now decode (reverse process)
decoder = BroadcastDecoder(DecoderConfig())

# Extract one symbol
cp_len = decoder.cp_length
symbol_with_cp = signal[:decoder.symbol_length]
symbol = symbol_with_cp[cp_len:]  # Remove CP

# FFT
freq_recovered = np.fft.fft(symbol)
freq_recovered = np.fft.fftshift(freq_recovered)

print(f"\n3. AFTER FFT (RECOVERED)")
print("-" * 70)

# Extract the same carriers
recovered = []
carrier_idx = guard_size
for i in range(12):
    if carrier_idx == dc_index:
        carrier_idx += 1
    recovered.append(freq_recovered[carrier_idx])
    carrier_idx += 1

print(f"First 12 recovered: {recovered[:4]}")

# Compare
print(f"\n4. COMPARISON")
print("-" * 70)
match_count = 0
for i in range(12):
    tx = test_pattern[i]
    rx = recovered[i]
    error = np.abs(tx - rx)
    match = error < 0.01
    symbol = "✓" if match else "✗"
    print(f"  [{i:2d}] TX: {tx}  RX: {rx}  Error: {error:.6f}  {symbol}")
    if match:
        match_count += 1

print(f"\nMatch rate: {match_count}/12")

if match_count == 12:
    print("✓ IFFT/FFT chain is CORRECT")
else:
    print("✗ IFFT/FFT chain has issues")

print("=" * 70)