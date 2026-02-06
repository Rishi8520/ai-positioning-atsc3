# test_ofdm_symbol_placement.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.framebuilder import FrameConfig
from broadcast.config import ModulationScheme, FFTSize, GuardInterval

print("=" * 70)
print("OFDM SYMBOL PLACEMENT TEST")
print("=" * 70)

modulator = OFDMModulator()
qpsk = modulator.CONSTELLATIONS[ModulationScheme.QPSK]

# Create test pattern: [0,1,2,3, 0,1,2,3, ...]
num_symbols = 20
test_symbols = np.array([qpsk[i % 4] for i in range(num_symbols)], dtype=np.complex64)

print(f"\n1. INPUT SYMBOLS")
print("-" * 70)
print(f"Count: {num_symbols}")
for i in range(min(12, num_symbols)):
    idx = i % 4
    print(f"  [{i:2d}] Index {idx} → {test_symbols[i]}")

# Create config (no pilots for simplicity)
config = FrameConfig(
    fft_size=FFTSize.FFT_8K,
    guard_interval=GuardInterval.GI_1_8,
    modulation=ModulationScheme.QPSK,
    frame_duration_ms=50.0,
    pilots_enabled=False
)

# Call _build_ofdm_symbols directly
ofdm_grid = modulator._build_ofdm_symbols(test_symbols, None, config)

print(f"\n2. OFDM GRID")
print("-" * 70)
print(f"Shape: {ofdm_grid.shape}")

# Extract carriers from first OFDM symbol
first_sym = ofdm_grid[0]
guard_size = 639
dc_index = 4096
fft_size = 8192

extracted = []
carrier_idx = guard_size
active_count = 0

while active_count < 6913 and len(extracted) < 20:
    if carrier_idx == dc_index:
        carrier_idx += 1
        continue
    if carrier_idx >= fft_size - guard_size:
        break
    
    val = first_sym[carrier_idx]
    if np.abs(val) > 0.01:  # Non-zero
        extracted.append(val)
    
    carrier_idx += 1
    active_count += 1

print(f"\n3. EXTRACTED FROM FIRST OFDM SYMBOL")
print("-" * 70)
print(f"Count: {len(extracted)}")
for i in range(min(12, len(extracted))):
    print(f"  [{i:2d}] {extracted[i]}")

# Compare
print(f"\n4. COMPARISON")
print("-" * 70)
match_count = 0
for i in range(min(12, len(extracted), num_symbols)):
    tx = test_symbols[i]
    rx = extracted[i]
    match = np.abs(tx - rx) < 0.01
    symbol = "✓" if match else "✗"
    print(f"  [{i:2d}] TX: {tx}  RX: {rx}  {symbol}")
    if match:
        match_count += 1

print(f"\nMatch rate: {match_count}/{min(12, len(extracted), num_symbols)}")

if match_count == min(12, len(extracted), num_symbols):
    print("✓ Symbol placement is CORRECT")
else:
    print("✗ Symbol placement is WRONG")
    print("   Symbols are being scrambled in _build_ofdm_symbols()")

print("=" * 70)