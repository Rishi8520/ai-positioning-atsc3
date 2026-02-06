# test_symbol_ordering.py (FIXED)
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.framebuilder import ATSCFrame, FrameType, FrameConfig
from broadcast.config import ModulationScheme, FFTSize, GuardInterval
from broadcast.utils import get_timestamp_ms

print("=" * 70)
print("SYMBOL ORDERING TEST")
print("=" * 70)

# Create unique symbols (incrementing pattern)
modulator = OFDMModulator()
qpsk = modulator.CONSTELLATIONS[ModulationScheme.QPSK]

# Use different symbol for each position: pattern [0,1,2,3,0,1,2,3,...]
num_symbols = 100
test_symbols = np.array([qpsk[i % 4] for i in range(num_symbols)], dtype=np.complex64)

print(f"\n1. TEST PATTERN")
print("-" * 70)
print(f"Symbols: {num_symbols}")
print(f"Pattern: [0,1,2,3,0,1,2,3...] repeating")
print(f"First 12 symbols: {test_symbols[:12]}")

# Build frame (no pilots)
config = FrameConfig(
    fft_size=FFTSize.FFT_8K,
    guard_interval=GuardInterval.GI_1_8,
    modulation=ModulationScheme.QPSK,
    frame_duration_ms=50.0,
    pilots_enabled=False
)

frame = ATSCFrame(
    frame_id=0,
    frame_type=FrameType.SUBFRAME,
    preamble=b'',
    payload=test_symbols.tobytes(),
    pilots=None,
    total_symbols=1,
    modulation=ModulationScheme.QPSK,
    timestamp=get_timestamp_ms(),
    metadata={
        'fft_size': 8192,
        'guard_interval': 1024,
        'modulation': 'QPSK',
        'frame_duration_ms': 50.0,
        'pilot_pattern': None,
        'pilot_boost_db': 0,
        'time_interleaving_depth': 0,
        'data_symbols': num_symbols,
        'pilot_symbols': 0,
        'active_carriers': 6913,
        'original_data_size': 100,
        'fec_overhead_bytes': 0,
        'fec_code_rate': 1.0,
    }
)

# Modulate
signal = modulator.modulate(frame, config)
freq_domain = signal.frequency_domain_symbols

print(f"\n2. CHECK OFDM GRID")
print("-" * 70)

# Extract carriers from first OFDM symbol manually
first_symbol = freq_domain[0]
guard_size = 639
dc_index = 4096
fft_size = 8192

extracted = []
carrier_idx = guard_size
active_count = 0

while active_count < 6913:
    if carrier_idx == dc_index:
        carrier_idx += 1
        continue
    if carrier_idx >= fft_size - guard_size:
        break
    
    if first_symbol[carrier_idx] != 0:
        extracted.append(first_symbol[carrier_idx])
    
    carrier_idx += 1
    active_count += 1

print(f"Extracted from first OFDM symbol: {len(extracted)} symbols")
print(f"First 12 extracted: {np.array(extracted[:12])}")

# Compare with original
print(f"\n3. COMPARISON")
print("-" * 70)
match_count = 0
for i in range(min(12, len(extracted))):
    tx = test_symbols[i]
    rx = extracted[i] if i < len(extracted) else 0
    match = "✓" if np.abs(tx - rx) < 0.01 else "✗"
    print(f"  [{i:2d}] TX: {tx}  RX: {rx}  {match}")
    if np.abs(tx - rx) < 0.01:
        match_count += 1

if match_count == 12:
    print("\n✓ ORDERING IS CORRECT")
else:
    print(f"\n✗ ORDERING MISMATCH: {match_count}/12 correct")
    print("\nThis means symbols are being SCRAMBLED during modulation!")
    print("The _map_bits_to_symbols() output doesn't match what's in the OFDM grid.")

print("=" * 70)