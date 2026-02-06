# test_symbol_mapping.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.framebuilder import FrameBuilder
from broadcast.fecencoder import FECEncoder
from broadcast.config import FFTSize, PilotPattern

print("=" * 70)
print("SYMBOL MAPPING TEST")
print("=" * 70)

# Simple test: just 100 bytes
test_data = b"A" * 100

# Encode
print("\n1. ENCODING")
print("-" * 70)
fec_encoder = FECEncoder()
frame_builder = FrameBuilder()
modulator = OFDMModulator()

fec_result = fec_encoder.encode(test_data)
print(f"After FEC: {len(fec_result.encoded_data)} bytes")

frame = frame_builder.build_frame(fec_result)
print(f"After framing: {len(frame.payload)} bytes")
print(f"Data symbols in payload: {frame.metadata['data_symbols']}")

# Check OFDM modulation
signal = modulator.modulate(frame)
freq_symbols = signal.frequency_domain_symbols

print(f"\nOFDM frequency domain shape: {freq_symbols.shape}")
print(f"  {freq_symbols.shape[0]} symbols × {freq_symbols.shape[1]} carriers")

# Count non-zero carriers in first symbol
first_sym = freq_symbols[0]
nonzero = np.count_nonzero(first_sym)
print(f"\nNon-zero carriers in first symbol: {nonzero}")

# Check carrier positions
fft_size = FFTSize.FFT_8K.value
guard_size = 639
dc_index = fft_size // 2

active_start = guard_size
active_end = fft_size - guard_size

print(f"\nCarrier allocation:")
print(f"  Guard band: [0:{guard_size}]")
print(f"  Active: [{active_start}:{active_end}]")
print(f"  DC index: {dc_index}")

# Check specific positions
dx, dy = 3, 4  # SP3_4 pattern
pilot_positions = []
data_positions = []

for i in range(active_start, active_end):
    if i == dc_index:
        continue
    if (i - guard_size) % dx == 0:
        pilot_positions.append(i)
    else:
        data_positions.append(i)

print(f"\nExpected pilots per symbol: {len(pilot_positions)}")
print(f"Expected data carriers per symbol: {len(data_positions)}")

# Check what's actually in the OFDM symbol
pilot_count = 0
data_count = 0
for i in range(fft_size):
    if first_sym[i] != 0:
        if i in pilot_positions:
            pilot_count += 1
        elif i in data_positions:
            data_count += 1

print(f"\nActual in first OFDM symbol:")
print(f"  Pilots placed: {pilot_count}")
print(f"  Data placed: {data_count}")

# Decode
print("\n2. DECODING")
print("-" * 70)
decoder = BroadcastDecoder(DecoderConfig())
result = decoder.decode(signal.time_domain_signal)

print(f"Recovered: {len(result.data)} bytes")
print(f"EVM: {result.metrics.evm_percent:.2f}%")
print(f"Symbols processed: {result.metrics.symbols_processed}")

expected_data_symbols = len(data_positions) * freq_symbols.shape[0]
print(f"\nExpected data symbols: {expected_data_symbols}")
print(f"Actually processed: {result.metrics.symbols_processed}")

if result.metrics.symbols_processed == expected_data_symbols:
    print("✓ Symbol count MATCHES")
else:
    print(f"✗ MISMATCH: {expected_data_symbols} vs {result.metrics.symbols_processed}")

print("\n" + "=" * 70)