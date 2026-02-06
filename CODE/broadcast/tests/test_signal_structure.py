# test_signal_structure.py
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.framebuilder import FrameBuilder
from broadcast.fecencoder import FECEncoder
from broadcast.config import FFTSize, GuardInterval

print("=" * 70)
print("SIGNAL STRUCTURE TEST")
print("=" * 70)

# Test data
test_data = b"HELLO WORLD! " * 200

# Encode
fec_encoder = FECEncoder()
frame_builder = FrameBuilder()
modulator = OFDMModulator()

fec_result = fec_encoder.encode(test_data)
frame = frame_builder.build_frame(fec_result)
signal = modulator.modulate(frame)

print("\n1. ENCODER OUTPUT")
print("-" * 70)
print(f"Total samples: {signal.num_samples}")
print(f"OFDM symbols: {signal.metadata['num_ofdm_symbols']}")
print(f"FFT size: {signal.metadata['fft_size']}")

# Calculate expected structure
fft_size = FFTSize.FFT_8K.value  # 8192
gi_fraction = 1/8  # GI_1_8
cp_length = int(fft_size * gi_fraction)  # 1024
symbol_length = fft_size + cp_length  # 9216

print(f"\nCalculated per symbol:")
print(f"  FFT size: {fft_size}")
print(f"  CP length: {cp_length}")
print(f"  Symbol length: {symbol_length}")

num_symbols_calc = signal.num_samples // symbol_length
print(f"\nOFDM symbols (from signal length): {num_symbols_calc}")
print(f"OFDM symbols (from metadata): {signal.metadata['num_ofdm_symbols']}")

if num_symbols_calc == signal.metadata['num_ofdm_symbols']:
    print("✓ Symbol count MATCHES")
else:
    print(f"✗ MISMATCH: {num_symbols_calc} vs {signal.metadata['num_ofdm_symbols']}")

print("\n2. DECODER CALCULATION")
print("-" * 70)
decoder_config = DecoderConfig()
print(f"Decoder FFT size: {decoder_config.fft_size.value}")
print(f"Decoder GI: {decoder_config.guard_interval.name}")

decoder_cp = int(decoder_config.fft_size.value * (1/8))
decoder_sym_len = decoder_config.fft_size.value + decoder_cp

print(f"Decoder symbol length: {decoder_sym_len}")
decoder_num_syms = signal.num_samples // decoder_sym_len
print(f"Decoder will decode: {decoder_num_syms} symbols")

if decoder_sym_len == symbol_length:
    print("✓ Symbol lengths MATCH")
else:
    print(f"✗ MISMATCH: {symbol_length} (encoder) vs {decoder_sym_len} (decoder)")

print("\n3. DETAILED BREAKDOWN")
print("-" * 70)
print(f"Frame payload: {len(frame.payload)} bytes")
print(f"Data symbols (metadata): {frame.metadata['data_symbols']}")
print(f"Bits per carrier (QPSK): 2")
print(f"Data carriers per symbol: {frame.metadata['active_carriers']}")

bits_per_ofdm_sym = frame.metadata['active_carriers'] * 2
bytes_per_ofdm_sym = bits_per_ofdm_sym // 8

print(f"\nData capacity per OFDM symbol:")
print(f"  {bits_per_ofdm_sym} bits = {bytes_per_ofdm_sym} bytes")

symbols_needed = (len(frame.payload) * 8 + bits_per_ofdm_sym - 1) // bits_per_ofdm_sym
print(f"\nSymbols needed for payload: {symbols_needed}")
print(f"Actual OFDM symbols generated: {signal.metadata['num_ofdm_symbols']}")

print("\n" + "=" * 70)