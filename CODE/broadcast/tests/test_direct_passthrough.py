# test_direct_passthrough.py (CORRECTED)
import sys
import numpy as np
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.framebuilder import ATSCFrame, FrameType, FrameConfig
from broadcast.config import ModulationScheme, FFTSize, GuardInterval
from broadcast.utils import get_timestamp_ms

print("=" * 70)
print("DIRECT PASSTHROUGH TEST (No FEC, No Pilots)")
print("=" * 70)

# Simple QPSK modulated symbols
modulator = OFDMModulator()
constellation = modulator.CONSTELLATIONS[ModulationScheme.QPSK]

# Create test symbols (1000 identical QPSK symbols)
num_test_symbols = 1000
test_symbols = np.full(num_test_symbols, constellation[0], dtype=np.complex64)

print(f"\n1. CREATING TEST SIGNAL")
print("-" * 70)
print(f"Test symbols: {num_test_symbols} × {constellation[0]}")

# Create config
config = FrameConfig(
    fft_size=FFTSize.FFT_8K,
    guard_interval=GuardInterval.GI_1_8,
    modulation=ModulationScheme.QPSK,
    frame_duration_ms=50.0,
    pilots_enabled=False  # NO PILOTS for this test
)

# Create minimal frame
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
        'fft_size': config.fft_size.value,
        'guard_interval': config.guard_interval.value,
        'modulation': 'QPSK',
        'frame_duration_ms': config.frame_duration_ms,
        'pilot_pattern': None,
        'pilot_boost_db': 0,
        'time_interleaving_depth': 0,
        'data_symbols': num_test_symbols,
        'pilot_symbols': 0,
        'active_carriers': 6913,
        'original_data_size': 100,
        'fec_overhead_bytes': 0,
        'fec_code_rate': 1.0,
    }
)

# Modulate
signal = modulator.modulate(frame, config)

print(f"TX Signal: {signal.num_samples} samples")
print(f"OFDM symbols: {signal.metadata['num_ofdm_symbols']}")

# Check frequency domain
freq_symbols = signal.frequency_domain_symbols
nonzero = np.count_nonzero(freq_symbols[0])
print(f"Non-zero carriers in first symbol: {nonzero}")

# Decode
print("\n2. DECODING (No FEC)")
print("-" * 70)

decoder_config = DecoderConfig(
    pilots_enabled=False,
    fec_rs_symbols=0,
    time_interleaving=False
)
decoder = BroadcastDecoder(decoder_config)

# Just OFDM demod
payload_bytes, snr, evm = decoder._ofdm_demodulate(signal.time_domain_signal)

print(f"Demodulated: {len(payload_bytes)} bytes")
print(f"EVM: {evm:.2f}%")
print(f"SNR: {snr:.2f} dB")

# Convert back to symbols
bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
rx_symbols = []
for i in range(0, len(bits)-1, 2):
    idx = bits[i] * 2 + bits[i+1]
    if idx < len(constellation):
        rx_symbols.append(constellation[idx])

print(f"Recovered symbols: {len(rx_symbols)}")

# Check first 10 symbols
print(f"\nFirst 10 TX: {test_symbols[:10]}")
print(f"First 10 RX: {np.array(rx_symbols[:10])}")

if evm < 5.0:
    print("\n✓ OFDM chain is WORKING")
else:
    print(f"\n✗ OFDM chain has issues (EVM={evm:.1f}%)")

print("=" * 70)