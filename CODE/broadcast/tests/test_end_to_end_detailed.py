# test_end_to_end_detailed.py
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.framebuilder import FrameBuilder, FrameConfig
from broadcast.fecencoder import FECEncoder
from broadcast.config import ModulationScheme, FFTSize, GuardInterval

print("=" * 70)
print("DETAILED END-TO-END TEST")
print("=" * 70)

test_data = b"HELLO" * 20  # 100 bytes

print(f"\n1. ENCODING")
print("-" * 70)
print(f"Input: {len(test_data)} bytes")
print(f"Content: {test_data.hex()}")

# Encode
fec_encoder = FECEncoder()
fec_result = fec_encoder.encode(test_data)

frame_config = FrameConfig(
    fft_size=FFTSize.FFT_8K,
    guard_interval=GuardInterval.GI_1_8,
    modulation=ModulationScheme.QPSK,
    frame_duration_ms=50.0,
    pilots_enabled=False,
    time_interleaving=False
)

frame_builder = FrameBuilder(default_config=frame_config)
frame = frame_builder.build_frame(fec_result, frame_config)

modulator = OFDMModulator()
signal = modulator.modulate(frame, frame_config)

print(f"After encoding: {signal.num_samples} samples")

# Decode
decoder_config = DecoderConfig(
    pilots_enabled=False,
    time_interleaving=False,
    fec_rs_symbols=16
)

decoder = BroadcastDecoder(decoder_config)
result = decoder.decode(signal.time_domain_signal)

print(f"\n2. DECODING")
print("-" * 70)
print(f"Recovered: {len(result.data)} bytes")
print(f"EVM: {result.metrics.evm_percent:.2f}%")
print(f"SNR: {result.metrics.estimated_snr_db:.2f} dB")

# Check ALL bytes
print(f"\n3. BYTE-BY-BYTE COMPARISON")
print("-" * 70)

match_count = 0
for i in range(100):
    tx = test_data[i]
    rx = result.data[i] if i < len(result.data) else 0x00
    match = tx == rx
    if match:
        match_count += 1
    if i < 20 or not match:
        status = "✓" if match else "✗"
        print(f"  [{i:3d}] TX: 0x{tx:02x}  RX: 0x{rx:02x}  {status}")

print(f"\nMatch rate: {match_count}/100 = {match_count}%")

# Check extra bytes
if len(result.data) > 100:
    print(f"\n4. EXTRA BYTES (101-{len(result.data)})")
    print("-" * 70)
    extra = result.data[100:]
    print(f"Content: {extra.hex()}")
    if extra == b'\x00' * len(extra):
        print("✓ All zeros (padding)")
    else:
        print("✗ Contains non-zero data")

print("=" * 70)