# test_carrier_mismatch.py
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.config import FFTSize, PilotPattern

# ATSC 3.0 spec values
CARRIER_ALLOCATION = {
    FFTSize.FFT_8K: {
        'total': 8192,
        'active': 6913,
        'guard_lower': 639,
        'guard_upper': 639,
        'dc': 1,
    }
}

fft_size = FFTSize.FFT_8K
pilot_pattern = PilotPattern.SP3_4  # (3, 4)

allocation = CARRIER_ALLOCATION[fft_size]
active_carriers = allocation['active']
dx, dy = pilot_pattern.value

print("=" * 60)
print("Carrier Allocation Analysis")
print("=" * 60)
print(f"\nFFT Size: {fft_size.value}")
print(f"Active carriers (ATSC spec): {active_carriers}")
print(f"Pilot pattern: SP{dx}_{dy} (every {dx} carriers, every {dy} symbols)")

pilots_per_symbol = active_carriers // dx
data_per_symbol = active_carriers - pilots_per_symbol

print(f"\nPer OFDM symbol:")
print(f"  Pilots: {pilots_per_symbol}")
print(f"  Data: {data_per_symbol}")
print(f"  Total: {pilots_per_symbol + data_per_symbol}")

# For 6 OFDM symbols (from your test: 55296 / 9216 = 6 symbols)
num_ofdm_symbols = 6
total_pilots = pilots_per_symbol * num_ofdm_symbols
total_data = data_per_symbol * num_ofdm_symbols

print(f"\nFor {num_ofdm_symbols} OFDM symbols:")
print(f"  Total pilots: {total_pilots}")
print(f"  Total data carriers: {total_data}")

print("=" * 60)