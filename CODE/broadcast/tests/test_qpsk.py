#!/usr/bin/env python3
"""Test QPSK bit round-trip to find bit ordering issue."""

import numpy as np
import sys
sys.path.insert(0, '/media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE')

from broadcast.ofdmmodulator import OFDMModulator
from broadcast.decoder import BroadcastDecoder
from broadcast.config import ModulationScheme

def test_qpsk_roundtrip():
    """Test if QPSK modulation + LLR demapping preserves bit order."""
    
    # Test pattern: known bit sequence
    test_bits = np.array([0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0], dtype=np.uint8)
    test_bytes = np.packbits(test_bits).tobytes()
    
    print(f"Original bits: {test_bits}")
    print(f"Original bytes (hex): {test_bytes.hex()}")
    
    # TX: Modulate
    modulator = OFDMModulator()
    symbols = modulator._map_bits_to_symbols(test_bytes, ModulationScheme.QPSK)
    print(f"\nModulated to {len(symbols)} QPSK symbols")
    print(f"First 4 symbols: {symbols[:4]}")
    
    # RX: Compute LLRs
    decoder = BroadcastDecoder()
    llrs = decoder.qpsk_llrs(symbols, noise_var=0.1)
    
    # Hard decision
    recovered_bits = (llrs < 0).astype(np.uint8)
    
    print(f"\nRecovered bits: {recovered_bits[:16]}")
    print(f"\nBit-by-bit comparison:")
    print("Pos | Sent | Recv | Match")
    print("----|------|------|------")
    
    errors = 0
    for i in range(16):
        match = "✓" if test_bits[i] == recovered_bits[i] else "✗"
        if test_bits[i] != recovered_bits[i]:
            errors += 1
        print(f" {i:2d} |  {test_bits[i]}   |  {recovered_bits[i]}   |  {match}")
    
    print(f"\nTotal errors: {errors}/16")
    
    if errors == 0:
        print("✅ QPSK round-trip is BIT-PERFECT!")
    else:
        print(f"❌ {errors} bit errors detected - bit ordering mismatch!")
        
        # Analyze pattern
        if errors == 8:
            print("\n🔍 Exactly 50% errors suggests:")
            print("   - Possible I/Q swap")
            print("   - OR bit reversal within symbols")

if __name__ == "__main__":
    test_qpsk_roundtrip()