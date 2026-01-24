"""
ATSC 3.0 Broadcast Pipeline Demo
Complete end-to-end demonstration of RTK correction broadcast.
"""

import time
from broadcast.pipeline import BroadcastPipeline, BroadcastConfig, broadcast_data
from broadcast.config import ModulationScheme, FFTSize, FECCodeRate
import numpy as np


def print_header(title):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_basic_pipeline():
    """Demo 1: Basic pipeline with default settings."""
    print_header("Demo 1: Basic Pipeline (Default QPSK, 8K FFT)")
    
    # Create pipeline
    pipeline = BroadcastPipeline()
    
    # Simulate RTCM correction message
    rtcm_data = b"\xd3\x00\x13\x3e\xd0\x00\x03" + b"\x12\x34\x56\x78" * 10
    
    print(f"Input: RTCM message ({len(rtcm_data)} bytes)")
    
    # Process
    result = pipeline.process(rtcm_data)
    
    # Display results
    print(f"\nResults:")
    print(f"  ALP Packet:       {len(result.alp_packet.to_bytes())} bytes" if result.alp_packet else "  ALP: Disabled")
    print(f"  FEC Encoded:      {len(result.fec_result.encoded_data)} bytes")
    print(f"  FEC Code Rate:    {result.fec_result.code_rate:.3f}")
    print(f"  Frame Size:       {result.frame.total_bytes} bytes")
    print(f"  OFDM Samples:     {result.signal.num_samples:,} samples")
    print(f"  Signal Duration:  {result.signal.duration_ms:.2f} ms")
    print(f"  Processing Time:  {result.processing_time_ms:.2f} ms")
    print(f"  Expansion Ratio:  {result.expansion_ratio:.1f}x")
    print(f"  Spectral Eff:     {result.spectral_efficiency:.4f} bits/sample")
    
    # Power analysis
    if result.signal.metadata['average_power'] > 0:
        papr_db = 10 * np.log10(
            result.signal.metadata['peak_power'] / 
            result.signal.metadata['average_power']
        )
        print(f"  PAPR:             {papr_db:.1f} dB")


def demo_adaptive_modulation():
    """Demo 2: Different modulation schemes."""
    print_header("Demo 2: Adaptive Modulation (QPSK vs QAM64)")
    
    test_data = b"RTK Correction Data - 100 bytes" * 3
    
    modulations = [
        (ModulationScheme.QPSK, "QPSK (Robust)"),
        (ModulationScheme.QAM64, "64-QAM (High Capacity)")
    ]
    
    for mod_scheme, label in modulations:
        config = BroadcastConfig(modulation=mod_scheme)
        pipeline = BroadcastPipeline(config)
        result = pipeline.process(test_data)
        
        print(f"\n{label}:")
        print(f"  Spectral Efficiency: {result.spectral_efficiency:.4f} bits/sample")
        print(f"  Signal Duration:     {result.signal.duration_ms:.2f} ms")
        print(f"  OFDM Symbols:        {result.signal.num_symbols}")


def demo_adaptive_fec():
    """Demo 3: AI-driven adaptive FEC."""
    print_header("Demo 3: Adaptive FEC (AI-Controlled Overhead)")
    
    test_data = b"GPS Correction Frame with Adaptive FEC"
    pipeline = BroadcastPipeline()
    
    # Simulate different channel conditions
    scenarios = [
        (10.0, "Excellent Channel (Urban, LOS)"),
        (20.0, "Good Channel (Suburban)"),
        (35.0, "Poor Channel (Deep Indoor, NLOS)")
    ]
    
    print(f"Input: {len(test_data)} bytes\n")
    
    for overhead_pct, scenario in scenarios:
        result = pipeline.process_with_adaptive_fec(test_data, overhead_pct)
        
        print(f"{scenario}:")
        print(f"  FEC Overhead:     {overhead_pct}%")
        print(f"  Code Rate:        {result.fec_result.code_rate:.3f}")
        print(f"  Overhead Bytes:   {result.fec_result.overhead_bytes}")
        print(f"  Total Output:     {result.signal.num_samples:,} samples")
        print()


def demo_batch_processing():
    """Demo 4: Continuous stream processing."""
    print_header("Demo 4: Continuous RTCM Stream (20 packets)")
    
    pipeline = BroadcastPipeline()
    
    # Simulate 1 second of RTCM updates (20 packets @ 50ms each)
    packets = [
        f"RTCM Update #{i:03d} - GPS L1/L2 Corrections".encode()
        for i in range(20)
    ]
    
    print(f"Processing {len(packets)} RTCM packets...")
    start_time = time.time()
    
    results = pipeline.process_batch(packets)
    
    elapsed = time.time() - start_time
    
    # Statistics
    total_input = sum(len(r.original_data) for r in results)
    total_samples = sum(r.signal.num_samples for r in results)
    total_duration_ms = sum(r.signal.duration_ms for r in results)
    avg_processing_time = sum(r.processing_time_ms for r in results) / len(results)
    
    print(f"\nBatch Results:")
    print(f"  Packets Processed:     {len(results)}")
    print(f"  Total Input:           {total_input:,} bytes")
    print(f"  Total Samples:         {total_samples:,}")
    print(f"  Total Air Time:        {total_duration_ms:.2f} ms")
    print(f"  Avg Processing Time:   {avg_processing_time:.2f} ms/packet")
    print(f"  Wall Clock Time:       {elapsed:.2f} seconds")
    print(f"  Throughput:            {len(results)/elapsed:.1f} packets/sec")


def demo_iq_signal_export():
    """Demo 5: I/Q signal ready for SDR transmission."""
    print_header("Demo 5: I/Q Signal Export (Ready for SDR)")
    
    result = broadcast_data(b"RTK Base Station Corrections - Ready for TX")
    
    # Get I/Q components
    i_data, q_data = result.signal.get_iq_data()
    
    print(f"Signal Properties:")
    print(f"  Sample Rate:      {result.signal.sample_rate/1e6:.3f} MHz")
    print(f"  I/Q Samples:      {len(i_data):,}")
    print(f"  Duration:         {result.signal.duration_ms:.2f} ms")
    print(f"  I Range:          [{i_data.min():.4f}, {i_data.max():.4f}]")
    print(f"  Q Range:          [{q_data.min():.4f}, {q_data.max():.4f}]")
    print(f"\nReady for transmission via:")
    print(f"  • USRP/Ettus SDR")
    print(f"  • HackRF/bladeRF")
    print(f"  • LimeSDR")
    print(f"  • GNU Radio integration")


def demo_performance_metrics():
    """Demo 6: Comprehensive performance metrics."""
    print_header("Demo 6: Performance Metrics & Statistics")
    
    config = BroadcastConfig(
        modulation=ModulationScheme.QPSK,
        fft_size=FFTSize.FFT_8K,
        fec_ldpc_rate=FECCodeRate.RATE_8_15
    )
    
    pipeline = BroadcastPipeline(config)
    
    # Process multiple packets
    for i in range(10):
        data = f"Performance test packet {i}".encode()
        pipeline.process(data)
    
    # Get comprehensive stats
    stats = pipeline.get_stats()
    
    print(f"Pipeline Statistics:")
    print(f"  Packets Processed:     {stats['packets_processed']}")
    print(f"  Total Input:           {stats['total_input_bytes']:,} bytes")
    print(f"  Total Output:          {stats['total_output_bytes']:,} bytes")
    print(f"  Avg Expansion:         {stats['average_compression_ratio']:.1f}x")
    
    print(f"\nComponent Statistics:")
    
    # FEC Encoder stats
    if 'fec_encoder_stats' in stats:
        print(f"  FEC Encoder:")
        fec_stats = stats['fec_encoder_stats']
        for key, value in fec_stats.items():
            print(f"    {key}: {value}")
    
    # Frame Builder stats
    if 'frame_builder_stats' in stats:
        print(f"  Frame Builder:")
        frame_stats = stats['frame_builder_stats']
        for key, value in frame_stats.items():
            print(f"    {key}: {value}")
    
    # OFDM Modulator stats
    if 'modulator_stats' in stats:
        print(f"  OFDM Modulator:")
        mod_stats = stats['modulator_stats']
        for key, value in mod_stats.items():
            if key == 'sample_rate_hz':
                print(f"    {key}: {value/1e6:.3f} MHz")
            else:
                print(f"    {key}: {value}")
    
    # ALP Encoder stats
    if 'alp_encoder_stats' in stats:
        print(f"  ALP Encoder:")
        alp_stats = stats['alp_encoder_stats']
        for key, value in alp_stats.items():
            print(f"    {key}: {value}")

def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  ATSC 3.0 Broadcast Pipeline - Complete Demonstration")
    print("  RTK Correction Broadcast System")
    print("="*70)
    
    try:
        demo_basic_pipeline()
        demo_adaptive_modulation()
        demo_adaptive_fec()
        demo_batch_processing()
        demo_iq_signal_export()
        demo_performance_metrics()
        
        print_header("All Demos Complete! ✅")
        print("Your ATSC 3.0 broadcast pipeline is ready for production!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()