#!/usr/bin/env python3
"""
RTCM to Broadcast Pipeline Handoff Test

Reads RTCM data from a file and feeds it through the ATSC 3.0 broadcast pipeline.
Produces proof artifacts demonstrating successful handoff.

Usage:
  python rtcm_to_broadcast_handoff.py --scenario scenario1
  python rtcm_to_broadcast_handoff.py --rtcm path/to/corrections.rtcm --out-dir OUTPUTS/handoff_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broadcast.pipeline import BroadcastPipeline, BroadcastConfig
from broadcast.config import FECCodeRate, ModulationScheme, FFTSize, GuardInterval


def format_bytes(n: int) -> str:
    """Format byte count for human readability."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.2f} KB"
    else:
        return f"{n / (1024 * 1024):.2f} MB"


def run_handoff_test(
    rtcm_path: Path,
    output_dir: Path,
    scenario_name: str = "unknown"
) -> dict:
    """
    Run the RTCM to broadcast handoff test.
    
    Args:
        rtcm_path: Path to RTCM file
        output_dir: Directory for output artifacts
        scenario_name: Name of scenario for logging
    
    Returns:
        Dictionary with test results
    """
    results = {
        "scenario": scenario_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_file": str(rtcm_path),
        "output_dir": str(output_dir),
        "success": False,
        "stages": {},
        "metrics": {},
        "errors": []
    }
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Read RTCM data
    print(f"\n{'='*70}")
    print(f"RTCM → BROADCAST HANDOFF TEST: {scenario_name}")
    print(f"{'='*70}\n")
    
    print("[Stage 1] Reading RTCM input...")
    try:
        rtcm_data = rtcm_path.read_bytes()
        results["stages"]["read_rtcm"] = {
            "status": "OK",
            "bytes": len(rtcm_data),
            "file": str(rtcm_path)
        }
        print(f"  ✓ Read {format_bytes(len(rtcm_data))} from {rtcm_path.name}")
        
        # Validate RTCM preamble
        if len(rtcm_data) >= 1 and rtcm_data[0] == 0xD3:
            print(f"  ✓ Valid RTCM 3.x preamble detected (0xD3)")
            results["stages"]["read_rtcm"]["valid_preamble"] = True
        else:
            print(f"  ⚠ Warning: RTCM preamble not 0xD3 (got 0x{rtcm_data[0]:02X})")
            results["stages"]["read_rtcm"]["valid_preamble"] = False
            
    except FileNotFoundError:
        error = f"RTCM file not found: {rtcm_path}"
        print(f"  ✗ {error}")
        results["errors"].append(error)
        return results
    except Exception as e:
        error = f"Failed to read RTCM file: {e}"
        print(f"  ✗ {error}")
        results["errors"].append(error)
        return results
    
    # Stage 2: Initialize broadcast pipeline
    print("\n[Stage 2] Initializing broadcast pipeline...")
    try:
        config = BroadcastConfig(
            use_alp=True,
            fec_ldpc_rate=FECCodeRate.RATE_8_15,
            fec_rs_symbols=16,
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
        )
        pipeline = BroadcastPipeline(config)
        results["stages"]["init_pipeline"] = {
            "status": "OK",
            "config": {
                "fec_rate": config.fec_ldpc_rate.name,
                "modulation": config.modulation.name,
                "fft_size": config.fft_size.name,
                "guard_interval": config.guard_interval.name,
                "alp_enabled": config.use_alp
            }
        }
        print(f"  ✓ Pipeline initialized")
        print(f"    FEC: {config.fec_ldpc_rate.name}, Modulation: {config.modulation.name}")
        print(f"    FFT: {config.fft_size.name}, Guard: {config.guard_interval.name}")
    except Exception as e:
        error = f"Failed to initialize pipeline: {e}"
        print(f"  ✗ {error}")
        results["errors"].append(error)
        return results
    
    # Stage 3: Process through pipeline
    print("\n[Stage 3] Processing RTCM through broadcast pipeline...")
    try:
        start_time = time.time()
        result = pipeline.process(rtcm_data)
        processing_time = (time.time() - start_time) * 1000
        
        # Extract key metrics
        ofdm_samples = result.signal.time_domain_signal
        ofdm_bytes = len(ofdm_samples.tobytes())
        
        results["stages"]["process"] = {
            "status": "OK",
            "processing_time_ms": processing_time,
        }
        
        results["metrics"] = {
            "input_bytes": len(rtcm_data),
            "output_samples": len(ofdm_samples),
            "output_bytes": ofdm_bytes,
            "expansion_ratio": result.expansion_ratio,
            "signal_duration_ms": result.signal.duration_ms,
            "sample_rate_hz": result.signal.sample_rate,
            "spectral_efficiency_bps": result.spectral_efficiency,
            "fec_overhead_bytes": result.fec_result.overhead_bytes,
            "processing_time_ms": processing_time
        }
        
        print(f"  ✓ Processing complete in {processing_time:.2f} ms")
        print(f"\n  Pipeline Metrics:")
        print(f"    Input:           {format_bytes(len(rtcm_data))}")
        print(f"    FEC Overhead:    {format_bytes(result.fec_result.overhead_bytes)}")
        print(f"    Output Samples:  {len(ofdm_samples):,}")
        print(f"    Output Size:     {format_bytes(ofdm_bytes)}")
        print(f"    Expansion Ratio: {result.expansion_ratio:.1f}x")
        print(f"    Signal Duration: {result.signal.duration_ms:.2f} ms")
        print(f"    Sample Rate:     {result.signal.sample_rate / 1e6:.2f} MHz")
        
    except Exception as e:
        error = f"Pipeline processing failed: {e}"
        print(f"  ✗ {error}")
        results["errors"].append(error)
        import traceback
        traceback.print_exc()
        return results
    
    # Stage 4: Write output artifacts
    print("\n[Stage 4] Writing output artifacts...")
    try:
        # Save OFDM signal (first 1000 samples as preview)
        ofdm_preview_path = output_dir / "ofdm_preview.bin"
        preview_samples = ofdm_samples[:min(1000, len(ofdm_samples))]
        ofdm_preview_path.write_bytes(preview_samples.tobytes())
        print(f"  ✓ OFDM preview: {ofdm_preview_path.name} ({format_bytes(len(preview_samples.tobytes()))})")
        
        # Save full OFDM signal
        ofdm_full_path = output_dir / "ofdm_signal.bin"
        ofdm_full_path.write_bytes(ofdm_samples.tobytes())
        print(f"  ✓ OFDM signal:  {ofdm_full_path.name} ({format_bytes(ofdm_bytes)})")
        
        # Mark success before writing JSON
        results["success"] = True
        
        # Save test results as JSON
        results_path = output_dir / "handoff_results.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Results JSON: {results_path.name}")
        
        # Save human-readable log
        log_path = output_dir / "handoff_log.txt"
        with log_path.open("w") as f:
            f.write(f"RTCM → Broadcast Handoff Test Log\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Scenario:    {scenario_name}\n")
            f.write(f"Timestamp:   {results['timestamp']}\n")
            f.write(f"Input File:  {rtcm_path}\n")
            f.write(f"Output Dir:  {output_dir}\n\n")
            f.write(f"Input Size:       {format_bytes(len(rtcm_data))}\n")
            f.write(f"Output Samples:   {len(ofdm_samples):,}\n")
            f.write(f"Output Size:      {format_bytes(ofdm_bytes)}\n")
            f.write(f"Expansion Ratio:  {result.expansion_ratio:.1f}x\n")
            f.write(f"Signal Duration:  {result.signal.duration_ms:.2f} ms\n")
            f.write(f"Sample Rate:      {result.signal.sample_rate / 1e6:.2f} MHz\n")
            f.write(f"Processing Time:  {processing_time:.2f} ms\n\n")
            f.write(f"Status: SUCCESS\n")
        print(f"  ✓ Log file:     {log_path.name}")
        
        results["stages"]["write_artifacts"] = {
            "status": "OK",
            "files": [
                str(ofdm_preview_path),
                str(ofdm_full_path),
                str(results_path),
                str(log_path)
            ]
        }
        
    except Exception as e:
        error = f"Failed to write artifacts: {e}"
        print(f"  ✗ {error}")
        results["errors"].append(error)
        return results
    
    # Success!
    results["success"] = True
    print(f"\n{'='*70}")
    print(f"✓ HANDOFF TEST PASSED: {scenario_name}")
    print(f"  Artifacts written to: {output_dir}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="RTCM to Broadcast Pipeline Handoff Test"
    )
    
    # Input options
    parser.add_argument(
        "--scenario", type=str,
        help="Scenario name (looks for OUTPUTS/<scenario>/corrections.rtcm)"
    )
    parser.add_argument(
        "--rtcm", type=str,
        help="Direct path to RTCM file"
    )
    
    # Output options
    parser.add_argument(
        "--out-dir", type=str,
        help="Output directory for artifacts (default: OUTPUTS/<scenario>/handoff/)"
    )
    
    # Paths
    parser.add_argument(
        "--outputs-root", type=str,
        default="../OUTPUTS",
        help="Root OUTPUTS directory"
    )
    
    args = parser.parse_args()
    
    # Determine RTCM path
    outputs_root = Path(args.outputs_root).resolve()
    
    if args.rtcm:
        rtcm_path = Path(args.rtcm).resolve()
        scenario_name = rtcm_path.stem
    elif args.scenario:
        rtcm_path = outputs_root / args.scenario / "corrections.rtcm"
        scenario_name = args.scenario
    else:
        print("Error: Must specify --scenario or --rtcm")
        sys.exit(1)
    
    # Determine output directory
    if args.out_dir:
        output_dir = Path(args.out_dir).resolve()
    else:
        output_dir = outputs_root / scenario_name / "handoff"
    
    # Run test
    results = run_handoff_test(rtcm_path, output_dir, scenario_name)
    
    # Exit code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
