#!/usr/bin/env python3
"""
RTCM Generator for GNSS Corrections

Generates RTCM 3.x format messages from base station observations.

Usage:
  python rtcm_generator.py --scenario scenario1 --out corrections.rtcm
  python rtcm_generator.py --base path/to/base.obs --nav path/to/nav.nav --out corrections.rtcm

Output Modes:
  1. If RTKLIB convbin is available: Convert RINEX to RTCM via convbin
  2. Otherwise: Generate deterministic RTCM-like payload for pipeline testing

RTCM 3.x Message Types Generated:
  - 1005: Stationary RTK Reference Station ARP
  - 1077: GPS MSM7 (Full GPS Observations)
  - 1087: GLONASS MSM7 (if GLONASS enabled)

The output can be consumed by:
  - gnss_baseline_runner.py (future --corrections support)
  - broadcast/pipeline.py via BroadcastPipeline.process(rtcm_bytes)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Constants
# ----------------------------

DEFAULT_SCENARIO_ROOT = Path("DATA/scenarios")

# RTCM 3.x constants
RTCM_PREAMBLE = 0xD3
RTCM_HEADER_LEN = 3  # preamble + 2 bytes (reserved + length)
RTCM_CRC_LEN = 3

# Common RTCM message types for RTK
MSG_1005_STATION_ARP = 1005      # Reference station ARP (antenna reference point)
MSG_1077_GPS_MSM7 = 1077         # GPS MSM7 (full observables)
MSG_1087_GLO_MSM7 = 1087         # GLONASS MSM7
MSG_1097_GAL_MSM7 = 1097         # Galileo MSM7
MSG_1127_BDS_MSM7 = 1127         # BeiDou MSM7


# ----------------------------
# CRC-24Q Implementation
# ----------------------------

CRC24Q_POLY = 0x1864CFB

def crc24q(data: bytes) -> int:
    """
    Compute CRC-24Q as used in RTCM 3.x.
    Polynomial: 0x1864CFB
    """
    crc = 0
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= CRC24Q_POLY
    return crc & 0xFFFFFF


def build_rtcm_frame(message_type: int, payload: bytes) -> bytes:
    """
    Build a complete RTCM 3.x frame.
    
    Frame structure:
      [Preamble 1B] [Reserved 6b + Length 10b = 2B] [Payload] [CRC-24Q 3B]
    
    Args:
        message_type: RTCM message number (embedded in payload's first 12 bits)
        payload: Variable-length payload (including message type bits)
    
    Returns:
        Complete RTCM frame with preamble, length, payload, and CRC
    """
    length = len(payload)
    if length > 1023:
        raise ValueError(f"RTCM payload too large: {length} bytes (max 1023)")
    
    # Header: preamble + reserved (6 bits = 0) + length (10 bits)
    header = bytes([
        RTCM_PREAMBLE,
        (length >> 8) & 0x03,  # High 2 bits of length (reserved bits = 0)
        length & 0xFF          # Low 8 bits of length
    ])
    
    # CRC over header + payload
    frame_without_crc = header + payload
    crc = crc24q(frame_without_crc)
    crc_bytes = bytes([
        (crc >> 16) & 0xFF,
        (crc >> 8) & 0xFF,
        crc & 0xFF
    ])
    
    return frame_without_crc + crc_bytes


# ----------------------------
# RTCM Message Generators
# ----------------------------

@dataclass
class StationCoordinates:
    """WGS84 ECEF coordinates for base station."""
    x: float  # meters
    y: float  # meters
    z: float  # meters
    antenna_height: float = 0.0


def generate_msg_1005(station: StationCoordinates, station_id: int = 0) -> bytes:
    """
    Generate RTCM Message 1005: Stationary RTK Reference Station ARP.
    
    This message provides the antenna reference point (ARP) coordinates.
    """
    # Message 1005 structure (simplified):
    # - Message Number (12 bits): 1005
    # - Reference Station ID (12 bits)
    # - ITRF Realization Year (6 bits)
    # - GPS Indicator (1 bit)
    # - GLONASS Indicator (1 bit)
    # - Galileo Indicator (1 bit)
    # - Reference Station Indicator (1 bit)
    # - Antenna Reference Point ECEF-X (38 bits, 0.0001m resolution)
    # - Single Receiver Oscillator Indicator (1 bit)
    # - Reserved (1 bit)
    # - Antenna Reference Point ECEF-Y (38 bits)
    # - Quarter Cycle Indicator (2 bits)
    # - Antenna Reference Point ECEF-Z (38 bits)
    
    # For simplicity, we build a deterministic payload
    # Real implementation would pack bits precisely per spec
    
    # Convert coordinates to 0.0001m resolution integers
    x_int = int(station.x * 10000)
    y_int = int(station.y * 10000)
    z_int = int(station.z * 10000)
    
    # Build payload (simplified encoding for PoC)
    # In production, use proper bit-level packing
    payload = bytearray()
    
    # Message type (12 bits) + station ID (12 bits) = 3 bytes
    msg_station = (MSG_1005_STATION_ARP << 12) | (station_id & 0xFFF)
    payload.extend([
        (msg_station >> 16) & 0xFF,
        (msg_station >> 8) & 0xFF,
        msg_station & 0xFF,
    ])
    
    # Flags byte (GPS=1, GLONASS=0, Galileo=0, etc.)
    payload.append(0b10000000)  # GPS indicator set
    
    # ECEF coordinates (simplified: 5 bytes each for demonstration)
    # Real encoding uses 38 bits with sign extension
    for coord in [x_int, y_int, z_int]:
        # Pack as signed 40-bit (5 bytes) for simplicity
        coord_bytes = coord.to_bytes(5, byteorder='big', signed=True)
        payload.extend(coord_bytes)
    
    return build_rtcm_frame(MSG_1005_STATION_ARP, bytes(payload))


def generate_msg_1077_stub(station_id: int = 0, gps_time_ms: int = 0, num_sats: int = 8) -> bytes:
    """
    Generate a stub RTCM Message 1077: GPS MSM7.
    
    This is a simplified/deterministic version for pipeline testing.
    Real MSM7 encoding requires actual observation data.
    """
    payload = bytearray()
    
    # Message header (12 bits msg type + 12 bits station ID)
    msg_station = (MSG_1077_GPS_MSM7 << 12) | (station_id & 0xFFF)
    payload.extend([
        (msg_station >> 16) & 0xFF,
        (msg_station >> 8) & 0xFF,
        msg_station & 0xFF,
    ])
    
    # GPS Epoch Time (30 bits, ms of week)
    epoch = gps_time_ms & 0x3FFFFFFF
    payload.extend([
        (epoch >> 24) & 0xFF,
        (epoch >> 16) & 0xFF,
        (epoch >> 8) & 0xFF,
        epoch & 0xFF,
    ])
    
    # Multiple Message Bit (1 bit) + IODS (3 bits) + Reserved (7 bits) + Clock Steering (2 bits)
    # + External Clock Indicator (2 bits) + GNSS Divergence-free Smoothing Indicator (1 bit)
    # + GNSS Smoothing Interval (3 bits) = 19 bits ~= 3 bytes
    payload.extend([0x00, 0x00, 0x00])
    
    # Satellite mask (64 bits for GPS PRN 1-64)
    # Set bits for num_sats satellites (PRN 1 to num_sats)
    sat_mask = (1 << num_sats) - 1
    payload.extend(sat_mask.to_bytes(8, byteorder='big'))
    
    # Signal mask (32 bits) - indicate L1 C/A for simplicity
    sig_mask = 0x80000000  # L1 C/A
    payload.extend(sig_mask.to_bytes(4, byteorder='big'))
    
    # Cell mask (num_sats * num_signals bits) - all cells present
    # Simplified: 1 byte per satellite
    for _ in range(num_sats):
        payload.append(0x80)  # L1 C/A present
    
    # Satellite data (rough range, etc.) - stub values
    for i in range(num_sats):
        # Rough range (8 bits) + extended info (4 bits) = 12 bits
        payload.extend([0x80 + i, 0x00])
    
    # Signal data (pseudorange, phase, etc.) - stub values
    for i in range(num_sats):
        # Fine pseudorange (15 bits) + fine phase (22 bits) + etc.
        # Simplified: 6 bytes per signal
        payload.extend([0x00, 0x00, 0x00, 0x00, 0x00, i])
    
    return build_rtcm_frame(MSG_1077_GPS_MSM7, bytes(payload))


# ----------------------------
# RTCM Stream Generator
# ----------------------------

@dataclass
class RTCMGeneratorConfig:
    """Configuration for RTCM generation."""
    station_id: int = 0
    station_x: float = -2694892.4600    # Default: approximate GPS reference station
    station_y: float = -4297557.4700
    station_z: float = 3854813.4200
    include_gps: bool = True
    include_glonass: bool = False
    num_epochs: int = 10
    epoch_interval_ms: int = 1000  # 1 Hz
    num_sats: int = 8


def generate_rtcm_stream(config: RTCMGeneratorConfig) -> bytes:
    """
    Generate a stream of RTCM messages.
    
    Stream structure:
      - 1x Message 1005 (station coordinates)
      - N x Message 1077 (GPS observations, one per epoch)
    
    Returns:
        Concatenated RTCM frames
    """
    frames: List[bytes] = []
    
    # Station coordinates message (sent once at start)
    station = StationCoordinates(
        x=config.station_x,
        y=config.station_y,
        z=config.station_z
    )
    frames.append(generate_msg_1005(station, config.station_id))
    
    # Observation messages (one per epoch)
    base_time_ms = 0  # GPS time of week in ms
    for epoch in range(config.num_epochs):
        gps_time_ms = base_time_ms + (epoch * config.epoch_interval_ms)
        
        if config.include_gps:
            frames.append(generate_msg_1077_stub(
                station_id=config.station_id,
                gps_time_ms=gps_time_ms,
                num_sats=config.num_sats
            ))
    
    return b''.join(frames)


# ----------------------------
# RTKLIB Integration (convbin)
# ----------------------------

def find_convbin() -> Optional[Path]:
    """Find RTKLIB convbin binary."""
    # 1) Explicit override if user sets it
    convbin_env = os.environ.get("CONVBIN_PATH", "").strip()
    if convbin_env:
        p = Path(convbin_env)
        if p.exists() and os.access(p, os.X_OK):
            return p.resolve()

    # 2) RTKLIB_HOME common locations (source tree builds)
    rtk_home = os.environ.get("RTKLIB_HOME", "").strip()
    if rtk_home:
        candidates = [
            Path(rtk_home) / "app" / "convbin" / "gcc" / "convbin",
            Path(rtk_home) / "app" / "convbin" / "convbin",
            Path(rtk_home) / "bin" / "convbin",
        ]
        for c in candidates:
            if c.exists() and os.access(c, os.X_OK):
                return c.resolve()

    # 3) PATH lookup
    which = shutil.which("convbin")
    if which:
        return Path(which).resolve()

    return None


def convert_rinex_to_rtcm_via_convbin(
    base_obs: Path,
    nav_file: Path,
    out_rtcm: Path,
    timeout_sec: int = 60
) -> Tuple[bool, str]:
    """
    Convert RINEX to RTCM using RTKLIB convbin.
    
    Command: convbin -r rtcm3 -o out.rtcm base.obs nav.nav
    """
    convbin = find_convbin()
    if not convbin:
        return False, "convbin not found"
    
    cmd = [
        str(convbin),
        "-r", "rtcm3",
        "-o", str(out_rtcm),
        str(base_obs),
        str(nav_file)
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,              # <-- IMPORTANT: keep bytes
            timeout=timeout_sec
        )
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return False, f"convbin failed (rc={proc.returncode}): {stderr.strip() or stdout.strip()}"

        if not out_rtcm.exists() or out_rtcm.stat().st_size == 0:
            return False, "convbin produced empty output"
        
        return True, f"Generated {out_rtcm.stat().st_size} bytes via convbin"
    
    except subprocess.TimeoutExpired:
        return False, "convbin timed out"
    except Exception as e:
        return False, f"convbin error: {e}"


# ----------------------------
# Scenario Loading
# ----------------------------

def load_scenario_base_station(scenario_root: Path, scenario: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Load base station files from scenario.
    
    Returns:
        (base_obs_path, nav_file_path) or (None, None) if not found
    """
    scen_dir = scenario_root / scenario
    profile_path = scen_dir / "scenario_profile.json"
    
    if not profile_path.exists():
        return None, None
    
    try:
        with profile_path.open("r") as f:
            cfg = json.load(f)
    except Exception:
        return None, None
    
    def resolve(key: str) -> Optional[Path]:
        v = cfg.get(key)
        if not v:
            return None
        p = scen_dir / v if not Path(v).is_absolute() else Path(v)
        return p.resolve() if p.exists() and p.stat().st_size > 0 else None
    
    return resolve("base_obs"), resolve("nav_file")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTCM Generator for GNSS Corrections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from scenario
  python rtcm_generator.py --scenario scenario1 --out corrections.rtcm

  # Generate with explicit files
  python rtcm_generator.py --base base.obs --nav nav.nav --out corrections.rtcm

  # Generate synthetic RTCM (no input files needed)
  python rtcm_generator.py --synthetic --out test_corrections.rtcm --epochs 100
        """
    )
    
    # Input options
    p.add_argument("--scenario", type=str, help="Scenario name")
    p.add_argument("--scenario-root", type=str, default=str(DEFAULT_SCENARIO_ROOT))
    p.add_argument("--base", type=str, help="Path to base observation file (RINEX)")
    p.add_argument("--nav", type=str, help="Path to navigation file (RINEX)")
    
    # Output options
    p.add_argument("--out", "-o", type=str, required=True, help="Output RTCM file path")
    
    # Synthetic mode
    p.add_argument("--synthetic", action="store_true", 
                   help="Generate synthetic RTCM (no real RINEX required)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of epochs for synthetic generation")
    p.add_argument("--num-sats", type=int, default=8,
                   help="Number of satellites in synthetic observations")
    p.add_argument("--station-id", type=int, default=0,
                   help="Reference station ID")
    
    # Station coordinates (for synthetic mode)
    p.add_argument("--station-x", type=float, default=-2694892.46,
                   help="Station ECEF X coordinate (m)")
    p.add_argument("--station-y", type=float, default=-4297557.47,
                   help="Station ECEF Y coordinate (m)")
    p.add_argument("--station-z", type=float, default=3854813.42,
                   help="Station ECEF Z coordinate (m)")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).resolve()
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Mode 1: Synthetic generation
    if args.synthetic:
        print(f"[INFO] Generating synthetic RTCM stream...")
        config = RTCMGeneratorConfig(
            station_id=args.station_id,
            station_x=args.station_x,
            station_y=args.station_y,
            station_z=args.station_z,
            num_epochs=args.epochs,
            num_sats=args.num_sats,
        )
        
        rtcm_data = generate_rtcm_stream(config)
        out_path.write_bytes(rtcm_data)
        
        print(f"[OK] Generated {len(rtcm_data)} bytes ({args.epochs} epochs)")
        print(f"     Output: {out_path}")
        return
    
    # Mode 2: From scenario
    base_obs: Optional[Path] = None
    nav_file: Optional[Path] = None
    
    if args.scenario:
        scenario_root = Path(args.scenario_root)
        base_obs, nav_file = load_scenario_base_station(scenario_root, args.scenario)
        
        if not base_obs or not nav_file:
            print(f"[WARN] Scenario '{args.scenario}' has empty/missing base or nav files.")
            print(f"       Falling back to synthetic RTCM generation.")
            # Generate synthetic RTCM directly (avoid recursive main() call)
            config = RTCMGeneratorConfig(
                station_id=args.station_id,
                station_x=args.station_x,
                station_y=args.station_y,
                station_z=args.station_z,
                num_epochs=args.epochs,
                num_sats=args.num_sats,
            )
            rtcm_data = generate_rtcm_stream(config)
            out_path.write_bytes(rtcm_data)
            print(f"[OK] Generated {len(rtcm_data)} bytes ({args.epochs} epochs, synthetic)")
            print(f"     Output: {out_path}")
            return
    
    # Mode 3: Explicit paths
    if args.base:
        base_obs = Path(args.base).resolve()
    if args.nav:
        nav_file = Path(args.nav).resolve()
    
    # Validate inputs
    if not base_obs or not nav_file:
        print("[ERROR] No valid base observation or navigation file.")
        print("        Use --synthetic for testing, or provide --base and --nav.")
        sys.exit(1)
    
    if not base_obs.exists() or base_obs.stat().st_size == 0:
        print(f"[WARN] Base file empty or missing: {base_obs}")
        print("       Falling back to synthetic RTCM generation.")
        config = RTCMGeneratorConfig(num_epochs=args.epochs, num_sats=args.num_sats)
        rtcm_data = generate_rtcm_stream(config)
        out_path.write_bytes(rtcm_data)
        print(f"[OK] Generated {len(rtcm_data)} bytes (synthetic)")
        print(f"     Output: {out_path}")
        return
    
    # Try RTKLIB convbin first
    print(f"[INFO] Attempting RTCM generation via RTKLIB convbin...")
    success, msg = convert_rinex_to_rtcm_via_convbin(base_obs, nav_file, out_path)
    
    if success:
        print(f"[OK] {msg}")
        print(f"     Output: {out_path}")
    else:
        print(f"[WARN] {msg}")
        print("       Falling back to synthetic RTCM generation.")
        config = RTCMGeneratorConfig(num_epochs=args.epochs, num_sats=args.num_sats)
        rtcm_data = generate_rtcm_stream(config)
        out_path.write_bytes(rtcm_data)
        print(f"[OK] Generated {len(rtcm_data)} bytes (synthetic)")
        print(f"     Output: {out_path}")


if __name__ == "__main__":
    main()
