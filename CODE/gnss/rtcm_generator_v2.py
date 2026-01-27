#!/usr/bin/env python3
"""
RTCM Generator for GNSS Corrections

Generates RTCM 3.x format messages from base station observations.

MODIFICATIONS FROM ORIGINAL:
1. ✅ Automatic station coordinate extraction from RINEX header
2. ✅ Scenario profile integration for station coordinates
3. ✅ Increased default epochs/satellites for realistic testing
4. ✅ Robust fallback chain: RINEX → Scenario Profile → CLI defaults

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
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    source: str = "unknown"  # Track where coordinates came from
    
    def __str__(self) -> str:
        return f"ECEF(X={self.x:.4f}, Y={self.y:.4f}, Z={self.z:.4f}m, H={self.antenna_height:.4f}m) [source: {self.source}]"


# ✅ NEW FUNCTION: Extract coordinates from RINEX header
def extract_station_coords_from_rinex(rinex_path: Path) -> Optional[StationCoordinates]:
    """
    Extract station coordinates from RINEX observation file header.
    
    RINEX format includes station coordinates in header:
      APPROX POSITION XYZ  -2694892.4600 -4297557.4700  3854813.4200
    
    Coordinate format (RINEX 2.x and 3.x):
      Columns 1-14:   X coordinate (m)
      Columns 15-28:  Y coordinate (m)
      Columns 29-42:  Z coordinate (m)
    
    Args:
        rinex_path: Path to RINEX observation file
    
    Returns:
        StationCoordinates object or None if not found
    """
    print(f"[INFO] Attempting to extract station coordinates from RINEX: {rinex_path.name}")
    
    if not rinex_path.exists():
        print(f"[WARN] RINEX file not found: {rinex_path}")
        return None
    
    try:
        # Try UTF-8 first, fallback to latin-1 for older RINEX files
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                with rinex_path.open("r", encoding=encoding, errors="ignore") as f:
                    line_num = 0
                    for line in f:
                        line_num += 1
                        
                        # Look for APPROX POSITION XYZ marker
                        if "APPROX POSITION XYZ" in line:
                            # Parse fixed-width format: cols 0-14, 14-28, 28-42
                            try:
                                x_str = line[0:14].strip()
                                y_str = line[14:28].strip()
                                z_str = line[28:42].strip()
                                
                                if not x_str or not y_str or not z_str:
                                    print(f"[WARN] Empty coordinate fields in RINEX header line {line_num}")
                                    continue
                                
                                x = float(x_str)
                                y = float(y_str)
                                z = float(z_str)
                                
                                # Sanity check: ECEF coordinates should be large values
                                if abs(x) < 1e5 or abs(y) < 1e5 or abs(z) < 1e5:
                                    print(f"[WARN] Suspicious coordinates (too small): X={x}, Y={y}, Z={z}")
                                    continue
                                
                                coords = StationCoordinates(
                                    x=x,
                                    y=y,
                                    z=z,
                                    antenna_height=0.0,
                                    source=f"RINEX:{rinex_path.name}"
                                )
                                
                                print(f"[OK] Extracted station coordinates from RINEX:")
                                print(f"     {coords}")
                                
                                return coords
                            
                            except ValueError as e:
                                print(f"[WARN] Failed to parse coordinate values: {e}")
                                print(f"       Line: {line.strip()}")
                                continue
                        
                        # Stop reading after header
                        if "END OF HEADER" in line:
                            break
                
                # If we successfully read the file but didn't find coords, break the encoding loop
                break
                
            except UnicodeDecodeError:
                # Try next encoding
                continue
        
        print(f"[WARN] APPROX POSITION XYZ not found in RINEX header")
        return None
    
    except Exception as e:
        print(f"[ERROR] Failed to read RINEX file: {e}")
        return None


# ✅ NEW FUNCTION: Load coordinates from scenario profile
def load_station_coords_from_scenario(scenario_root: Path, scenario: str) -> Optional[StationCoordinates]:
    """
    Load station coordinates from scenario profile JSON.
    
    Expected format in scenario_profile.json:
      "station_coordinates": {
        "x": -2694892.46,
        "y": -4297557.47,
        "z": 3854813.42,
        "antenna_height": 0.0,
        "description": "Seoul Base Station"
      }
    
    Args:
        scenario_root: Root directory for scenarios
        scenario: Scenario name
    
    Returns:
        StationCoordinates object or None if not found
    """
    scen_dir = scenario_root / scenario
    profile_path = scen_dir / "scenario_profile.json"
    
    if not profile_path.exists():
        print(f"[INFO] Scenario profile not found: {profile_path}")
        return None
    
    try:
        with profile_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        coords_cfg = cfg.get("station_coordinates")
        if not coords_cfg:
            print(f"[INFO] No 'station_coordinates' section in scenario profile")
            return None
        
        # Validate required fields
        if "x" not in coords_cfg or "y" not in coords_cfg or "z" not in coords_cfg:
            print(f"[WARN] Incomplete station_coordinates in profile (missing x/y/z)")
            return None
        
        coords = StationCoordinates(
            x=float(coords_cfg["x"]),
            y=float(coords_cfg["y"]),
            z=float(coords_cfg["z"]),
            antenna_height=float(coords_cfg.get("antenna_height", 0.0)),
            source=f"scenario_profile:{scenario}"
        )
        
        print(f"[OK] Loaded station coordinates from scenario profile:")
        print(f"     {coords}")
        
        return coords
    
    except Exception as e:
        print(f"[ERROR] Failed to load station coordinates from profile: {e}")
        return None


# ✅ NEW FUNCTION: Smart coordinate resolution with fallback chain
def resolve_station_coordinates(
    scenario_root: Optional[Path],
    scenario: Optional[str],
    base_obs: Optional[Path],
    cli_x: float,
    cli_y: float,
    cli_z: float,
    cli_antenna_height: float
) -> StationCoordinates:
    """
    Resolve station coordinates using intelligent fallback chain:
    
    Priority:
      1. RINEX observation file header (most reliable)
      2. Scenario profile JSON
      3. CLI arguments (defaults or user-provided)
    
    Args:
        scenario_root: Root directory for scenarios
        scenario: Scenario name
        base_obs: Path to base observation RINEX file
        cli_x, cli_y, cli_z: CLI-provided coordinates
        cli_antenna_height: CLI-provided antenna height
    
    Returns:
        StationCoordinates object (always returns valid coordinates)
    """
    print("\n" + "="*70)
    print("STATION COORDINATE RESOLUTION")
    print("="*70)
    
    # Priority 1: Extract from RINEX header (most reliable)
    if base_obs and base_obs.exists():
        coords = extract_station_coords_from_rinex(base_obs)
        if coords:
            print(f"[✓] Using coordinates from RINEX header")
            return coords
    
    # Priority 2: Load from scenario profile
    if scenario_root and scenario:
        coords = load_station_coords_from_scenario(scenario_root, scenario)
        if coords:
            print(f"[✓] Using coordinates from scenario profile")
            return coords
    
    # Priority 3: Use CLI arguments (defaults or user-provided)
    coords = StationCoordinates(
        x=cli_x,
        y=cli_y,
        z=cli_z,
        antenna_height=cli_antenna_height,
        source="CLI arguments"
    )
    
    print(f"[✓] Using coordinates from CLI arguments (defaults or user-provided)")
    print(f"    {coords}")
    print("="*70 + "\n")
    
    return coords


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


@dataclass
class RTCMGeneratorConfig:
    """Configuration for RTCM stream generation."""
    station: StationCoordinates
    station_id: int = 0
    num_epochs: int = 100  # ✅ CHANGED from 10 to 100
    num_sats: int = 12     # ✅ CHANGED from 8 to 12
    epoch_interval_ms: int = 1000  # 1 Hz update rate


def generate_rtcm_stream(config: RTCMGeneratorConfig) -> bytes:
    """
    Generate a complete RTCM correction stream.
    
    Stream structure:
      1. Message 1005: Station coordinates (sent once)
      2. Message 1077: GPS observations (sent every epoch)
    
    Args:
        config: Generator configuration
    
    Returns:
        Complete RTCM binary stream
    """
    stream = bytearray()
    
    # Send station coordinates (Message 1005) at start
    msg_1005 = generate_msg_1005(config.station, config.station_id)
    stream.extend(msg_1005)
    
    # Generate observations for each epoch
    for epoch in range(config.num_epochs):
        gps_time_ms = epoch * config.epoch_interval_ms
        
        # GPS MSM7 observations
        msg_1077 = generate_msg_1077_stub(
            station_id=config.station_id,
            gps_time_ms=gps_time_ms,
            num_sats=config.num_sats
        )
        stream.extend(msg_1077)
    
    return bytes(stream)


# ----------------------------
# RTKLIB convbin Integration
# ----------------------------

def find_convbin() -> Optional[Path]:
    """
    Find RTKLIB convbin binary.
    
    Search order:
      1. RTKLIB_HOME environment variable
      2. Common RTKLIB installation paths
      3. PATH lookup
    """
    # 1) RTKLIB_HOME env var
    rtk_home = os.environ.get("RTKLIB_HOME", "").strip()
    if rtk_home:
        candidates = [
            Path(rtk_home) / "bin" / "convbin",
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
    
    Command variants tried:
      1. convbin -r rtcm3 -o out.rtcm base.obs nav.nav
      2. convbin -r rtcm3 base.obs nav.nav (output to base.rtcm3)
    """
    convbin = find_convbin()
    if not convbin:
        return False, "convbin not found"
    
    # Try Method 1: Explicit output file with -o
    cmd = [
        str(convbin),
        "-r", "rtcm3",
        "-o", str(out_rtcm),
        str(base_obs),
        str(nav_file)
    ]
    
    print(f"[DEBUG] Running: {' '.join(cmd)}")
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,              # <-- IMPORTANT: keep bytes
            timeout=timeout_sec,
            cwd=base_obs.parent      # Run in same directory as input files
        )
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        
        print(f"[DEBUG] convbin return code: {proc.returncode}")
        if stderr.strip():
            print(f"[DEBUG] convbin stderr: {stderr.strip()}")
        if stdout.strip():
            print(f"[DEBUG] convbin stdout: {stdout.strip()}")

        if proc.returncode != 0:
            return False, f"convbin failed (rc={proc.returncode}): {stderr.strip() or stdout.strip()}"

        if not out_rtcm.exists():
            # Try Method 2: Check for auto-generated output file
            # convbin sometimes creates base.rtcm3 automatically
            auto_output = base_obs.with_suffix('.rtcm3')
            if auto_output.exists() and auto_output.stat().st_size > 0:
                # Move to desired output location
                import shutil
                shutil.move(str(auto_output), str(out_rtcm))
                return True, f"Generated {out_rtcm.stat().st_size} bytes via convbin (auto-output)"
            
            return False, f"convbin produced no output file (expected: {out_rtcm})"
        
        if out_rtcm.stat().st_size == 0:
            return False, "convbin produced empty output file (0 bytes)"
        
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
  # Generate from scenario (coordinates auto-extracted from RINEX)
  python rtcm_generator.py --scenario scenario1 --out corrections.rtcm

  # Generate with explicit files (coordinates auto-extracted)
  python rtcm_generator.py --base base.obs --nav nav.nav --out corrections.rtcm

  # Generate synthetic RTCM (no input files needed)
  python rtcm_generator.py --synthetic --out test_corrections.rtcm --epochs 100

  # Override coordinates manually (not recommended)
  python rtcm_generator.py --base base.obs --nav nav.nav --out corrections.rtcm \\
      --station-x 1234567.89 --station-y 9876543.21 --station-z 1111111.11
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
    p.add_argument("--epochs", type=int, default=100,  # ✅ CHANGED from 10 to 100
                   help="Number of epochs for synthetic generation (default: 100 = ~100 seconds at 1 Hz)")
    p.add_argument("--num-sats", type=int, default=12,  # ✅ CHANGED from 8 to 12
                   help="Number of satellites in synthetic observations (default: 12 = typical GPS)")
    p.add_argument("--station-id", type=int, default=0,
                   help="Reference station ID")
    
    # Station coordinates (now with informative defaults showing Seoul example)
    # ✅ CHANGED: Added warning in help text that auto-extraction is preferred
    p.add_argument("--station-x", type=float, default=-2694892.46,
                   help="Station ECEF X coordinate (m). DEFAULT: Seoul coordinates. RECOMMENDED: Let program auto-extract from RINEX header.")
    p.add_argument("--station-y", type=float, default=-4297557.47,
                   help="Station ECEF Y coordinate (m). DEFAULT: Seoul coordinates. RECOMMENDED: Let program auto-extract from RINEX header.")
    p.add_argument("--station-z", type=float, default=3854813.42,
                   help="Station ECEF Z coordinate (m). DEFAULT: Seoul coordinates. RECOMMENDED: Let program auto-extract from RINEX header.")
    p.add_argument("--antenna-height", type=float, default=0.0,
                   help="Antenna height above marker (m)")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).resolve()
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Mode 1: Synthetic generation
    if args.synthetic:
        print(f"[INFO] Generating synthetic RTCM stream...")
        
        # Use CLI coordinates for synthetic (no RINEX to extract from)
        station = StationCoordinates(
            x=args.station_x,
            y=args.station_y,
            z=args.station_z,
            antenna_height=args.antenna_height,
            source="CLI (synthetic mode)"
        )
        
        config = RTCMGeneratorConfig(
            station=station,
            station_id=args.station_id,
            num_epochs=args.epochs,
            num_sats=args.num_sats,
        )
        
        rtcm_data = generate_rtcm_stream(config)
        out_path.write_bytes(rtcm_data)
        
        print(f"[OK] Generated {len(rtcm_data)} bytes ({args.epochs} epochs, {args.num_sats} sats)")
        print(f"     Station: {station}")
        print(f"     Output: {out_path}")
        return
    
    # Mode 2: From scenario
    base_obs: Optional[Path] = None
    nav_file: Optional[Path] = None
    scenario_root: Optional[Path] = None
    scenario_name: Optional[str] = None
    
    if args.scenario:
        scenario_root = Path(args.scenario_root)
        scenario_name = args.scenario
        base_obs, nav_file = load_scenario_base_station(scenario_root, args.scenario)
        
        if not base_obs or not nav_file:
            print(f"[WARN] Scenario '{args.scenario}' has empty/missing base or nav files.")
            print(f"       Falling back to synthetic RTCM generation.")
            
            # ✅ FIXED: Use resolved station coordinates for synthetic fallback
            station = resolve_station_coordinates(
                scenario_root=scenario_root,
                scenario=scenario_name,
                base_obs=None,
                cli_x=args.station_x,
                cli_y=args.station_y,
                cli_z=args.station_z,
                cli_antenna_height=args.antenna_height
            )
            
            config = RTCMGeneratorConfig(
                station=station,
                station_id=args.station_id,
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
        
        # ✅ FIXED: Use resolved station coordinates for synthetic fallback
        station = resolve_station_coordinates(
            scenario_root=scenario_root,
            scenario=scenario_name,
            base_obs=None,
            cli_x=args.station_x,
            cli_y=args.station_y,
            cli_z=args.station_z,
            cli_antenna_height=args.antenna_height
        )
        
        config = RTCMGeneratorConfig(
            station=station,
            station_id=args.station_id,
            num_epochs=args.epochs,
            num_sats=args.num_sats
        )
        rtcm_data = generate_rtcm_stream(config)
        out_path.write_bytes(rtcm_data)
        print(f"[OK] Generated {len(rtcm_data)} bytes (synthetic)")
        print(f"     Output: {out_path}")
        return
    
    # ✅ CRITICAL: Resolve station coordinates using intelligent fallback chain
    station = resolve_station_coordinates(
        scenario_root=scenario_root,
        scenario=scenario_name,
        base_obs=base_obs,
        cli_x=args.station_x,
        cli_y=args.station_y,
        cli_z=args.station_z,
        cli_antenna_height=args.antenna_height
    )
    
    # Try RTKLIB convbin first
    print(f"[INFO] Attempting RTCM generation via RTKLIB convbin...")
    success, msg = convert_rinex_to_rtcm_via_convbin(base_obs, nav_file, out_path)
    
    if success:
        print(f"[OK] {msg}")
        print(f"     Station: {station}")
        print(f"     Output: {out_path}")
    else:
        print(f"[WARN] {msg}")
        print("       Falling back to synthetic RTCM generation.")
        config = RTCMGeneratorConfig(
            station=station,
            station_id=args.station_id,
            num_epochs=args.epochs,
            num_sats=args.num_sats
        )
        rtcm_data = generate_rtcm_stream(config)
        out_path.write_bytes(rtcm_data)
        print(f"[OK] Generated {len(rtcm_data)} bytes (synthetic)")
        print(f"     Output: {out_path}")


if __name__ == "__main__":
    main()