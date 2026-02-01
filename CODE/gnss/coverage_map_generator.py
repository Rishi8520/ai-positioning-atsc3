#!/usr/bin/env python3
"""
Coverage Map Generator for ATSC3 Broadcast Signal Analysis

Generates a simplified coverage proxy map based on free-space path loss (FSPL)
and SNR estimation. This is NOT RF-accurate but provides a deterministic baseline
for pipeline testing and demonstration purposes.

The coverage map divides the area around a transmitter into grid cells and
classifies each cell based on estimated SNR: good, marginal, or bad.

ASSUMPTIONS:
- Free-space path loss model (no terrain, obstacles, or atmospheric effects)
- Fixed noise floor assumption for SNR calculation
- Flat earth approximation for short distances (<100 km)

USAGE:
  # From scenario profile
  python coverage_map_generator.py --scenario scenario1 --out-dir OUTPUTS

  # With explicit transmitter location
  python coverage_map_generator.py --tx-lat 37.5665 --tx-lon 126.9780 \\
      --max-range-km 25 --grid-res-m 100 --out-dir OUTPUTS/scenario1

OUTPUT:
  <out-dir>/<scenario>/<scenario>_coverage_<timestamp>/coverage_map.csv
  <out-dir>/<scenario>/<scenario>_coverage_<timestamp>/coverage_summary.json
  <out-dir>/<scenario>/<scenario>_coverage_<timestamp>/run_manifest.json

Author: GNSS Module
Date: 2026-02-01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Constants
# ----------------------------

DEFAULT_SCENARIO_ROOT = Path("DATA/scenarios")
DEFAULT_OUT_DIR = Path("OUTPUTS")

# ATSC3 typical parameters (simplified)
# Frequency: ~600 MHz typical for ATSC3 low-VHF/UHF
DEFAULT_FREQUENCY_MHZ = 600.0
SPEED_OF_LIGHT_M_S = 299792458.0

# Default noise floor for SNR estimation (dBm)
# Typical receiver noise floor: -100 to -110 dBm
DEFAULT_NOISE_FLOOR_DBM = -105.0

# Default transmit power (EIRP in dBm) - typical ATSC3 station
# Note: Real stations vary from ~10 kW to 1 MW EIRP
DEFAULT_TX_POWER_DBM = 60.0  # ~1 kW EIRP

# SNR thresholds for coverage classification (dB)
# These are simplified; real ATSC3 mode-dependent
SNR_GOOD_THRESHOLD_DB = 20.0      # Robust reception
SNR_MARGINAL_THRESHOLD_DB = 10.0  # Marginal reception
# Below marginal = bad

# Earth radius for distance calculations (meters)
EARTH_RADIUS_M = 6371000.0


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class TransmitterConfig:
    """Transmitter configuration for coverage calculation."""
    lat: float              # Latitude (degrees)
    lon: float              # Longitude (degrees)
    height_m: float         # Antenna height above ground (meters)
    power_dbm: float        # Transmit power EIRP (dBm)
    frequency_mhz: float    # Center frequency (MHz)
    
    def __str__(self) -> str:
        return (f"TX(lat={self.lat:.6f}, lon={self.lon:.6f}, "
                f"h={self.height_m}m, P={self.power_dbm}dBm, f={self.frequency_mhz}MHz)")


@dataclass
class CoverageCell:
    """Single cell in the coverage grid."""
    lat: float
    lon: float
    distance_m: float
    fspl_db: float
    snr_db: float
    coverage_class: str  # "good", "marginal", "bad"


@dataclass
class CoverageSummary:
    """Summary statistics for coverage map."""
    total_cells: int
    good_cells: int
    marginal_cells: int
    bad_cells: int
    good_pct: float
    marginal_pct: float
    bad_pct: float
    snr_good_threshold_db: float
    snr_marginal_threshold_db: float
    grid_resolution_m: float
    max_range_km: float
    tx_config: Dict[str, Any]
    rx_height_m: float
    noise_floor_dbm: float


@dataclass
class RunManifest:
    """Metadata for a coverage map generation run."""
    run_id: str
    scenario: str
    timestamp: str
    args: Dict[str, Any]
    success: bool = False
    error_message: str = ""
    output_files: List[str] = field(default_factory=list)
    duration_sec: float = 0.0


# ----------------------------
# Utility Functions
# ----------------------------

def generate_run_id(scenario: str) -> Tuple[str, str]:
    """
    Generate a unique run ID with UTC timestamp.
    
    Returns:
        Tuple of (run_id, timestamp_str)
        run_id format: <scenario>_coverage_<YYYYMMDDTHHMMSSZ>
    """
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{scenario}_coverage_{timestamp_str}"
    return run_id, timestamp_str


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
    
    Returns:
        Distance in meters
    """
    # Convert to radians
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_M * c


def meters_to_degrees_lat(meters: float, lat: float) -> float:
    """
    Convert meters to degrees latitude (approximate).
    
    At the equator, 1 degree ≈ 111 km.
    """
    return meters / 111000.0


def meters_to_degrees_lon(meters: float, lat: float) -> float:
    """
    Convert meters to degrees longitude (approximate).
    
    Longitude degrees shrink with latitude: 1 deg ≈ 111 km * cos(lat).
    """
    lat_rad = math.radians(lat)
    return meters / (111000.0 * math.cos(lat_rad))


def calculate_fspl_db(distance_m: float, frequency_mhz: float) -> float:
    """
    Calculate Free-Space Path Loss (FSPL) in dB.
    
    FSPL (dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
              = 20*log10(d_km) + 20*log10(f_MHz) + 32.45
    
    Args:
        distance_m: Distance in meters (must be > 0)
        frequency_mhz: Frequency in MHz
    
    Returns:
        FSPL in dB (always positive, representing loss)
    """
    if distance_m <= 0:
        return 0.0  # At transmitter location
    
    distance_km = distance_m / 1000.0
    
    # FSPL formula: 20*log10(d_km) + 20*log10(f_MHz) + 32.45
    fspl = 20 * math.log10(distance_km) + 20 * math.log10(frequency_mhz) + 32.45
    
    return max(0.0, fspl)  # FSPL should not be negative


def calculate_snr_db(
    tx_power_dbm: float,
    fspl_db: float,
    noise_floor_dbm: float,
    tx_height_m: float = 30.0,
    rx_height_m: float = 1.5
) -> float:
    """
    Calculate estimated SNR at receiver.
    
    SNR = Received Power - Noise Floor
    Received Power = TX Power - FSPL + Antenna Gains (simplified to 0 dBi)
    
    Note: This is a simplified model. Real calculations would include:
    - Antenna gain patterns
    - Cable/connector losses
    - Atmospheric absorption
    - Terrain effects
    
    Args:
        tx_power_dbm: Transmit power EIRP (dBm)
        fspl_db: Free-space path loss (dB)
        noise_floor_dbm: Receiver noise floor (dBm)
        tx_height_m: Transmitter height (not used in FSPL, for reference)
        rx_height_m: Receiver height (not used in FSPL, for reference)
    
    Returns:
        Estimated SNR in dB
    """
    # Simple link budget: P_rx = P_tx - FSPL
    received_power_dbm = tx_power_dbm - fspl_db
    
    # SNR = P_rx - Noise Floor
    snr_db = received_power_dbm - noise_floor_dbm
    
    return snr_db


def classify_coverage(snr_db: float, good_thresh: float, marginal_thresh: float) -> str:
    """
    Classify coverage quality based on SNR.
    
    Args:
        snr_db: Signal-to-noise ratio (dB)
        good_thresh: SNR threshold for "good" classification
        marginal_thresh: SNR threshold for "marginal" classification
    
    Returns:
        "good", "marginal", or "bad"
    """
    if snr_db >= good_thresh:
        return "good"
    elif snr_db >= marginal_thresh:
        return "marginal"
    else:
        return "bad"


# ----------------------------
# Coverage Map Generation
# ----------------------------

def generate_coverage_grid(
    tx_config: TransmitterConfig,
    grid_res_m: float,
    max_range_km: float,
    rx_height_m: float = 1.5,
    noise_floor_dbm: float = DEFAULT_NOISE_FLOOR_DBM,
    snr_good_thresh: float = SNR_GOOD_THRESHOLD_DB,
    snr_marginal_thresh: float = SNR_MARGINAL_THRESHOLD_DB
) -> List[CoverageCell]:
    """
    Generate coverage grid cells around transmitter.
    
    Creates a rectangular grid centered on the transmitter, calculates FSPL
    and SNR for each cell, and classifies coverage quality.
    
    Args:
        tx_config: Transmitter configuration
        grid_res_m: Grid cell resolution in meters
        max_range_km: Maximum range from transmitter in km
        rx_height_m: Receiver height above ground in meters
        noise_floor_dbm: Receiver noise floor in dBm
        snr_good_thresh: SNR threshold for "good" coverage
        snr_marginal_thresh: SNR threshold for "marginal" coverage
    
    Returns:
        List of CoverageCell objects
    """
    cells: List[CoverageCell] = []
    
    max_range_m = max_range_km * 1000.0
    
    # Calculate grid extent in degrees
    lat_step = meters_to_degrees_lat(grid_res_m, tx_config.lat)
    lon_step = meters_to_degrees_lon(grid_res_m, tx_config.lat)
    
    # Calculate number of steps in each direction
    lat_extent_deg = meters_to_degrees_lat(max_range_m, tx_config.lat)
    lon_extent_deg = meters_to_degrees_lon(max_range_m, tx_config.lat)
    
    n_steps_lat = int(max_range_m / grid_res_m)
    n_steps_lon = int(max_range_m / grid_res_m)
    
    print(f"[INFO] Generating grid: {2*n_steps_lat+1} x {2*n_steps_lon+1} = "
          f"{(2*n_steps_lat+1) * (2*n_steps_lon+1)} cells")
    print(f"       Grid resolution: {grid_res_m}m, Max range: {max_range_km}km")
    
    # Generate grid (centered on transmitter)
    for i in range(-n_steps_lat, n_steps_lat + 1):
        for j in range(-n_steps_lon, n_steps_lon + 1):
            # Calculate cell center coordinates
            cell_lat = tx_config.lat + i * lat_step
            cell_lon = tx_config.lon + j * lon_step
            
            # Calculate distance from transmitter
            distance_m = haversine_distance_m(
                tx_config.lat, tx_config.lon,
                cell_lat, cell_lon
            )
            
            # Skip cells outside max range (circular boundary)
            if distance_m > max_range_m:
                continue
            
            # Calculate FSPL
            fspl_db = calculate_fspl_db(distance_m, tx_config.frequency_mhz)
            
            # Calculate SNR
            snr_db = calculate_snr_db(
                tx_power_dbm=tx_config.power_dbm,
                fspl_db=fspl_db,
                noise_floor_dbm=noise_floor_dbm,
                tx_height_m=tx_config.height_m,
                rx_height_m=rx_height_m
            )
            
            # Classify coverage
            coverage_class = classify_coverage(snr_db, snr_good_thresh, snr_marginal_thresh)
            
            cells.append(CoverageCell(
                lat=cell_lat,
                lon=cell_lon,
                distance_m=distance_m,
                fspl_db=round(fspl_db, 2),
                snr_db=round(snr_db, 2),
                coverage_class=coverage_class
            ))
    
    print(f"[INFO] Generated {len(cells)} cells within circular coverage area")
    
    return cells


def calculate_coverage_summary(
    cells: List[CoverageCell],
    tx_config: TransmitterConfig,
    grid_res_m: float,
    max_range_km: float,
    rx_height_m: float,
    noise_floor_dbm: float,
    snr_good_thresh: float,
    snr_marginal_thresh: float
) -> CoverageSummary:
    """
    Calculate summary statistics for coverage map.
    
    Args:
        cells: List of coverage cells
        tx_config: Transmitter configuration
        grid_res_m: Grid resolution in meters
        max_range_km: Maximum range in km
        rx_height_m: Receiver height in meters
        noise_floor_dbm: Noise floor in dBm
        snr_good_thresh: Good SNR threshold
        snr_marginal_thresh: Marginal SNR threshold
    
    Returns:
        CoverageSummary object
    """
    total = len(cells)
    good = sum(1 for c in cells if c.coverage_class == "good")
    marginal = sum(1 for c in cells if c.coverage_class == "marginal")
    bad = sum(1 for c in cells if c.coverage_class == "bad")
    
    good_pct = (good / total * 100) if total > 0 else 0.0
    marginal_pct = (marginal / total * 100) if total > 0 else 0.0
    bad_pct = (bad / total * 100) if total > 0 else 0.0
    
    return CoverageSummary(
        total_cells=total,
        good_cells=good,
        marginal_cells=marginal,
        bad_cells=bad,
        good_pct=round(good_pct, 2),
        marginal_pct=round(marginal_pct, 2),
        bad_pct=round(bad_pct, 2),
        snr_good_threshold_db=snr_good_thresh,
        snr_marginal_threshold_db=snr_marginal_thresh,
        grid_resolution_m=grid_res_m,
        max_range_km=max_range_km,
        tx_config={
            "lat": tx_config.lat,
            "lon": tx_config.lon,
            "height_m": tx_config.height_m,
            "power_dbm": tx_config.power_dbm,
            "frequency_mhz": tx_config.frequency_mhz
        },
        rx_height_m=rx_height_m,
        noise_floor_dbm=noise_floor_dbm
    )


# ----------------------------
# File I/O
# ----------------------------

def write_coverage_csv(cells: List[CoverageCell], output_path: Path) -> None:
    """
    Write coverage map to CSV file.
    
    Columns: lat, lon, distance_m, fspl_db, snr_db, class
    
    Args:
        cells: List of coverage cells
        output_path: Path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["lat", "lon", "distance_m", "fspl_db", "snr_db", "class"])
        
        # Data rows
        for cell in cells:
            writer.writerow([
                f"{cell.lat:.8f}",
                f"{cell.lon:.8f}",
                f"{cell.distance_m:.2f}",
                f"{cell.fspl_db:.2f}",
                f"{cell.snr_db:.2f}",
                cell.coverage_class
            ])
    
    print(f"[OK] Wrote coverage map: {output_path} ({len(cells)} cells)")


def write_coverage_summary(summary: CoverageSummary, output_path: Path) -> None:
    """
    Write coverage summary to JSON file.
    
    Args:
        summary: CoverageSummary object
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    
    print(f"[OK] Wrote coverage summary: {output_path}")


def write_run_manifest(manifest: RunManifest, output_path: Path) -> None:
    """
    Write run manifest to JSON file.
    
    Args:
        manifest: RunManifest object
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
    
    print(f"[OK] Wrote run manifest: {output_path}")


# ----------------------------
# Scenario Loading
# ----------------------------

def load_scenario_config(
    scenario_root: Path,
    scenario: str
) -> Optional[Dict[str, Any]]:
    """
    Load scenario configuration from scenario_profile.json.
    
    Args:
        scenario_root: Root directory for scenarios
        scenario: Scenario name
    
    Returns:
        Configuration dict or None if not found
    """
    profile_path = scenario_root / scenario / "scenario_profile.json"
    
    if not profile_path.exists():
        print(f"[WARN] Scenario profile not found: {profile_path}")
        return None
    
    try:
        with profile_path.open("r", encoding="utf-8") as f:
            # Handle JSON with comments (common in this project)
            content = f.read()
            # Remove single-line comments (// ...)
            lines = content.split("\n")
            clean_lines = [line for line in lines if not line.strip().startswith("//")]
            clean_content = "\n".join(clean_lines)
            
            return json.loads(clean_content)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse scenario profile: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load scenario profile: {e}")
        return None


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic (lat, lon, alt).
    
    Uses iterative method for WGS84 ellipsoid.
    
    Args:
        x, y, z: ECEF coordinates in meters
    
    Returns:
        (latitude_deg, longitude_deg, altitude_m)
    """
    # WGS84 parameters
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f ** 2  # First eccentricity squared
    
    # Longitude is straightforward
    lon = math.atan2(y, x)
    
    # Iterative solution for latitude and altitude
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - e2))  # Initial estimate
    
    for _ in range(10):  # Iterate until convergence
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat ** 2)
        lat_new = math.atan2(z + e2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    
    # Altitude
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)
    alt = p / math.cos(lat) - N
    
    return (math.degrees(lat), math.degrees(lon), alt)


def get_tx_coords_from_scenario(
    scenario_cfg: Dict[str, Any]
) -> Optional[Tuple[float, float]]:
    """
    Extract transmitter coordinates from scenario configuration.
    
    Tries multiple sources:
    1. Explicit tx_lat/tx_lon in scenario profile
    2. station_coordinates (converted from ECEF)
    
    Args:
        scenario_cfg: Scenario configuration dict
    
    Returns:
        (lat, lon) tuple or None
    """
    # Try explicit tx coordinates first
    if "tx_lat" in scenario_cfg and "tx_lon" in scenario_cfg:
        return (scenario_cfg["tx_lat"], scenario_cfg["tx_lon"])
    
    # Try station_coordinates (ECEF)
    station_coords = scenario_cfg.get("station_coordinates", {})
    if station_coords:
        x = station_coords.get("x_ecef") or station_coords.get("x")
        y = station_coords.get("y_ecef") or station_coords.get("y")
        z = station_coords.get("z_ecef") or station_coords.get("z")
        
        if x is not None and y is not None and z is not None:
            lat, lon, _ = ecef_to_geodetic(float(x), float(y), float(z))
            return (lat, lon)
    
    return None


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Coverage Map Generator for ATSC3 Broadcast Signal Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate coverage map for scenario1
  python coverage_map_generator.py --scenario scenario1

  # Generate with explicit transmitter location
  python coverage_map_generator.py --tx-lat 37.5665 --tx-lon 126.9780 \\
      --scenario scenario1

  # Custom grid resolution and range
  python coverage_map_generator.py --scenario scenario1 \\
      --grid-res-m 100 --max-range-km 30

  # Full custom configuration
  python coverage_map_generator.py --tx-lat 37.5 --tx-lon 127.0 \\
      --tx-height-m 50 --rx-height-m 2.0 \\
      --grid-res-m 25 --max-range-km 20 \\
      --out-dir OUTPUTS/custom

Output Files:
  <out-dir>/<scenario>/coverage/coverage_map.csv
    - Columns: lat, lon, distance_m, fspl_db, snr_db, class
    - class: "good", "marginal", or "bad" based on SNR thresholds

  <out-dir>/<scenario>/coverage/coverage_summary.json
    - Cell counts and percentages by class
    - Configuration parameters used
    - Thresholds applied

Notes:
  - Uses Free-Space Path Loss (FSPL) model - NOT RF-accurate
  - Suitable for pipeline testing and demonstration
  - Does not account for terrain, buildings, or atmospheric effects
        """
    )
    
    # Scenario options
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name (e.g., scenario1). If provided, loads config from scenario profile."
    )
    parser.add_argument(
        "--scenario-root",
        type=str,
        default=str(DEFAULT_SCENARIO_ROOT),
        help=f"Root directory for scenarios (default: {DEFAULT_SCENARIO_ROOT})"
    )
    
    # Output options
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory root (default: {DEFAULT_OUT_DIR})"
    )
    
    # Grid configuration
    parser.add_argument(
        "--grid-res-m",
        type=float,
        default=50.0,
        help="Grid cell resolution in meters (default: 50)"
    )
    parser.add_argument(
        "--max-range-km",
        type=float,
        default=50.0,
        help="Maximum range from transmitter in km (default: 50)"
    )
    
    # Transmitter configuration
    parser.add_argument(
        "--tx-lat",
        type=float,
        help="Transmitter latitude (degrees). Overrides scenario profile."
    )
    parser.add_argument(
        "--tx-lon",
        type=float,
        help="Transmitter longitude (degrees). Overrides scenario profile."
    )
    parser.add_argument(
        "--tx-height-m",
        type=float,
        default=30.0,
        help="Transmitter antenna height above ground (meters, default: 30)"
    )
    parser.add_argument(
        "--tx-power-dbm",
        type=float,
        default=DEFAULT_TX_POWER_DBM,
        help=f"Transmitter EIRP (dBm, default: {DEFAULT_TX_POWER_DBM})"
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        default=DEFAULT_FREQUENCY_MHZ,
        help=f"Center frequency (MHz, default: {DEFAULT_FREQUENCY_MHZ})"
    )
    
    # Receiver configuration
    parser.add_argument(
        "--rx-height-m",
        type=float,
        default=1.5,
        help="Receiver antenna height above ground (meters, default: 1.5)"
    )
    parser.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=DEFAULT_NOISE_FLOOR_DBM,
        help=f"Receiver noise floor (dBm, default: {DEFAULT_NOISE_FLOOR_DBM})"
    )
    
    # SNR thresholds
    parser.add_argument(
        "--snr-good-db",
        type=float,
        default=SNR_GOOD_THRESHOLD_DB,
        help=f"SNR threshold for 'good' coverage (dB, default: {SNR_GOOD_THRESHOLD_DB})"
    )
    parser.add_argument(
        "--snr-marginal-db",
        type=float,
        default=SNR_MARGINAL_THRESHOLD_DB,
        help=f"SNR threshold for 'marginal' coverage (dB, default: {SNR_MARGINAL_THRESHOLD_DB})"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    import time
    start_time = time.time()
    
    args = parse_args()
    
    print("=" * 70)
    print("COVERAGE MAP GENERATOR")
    print("=" * 70)
    
    # Resolve paths
    scenario_root = Path(args.scenario_root)
    out_dir = Path(args.out_dir)
    
    # Determine scenario name for output path
    scenario_name = args.scenario if args.scenario else "custom"
    
    # Generate unique run ID with UTC timestamp
    run_id, timestamp_str = generate_run_id(scenario_name)
    
    # Create run directory: OUTPUTS/<scenario>/<run_id>/
    run_dir = out_dir / scenario_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Output directory: {run_dir}")
    
    # Prepare args dict for manifest
    args_dict = {
        "scenario": args.scenario,
        "scenario_root": args.scenario_root,
        "out_dir": args.out_dir,
        "grid_res_m": args.grid_res_m,
        "max_range_km": args.max_range_km,
        "tx_lat": args.tx_lat,
        "tx_lon": args.tx_lon,
        "tx_height_m": args.tx_height_m,
        "tx_power_dbm": args.tx_power_dbm,
        "frequency_mhz": args.frequency_mhz,
        "rx_height_m": args.rx_height_m,
        "noise_floor_dbm": args.noise_floor_dbm,
        "snr_good_db": args.snr_good_db,
        "snr_marginal_db": args.snr_marginal_db,
    }
    
    # Initialize manifest
    manifest = RunManifest(
        run_id=run_id,
        scenario=scenario_name,
        timestamp=timestamp_str,
        args=args_dict,
        success=False,
        output_files=[]
    )
    
    # Load scenario profile if available
    scenario_cfg: Optional[Dict[str, Any]] = None
    if args.scenario:
        scenario_cfg = load_scenario_config(scenario_root, args.scenario)
        if scenario_cfg:
            print(f"[OK] Loaded scenario profile: {args.scenario}")
            if "description" in scenario_cfg:
                print(f"     Description: {scenario_cfg['description']}")
    
    # Resolve transmitter coordinates
    tx_lat: Optional[float] = args.tx_lat
    tx_lon: Optional[float] = args.tx_lon
    
    # If not provided via CLI, try scenario profile
    if tx_lat is None or tx_lon is None:
        if scenario_cfg:
            coords = get_tx_coords_from_scenario(scenario_cfg)
            if coords:
                tx_lat, tx_lon = coords
                print(f"[INFO] Using TX coordinates from scenario profile: "
                      f"lat={tx_lat:.6f}, lon={tx_lon:.6f}")
    
    # Validate we have coordinates
    if tx_lat is None or tx_lon is None:
        manifest.error_message = "Transmitter coordinates not specified"
        manifest.duration_sec = time.time() - start_time
        write_run_manifest(manifest, run_dir / "run_manifest.json")
        print("[ERROR] Transmitter coordinates not specified.")
        print("        Provide --tx-lat and --tx-lon, or use --scenario with a valid profile.")
        return 1
    
    # Update args with resolved coordinates
    args_dict["tx_lat_resolved"] = tx_lat
    args_dict["tx_lon_resolved"] = tx_lon
    
    # Get additional parameters from scenario profile if available
    tx_height = args.tx_height_m
    frequency = args.frequency_mhz
    
    if scenario_cfg:
        env = scenario_cfg.get("environment", {})
        # Could extend to read broadcast config for frequency, etc.
    
    # Create transmitter configuration
    tx_config = TransmitterConfig(
        lat=tx_lat,
        lon=tx_lon,
        height_m=tx_height,
        power_dbm=args.tx_power_dbm,
        frequency_mhz=frequency
    )
    
    print(f"\n[CONFIG] Transmitter: {tx_config}")
    print(f"[CONFIG] Receiver height: {args.rx_height_m}m")
    print(f"[CONFIG] Grid resolution: {args.grid_res_m}m")
    print(f"[CONFIG] Max range: {args.max_range_km}km")
    print(f"[CONFIG] Noise floor: {args.noise_floor_dbm}dBm")
    print(f"[CONFIG] SNR thresholds: good>{args.snr_good_db}dB, marginal>{args.snr_marginal_db}dB")
    print()
    
    # Generate coverage grid
    cells = generate_coverage_grid(
        tx_config=tx_config,
        grid_res_m=args.grid_res_m,
        max_range_km=args.max_range_km,
        rx_height_m=args.rx_height_m,
        noise_floor_dbm=args.noise_floor_dbm,
        snr_good_thresh=args.snr_good_db,
        snr_marginal_thresh=args.snr_marginal_db
    )
    
    if not cells:
        manifest.error_message = "No coverage cells generated"
        manifest.duration_sec = time.time() - start_time
        write_run_manifest(manifest, run_dir / "run_manifest.json")
        print("[ERROR] No coverage cells generated. Check parameters.")
        return 1
    
    # Calculate summary
    summary = calculate_coverage_summary(
        cells=cells,
        tx_config=tx_config,
        grid_res_m=args.grid_res_m,
        max_range_km=args.max_range_km,
        rx_height_m=args.rx_height_m,
        noise_floor_dbm=args.noise_floor_dbm,
        snr_good_thresh=args.snr_good_db,
        snr_marginal_thresh=args.snr_marginal_db
    )
    
    # Print summary
    print()
    print("-" * 50)
    print("COVERAGE SUMMARY")
    print("-" * 50)
    print(f"  Total cells:    {summary.total_cells}")
    print(f"  Good cells:     {summary.good_cells} ({summary.good_pct:.1f}%)")
    print(f"  Marginal cells: {summary.marginal_cells} ({summary.marginal_pct:.1f}%)")
    print(f"  Bad cells:      {summary.bad_cells} ({summary.bad_pct:.1f}%)")
    print("-" * 50)
    
    # Write outputs to run directory
    csv_path = run_dir / "coverage_map.csv"
    json_path = run_dir / "coverage_summary.json"
    manifest_path = run_dir / "run_manifest.json"
    
    write_coverage_csv(cells, csv_path)
    write_coverage_summary(summary, json_path)
    
    # Update and write manifest
    manifest.success = True
    manifest.duration_sec = time.time() - start_time
    manifest.output_files = [
        str(csv_path.name),
        str(json_path.name),
        str(manifest_path.name)
    ]
    write_run_manifest(manifest, manifest_path)
    
    print()
    print("[DONE] Coverage map generation complete")
    print(f"       Run ID: {run_id}")
    print(f"       Output directory: {run_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
