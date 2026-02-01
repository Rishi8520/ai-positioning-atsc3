#!/usr/bin/env python3
"""
Scenario Simulator for Synthetic GNSS Dataset Generation

Generates deterministic synthetic datasets for scenario2 (sparse RTCM) and
scenario3 (urban multipath) testing. Creates ground truth files, corrections
schedules, and mode transition profiles.

This tool is intended for pipeline testing when real GNSS data is unavailable.
All outputs are deterministic given the same seed for reproducibility.

USAGE:
  # Generate scenario2 synthetic data
  python scenario_simulator.py --scenario scenario2 --seed 42 --n-epochs 300

  # Generate scenario3 synthetic data
  python scenario_simulator.py --scenario scenario3 --seed 42 --n-epochs 300

OUTPUT FILES:
  DATA/scenarios/<scenario>/
    ├── gt.csv                    # Ground truth trajectory
    ├── scenario_profile.json     # Updated with synthetic:true
    └── <scenario-specific files>

  scenario2 outputs:
    └── corrections_schedule.json # RTCM availability schedule

  scenario3 outputs:
    ├── multipath_profile.json    # Multipath model per epoch
    └── mode_transitions.csv      # FIX/FLOAT/STANDALONE per epoch

Author: GNSS Module
Date: 2026-02
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Constants
# ----------------------------

DEFAULT_SCENARIO_ROOT = Path("DATA/scenarios")
VALID_SCENARIOS = {"scenario2", "scenario3"}

# WGS84 constants
EARTH_RADIUS_M = 6371000.0

# Default trajectory parameters
# Base location: Seoul area (scenario profiles use this region)
DEFAULT_BASE_LAT = 37.5665
DEFAULT_BASE_LON = 126.9780
DEFAULT_BASE_HEIGHT = 50.0  # meters above ellipsoid

# Vehicle dynamics (meters per epoch at 1 Hz)
# Scenario 2: suburban/highway - faster, smoother
SCENARIO2_SPEED_MPS = 20.0  # ~72 km/h highway speed
SCENARIO2_HEADING_NOISE_DEG = 2.0
SCENARIO2_HEIGHT_VARIATION_M = 5.0

# Scenario 3: urban - slower, more variable
SCENARIO3_SPEED_MPS = 8.0  # ~29 km/h urban speed
SCENARIO3_HEADING_NOISE_DEG = 15.0  # More turns
SCENARIO3_HEIGHT_VARIATION_M = 10.0  # More elevation changes

# RTCM corrections schedule (scenario2)
RTCM_UPDATE_INTERVAL_MIN_S = 5
RTCM_UPDATE_INTERVAL_MAX_S = 10
RTCM_DROPOUT_PROBABILITY = 0.15  # 15% chance of dropout per interval

# Mode transitions (scenario3)
# Urban environment: frequent mode changes
MODE_FIX = 1
MODE_FLOAT = 2
MODE_STANDALONE = 5

# Probability of transitioning TO each mode (from any other mode)
# In urban canyon, standalone and float are more common
URBAN_MODE_PROBS = {
    MODE_FIX: 0.25,
    MODE_FLOAT: 0.35,
    MODE_STANDALONE: 0.40,
}

# Multipath parameters (scenario3)
MULTIPATH_BIAS_RANGE_M = (0.5, 5.0)  # Pseudorange bias from multipath
MULTIPATH_NOISE_RANGE_M = (0.1, 1.0)  # Additional noise std dev
NLOS_PROBABILITY = 0.4  # Probability of NLOS per epoch


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class GroundTruthEpoch:
    """Single epoch of ground truth position."""
    epoch: int
    timestamp: float  # seconds of day
    lat: float
    lon: float
    height: float
    velocity_mps: float
    heading_deg: float


@dataclass
class RTCMScheduleEntry:
    """Single entry in RTCM corrections schedule."""
    epoch_start: int
    epoch_end: int
    available: bool
    update_interval_s: float
    reason: str = ""


@dataclass
class MultipathEntry:
    """Multipath characteristics for a single epoch."""
    epoch: int
    multipath_bias_m: float
    multipath_noise_m: float
    nlos_affected: bool
    sky_view_pct: float
    num_visible_sats: int


@dataclass
class ModeTransitionEntry:
    """RTK mode for a single epoch."""
    epoch: int
    timestamp: float
    mode: int  # 1=FIX, 2=FLOAT, 5=STANDALONE
    mode_name: str
    duration_in_mode: int  # consecutive epochs in this mode


@dataclass
class SimulationResult:
    """Results from scenario simulation."""
    scenario: str
    seed: int
    n_epochs: int
    ground_truth: List[GroundTruthEpoch]
    corrections_schedule: Optional[List[RTCMScheduleEntry]] = None
    multipath_profile: Optional[List[MultipathEntry]] = None
    mode_transitions: Optional[List[ModeTransitionEntry]] = None
    timestamp: str = ""
    success: bool = False
    error_message: str = ""


# ----------------------------
# Utility Functions
# ----------------------------

def meters_to_deg_lat(meters: float) -> float:
    """Convert meters to degrees latitude."""
    return meters / 111000.0


def meters_to_deg_lon(meters: float, lat: float) -> float:
    """Convert meters to degrees longitude at given latitude."""
    return meters / (111000.0 * math.cos(math.radians(lat)))


def normalize_heading(heading: float) -> float:
    """Normalize heading to [0, 360) degrees."""
    while heading < 0:
        heading += 360.0
    while heading >= 360.0:
        heading -= 360.0
    return heading


def mode_name(mode: int) -> str:
    """Get human-readable mode name."""
    return {1: "FIX", 2: "FLOAT", 5: "STANDALONE"}.get(mode, f"Q{mode}")


# ----------------------------
# Ground Truth Generation
# ----------------------------

def generate_ground_truth(
    n_epochs: int,
    seed: int,
    scenario: str,
    base_lat: float = DEFAULT_BASE_LAT,
    base_lon: float = DEFAULT_BASE_LON,
    base_height: float = DEFAULT_BASE_HEIGHT,
) -> List[GroundTruthEpoch]:
    """
    Generate deterministic ground truth trajectory.
    
    Creates a realistic vehicle trajectory with:
    - Smooth heading changes
    - Speed variations
    - Height variations
    
    Args:
        n_epochs: Number of epochs to generate
        seed: Random seed for reproducibility
        scenario: "scenario2" or "scenario3" (affects dynamics)
        base_lat, base_lon, base_height: Starting position
    
    Returns:
        List of GroundTruthEpoch objects
    """
    rng = random.Random(seed)
    
    # Select scenario-specific parameters
    if scenario == "scenario2":
        base_speed = SCENARIO2_SPEED_MPS
        heading_noise = SCENARIO2_HEADING_NOISE_DEG
        height_var = SCENARIO2_HEIGHT_VARIATION_M
    else:  # scenario3
        base_speed = SCENARIO3_SPEED_MPS
        heading_noise = SCENARIO3_HEADING_NOISE_DEG
        height_var = SCENARIO3_HEIGHT_VARIATION_M
    
    epochs: List[GroundTruthEpoch] = []
    
    # Initial state
    lat = base_lat
    lon = base_lon
    height = base_height
    heading = rng.uniform(0, 360)  # Random initial heading
    
    # Base timestamp: start at 09:00:00 UTC (32400 seconds)
    base_timestamp = 32400.0
    
    for i in range(n_epochs):
        # Speed variation (±20%)
        speed = base_speed * (1.0 + rng.uniform(-0.2, 0.2))
        
        # Heading change (smoothed random walk)
        heading_delta = rng.gauss(0, heading_noise)
        heading = normalize_heading(heading + heading_delta)
        
        # Calculate displacement
        heading_rad = math.radians(heading)
        dx = speed * math.sin(heading_rad)  # East component
        dy = speed * math.cos(heading_rad)  # North component
        
        # Update position
        lat += meters_to_deg_lat(dy)
        lon += meters_to_deg_lon(dx, lat)
        
        # Height variation (smooth random walk)
        height += rng.gauss(0, height_var * 0.1)
        height = max(0.0, height)  # Keep above ground
        
        timestamp = base_timestamp + i
        
        epochs.append(GroundTruthEpoch(
            epoch=i,
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            height=height,
            velocity_mps=speed,
            heading_deg=heading,
        ))
    
    return epochs


# ----------------------------
# Scenario 2: RTCM Schedule Generation
# ----------------------------

def generate_corrections_schedule(
    n_epochs: int,
    seed: int,
) -> List[RTCMScheduleEntry]:
    """
    Generate RTCM corrections availability schedule for scenario2.
    
    Creates a sparse schedule with:
    - Updates every 5-10 seconds
    - Random dropouts (~15% probability)
    
    Args:
        n_epochs: Total number of epochs
        seed: Random seed
    
    Returns:
        List of RTCMScheduleEntry objects
    """
    rng = random.Random(seed + 100)  # Offset seed for different sequence
    
    schedule: List[RTCMScheduleEntry] = []
    epoch = 0
    
    while epoch < n_epochs:
        # Determine interval duration
        interval = rng.randint(RTCM_UPDATE_INTERVAL_MIN_S, RTCM_UPDATE_INTERVAL_MAX_S)
        end_epoch = min(epoch + interval - 1, n_epochs - 1)
        
        # Determine availability (dropout check)
        if rng.random() < RTCM_DROPOUT_PROBABILITY:
            available = False
            reason = "network_dropout"
        else:
            available = True
            reason = ""
        
        schedule.append(RTCMScheduleEntry(
            epoch_start=epoch,
            epoch_end=end_epoch,
            available=available,
            update_interval_s=float(interval),
            reason=reason,
        ))
        
        epoch = end_epoch + 1
    
    return schedule


# ----------------------------
# Scenario 3: Multipath & Mode Generation
# ----------------------------

def generate_multipath_profile(
    n_epochs: int,
    seed: int,
    ground_truth: List[GroundTruthEpoch],
) -> List[MultipathEntry]:
    """
    Generate multipath profile for scenario3 (urban canyon).
    
    Models:
    - Pseudorange bias from reflections
    - Additional measurement noise
    - NLOS (non-line-of-sight) events
    - Sky view percentage
    
    Args:
        n_epochs: Number of epochs
        seed: Random seed
        ground_truth: Ground truth trajectory (for context)
    
    Returns:
        List of MultipathEntry objects
    """
    rng = random.Random(seed + 200)
    
    profile: List[MultipathEntry] = []
    
    # State for correlated multipath (urban canyons are spatially correlated)
    current_severity = rng.uniform(0.3, 0.7)  # 0=open sky, 1=deep canyon
    
    for i in range(n_epochs):
        # Smooth severity changes (represents driving through urban environment)
        severity_delta = rng.gauss(0, 0.05)
        current_severity = max(0.1, min(0.9, current_severity + severity_delta))
        
        # NLOS probability scales with severity
        nlos = rng.random() < (NLOS_PROBABILITY * current_severity * 2)
        
        # Multipath bias (higher in deep canyon or NLOS)
        bias_min, bias_max = MULTIPATH_BIAS_RANGE_M
        if nlos:
            bias = rng.uniform(bias_max * 0.7, bias_max * 1.5)
        else:
            bias = rng.uniform(bias_min, bias_max) * current_severity
        
        # Additional noise
        noise_min, noise_max = MULTIPATH_NOISE_RANGE_M
        noise = rng.uniform(noise_min, noise_max) * (1 + current_severity)
        
        # Sky view (inverse of severity)
        sky_view = max(10, min(90, (1 - current_severity) * 100 + rng.gauss(0, 10)))
        
        # Visible satellites (fewer in urban canyon)
        base_sats = 12
        visible_sats = max(4, int(base_sats * (1 - current_severity * 0.6) + rng.gauss(0, 1)))
        
        profile.append(MultipathEntry(
            epoch=i,
            multipath_bias_m=round(bias, 3),
            multipath_noise_m=round(noise, 3),
            nlos_affected=nlos,
            sky_view_pct=round(sky_view, 1),
            num_visible_sats=visible_sats,
        ))
    
    return profile


def generate_mode_transitions(
    n_epochs: int,
    seed: int,
    multipath_profile: List[MultipathEntry],
    ground_truth: List[GroundTruthEpoch],
) -> List[ModeTransitionEntry]:
    """
    Generate RTK mode transitions for scenario3.
    
    Mode transitions are influenced by:
    - Multipath severity
    - NLOS events
    - Number of visible satellites
    
    Args:
        n_epochs: Number of epochs
        seed: Random seed
        multipath_profile: Generated multipath data
        ground_truth: Ground truth for timestamps
    
    Returns:
        List of ModeTransitionEntry objects
    """
    rng = random.Random(seed + 300)
    
    transitions: List[ModeTransitionEntry] = []
    current_mode = MODE_FLOAT  # Start in FLOAT mode
    duration = 0
    
    for i in range(n_epochs):
        mp = multipath_profile[i]
        gt = ground_truth[i]
        
        # Determine mode based on conditions
        if mp.nlos_affected or mp.num_visible_sats < 5:
            # Poor conditions: likely standalone
            if rng.random() < 0.7:
                new_mode = MODE_STANDALONE
            else:
                new_mode = MODE_FLOAT
        elif mp.sky_view_pct > 60 and mp.num_visible_sats >= 8:
            # Good conditions: chance of FIX
            if rng.random() < 0.5:
                new_mode = MODE_FIX
            elif rng.random() < 0.7:
                new_mode = MODE_FLOAT
            else:
                new_mode = MODE_STANDALONE
        else:
            # Medium conditions: mostly FLOAT
            if rng.random() < 0.6:
                new_mode = MODE_FLOAT
            elif rng.random() < 0.5:
                new_mode = MODE_STANDALONE
            else:
                new_mode = MODE_FIX
        
        # Add persistence (modes don't change every epoch)
        if current_mode == new_mode:
            duration += 1
        else:
            # 30% chance to stay in current mode (hysteresis)
            if duration > 0 and rng.random() < 0.3:
                new_mode = current_mode
                duration += 1
            else:
                current_mode = new_mode
                duration = 1
        
        transitions.append(ModeTransitionEntry(
            epoch=i,
            timestamp=gt.timestamp,
            mode=current_mode,
            mode_name=mode_name(current_mode),
            duration_in_mode=duration,
        ))
    
    return transitions


# ----------------------------
# File I/O
# ----------------------------

def write_ground_truth_csv(
    gt: List[GroundTruthEpoch],
    out_path: Path,
) -> None:
    """Write ground truth to CSV file."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "timestamp", "lat_deg", "lon_deg", "height_m",
            "velocity_mps", "heading_deg"
        ])
        for e in gt:
            writer.writerow([
                e.epoch,
                f"{e.timestamp:.3f}",
                f"{e.lat:.9f}",
                f"{e.lon:.9f}",
                f"{e.height:.3f}",
                f"{e.velocity_mps:.2f}",
                f"{e.heading_deg:.2f}",
            ])


def write_corrections_schedule_json(
    schedule: List[RTCMScheduleEntry],
    out_path: Path,
) -> None:
    """Write RTCM corrections schedule to JSON."""
    data = {
        "description": "RTCM corrections availability schedule",
        "synthetic": True,
        "total_intervals": len(schedule),
        "available_intervals": sum(1 for s in schedule if s.available),
        "dropout_intervals": sum(1 for s in schedule if not s.available),
        "schedule": [asdict(s) for s in schedule],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_multipath_profile_json(
    profile: List[MultipathEntry],
    out_path: Path,
) -> None:
    """Write multipath profile to JSON."""
    # Compute summary statistics
    avg_bias = sum(p.multipath_bias_m for p in profile) / len(profile)
    avg_noise = sum(p.multipath_noise_m for p in profile) / len(profile)
    nlos_count = sum(1 for p in profile if p.nlos_affected)
    avg_sky_view = sum(p.sky_view_pct for p in profile) / len(profile)
    avg_sats = sum(p.num_visible_sats for p in profile) / len(profile)
    
    data = {
        "description": "Urban multipath characteristics per epoch",
        "synthetic": True,
        "total_epochs": len(profile),
        "summary": {
            "avg_multipath_bias_m": round(avg_bias, 3),
            "avg_multipath_noise_m": round(avg_noise, 3),
            "nlos_affected_epochs": nlos_count,
            "nlos_percentage": round(100 * nlos_count / len(profile), 1),
            "avg_sky_view_pct": round(avg_sky_view, 1),
            "avg_visible_sats": round(avg_sats, 1),
        },
        "epochs": [asdict(p) for p in profile],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_mode_transitions_csv(
    transitions: List[ModeTransitionEntry],
    out_path: Path,
) -> None:
    """Write mode transitions to CSV."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "timestamp", "mode", "mode_name", "duration_in_mode"])
        for t in transitions:
            writer.writerow([
                t.epoch,
                f"{t.timestamp:.3f}",
                t.mode,
                t.mode_name,
                t.duration_in_mode,
            ])


def update_scenario_profile(
    profile_path: Path,
    scenario: str,
    n_epochs: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Update scenario_profile.json to mark as synthetic.
    
    Adds or updates the "synthetic_metadata" section.
    """
    # Read existing profile
    with profile_path.open("r", encoding="utf-8") as f:
        # Handle JSON with comments (remove // comments)
        content = f.read()
        lines = content.split("\n")
        clean_lines = [line for line in lines if not line.strip().startswith("//")]
        clean_content = "\n".join(clean_lines)
        profile = json.loads(clean_content)
    
    # Add synthetic metadata
    profile["synthetic_metadata"] = {
        "synthetic": True,
        "generator": "scenario_simulator.py",
        "seed": seed,
        "n_epochs": n_epochs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_type": scenario,
    }
    
    # Update ground_truth reference if needed
    if "files" in profile:
        profile["files"]["ground_truth"] = "gt.csv"
    else:
        profile["ground_truth"] = "gt.csv"
    
    # Write back
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    
    return profile


# ----------------------------
# Main Simulation Logic
# ----------------------------

def simulate_scenario(
    scenario: str,
    seed: int,
    n_epochs: int,
    scenario_root: Path,
) -> SimulationResult:
    """
    Run simulation for the specified scenario.
    
    Args:
        scenario: "scenario2" or "scenario3"
        seed: Random seed for reproducibility
        n_epochs: Number of epochs to simulate
        scenario_root: Root directory for scenarios
    
    Returns:
        SimulationResult with all generated data
    """
    result = SimulationResult(
        scenario=scenario,
        seed=seed,
        n_epochs=n_epochs,
        ground_truth=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    try:
        scen_dir = scenario_root / scenario
        if not scen_dir.exists():
            result.error_message = f"Scenario directory not found: {scen_dir}"
            return result
        
        profile_path = scen_dir / "scenario_profile.json"
        if not profile_path.exists():
            result.error_message = f"scenario_profile.json not found in {scen_dir}"
            return result
        
        # Load scenario profile for base coordinates
        with profile_path.open("r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
            clean_lines = [line for line in lines if not line.strip().startswith("//")]
            clean_content = "\n".join(clean_lines)
            profile = json.loads(clean_content)
        
        # Extract base coordinates from station_coordinates
        station_coords = profile.get("station_coordinates", {})
        # Convert ECEF to approximate lat/lon (simplified)
        # For now, use Seoul defaults if not available
        base_lat = DEFAULT_BASE_LAT
        base_lon = DEFAULT_BASE_LON
        base_height = DEFAULT_BASE_HEIGHT
        
        print(f"[INFO] Simulating {scenario} with seed={seed}, n_epochs={n_epochs}")
        print(f"[INFO] Base location: lat={base_lat:.4f}, lon={base_lon:.4f}")
        
        # Generate ground truth trajectory
        print("[INFO] Generating ground truth trajectory...")
        result.ground_truth = generate_ground_truth(
            n_epochs=n_epochs,
            seed=seed,
            scenario=scenario,
            base_lat=base_lat,
            base_lon=base_lon,
            base_height=base_height,
        )
        
        # Scenario-specific generation
        if scenario == "scenario2":
            print("[INFO] Generating RTCM corrections schedule...")
            result.corrections_schedule = generate_corrections_schedule(
                n_epochs=n_epochs,
                seed=seed,
            )
            
        elif scenario == "scenario3":
            print("[INFO] Generating multipath profile...")
            result.multipath_profile = generate_multipath_profile(
                n_epochs=n_epochs,
                seed=seed,
                ground_truth=result.ground_truth,
            )
            
            print("[INFO] Generating mode transitions...")
            result.mode_transitions = generate_mode_transitions(
                n_epochs=n_epochs,
                seed=seed,
                multipath_profile=result.multipath_profile,
                ground_truth=result.ground_truth,
            )
        
        # Write output files
        print("[INFO] Writing output files...")
        
        # Ground truth CSV
        gt_path = scen_dir / "gt.csv"
        write_ground_truth_csv(result.ground_truth, gt_path)
        print(f"  -> {gt_path}")
        
        if scenario == "scenario2":
            # Corrections schedule JSON
            schedule_path = scen_dir / "corrections_schedule.json"
            write_corrections_schedule_json(result.corrections_schedule, schedule_path)
            print(f"  -> {schedule_path}")
            
        elif scenario == "scenario3":
            # Multipath profile JSON
            mp_path = scen_dir / "multipath_profile.json"
            write_multipath_profile_json(result.multipath_profile, mp_path)
            print(f"  -> {mp_path}")
            
            # Mode transitions CSV
            mt_path = scen_dir / "mode_transitions.csv"
            write_mode_transitions_csv(result.mode_transitions, mt_path)
            print(f"  -> {mt_path}")
        
        # Update scenario profile
        print("[INFO] Updating scenario_profile.json...")
        update_scenario_profile(profile_path, scenario, n_epochs, seed)
        print(f"  -> {profile_path}")
        
        result.success = True
        print(f"[OK] Simulation complete for {scenario}")
        
    except Exception as e:
        result.error_message = str(e)
        print(f"[ERROR] Simulation failed: {e}")
    
    return result


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic GNSS scenario data for pipeline testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate scenario2 with default settings
  python scenario_simulator.py --scenario scenario2

  # Generate scenario3 with custom seed and epochs
  python scenario_simulator.py --scenario scenario3 --seed 123 --n-epochs 600

  # Specify custom scenario root
  python scenario_simulator.py --scenario scenario2 --scenario-root /path/to/scenarios
        """,
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=sorted(VALID_SCENARIOS),
        help="Scenario to simulate (scenario2 or scenario3)",
    )
    
    parser.add_argument(
        "--scenario-root",
        type=str,
        default=str(DEFAULT_SCENARIO_ROOT),
        help=f"Root directory for scenarios (default: {DEFAULT_SCENARIO_ROOT})",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=300,
        help="Number of epochs to simulate (default: 300)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    scenario_root = Path(args.scenario_root)
    
    if not scenario_root.exists():
        print(f"[ERROR] Scenario root not found: {scenario_root}")
        sys.exit(1)
    
    result = simulate_scenario(
        scenario=args.scenario,
        seed=args.seed,
        n_epochs=args.n_epochs,
        scenario_root=scenario_root,
    )
    
    if result.success:
        # Print summary
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Scenario:    {result.scenario}")
        print(f"  Seed:        {result.seed}")
        print(f"  Epochs:      {result.n_epochs}")
        print(f"  Timestamp:   {result.timestamp}")
        
        if result.corrections_schedule:
            avail = sum(1 for s in result.corrections_schedule if s.available)
            total = len(result.corrections_schedule)
            print(f"  RTCM intervals: {avail}/{total} available ({100*avail/total:.1f}%)")
        
        if result.multipath_profile:
            nlos = sum(1 for p in result.multipath_profile if p.nlos_affected)
            print(f"  NLOS epochs: {nlos}/{len(result.multipath_profile)} ({100*nlos/len(result.multipath_profile):.1f}%)")
        
        if result.mode_transitions:
            fix_epochs = sum(1 for t in result.mode_transitions if t.mode == MODE_FIX)
            float_epochs = sum(1 for t in result.mode_transitions if t.mode == MODE_FLOAT)
            standalone_epochs = sum(1 for t in result.mode_transitions if t.mode == MODE_STANDALONE)
            print(f"  Mode distribution:")
            print(f"    FIX:        {fix_epochs} ({100*fix_epochs/len(result.mode_transitions):.1f}%)")
            print(f"    FLOAT:      {float_epochs} ({100*float_epochs/len(result.mode_transitions):.1f}%)")
            print(f"    STANDALONE: {standalone_epochs} ({100*standalone_epochs/len(result.mode_transitions):.1f}%)")
        
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"\n[ERROR] Simulation failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
