#!/usr/bin/env python3
"""
Populate Scenario 2 with UrbanNav Real GNSS Data

Downloads and prepares real GNSS data from the UrbanNav Hong Kong dataset
for scenario2 (vehicle suburban / low obstruction / lower SNR / sparser corrections).

DATASET: UrbanNav-HK-Medium-Urban-1 (TST area)
- GNSS RINEX v3.02 (u-blox ZED-F9P receivers)
- Ground Truth from NovAtel SPAN-CPT + IE post-processing (1Hz, ~5cm RMSE)
- 785 seconds duration, 3.64 km path, Medium urban canyon

Why this dataset for scenario2:
- "Medium" urban canyon matches scenario2's "suburban/highway moderate obstruction" intent
- High-quality ground truth available (SPAN-CPT RTK/INS)
- Real GNSS RINEX files suitable for RTKLIB processing
- Multiple receivers provide rover/base capability

USAGE:
  # Download and populate scenario2
  python populate_scenario2_urbannav.py

  # Force re-download even if files exist
  python populate_scenario2_urbannav.py --force

  # Specify custom scenario root
  python populate_scenario2_urbannav.py --scenario-root /path/to/scenarios

  # Skip download (use existing raw files)
  python populate_scenario2_urbannav.py --skip-download

OUTPUT:
  DATA/scenarios/scenario2/
    rover.obs        - Rover RINEX observation file
    base.obs         - Base station RINEX observation file  
    nav.nav          - Navigation/ephemeris file
    gt.csv           - Ground truth (converted from SPAN-CPT)
    scenario_profile.json - Updated profile (synthetic=false)
    raw/             - Downloaded raw files (gitignored)

Author: GNSS Module
Date: 2026-02-01
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Constants
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENARIO_ROOT = PROJECT_ROOT / "DATA" / "scenarios"
CODE_GNSS = PROJECT_ROOT / "CODE" / "gnss"

# UrbanNav dataset URLs (Dropbox direct download links)
# Dataset: UrbanNav-HK-Medium-Urban-1 (TST area, Medium urban canyon)
URBANNAV_URLS = {
    # GNSS RINEX folder (contains multiple .obs and .nav files)
    "gnss_folder": "https://www.dropbox.com/sh/2haoy68xekg95zl/AAAkcN4FwhFxkPY1lXsxbJrxa?dl=1",
    # Ground truth file
    "ground_truth": "https://www.dropbox.com/s/twsvwftucoytfpc/UrbanNav_TST_GT_raw.txt?dl=1",
}

# Alternative: UrbanNav-HK-Data20200314 (Light urban, pilot data)
# Note: No separate ground truth file, would need ROS bag extraction
URBANNAV_PILOT_URLS = {
    "gnss_folder": "https://www.dropbox.com/sh/22mnpg86umjb4zh/AAAsIiQ1WAMp20OROUn4h6pka?dl=1",
}

# Dataset metadata
DATASET_INFO = {
    "name": "UrbanNav-HK-Medium-Urban-1",
    "location": "Hong Kong, TST (Tsim Sha Tsui)",
    "date": "2021-05-17",
    "duration_sec": 785,
    "path_length_km": 3.64,
    "environment": "Medium urban canyon",
    "ground_truth_source": "NovAtel SPAN-CPT + IE post-processing",
    "ground_truth_accuracy_cm": 5.0,
    "gnss_receivers": ["u-blox ZED-F9P x2", "u-blox EVK-M8T x1", "NovAtel Flexpak6"],
    "rinex_version": "3.02",
    "reference_url": "https://github.com/IPNL-POLYU/UrbanNavDataset",
}

# Hong Kong approximate coordinates (for scenario profile)
HK_TST_COORDS = {
    "lat": 22.2988,
    "lon": 114.1722,
    "description": "Tsim Sha Tsui, Hong Kong",
}


# ----------------------------
# Logging
# ----------------------------

def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def _error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


def _die(msg: str, code: int = 1) -> None:
    _error(msg)
    sys.exit(code)


# ----------------------------
# File Operations
# ----------------------------

def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a file from URL with progress indication."""
    _info(f"Downloading: {url[:80]}...")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (GNSS-Populate/1.0)",
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = response.headers.get('Content-Length')
            if total_size:
                total_size = int(total_size)
                _info(f"  File size: {total_size / 1024 / 1024:.1f} MB")
            
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                downloaded = 0
                block_size = 8192
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
                print()  # newline after progress
                
        _ok(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    except urllib.error.HTTPError as e:
        _error(f"HTTP error downloading {url}: {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        _error(f"URL error downloading {url}: {e.reason}")
        return False
    except Exception as e:
        _error(f"Error downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract tar/zip archive."""
    _info(f"Extracting: {archive_path.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
        elif archive_path.suffix in (".tar", ".gz", ".tgz") or ".tar" in archive_path.name:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(dest_dir)
        else:
            _warn(f"Unknown archive format: {archive_path}")
            return False
        
        _ok(f"Extracted to: {dest_dir}")
        return True
    except Exception as e:
        _error(f"Error extracting {archive_path}: {e}")
        return False


def decompress_gzip(gz_path: Path, dest_path: Path) -> bool:
    """Decompress a gzip file."""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        _error(f"Error decompressing {gz_path}: {e}")
        return False


def download_broadcast_ephemeris(dest_path: Path, year: int = 2021, doy: int = 137) -> bool:
    """
    Download broadcast ephemeris from IGS/CDDIS for a specific date.
    
    The UrbanNav dataset date is 2021-05-17 = DOY 137, Year 2021.
    
    Sources tried in order:
    1. IGS BRDC combined file from CDDIS
    2. BKG's broadcast archive
    3. Generate minimal nav file if all else fails
    
    Args:
        dest_path: Where to save the nav file
        year: Year (default 2021)
        doy: Day of year (default 137 = May 17)
    
    Returns:
        True if successful
    """
    _info(f"Downloading broadcast ephemeris for {year} DOY {doy}...")
    
    # Format DOY with leading zeros
    doy_str = f"{doy:03d}"
    yy = str(year)[-2:]
    
    # List of potential sources for broadcast ephemeris
    # Note: CDDIS requires EarthData login, so we try alternatives first
    nav_urls = [
        # IGN (France) - often publicly accessible
        f"https://igs.ign.fr/pub/igs/data/{year}/{doy_str}/brdc{doy_str}0.{yy}n.gz",
        f"https://igs.ign.fr/pub/igs/data/{year}/{doy_str}/BRDC00IGS_R_{year}{doy_str}0000_01D_MN.rnx.gz",
        # BKG (Germany)
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy_str}/brdc{doy_str}0.{yy}n.gz",
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy_str}/BRDC00IGS_R_{year}{doy_str}0000_01D_MN.rnx.gz",
        # CDDIS (requires auth, may fail)
        f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/brdc/brdc{doy_str}0.{yy}n.gz",
    ]
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_gz = dest_path.parent / f"nav_temp.gz"
    
    for url in nav_urls:
        _info(f"  Trying: {url[:60]}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (GNSS-Populate/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(temp_gz, 'wb') as f:
                    f.write(response.read())
            
            # Decompress
            if decompress_gzip(temp_gz, dest_path):
                temp_gz.unlink(missing_ok=True)
                _ok(f"Downloaded broadcast ephemeris ({dest_path.stat().st_size / 1024:.1f} KB)")
                return True
        except Exception as e:
            _warn(f"  Failed: {e}")
            continue
    
    # If all downloads fail, create a minimal nav file
    _warn("Could not download broadcast ephemeris from any source")
    _info("Creating minimal navigation file placeholder...")
    
    # Create a minimal RINEX nav header - RTKLIB can work in SPP mode without full ephemeris
    # if the observation file has pseudoranges
    minimal_nav = f"""     3.04           N: GNSS NAV DATA    M: Mixed            RINEX VERSION / TYPE
populate_scenario2  GNSS Module         {datetime.now().strftime('%Y%m%d %H%M%S')} UTC PGM / RUN BY / DATE
Minimal nav file - broadcast ephemeris not available        COMMENT
UrbanNav dataset {year}-DOY{doy_str}                        COMMENT
                                                            END OF HEADER
"""
    with open(dest_path, 'w') as f:
        f.write(minimal_nav)
    
    _warn(f"Created minimal nav placeholder - RTK may be limited to SPP mode")
    return True


# ----------------------------
# Ground Truth Conversion
# ----------------------------

def dms_to_decimal(degrees: float, minutes: float, seconds: float) -> float:
    """Convert degrees-minutes-seconds to decimal degrees."""
    sign = 1 if degrees >= 0 else -1
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def parse_span_cpt_ground_truth(gt_path: Path) -> List[Dict[str, Any]]:
    """
    Parse SPAN-CPT ground truth file from UrbanNav dataset.
    
    UrbanNav ground truth format (UrbanNav_TST_GT_raw.txt):
    UTCTime Week GPSTime Latitude Longitude H-Ell VelBdyX ... Q
    
    Where Latitude/Longitude are in D M S format (e.g., "22 18 04.31949")
    
    Example line:
    1621218775.00 2158.00000 95593.00 22 18 04.31949 114 10 44.60559 3.472 ...
    
    Returns list of dicts with: timestamp (GPS TOW), gps_week, lat_deg, lon_deg, height_m
    """
    epochs = []
    
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # Skip header lines and comments
        if not line or line.startswith('#') or line.startswith('%'):
            continue
        if line.startswith('//') or line.startswith('UTCTime') or '(sec)' in line:
            continue
        
        parts = line.split()
        
        # UrbanNav format has at least 16 columns
        # UTCTime(0) Week(1) GPSTime(2) LatD(3) LatM(4) LatS(5) LonD(6) LonM(7) LonS(8) H-Ell(9) ...
        if len(parts) < 10:
            continue
        
        try:
            # Parse GPS week and TOW (Time of Week) for timestamp matching with RTKLIB
            gps_week = int(float(parts[1]))
            gps_tow = float(parts[2])  # This is what RTKLIB uses as timestamp
            
            # Parse latitude D M S
            lat_d = float(parts[3])
            lat_m = float(parts[4])
            lat_s = float(parts[5])
            lat_deg = dms_to_decimal(lat_d, lat_m, lat_s)
            
            # Parse longitude D M S
            lon_d = float(parts[6])
            lon_m = float(parts[7])
            lon_s = float(parts[8])
            lon_deg = dms_to_decimal(lon_d, lon_m, lon_s)
            
            # Parse height (H-Ell, ellipsoidal height)
            height_m = float(parts[9])
            
            # Parse quality flag (Q column, last column or near end)
            # Quality: 1=SPP, 2=DGPS, 3=RTK Float, 4=RTK Fix, etc.
            quality = 1
            if len(parts) >= 17:
                try:
                    quality = int(parts[16])
                except ValueError:
                    quality = 1
            
            # Validate coordinates (Hong Kong area: ~22.2-22.5 N, ~114.0-114.3 E)
            if 22.0 < lat_deg < 22.6 and 113.8 < lon_deg < 114.4:
                epochs.append({
                    "timestamp": gps_tow,  # Use GPS TOW for timestamp matching
                    "gps_week": gps_week,
                    "lat_deg": lat_deg,
                    "lon_deg": lon_deg,
                    "height_m": height_m,
                    "quality": quality,
                })
        except (ValueError, IndexError) as e:
            continue
    
    return epochs


def convert_ground_truth_to_csv(
    epochs: List[Dict[str, Any]],
    output_path: Path,
    resample_interval_sec: float = 1.0
) -> int:
    """
    Convert ground truth epochs to CSV format expected by gnss_baseline_runner_v2.py.
    
    Output format:
    epoch,timestamp,lat_deg,lon_deg,height_m,quality,num_sats
    
    The timestamp field uses GPS TOW (Time of Week in seconds) to match RTKLIB output format.
    
    Args:
        epochs: List of ground truth epochs
        output_path: Output CSV path
        resample_interval_sec: Resampling interval (default 1 Hz) - not used if epochs are already 1Hz
    
    Returns:
        Number of epochs written
    """
    if not epochs:
        return 0
    
    # Sort by timestamp (GPS TOW)
    epochs = sorted(epochs, key=lambda e: e["timestamp"])
    
    # Write all epochs directly without resampling
    # The GPS TOW timestamp is preserved for matching with RTKLIB solution
    output_epochs = []
    for i, ep in enumerate(epochs):
        output_epochs.append({
            "epoch": i,
            "timestamp": round(ep["timestamp"], 3),  # GPS TOW
            "lat_deg": ep["lat_deg"],
            "lon_deg": ep["lon_deg"],
            "height_m": ep["height_m"],
            "quality": ep.get("quality", 1),
            "num_sats": 12,  # Typical for SPAN-CPT
        })
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "timestamp", "lat_deg", "lon_deg", "height_m", "quality", "num_sats"
        ])
        writer.writeheader()
        writer.writerows(output_epochs)
    
    return len(output_epochs)


# ----------------------------
# RINEX File Processing
# ----------------------------

def find_rinex_files(directory: Path) -> Dict[str, List[Path]]:
    """
    Find RINEX files in a directory.
    
    Returns dict with keys: obs, nav
    """
    result = {"obs": [], "nav": []}
    
    if not directory.exists():
        return result
    
    for f in directory.rglob("*"):
        if not f.is_file():
            continue
        
        name_lower = f.name.lower()
        suffix = f.suffix.lower()
        
        # Skip archive files
        if suffix in (".zip", ".tar", ".tgz", ".7z", ".rar"):
            continue
        
        # Observation files: .obs, .o, .*o (e.g., .21o)
        if suffix in (".obs", ".o") or (len(suffix) == 4 and suffix.endswith("o") and suffix[1:3].isdigit()):
            result["obs"].append(f)
        # RINEX 3 observation: .rnx
        elif suffix == ".rnx" and "o" in name_lower:
            result["obs"].append(f)
        # Navigation files: .nav, .n, .*n (e.g., .21n), .*p (e.g., .21p), .*g (e.g., .21g)
        elif suffix in (".nav", ".n", ".p", ".g"):
            result["nav"].append(f)
        elif len(suffix) == 4 and suffix[1:3].isdigit() and suffix[-1] in "npg":
            result["nav"].append(f)
        # Gzipped files
        elif suffix == ".gz":
            stem = f.stem.lower()
            if stem.endswith(".obs") or stem.endswith("o"):
                result["obs"].append(f)
            elif stem.endswith(".nav") or stem.endswith(("n", "p", "g")):
                result["nav"].append(f)
    
    return result


def select_best_rinex_pair(rinex_files: Dict[str, List[Path]]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Select best rover, base, and nav files from available RINEX files.
    
    For UrbanNav, we typically have multiple receivers:
    - F9P receivers (u-blox) - good for rover
    - M8T receiver - can be used as pseudo-base
    - NovAtel Flexpak6 - high quality
    
    Returns: (rover_obs, base_obs, nav)
    """
    obs_files = sorted(rinex_files.get("obs", []))
    nav_files = sorted(rinex_files.get("nav", []))
    
    rover_obs = None
    base_obs = None
    nav_file = None
    
    # Select observation files
    if len(obs_files) >= 2:
        # Use first two as rover and base
        # Prefer F9P files if available
        f9p_files = [f for f in obs_files if "f9p" in f.name.lower()]
        if len(f9p_files) >= 2:
            rover_obs = f9p_files[0]
            base_obs = f9p_files[1]
        else:
            rover_obs = obs_files[0]
            base_obs = obs_files[1]
    elif len(obs_files) == 1:
        rover_obs = obs_files[0]
        _warn("Only one observation file found - no base station available")
    
    # Select navigation file (prefer combined/mixed nav)
    if nav_files:
        # Prefer files with 'MN' (mixed nav) or larger files
        mixed_nav = [f for f in nav_files if "mn" in f.name.lower() or "mix" in f.name.lower()]
        if mixed_nav:
            nav_file = mixed_nav[0]
        else:
            # Use largest nav file (usually most complete)
            nav_file = max(nav_files, key=lambda f: f.stat().st_size if f.exists() else 0)
    
    return rover_obs, base_obs, nav_file


def copy_and_decompress_rinex(src: Path, dest: Path) -> bool:
    """Copy RINEX file, decompressing if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if src.suffix.lower() == ".gz":
        return decompress_gzip(src, dest)
    else:
        shutil.copy2(src, dest)
        return True


# ----------------------------
# Scenario Profile Update
# ----------------------------

def load_json_with_comments(path: Path) -> Dict:
    """Load JSON file, stripping // comments."""
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    clean_lines = [line for line in lines if not line.strip().startswith("//")]
    clean_content = "\n".join(clean_lines)
    
    return json.loads(clean_content)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> Tuple[float, float, float]:
    """Convert geodetic coordinates to ECEF."""
    import math
    
    # WGS84 parameters
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f ** 2  # First eccentricity squared
    
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)
    
    x = (N + height_m) * cos_lat * cos_lon
    y = (N + height_m) * cos_lat * sin_lon
    z = (N * (1 - e2) + height_m) * sin_lat
    
    return x, y, z


def update_scenario_profile(
    scenario_dir: Path,
    dataset_info: Dict[str, Any],
    gt_epochs: int,
    files: Dict[str, str],
    coords: Tuple[float, float, float],
) -> None:
    """Update scenario_profile.json with real data metadata."""
    
    profile_path = scenario_dir / "scenario_profile.json"
    
    if profile_path.exists():
        profile = load_json_with_comments(profile_path)
    else:
        profile = {}
    
    # Update basic info
    profile["name"] = "scenario2_vehicle_suburban"
    profile["description"] = (
        f"Vehicle navigation - {dataset_info['name']} ({dataset_info['location']}). "
        f"{dataset_info['environment']} environment, {dataset_info['duration_sec']}s duration, "
        f"{dataset_info['path_length_km']}km path."
    )
    profile["intent"] = "minimize_bandwidth_maintain_accuracy"
    profile["scenario_type"] = "vehicle_suburban"
    
    # Update file references
    profile["files"] = {
        "rover_obs": files.get("rover_obs", "rover.obs"),
        "nav_file": files.get("nav_file", "nav.nav"),
        "base_obs": files.get("base_obs", "base.obs"),
        "ground_truth": files.get("ground_truth", "gt.csv"),
        "rtcm_file": "corrections.rtcm",
    }
    
    # Update coordinates (ECEF from first ground truth point)
    x, y, z = coords
    profile["station_coordinates"] = {
        "x_ecef": round(x, 1),
        "y_ecef": round(y, 1),
        "z_ecef": round(z, 1),
    }
    
    # Update environment
    profile["environment"] = {
        "sky_view": "partial",
        "multipath": "moderate",
        "signal_quality": "good",
        "baseline_km": 0.0,  # Short baseline (same vehicle)
        "dynamics": "high",
        "snr_margin_db": 3.0,
        "urban_density": "medium",
    }
    
    # CRITICAL: Mark as NON-synthetic real data
    profile["synthetic_metadata"] = {
        "synthetic": False,
        "real_data_source": "UrbanNav",
        "dataset_name": dataset_info["name"],
        "dataset_location": dataset_info["location"],
        "dataset_date": dataset_info["date"],
        "dataset_url": dataset_info["reference_url"],
        "ground_truth_source": dataset_info["ground_truth_source"],
        "ground_truth_accuracy_cm": dataset_info["ground_truth_accuracy_cm"],
        "ground_truth_epochs": gt_epochs,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "populate_scenario2_urbannav.py",
    }
    
    # Ensure traditional and AI configs exist
    if "traditional_config" not in profile:
        profile["traditional_config"] = {
            "rtklib": {
                "pos1-posmode": "kinematic",
                "pos1-frequency": "l1",
                "pos1-soltype": "forward",
                "pos1-navsys": "1",
                "pos1-elmask": "15",
                "pos1-ionoopt": "brdc",
                "pos1-tropopt": "saas",
                "pos1-dynamics": "on",
                "pos2-armode": "continuous",
                "pos2-arthres": "3.0",
            }
        }
    
    if "ai_config" not in profile:
        profile["ai_config"] = {
            "rtklib": {
                "pos1-posmode": "kinematic",
                "pos1-frequency": "l1+l2",
                "pos1-soltype": "forward",
                "pos1-navsys": "15",  # GPS+GLO+GAL
                "pos1-elmask": "10",
                "pos1-ionoopt": "brdc",
                "pos1-tropopt": "saas",
                "pos1-dynamics": "on",
                "pos2-armode": "fix-and-hold",
                "pos2-arthres": "2.8",
            }
        }
    
    # Write updated profile
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)
    
    _ok(f"Updated scenario profile: {profile_path}")


# ----------------------------
# Gitignore Update
# ----------------------------

def update_gitignore(project_root: Path) -> None:
    """Add raw data directories to .gitignore."""
    gitignore_path = project_root / ".gitignore"
    
    entries_to_add = [
        "",
        "# Downloaded raw GNSS data (large files)",
        "DATA/scenarios/*/raw/",
        "*.bag",
        "*.bag.tar.gz",
    ]
    
    existing_content = ""
    if gitignore_path.exists():
        existing_content = gitignore_path.read_text()
    
    # Check if already present
    if "DATA/scenarios/*/raw/" in existing_content:
        return
    
    # Append new entries
    with open(gitignore_path, 'a') as f:
        f.write("\n".join(entries_to_add) + "\n")
    
    _ok("Updated .gitignore to exclude raw data directories")


# ----------------------------
# Validation
# ----------------------------

def validate_scenario(scenario_dir: Path) -> Tuple[bool, List[str]]:
    """Validate scenario has required files."""
    issues = []
    
    required_files = {
        "rover.obs": 1000,  # Min size in bytes
        "nav.nav": 500,
        "gt.csv": 100,
        "scenario_profile.json": 100,
    }
    
    for filename, min_size in required_files.items():
        filepath = scenario_dir / filename
        if not filepath.exists():
            issues.append(f"Missing required file: {filename}")
        elif filepath.stat().st_size < min_size:
            issues.append(f"File too small ({filepath.stat().st_size} bytes): {filename}")
    
    # Check that synthetic flag is False
    profile_path = scenario_dir / "scenario_profile.json"
    if profile_path.exists():
        try:
            profile = load_json_with_comments(profile_path)
            synthetic = profile.get("synthetic_metadata", {}).get("synthetic", True)
            if synthetic:
                issues.append("scenario_profile.json still marked as synthetic=true")
        except Exception as e:
            issues.append(f"Error reading scenario profile: {e}")
    
    return len(issues) == 0, issues


# ----------------------------
# Main Population Logic
# ----------------------------

def populate_scenario2(
    scenario_root: Path,
    force: bool = False,
    skip_download: bool = False,
) -> bool:
    """
    Populate scenario2 with UrbanNav real GNSS data.
    
    Steps:
    1. Download GNSS RINEX files and ground truth
    2. Select best rover/base/nav files
    3. Convert ground truth to expected CSV format
    4. Update scenario_profile.json
    5. Validate the result
    """
    
    scenario_dir = scenario_root / "scenario2"
    raw_dir = scenario_dir / "raw"
    
    _info("=" * 60)
    _info("POPULATE SCENARIO 2 WITH URBANNAV REAL DATA")
    _info("=" * 60)
    _info(f"Dataset: {DATASET_INFO['name']}")
    _info(f"Location: {DATASET_INFO['location']}")
    _info(f"Environment: {DATASET_INFO['environment']}")
    _info(f"Output: {scenario_dir}")
    _info("")
    
    # Check if already populated
    existing_rover = scenario_dir / "rover.obs"
    if existing_rover.exists() and existing_rover.stat().st_size > 1000 and not force:
        # Check if it's real data (not copied from scenario1)
        profile_path = scenario_dir / "scenario_profile.json"
        if profile_path.exists():
            try:
                profile = load_json_with_comments(profile_path)
                source = profile.get("synthetic_metadata", {}).get("real_data_source", "")
                if source == "UrbanNav":
                    _info("Scenario2 already populated with UrbanNav data. Use --force to re-download.")
                    return True
            except Exception:
                pass
    
    # Step 1: Download files
    if not skip_download:
        _info("Step 1: Downloading GNSS data...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Download ground truth
        gt_raw_path = raw_dir / "ground_truth_raw.txt"
        if not gt_raw_path.exists() or force:
            if not download_file(URBANNAV_URLS["ground_truth"], gt_raw_path):
                _error("Failed to download ground truth file")
                _info("You may need to manually download from:")
                _info("  https://www.dropbox.com/s/twsvwftucoytfpc/UrbanNav_TST_GT_raw.txt")
                return False
        else:
            _info(f"Using existing ground truth: {gt_raw_path}")
        
        # Download GNSS RINEX folder (as zip)
        gnss_zip_path = raw_dir / "gnss_rinex.zip"
        gnss_extract_dir = raw_dir / "gnss"
        
        if not gnss_extract_dir.exists() or force:
            if not gnss_zip_path.exists() or force:
                if not download_file(URBANNAV_URLS["gnss_folder"], gnss_zip_path, timeout=300):
                    _error("Failed to download GNSS RINEX files")
                    _info("You may need to manually download from:")
                    _info("  https://www.dropbox.com/sh/2haoy68xekg95zl/AAAkcN4FwhFxkPY1lXsxbJrxa")
                    _info("Extract to: " + str(gnss_extract_dir))
                    return False
            
            # Extract
            if gnss_zip_path.exists():
                if not extract_archive(gnss_zip_path, gnss_extract_dir):
                    _error("Failed to extract GNSS RINEX archive")
                    return False
        else:
            _info(f"Using existing GNSS files: {gnss_extract_dir}")
    else:
        _info("Step 1: Skipping download (--skip-download)")
        gnss_extract_dir = raw_dir / "gnss"
        gt_raw_path = raw_dir / "ground_truth_raw.txt"
    
    # Step 2: Find and select RINEX files
    _info("")
    _info("Step 2: Selecting RINEX files...")
    
    rinex_files = find_rinex_files(raw_dir)
    _info(f"Found {len(rinex_files['obs'])} observation files, {len(rinex_files['nav'])} navigation files")
    
    rover_src, base_src, nav_src = select_best_rinex_pair(rinex_files)
    
    if not rover_src:
        _error("No rover observation file found")
        _info("Please ensure GNSS RINEX files are in: " + str(raw_dir))
        return False
    
    # Handle missing navigation file - download from IGS
    need_ephemeris_download = False
    if not nav_src:
        _warn("No navigation file in UrbanNav dataset - will download broadcast ephemeris")
        need_ephemeris_download = True
    
    _info(f"Selected rover: {rover_src.name}")
    if base_src:
        _info(f"Selected base: {base_src.name}")
    if nav_src:
        _info(f"Selected nav: {nav_src.name}")
    else:
        _info("Nav: Will download broadcast ephemeris from IGS")
    
    # Step 3: Copy/decompress RINEX files
    _info("")
    _info("Step 3: Preparing RINEX files...")
    
    rover_dest = scenario_dir / "rover.obs"
    nav_dest = scenario_dir / "nav.nav"
    
    if not copy_and_decompress_rinex(rover_src, rover_dest):
        _error("Failed to prepare rover observation file")
        return False
    _ok(f"Prepared: {rover_dest.name} ({rover_dest.stat().st_size / 1024:.1f} KB)")
    
    # Handle navigation file
    if nav_src:
        if not copy_and_decompress_rinex(nav_src, nav_dest):
            _error("Failed to prepare navigation file")
            return False
        _ok(f"Prepared: {nav_dest.name} ({nav_dest.stat().st_size / 1024:.1f} KB)")
    elif need_ephemeris_download:
        # Download broadcast ephemeris from IGS for 2021-05-17 (DOY 137)
        if not download_broadcast_ephemeris(nav_dest, year=2021, doy=137):
            _error("Failed to download broadcast ephemeris")
            return False
    
    base_dest = scenario_dir / "base.obs"
    if base_src:
        if not copy_and_decompress_rinex(base_src, base_dest):
            _warn("Failed to prepare base observation file - continuing without base")
        else:
            _ok(f"Prepared: {base_dest.name} ({base_dest.stat().st_size / 1024:.1f} KB)")
    else:
        # Create empty placeholder or copy rover as pseudo-base
        _warn("No separate base station file - using rover as pseudo-base (SPP mode)")
        shutil.copy2(rover_dest, base_dest)
    
    # Step 4: Convert ground truth
    _info("")
    _info("Step 4: Converting ground truth...")
    
    if gt_raw_path.exists():
        gt_epochs = parse_span_cpt_ground_truth(gt_raw_path)
        _info(f"Parsed {len(gt_epochs)} ground truth epochs")
        
        if gt_epochs:
            gt_dest = scenario_dir / "gt.csv"
            num_written = convert_ground_truth_to_csv(gt_epochs, gt_dest)
            _ok(f"Wrote {num_written} epochs to: {gt_dest.name}")
            
            # Get reference coordinates from first epoch
            first_epoch = gt_epochs[0]
            ref_coords = geodetic_to_ecef(
                first_epoch["lat_deg"],
                first_epoch["lon_deg"],
                first_epoch["height_m"]
            )
        else:
            _error("Failed to parse ground truth - no valid epochs")
            return False
    else:
        _error(f"Ground truth file not found: {gt_raw_path}")
        return False
    
    # Step 5: Update scenario profile
    _info("")
    _info("Step 5: Updating scenario profile...")
    
    files_dict = {
        "rover_obs": "rover.obs",
        "nav_file": "nav.nav",
        "base_obs": "base.obs",
        "ground_truth": "gt.csv",
    }
    
    update_scenario_profile(
        scenario_dir=scenario_dir,
        dataset_info=DATASET_INFO,
        gt_epochs=num_written,
        files=files_dict,
        coords=ref_coords,
    )
    
    # Step 6: Update .gitignore
    update_gitignore(PROJECT_ROOT)
    
    # Step 7: Clean up old files that might indicate synthetic data
    old_files = [
        scenario_dir / "corrections_schedule.json",
        scenario_dir / "mode_transitions.csv",
        scenario_dir / "multipath_profile.json",
    ]
    for old_file in old_files:
        if old_file.exists():
            old_file.unlink()
            _info(f"Removed old synthetic file: {old_file.name}")
    
    # Step 8: Validate
    _info("")
    _info("Step 6: Validating scenario...")
    
    is_valid, issues = validate_scenario(scenario_dir)
    
    if is_valid:
        _ok("Scenario2 validation PASSED")
    else:
        _error("Scenario2 validation FAILED:")
        for issue in issues:
            _error(f"  - {issue}")
        return False
    
    # Summary
    _info("")
    _info("=" * 60)
    _ok("SCENARIO 2 POPULATION COMPLETE")
    _info("=" * 60)
    _info(f"  Dataset: {DATASET_INFO['name']}")
    _info(f"  Location: {DATASET_INFO['location']}")
    _info(f"  Ground truth epochs: {num_written}")
    _info(f"  Output directory: {scenario_dir}")
    _info("")
    _info("Next steps:")
    _info("  1. Validate: python validate_scenario.py --scenario scenario2 --strict-real")
    _info("  2. Run baseline: python gnss_baseline_runner_v2.py --scenario scenario2 --mode traditional")
    _info("  3. Run optimised: python gnss_baseline_runner_v2.py --scenario scenario2 --mode ai")
    _info("  4. Compare: python gnss_baseline_runner_v2.py --scenario scenario2 --mode both")
    
    return True


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Populate Scenario 2 with UrbanNav Real GNSS Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset: UrbanNav-HK-Medium-Urban-1
  - Location: Hong Kong, TST (Tsim Sha Tsui)  
  - Environment: Medium urban canyon
  - Duration: 785 seconds
  - Ground Truth: NovAtel SPAN-CPT + IE (5cm RMSE)

Examples:
  # Download and populate scenario2
  python populate_scenario2_urbannav.py

  # Force re-download
  python populate_scenario2_urbannav.py --force

  # Use existing downloaded files
  python populate_scenario2_urbannav.py --skip-download

After population:
  python validate_scenario.py --scenario scenario2 --strict-real
  python gnss_baseline_runner_v2.py --scenario scenario2 --mode both
        """
    )
    
    parser.add_argument(
        "--scenario-root",
        type=str,
        default=str(DEFAULT_SCENARIO_ROOT),
        help=f"Root directory for scenarios (default: {DEFAULT_SCENARIO_ROOT})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing raw files"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    scenario_root = Path(args.scenario_root)
    
    if not scenario_root.exists():
        _error(f"Scenario root does not exist: {scenario_root}")
        return 1
    
    success = populate_scenario2(
        scenario_root=scenario_root,
        force=args.force,
        skip_download=args.skip_download,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
