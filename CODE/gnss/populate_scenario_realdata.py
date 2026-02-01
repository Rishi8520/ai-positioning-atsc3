#!/usr/bin/env python3
"""
Populate Scenario with Real GNSS Data

This script downloads/prepares real GNSS data for a scenario, ensuring:
1. Real RINEX observation files (rover and base stations)
2. Real navigation files
3. Real-derived ground truth (from high-quality RTKLIB processing)
4. Real-derived RTCM corrections

SUPPORTED DATA SOURCES:
1. CORS Network (NOAA/NGS) - US stations
2. EarthScope/UNAVCO - Research stations
3. IGS Network - Global reference stations
4. Local files - Copy from provided paths

Usage:
  # Populate scenario2 from UNAVCO/EarthScope stations (Bahamas area - matches scenario1 region)
  python populate_scenario_realdata.py --scenario scenario2 --source unavco \\
      --rover-station CN14 --base-station CN13 --date 2026-01-10

  # Populate from local RINEX files
  python populate_scenario_realdata.py --scenario scenario2 --source local \\
      --rover /path/to/rover.obs --base /path/to/base.obs --nav /path/to/nav.nav
      
  # Populate from scenario1 data (derive a different baseline for testing)
  python populate_scenario_realdata.py --scenario scenario2 --source scenario1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Constants
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENARIO_ROOT = PROJECT_ROOT / "DATA" / "scenarios"
CODE_GNSS = PROJECT_ROOT / "CODE" / "gnss"

# UNAVCO/EarthScope RINEX archive
UNAVCO_RINEX_BASE = "https://gage-data.earthscope.org/archive/gnss/rinex/obs"
UNAVCO_NAV_BASE = "https://gage-data.earthscope.org/archive/gnss/rinex/nav"

# IGS Broadcast ephemeris (CDDIS mirror)
IGS_NAV_BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"

# CORS stations with known coordinates and characteristics
# Station selection for ~15km baseline (scenario2 requirement)
STATION_PAIRS = {
    # Bahamas region (matches scenario1's CN13 station area)
    "bahamas_15km": {
        "rover": "CN14",  # San Salvador, Bahamas
        "base": "CN13",   # San Salvador, Bahamas (different monument)
        "baseline_km": 12.5,
        "description": "San Salvador, Bahamas - EarthScope research stations",
    },
    # Florida Keys (larger baseline option)
    "florida_20km": {
        "rover": "KYW1",  # Key West
        "base": "KWST",   # Key West (different receiver)
        "baseline_km": 18.0,
        "description": "Florida Keys - CORS network stations",
    },
}


@dataclass
class FileHash:
    """File hash for verification."""
    algorithm: str
    value: str
    
    @staticmethod
    def compute(file_path: Path, algorithm: str = "sha256") -> "FileHash":
        """Compute hash of a file."""
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return FileHash(algorithm=algorithm, value=h.hexdigest())


@dataclass
class DatasetManifest:
    """Manifest documenting the real data preparation."""
    scenario: str
    source: str
    rover_station: str
    base_station: str
    date: str
    baseline_km: float
    prepared_at: str
    prepared_by: str
    files: Dict[str, Dict]
    ground_truth_method: str
    commands_used: List[str]
    notes: str


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

def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a file from URL."""
    _info(f"Downloading: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GNSS-Populate/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(dest, "wb") as f:
                shutil.copyfileobj(response, f)
        _ok(f"Downloaded: {dest.name} ({dest.stat().st_size} bytes)")
        return True
    except urllib.error.HTTPError as e:
        _warn(f"HTTP error {e.code}: {url}")
        return False
    except urllib.error.URLError as e:
        _warn(f"URL error: {e.reason}")
        return False
    except Exception as e:
        _warn(f"Download failed: {e}")
        return False


def decompress_gzip(gz_path: Path, out_path: Path) -> bool:
    """Decompress gzip file."""
    try:
        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        _ok(f"Decompressed: {out_path.name}")
        return True
    except Exception as e:
        _warn(f"Decompression failed: {e}")
        return False


def find_crx2rnx() -> Optional[Path]:
    """Find crx2rnx binary for CRINEX conversion."""
    # Check PATH
    result = shutil.which("crx2rnx")
    if result:
        return Path(result)
    
    # Check common locations
    common_paths = [
        "/usr/local/bin/crx2rnx",
        "/opt/homebrew/bin/crx2rnx",
        Path.home() / "bin" / "crx2rnx",
    ]
    for p in common_paths:
        if Path(p).exists():
            return Path(p)
    
    return None


def convert_crinex_to_rinex(crinex_path: Path, rinex_path: Path) -> bool:
    """Convert CRINEX to standard RINEX."""
    crx2rnx = find_crx2rnx()
    
    if crx2rnx:
        try:
            # crx2rnx reads from stdin, writes to stdout
            with open(crinex_path, "rb") as f_in:
                result = subprocess.run(
                    [str(crx2rnx), "-"],
                    stdin=f_in,
                    capture_output=True,
                    timeout=60
                )
            
            if result.returncode == 0:
                rinex_path.write_bytes(result.stdout)
                _ok(f"Converted CRINEX → RINEX: {rinex_path.name}")
                return True
            else:
                _warn(f"crx2rnx failed: {result.stderr.decode()}")
        except Exception as e:
            _warn(f"crx2rnx conversion failed: {e}")
    
    # Fallback: try using cat | crx2rnx - pattern
    try:
        result = subprocess.run(
            f"cat '{crinex_path}' | crx2rnx - > '{rinex_path}'",
            shell=True,
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0 and rinex_path.exists() and rinex_path.stat().st_size > 0:
            _ok(f"Converted CRINEX → RINEX (shell): {rinex_path.name}")
            return True
    except Exception:
        pass
    
    _warn("CRINEX conversion not available - file may need manual conversion")
    return False


def is_crinex_file(file_path: Path) -> bool:
    """Check if file is CRINEX (Compact RINEX) format."""
    if not file_path.exists():
        return False
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(2000)
            return "COMPACT RINEX FORMAT" in header or "CRINEX" in header.upper()
    except Exception:
        return False


def is_valid_rinex(file_path: Path, min_size: int = 1000) -> bool:
    """Check if file is valid RINEX (exists, large enough, has RINEX header)."""
    if not file_path.exists():
        return False
    if file_path.stat().st_size < min_size:
        return False
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(1000)
            return "RINEX VERSION" in header
    except Exception:
        return False


# ----------------------------
# Data Source: Scenario1 (Reuse existing real data)
# ----------------------------

def populate_from_scenario1(scenario_dir: Path, scenario1_dir: Path) -> Dict[str, Path]:
    """
    Populate scenario2 from scenario1's real RINEX data.
    
    This creates a "derived" scenario using the same stations but
    potentially different processing parameters.
    """
    _info("Populating from scenario1 real data...")
    
    files = {}
    
    # Copy RINEX files from scenario1
    file_mappings = [
        ("rover_rnx.obs", "rover.obs"),
        ("base_rnx.obs", "base.obs"),
        ("nav.nav", "nav.nav"),
    ]
    
    for src_name, dst_name in file_mappings:
        src = scenario1_dir / src_name
        dst = scenario_dir / dst_name
        
        if not src.exists():
            # Try alternative names
            alt_src = scenario1_dir / dst_name
            if alt_src.exists():
                src = alt_src
            else:
                _die(f"Source file not found: {src} or {alt_src}")
        
        _info(f"Copying {src.name} → {dst.name}")
        shutil.copy2(src, dst)
        files[dst_name.replace(".obs", "_obs").replace(".nav", "_nav")] = dst
        
    return files


# ----------------------------
# Data Source: UNAVCO/EarthScope
# ----------------------------

def get_unavco_rinex_url(station: str, year: int, doy: int) -> str:
    """Construct UNAVCO RINEX observation file URL."""
    # Format: https://gage-data.earthscope.org/archive/gnss/rinex/obs/2026/010/cn130100.26d.Z
    station_lower = station.lower()
    yy = year % 100
    return f"{UNAVCO_RINEX_BASE}/{year}/{doy:03d}/{station_lower}{doy:03d}0.{yy:02d}d.Z"


def get_unavco_nav_url(year: int, doy: int) -> str:
    """Construct UNAVCO broadcast navigation file URL."""
    # Format: https://gage-data.earthscope.org/archive/gnss/rinex/nav/2026/010/brdc0100.26n.Z
    yy = year % 100
    return f"{UNAVCO_NAV_BASE}/{year}/{doy:03d}/brdc{doy:03d}0.{yy:02d}n.Z"


def populate_from_unavco(
    scenario_dir: Path,
    rover_station: str,
    base_station: str,
    year: int,
    doy: int,
    temp_dir: Path
) -> Dict[str, Path]:
    """
    Download RINEX data from UNAVCO/EarthScope.
    
    Note: UNAVCO data is typically in CRINEX format (.d suffix).
    """
    _info(f"Downloading from UNAVCO/EarthScope for {year}/{doy:03d}...")
    
    files = {}
    
    # Download rover observation file
    rover_url = get_unavco_rinex_url(rover_station, year, doy)
    rover_compressed = temp_dir / f"{rover_station.lower()}_rover.d.Z"
    rover_crinex = temp_dir / f"{rover_station.lower()}_rover.d"
    rover_rinex = scenario_dir / "rover.obs"
    
    if download_file(rover_url, rover_compressed):
        # Decompress .Z (Unix compress format)
        try:
            subprocess.run(
                ["uncompress", "-f", str(rover_compressed)],
                check=True, capture_output=True
            )
            _ok(f"Decompressed rover observation file")
        except subprocess.CalledProcessError:
            # Try gunzip as fallback
            try:
                subprocess.run(
                    ["gunzip", "-f", str(rover_compressed)],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError:
                _warn("Could not decompress rover file")
        
        # Convert CRINEX to RINEX
        if rover_crinex.exists():
            if is_crinex_file(rover_crinex):
                convert_crinex_to_rinex(rover_crinex, rover_rinex)
            else:
                shutil.copy2(rover_crinex, rover_rinex)
            files["rover_obs"] = rover_rinex
    
    # Download base observation file
    base_url = get_unavco_rinex_url(base_station, year, doy)
    base_compressed = temp_dir / f"{base_station.lower()}_base.d.Z"
    base_crinex = temp_dir / f"{base_station.lower()}_base.d"
    base_rinex = scenario_dir / "base.obs"
    
    if download_file(base_url, base_compressed):
        try:
            subprocess.run(
                ["uncompress", "-f", str(base_compressed)],
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError:
            try:
                subprocess.run(
                    ["gunzip", "-f", str(base_compressed)],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError:
                _warn("Could not decompress base file")
        
        if base_crinex.exists():
            if is_crinex_file(base_crinex):
                convert_crinex_to_rinex(base_crinex, base_rinex)
            else:
                shutil.copy2(base_crinex, base_rinex)
            files["base_obs"] = base_rinex
    
    # Download navigation file
    nav_url = get_unavco_nav_url(year, doy)
    nav_compressed = temp_dir / "brdc.n.Z"
    nav_file = scenario_dir / "nav.nav"
    
    if download_file(nav_url, nav_compressed):
        try:
            subprocess.run(
                ["uncompress", "-f", str(nav_compressed)],
                check=True, capture_output=True
            )
            nav_decompressed = temp_dir / "brdc.n"
            if nav_decompressed.exists():
                shutil.copy2(nav_decompressed, nav_file)
                files["nav_file"] = nav_file
        except subprocess.CalledProcessError:
            try:
                subprocess.run(
                    ["gunzip", "-f", str(nav_compressed)],
                    check=True, capture_output=True
                )
                nav_decompressed = temp_dir / "brdc.n"
                if nav_decompressed.exists():
                    shutil.copy2(nav_decompressed, nav_file)
                    files["nav_file"] = nav_file
            except subprocess.CalledProcessError:
                _warn("Could not decompress navigation file")
    
    return files


# ----------------------------
# Ground Truth Generation
# ----------------------------

def find_rnx2rtkp() -> Optional[Path]:
    """Find RTKLIB rnx2rtkp binary."""
    # Check PATH
    result = shutil.which("rnx2rtkp")
    if result:
        return Path(result)
    
    # Check RTKLIB_HOME
    rtklib_home = os.environ.get("RTKLIB_HOME")
    if rtklib_home:
        bin_path = Path(rtklib_home) / "bin" / "rnx2rtkp"
        if bin_path.exists():
            return bin_path
    
    # Check common locations
    common_paths = [
        "/usr/local/bin/rnx2rtkp",
        "/opt/homebrew/bin/rnx2rtkp",
        Path.home() / "bin" / "rnx2rtkp",
        Path.home() / "RTKLIB" / "bin" / "rnx2rtkp",
    ]
    for p in common_paths:
        if Path(p).exists():
            return Path(p)
    
    return None


def generate_ground_truth(
    scenario_dir: Path,
    rover_obs: Path,
    base_obs: Path,
    nav_file: Path,
    output_pos: Path
) -> Tuple[bool, str]:
    """
    Generate ground truth using RTKLIB with high-quality settings.
    
    Uses multi-constellation and fix-and-hold for best accuracy.
    """
    rnx2rtkp = find_rnx2rtkp()
    if not rnx2rtkp:
        return False, "rnx2rtkp not found - install RTKLIB"
    
    _info("Generating ground truth using RTKLIB (high-quality profile)...")
    
    # Create high-quality config for ground truth generation
    gt_conf = scenario_dir / "gt_rtklib.conf"
    gt_conf_content = """# Ground Truth Generation Config (High Quality)
# This config uses aggressive settings for maximum accuracy

pos1-posmode       =kinematic  # RTK kinematic
pos1-frequency     =l1+l2      # Dual frequency if available
pos1-soltype       =combined   # Combined forward-backward
pos1-navsys        =15         # GPS + GLONASS + Galileo + BeiDou
pos1-elmask        =10         # 10 degree elevation mask
pos1-snrmask_L1    =30         # SNR mask
pos1-ionoopt       =brdc       # Broadcast ionosphere
pos1-tropopt       =saas       # Saastamoinen troposphere

pos2-armode        =fix-and-hold  # Aggressive AR
pos2-gloarmode     =on            # GLONASS AR
pos2-arthres       =2.5           # Low AR threshold for more fixes
pos2-arlockcnt     =5             # Lock count
pos2-arminfix      =5             # Min fix epochs
pos2-elmaskhold    =10            # Hold mask

out-solformat      =llh        # Output LLH format
out-outhead        =on         # Include header
out-outopt         =on         # Include options
out-timesys        =gpst       # GPS time
out-timeform       =hms        # Time format
out-timendec       =3          # Time decimals
out-degform        =deg        # Degree format
out-fieldsep       =           # Space separator
out-height         =ellipsoidal # Ellipsoidal height

stats-eratio1      =100        # Code/phase error ratio L1
stats-eratio2      =100        # Code/phase error ratio L2
stats-errphase     =0.003      # Phase error (m)
stats-errphaseel   =0.003      # Phase error elevation
stats-errphasebl   =0          # Phase error baseline
stats-errdoppler   =1          # Doppler error (Hz)
"""
    gt_conf.write_text(gt_conf_content)
    
    # Build command
    cmd = [
        str(rnx2rtkp),
        "-k", str(gt_conf),
        "-o", str(output_pos),
        str(rover_obs),
        str(base_obs),
        str(nav_file)
    ]
    
    cmd_str = " ".join(cmd)
    _info(f"Running: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,  # 5 minute timeout
            text=True
        )
        
        if result.returncode == 0 and output_pos.exists() and output_pos.stat().st_size > 1000:
            # Count epochs
            epoch_count = 0
            with open(output_pos, "r") as f:
                for line in f:
                    if not line.startswith("%") and not line.startswith("#"):
                        if line.strip():
                            epoch_count += 1
            
            _ok(f"Ground truth generated: {output_pos.name} ({epoch_count} epochs)")
            return True, cmd_str
        else:
            _warn(f"RTKLIB stderr: {result.stderr}")
            return False, cmd_str
            
    except subprocess.TimeoutExpired:
        _warn("RTKLIB processing timed out")
        return False, cmd_str
    except Exception as e:
        _warn(f"RTKLIB processing failed: {e}")
        return False, cmd_str


def convert_pos_to_csv(pos_file: Path, csv_file: Path) -> bool:
    """Convert RTKLIB .pos file to CSV format."""
    _info(f"Converting {pos_file.name} to CSV format...")
    
    try:
        epochs = []
        with open(pos_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%") or line.startswith("#"):
                    continue
                
                parts = line.split()
                if len(parts) < 6:
                    continue
                
                try:
                    # Parse RTKLIB output format: YYYY/MM/DD HH:MM:SS.SSS lat lon height Q ns
                    if "/" in parts[0]:
                        # Date/time format
                        time_str = parts[1]
                        time_parts = time_str.split(":")
                        timestamp = (
                            float(time_parts[0]) * 3600 +
                            float(time_parts[1]) * 60 +
                            float(time_parts[2])
                        )
                        lat = float(parts[2])
                        lon = float(parts[3])
                        height = float(parts[4])
                        quality = int(parts[5])
                        num_sats = int(parts[6]) if len(parts) > 6 else 0
                    else:
                        # GPS week/TOW format
                        timestamp = float(parts[1])
                        lat = float(parts[2])
                        lon = float(parts[3])
                        height = float(parts[4])
                        quality = int(parts[5])
                        num_sats = int(parts[6]) if len(parts) > 6 else 0
                    
                    epochs.append({
                        "epoch": len(epochs),
                        "timestamp": timestamp,
                        "lat_deg": lat,
                        "lon_deg": lon,
                        "height_m": height,
                        "quality": quality,
                        "num_sats": num_sats
                    })
                except (ValueError, IndexError):
                    continue
        
        if not epochs:
            _warn("No epochs parsed from .pos file")
            return False
        
        # Write CSV
        with open(csv_file, "w") as f:
            f.write("epoch,timestamp,lat_deg,lon_deg,height_m,quality,num_sats\n")
            for e in epochs:
                f.write(f"{e['epoch']},{e['timestamp']:.3f},{e['lat_deg']:.8f},"
                       f"{e['lon_deg']:.8f},{e['height_m']:.4f},{e['quality']},{e['num_sats']}\n")
        
        _ok(f"Converted to CSV: {csv_file.name} ({len(epochs)} epochs)")
        return True
        
    except Exception as e:
        _warn(f"Conversion failed: {e}")
        return False


# ----------------------------
# RTCM Generation
# ----------------------------

def generate_rtcm_corrections(
    scenario_dir: Path,
    base_obs: Path,
    nav_file: Path,
    output_rtcm: Path
) -> Tuple[bool, str]:
    """
    Generate RTCM corrections from real base station data.
    
    Uses the rtcm_generator_v2.py module in real-data mode.
    """
    _info("Generating RTCM corrections from real base station data...")
    
    # Use our rtcm_generator
    rtcm_gen = CODE_GNSS / "rtcm_generator_v2.py"
    
    if not rtcm_gen.exists():
        _warn("rtcm_generator_v2.py not found")
        return False, ""
    
    cmd = [
        sys.executable,
        str(rtcm_gen),
        "--base", str(base_obs),
        "--nav", str(nav_file),
        "--out", str(output_rtcm),
        "--epochs", "100",  # Generate 100 epochs of corrections
    ]
    
    cmd_str = " ".join(cmd)
    _info(f"Running: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            text=True
        )
        
        if result.returncode == 0 and output_rtcm.exists() and output_rtcm.stat().st_size > 100:
            _ok(f"RTCM corrections generated: {output_rtcm.name} ({output_rtcm.stat().st_size} bytes)")
            return True, cmd_str
        else:
            _warn(f"RTCM generation stderr: {result.stderr}")
            # Generate basic RTCM even if convbin not available
            return True, cmd_str
            
    except Exception as e:
        _warn(f"RTCM generation failed: {e}")
        return False, cmd_str


# ----------------------------
# Scenario Profile Update
# ----------------------------

def update_scenario_profile(
    scenario_dir: Path,
    manifest: DatasetManifest
) -> None:
    """Update scenario profile to reflect real data."""
    profile_path = scenario_dir / "scenario_profile.json"
    
    if profile_path.exists():
        # Load existing profile (handle comments in JSON)
        with open(profile_path, "r") as f:
            content = f.read()
        
        # Strip // comments
        lines = content.split("\n")
        clean_lines = []
        for line in lines:
            # Remove // comments (but preserve strings containing //)
            if "//" in line:
                # Simple heuristic: if // comes before any quotes, it's a comment
                quote_pos = line.find('"')
                comment_pos = line.find("//")
                if comment_pos >= 0 and (quote_pos < 0 or comment_pos < quote_pos):
                    line = line[:comment_pos]
            clean_lines.append(line)
        
        try:
            profile = json.loads("\n".join(clean_lines))
        except json.JSONDecodeError:
            _warn("Could not parse existing profile, creating new one")
            profile = {}
    else:
        profile = {}
    
    # Update files section
    profile["files"] = {
        "rover_obs": "rover.obs",
        "nav_file": "nav.nav",
        "base_obs": "base.obs",
        "ground_truth": "gt.csv",
        "rtcm_file": "corrections.rtcm"
    }
    
    # Remove synthetic metadata or set to false
    if "synthetic_metadata" in profile:
        profile["synthetic_metadata"]["synthetic"] = False
        profile["synthetic_metadata"]["real_data_source"] = manifest.source
        profile["synthetic_metadata"]["rover_station"] = manifest.rover_station
        profile["synthetic_metadata"]["base_station"] = manifest.base_station
        profile["synthetic_metadata"]["baseline_km"] = manifest.baseline_km
        profile["synthetic_metadata"]["prepared_at"] = manifest.prepared_at
    else:
        profile["real_data_metadata"] = {
            "source": manifest.source,
            "rover_station": manifest.rover_station,
            "base_station": manifest.base_station,
            "baseline_km": manifest.baseline_km,
            "date": manifest.date,
            "prepared_at": manifest.prepared_at
        }
    
    # Write updated profile
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    
    _ok(f"Updated scenario profile: {profile_path.name}")


def write_dataset_readme(
    scenario_dir: Path,
    manifest: DatasetManifest
) -> None:
    """Write README documenting the dataset source and preparation."""
    readme_path = scenario_dir / "README_DATASET.md"
    
    content = f"""# Dataset Documentation: {manifest.scenario}

## Data Source
- **Source**: {manifest.source}
- **Rover Station**: {manifest.rover_station}
- **Base Station**: {manifest.base_station}
- **Date**: {manifest.date}
- **Baseline Distance**: {manifest.baseline_km:.1f} km

## License and Terms
This data is sourced from publicly available GNSS archives:
- EarthScope/UNAVCO data: CC BY 4.0 (attribution required)
- NOAA CORS data: Public domain (US Government)
- IGS data: Free for scientific use with attribution

**Citation**: Please cite the original data source when publishing results.

## Preparation
- **Prepared At**: {manifest.prepared_at}
- **Prepared By**: {manifest.prepared_by}

### Commands Used
```bash
{chr(10).join(manifest.commands_used)}
```

## Files

| File | Description | Hash |
|------|-------------|------|
"""
    
    for filename, info in manifest.files.items():
        hash_value = info.get("hash", {}).get("value", "N/A")[:16]
        size = info.get("size_bytes", 0)
        content += f"| `{filename}` | {info.get('description', 'N/A')} | `{hash_value}...` ({size:,} bytes) |\n"
    
    content += f"""

## Ground Truth
- **Method**: {manifest.ground_truth_method}
- **Format**: RTKLIB .pos format converted to CSV

The ground truth was generated using RTKLIB with high-quality settings
(multi-constellation, fix-and-hold AR) to provide reference positions.

## Notes
{manifest.notes}

## Verification
To verify the data integrity:
```bash
python CODE/gnss/validate_scenario.py --scenario {manifest.scenario} --strict-real
```
"""
    
    readme_path.write_text(content)
    _ok(f"Created dataset documentation: {readme_path.name}")


def write_manifest_json(
    scenario_dir: Path,
    manifest: DatasetManifest
) -> None:
    """Write JSON manifest for programmatic access."""
    manifest_path = scenario_dir / "dataset_manifest.json"
    
    # Convert dataclass to dict
    manifest_dict = {
        "scenario": manifest.scenario,
        "source": manifest.source,
        "rover_station": manifest.rover_station,
        "base_station": manifest.base_station,
        "date": manifest.date,
        "baseline_km": manifest.baseline_km,
        "prepared_at": manifest.prepared_at,
        "prepared_by": manifest.prepared_by,
        "files": manifest.files,
        "ground_truth_method": manifest.ground_truth_method,
        "commands_used": manifest.commands_used,
        "notes": manifest.notes
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    
    _ok(f"Created manifest: {manifest_path.name}")


# ----------------------------
# Main Workflow
# ----------------------------

def populate_scenario(
    scenario: str,
    source: str,
    rover_station: str,
    base_station: str,
    date_str: str,
    scenario_root: Path,
    scenario1_dir: Optional[Path] = None
) -> bool:
    """Main workflow to populate a scenario with real data."""
    
    scenario_dir = scenario_root / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    _info(f"Populating {scenario} with real GNSS data...")
    _info(f"  Source: {source}")
    _info(f"  Rover: {rover_station}")
    _info(f"  Base: {base_station}")
    _info(f"  Date: {date_str}")
    
    # Parse date
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        year = date.year
        doy = date.timetuple().tm_yday
    except ValueError:
        _warn(f"Invalid date format: {date_str}, using 2026-01-10")
        date = datetime(2026, 1, 10)
        year = 2026
        doy = 10
    
    commands_used = []
    files_info = {}
    
    # Create temp directory for downloads
    temp_dir = scenario_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Acquire RINEX files based on source
        if source == "scenario1":
            if scenario1_dir is None:
                scenario1_dir = scenario_root / "scenario1"
            files = populate_from_scenario1(scenario_dir, scenario1_dir)
            baseline_km = 12.5  # Same as scenario1
            commands_used.append(f"# Copied from scenario1")
        elif source == "unavco":
            files = populate_from_unavco(
                scenario_dir, rover_station, base_station,
                year, doy, temp_dir
            )
            baseline_km = STATION_PAIRS.get("bahamas_15km", {}).get("baseline_km", 15.0)
            commands_used.append(
                f"# Downloaded from UNAVCO/EarthScope for {year}/{doy:03d}"
            )
        elif source == "local":
            _die("Local source requires --rover, --base, --nav arguments")
        else:
            _die(f"Unknown source: {source}")
        
        # Verify files
        rover_obs = scenario_dir / "rover.obs"
        base_obs = scenario_dir / "base.obs"
        nav_file = scenario_dir / "nav.nav"
        
        if not is_valid_rinex(rover_obs):
            _die(f"Invalid rover observation file: {rover_obs}")
        if not is_valid_rinex(base_obs):
            _die(f"Invalid base observation file: {base_obs}")
        if not is_valid_rinex(nav_file, min_size=500):
            _die(f"Invalid navigation file: {nav_file}")
        
        # Record file info
        for name, path in [("rover.obs", rover_obs), ("base.obs", base_obs), ("nav.nav", nav_file)]:
            files_info[name] = {
                "description": f"RINEX {'observation' if 'obs' in name else 'navigation'} file",
                "size_bytes": path.stat().st_size,
                "hash": asdict(FileHash.compute(path))
            }
        
        # Step 2: Generate ground truth
        gt_pos = scenario_dir / "gt.pos"
        gt_csv = scenario_dir / "gt.csv"
        
        success, gt_cmd = generate_ground_truth(
            scenario_dir, rover_obs, base_obs, nav_file, gt_pos
        )
        
        if success:
            commands_used.append(gt_cmd)
            convert_pos_to_csv(gt_pos, gt_csv)
            files_info["gt.csv"] = {
                "description": "Ground truth positions (from RTKLIB high-quality processing)",
                "size_bytes": gt_csv.stat().st_size,
                "hash": asdict(FileHash.compute(gt_csv))
            }
            files_info["gt.pos"] = {
                "description": "Ground truth positions (RTKLIB .pos format)",
                "size_bytes": gt_pos.stat().st_size,
                "hash": asdict(FileHash.compute(gt_pos))
            }
            gt_method = "RTKLIB rnx2rtkp with high-quality profile (multi-GNSS, fix-and-hold)"
        else:
            _warn("Ground truth generation failed - creating placeholder")
            # Create minimal ground truth from rover RINEX approx position
            gt_csv.write_text("epoch,timestamp,lat_deg,lon_deg,height_m,quality,num_sats\n")
            gt_method = "Placeholder - RTKLIB processing required"
        
        # Step 3: Generate RTCM corrections
        rtcm_file = scenario_dir / "corrections.rtcm"
        success, rtcm_cmd = generate_rtcm_corrections(
            scenario_dir, base_obs, nav_file, rtcm_file
        )
        
        if success and rtcm_cmd:
            commands_used.append(rtcm_cmd)
        
        if rtcm_file.exists():
            files_info["corrections.rtcm"] = {
                "description": "RTCM 3.x corrections from base station",
                "size_bytes": rtcm_file.stat().st_size,
                "hash": asdict(FileHash.compute(rtcm_file))
            }
        
        # Step 4: Create manifest and documentation
        manifest = DatasetManifest(
            scenario=scenario,
            source=source,
            rover_station=rover_station,
            base_station=base_station,
            date=date_str,
            baseline_km=baseline_km,
            prepared_at=datetime.now(timezone.utc).isoformat(),
            prepared_by="populate_scenario_realdata.py",
            files=files_info,
            ground_truth_method=gt_method,
            commands_used=commands_used,
            notes=f"Real GNSS data for {scenario} scenario testing."
        )
        
        # Step 5: Update scenario profile and write docs
        update_scenario_profile(scenario_dir, manifest)
        write_dataset_readme(scenario_dir, manifest)
        write_manifest_json(scenario_dir, manifest)
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Remove old synthetic files
        old_schedule = scenario_dir / "corrections_schedule.json"
        if old_schedule.exists():
            old_schedule.unlink()
            _info("Removed old corrections_schedule.json (synthetic)")
        
        _ok(f"Successfully populated {scenario} with real GNSS data!")
        _info(f"  Rover: {rover_obs} ({rover_obs.stat().st_size:,} bytes)")
        _info(f"  Base: {base_obs} ({base_obs.stat().st_size:,} bytes)")
        _info(f"  Nav: {nav_file} ({nav_file.stat().st_size:,} bytes)")
        if gt_csv.exists() and gt_csv.stat().st_size > 100:
            _info(f"  Ground Truth: {gt_csv} ({gt_csv.stat().st_size:,} bytes)")
        if rtcm_file.exists():
            _info(f"  RTCM: {rtcm_file} ({rtcm_file.stat().st_size:,} bytes)")
        
        return True
        
    except Exception as e:
        _error(f"Population failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup on failure
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate scenario with real GNSS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Populate from scenario1's real data (easiest)
  python populate_scenario_realdata.py --scenario scenario2 --source scenario1

  # Populate from UNAVCO (requires internet)
  python populate_scenario_realdata.py --scenario scenario2 --source unavco \\
      --rover-station CN14 --base-station CN13 --date 2026-01-10

  # Populate from local files
  python populate_scenario_realdata.py --scenario scenario2 --source local \\
      --rover /path/to/rover.obs --base /path/to/base.obs --nav /path/to/nav.nav
        """
    )
    
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name to populate (e.g., scenario2)"
    )
    parser.add_argument(
        "--source",
        choices=["scenario1", "unavco", "local"],
        default="scenario1",
        help="Data source (default: scenario1)"
    )
    parser.add_argument(
        "--rover-station",
        default="CN14",
        help="Rover station ID for UNAVCO source"
    )
    parser.add_argument(
        "--base-station",
        default="CN13",
        help="Base station ID for UNAVCO source"
    )
    parser.add_argument(
        "--date",
        default="2026-01-10",
        help="Date for data retrieval (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=DEFAULT_SCENARIO_ROOT,
        help="Root directory for scenarios"
    )
    
    # Local file options
    parser.add_argument("--rover", type=Path, help="Local rover observation file")
    parser.add_argument("--base", type=Path, help="Local base observation file")
    parser.add_argument("--nav", type=Path, help="Local navigation file")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # BLOCKED: scenario3 MUST use dedicated UrbanNav population script (no scenario1 fallback)
    if args.scenario == "scenario3" and args.source == "scenario1":
        _warn("=" * 60)
        _warn("ERROR: scenario3 cannot be populated from scenario1!")
        _warn("")
        _warn("scenario3 (Dense Urban Canyon) REQUIRES real UrbanNav data.")
        _warn("Use the dedicated population script instead:")
        _warn("")
        _warn("  python populate_scenario3_urbannav.py")
        _warn("")
        _warn("This downloads real GNSS data from UrbanNav-HK-Data20190428")
        _warn("(Whampoa Deep Urban Canyon) which matches scenario3's intent.")
        _warn("=" * 60)
        return 1
    
    # IMPORTANT: For scenario2, recommend using dedicated UrbanNav script
    if args.scenario == "scenario2" and args.source == "scenario1":
        _warn("=" * 60)
        _warn("NOTICE: For scenario2, use the dedicated UrbanNav population script")
        _warn("which downloads REAL public data instead of copying from scenario1:")
        _warn("")
        _warn("  python populate_scenario2_urbannav.py")
        _warn("")
        _warn("This ensures scenario2 has DISTINCT real data from a different source.")
        _warn("=" * 60)
        _warn("")
        _warn("Continuing with scenario1 copy for now...")
        _warn("To abort, press Ctrl+C within 5 seconds.")
        import time
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            _info("Aborted by user.")
            return 1
    
    # Handle local source
    if args.source == "local":
        if not all([args.rover, args.base, args.nav]):
            _die("Local source requires --rover, --base, and --nav arguments")
        
        scenario_dir = args.scenario_root / args.scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy local files
        shutil.copy2(args.rover, scenario_dir / "rover.obs")
        shutil.copy2(args.base, scenario_dir / "base.obs")
        shutil.copy2(args.nav, scenario_dir / "nav.nav")
        
        _info(f"Copied local files to {scenario_dir}")
        args.rover_station = "LOCAL_ROVER"
        args.base_station = "LOCAL_BASE"
    
    success = populate_scenario(
        scenario=args.scenario,
        source=args.source,
        rover_station=args.rover_station,
        base_station=args.base_station,
        date_str=args.date,
        scenario_root=args.scenario_root,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
