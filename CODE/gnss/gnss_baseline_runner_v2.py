"""
GNSS Baseline Runner (RTKLIB rnx2rtkp wrapper)

PURPOSE:
Run baseline RTK/PPK positioning and compute metrics for Traditional vs AI-Native comparison.
This is the core evaluation tool for the three PoC scenarios.

MODIFICATIONS FROM ORIGINAL:
1. ✅ Traditional vs AI mode support (--mode flag)
2. ✅ Scenario-specific RTKLIB configuration loading
3. ✅ Ground truth comparison with HPE calculation
4. ✅ Enhanced metrics: convergence time, mode transitions, availability
5. ✅ Output structure for demo readiness
6. ✅ Comparison report generation

USAGE:
  # Run traditional baseline for scenario 1
  python gnss_baseline_runner.py --scenario scenario1 --mode traditional

  # Run AI-optimized configuration for scenario 1
  python gnss_baseline_runner.py --scenario scenario1 --mode ai

  # Run with explicit paths
  python gnss_baseline_runner.py --rover path/to/rover.obs --nav path/to/nav.nav \\
      --base path/to/base.obs --mode traditional

  # Run both modes and generate comparison
  python gnss_baseline_runner.py --scenario scenario1 --mode both

SCENARIO PROFILE SCHEMA:
  The scenario_profile.json should contain:
  - files: rover_obs, nav_file, base_obs, ground_truth
  - station_coordinates: x_ecef, y_ecef, z_ecef
  - traditional_config: rtklib settings for baseline
  - ai_config: rtklib settings for AI-optimized mode
  - evaluation: accuracy_target_cm, availability_target_pct, convergence_target_sec

OUTPUT STRUCTURE:
  OUTPUTs/<scenario>/<mode>/run_<timestamp>/
    - solution.pos: RTKLIB position output
    - solution.csv: Parsed positions
    - metrics.json: Computed metrics (fix%, convergence, etc.)
    - accuracy_report.json: HPE against ground truth
    - mode_transitions.csv: FIX/FLOAT/SINGLE transitions
    - config_used.conf: RTKLIB config for reproducibility
    - manifest.json: Run metadata

INTEGRATION NOTES:
- Phase 0/1: Post-processing with base+rover RINEX files
- Phase 2: Integration with broadcast pipeline (RTCM flow verification)
- Reads scenario profiles from: DATA/scenarios/<scenario>/scenario_profile.json
- Writes results to: OUTPUTs/<scenario>/<mode>/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Constants / Defaults
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENARIO_ROOT = PROJECT_ROOT / "DATA" / "scenarios"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "OUTPUTs"

# Override with environment variables if set
if os.getenv("SCENARIO_ROOT"):
    DEFAULT_SCENARIO_ROOT = Path(os.getenv("SCENARIO_ROOT"))
if os.getenv("RESULTS_ROOT"):
    DEFAULT_RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT"))

RTKLIB_BINARIES = ["rnx2rtkp"]

# RTKLIB solution quality codes (from RTKLIB manual):
# 1=FIX, 2=FLOAT, 3=SBAS, 4=DGPS, 5=SINGLE, 6=PPP
Q_FIX = 1
Q_FLOAT = 2
Q_SBAS = 3
Q_DGPS = 4
Q_SINGLE = 5
Q_PPP = 6

QUALITY_NAMES = {
    Q_FIX: "FIX",
    Q_FLOAT: "FLOAT",
    Q_SBAS: "SBAS",
    Q_DGPS: "DGPS",
    Q_SINGLE: "SINGLE",
    Q_PPP: "PPP",
}

# Mode types for Traditional vs AI comparison
MODE_TRADITIONAL = "traditional"
MODE_AI = "ai"
MODE_BOTH = "both"

# Earth parameters for coordinate conversion (WGS84)
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared


# ----------------------------
# Data Structures
# ----------------------------


@dataclass
class ScenarioConfig:
    """Configuration loaded from scenario profile."""

    name: str
    description: str = ""
    intent: str = ""
    scenario_type: str = ""

    # File paths
    rover_obs: Optional[Path] = None
    nav_file: Optional[Path] = None
    base_obs: Optional[Path] = None
    ground_truth: Optional[Path] = None

    # Station coordinates (ECEF)
    station_x: Optional[float] = None
    station_y: Optional[float] = None
    station_z: Optional[float] = None

    # RTKLIB configuration for each mode
    traditional_rtklib: Dict[str, str] = field(default_factory=dict)
    ai_rtklib: Dict[str, str] = field(default_factory=dict)

    # Broadcast configuration (for reference/logging)
    traditional_broadcast: Dict[str, Any] = field(default_factory=dict)
    ai_broadcast: Dict[str, Any] = field(default_factory=dict)

    # Evaluation targets
    accuracy_target_cm: float = 10.0
    availability_target_pct: float = 95.0
    convergence_target_sec: float = 30.0


@dataclass
class PositionEpoch:
    """Single epoch position solution."""

    timestamp: float  # GPS time or seconds of day
    lat: float  # Latitude (degrees)
    lon: float  # Longitude (degrees)
    height: float  # Ellipsoidal height (m)
    quality: int  # Solution quality (1=FIX, 2=FLOAT, 5=SINGLE)
    num_sats: int  # Number of satellites used
    sdn: float = 0.0  # North std dev (m)
    sde: float = 0.0  # East std dev (m)
    sdu: float = 0.0  # Up std dev (m)
    age: float = 0.0  # Age of differential (s)
    ratio: float = 0.0  # AR ratio

    @property
    def quality_name(self) -> str:
        return QUALITY_NAMES.get(self.quality, f"Q{self.quality}")


@dataclass
class ModeTransition:
    """Record of a solution mode transition."""

    epoch_index: int
    timestamp: float
    from_mode: str
    to_mode: str
    duration_in_previous: float  # Seconds spent in previous mode


@dataclass
class AccuracyMetrics:
    """Accuracy metrics against ground truth."""

    num_epochs_compared: int = 0
    mean_hpe_m: float = 0.0  # Mean Horizontal Position Error
    rms_hpe_m: float = 0.0  # RMS Horizontal Position Error
    max_hpe_m: float = 0.0  # Maximum HPE
    pct95_hpe_m: float = 0.0  # 95th percentile HPE
    mean_vpe_m: float = 0.0  # Mean Vertical Position Error
    rms_vpe_m: float = 0.0  # RMS VPE
    pct_within_target: float = 0.0  # Percentage within target accuracy


@dataclass
class RunMetrics:
    """Comprehensive metrics for a single run."""

    # Basic counts
    total_epochs: int = 0
    fix_epochs: int = 0
    float_epochs: int = 0
    single_epochs: int = 0
    other_epochs: int = 0

    # Percentages
    fix_pct: float = 0.0
    float_pct: float = 0.0
    single_pct: float = 0.0
    availability_pct: float = 0.0  # FIX + FLOAT

    # Timing
    total_duration_sec: float = 0.0
    mean_convergence_sec: float = 0.0
    max_convergence_sec: float = 0.0
    num_convergences: int = 0

    # Mode stability
    num_mode_transitions: int = 0
    num_fix_losses: int = 0  # Transitions from FIX to non-FIX

    # Accuracy (if ground truth available)
    accuracy: Optional[AccuracyMetrics] = None

    # Satellite statistics
    mean_num_sats: float = 0.0
    min_num_sats: int = 0
    max_num_sats: int = 0


@dataclass
class RunManifest:
    """Metadata for a single run."""

    run_id: str
    scenario_name: str
    mode: str  # "traditional" or "ai"
    intent: str
    timestamp: str
    rtklib_version: str = ""
    config_file: str = ""
    rover_file: str = ""
    base_file: str = ""
    nav_file: str = ""
    ground_truth_file: str = ""
    success: bool = False
    error_message: str = ""
    duration_sec: float = 0.0


# ----------------------------
# Utility Functions
# ----------------------------


def _now_run_id(prefix: str = "run") -> str:
    """Generate timestamped run ID."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _die(msg: str, code: int = 2) -> None:
    """Print error message and exit."""
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def _warn(msg: str) -> None:
    """Print warning message."""
    print(f"[WARN] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    """Print info message."""
    print(f"[INFO] {msg}")


def _ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    """Write object to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def resolve_rtklib_binary(bin_name: str) -> Path:
    """
    Find RTKLIB binary in system PATH or RTKLIB_HOME.

    Search order:
    1. RTKLIB_HOME environment variable
    2. System PATH
    3. Common installation locations
    """
    # Check RTKLIB_HOME
    rtklib_home = os.getenv("RTKLIB_HOME")
    if rtklib_home:
        candidates = [
            Path(rtklib_home) / "app" / bin_name / "gcc" / bin_name,
            Path(rtklib_home) / "bin" / bin_name,
            Path(rtklib_home) / bin_name,
        ]
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate

    # Check system PATH
    result = shutil.which(bin_name)
    if result:
        return Path(result)

    # Check common locations
    common_paths = [
        Path.home() / "tools" / "RTKLIB" / "app" / bin_name / "gcc" / bin_name,
        Path("/usr/local/bin") / bin_name,
        Path("/usr/bin") / bin_name,
    ]
    for p in common_paths:
        if p.exists() and os.access(p, os.X_OK):
            return p

    _die(
        f"Could not find RTKLIB binary '{bin_name}'. "
        f"Set RTKLIB_HOME or add to PATH."
    )
    return Path()  # Never reached


def get_rtklib_version(bin_path: Path) -> str:
    """Get RTKLIB version string."""
    try:
        result = subprocess.run(
            [str(bin_path), "-h"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse version from help output
        for line in result.stderr.split("\n") + result.stdout.split("\n"):
            if "ver." in line.lower() or "version" in line.lower():
                return line.strip()
        return "unknown"
    except Exception:
        return "unknown"


# ----------------------------
# Coordinate Conversion
# ----------------------------


def ecef_to_llh(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to LLH (Latitude, Longitude, Height).

    Uses iterative algorithm for WGS84 ellipsoid.

    Returns:
        Tuple of (latitude_deg, longitude_deg, height_m)
    """
    lon = math.atan2(y, x)

    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - WGS84_E2))

    # Iterate to converge on latitude
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)
        lat_new = math.atan2(z + WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)

    if abs(math.cos(lat)) > 1e-10:
        height = p / math.cos(lat) - N
    else:
        height = abs(z) - WGS84_A * math.sqrt(1 - WGS84_E2)

    return (math.degrees(lat), math.degrees(lon), height)


def llh_to_enu(
    lat: float, lon: float, height: float, ref_lat: float, ref_lon: float, ref_height: float
) -> Tuple[float, float, float]:
    """
    Convert LLH to local ENU (East, North, Up) relative to reference point.

    Args:
        lat, lon, height: Point coordinates (degrees, degrees, meters)
        ref_lat, ref_lon, ref_height: Reference point coordinates

    Returns:
        Tuple of (east_m, north_m, up_m)
    """
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    # Approximate local radius
    sin_ref_lat = math.sin(ref_lat_rad)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_ref_lat**2)
    M = WGS84_A * (1 - WGS84_E2) / ((1 - WGS84_E2 * sin_ref_lat**2) ** 1.5)

    # Differences
    dlat = lat_rad - ref_lat_rad
    dlon = lon_rad - ref_lon_rad
    dh = height - ref_height

    # ENU coordinates
    east = dlon * (N + ref_height) * math.cos(ref_lat_rad)
    north = dlat * (M + ref_height)
    up = dh

    return (east, north, up)


# ----------------------------
# Scenario Loading
# ----------------------------


def load_scenario_config(
    scenario_name: str,
    scenario_root: Path = DEFAULT_SCENARIO_ROOT,
    mode: str = MODE_TRADITIONAL,
) -> ScenarioConfig:
    """
    Load scenario configuration from profile JSON.

    Args:
        scenario_name: Name of the scenario directory
        scenario_root: Root directory containing scenarios
        mode: "traditional" or "ai" - determines which config to use

    Returns:
        ScenarioConfig with all settings loaded
    """
    scenario_dir = scenario_root / scenario_name
    profile_path = scenario_dir / "scenario_profile.json"

    if not profile_path.exists():
        _die(f"Scenario profile not found: {profile_path}")

    profile = _load_json(profile_path)

    # Build configuration
    config = ScenarioConfig(
        name=profile.get("name", scenario_name),
        description=profile.get("description", ""),
        intent=profile.get("intent", ""),
        scenario_type=profile.get("scenario_type", ""),
    )

    # Load file paths
    files = profile.get("files", profile)  # Support both nested and flat structure
    if files.get("rover_obs"):
        config.rover_obs = scenario_dir / files["rover_obs"]
    if files.get("nav_file"):
        config.nav_file = scenario_dir / files["nav_file"]
    if files.get("base_obs"):
        config.base_obs = scenario_dir / files["base_obs"]
    if files.get("ground_truth"):
        config.ground_truth = scenario_dir / files["ground_truth"]

    # Load station coordinates
    station = profile.get("station_coordinates", {})
    config.station_x = station.get("x_ecef") or station.get("x")
    config.station_y = station.get("y_ecef") or station.get("y")
    config.station_z = station.get("z_ecef") or station.get("z")

    # Load RTKLIB configurations
    trad_config = profile.get("traditional_config", {})
    ai_config = profile.get("ai_config", {})

    config.traditional_rtklib = trad_config.get("rtklib", {})
    config.ai_rtklib = ai_config.get("rtklib", {})

    config.traditional_broadcast = trad_config.get("broadcast", {})
    config.ai_broadcast = ai_config.get("broadcast", {})

    # Load evaluation targets
    evaluation = profile.get("evaluation", {})
    config.accuracy_target_cm = evaluation.get("accuracy_target_cm", 10.0)
    config.availability_target_pct = evaluation.get("availability_target_pct", 95.0)
    config.convergence_target_sec = evaluation.get("convergence_target_sec", 30.0)

    return config


def load_explicit_inputs(args: argparse.Namespace) -> ScenarioConfig:
    """
    Create ScenarioConfig from explicit command-line arguments.
    """
    config = ScenarioConfig(name="explicit")

    if args.rover:
        config.rover_obs = Path(args.rover).expanduser().resolve()
    if args.nav:
        config.nav_file = Path(args.nav).expanduser().resolve()
    if args.base:
        config.base_obs = Path(args.base).expanduser().resolve()
    if args.ground_truth:
        config.ground_truth = Path(args.ground_truth).expanduser().resolve()

    return config


# ----------------------------
# RTKLIB Configuration
# ----------------------------


def build_rtklib_conf(
    out_path: Path,
    use_base: bool,
    rtklib_settings: Dict[str, str],
    station_coords: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Generate RTKLIB configuration file from settings dictionary.

    Args:
        out_path: Path to write config file
        use_base: Whether base station is used (differential mode)
        rtklib_settings: Dictionary of RTKLIB settings
        station_coords: Optional (x, y, z) ECEF coordinates for base
    """
    # Default settings (can be overridden by rtklib_settings)
    defaults = {
        # Positioning mode: 0=single, 1=dgps, 2=kinematic, 3=static, 4=moving-base
        "pos1-posmode": "kinematic" if use_base else "single",
        "pos1-frequency": "l1",  # l1, l1+l2, l1+l2+l5
        "pos1-soltype": "forward",  # forward, backward, combined
        "pos1-elmask": "15",  # Elevation mask (degrees)
        "pos1-snrmask_r": "off",
        "pos1-snrmask_b": "off",
        "pos1-dynamics": "on" if use_base else "off",
        "pos1-tidecorr": "off",
        "pos1-ionoopt": "brdc",  # off, brdc, sbas, dual-freq, est-stec
        "pos1-tropopt": "saas",  # off, saas, sbas, est-ztd
        "pos1-sateph": "brdc",  # brdc, precise, brdc+sbas
        "pos1-posopt1": "off",  # Satellite PCV
        "pos1-posopt2": "off",  # Receiver PCV
        "pos1-posopt3": "off",  # Phase windup
        "pos1-posopt4": "off",  # Reject eclipsing
        "pos1-posopt5": "off",  # RAIM FDE
        "pos1-posopt6": "off",  # Handle day boundary
        "pos1-exclsats": "",
        "pos1-navsys": "1",  # 1=GPS, 2=SBAS, 4=GLO, 8=GAL, 16=QZS, 32=BDS
        # Ambiguity resolution
        "pos2-armode": "continuous" if use_base else "off",  # off, continuous, instantaneous, fix-and-hold
        "pos2-gloarmode": "off",
        "pos2-bdsarmode": "off",
        "pos2-arthres": "3.0",  # AR validation threshold
        "pos2-arthres1": "0.9999",
        "pos2-arthres2": "0.25",
        "pos2-arthres3": "0.1",
        "pos2-arthres4": "0.05",
        "pos2-arlockcnt": "0",
        "pos2-arelmask": "0",
        "pos2-arminfix": "10",
        "pos2-armaxiter": "1",
        "pos2-elmaskhold": "0",
        "pos2-aroutcnt": "5",
        "pos2-maxage": "30",
        "pos2-syncsol": "off",
        "pos2-slipthres": "0.05",
        "pos2-rejionno": "30",
        "pos2-rejgdop": "30",
        "pos2-niter": "1",
        "pos2-baselen": "0",
        "pos2-basesig": "0",
        # Output settings
        "out-solformat": "llh",  # llh, xyz, enu
        "out-outhead": "on",
        "out-outopt": "on",
        "out-outvel": "off",
        "out-timesys": "gpst",
        "out-timeform": "tow",
        "out-timendec": "3",
        "out-degform": "deg",
        "out-fieldsep": "",
        "out-outsingle": "off",
        "out-maxsolstd": "0",
        "out-height": "ellipsoidal",
        "out-geoid": "internal",
        "out-solstatic": "all",
        "out-nmeaintv1": "0",
        "out-nmeaintv2": "0",
        "out-outstat": "off",
        # Statistics
        "stats-eratio1": "100",
        "stats-eratio2": "100",
        "stats-errphase": "0.003",
        "stats-errphaseel": "0.003",
        "stats-errphasebl": "0",
        "stats-errdoppler": "1",
        "stats-stdbias": "30",
        "stats-stdiono": "0.03",
        "stats-stdtrop": "0.3",
        "stats-prnaccelh": "10",
        "stats-prnaccelv": "10",
        "stats-prnbias": "0.0001",
        "stats-prniono": "0.001",
        "stats-prntrop": "0.0001",
        "stats-prnpos": "0",
        "stats-clkstab": "5e-12",
        # Antenna
        "ant1-postype": "llh",
        "ant1-pos1": "0",
        "ant1-pos2": "0",
        "ant1-pos3": "0",
        "ant1-anttype": "",
        "ant1-antdele": "0",
        "ant1-antdeln": "0",
        "ant1-antdelu": "0",
        "ant2-postype": "llh",
        "ant2-pos1": "0",
        "ant2-pos2": "0",
        "ant2-pos3": "0",
        "ant2-anttype": "",
        "ant2-antdele": "0",
        "ant2-antdeln": "0",
        "ant2-antdelu": "0",
        "ant2-maxaveep": "0",
        "ant2-initrst": "off",
        # Misc
        "misc-timeinterp": "off",
        "misc-sbasatsel": "0",
        "misc-rnxopt1": "",
        "misc-rnxopt2": "",
        "misc-pppopt": "",
        # File paths (typically set externally)
        "file-satantfile": "",
        "file-rcvantfile": "",
        "file-staposfile": "",
        "file-geoidfile": "",
        "file-ionofile": "",
        "file-dcbfile": "",
        "file-eopfile": "",
        "file-blqfile": "",
        "file-tempdir": "",
        "file-geexefile": "",
        "file-solstatfile": "",
        "file-tracefile": "",
    }

    # Apply custom settings (override defaults)
    for key, value in rtklib_settings.items():
        # Normalize key format (accept both pos1-posmode and pos1_posmode)
        normalized_key = key.replace("_", "-")
        defaults[normalized_key] = str(value)

    # Set base station coordinates if provided
    if station_coords and use_base:
        x, y, z = station_coords
        lat, lon, height = ecef_to_llh(x, y, z)
        defaults["ant2-postype"] = "llh"
        defaults["ant2-pos1"] = f"{lat:.9f}"
        defaults["ant2-pos2"] = f"{lon:.9f}"
        defaults["ant2-pos3"] = f"{height:.4f}"

    # Write config file
    with open(out_path, "w", encoding="utf-8") as f:
        for key, value in sorted(defaults.items()):
            f.write(f"{key}={value}\n")

    _info(f"Generated RTKLIB config: {out_path}")


# ----------------------------
# Solution Parsing
# ----------------------------


def parse_rtk_pos_file(pos_path: Path) -> List[PositionEpoch]:
    """
    Parse RTKLIB .pos output file.

    Handles both LLH and XYZ output formats.

    Returns:
        List of PositionEpoch objects
    """
    if not pos_path.exists():
        _warn(f"Position file not found: {pos_path}")
        return []

    epochs = []

    with open(pos_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip header lines and comments
            if not line or line.startswith("%") or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                # Parse timestamp (could be various formats)
                # Common format: YYYY/MM/DD HH:MM:SS.SSS or GPS week/tow
                if "/" in parts[0]:
                    # Date/time format: YYYY/MM/DD HH:MM:SS.SSS
                    # Combine date and time parts
                    if len(parts) >= 6:
                        time_str = parts[1]
                        # Extract seconds of day as timestamp
                        time_parts = time_str.split(":")
                        timestamp = (
                            float(time_parts[0]) * 3600
                            + float(time_parts[1]) * 60
                            + float(time_parts[2])
                        )
                        lat = float(parts[2])
                        lon = float(parts[3])
                        height = float(parts[4])
                        quality = int(parts[5])
                        num_sats = int(parts[6]) if len(parts) > 6 else 0

                        # Optional fields
                        sdn = float(parts[7]) if len(parts) > 7 else 0.0
                        sde = float(parts[8]) if len(parts) > 8 else 0.0
                        sdu = float(parts[9]) if len(parts) > 9 else 0.0
                        age = float(parts[10]) if len(parts) > 10 else 0.0
                        ratio = float(parts[11]) if len(parts) > 11 else 0.0
                else:
                    # GPS week/tow format
                    timestamp = float(parts[1])  # TOW
                    lat = float(parts[2])
                    lon = float(parts[3])
                    height = float(parts[4])
                    quality = int(parts[5])
                    num_sats = int(parts[6]) if len(parts) > 6 else 0
                    sdn = float(parts[7]) if len(parts) > 7 else 0.0
                    sde = float(parts[8]) if len(parts) > 8 else 0.0
                    sdu = float(parts[9]) if len(parts) > 9 else 0.0
                    age = float(parts[10]) if len(parts) > 10 else 0.0
                    ratio = float(parts[11]) if len(parts) > 11 else 0.0

                epochs.append(
                    PositionEpoch(
                        timestamp=timestamp,
                        lat=lat,
                        lon=lon,
                        height=height,
                        quality=quality,
                        num_sats=num_sats,
                        sdn=sdn,
                        sde=sde,
                        sdu=sdu,
                        age=age,
                        ratio=ratio,
                    )
                )

            except (ValueError, IndexError) as e:
                _warn(f"Could not parse line: {line[:50]}... ({e})")
                continue

    _info(f"Parsed {len(epochs)} epochs from {pos_path}")
    return epochs


def parse_ground_truth(gt_path: Path) -> List[PositionEpoch]:
    """
    Parse ground truth file. Supports:
    - CSV format: epoch,timestamp,lat_deg,lon_deg,height_m,quality,num_sats
    - RTKLIB .pos format (same as solution files)
    """
    import csv
    
    epochs = []
    
    # Check if file exists
    if not gt_path.exists():
        _warn(f"Ground truth file not found: {gt_path}")
        return []
    
    # First, check if it's a CSV file with header
    with open(gt_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    # Check for CSV header
    if first_line.startswith("epoch,") or "lat_deg" in first_line or "timestamp" in first_line:
        # Parse as CSV
        _info(f"Parsing ground truth as CSV: {gt_path}")
        with open(gt_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = float(row.get("timestamp", row.get("time", 0)))
                    lat = float(row.get("lat_deg", row.get("lat", row.get("latitude", 0))))
                    lon = float(row.get("lon_deg", row.get("lon", row.get("longitude", 0))))
                    height = float(row.get("height_m", row.get("height", row.get("alt", 0))))
                    quality = int(row.get("quality", row.get("fix_type", 1)))
                    num_sats = int(row.get("num_sats", row.get("nsv", 12)))
                    
                    epochs.append(
                        PositionEpoch(
                            timestamp=timestamp,
                            lat=lat,
                            lon=lon,
                            height=height,
                            quality=quality,
                            num_sats=num_sats,
                            sdn=0.0,
                            sde=0.0,
                            sdu=0.0,
                            age=0.0,
                            ratio=0.0,
                        )
                    )
                except (ValueError, KeyError) as e:
                    _warn(f"Could not parse CSV row: {row} ({e})")
                    continue
        
        _info(f"Parsed {len(epochs)} ground truth epochs from CSV")
        return epochs
    
    # Fall back to RTKLIB .pos format
    return parse_rtk_pos_file(gt_path)


# ----------------------------
# Metrics Computation
# ----------------------------


def compute_mode_transitions(epochs: List[PositionEpoch]) -> List[ModeTransition]:
    """
    Identify mode transitions in the solution.

    Returns:
        List of ModeTransition objects
    """
    if len(epochs) < 2:
        return []

    transitions = []
    prev_mode = epochs[0].quality_name
    prev_timestamp = epochs[0].timestamp
    mode_start_time = epochs[0].timestamp

    for i, epoch in enumerate(epochs[1:], start=1):
        curr_mode = epoch.quality_name
        if curr_mode != prev_mode:
            duration = epoch.timestamp - mode_start_time
            transitions.append(
                ModeTransition(
                    epoch_index=i,
                    timestamp=epoch.timestamp,
                    from_mode=prev_mode,
                    to_mode=curr_mode,
                    duration_in_previous=duration,
                )
            )
            prev_mode = curr_mode
            mode_start_time = epoch.timestamp

    return transitions


def compute_convergence_times(
    epochs: List[PositionEpoch], transitions: List[ModeTransition]
) -> List[float]:
    """
    Compute convergence times (time from non-FIX to FIX).

    Returns:
        List of convergence times in seconds
    """
    convergence_times = []

    # Track when we lose FIX
    in_non_fix = False
    non_fix_start = 0.0

    for epoch in epochs:
        if epoch.quality != Q_FIX:
            if not in_non_fix:
                in_non_fix = True
                non_fix_start = epoch.timestamp
        else:
            if in_non_fix:
                # Just converged to FIX
                convergence_time = epoch.timestamp - non_fix_start
                convergence_times.append(convergence_time)
                in_non_fix = False

    return convergence_times


def compute_accuracy_metrics(
    solution: List[PositionEpoch],
    ground_truth: List[PositionEpoch],
    target_accuracy_m: float = 0.10,
) -> AccuracyMetrics:
    """
    Compute accuracy metrics by comparing solution to ground truth.

    Uses timestamp matching (within 0.5s tolerance) to pair epochs.

    Args:
        solution: List of solution epochs
        ground_truth: List of ground truth epochs
        target_accuracy_m: Target accuracy in meters

    Returns:
        AccuracyMetrics object
    """
    if not solution or not ground_truth:
        return AccuracyMetrics()

    # Build lookup table for ground truth by timestamp
    gt_lookup = {int(e.timestamp * 2): e for e in ground_truth}  # 0.5s resolution

    hpe_values = []
    vpe_values = []

    for sol_epoch in solution:
        # Find matching ground truth epoch
        key = int(sol_epoch.timestamp * 2)
        gt_epoch = gt_lookup.get(key)

        if gt_epoch is None:
            # Try nearby timestamps
            for offset in [-1, 1, -2, 2]:
                gt_epoch = gt_lookup.get(key + offset)
                if gt_epoch:
                    break

        if gt_epoch is None:
            continue

        # Compute ENU errors
        east, north, up = llh_to_enu(
            sol_epoch.lat,
            sol_epoch.lon,
            sol_epoch.height,
            gt_epoch.lat,
            gt_epoch.lon,
            gt_epoch.height,
        )

        # Horizontal position error
        hpe = math.sqrt(east**2 + north**2)
        vpe = abs(up)

        hpe_values.append(hpe)
        vpe_values.append(vpe)

    if not hpe_values:
        return AccuracyMetrics()

    # Compute statistics
    import statistics

    hpe_sorted = sorted(hpe_values)
    pct95_idx = int(len(hpe_sorted) * 0.95)

    metrics = AccuracyMetrics(
        num_epochs_compared=len(hpe_values),
        mean_hpe_m=statistics.mean(hpe_values),
        rms_hpe_m=math.sqrt(sum(h**2 for h in hpe_values) / len(hpe_values)),
        max_hpe_m=max(hpe_values),
        pct95_hpe_m=hpe_sorted[pct95_idx] if pct95_idx < len(hpe_sorted) else hpe_sorted[-1],
        mean_vpe_m=statistics.mean(vpe_values),
        rms_vpe_m=math.sqrt(sum(v**2 for v in vpe_values) / len(vpe_values)),
        pct_within_target=sum(1 for h in hpe_values if h <= target_accuracy_m)
        / len(hpe_values)
        * 100,
    )

    return metrics


def compute_run_metrics(
    epochs: List[PositionEpoch],
    ground_truth: Optional[List[PositionEpoch]] = None,
    target_accuracy_m: float = 0.10,
) -> RunMetrics:
    """
    Compute comprehensive metrics for a single run.

    Args:
        epochs: List of solution epochs
        ground_truth: Optional list of ground truth epochs
        target_accuracy_m: Target accuracy in meters

    Returns:
        RunMetrics object
    """
    if not epochs:
        return RunMetrics()

    metrics = RunMetrics(total_epochs=len(epochs))

    # Count by quality
    for epoch in epochs:
        if epoch.quality == Q_FIX:
            metrics.fix_epochs += 1
        elif epoch.quality == Q_FLOAT:
            metrics.float_epochs += 1
        elif epoch.quality == Q_SINGLE:
            metrics.single_epochs += 1
        else:
            metrics.other_epochs += 1

    # Percentages
    metrics.fix_pct = metrics.fix_epochs / metrics.total_epochs * 100
    metrics.float_pct = metrics.float_epochs / metrics.total_epochs * 100
    metrics.single_pct = metrics.single_epochs / metrics.total_epochs * 100
    metrics.availability_pct = (metrics.fix_epochs + metrics.float_epochs) / metrics.total_epochs * 100

    # Duration
    if len(epochs) >= 2:
        metrics.total_duration_sec = epochs[-1].timestamp - epochs[0].timestamp

    # Mode transitions
    transitions = compute_mode_transitions(epochs)
    metrics.num_mode_transitions = len(transitions)
    metrics.num_fix_losses = sum(
        1 for t in transitions if t.from_mode == "FIX" and t.to_mode != "FIX"
    )

    # Convergence times
    convergence_times = compute_convergence_times(epochs, transitions)
    if convergence_times:
        metrics.num_convergences = len(convergence_times)
        metrics.mean_convergence_sec = sum(convergence_times) / len(convergence_times)
        metrics.max_convergence_sec = max(convergence_times)

    # Satellite statistics
    sat_counts = [e.num_sats for e in epochs if e.num_sats > 0]
    if sat_counts:
        metrics.mean_num_sats = sum(sat_counts) / len(sat_counts)
        metrics.min_num_sats = min(sat_counts)
        metrics.max_num_sats = max(sat_counts)

    # Accuracy metrics (if ground truth available)
    if ground_truth:
        metrics.accuracy = compute_accuracy_metrics(epochs, ground_truth, target_accuracy_m)

    return metrics


# ----------------------------
# Output Generation
# ----------------------------


def write_solution_csv(epochs: List[PositionEpoch], csv_path: Path) -> None:
    """Write solution epochs to CSV file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "lat_deg",
                "lon_deg",
                "height_m",
                "quality",
                "quality_name",
                "num_sats",
                "sdn_m",
                "sde_m",
                "sdu_m",
                "age_s",
                "ratio",
            ]
        )
        for e in epochs:
            writer.writerow(
                [
                    f"{e.timestamp:.3f}",
                    f"{e.lat:.9f}",
                    f"{e.lon:.9f}",
                    f"{e.height:.4f}",
                    e.quality,
                    e.quality_name,
                    e.num_sats,
                    f"{e.sdn:.4f}",
                    f"{e.sde:.4f}",
                    f"{e.sdu:.4f}",
                    f"{e.age:.1f}",
                    f"{e.ratio:.1f}",
                ]
            )
    _info(f"Wrote solution CSV: {csv_path}")


def write_transitions_csv(transitions: List[ModeTransition], csv_path: Path) -> None:
    """Write mode transitions to CSV file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch_index", "timestamp", "from_mode", "to_mode", "duration_in_previous_s"]
        )
        for t in transitions:
            writer.writerow(
                [
                    t.epoch_index,
                    f"{t.timestamp:.3f}",
                    t.from_mode,
                    t.to_mode,
                    f"{t.duration_in_previous:.3f}",
                ]
            )
    _info(f"Wrote transitions CSV: {csv_path}")


# ----------------------------
# Main Run Logic
# ----------------------------


def run_rtklib(
    config: ScenarioConfig,
    mode: str,
    output_dir: Path,
    timeout_sec: int = 300,
) -> Tuple[bool, RunManifest, RunMetrics]:
    """
    Run RTKLIB rnx2rtkp with the given configuration.

    Args:
        config: Scenario configuration
        mode: "traditional" or "ai"
        output_dir: Directory for output files
        timeout_sec: Timeout in seconds

    Returns:
        Tuple of (success, manifest, metrics)
    """
    run_id = _now_run_id()
    run_dir = output_dir / mode / run_id
    _ensure_dir(run_dir)

    # Initialize manifest
    manifest = RunManifest(
        run_id=run_id,
        scenario_name=config.name,
        mode=mode,
        intent=config.intent,
        timestamp=datetime.now().isoformat(),
    )

    # Find RTKLIB binary
    rnx2rtkp_bin = resolve_rtklib_binary("rnx2rtkp")
    manifest.rtklib_version = get_rtklib_version(rnx2rtkp_bin)

    # Validate input files
    if not config.rover_obs or not config.rover_obs.exists():
        manifest.error_message = f"Rover observation file not found: {config.rover_obs}"
        return False, manifest, RunMetrics()

    if not config.nav_file or not config.nav_file.exists():
        manifest.error_message = f"Navigation file not found: {config.nav_file}"
        return False, manifest, RunMetrics()

    manifest.rover_file = str(config.rover_obs)
    manifest.nav_file = str(config.nav_file)

    # Check for base station (differential mode)
    use_base = config.base_obs is not None and config.base_obs.exists()
    if use_base:
        manifest.base_file = str(config.base_obs)

    # Select RTKLIB settings based on mode
    # First, try to load from rtk_profiles.yaml for scenario-specific profiles
    try:
        from strict_preflight import load_rtk_profiles, get_profile_settings
        profiles = load_rtk_profiles()
        profile_settings = get_profile_settings(
            profiles, mode, config.intent, scenario_name=config.name
        )
        if profile_settings:
            _info(f"[PROFILE] Using {mode} profile from rtk_profiles.yaml for {config.name}")
            rtklib_settings = profile_settings
        else:
            # Fall back to scenario profile settings
            if mode == MODE_AI:
                rtklib_settings = config.ai_rtklib
            else:
                rtklib_settings = config.traditional_rtklib
    except ImportError:
        # Fall back to scenario profile settings
        if mode == MODE_AI:
            rtklib_settings = config.ai_rtklib
        else:
            rtklib_settings = config.traditional_rtklib

    # Get station coordinates
    station_coords = None
    if config.station_x and config.station_y and config.station_z:
        station_coords = (config.station_x, config.station_y, config.station_z)

    # Generate RTKLIB config file
    conf_file = run_dir / "rtklib.conf"
    build_rtklib_conf(conf_file, use_base, rtklib_settings, station_coords)
    manifest.config_file = str(conf_file)

    # Output file
    out_pos = run_dir / "solution.pos"

    # Build command
    cmd = [
        str(rnx2rtkp_bin),
        "-k",
        str(conf_file),
        "-o",
        str(out_pos),
        str(config.rover_obs),
    ]

    if use_base:
        cmd.append(str(config.base_obs))

    cmd.append(str(config.nav_file))

    _info(f"Running RTKLIB: {' '.join(cmd)}")

    # Execute
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        # Log output
        if result.stdout:
            with open(run_dir / "stdout.log", "w") as f:
                f.write(result.stdout)
        if result.stderr:
            with open(run_dir / "stderr.log", "w") as f:
                f.write(result.stderr)

        if result.returncode != 0:
            manifest.error_message = f"RTKLIB exited with code {result.returncode}"
            _warn(manifest.error_message)
            if result.stderr:
                _warn(f"STDERR: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        manifest.error_message = f"RTKLIB timed out after {timeout_sec}s"
        return False, manifest, RunMetrics()
    except Exception as e:
        manifest.error_message = f"Failed to run RTKLIB: {e}"
        return False, manifest, RunMetrics()

    end_time = datetime.now()
    manifest.duration_sec = (end_time - start_time).total_seconds()

    # Parse solution
    if not out_pos.exists():
        manifest.error_message = "RTKLIB did not produce output file"
        return False, manifest, RunMetrics()

    epochs = parse_rtk_pos_file(out_pos)
    if not epochs:
        manifest.error_message = "No epochs parsed from solution file"
        return False, manifest, RunMetrics()

    # Load ground truth if available
    ground_truth = None
    if config.ground_truth and config.ground_truth.exists():
        manifest.ground_truth_file = str(config.ground_truth)
        ground_truth = parse_ground_truth(config.ground_truth)
        _info(f"Loaded {len(ground_truth)} ground truth epochs")

    # Compute metrics
    target_m = config.accuracy_target_cm / 100.0  # Convert cm to m
    metrics = compute_run_metrics(epochs, ground_truth, target_m)

    # Write outputs
    write_solution_csv(epochs, run_dir / "solution.csv")

    transitions = compute_mode_transitions(epochs)
    write_transitions_csv(transitions, run_dir / "mode_transitions.csv")

    # Write metrics
    metrics_dict = asdict(metrics)
    if metrics.accuracy:
        metrics_dict["accuracy"] = asdict(metrics.accuracy)
    _write_json(run_dir / "metrics.json", metrics_dict)

    # Write manifest
    manifest.success = True
    _write_json(run_dir / "manifest.json", asdict(manifest))

    _info(f"Run complete: {run_dir}")
    _info(f"  Total epochs: {metrics.total_epochs}")
    _info(f"  FIX: {metrics.fix_pct:.1f}%, FLOAT: {metrics.float_pct:.1f}%, SINGLE: {metrics.single_pct:.1f}%")
    _info(f"  Availability: {metrics.availability_pct:.1f}%")
    if metrics.accuracy and metrics.accuracy.num_epochs_compared > 0:
        _info(f"  Mean HPE: {metrics.accuracy.mean_hpe_m*100:.2f} cm")
        _info(f"  95% HPE: {metrics.accuracy.pct95_hpe_m*100:.2f} cm")

    return True, manifest, metrics


def generate_comparison_report(
    scenario_name: str,
    traditional_metrics: Optional[RunMetrics],
    ai_metrics: Optional[RunMetrics],
    output_dir: Path,
) -> Dict:
    """
    Generate comparison report between Traditional and AI modes.

    Returns:
        Dictionary containing comparison data
    """
    report = {
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "traditional": None,
        "ai": None,
        "comparison": {},
    }

    if traditional_metrics:
        report["traditional"] = asdict(traditional_metrics)
        if traditional_metrics.accuracy:
            report["traditional"]["accuracy"] = asdict(traditional_metrics.accuracy)

    if ai_metrics:
        report["ai"] = asdict(ai_metrics)
        if ai_metrics.accuracy:
            report["ai"]["accuracy"] = asdict(ai_metrics.accuracy)

    # Compute comparison metrics
    if traditional_metrics and ai_metrics:
        comparison = report["comparison"]

        # Fix rate improvement
        comparison["fix_pct_improvement"] = ai_metrics.fix_pct - traditional_metrics.fix_pct

        # Availability improvement
        comparison["availability_improvement"] = (
            ai_metrics.availability_pct - traditional_metrics.availability_pct
        )

        # Convergence time improvement (negative = faster = better)
        if traditional_metrics.mean_convergence_sec > 0 and ai_metrics.mean_convergence_sec > 0:
            comparison["convergence_time_reduction_sec"] = (
                traditional_metrics.mean_convergence_sec - ai_metrics.mean_convergence_sec
            )
            comparison["convergence_time_reduction_pct"] = (
                (traditional_metrics.mean_convergence_sec - ai_metrics.mean_convergence_sec)
                / traditional_metrics.mean_convergence_sec
                * 100
            )

        # Accuracy improvement (if ground truth available)
        if traditional_metrics.accuracy and ai_metrics.accuracy:
            t_acc = traditional_metrics.accuracy
            a_acc = ai_metrics.accuracy
            if t_acc.mean_hpe_m > 0 and a_acc.mean_hpe_m > 0:
                comparison["hpe_improvement_m"] = t_acc.mean_hpe_m - a_acc.mean_hpe_m
                comparison["hpe_improvement_pct"] = (
                    (t_acc.mean_hpe_m - a_acc.mean_hpe_m) / t_acc.mean_hpe_m * 100
                )
                comparison["pct_within_target_improvement"] = (
                    a_acc.pct_within_target - t_acc.pct_within_target
                )

        # Mode stability
        comparison["fix_loss_reduction"] = (
            traditional_metrics.num_fix_losses - ai_metrics.num_fix_losses
        )

        # Summary verdict
        improvements = []
        if comparison.get("fix_pct_improvement", 0) > 1:
            improvements.append("higher FIX rate")
        if comparison.get("hpe_improvement_pct", 0) > 5:
            improvements.append("better accuracy")
        if comparison.get("convergence_time_reduction_pct", 0) > 10:
            improvements.append("faster convergence")
        if comparison.get("fix_loss_reduction", 0) > 0:
            improvements.append("fewer FIX losses")

        if improvements:
            comparison["summary"] = f"AI mode shows: {', '.join(improvements)}"
        else:
            comparison["summary"] = "No significant improvement observed"

    # Write report
    report_path = output_dir / "comparison_report.json"
    _write_json(report_path, report)
    _info(f"Wrote comparison report: {report_path}")

    return report


# ----------------------------
# Argument Parsing
# ----------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GNSS Baseline Runner - Traditional vs AI-Native Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run traditional baseline for scenario 1
  python gnss_baseline_runner.py --scenario scenario1 --mode traditional

  # Run AI-optimized configuration for scenario 1  
  python gnss_baseline_runner.py --scenario scenario1 --mode ai

  # Run both modes and generate comparison
  python gnss_baseline_runner.py --scenario scenario1 --mode both

  # Run with explicit file paths
  python gnss_baseline_runner.py --rover rover.obs --nav nav.nav --base base.obs --mode traditional
        """,
    )

    # Scenario-based input
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name (loads from DATA/scenarios/<name>/scenario_profile.json)",
    )

    # Explicit file inputs
    parser.add_argument("--rover", type=str, help="Path to rover observation file")
    parser.add_argument("--nav", type=str, help="Path to navigation file")
    parser.add_argument("--base", type=str, help="Path to base station observation file")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth file")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MODE_TRADITIONAL, MODE_AI, MODE_BOTH],
        default=MODE_TRADITIONAL,
        help="Run mode: traditional (baseline), ai (optimized), or both (comparison)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Output directory (default: {DEFAULT_RESULTS_ROOT})",
    )

    # Execution options
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Timeout for RTKLIB execution in seconds (default: 300)",
    )

    parser.add_argument(
        "--scenario-root",
        type=str,
        default=str(DEFAULT_SCENARIO_ROOT),
        help=f"Root directory for scenarios (default: {DEFAULT_SCENARIO_ROOT})",
    )

    # Strict mode options
    parser.add_argument(
        "--auto-crinex-convert",
        action="store_true",
        help="Automatically convert CRINEX files to standard RINEX using crx2rnx",
    )

    return parser.parse_args()


# ----------------------------
# Main Entry Point
# ----------------------------


def main() -> int:
    """Main entry point."""
    # Import strict preflight module
    from strict_preflight import (
        strict_preflight_check,
        print_preflight_errors,
        load_rtk_profiles,
        get_profile_settings,
    )
    
    args = parse_args()

    # Determine scenario configuration
    if args.scenario:
        scenario_root = Path(args.scenario_root).expanduser().resolve()
        config = load_scenario_config(args.scenario, scenario_root, args.mode)
        _info(f"Loaded scenario: {config.name}")
        _info(f"  Intent: {config.intent}")
        _info(f"  Description: {config.description}")
    elif args.rover and args.nav:
        config = load_explicit_inputs(args)
        _info("Using explicit file paths")
    else:
        _die("Must specify either --scenario or (--rover and --nav)")
        return 1

    # Override with explicit arguments if provided
    if args.rover:
        config.rover_obs = Path(args.rover).expanduser().resolve()
    if args.nav:
        config.nav_file = Path(args.nav).expanduser().resolve()
    if args.base:
        config.base_obs = Path(args.base).expanduser().resolve()
    if args.ground_truth:
        config.ground_truth = Path(args.ground_truth).expanduser().resolve()

    # Setup output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.scenario:
        output_dir = output_dir / args.scenario
    _ensure_dir(output_dir)

    # STRICT PREFLIGHT VALIDATION
    # This is the single source of truth - ensures real RINEX data only
    _info("=" * 60)
    _info("STRICT PREFLIGHT VALIDATION")
    _info("=" * 60)
    
    # Create temporary run dir for potential CRINEX conversion
    preflight_run_dir = output_dir / "preflight_temp"
    
    preflight_result = strict_preflight_check(
        rover_obs=config.rover_obs,
        nav_file=config.nav_file,
        base_obs=config.base_obs,
        auto_crinex_convert=args.auto_crinex_convert,
        run_dir=preflight_run_dir,
    )
    
    if not preflight_result.valid:
        print_preflight_errors(preflight_result, config.name)
        return 2  # Exit code 2 for validation failure
    
    # Update config with converted paths if CRINEX was converted
    if preflight_result.rover_path:
        config.rover_obs = preflight_result.rover_path
    if preflight_result.base_path:
        config.base_obs = preflight_result.base_path
    
    for warning in preflight_result.warnings:
        _info(f"[PREFLIGHT] {warning}")
    
    _info("[PREFLIGHT] Validation PASSED - all inputs are real RINEX data")

    # Load RTK profiles for profile-driven configuration
    profiles = load_rtk_profiles()
    if profiles:
        _info("[PROFILES] Loaded RTK profiles from rtk_profiles.yaml")

    # Run based on mode
    traditional_metrics = None
    ai_metrics = None

    if args.mode in [MODE_TRADITIONAL, MODE_BOTH]:
        _info("=" * 60)
        _info("Running TRADITIONAL mode (baseline)")
        _info("=" * 60)
        success, manifest, traditional_metrics = run_rtklib(
            config, MODE_TRADITIONAL, output_dir, args.timeout_sec
        )
        if not success:
            _warn(f"Traditional run failed: {manifest.error_message}")

    if args.mode in [MODE_AI, MODE_BOTH]:
        _info("=" * 60)
        _info("Running AI mode (intent-driven)")
        _info("=" * 60)
        success, manifest, ai_metrics = run_rtklib(
            config, MODE_AI, output_dir, args.timeout_sec
        )
        if not success:
            _warn(f"AI run failed: {manifest.error_message}")

    # Generate comparison report if both modes were run
    if args.mode == MODE_BOTH and traditional_metrics and ai_metrics:
        _info("=" * 60)
        _info("Generating comparison report")
        _info("=" * 60)
        report = generate_comparison_report(
            config.name, traditional_metrics, ai_metrics, output_dir
        )

        # Print summary
        if "comparison" in report and report["comparison"]:
            comp = report["comparison"]
            print("\n" + "=" * 60)
            print("COMPARISON SUMMARY: Traditional vs AI-Native")
            print("=" * 60)
            print(f"  Fix rate improvement:     {comp.get('fix_pct_improvement', 0):+.1f}%")
            print(f"  Availability improvement: {comp.get('availability_improvement', 0):+.1f}%")
            if "convergence_time_reduction_pct" in comp:
                print(f"  Convergence time:         {comp['convergence_time_reduction_pct']:+.1f}% faster")
            if "hpe_improvement_pct" in comp:
                print(f"  HPE improvement:          {comp['hpe_improvement_pct']:+.1f}%")
            print(f"\n  {comp.get('summary', 'N/A')}")
            print("=" * 60)

    _info(f"\nResults written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())