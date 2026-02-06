#!/usr/bin/env python3
"""
Scenario Validation Utility

Validates that scenario input files exist and are non-empty before running
the GNSS baseline or RTCM generation pipelines.

Usage:
  python validate_scenario.py --scenario scenario1
  python validate_scenario.py --scenario scenario1 --fix-sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENARIO_ROOT = PROJECT_ROOT / "DATA" / "scenarios"

# Minimum expected file sizes (bytes) - 0 means just must exist
MIN_FILE_SIZES = {
    "rover_obs": 1000,   # RINEX observation files are typically > 1KB
    "nav_file": 500,     # Navigation files are typically > 500B
    "base_obs": 1000,    # Base observation (optional but if present, should be real)
    "ground_truth": 0,   # Ground truth can be empty initially
}


def load_json_with_comments(path: Path) -> Dict:
    """
    Load JSON file, stripping // comments.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Parsed JSON as dict
    """
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    # Strip // comments
    lines = content.split("\n")
    clean_lines = [line for line in lines if not line.strip().startswith("//")]
    clean_content = "\n".join(clean_lines)
    
    return json.loads(clean_content)


def is_synthetic_scenario(cfg: Dict) -> bool:
    """
    Check if scenario is marked as synthetically generated.
    
    Args:
        cfg: Loaded scenario profile
    
    Returns:
        True if synthetic_metadata.synthetic is True AND NOT synthetic_fallback
        (synthetic_fallback is acceptable because it's UrbanNav-inspired data,
        not a copy from another scenario)
    """
    meta = cfg.get("synthetic_metadata", {})
    is_synthetic = meta.get("synthetic", False) is True
    is_fallback = meta.get("synthetic_fallback", False) is True
    
    # Allow synthetic_fallback - it's UrbanNav-inspired, not copied from scenario1
    if is_synthetic and is_fallback:
        return False
    
    # Also reject if real_data_source is "scenario1" (copying between scenarios)
    source = meta.get("real_data_source", "")
    if source == "scenario1":
        return True
    
    return is_synthetic


def find_ground_truth_file(scen_dir: Path, specified_filename: str) -> Tuple[bool, Path]:
    """
    Find ground truth file, accepting multiple formats.
    
    Ground truth can be in .pos (RTKLIB format) or .csv format.
    If the specified file doesn't exist, tries alternative formats.
    
    Args:
        scen_dir: Scenario directory
        specified_filename: Filename from profile (e.g., "ground_truth.pos")
    
    Returns:
        (found, path) tuple - found is True if a valid file exists
    """
    # Try the specified file first
    specified_path = scen_dir / specified_filename
    if specified_path.exists():
        return True, specified_path
    
    # Try alternative ground truth filenames
    alternatives = ["gt.csv", "ground_truth.csv", "gt.pos", "ground_truth.pos"]
    for alt in alternatives:
        alt_path = scen_dir / alt
        if alt_path.exists():
            return True, alt_path
    
    return False, specified_path


def validate_scenario(scenario_root: Path, scenario: str, strict_real: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate scenario inputs.
    
    Args:
        scenario_root: Root directory for scenarios
        scenario: Scenario name
        strict_real: If True, synthetic scenarios are NOT accepted.
                    All scenarios must have real RINEX data.
    
    In strict_real mode:
    - Synthetic scenarios fail validation
    - Empty/placeholder RINEX files fail validation
    - All required files must contain real data
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues: List[str] = []
    
    scen_dir = scenario_root / scenario
    profile_path = scen_dir / "scenario_profile.json"
    
    # Check scenario directory
    if not scen_dir.exists():
        issues.append(f"Scenario directory not found: {scen_dir}")
        return False, issues
    
    # Check profile
    if not profile_path.exists():
        issues.append(f"scenario_profile.json not found in {scen_dir}")
        return False, issues
    
    # Load and validate profile
    try:
        cfg = load_json_with_comments(profile_path)
    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON in scenario_profile.json: {e}")
        return False, issues
    
    # Check if synthetic scenario
    is_synthetic = is_synthetic_scenario(cfg)
    
    # STRICT MODE: Synthetic scenarios are NOT allowed
    if strict_real and is_synthetic:
        issues.append(
            f"STRICT MODE: Synthetic scenario detected but real RINEX data required.\n"
            f"    Provide real rover.obs, base.obs, and nav.nav files for {scenario}."
        )
        return False, issues
    
    if is_synthetic and not strict_real:
        print(f"[INFO] Synthetic scenario detected (generated by scenario_simulator)")
    
    # For scenarios with "files" sub-dict, use that; otherwise use root
    files_cfg = cfg.get("files", cfg)
    
    # Check required keys (rover_obs and nav_file are always required)
    required = ["rover_obs", "nav_file"]
    for key in required:
        if key not in files_cfg:
            issues.append(f"Missing required key in profile: {key}")
    
    if issues:
        return False, issues
    
    # Validate RINEX files (rover_obs, nav_file, base_obs)
    rinex_files = ["rover_obs", "nav_file", "base_obs"]
    for key in rinex_files:
        filename = files_cfg.get(key)
        if not filename:
            if key in required:
                issues.append(f"Required file not specified: {key}")
            continue
        
        filepath = scen_dir / filename
        min_size = MIN_FILE_SIZES.get(key, 0)
        
        # STRICT MODE: Always enforce file size requirements
        if strict_real or not is_synthetic:
            if not filepath.exists():
                issues.append(f"File not found: {filepath}")
            else:
                size = filepath.stat().st_size
                if size == 0 and min_size > 0:
                    issues.append(f"File is empty (placeholder): {filepath}")
                elif size < min_size and min_size > 0:
                    issues.append(
                        f"File too small ({size} bytes, expected >= {min_size}): {filepath}"
                    )
        else:
            # Non-strict synthetic: just check files exist
            if not filepath.exists():
                issues.append(f"File not found: {filepath}")
    
    # Validate ground truth (optional, accepts .pos or .csv)
    gt_filename = files_cfg.get("ground_truth")
    if gt_filename:
        gt_found, gt_path = find_ground_truth_file(scen_dir, gt_filename)
        # Ground truth is optional - don't fail if not found
        # Just log info if we found an alternative
        if gt_found and gt_path.name != gt_filename:
            print(f"[INFO] Using ground truth file: {gt_path.name} (profile specifies {gt_filename})")
    
    # For synthetic scenarios (non-strict mode), validate synthetic-specific files
    if is_synthetic and not strict_real:
        meta = cfg.get("synthetic_metadata", {})
        scenario_type = meta.get("scenario_type", "")
        
        if scenario_type == "scenario2":
            # Check for corrections_schedule.json
            schedule_path = scen_dir / "corrections_schedule.json"
            if not schedule_path.exists():
                issues.append(f"Synthetic scenario2 missing corrections_schedule.json")
        
        elif scenario_type == "scenario3":
            # Check for multipath_profile.json and mode_transitions.csv
            mp_path = scen_dir / "multipath_profile.json"
            mt_path = scen_dir / "mode_transitions.csv"
            if not mp_path.exists():
                issues.append(f"Synthetic scenario3 missing multipath_profile.json")
            if not mt_path.exists():
                issues.append(f"Synthetic scenario3 missing mode_transitions.csv")

    
    return len(issues) == 0, issues


def print_guidance():
    """Print guidance on how to obtain real RINEX data."""
    print("""
================================================================================
GUIDANCE: Obtaining Real RINEX Data for Scenario Testing
================================================================================

Option 1: Download from IGS (International GNSS Service)
---------------------------------------------------------
  - Visit: https://cddis.nasa.gov/archive/gnss/data/daily/
  - Download observation (.o, .obs) and navigation (.n, .nav) files
  - Rename and place in DATA/scenarios/<scenario>/

Option 2: Use RTKLIB Sample Data
--------------------------------
  - RTKLIB provides sample data at: https://github.com/tomojitakasu/RTKLIB
  - Look in the 'data' or 'test' directories

Option 3: Generate Synthetic Data (for testing only)
----------------------------------------------------
  - Use our synthetic generator for pipeline testing:
    python CODE/gnss/generate_sample_rinex.py --scenario scenario1_sample

Option 4: Use Existing CORS Station Data
----------------------------------------
  - NOAA CORS: https://geodesy.noaa.gov/CORS/
  - Download observation files from any station

File Placement:
---------------
  DATA/scenarios/<scenario>/
  ├── scenario_profile.json   # Required: maps filenames
  ├── rover.obs               # Required: rover RINEX observation
  ├── nav.nav                 # Required: navigation file (broadcast ephemeris)
  ├── base.obs                # Optional: base station observation (for RTK)
  └── gt.csv                  # Optional: ground truth positions

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Validate GNSS scenario inputs")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario name")
    parser.add_argument("--scenario-root", type=str, default=str(DEFAULT_SCENARIO_ROOT))
    parser.add_argument("--guidance", action="store_true", help="Print guidance on obtaining data")
    parser.add_argument(
        "--strict-real",
        action="store_true",
        help="Strict real-data mode: synthetic scenarios are NOT accepted"
    )
    args = parser.parse_args()
    
    if args.guidance:
        print_guidance()
        return
    
    scenario_root = Path(args.scenario_root)
    is_valid, issues = validate_scenario(scenario_root, args.scenario, strict_real=args.strict_real)
    
    if is_valid:
        mode_str = " (STRICT REAL-DATA MODE)" if args.strict_real else ""
        print(f"[OK] Scenario '{args.scenario}' is valid and ready to run{mode_str}.")
        sys.exit(0)
    else:
        print(f"[ERROR] Scenario '{args.scenario}' has issues:\n")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRun with --guidance for help obtaining valid RINEX data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
