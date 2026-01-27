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

DEFAULT_SCENARIO_ROOT = Path("DATA/scenarios")

# Minimum expected file sizes (bytes) - 0 means just must exist
MIN_FILE_SIZES = {
    "rover_obs": 1000,   # RINEX observation files are typically > 1KB
    "nav_file": 500,     # Navigation files are typically > 500B
    "base_obs": 1000,    # Base observation (optional but if present, should be real)
    "ground_truth": 0,   # Ground truth can be empty initially
}


def validate_scenario(scenario_root: Path, scenario: str) -> Tuple[bool, List[str]]:
    """
    Validate scenario inputs.
    
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
        with profile_path.open("r") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON in scenario_profile.json: {e}")
        return False, issues
    
    # Check required keys
    required = ["rover_obs", "nav_file"]
    for key in required:
        if key not in cfg:
            issues.append(f"Missing required key in profile: {key}")
    
    if issues:
        return False, issues
    
    # Check files
    for key in ["rover_obs", "nav_file", "base_obs", "ground_truth"]:
        filename = cfg.get(key)
        if not filename:
            if key in required:
                issues.append(f"Required file not specified: {key}")
            continue
        
        filepath = scen_dir / filename
        min_size = MIN_FILE_SIZES.get(key, 0)
        
        if not filepath.exists():
            issues.append(f"File not found: {filepath}")
        else:
            size = filepath.stat().st_size

            # Only enforce "non-empty" for files that have a positive min_size
            # (ground_truth is allowed to be empty during early pipeline bring-up)
            if size == 0 and min_size > 0:
                issues.append(f"File is empty (placeholder): {filepath}")
            elif size < min_size and min_size > 0:
                issues.append(
                    f"File too small ({size} bytes, expected >= {min_size}): {filepath}"
                )

    
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
    args = parser.parse_args()
    
    if args.guidance:
        print_guidance()
        return
    
    scenario_root = Path(args.scenario_root)
    is_valid, issues = validate_scenario(scenario_root, args.scenario)
    
    if is_valid:
        print(f"[OK] Scenario '{args.scenario}' is valid and ready to run.")
        sys.exit(0)
    else:
        print(f"[ERROR] Scenario '{args.scenario}' has issues:\n")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRun with --guidance for help obtaining valid RINEX data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
