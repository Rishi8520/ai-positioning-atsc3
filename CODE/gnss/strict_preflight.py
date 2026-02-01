"""
Strict Preflight Validator for GNSS Baseline Runner

This module provides strict validation ensuring all scenarios use real RINEX data
through RTKLIB rnx2rtkp. NO synthetic/simulated epochs are allowed.

Key Features:
- Single source of truth for input validation
- CRINEX detection and optional auto-conversion
- RTK profile loading from YAML
- Fail-fast with actionable error messages

Usage:
    from strict_preflight import strict_preflight_check, PreflightResult
    
    result = strict_preflight_check(config, auto_crinex_convert=False)
    if not result.valid:
        print_preflight_errors(result, scenario_name)
        sys.exit(2)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class PreflightResult:
    """Result of preflight validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    rover_path: Optional[Path] = None
    base_path: Optional[Path] = None
    nav_path: Optional[Path] = None
    crinex_files: List[str] = field(default_factory=list)


# ----------------------------
# CRINEX Detection
# ----------------------------

def is_crinex_file(file_path: Optional[Path]) -> bool:
    """
    Detect if a file is in CRINEX (Compact RINEX) format.
    
    CRINEX files have "COMPACT RINEX FORMAT" or "CRINEX" in the header.
    """
    if not file_path or not file_path.exists():
        return False
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                line_upper = line.upper()
                if "COMPACT RINEX FORMAT" in line_upper or "CRINEX" in line_upper:
                    return True
    except Exception:
        pass
    
    return False


def is_crx2rnx_available() -> bool:
    """Check if crx2rnx tool is available in PATH."""
    return shutil.which("crx2rnx") is not None


# ----------------------------
# RINEX Validation
# ----------------------------

def is_valid_rinex_obs(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a file is a valid (non-empty) RINEX observation file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    size = file_path.stat().st_size
    if size == 0:
        return False, f"File is empty (placeholder): {file_path}"
    
    if size < 1000:
        return False, f"File too small ({size} bytes), likely not real RINEX: {file_path}"
    
    # Check for RINEX header
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            if "RINEX" not in first_line.upper() and "COMPACT" not in first_line.upper():
                return False, f"File does not appear to be RINEX format: {file_path}"
    except Exception as e:
        return False, f"Cannot read file: {file_path} ({e})"
    
    return True, ""


def is_valid_rinex_nav(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a file is a valid (non-empty) RINEX navigation file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"Navigation file not found: {file_path}"
    
    size = file_path.stat().st_size
    if size == 0:
        return False, f"Navigation file is empty: {file_path}"
    
    if size < 500:
        return False, f"Navigation file too small ({size} bytes): {file_path}"
    
    return True, ""


# ----------------------------
# CRINEX Conversion
# ----------------------------

def convert_crinex_file(input_path: Path, output_path: Path) -> bool:
    """
    Convert a single CRINEX file using: cat <input> | crx2rnx - > <output>
    
    Returns:
        True if conversion succeeded
    """
    try:
        cmd = f"cat '{input_path}' | crx2rnx - > '{output_path}'"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            return False
        
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False
        
        return True
        
    except Exception:
        return False


def convert_crinex_files(
    rover_crinex: Optional[Path],
    base_crinex: Optional[Path],
    run_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Convert CRINEX files to standard RINEX.
    
    Converted files are placed in run_dir to avoid modifying original data.
    
    Returns:
        Tuple of (converted_rover_path, converted_base_path)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    converted_rover = None
    converted_base = None
    
    if rover_crinex and rover_crinex.exists():
        output_path = run_dir / "rover_converted.obs"
        if convert_crinex_file(rover_crinex, output_path):
            converted_rover = output_path
    
    if base_crinex and base_crinex.exists():
        output_path = run_dir / "base_converted.obs"
        if convert_crinex_file(base_crinex, output_path):
            converted_base = output_path
    
    return converted_rover, converted_base


# ----------------------------
# RTKLIB Binary Check
# ----------------------------

def find_rtklib_binary(bin_name: str = "rnx2rtkp") -> Optional[Path]:
    """
    Find RTKLIB binary in system PATH or RTKLIB_HOME.
    
    Returns:
        Path to binary or None if not found
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

    return None


# ----------------------------
# RTK Profiles
# ----------------------------

def load_rtk_profiles(profiles_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load RTK processing profiles from YAML file.
    
    Args:
        profiles_path: Path to rtk_profiles.yaml. If None, uses default location.
        
    Returns:
        Dictionary with 'baseline' and 'optimised' profile sections
    """
    if profiles_path is None:
        profiles_path = Path(__file__).parent / "rtk_profiles.yaml"
    
    if not profiles_path.exists():
        return {}
    
    try:
        import yaml
        with open(profiles_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("[WARN] PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"[WARN] Failed to load RTK profiles: {e}", file=sys.stderr)
        return {}


def get_profile_settings(
    profiles: Dict[str, Any],
    mode: str,
    intent: str,
    scenario_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Get RTKLIB settings from profile based on mode and intent.
    
    Args:
        profiles: Loaded RTK profiles
        mode: "traditional" or "ai"
        intent: Scenario intent (for optimised profile selection)
        scenario_name: Optional scenario name for scenario-specific profiles
        
    Returns:
        RTKLIB settings dictionary
    """
    settings = {}
    
    # First check for scenario-specific profiles
    if scenario_name:
        scenarios = profiles.get("scenarios", {})
        if scenario_name in scenarios:
            scenario_profiles = scenarios[scenario_name]
            profile_key = "baseline" if mode == "traditional" else "optimised"
            if profile_key in scenario_profiles:
                profile = scenario_profiles[profile_key]
                for key, value in profile.items():
                    if key != "description":
                        settings[key] = str(value)
                return settings
    
    # Fall back to global profiles
    if mode == "traditional":
        profile = profiles.get("baseline", {})
    else:
        optimised = profiles.get("optimised", {})
        # Try exact intent match, then default
        profile = optimised.get(intent, optimised.get("default", {}))
    
    for key, value in profile.items():
        if key != "description":
            settings[key] = str(value)
    
    return settings


# ----------------------------
# Main Preflight Check
# ----------------------------

def strict_preflight_check(
    rover_obs: Optional[Path],
    nav_file: Optional[Path],
    base_obs: Optional[Path],
    auto_crinex_convert: bool = False,
    run_dir: Optional[Path] = None,
) -> PreflightResult:
    """
    Strict preflight validation ensuring all inputs are real RINEX data.
    
    This is the SINGLE SOURCE OF TRUTH for input validation.
    Called before ANY processing to ensure real data mode.
    
    Args:
        rover_obs: Path to rover observation file
        nav_file: Path to navigation file
        base_obs: Path to base observation file (optional)
        auto_crinex_convert: If True, auto-convert CRINEX files
        run_dir: Directory for converted files (required if auto_crinex_convert=True)
        
    Returns:
        PreflightResult with validation status and file paths
    """
    result = PreflightResult(valid=True, errors=[], warnings=[])
    
    # 1. Check RTKLIB binary availability
    rnx2rtkp_bin = find_rtklib_binary("rnx2rtkp")
    if not rnx2rtkp_bin:
        result.valid = False
        result.errors.append(
            "RTKLIB rnx2rtkp binary not found. "
            "Set RTKLIB_HOME environment variable or add rnx2rtkp to PATH."
        )
    
    # 2. Validate rover observation file
    if not rover_obs:
        result.valid = False
        result.errors.append("Rover observation file not specified")
    else:
        is_valid, error = is_valid_rinex_obs(rover_obs)
        if not is_valid:
            result.valid = False
            result.errors.append(error)
        elif is_crinex_file(rover_obs):
            result.crinex_files.append(str(rover_obs))
    
    # 3. Validate navigation file
    if not nav_file:
        result.valid = False
        result.errors.append("Navigation file not specified")
    else:
        is_valid, error = is_valid_rinex_nav(nav_file)
        if not is_valid:
            result.valid = False
            result.errors.append(error)
    
    # 4. Validate base observation file (if specified)
    if base_obs:
        is_valid, error = is_valid_rinex_obs(base_obs)
        if not is_valid:
            result.valid = False
            result.errors.append(error)
        elif is_crinex_file(base_obs):
            result.crinex_files.append(str(base_obs))
    
    # 5. Handle CRINEX files
    if result.crinex_files:
        if auto_crinex_convert:
            if not run_dir:
                result.valid = False
                result.errors.append("CRINEX auto-convert requires run_dir")
            elif not is_crx2rnx_available():
                result.valid = False
                result.errors.append(
                    "CRINEX auto-convert requested but crx2rnx not found. "
                    "Install: brew install rnxcmp (macOS) or apt install rnxcmp (Linux)"
                )
            else:
                # Convert CRINEX files
                rover_to_convert = rover_obs if (rover_obs and str(rover_obs) in result.crinex_files) else None
                base_to_convert = base_obs if (base_obs and str(base_obs) in result.crinex_files) else None
                
                converted_rover, converted_base = convert_crinex_files(
                    rover_to_convert, base_to_convert, run_dir
                )
                
                if rover_to_convert:
                    if converted_rover:
                        result.rover_path = converted_rover
                        result.warnings.append(f"Converted rover CRINEX: {converted_rover}")
                    else:
                        result.valid = False
                        result.errors.append(f"Failed to convert rover CRINEX: {rover_obs}")
                
                if base_to_convert:
                    if converted_base:
                        result.base_path = converted_base
                        result.warnings.append(f"Converted base CRINEX: {converted_base}")
                    else:
                        result.valid = False
                        result.errors.append(f"Failed to convert base CRINEX: {base_obs}")
                
                # Clear CRINEX files if all converted successfully
                if result.valid:
                    result.crinex_files = []
        else:
            # CRINEX without auto-convert is a fatal error
            result.valid = False
            crinex_list = "\n      ".join(result.crinex_files)
            result.errors.append(
                f"CRINEX files detected but --auto-crinex-convert not enabled:\n"
                f"      {crinex_list}\n"
                f"    Options:\n"
                f"      1. Use --auto-crinex-convert flag\n"
                f"      2. Manually convert: cat <file> | crx2rnx - > <output>\n"
                f"      3. Use pre-converted files (e.g., rover_rnx.obs)"
            )
    
    # Set final paths if not already set
    if not result.rover_path and rover_obs:
        result.rover_path = rover_obs
    if not result.base_path and base_obs:
        result.base_path = base_obs
    if nav_file:
        result.nav_path = nav_file
    
    return result


def print_preflight_errors(result: PreflightResult, scenario_name: str) -> None:
    """Print formatted preflight error report."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("STRICT PREFLIGHT VALIDATION FAILED", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"\nScenario: {scenario_name}", file=sys.stderr)
    print(f"\nThe following issues must be resolved:\n", file=sys.stderr)
    
    for i, error in enumerate(result.errors, 1):
        print(f"  {i}. {error}", file=sys.stderr)
    
    if result.warnings:
        print(f"\nWarnings:", file=sys.stderr)
        for warning in result.warnings:
            print(f"  - {warning}", file=sys.stderr)
    
    print("\n" + "-" * 70, file=sys.stderr)
    print("REQUIRED FOR STRICT REAL-DATA MODE:", file=sys.stderr)
    print("  - All scenarios must have real RINEX rover/base/nav files", file=sys.stderr)
    print("  - scenario2/scenario3 need real data (not empty placeholders)", file=sys.stderr)
    print("  - For CRINEX: use --auto-crinex-convert or pre-convert", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)
