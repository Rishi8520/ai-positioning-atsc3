#!/usr/bin/env python3
"""
GNSS Baseline Runner (RTKLIB rnx2rtkp wrapper)

Immediate goal (Point 1):
- Run baseline RTK/PPK without broadcast transport and without live RTCM.
- Produce RTKLIB solution output (.pos) and a compact metrics summary.

Usage (scenario mode):
  python gnss_baseline_runner.py --scenario scenario1 --corrections none

Usage (explicit paths):
  python gnss_baseline_runner.py --rover path/to/rover.obs --nav path/to/nav.nav \
      --base path/to/base.obs --corrections none

If base is omitted, it runs standalone positioning (SINGLE).
If base is provided, it runs differential/RTK depending on config settings.

Integration notes:
- Reads scenario profiles from: data/scenarios/<scenario>/scenario_profile.json
- Writes results to: results/<scenario>/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Constants / Defaults
# ----------------------------

DEFAULT_SCENARIO_ROOT = Path("DATA/scenarios")
DEFAULT_RESULTS_ROOT = Path("OUTPUTS")

RTKLIB_BINARIES = ["rnx2rtkp"]  # add others if needed

# RTKLIB solution quality codes (typical):
# 1=FIX, 2=FLOAT, 3=SBAS, 4=DGPS, 5=SINGLE, 6=PPP
Q_FIX = 1
Q_FLOAT = 2
Q_SINGLE = 5

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ScenarioInputs:
    scenario: str
    rover_obs: Path
    nav_file: Path
    base_obs: Optional[Path] = None
    # optional extra metadata
    ground_truth: Optional[Path] = None


@dataclass
class RunOutputs:
    run_id: str
    out_dir: Path
    pos_file: Path
    metrics_csv: Path
    metrics_json: Path
    manifest_json: Path
    rtk_conf_file: Path


# ----------------------------
# Utilities
# ----------------------------

def _now_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"


def _die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def resolve_rtklib_binary(bin_name: str) -> Path:
    """
    Resolve RTKLIB binary via:
      1) RTKLIB_HOME env var (preferred)
      2) PATH lookup
    """
    rtk_home = os.environ.get("RTKLIB_HOME", "").strip()
    if rtk_home:
        candidate = Path(rtk_home) / "bin" / bin_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    which = shutil.which(bin_name)
    if which:
        return Path(which).resolve()

    _die(
        f"Unable to find RTKLIB binary '{bin_name}'. "
        f"Set RTKLIB_HOME or ensure '{bin_name}' is on PATH."
    )
    raise RuntimeError("unreachable")


def build_default_rnx2rtkp_conf(
    out_path: Path,
    use_base: bool,
    navsys: str = "1",  # 1=GPS; can be ORed bitmask in RTKLIB configs; keep simple by default
) -> None:
    """
    Create a minimal RTKLIB options file suitable for PoC baselining.

    Notes:
    - If base is present, we keep posmode=kinematic and allow ambiguity resolution.
    - If no base, SINGLE mode will apply (base omitted => no differential corrections).
    - You can later expand this to match tighter PoC specs (e.g., specific constellations, elevation mask).
    """
    # RTKLIB option file format: "key = value"
    # This is intentionally minimal and conservative.
    lines = [
        "pos1-posmode     = 0",         # 0: single, 1: dgps, 2: kinematic, 3: static, 4: moving-base, 5: fixed, 6: ppp-kinematic, 7: ppp-static
        "pos1-frequency   = 2",         # 1:L1, 2:L1+L2
        "pos1-soltype     = 0",         # 0: forward
        "pos1-elmask      = 15",        # deg
        f"pos1-navsys      = {navsys}", # keep as GPS default unless you need multi-constellation
        "pos2-armode      = 1",         # 0: off, 1: continuous, 2: instantaneous, 3: fix and hold
        "pos2-arthres     = 3.0",       # ambiguity validation threshold
        "pos2-arminfix    = 10",        # min fix count
        "pos2-slipthres   = 0.05",
        "pos2-rejionno    = 30",
        "pos2-rejgdop     = 30",
        "pos2-niter       = 1",
        "out-solformat    = 0",         # 0: llh
        "out-outhead      = 1",
        "out-outopt       = 1",
        "out-outstat      = 0",
        "out-outvel       = 0",
        "out-timesys      = 0",         # 0: gpst
        "out-timeform     = 1",         # 1: yyyy/mm/dd hh:mm:ss
        "out-timendec     = 3",
        "out-degform      = 0",
        "out-fieldsep     = 0",
        "out-height       = 0",         # 0: ellipsoidal
    ]

    # If base exists, switch posmode to kinematic RTK (2) by default.
    # Without base, rnx2rtkp will still run but effectively yield SINGLE solutions.
    if use_base:
        # kinematic (RTK): 2
        lines[0] = "pos1-posmode     = 2"

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNSS Baseline Runner (RTKLIB rnx2rtkp)")
    p.add_argument("--scenario", type=str, default=None, help="Scenario name under data/scenarios/<scenario>/")
    p.add_argument("--scenario-root", type=str, default=str(DEFAULT_SCENARIO_ROOT), help="Root folder for scenarios")
    p.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT), help="Root folder for results")

    p.add_argument("--rover", type=str, default=None, help="Path to rover observation file (RINEX)")
    p.add_argument("--base", type=str, default=None, help="Path to base observation file (RINEX) (optional)")
    p.add_argument("--nav", type=str, default=None, help="Path to navigation file (RINEX nav)")

    p.add_argument("--corrections", type=str, default="none",
                   help="Corrections source. For Point-1 baseline use 'none'. (Future: path to RTCM file/stream)")
    p.add_argument("--conf", type=str, default=None,
                   help="Optional RTKLIB config file. If omitted, one is generated per run.")
    p.add_argument("--run-id", type=str, default=None, help="Override run id folder name")
    p.add_argument("--timeout-sec", type=int, default=180, help="Timeout for RTKLIB execution")
    p.add_argument("--navsys", type=str, default="1", help="RTKLIB navsys bitmask; default 1=GPS")

    return p.parse_args()


# ----------------------------
# Scenario loading
# ----------------------------

def load_scenario_inputs(scenario_root: Path, scenario: str) -> ScenarioInputs:
    """
    Expected file: data/scenarios/<scenario>/scenario_profile.json

    Minimal required keys:
      - rover_obs
      - nav_file
    Optional:
      - base_obs
      - ground_truth
    """
    scen_dir = scenario_root / scenario
    profile = scen_dir / "scenario_profile.json"
    if not profile.exists():
        _die(f"Scenario profile not found: {profile}")

    cfg = _load_json(profile)

    def _p(key: str) -> Optional[Path]:
        v = cfg.get(key)
        if not v:
            return None
        return (scen_dir / v).resolve() if not Path(v).is_absolute() else Path(v).resolve()

    rover = _p("rover_obs")
    nav = _p("nav_file")
    base = _p("base_obs")
    gt = _p("ground_truth")

    if rover is None or nav is None:
        _die(f"Scenario profile must include rover_obs and nav_file. Found keys: {list(cfg.keys())}")

    for needed in [rover, nav] + ([base] if base else []):
        if needed and not needed.exists():
            _die(f"Missing scenario input file: {needed}")

    return ScenarioInputs(
        scenario=scenario,
        rover_obs=rover,
        nav_file=nav,
        base_obs=base,
        ground_truth=gt
    )


def resolve_inputs_from_args(args: argparse.Namespace) -> ScenarioInputs:
    scenario_root = Path(args.scenario_root)

    if args.scenario:
        return load_scenario_inputs(scenario_root, args.scenario)

    # explicit mode
    if not args.rover or not args.nav:
        _die("Provide either --scenario OR explicit --rover and --nav (and optionally --base).")

    rover = Path(args.rover).resolve()
    nav = Path(args.nav).resolve()
    base = Path(args.base).resolve() if args.base else None

    for needed in [rover, nav] + ([base] if base else []):
        if needed and not needed.exists():
            _die(f"Missing input file: {needed}")

    return ScenarioInputs(
        scenario="adhoc",
        rover_obs=rover,
        nav_file=nav,
        base_obs=base,
        ground_truth=None
    )


# ----------------------------
# Metrics extraction (lightweight, robust)
# ----------------------------

def parse_rtk_pos_file(pos_path: Path) -> List[Dict]:
    """
    Parse RTKLIB .pos output in LLH format.
    Expected columns (typical):
      date time lat lon height Q ns sdn sde sdu sdne sdeu sdun age ratio
    Header lines begin with '%' or are blank.

    Returns list of epochs with keys:
      - ts (string)
      - Q (int)
      - sdn, sde, sdu (float or None)
      - ratio (float or None)
    """
    epochs: List[Dict] = []
    with pos_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue

            # date + time = timestamp
            ts = f"{parts[0]} {parts[1]}"
            # lat lon height present but not needed for immediate metrics
            # Q typically at index 5 in LLH format
            try:
                q = int(parts[5])
            except Exception:
                continue

            # Attempt to parse sdn/sde/sdu and ratio if present
            sdn = sde = sdu = None
            ratio = None
            try:
                # indices based on standard RTKLIB llh output:
                # 0 date,1 time,2 lat,3 lon,4 height,5 Q,6 ns,7 sdn,8 sde,9 sdu,..., last ratio
                if len(parts) >= 10:
                    sdn = float(parts[7])
                    sde = float(parts[8])
                    sdu = float(parts[9])
                if len(parts) >= 15:
                    ratio = float(parts[14])
            except Exception:
                pass

            epochs.append({
                "ts": ts,
                "Q": q,
                "sdn": sdn,
                "sde": sde,
                "sdu": sdu,
                "ratio": ratio
            })
    return epochs


def compute_metrics(epochs: List[Dict]) -> Dict:
    """
    Compute minimal PoC metrics:
      - counts and percentages of FIX/FLOAT/SINGLE/OTHER
      - time_to_first_fix_epochs (index-based)
      - horizontal_sigma stats (if available)
      - ratio stats (if available)
    """
    n = len(epochs)
    if n == 0:
        return {
            "epochs": 0,
            "fix_pct": 0.0,
            "float_pct": 0.0,
            "single_pct": 0.0,
            "other_pct": 0.0,
            "time_to_first_fix_epochs": None,
            "horiz_sigma_mean": None,
            "horiz_sigma_p50": None,
            "horiz_sigma_p95": None,
            "ratio_mean": None,
            "ratio_p50": None,
            "ratio_p95": None,
        }

    fix = sum(1 for e in epochs if e["Q"] == Q_FIX)
    flt = sum(1 for e in epochs if e["Q"] == Q_FLOAT)
    sgl = sum(1 for e in epochs if e["Q"] == Q_SINGLE)
    oth = n - fix - flt - sgl

    tff = None
    for i, e in enumerate(epochs):
        if e["Q"] == Q_FIX:
            tff = i
            break

    # Horizontal sigma = sqrt(sdn^2 + sde^2)
    hs: List[float] = []
    ratios: List[float] = []
    for e in epochs:
        if e["sdn"] is not None and e["sde"] is not None:
            hs.append((e["sdn"] ** 2 + e["sde"] ** 2) ** 0.5)
        if e["ratio"] is not None:
            ratios.append(float(e["ratio"]))

    def _pct(x: int) -> float:
        return 100.0 * x / n if n else 0.0

    def _quantile(vals: List[float], q: float) -> Optional[float]:
        if not vals:
            return None
        v = sorted(vals)
        idx = int(round((len(v) - 1) * q))
        return v[max(0, min(idx, len(v) - 1))]

    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return sum(vals) / len(vals)

    return {
        "epochs": n,
        "fix_pct": _pct(fix),
        "float_pct": _pct(flt),
        "single_pct": _pct(sgl),
        "other_pct": _pct(oth),
        "time_to_first_fix_epochs": tff,
        "horiz_sigma_mean": _mean(hs),
        "horiz_sigma_p50": _quantile(hs, 0.50),
        "horiz_sigma_p95": _quantile(hs, 0.95),
        "ratio_mean": _mean(ratios),
        "ratio_p50": _quantile(ratios, 0.50),
        "ratio_p95": _quantile(ratios, 0.95),
    }


def write_metrics_csv(path: Path, metrics: Dict, extra_cols: Dict) -> None:
    """
    One-row CSV for easy aggregation.
    """
    cols = {**extra_cols, **metrics}
    headers = list(cols.keys())
    values = [cols[h] for h in headers]

    # Convert None to empty
    values = [("" if v is None else v) for v in values]

    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        f.write(",".join(str(v) for v in values) + "\n")


# ----------------------------
# RTK execution
# ----------------------------

def run_rnx2rtkp(
    rnx2rtkp_bin: Path,
    rover_obs: Path,
    nav_file: Path,
    out_pos: Path,
    conf_file: Path,
    base_obs: Optional[Path],
    timeout_sec: int,
) -> Tuple[int, str, str]:
    """
    Execute RTKLIB rnx2rtkp.

    Command (typical):
      rnx2rtkp -k conf -o out.pos rover.obs base.obs nav.nav

    RTKLIB argument ordering:
      rnx2rtkp [options] rover [base] [nav...]
    """
    cmd = [str(rnx2rtkp_bin), "-k", str(conf_file), "-o", str(out_pos), str(rover_obs)]
    if base_obs:
        cmd.append(str(base_obs))
    cmd.append(str(nav_file))

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec
    )
    return proc.returncode, proc.stdout, proc.stderr


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()

    # Corrections handling:
    # - "none" => baseline
    # - otherwise treat as a local RTCM (or other) file path for now (validated + logged)
    corr_raw = (args.corrections or "").strip()
    corr_is_none = (corr_raw.lower() == "none" or corr_raw == "")

    corrections_path = None  # Path | None
    corrections_label = "none"

    if not corr_is_none:
        corrections_path = Path(corr_raw).expanduser().resolve()
        if not corrections_path.exists():
            _die(f"Provided --corrections file does not exist: {corrections_path}")
        if corrections_path.stat().st_size == 0:
            _die(f"Provided --corrections file is empty: {corrections_path}")
        corrections_label = str(corrections_path)

    inputs = resolve_inputs_from_args(args)

    results_root = Path(args.results_root)
    run_kind = "baseline" if corr_is_none else "with_corrections"
    run_id = args.run_id or _now_run_id(prefix=f"{inputs.scenario}_{run_kind}")


    out_dir = results_root / inputs.scenario / run_id
    _ensure_dir(out_dir)

    pos_file = out_dir / "solution.pos"
    metrics_csv = out_dir / "metrics.csv"
    metrics_json = out_dir / "metrics.json"
    manifest_json = out_dir / "run_manifest.json"
    rtk_conf_file = out_dir / "rtk_options.conf"

    outputs = RunOutputs(
        run_id=run_id,
        out_dir=out_dir,
        pos_file=pos_file,
        metrics_csv=metrics_csv,
        metrics_json=metrics_json,
        manifest_json=manifest_json,
        rtk_conf_file=rtk_conf_file,
    )

    # Resolve RTKLIB binary
    rnx2rtkp_bin = resolve_rtklib_binary("rnx2rtkp")

    # Choose config file
    if args.conf:
        conf_path = Path(args.conf).resolve()
        if not conf_path.exists():
            _die(f"Provided --conf file does not exist: {conf_path}")
        # copy into run dir for manifest consistency
        shutil.copy2(conf_path, outputs.rtk_conf_file)
    else:
        build_default_rnx2rtkp_conf(
            out_path=outputs.rtk_conf_file,
            use_base=bool(inputs.base_obs),
            navsys=args.navsys,
        )

    # Execute
    t0 = time.time()
    rc, stdout, stderr = run_rnx2rtkp(
        rnx2rtkp_bin=rnx2rtkp_bin,
        rover_obs=inputs.rover_obs,
        base_obs=inputs.base_obs,
        nav_file=inputs.nav_file,
        out_pos=outputs.pos_file,
        conf_file=outputs.rtk_conf_file,
        timeout_sec=args.timeout_sec,
    )
    elapsed = time.time() - t0

    # Validate output
    if rc != 0:
        _write_json(outputs.manifest_json, {
            "status": "failed",
            "return_code": rc,
            "elapsed_sec": elapsed,
            "rtklib_binary": str(rnx2rtkp_bin),
            "inputs": {
                "scenario": inputs.scenario,
                "rover_obs": str(inputs.rover_obs),
                "base_obs": (str(inputs.base_obs) if inputs.base_obs else None),
                "nav_file": str(inputs.nav_file),
                "corrections": corrections_label
            },
            "stderr": stderr,
            "stdout": stdout,
        })
        _die(
            f"RTKLIB rnx2rtkp failed (rc={rc}). "
            f"See manifest: {outputs.manifest_json}"
        )

    if not outputs.pos_file.exists() or outputs.pos_file.stat().st_size == 0:
        _die(f"RTKLIB did not produce a valid .pos output: {outputs.pos_file}")

    # Metrics
    epochs = parse_rtk_pos_file(outputs.pos_file)
    metrics = compute_metrics(epochs)

    extra = {
        "scenario": inputs.scenario,
        "run_id": outputs.run_id,
        "elapsed_sec": round(elapsed, 3),
        "mode": ("rtk" if inputs.base_obs else "standalone"),
        "corrections": corrections_label,
        "rtklib_binary": str(rnx2rtkp_bin),
        "rover_obs": str(inputs.rover_obs),
        "base_obs": (str(inputs.base_obs) if inputs.base_obs else ""),
        "nav_file": str(inputs.nav_file),
    }

    _write_json(outputs.metrics_json, {**extra, **metrics})
    write_metrics_csv(outputs.metrics_csv, metrics, extra)

    # Manifest
    _write_json(outputs.manifest_json, {
        "status": "ok",
        "elapsed_sec": elapsed,
        "rtklib_binary": str(rnx2rtkp_bin),
        "rtk_conf_file": str(outputs.rtk_conf_file),
        "inputs": {
            "scenario": inputs.scenario,
            "rover_obs": str(inputs.rover_obs),
            "base_obs": (str(inputs.base_obs) if inputs.base_obs else None),
            "nav_file": str(inputs.nav_file),
            "corrections": corrections_label
        },
        "outputs": {
            "pos_file": str(outputs.pos_file),
            "metrics_csv": str(outputs.metrics_csv),
            "metrics_json": str(outputs.metrics_json),
        },
        "rtklib_stdout": stdout,
        "rtklib_stderr": stderr,
    })

    print(f"[OK] Output written to: {outputs.out_dir}")
    print(f"     POS: {outputs.pos_file}")
    print(f"     CSV: {outputs.metrics_csv}")
    print(f"     JSON: {outputs.metrics_json}")


if __name__ == "__main__":
    main()
