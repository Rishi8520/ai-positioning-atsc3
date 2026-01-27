# GNSS Module - README

This module provides GNSS positioning tools for the AI Positioning PoC, including:
- **Baseline RTK/PPK** processing via RTKLIB
- **RTCM generation** for broadcast pipeline integration
- **Scenario validation** utilities

---

## Quick Start

### Prerequisites

1. **RTKLIB** installed and available:
   ```bash
   # Option A: Set RTKLIB_HOME environment variable
   export RTKLIB_HOME=/path/to/rtklib
   
   # Option B: Ensure rnx2rtkp and convbin are on PATH
   which rnx2rtkp  # Should return path to binary
   ```

2. **Python dependencies** (from project root):
   ```bash
   pip install -r CODE/requirements.txt
   ```

---

## Usage Examples

### 1. Validate Scenario Inputs

Before running, validate that scenario files are properly configured:

```bash
cd /path/to/ai-positioning-atsc3-main

# Validate scenario1
python CODE/gnss/validate_scenario.py --scenario scenario1

# If validation fails, get guidance on obtaining data
python CODE/gnss/validate_scenario.py --scenario scenario1 --guidance
```

### 2. Run Baseline Positioning (No Corrections)

This is "Point 1" - baseline RTK/PPK without broadcast transport:

```bash
# Using scenario mode
python CODE/gnss/gnss_baseline_runner.py --scenario scenario1 --corrections none

# Using explicit file paths
python CODE/gnss/gnss_baseline_runner.py \
    --rover DATA/scenarios/scenario1/rover.obs \
    --nav DATA/scenarios/scenario1/nav.nav \
    --base DATA/scenarios/scenario1/base.obs \
    --corrections none
```

**Output location**: `OUTPUTS/<scenario>/<run_id>/`
- `solution.pos` - RTKLIB position solution (LLH format)
- `metrics.csv` - Single-row metrics for aggregation
- `metrics.json` - Full metrics object
- `run_manifest.json` - Run metadata and diagnostics
- `rtk_options.conf` - RTKLIB configuration used

### 3. Generate RTCM Corrections

Generate RTCM data for broadcast pipeline testing:

```bash
# From scenario (auto-detects base station files)
python CODE/gnss/rtcm_generator.py --scenario scenario1 --out OUTPUTS/scenario1/corrections.rtcm

# From explicit files
python CODE/gnss/rtcm_generator.py \
    --base DATA/scenarios/scenario1/base.obs \
    --nav DATA/scenarios/scenario1/nav.nav \
    --out corrections.rtcm

# Generate synthetic RTCM (no real data required)
python CODE/gnss/rtcm_generator.py --synthetic --out test.rtcm --epochs 100
```

**RTCM Output Format**:
- RTCM 3.x binary format
- Contains Message 1005 (station coordinates) + Message 1077 (GPS MSM7 observations)
- Compatible with `broadcast/pipeline.py`

### 4. Feed RTCM to Broadcast Pipeline

```python
from broadcast.pipeline import BroadcastPipeline

# Read generated RTCM
with open("OUTPUTS/scenario1/corrections.rtcm", "rb") as f:
    rtcm_bytes = f.read()

# Process through broadcast pipeline
pipeline = BroadcastPipeline()
result = pipeline.process(rtcm_bytes)

# Result contains OFDM signal ready for transmission
print(f"OFDM samples: {result.signal.num_samples}")
print(f"Duration: {result.signal.duration_ms:.2f} ms")
```

### 5. End-to-End Workflow

```bash
# Step 1: Validate inputs
python CODE/gnss/validate_scenario.py --scenario scenario1

# Step 2: Run baseline (establishes "ground truth" performance)
python CODE/gnss/gnss_baseline_runner.py --scenario scenario1 --corrections none

# Step 3: Generate RTCM corrections
python CODE/gnss/rtcm_generator.py --scenario scenario1 --out OUTPUTS/scenario1/corrections.rtcm

# Step 4: Process RTCM through broadcast (see demo_broadcast_pipeline.py)
python CODE/demo_broadcast_pipeline.py
```

---

## File Structure

```
CODE/gnss/
├── __init__.py
├── gnss_baseline_runner.py   # RTKLIB rnx2rtkp wrapper
├── rtcm_generator.py         # RTCM generation utility
├── validate_scenario.py      # Scenario validation
├── README.md                  # This file
└── tests/                     # Unit tests

DATA/scenarios/<scenario>/
├── scenario_profile.json     # Required: file mappings
├── rover.obs                 # Required: rover RINEX observation
├── nav.nav                   # Required: navigation file
├── base.obs                  # Optional: base station observation
└── gt.csv                    # Optional: ground truth positions

OUTPUTS/<scenario>/<run_id>/
├── solution.pos              # RTKLIB solution
├── metrics.csv               # Aggregatable metrics
├── metrics.json              # Full metrics
├── run_manifest.json         # Run metadata
└── rtk_options.conf          # RTKLIB config used
```

---

## Scenario Profile Format

`scenario_profile.json` example:
```json
{
  "rover_obs": "rover.obs",
  "nav_file": "nav.nav",
  "base_obs": "base.obs",
  "ground_truth": "gt.csv"
}
```

**Required keys**: `rover_obs`, `nav_file`
**Optional keys**: `base_obs`, `ground_truth`

Paths are relative to the scenario directory unless absolute.

---

## CLI Reference

### gnss_baseline_runner.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | None | Scenario name under DATA/scenarios/ |
| `--scenario-root` | `DATA/scenarios` | Root folder for scenarios |
| `--results-root` | `OUTPUTS` | Root folder for results |
| `--rover` | None | Explicit rover observation file |
| `--base` | None | Explicit base observation file |
| `--nav` | None | Explicit navigation file |
| `--corrections` | `none` | Corrections source (currently only `none`) |
| `--conf` | None | Custom RTKLIB config file |
| `--run-id` | Auto | Override run folder name |
| `--timeout-sec` | 180 | RTKLIB execution timeout |
| `--navsys` | `1` | RTKLIB navsys bitmask (1=GPS) |

### rtcm_generator.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | None | Scenario name |
| `--scenario-root` | `DATA/scenarios` | Root folder for scenarios |
| `--base` | None | Explicit base observation file |
| `--nav` | None | Explicit navigation file |
| `--out` / `-o` | Required | Output RTCM file path |
| `--synthetic` | False | Generate synthetic RTCM |
| `--epochs` | 10 | Number of epochs (synthetic) |
| `--num-sats` | 8 | Number of satellites (synthetic) |
| `--station-id` | 0 | Reference station ID |
| `--station-x/y/z` | Default ECEF | Station coordinates |

---

## Troubleshooting

### "Unable to find RTKLIB binary"
- Install RTKLIB from https://github.com/tomojitakasu/RTKLIB
- Set `RTKLIB_HOME` environment variable, or add to PATH

### "Scenario files are empty"
- The default scenario files are placeholders
- Download real RINEX data or use `--synthetic` mode for testing
- Run `validate_scenario.py --guidance` for data sources

### "RTKLIB rnx2rtkp failed"
- Check that input files are valid RINEX format
- Review `run_manifest.json` for stderr output
- Try with `--navsys 1` (GPS only) for simpler processing

---

## Integration with Broadcast Pipeline

The RTCM output is designed to work directly with the broadcast module:

```python
# RTCM → ALP → FEC → Frame → OFDM
from broadcast.pipeline import BroadcastPipeline

pipeline = BroadcastPipeline()
result = pipeline.process(rtcm_bytes)

# Access I/Q data for SDR transmission
i_data, q_data = result.signal.get_iq_data()
```

See `CODE/demo_broadcast_pipeline.py` for complete examples.
