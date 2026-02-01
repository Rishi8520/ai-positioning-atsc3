# GNSS Module - README

This module provides GNSS positioning tools for the AI Positioning PoC, including:
- **Baseline RTK/PPK** processing via RTKLIB
- **RTCM generation** for broadcast pipeline integration
- **Coverage map generation** for broadcast signal analysis
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

3. **CRX2RNX** (optional, for CRINEX conversion):
   ```bash
   # Check if crx2rnx is available
   which crx2rnx
   
   # Install if needed (via RNXCMP package)
   # https://terras.gsi.go.jp/ja/crx2rnx.html
   ```

---

## Strict Real-Data Mode

The GNSS module enforces **strict real-data mode** by default. This ensures:

1. **ALL scenarios use REAL RINEX data** - processed end-to-end through RTKLIB rnx2rtkp
2. **ZERO synthetic/simulated epochs** - comparison_report.json computed from actual RTKLIB outputs only
3. **Fail-fast validation** - actionable error messages if RINEX data is missing or invalid

### Preflight Validation

Before running RTKLIB, strict preflight checks verify:

- ✅ Rover observation file exists and is >1000 bytes
- ✅ Navigation file exists and is >500 bytes  
- ✅ Base observation file (if specified) exists and is >1000 bytes
- ✅ Files contain valid RINEX headers (not CRINEX compressed)
- ✅ RTKLIB rnx2rtkp binary is available

### CRINEX Auto-Conversion

Real RINEX data from CORS archives is often distributed in **CRINEX** (Compact RINEX) format.
Use the `--auto-crinex-convert` flag to automatically convert:

```bash
# Automatically convert CRINEX → RINEX before processing
python CODE/gnss/gnss_baseline_runner_v2.py \
    --scenario scenario1 \
    --auto-crinex-convert

# scenario1 has pre-converted files (rover_rnx.obs, base_rnx.obs)
# You can also use these directly without the flag
```

**Note**: Auto-conversion requires `crx2rnx` to be available on PATH.

### Scenario Validation with --strict-real

```bash
# Strict validation - fails if RINEX files are missing/empty
python CODE/gnss/validate_scenario.py --scenario scenario1 --strict-real

# Expected results:
# scenario1: PASS (has real RINEX data)
# scenario2: FAIL (empty placeholder files)
# scenario3: FAIL (empty placeholder files)
```

### RTK Profile Configuration

Processing profiles are configured in `rtk_profiles.yaml`:

```yaml
# Baseline (traditional) profile - standard RTKLIB settings
baseline:
  description: "Standard RTK processing"
  pos2-armode: continuous
  pos2-arthres: "3.0"

# Optimised (AI) profiles - keyed by intent
optimised:
  provide_sub_3cm_accuracy:
    pos2-armode: fix-and-hold
    pos2-arthres: "2.5"
    pos2-arlockcnt: "5"
```

Select profile via `--profile`:
```bash
python CODE/gnss/gnss_baseline_runner_v2.py \
    --scenario scenario1 \
    --profile baseline    # or: optimised
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error |
| 2 | Preflight validation failure (missing/invalid RINEX) |

### Verification Commands

```bash
# 1. Verify scenario1 has valid RINEX data
python CODE/gnss/validate_scenario.py --scenario scenario1 --strict-real

# 2. Verify scenario2 has valid real RINEX data  
python CODE/gnss/validate_scenario.py --scenario scenario2 --strict-real

# 3. Check for CRINEX files
head -5 DATA/scenarios/scenario1/rover.obs | grep -i "COMPACT RINEX"

# 4. Run with preflight validation
python CODE/gnss/gnss_baseline_runner_v2.py --scenario scenario1 \
    --dry-run  # Shows what would run without executing

# 5. Full end-to-end with auto-conversion
python CODE/gnss/gnss_baseline_runner_v2.py \
    --scenario scenario1 \
    --auto-crinex-convert \
    --profile baseline

# 6. Run all tests to verify the implementation
cd CODE/gnss && python3 -m pytest tests/ -v
```

---

## Evaluation Layer

The evaluation layer provides end-to-end **baseline vs optimised** comparison for each scenario, producing stable output schemas suitable for automated CI/CD validation and reporting.

### Running Evaluations

```bash
# Evaluate scenario1 (requires real RINEX data)
python CODE/gnss/rtk_evaluate.py --scenario scenario1

# Evaluate scenario2 (requires real RINEX data)
python CODE/gnss/rtk_evaluate.py --scenario scenario2

# Evaluate scenario3 (allows synthetic_fallback if flagged)
python CODE/gnss/rtk_evaluate.py --scenario scenario3

# Custom scenario/output roots
python CODE/gnss/rtk_evaluate.py --scenario scenario1 \
    --scenario-root DATA/scenarios \
    --output-root OUTPUTS
```

### Data Requirements

| Scenario | Real Data Required | Synthetic Fallback Allowed |
|----------|-------------------|---------------------------|
| scenario1 | ✅ Yes | ❌ No - fails if synthetic |
| scenario2 | ✅ Yes | ❌ No - fails if synthetic |
| scenario3 | ✅ Preferred | ✅ Yes - with `simulated=true` flag |

**Note**: Scenario3 allows simulated comparison ONLY when `synthetic_fallback: true` is set in `scenario_profile.json`. The output will clearly indicate `simulated=true` in this case.

### Output Schema

Each evaluation run creates a timestamped directory:
```
OUTPUTS/<scenario>/evaluation/<timestamp>/
├── comparison.json      # Structured comparison result
├── comparison.csv       # Tabular format for aggregation
├── run_manifest.json    # Provenance, inputs, commands
├── baseline_run/        # RTKLIB baseline outputs
│   └── solution.pos
└── optimised_run/       # RTKLIB optimised outputs (or simulated)
    └── solution.pos
```

#### comparison.json Schema

```json
{
  "scenario": "scenario1",
  "timestamp": "20260202T120000Z",
  "simulated": false,
  "data_source": "real",
  "simulation_reason": null,
  "simulation_method": null,
  "baseline": {
    "horizontal_error_rms_m": 0.0523,
    "horizontal_error_p95_m": 0.1045,
    "vertical_error_rms_m": 0.0891,
    "vertical_error_p95_m": 0.1523,
    "fix_rate_pct": 72.5,
    "float_rate_pct": 18.3,
    "single_rate_pct": 9.2,
    "availability_pct": 90.8,
    "ttff_sec": 5.2,
    "num_mode_transitions": 12,
    "num_fix_losses": 4,
    "total_epochs": 300
  },
  "optimised": {
    "horizontal_error_rms_m": 0.0418,
    "horizontal_error_p95_m": 0.0836,
    "vertical_error_rms_m": 0.0757,
    "vertical_error_p95_m": 0.1295,
    "fix_rate_pct": 85.2,
    "float_rate_pct": 10.5,
    "single_rate_pct": 4.3,
    "availability_pct": 95.7,
    "ttff_sec": 3.6,
    "num_mode_transitions": 7,
    "num_fix_losses": 2,
    "total_epochs": 300
  },
  "delta": {
    "horizontal_error_rms_delta_m": -0.0105,
    "horizontal_error_p95_delta_m": -0.0209,
    "vertical_error_rms_delta_m": -0.0134,
    "vertical_error_p95_delta_m": -0.0228,
    "fix_rate_delta_pct": 12.7,
    "float_rate_delta_pct": -7.8,
    "availability_delta_pct": 4.9,
    "ttff_delta_sec": -1.6,
    "mode_transitions_delta": -5,
    "fix_losses_delta": -2
  }
}
```

#### Metrics Definitions

| Metric | Unit | Description |
|--------|------|-------------|
| `horizontal_error_rms_m` | meters | RMS of horizontal position error (Haversine) |
| `horizontal_error_p95_m` | meters | 95th percentile horizontal error |
| `vertical_error_rms_m` | meters | RMS of vertical (height) error |
| `vertical_error_p95_m` | meters | 95th percentile vertical error |
| `fix_rate_pct` | % | Percentage of epochs with RTK FIX (Q=1) |
| `float_rate_pct` | % | Percentage of epochs with RTK FLOAT (Q=2) |
| `single_rate_pct` | % | Percentage of epochs with SINGLE (Q=5) |
| `availability_pct` | % | FIX + FLOAT percentage |
| `ttff_sec` | seconds | Time to first FIX from first epoch |
| `num_mode_transitions` | count | Total quality mode changes |
| `num_fix_losses` | count | FIX → non-FIX transitions |
| `total_epochs` | count | Total solution epochs |

#### Delta Interpretation

- **Error deltas**: Negative = improvement (optimised has lower error)
- **Rate deltas**: Positive = improvement for fix_rate/availability
- **TTFF delta**: Negative = improvement (faster convergence)
- **Transition deltas**: Negative = improvement (more stable)

### Simulated Mode (Scenario3 Fallback)

When scenario3 uses `synthetic_fallback` data:

1. Baseline is computed from real RTKLIB run on available data
2. Optimised is **simulated** using deterministic improvement factors:
   - Fix rate: +15%
   - Availability: +10%
   - TTFF: -30%
   - Mode transitions: -25%
   - Fix losses: -30%
   - Horizontal error: -20%
   - Vertical error: -15%

3. Output includes:
   ```json
   {
     "simulated": true,
     "simulation_reason": "scenario3 synthetic_fallback data",
     "simulation_method": "deterministic_improvement_factors"
   }
   ```

### Running Tests

```bash
# Run evaluation layer tests
cd CODE/gnss && python3 -m pytest tests/test_rtk_evaluate.py -v

# Run all GNSS tests
cd CODE/gnss && python3 -m pytest tests/ -v
```

---

## Intent-Driven Scoring

The evaluation layer supports **intent-driven scoring** to quantify how well the optimised configuration meets specific use-case requirements.

### Available Intents

| Intent | Description | Key Metrics |
|--------|-------------|-------------|
| `accuracy` | Prioritize horizontal/vertical accuracy | HPE/VPE RMS, P95, fix rate |
| `robustness` | Prioritize stability and availability | Availability, fix losses, mode transitions |
| `latency` | Prioritize fast convergence | TTFF, fix rate |
| `balanced` | Balanced weighting across all factors | All metrics with moderate weights |

### Usage

```bash
# Evaluate with accuracy intent
python CODE/gnss/rtk_evaluate.py --scenario scenario1 --intent accuracy

# Evaluate with robustness intent
python CODE/gnss/rtk_evaluate.py --scenario scenario1 --intent robustness

# Evaluate with latency intent
python CODE/gnss/rtk_evaluate.py --scenario scenario1 --intent latency

# Evaluate with balanced intent
python CODE/gnss/rtk_evaluate.py --scenario scenario2 --intent balanced
```

### Output Files

When `--intent` is specified, additional fields are added:

```
OUTPUTS/<scenario>/evaluation/<timestamp>/
├── comparison.json           # Now includes intent, intent_score, intent_score_breakdown
├── intent_score_explain.json # Detailed formula substitution with actual numbers
└── ...
```

### Score Formula

```
score = sum(weight_i * normalized_value_i) / sum(|weight_i|)
```

- **Range**: Roughly [-1, 1], higher = better for the intent
- **Deterministic**: Same metrics + same intent = same score (no randomness)
- **Normalized**: Each metric is normalized to [0, 1] range before weighting

### Normalizer Types

| Type | Formula | Use Case |
|------|---------|----------|
| `linear` | `value * scale` | Percentages (0-100 → 0-1) |
| `clamp_ratio` | `clamp(value, min, max) / max` | Errors (low error = low ratio, combined with negative weight) |
| `count_ratio` | `value / base` | Counts (0 events = 0, combined with negative weight) |

### Intent Configuration

Intents are defined in `CODE/gnss/intents.json`:

```json
{
  "intents": {
    "accuracy": {
      "description": "Prioritize horizontal/vertical accuracy and fix rate",
      "weights": {
        "horizontal_error_rms_m": -10.0,
        "vertical_error_rms_m": -8.0,
        "fix_rate_pct": 8.0,
        "availability_pct": 3.0
      },
      "normalizers": {
        "horizontal_error_rms_m": {"type": "clamp_ratio", "max_val": 1.0},
        "fix_rate_pct": {"type": "linear", "scale": 0.01}
      }
    }
  }
}
```

**Weight semantics:**
- **Positive weights** (fix_rate, availability): Higher metric value = higher contribution (good)
- **Negative weights** (errors, counts): Lower metric value = smaller negative contribution (good)

### Example Output

#### comparison.json (with intent)

```json
{
  "scenario": "scenario1",
  "timestamp": "20260202T120000Z",
  "simulated": false,
  "intent": "accuracy",
  "intent_score": 0.7234,
  "intent_score_breakdown": {
    "description": "Prioritize horizontal/vertical accuracy and fix rate",
    "total_weight": 38.0,
    "simulated": false,
    "metrics_used": ["horizontal_error_rms_m", "fix_rate_pct", "availability_pct"]
  },
  "baseline": {...},
  "optimised": {...},
  "delta": {...}
}
```

#### intent_score_explain.json

```json
{
  "intent": "accuracy",
  "intent_description": "Prioritize horizontal/vertical accuracy and fix rate",
  "simulated": false,
  "final_score": 0.7234,
  "formula": "score = sum(weight_i * normalized_value_i) / sum(|weight_i|)",
  "total_weight": 38.0,
  "weighted_sum": 27.49,
  "computation_steps": [
    "-10.0 * normalize(horizontal_error_rms_m=0.05) = -10.0 * 0.95 = -9.50",
    "+8.0 * normalize(fix_rate_pct=85.0) = +8.0 * 0.85 = +6.80"
  ],
  "breakdown": [
    {
      "metric": "horizontal_error_rms_m",
      "raw_value": 0.05,
      "normalized_value": 0.95,
      "weight": -10.0,
      "weighted_contribution": -9.5,
      "normalizer_type": "inverse_clamp",
      "normalizer_params": {"max_val": 1.0, "min_val": 0.001}
    }
  ]
}
```

### Simulated Scoring

For scenario3 with `synthetic_fallback`, intent scoring still works but:

1. The `simulated` flag is set to `true` in the score result
2. `intent_score_explain.json` includes a `simulated_note` field
3. Scores are computed from simulated optimised metrics

---

### Populating Scenarios with Real Data

Each scenario should have its own **distinct real data** source for meaningful comparisons.

#### Scenario 2: UrbanNav Hong Kong Dataset (Recommended)

Use the dedicated UrbanNav population script for scenario2:

```bash
# Download and populate scenario2 from UrbanNav public dataset
python CODE/gnss/populate_scenario2_urbannav.py

# Force re-download
python CODE/gnss/populate_scenario2_urbannav.py --force

# Use existing downloaded files (in DATA/scenarios/scenario2/raw/)
python CODE/gnss/populate_scenario2_urbannav.py --skip-download
```

**Dataset**: UrbanNav-HK-Medium-Urban-1
- Location: Hong Kong, TST (Tsim Sha Tsui)
- Duration: 785 seconds, 3.64 km path
- Environment: Medium urban canyon
- Ground Truth: NovAtel SPAN-CPT + IE (5cm RMSE)
- Source: https://github.com/IPNL-POLYU/UrbanNavDataset

#### Scenario 3: UrbanNav Deep Urban Canyon (Dense Urban)

Use the dedicated UrbanNav population script for scenario3:

```bash
# Download and populate scenario3 from UrbanNav public dataset
python CODE/gnss/populate_scenario3_urbannav.py

# Force re-download
python CODE/gnss/populate_scenario3_urbannav.py --force

# Use existing downloaded files (in DATA/scenarios/scenario3/raw/)
python CODE/gnss/populate_scenario3_urbannav.py --skip-download
```

**Dataset**: UrbanNav-HK-Data20190428 (Whampoa)
- Location: Hong Kong, Whampoa (Deep Urban Canyon)
- Duration: 600 seconds, 2.5 km path
- Environment: **Deep urban canyon** with high-rise buildings
- Ground Truth: NovAtel SPAN-CPT + IE (5cm RMSE)
- Source: https://github.com/IPNL-POLYU/UrbanNavDataset

**NOTE**: If Dropbox downloads fail, the script generates UrbanNav-inspired synthetic data that matches urban canyon characteristics. This is clearly marked in `scenario_profile.json` with `synthetic_fallback: true`.

**Key differences from scenario2:**
- More severe multipath and NLOS conditions
- Lower expected FIX percentage
- More frequent mode transitions (FIX→FLOAT→SINGLE)
- Higher position errors during signal blockage

**Verification commands for scenario3:**
```bash
# 1. Validate scenario3 has proper data (not from scenario1)
python CODE/gnss/validate_scenario.py --scenario scenario3 --strict-real

# 2. Check scenario3 profile shows UrbanNav source
grep -A5 "real_data_source" DATA/scenarios/scenario3/scenario_profile.json

# 3. Verify scenario3 coordinates are in Hong Kong (not Bahamas)
grep "station_coordinates" -A4 DATA/scenarios/scenario3/scenario_profile.json

# 4. Run baseline vs optimised comparison
python CODE/gnss/gnss_baseline_runner_v2.py --scenario scenario3 --mode both

# 5. Check comparison report
cat OUTPUTs/scenario3/latest_run/comparison_report.json
```

#### General Population Script

For other scenarios or custom sources:

```bash
# Populate scenario3 from scenario1's real data (for testing)
python CODE/gnss/populate_scenario_realdata.py --scenario scenario3 --source scenario1

# Populate from UNAVCO/EarthScope (requires internet)
python CODE/gnss/populate_scenario_realdata.py --scenario scenario3 --source unavco \
    --rover-station CN14 --base-station CN13 --date 2026-01-10

# Populate from local RINEX files
python CODE/gnss/populate_scenario_realdata.py --scenario scenario3 --source local \
    --rover /path/to/rover.obs --base /path/to/base.obs --nav /path/to/nav.nav
```

The scripts will:
1. Download/copy RINEX observation and navigation files
2. Generate ground truth using RTKLIB with high-quality settings
3. Generate RTCM corrections from real base station data
4. Update scenario profile (set `synthetic_metadata.synthetic=false`)
5. Create documentation (`README_DATASET.md`, `dataset_manifest.json`)

#### Running Baseline vs Optimised Comparison

After populating a scenario with real data:

```bash
# Run traditional baseline mode
python CODE/gnss/gnss_baseline_runner_v2.py --scenario scenario2 --mode traditional

# Run AI-optimised mode  
python CODE/gnss/gnss_baseline_runner_v2.py --scenario scenario2 --mode ai

# Run both modes and generate comparison_report.json
python CODE/gnss/gnss_baseline_runner_v2.py --scenario scenario2 --mode both
```

The comparison report is computed purely from real RTKLIB outputs.

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

### 4. Generate Coverage Map

Generate a coverage proxy map based on free-space path loss (FSPL) and SNR estimation:

```bash
# From scenario (uses station coordinates from profile)
python CODE/gnss/coverage_map_generator.py --scenario scenario1

# With explicit transmitter location
python CODE/gnss/coverage_map_generator.py --tx-lat 37.5665 --tx-lon 126.9780 \
    --scenario scenario1

# Custom grid resolution and range
python CODE/gnss/coverage_map_generator.py --scenario scenario1 \
    --grid-res-m 100 --max-range-km 30

# Full custom configuration
python CODE/gnss/coverage_map_generator.py --tx-lat 37.5 --tx-lon 127.0 \
    --tx-height-m 50 --rx-height-m 2.0 \
    --grid-res-m 25 --max-range-km 20 \
    --out-dir OUTPUTS/custom
```

**Output location**: `OUTPUTS/<scenario>/<scenario>_coverage_<timestamp>/`

Each run creates a unique timestamped directory containing:
- `coverage_map.csv` - Grid cells with columns: `lat, lon, distance_m, fspl_db, snr_db, class`
- `coverage_summary.json` - Statistics and thresholds used
- `run_manifest.json` - Run metadata including scenario, timestamp, and CLI arguments

Example run directory: `OUTPUTS/scenario1/scenario1_coverage_20260201T143052Z/`

**Coverage Classes**:
- `good`: SNR ≥ 20 dB (robust reception)
- `marginal`: 10 dB ≤ SNR < 20 dB (marginal reception)
- `bad`: SNR < 10 dB (unreliable reception)

**Note**: This uses a simplified FSPL model for demonstration purposes. It does NOT account for terrain, buildings, or atmospheric effects. For accurate RF planning, use proper propagation modeling tools.

### 5. Generate Synthetic Scenario Data

Generate deterministic synthetic datasets for scenario2 (sparse RTCM) and scenario3 (urban multipath) testing:

```bash
# Generate scenario2 synthetic data (suburban/highway)
python CODE/gnss/scenario_simulator.py --scenario scenario2 --seed 42 --n-epochs 300

# Generate scenario3 synthetic data (urban canyon)
python CODE/gnss/scenario_simulator.py --scenario scenario3 --seed 42 --n-epochs 300

# Custom seed and epoch count
python CODE/gnss/scenario_simulator.py --scenario scenario2 --seed 123 --n-epochs 600
```

**Scenario2 outputs** (`DATA/scenarios/scenario2/`):
- `gt.csv` - Ground truth trajectory (lat, lon, height, velocity, heading)
- `corrections_schedule.json` - RTCM availability schedule with dropouts
- Updated `scenario_profile.json` with `synthetic_metadata`

**Scenario3 outputs** (`DATA/scenarios/scenario3/`):
- `gt.csv` - Ground truth trajectory
- `multipath_profile.json` - Multipath bias, noise, NLOS per epoch
- `mode_transitions.csv` - FIX/FLOAT/STANDALONE mode per epoch
- Updated `scenario_profile.json` with `synthetic_metadata`

**Key features**:
- Deterministic given the same seed (reproducible experiments)
- Scenario2: Sparse RTCM corrections (5-10s updates, ~15% dropout)
- Scenario3: Urban multipath with frequent mode transitions
- Validates correctly with `validate_scenario.py`

### 6. Feed RTCM to Broadcast Pipeline

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

### 7. End-to-End Workflow

```bash
# Step 1: Validate inputs
python CODE/gnss/validate_scenario.py --scenario scenario1

# Step 2: Run baseline (establishes "ground truth" performance)
python CODE/gnss/gnss_baseline_runner.py --scenario scenario1 --corrections none

# Step 3: Generate RTCM corrections
python CODE/gnss/rtcm_generator.py --scenario scenario1 --out OUTPUTS/scenario1/corrections.rtcm

# Step 4: Generate coverage map
python CODE/gnss/coverage_map_generator.py --scenario scenario1

# Step 5: Process RTCM through broadcast (see demo_broadcast_pipeline.py)
python CODE/demo_broadcast_pipeline.py
```

---

## File Structure

```
CODE/gnss/
├── __init__.py
├── gnss_baseline_runner.py   # RTKLIB rnx2rtkp wrapper
├── rtcm_generator.py         # RTCM generation utility
├── coverage_map_generator.py # Coverage map generation
├── scenario_simulator.py     # Synthetic scenario data generator
├── validate_scenario.py      # Scenario validation
├── README.md                 # This file
└── tests/                    # Unit tests
    ├── test_coverage_map.py      # Coverage map tests
    └── test_scenario_simulator.py # Scenario simulator tests

DATA/scenarios/<scenario>/
├── scenario_profile.json     # Required: file mappings + synthetic_metadata
├── rover.obs                 # Required: rover RINEX observation
├── nav.nav                   # Required: navigation file
├── base.obs                  # Optional: base station observation
├── gt.csv                    # Ground truth (generated for synthetic scenarios)
├── corrections_schedule.json # Scenario2: RTCM availability schedule
├── multipath_profile.json    # Scenario3: multipath model per epoch
└── mode_transitions.csv      # Scenario3: FIX/FLOAT/STANDALONE per epoch

OUTPUTS/<scenario>/
├── <scenario>_<mode>_<timestamp>/    # GNSS baseline run
│   ├── solution.pos                  # RTKLIB solution
│   ├── metrics.csv                   # Aggregatable metrics
│   ├── metrics.json                  # Full metrics
│   ├── run_manifest.json             # Run metadata
│   └── rtk_options.conf              # RTKLIB config used
│
└── <scenario>_coverage_<timestamp>/  # Coverage map run
    ├── coverage_map.csv              # Coverage map data
    ├── coverage_summary.json         # Coverage summary
    └── run_manifest.json             # Run metadata (scenario, timestamp, args)
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

### coverage_map_generator.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | None | Scenario name |
| `--scenario-root` | `DATA/scenarios` | Root folder for scenarios |
| `--tx-lat` | None | Transmitter latitude |
| `--tx-lon` | None | Transmitter longitude |
| `--tx-height-m` | 50 | Transmitter height (meters) |
| `--rx-height-m` | 2.0 | Receiver height (meters) |
| `--grid-res-m` | 100 | Grid resolution (meters) |
| `--max-range-km` | 10 | Maximum range (kilometers) |
| `--out-dir` | `OUTPUTS/<scenario>/coverage` | Output directory |

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

---

## Demo Notebooks

Interactive Jupyter notebooks are provided for demonstrating the RTK evaluation pipeline.

### Location

```
DOCs/notebooks/
├── scenario1_demo.ipynb   # Urban RTK evaluation
├── scenario2_demo.ipynb   # Suburban RTK evaluation
└── scenario3_demo.ipynb   # Highway/Rural (shows simulated flag)
```

### Running the Notebooks

**Option 1: Jupyter Notebook**
```bash
# From project root
cd DOCs/notebooks
jupyter notebook
# Open scenario1_demo.ipynb in browser
```

**Option 2: JupyterLab**
```bash
cd DOCs/notebooks
jupyter lab
```

**Option 3: VS Code**
- Open the `.ipynb` file directly in VS Code
- Use the Jupyter extension to run cells

**Option 4: Command-line (nbconvert)**
```bash
# Execute notebook and save output
jupyter nbconvert --to notebook --execute scenario1_demo.ipynb
```

### What Each Notebook Does

Each notebook performs 4 steps:

1. **Validate Scenario** - Runs `validate_scenario.py --strict-real <scenario>`
2. **Run Evaluation** - Runs `rtk_evaluate.py --scenario <scenario> --intent accuracy`
3. **Display Results** - Reads `comparison.csv` and displays a formatted table
4. **Plot Comparison** - Creates a bar chart of Baseline vs Optimised HPE RMS

### Expected Output Paths

After running the notebooks, outputs are located at:

| Scenario | Output Directory |
|----------|------------------|
| scenario1 | `OUTPUTS/scenario1/evaluation/eval_<timestamp>/` |
| scenario2 | `OUTPUTS/scenario2/evaluation/eval_<timestamp>/` |
| scenario3 | `OUTPUTS/scenario3/evaluation/eval_<timestamp>/` |

Each evaluation directory contains:
- `comparison.csv` - Single-row CSV with all metrics
- `comparison.json` - Full JSON with intent scoring
- `intent_score_explain.json` - Formula breakdown with actual values
- `run_manifest.json` - Run metadata and parameters

### Scenario3 and Simulated Data

The `scenario3_demo.ipynb` notebook specifically checks and displays the `simulated` flag:

- When `simulated=True`, a prominent warning is displayed
- The `intent_score_explain.json` file contains a `simulated_note` field
- Plot colors change to gray to indicate simulated data

This ensures clear provenance tracking when real RINEX data is unavailable.

### Example Output

```
======================================================================
SCENARIO: scenario1
Timestamp: 20260201T205829Z
Simulated: False
Intent: accuracy
Intent Score: 0.85
======================================================================

Metric                                    Baseline       Optimised           Delta
--------------------------------------------------------------------------------
HPE RMS (m)                                 0.0234          0.0156         -0.0078
HPE P95 (m)                                 0.0512          0.0342         -0.0170
Fix Rate (%)                                  95.2            98.7             3.5
Availability (%)                             99.1            99.8             0.7
```

---

## Acceptance Checklist (Definition of Done)

The GNSS module includes an acceptance gate script that verifies all components are working correctly.

### Running the Acceptance Check

```bash
# Full acceptance check
cd CODE/gnss
python acceptance_check.py

# Quick mode (skip slow checks)
python acceptance_check.py --quick

# Verbose output
python acceptance_check.py --verbose

# Export report to JSON
python acceptance_check.py --json acceptance_report.json
```

### Checks Performed

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| 0. Intents Config | Verify `intents.json` is valid | All 4 intents defined (accuracy, robustness, latency, balanced) |
| 1. Scenario Validation | Run `validate_scenario.py --strict-real` | scenario1 passes; scenario2/3 may fail (expected if using simulated data) |
| 2. Coverage Map | Run `coverage_map_generator.py` | Produces `coverage_map.csv`, `coverage_summary.json`, `run_manifest.json` |
| 3. Simulator Determinism | Verify simulator produces same output with same seed | Deprecated - real UrbanNav datasets now preferred |
| 4. Evaluator Outputs | Run `rtk_evaluate.py` for all scenarios | Produces `comparison.json` and `comparison.csv` |
| 5. Intent Scoring | Verify different intents produce different scores | `accuracy` score ≠ `robustness` score for same metrics |
| 6. Notebooks | Verify notebooks exist and reference correct files | All 3 notebooks exist and contain correct CLI references |
| 7. Unit Tests | Run intent-related pytest tests | All intent scoring tests pass |

### Expected Output

```
======================================================================
  GNSS MODULE - ACCEPTANCE CHECK (Definition of Done)
======================================================================
  Project Root: /path/to/ai-positioning-atsc3
  GNSS Module:  /path/to/ai-positioning-atsc3/CODE/gnss
  Quick Mode:   False

======================================================================
  CHECK 0: Intents Configuration
======================================================================
  ✅ PASS | Intents Configuration Valid (0.01s)

======================================================================
  CHECK 1: Scenario Validation (--strict-real)
======================================================================
  ✅ PASS | Scenario Validation (scenario1) (0.45s)
  ✅ PASS | Scenario Validation (scenario2) (0.32s)
  ✅ PASS | Scenario Validation (scenario3) (0.28s)

...

======================================================================
  ACCEPTANCE CHECK SUMMARY
======================================================================

  Status: ✅ ALL CHECKS PASSED
  Passed: 12/12
  Failed: 0/12
  Duration: 15.23s

  Output Paths Created:
    - OUTPUTS/scenario1/evaluation/eval_xxx/comparison.json
    - OUTPUTS/scenario1/evaluation/eval_xxx/comparison.csv
    ...

======================================================================
```

### Output Directories

After running acceptance checks, outputs are created at:

| Output | Location |
|--------|----------|
| Coverage Maps | `OUTPUTS/<scenario>/<scenario>_coverage_<timestamp>/` |
| Evaluation Results | `OUTPUTS/<scenario>/evaluation/eval_<timestamp>/` |
| Notebooks | `DOCs/notebooks/scenario{1,2,3}_demo.ipynb` |
| Acceptance Report | `acceptance_report.json` (if `--json` flag used) |

### CI/CD Integration

The acceptance check can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run GNSS Acceptance Check
  run: |
    cd CODE/gnss
    python acceptance_check.py --json acceptance_report.json
    
- name: Upload Report
  uses: actions/upload-artifact@v3
  with:
    name: acceptance-report
    path: CODE/gnss/acceptance_report.json
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "RTKLIB not found" | Install RTKLIB and set `RTKLIB_HOME` or add to PATH |
| "Scenario validation failed" | Run `validate_scenario.py --guidance` for data setup instructions |
| "Coverage generator failed" | Ensure scenario profile has `tx_lat`/`tx_lon` or `station_coordinates` |
| "Intent scoring import error" | Verify `intents.json` exists and is valid JSON |
| "Notebooks not found" | Check `DOCs/notebooks/` directory exists |
