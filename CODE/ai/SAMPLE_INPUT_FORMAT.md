# Training Data Input Format Guide

## Overview
The AI model expects **50-dimensional feature vectors** representing GNSS telemetry and channel state information. For 10k samples, you need to prepare data in one of the following formats.

---

## File Format Options

### Option 1: CSV Format (Recommended for 10k samples)

**File Extension:** `.csv`

**Structure:**
- Header row with feature names (optional but recommended)
- Data rows with 50 comma-separated values
- One sample per row
- Total rows: 10,001 (1 header + 10,000 data rows)

**Example File:** `training_data.csv`

```
signal_strength_0,signal_strength_1,signal_strength_2,signal_strength_3,carrier_phase_0,carrier_phase_1,carrier_phase_2,carrier_phase_3,pseudorange_error_0,pseudorange_error_1,pseudorange_error_2,pseudorange_error_3,doppler_shift_0,doppler_shift_1,doppler_shift_2,doppler_shift_3,tracking_lock_0,tracking_lock_1,tracking_lock_2,tracking_lock_3,received_power_0,received_power_1,received_power_2,received_power_3,carrier_power_0,carrier_power_1,carrier_power_2,carrier_power_3,noise_power_0,noise_power_1,snr_db_0,snr_db_1,snr_db_2,snr_db_3,cnr_db_hz_0,cnr_db_hz_1,cnr_db_hz_2,cnr_db_hz_3,multipath_indicator_0,multipath_indicator_1,urban_density_estimate,blockage_probability,multipath_likelihood,shadow_fading_estimate,nlos_probability,tunnel_probability,time_of_day_normalized,vehicle_speed_kmh,heading_deg,gnss_availability_pct
12.5,11.8,13.2,10.9,0.125,0.142,0.118,0.165,0.045,0.052,0.038,0.061,125.3,128.5,122.1,130.2,1.0,1.0,0.95,1.0,45.2,43.1,46.8,41.5,32.1,31.4,33.8,30.2,8.5,8.2,18.5,17.2,19.1,16.8,35.2,34.1,36.5,33.2,0.15,0.12,0.65,0.35,0.45,0.28,0.52,0.18,0.5,65.5,180.0,0.95
13.1,12.2,12.8,11.5,0.135,0.152,0.128,0.175,0.050,0.055,0.042,0.065,126.5,129.2,123.5,131.1,1.0,0.98,0.92,0.99,46.1,44.2,47.5,42.3,33.2,32.1,34.5,31.1,8.7,8.4,19.2,18.1,19.8,17.5,36.1,35.2,37.2,34.1,0.18,0.14,0.68,0.32,0.48,0.30,0.55,0.20,0.52,70.2,185.5,0.96
...
(10,000 more rows)
```

**Size:** ~2.5 MB for 10k samples (50 features × float values)

---

### Option 2: NumPy Binary Format (Fastest Loading)

**File Extension:** `.npy` or `.npz`

**Structure:**
- NumPy array shape: `(10000, 50)` for 10k samples
- Data type: `float32` or `float64`
- Single binary file (loads much faster than CSV)

**Example Python Code to Create:**

```python
import numpy as np

# Load your data (shape: 10000 × 50)
data = np.random.rand(10000, 50)  # Example synthetic data

# Save as binary format
np.save('training_data.npy', data)

# Or save as compressed:
np.savez_compressed('training_data.npz', features=data)
```

**Size:** ~2.0 MB (uncompressed) or ~0.5 MB (compressed .npz)

---

### Option 3: HDF5 Format (For Large Datasets)

**File Extension:** `.h5` or `.hdf5`

**Structure:**
- Hierarchical data format with metadata
- Can include both X (features) and y (targets)
- Supports efficient partial loading

**Example Python Code:**

```python
import h5py
import numpy as np

# Create file
with h5py.File('training_data.h5', 'w') as f:
    # Create datasets
    X = np.random.rand(10000, 50)  # Feature data
    y = np.random.rand(10000, 5)   # Target data
    
    f.create_dataset('features', data=X, compression='gzip')
    f.create_dataset('targets', data=y, compression='gzip')
    f.attrs['n_samples'] = 10000
    f.attrs['n_features'] = 50
```

**Size:** ~0.6 MB (compressed)

---

## Feature Descriptions (50 Input Features)

### Subchannel State (20 features)
- **signal_strength_0 to signal_strength_3:** Signal power level per subchannel (dBm)
- **carrier_phase_0 to carrier_phase_3:** Carrier phase per subchannel (radians, normalized 0-1)
- **pseudorange_error_0 to pseudorange_error_3:** Pseudorange error per subchannel (meters, normalized)
- **doppler_shift_0 to doppler_shift_3:** Doppler shift per subchannel (Hz, normalized)
- **tracking_lock_0 to tracking_lock_3:** PLL/DLL lock status (1.0=locked, 0.0=unlocked)

### Power Measurements (10 features)
- **received_power_0 to received_power_3:** Raw received power per subchannel (dBm, normalized 0-100)
- **carrier_power_0 to carrier_power_3:** Carrier power per subchannel (dBm, normalized 0-100)
- **noise_power_0 to noise_power_1:** Noise floor estimates (dBm, normalized 0-100)

### SNR Estimates (10 features)
- **snr_db_0 to snr_db_3:** Signal-to-Noise Ratio per subchannel (dB, normalized 0-50)
- **cnr_db_hz_0 to cnr_db_hz_3:** Carrier-to-Noise Ratio per subchannel (dB-Hz, normalized 0-50)
- **multipath_indicator_0 to multipath_indicator_1:** Multipath presence indicators (0-1)

### Environmental/Contextual (10 features)
- **urban_density_estimate:** Urban canyon density (0-1, where 1=dense urban)
- **blockage_probability:** Likelihood of signal blockage (0-1)
- **multipath_likelihood:** Probability of multipath reflections (0-1)
- **shadow_fading_estimate:** Shadow fading severity (0-1)
- **nlos_probability:** Non-line-of-sight probability (0-1)
- **tunnel_probability:** Likelihood in tunnel/covered area (0-1)
- **time_of_day_normalized:** Time of day (0-1, 0=midnight, 0.5=noon)
- **vehicle_speed_kmh:** Vehicle speed (0-200 kmh, normalized 0-1)
- **heading_deg:** Vehicle heading (0-360 degrees, normalized 0-1)
- **gnss_availability_pct:** Satellite availability percentage (0-1)

---

## Value Ranges

All features should be **normalized to [0, 1]** or **[-1, 1]** range:

| Feature Category | Min | Max | Normalized |
|---|---|---|---|
| Signal Strength | -160 dBm | -80 dBm | (value + 160) / 80 |
| Carrier Phase | 0 | 2π | value / (2π) |
| Pseudorange Error | 0 | 100 m | value / 100 |
| Doppler Shift | -5000 Hz | 5000 Hz | (value + 5000) / 10000 |
| Tracking Lock | 0 | 1 | as-is |
| Power Measurements | 0 | 100 dBm | value / 100 |
| SNR/CNR | 0 | 50 dB | value / 50 |
| Environmental | 0 | 1 | as-is |
| Speed | 0 | 200 kmh | value / 200 |
| Heading | 0 | 360° | value / 360 |
| Availability | 0 | 100% | value / 100 |

---

## Target Output Labels (Optional)

If you have ground truth labels, include a **target file with 5 output features per sample:**

**Output Features (y):**

1. **redundancy_ratio** (1.0 - 5.0): Number of redundant signals needed
2. **spectrum_mbps** (0.1 - 2.0): Required bandwidth in Mbps
3. **availability_pct** (0.80 - 0.99): Target availability percentage
4. **convergence_time_sec** (10 - 60): Time to convergence in seconds
5. **accuracy_hpe_cm** (1.0 - 50.0): Horizontal Position Error in cm

**Example targets.csv:**
```
redundancy_ratio,spectrum_mbps,availability_pct,convergence_time_sec,accuracy_hpe_cm
2.5,0.85,0.95,25.5,12.3
3.1,1.1,0.92,32.1,18.5
...
```

---

## Data Loading Code

### Load CSV (10k samples)
```python
from ai_data_preprocessor import TelemetryLoader

loader = TelemetryLoader()
X = loader.load_from_csv('training_data.csv', n_rows=10000)
# Shape: (10000, 50)
```

### Load NumPy
```python
X = loader.load_from_numpy('training_data.npy')
# Shape: (10000, 50)
```

### Load HDF5
```python
import h5py

with h5py.File('training_data.h5', 'r') as f:
    X = f['features'][:]
    y = f['targets'][:]
```

---

## Recommended Setup for 10k Samples

**Best Practice Configuration:**

```
training_data/
├── features.npy              # 10k × 50 (2.0 MB)
├── targets.npy              # 10k × 5  (0.2 MB)
├── features_info.json       # Metadata
└── README.md               # Documentation
```

**Size Estimation:**
- 10,000 samples × 50 features = 500,000 values
- At 4 bytes/float32 = **2.0 MB** (NumPy) or **2.5 MB** (CSV)
- With 5 targets: **+0.2 MB**

**Loading Time:**
- CSV: ~500ms
- NumPy (.npy): ~50ms
- NumPy compressed (.npz): ~100ms
- HDF5: ~75ms

---

## Validation Checklist

Before training, verify:

- [ ] Exactly 10,000 samples
- [ ] Exactly 50 input features per sample
- [ ] All values in [0, 1] range (normalized)
- [ ] No NaN or Inf values
- [ ] No missing data
- [ ] Target values (y) in expected ranges
- [ ] Data type: float32 or float64
- [ ] File size: ~2-2.5 MB for 10k samples

---

## Example: Create Synthetic 10k Sample Dataset

```python
import numpy as np
from ai_data_preprocessor import TelemetryLoader

# Generate 10k synthetic samples
loader = TelemetryLoader()
X = loader.generate_synthetic(num_samples=10000, seed=42)

# Save as NumPy (recommended)
np.save('training_features.npy', X)

# Or save as CSV
np.savetxt('training_features.csv', X, delimiter=',', header=','.join([f'feature_{i}' for i in range(50)]))

# Optional: Generate and save targets
y = np.random.rand(10000, 5)
y[:, 0] = y[:, 0] * 4.0 + 1.0      # redundancy_ratio (1-5)
y[:, 1] = y[:, 1] * 1.9 + 0.1      # spectrum_mbps (0.1-2.0)
y[:, 2] = y[:, 2] * 0.19 + 0.8     # availability_pct (0.8-0.99)
y[:, 3] = y[:, 3] * 50.0 + 10.0    # convergence_time_sec (10-60)
y[:, 4] = y[:, 4] * 49.0 + 1.0     # accuracy_hpe_cm (1-50)

np.save('training_targets.npy', y)
```

---

## Questions?

Refer to:
- [ai_data_preprocessor.py](ai_data_preprocessor.py) - Data loading & preprocessing
- [main.py](main.py) - Training pipeline example
- [config.py](config.py) - Configuration defaults
