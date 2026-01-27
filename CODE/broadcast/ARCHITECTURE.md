# Broadcast Subsystem Architecture

This document describes the complete architecture of the **broadcast subsystem**: how it takes GNSS/RTK correction data, processes it through the ATSC 3.0 pipeline, simulates channel effects, and decodes at the receiver. It covers the AI integration points and the feedback loop that enables adaptive broadcasting.

---

## 1. High-Level Overview

The broadcast stack transforms **GNSS correction data** (primarily RTCM frames) into an **ATSC 3.0-compliant OFDM signal**, simulates channel effects for robustness testing, and decodes back to verify end-to-end integrity.

### Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRANSMITTER SIDE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GNSS Corrections (RTCM)                                                   │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │  ALP Encoder  │  Link layer encapsulation (A/330)                       │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │  FEC Encoder  │  LDPC + Reed-Solomon error protection                   │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │ Frame Builder │  ATSC 3.0 baseband framing + pilots                     │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │OFDM Modulator │  IFFT + cyclic prefix → I/Q samples                     │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   I/Q Samples → SDR/RF                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Over-the-Air / Simulation
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CHANNEL SIMULATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Channel Simulator                               │   │
│   │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐ ┌───────────────┐  │   │
│   │  │Path Loss│→│Multipath │→│ Fading │→│ Doppler │→│ AWGN + Noise  │  │   │
│   │  └─────────┘ └──────────┘ └────────┘ └─────────┘ └───────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RECEIVER SIDE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Degraded I/Q Samples                                                      │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │OFDM Demodulator│  FFT + equalization + symbol detection                 │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │ Frame Parser  │  Extract payload from ATSC 3.0 frame                    │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │  FEC Decoder  │  RS + LDPC decoding with error correction               │
│   └───────┬───────┘                                                         |
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │  ALP Decoder  │  Extract RTCM payload                                   │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   Recovered RTCM Data → GNSS Engine                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Metrics & Feedback
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI ORCHESTRATOR INTEGRATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Broadcast Controller                             │   │
│   │                                                                     │   │
│   │  • Receives AI commands (intent-driven)                             │   │
│   │  • Translates intents to broadcast parameters                       │   │
│   │  • Applies configuration to pipeline                                │   │
│   │  • Collects metrics for AI feedback loop                            │   │
│   │  • Supports operator overrides (HYBRID mode)                        │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ▲           │                                  │
│                              │           │                                  │
│            Feedback Metrics  │           │  AI Commands                     │
│                              │           ▼                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       AI Orchestrator                               │   │
│   │  (Decides FEC, modulation, FFT based on intent + telemetry)         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Inventory

| Module                   | File                      | Purpose                                          |
|--------|-----------------|---------------------------|--------------------------------------------------|
| **Configuration**        |  `config.py`              | Enums, dataclasses, and configuration management |
| **ALP Encoder**          | `alpencoder.py`           | Link layer encapsulation (A/330)                 |
| **FEC Encoder**          | `fecencoder.py`           | LDPC + Reed-Solomon error protection             |
| **Frame Builder**        | `framebuilder.py`         | ATSC 3.0 baseband frame construction             |
| **OFDM Modulator**       | `ofdmmodulator.py`        | OFDM symbol generation (IFFT + CP)               |
| **Broadcast Pipeline**   | `pipeline.py`             | End-to-end transmitter orchestration             |
| **Channel Simulator**    | `channel_simulator.py`    | RF channel impairment simulation                 |
| **Decoder**              | `decoder.py`              | Complete receiver chain                          |
| **Broadcast Controller** | `broadcast_controller.py` | AI integration and adaptive control              |

---

## 3. Inputs From GNSS / Positioning

The broadcast subsystem expects the GNSS/RTK side to provide:

| Input Type             | Description                                     | Typical Size |
|------------------------|-------------------------------------------------|--------------|
| **RTCM Frames**        | Binary RTCM 3.x messages (1004/1005/1006/107x…) | 50–200 bytes |
| **Correction Bitmaps** | Regional correction grids (future)              | Variable     |
| **Telemetry**          | Satellite health, quality metrics               | ~50 bytes    |

The broadcast code treats payloads as **opaque bytes** – it does not parse RTCM semantics.

### Interface Example

```python
from broadcast.pipeline import BroadcastPipeline

# GNSS engine produces RTCM frame
rtcm_frame: bytes = gnss_engine.get_correction()

# Broadcast pipeline consumes
pipeline = BroadcastPipeline()
result = pipeline.process(rtcm_frame)

# Output: OFDM I/Q samples
iq_samples = result.signal.samples
```

---

## 4. Transmitter Components

### 4.1 ALP Layer (Link Layer)

**Component:** `ALPEncoder` (`alpencoder.py`)

**Goal:** Encapsulate GNSS correction data into **ALP packets** (A/330 link layer protocol).

#### Packet Structure

```
┌─────────────────────────────────────────────┐
│              ALP PACKET                     │
├──────────────┬──────────────────────────────┤
│ Header (2B)  │ Payload (variable)           │
├──────────────┼──────────────────────────────┤
│ Type (3 bits)│ RTCM/bitmap/telemetry data   │
│ Length (13b) │                              │
└──────────────┴──────────────────────────────┘
```

#### API

| Method                                 | Input         | Output            |
|----------------------------------------|---------------|-------------------|
| `encode_rtcm(rtcm_frame)`              | `bytes`       | `ALPPacket`       |
| `encode_bitmap(data, tile_id)`         | `bytes, int`  | `ALPPacket`       |
| `encode_batch(rtcm_list, bitmap_list)` | `List[bytes]` | `List[ALPPacket]` |
| `decode(data)`                         | `bytes`       | `ALPPacket`       |

---

### 4.2 FEC Encoder (Error Protection)

**Component:** `FECEncoder` (`fecencoder.py`)

**Goal:** Add forward error correction for reliable reception under harsh channel conditions.

#### Coding Scheme

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEC ENCODING                                 │
│                                                                 │
│  Input Data ──► [LDPC Encoder] ──► [RS Encoder] ──► FEC Data    │
│    (k bits)      (inner code)      (outer code)     (n bits)    │
│                                                                 │
│  Code Rate = k/n = LDPC_rate × RS_rate                          │
│  Overhead = (n-k)/k × 100%                                      │
└─────────────────────────────────────────────────────────────────┘
```

#### LDPC Code Rates (ATSC 3.0)

| Rate         | Overhead | Use Case            |
|--------------|----------|---------------------|
| `RATE_2_15`  | 650%     | Extreme robustness  |
| `RATE_4_15`  | 275%     | High robustness     |
| `RATE_6_15`  | 150%     | Balanced            |
| `RATE_8_15`  | 88%      | Default             |
| `RATE_10_15` | 50%      | Bandwidth efficient |
| `RATE_12_15` | 25%      | Maximum efficiency  |

#### API

| Method                 | Input                | Output           |
|------------------------|----------------------|------------------|
| `encode(data, config)` | `bytes, FECConfig`   | `FECEncodedData` |
| `get_stats()`          |                    – | `Dict[str, Any]` |

---

### 4.3 Frame Builder (Baseband Framing)

**Component:** `FrameBuilder` (`framebuilder.py`)

**Goal:** Construct ATSC 3.0-compliant baseband frames with pilots and signaling.

#### Frame Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATSC 3.0 FRAME                               │
├─────────────────────────────────────────────────────────────────┤
│  Bootstrap    │  Preamble    │  Payload Symbols                 │
│  (sync)       │  (L1 sig)    │  (FEC data + pilots)             │
├───────────────┼──────────────┼──────────────────────────────────┤
│    Fixed      │  FFT-based   │  Interleaved + pilot-inserted    │
└─────────────────────────────────────────────────────────────────┘
```

#### Configuration Parameters

| Parameter             | Options              | Effect                           |
|-----------------------|----------------------|----------------------------------|
| **FFT Size**          | 8K, 16K, 32K         | Spectral resolution vs. latency  |
| **Guard Interval**    | 1/4, 1/8, 1/16, 1/32 | Multipath tolerance vs. overhead |
| **Pilot Pattern**     | PP1–PP8              | Channel estimation accuracy      |
| **Time Interleaving** | 0–1024 symbols       | Burst error resilience           |

---

### 4.4 OFDM Modulator (Waveform Generation)

**Component:** `OFDMModulator` (`ofdmmodulator.py`)

**Goal:** Generate OFDM I/Q samples from baseband frames.

#### OFDM Symbol Generation

```
┌─────────────────────────────────────────────────────────────────┐
│                 OFDM MODULATION                                 │
│                                                                 │
│  Bits ──► Constellation ──► Carrier ──► IFFT ──► Add CP ──► I/Q │
│           Mapping           Allocation                          │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │   QPSK   │    │  Pilot   │    │   IFFT   │    │  Cyclic  │   │
│  │  16-QAM  │ ──►│  Insert  │ ──►│  (8K/16K │ ──►│  Prefix  │   │
│  │  64-QAM  │    │          │    │   /32K)  │    │          │   │
│  │ 256-QAM  │    │          │    │          │    │          │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Modulation Schemes

| Scheme      | Bits/Symbol | SNR Required | Use Case           |
|-------------|-------------|--------------|--------------------|
| **QPSK**    | 2           | ~5 dB        | Maximum robustness |
| **16-QAM**  | 4           | ~12 dB       | Balanced           |
| **64-QAM**  | 6           | ~18 dB       | High throughput    |
| **256-QAM** | 8           | ~24 dB       | Maximum efficiency |

---

### 4.5 Broadcast Pipeline (Orchestrator)

**Component:** `BroadcastPipeline` (`pipeline.py`)

**Goal:** Single entry point for end-to-end transmission.

#### Pipeline Flow

```python
# Complete transmission flow
result = pipeline.process(rtcm_data)

# Internally:
# 1. ALP encoding (optional)
# 2. FEC encoding
# 3. Frame building
# 4. OFDM modulation
# 5. Statistics collection
```

#### API

| Method                                          | Description               |
|-------------------------------------------------|---------------------------|
| `process(data, config)`                         | Process single payload    |
| `process_batch(data_list)`                      | Process multiple payloads |
| `process_with_adaptive_fec(data, overhead_pct)` | Adaptive FEC overhead     |
| `get_stats()`                                   | Get processing statistics |
| `reset_stats()`                                 | Reset counters            |

#### Output: BroadcastResult

```python
@dataclass
class BroadcastResult:
    original_data: bytes          # Input RTCM
    alp_packet: ALPPacket         # Link layer packet
    fec_result: FECEncodedData    # FEC-protected data
    frame: ATSCFrame              # Baseband frame
    signal: OFDMSignal            # I/Q samples
    metrics: Dict[str, Any]       # Processing metrics
```

---

## 5. Channel Simulation

**Component:** `ChannelSimulator` (`channel_simulator.py`)

**Goal:** Simulate real-world RF channel effects for robustness testing.

### Channel Effects

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHANNEL MODEL                                │
│                                                                 │
│  TX Signal ──► Path ──► Multi ──► Fading ──► Doppler ──► Noise  │
│               Loss     path                            (AWGN)   │
│                                                                 │
│  Effects:                                                       │
│  • Path Loss: Distance-based attenuation                        │
│  • Multipath: Echoes from reflections (urban/indoor)            │
│  • Fading: Rayleigh (urban), Rician (rural)                     │
│  • Doppler: Motion-induced frequency shift                      │
│  • AWGN: Thermal noise (controlled by SNR)                      │
│  • Impulse: Electrical interference                             │
└─────────────────────────────────────────────────────────────────┘
```

### Environment Presets

| Preset           | SNR   | Multipath    | Fading   | Doppler | Use Case           |
|------------------|-------|--------------|----------|---------|--------------------|
| **RURAL**        | 25 dB | Minimal      | None     | No      | Open sky           |
| **SUBURBAN**     | 20 dB | Moderate     | Rician   | No      | Residential        |
| **URBAN**        | 15 dB | Severe       | Rayleigh | No      | Dense buildings    |
| **URBAN_CANYON** | 10 dB | Very Severe  | Rayleigh | Yes     | Street canyon      |
| **HIGHWAY**      | 18 dB | Moderate     | Rician   | High    | Vehicle @ 120 km/h |
| **INDOOR**       | 22 dB | Short delays | Rayleigh | No      | Inside building    |

### Standard Multipath Profiles (ITU/3GPP)

| Profile      | Taps | Max Delay | Use Case     |
|--------------|------|-----------|--------------|
| Pedestrian A | 4    | 0.41 µs   | Walking user |
| Pedestrian B | 6    | 3.7 µs    | Walking user |
| Vehicular A  | 6    | 2.51 µs   | Vehicle      |
| Vehicular B  | 6    | 20 µs     | Fast vehicle |

### API

```python
# Quick preset usage
simulator = ChannelSimulator.from_preset("urban", snr_override=12.0)
result = simulator.apply(ofdm_signal)

# Custom configuration
config = ChannelConfig(
    snr_db=15.0,
    multipath_enabled=True,
    num_paths=6,
    fading_type=FadingType.RAYLEIGH,
    doppler_hz=50.0
)
simulator = ChannelSimulator(config)
result = simulator.apply(signal)

# SNR sweep for BER curves
results = simulator.sweep_snr(signal, [5, 10, 15, 20, 25])
```

---

## 6. Receiver Components

**Component:** `BroadcastDecoder` (`decoder.py`)

**Goal:** Recover original RTCM data from degraded OFDM signal.

### Decoder Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECEIVER CHAIN                               │
│                                                                 │
│  RX Signal ──► Sync ──► OFDM ──► Frame ──► FEC ──► ALP ──► Data │
│                        Demod    Parse    Decode   Decode        │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Timing  │ │   FFT    │ │ Channel  │ │  Symbol  │            │
│  │   Sync   │→│ + Remove │→│Equalizer │→│Detection │            │
│  │          │ │    CP    │ │          │ │          │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Frame   │ │   LDPC   │ │   RS     │ │   ALP    │            │
│  │  Parser  │→│  Decode  │→│  Decode  │→│  Decode  │ ──► RTCM   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Decoder Components

| Component            | Class              | Function                             |
|----------------------|--------------------|--------------------------------------|
| **OFDM Demodulator** | `OFDMDemodulator`  | FFT, equalization, symbol detection  |
| **Frame Parser**     | `FrameParser`      | Extract payload from frame structure |
| **FEC Decoder**      | `FECDecoder`       | RS + LDPC error correction           |
| **ALP Decoder**      | `ALPDecoder`       | Extract original payload             |
| **Full Decoder**     | `BroadcastDecoder` | End-to-end receiver                  |

### Decoder Output

```python
@dataclass
class DecodeResult:
    success: bool                    # Overall decode success
    recovered_data: Optional[bytes]  # Original payload
    
    # Per-stage results
    demod_result: DemodulationResult
    frame_result: FrameParseResult
    fec_result: FECDecodeResult
    alp_result: ALPDecodeResult
    
    # Quality metrics
    metrics: DecodeMetrics           # SNR, BER, errors corrected
```

### Quality Metrics

| Metric             | Description             | Good Value |
|--------------------|-------------------------|------------|
| `estimated_snr_db` | Signal-to-noise ratio   | > 15 dB    |
| `evm_percent`      | Error vector magnitude  | < 10%      |
| `estimated_ber`    | Bit error rate          | < 1e-6     |
| `errors_corrected` | FEC corrections made    | Low        |
| `data_integrity`   | CRC/checksum match      | True       |

---

## 7. AI Integration: Broadcast Controller

**Component:** `BroadcastController` (`broadcast_controller.py`)

**Goal:** Bridge between AI Orchestrator and broadcast pipeline, enabling intent-driven adaptive broadcasting.

### Operating Modes

| Mode              | Description                        | Use Case             |
|-------------------|------------------------------------|----------------------|
| **TRADITIONAL**   | Static configuration               | Manual operation     |
| **AI_CONTROLLED** | Dynamic AI-driven config           | Autonomous operation |
| **HYBRID**        | AI suggestions + operator override | Supervised operation |

### AI Command Format

```json
{
    "command_id": "uuid",
    "timestamp": "ISO-8601",
    "intent": "maximize_reliability",
    "priority": "high",
    "broadcast_config": {
        "fec_ldpc_rate": "RATE_6_15",
        "fec_rs_symbols": 24,
        "fec_overhead_pct": 40.0,
        "modulation": "QPSK",
        "fft_size": "FFT_8K",
        "guard_interval": "GI_1_4",
        "update_frequency_hz": 5.0,
        "tile_resolution": "high",
        "plp_mode": "mobile"
    },
    "constraints": {
        "max_latency_ms": 100,
        "min_reliability_pct": 99.0,
        "max_bandwidth_hz": 6000000
    }
}
```

### Supported Intents

| Intent                 | FEC Strategy                    | Modulation | Use Case       |
|------------------------|---------------------------------|------------|----------------|
| `maximize_reliability` | RATE_4_15, 60% overhead         | QPSK       | Urban canyon   |
| `minimize_bandwidth`   | RATE_10_15, 15% overhead        | 64-QAM     | Rural/clear    |
| `maximize_accuracy`    | RATE_6_15, high update rate     | QPSK       | Precision apps |
| `low_latency`          | RATE_8_15, minimal interleaving | 16-QAM     | Real-time      |
| `urban_robust`         | RATE_4_15, long guard           | QPSK       | Dense urban    |
| `rural_efficient`      | RATE_8_15, short guard          | 16-QAM     | Open rural     |

### Controller API

```python
from broadcast.broadcast_controller import (
    BroadcastController,
    ControllerMode,
    AIBroadcastCommand,
    Intent
)

# Create AI-controlled controller
controller = BroadcastController(mode=ControllerMode.AI_CONTROLLED)

# Apply AI command
result = controller.apply_command({
    "intent": "maximize_reliability",
    "broadcast_config": {
        "fec_ldpc_rate": "RATE_6_15",
        "modulation": "QPSK"
    }
})

# Process data
broadcast_result = controller.process(rtcm_data)

# Get metrics for AI feedback
feedback = controller.get_ai_feedback()
```

### Feedback Metrics (to AI Orchestrator)

```python
{
    "timestamp": "2026-01-27T10:30:00",
    "performance": {
        "latency_ms": 45.2,
        "processing_time_ms": 12.3,
        "spectral_efficiency": 1.8
    },
    "quality": {
        "ber": 1.2e-6,
        "reliability_pct": 99.8
    },
    "throughput": {
        "bytes_processed": 150000,
        "frames_transmitted": 1250
    },
    "configuration": {
        "mode": "ai_controlled",
        "intent": "maximize_reliability",
        "fec_rate": "RATE_6_15",
        "modulation": "QPSK"
    }
}
```

---

## 8. End-to-End Data Flow

### Complete Processing Chain

```
                     TRANSMITTER
                         │
    RTCM Frame ──────────┼─────────────────────────────────────┐
         │               │                                     │
         ▼               │                                     │
    [ALP Encoder]        │  Adds link layer header             │
         │               │                                     │
         ▼               │                                     │
    [FEC Encoder]        │  LDPC + RS protection               │
         │               │  (configured by AI intent)          │
         ▼               │                                     │
    [Frame Builder]      │  ATSC 3.0 framing                   │
         │               │  (FFT size, pilots per AI)          │
         ▼               │                                     │
    [OFDM Modulator]     │  I/Q sample generation              │
         │               │  (modulation per AI)                │
         ▼               │                                     │
    OFDM Signal ─────────┼─────────────────────────────────────┘
                         │
                     CHANNEL
                         │
    ┌────────────────────┼─────────────────────────────────────┐
    │  [Path Loss]       │  Distance attenuation               │
    │       │            │                                     │
    │       ▼            │                                     │
    │  [Multipath]       │  Echoes, reflections                │
    │       │            │                                     │
    │       ▼            │                                     │
    │  [Fading]          │  Rayleigh/Rician                    │
    │       │            │                                     │
    │       ▼            │                                     │
    │  [Doppler]         │  Motion shift                       │
    │       │            │                                     │
    │       ▼            │                                     │
    │  [AWGN]            │  Thermal noise                      │
    └────────────────────┼─────────────────────────────────────┘
                         │
    Degraded Signal ─────┼─────────────────────────────────────┐
                         │                                     │
                     RECEIVER                                  │
                         │                                     │
    [OFDM Demodulator]   │  FFT + equalization                 │
         │               │                                     │
         ▼               │                                     │
    [Frame Parser]       │  Extract payload                    │
         │               │                                     │
         ▼               │                                     │
    [FEC Decoder]        │  Error correction                   │
         │               │                                     │
         ▼               │                                     │
    [ALP Decoder]        │  Extract RTCM                       │
         │               │                                     │
         ▼               │                                     │
    Recovered RTCM ──────┼─────────────────────────────────────┘
                         │
                         ▼
                    GNSS Engine
```

---

## 9. Integration Points

### 9.1 GNSS Engine → Broadcast

```python
# GNSS engine provides RTCM data
from gnss.rtcm_handler import RTCMHandler

rtcm_handler = RTCMHandler()
rtcm_frame = rtcm_handler.get_correction()

# Broadcast pipeline consumes
from broadcast.pipeline import BroadcastPipeline

pipeline = BroadcastPipeline()
result = pipeline.process(rtcm_frame)
```

### 9.2 AI Orchestrator → Broadcast Controller

```python
# AI Orchestrator sends commands
from broadcast.broadcast_controller import BroadcastController

controller = BroadcastController(mode=ControllerMode.AI_CONTROLLED)

# AI sends configuration command
ai_command = {
    "intent": "maximize_reliability",
    "broadcast_config": {"fec_ldpc_rate": "RATE_6_15"}
}
controller.apply_command(ai_command)

# Process data with AI-selected config
result = controller.process(rtcm_data)

# Report back to AI
feedback = controller.get_ai_feedback()
```

### 9.3 Handoff Test Integration

```python
# Complete handoff test
from broadcast.pipeline import BroadcastPipeline
from broadcast.channel_simulator import ChannelSimulator
from broadcast.decoder import BroadcastDecoder

# Transmit
pipeline = BroadcastPipeline(config=broadcast_config)
tx_result = pipeline.process(rtcm_data)

# Channel simulation
channel = ChannelSimulator.from_preset("urban")
channel_result = channel.apply(tx_result.signal.samples)

# Receive
decoder = BroadcastDecoder(config=decoder_config)
rx_result = decoder.decode(channel_result.signal)

# Verify
if rx_result.success:
    assert rx_result.recovered_data == rtcm_data
```

---

## 10. Configuration Reference

### BroadcastConfig Parameters

| Parameter           | Type               | Default     | Description           |
|---------------------|--------------------|-------------|-----------------------|
| `fec_ldpc_rate`     | `FECCodeRate`      | `RATE_8_15` | LDPC code rate        |
| `fec_rs_symbols`    | `int`              | 16          | Reed-Solomon symbols  |
| `fec_overhead_pct`  | `float`            | 15.0        | Target FEC overhead % |
| `modulation`        | `ModulationScheme` | `QPSK`      | Constellation         |
| `fft_size`          | `FFTSize`          | `FFT_8K`    | OFDM FFT size         |
| `guard_interval`    | `GuardInterval`    | `GI_1_8`    | Cyclic prefix ratio   |
| `pilots_enabled`    | `bool`             | True        | Enable pilot carriers |
| `pilot_pattern`     | `PilotPattern`     | `PP3`       | Pilot density         |
| `time_interleaving` | `bool`             | True        | Enable interleaving   |
| `use_alp`           | `bool`             | True        | Use ALP encapsulation |

### ChannelConfig Parameters

| Parameter           | Type         | Default    | Description           |
|---------------------|--------------|------------|-----------------------|
| `snr_db`            | `float`      | 20.0       | Signal-to-noise ratio |
| `multipath_enabled` | `bool`       | True       | Enable multipath      |
| `num_paths`         | `int`        | 4          | Number of paths       |
| `max_delay_us`      | `float`      | 10.0       | Maximum delay spread  |
| `fading_type`       | `FadingType` | `RAYLEIGH` | Fading model          |
| `doppler_hz`        | `float`      | 0.0        | Doppler frequency     |
| `path_loss_enabled` | `bool`       | False      | Enable path loss      |

---

## 11. Performance Characteristics

### Latency Budget

| Stage              | Typical Time  | Notes                |
|--------------------|---------------|----------------------|
| ALP Encoding       | < 1 ms        | Fast header addition |
| FEC Encoding       | 2-10 ms       | Depends on code rate |
| Frame Building     | 1-5 ms        | Pilot insertion      |
| OFDM Modulation    | 5-20 ms       | FFT-dependent        |
| **Total TX**       | **10-40 ms**  |                      | 
| Channel Simulation | 5-15 ms       | For testing only     |
| OFDM Demodulation  | 10-30 ms      | FFT + equalization   |
| FEC Decoding       | 5-20 ms       | Iterative LDPC       |
| **Total E2E**      | **30-100 ms** |                      |

### Spectral Efficiency

| Modulation | Code Rate | Bits/Hz/s |
|------------|-----------|-----------|
| QPSK       | 8/15      | 1.07      |
| QPSK       | 6/15      | 0.80      |
| 16-QAM     | 8/15      | 2.13      |
| 64-QAM     | 8/15      | 3.20      |

---

## 12. Files and Dependencies

### Module Dependencies

```
broadcast/
├── __init__.py
├── config.py           ← Base (no dependencies)
├── alpencoder.py       ← Uses: config
├── fecencoder.py       ← Uses: config
├── framebuilder.py     ← Uses: config
├── ofdmmodulator.py    ← Uses: config, numpy, scipy
├── pipeline.py         ← Uses: all above
├── channel_simulator.py ← Uses: config, numpy, scipy
├── decoder.py          ← Uses: config, numpy, scipy
├── broadcast_controller.py ← Uses: pipeline, config
└── ARCHITECTURE.md     ← This document
```

### External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20 | Array operations |
| `scipy` | ≥1.7 | Signal processing |
| `galois` | ≥0.3 | Reed-Solomon codec |

---

## 13. Testing and Validation

### Unit Test Coverage

```bash
# Run all broadcast tests
pytest tests/test_broadcast/ -v

# Individual module tests
pytest tests/test_broadcast/test_alpencoder.py
pytest tests/test_broadcast/test_fecencoder.py
pytest tests/test_broadcast/test_pipeline.py
pytest tests/test_broadcast/test_decoder.py
pytest tests/test_broadcast/test_channel_simulator.py
```

### End-to-End Validation

```python
# Loopback test: TX → Channel → RX → Verify
from broadcast.pipeline import BroadcastPipeline
from broadcast.channel_simulator import ChannelSimulator
from broadcast.decoder import BroadcastDecoder

def validate_end_to_end(data: bytes, snr_db: float = 20.0) -> bool:
    # Transmit
    pipeline = BroadcastPipeline()
    tx = pipeline.process(data)
    
    # Channel
    channel = ChannelSimulator.from_preset("suburban", snr_db)
    degraded = channel.apply(tx.signal.samples)
    
    # Receive
    decoder = BroadcastDecoder()
    rx = decoder.decode(degraded.signal)
    
    return rx.success and rx.recovered_data == data
```

---

## 14. Summary

The broadcast subsystem provides a complete, AI-adaptive ATSC 3.0 transmission and reception chain:

| Feature                | Implementation                       |
|------------------------|--------------------------------------|
| **Link Layer**         | ALP encapsulation (A/330)            |
| **Error Protection**   | LDPC + Reed-Solomon                  |
| **Physical Layer**     | OFDM with ATSC 3.0 framing           |
| **Channel Simulation** | Multipath, fading, AWGN, Doppler     |
| **Receiver**           | Complete decode chain                |
| **AI Integration**     | Intent-driven adaptive configuration |
| **Feedback Loop**      | Metrics for AI learning              | 

The system is **deterministic** in its signal processing but **adaptive** in its configuration, with AI driving the selection of FEC rates, modulation, and other parameters based on channel conditions and service requirements.