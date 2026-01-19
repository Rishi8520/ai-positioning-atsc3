# AI-Positioning PoC: Complete Technical & Organizational Guide

**Document Version:** 1.0  
**Last Updated:** January 13, 2026  
**Status:** Complete Reference Document  
---

## 📌 Executive Summary

This document provides a complete reference for the **AI-Positioning Proof-of-Concept (PoC)**, a system that uses AI-driven broadcast optimization to maintain centimeter-level GPS positioning accuracy in extreme environments (tunnels, urban canyons, signal blockage zones).

**Key Innovation:** Real-time AI agent dynamically adjusts broadcast parameters based on fleet telemetry, ensuring vehicles maintain RTK FIX (centimeter accuracy) even during GNSS signal loss events.

**Team Structure:** 3 specialized engineers (GNSS, Broadcast, AI/ML)  
**Duration:** 12 weeks, 4 phases  
**Total Cost:** $0 (100% open-source tools)  
**Expected Outcome:** Production-ready PoC ready for deployment on Qualcomm AI Hub

---

---

# PART 1: UNDERSTANDING THE PROBLEM STATEMENT

## 1.1 The Core Problem We're Solving

### Real-World Scenario: Why This Matters

Imagine an autonomous vehicle driving through a city:
- **In open sky (highway):** GPS works perfectly → ±1.5 cm accuracy
- **Entering tunnel:** GPS signals blocked → accuracy drops to ±5-10 meters
- **In tunnel (1 minute of driving):** GPS completely unavailable → system must rely on dead-reckoning

**The Challenge:**
- Autonomous vehicles need centimeter-level precision for lane-keeping, collision avoidance
- GPS alone cannot provide this in urban/tunnel environments
- Current solutions: expensive LiDAR ($50K+), high-precision inertial systems ($30K+), or cellular networks (spotty coverage)

**Our Solution:**
We're building a **broadcast-based positioning system** that:
1. Uses stationary base stations with known locations to measure GPS errors
2. Encodes these corrections into RTCM format
3. Broadcasts corrections over ATSC 3.0 (free spectrum, 30-50 km range)
4. Uses AI to adapt the broadcast in real-time based on what vehicles experience
5. Helps vehicles achieve centimeter accuracy even in degraded environments

---

## 1.2 Current State of GPS Technology

### Standard GPS (What Everyone Uses)
```
Accuracy:       ±5-10 meters
Why Limited:    Code-phase measurements only
Use Case:       "You are somewhere in this city block"
Problem:        Not good enough for autonomous vehicles
```

### RTK GPS (Real-Time Kinematic - What We're Building Toward)
```
Accuracy:       ±1.5-2 cm (centimeter-level!)
How:            Uses carrier-phase + corrections from base station
Range Limit:    ±20-50 km from base station
Problem:        Requires constant base station signal
Solution:       We broadcast the signal using ATSC 3.0 (TV spectrum)
```

### Why ATSC 3.0 for Broadcasting?

```
┌────────────────────────────────────────────────────────────┐
│              ATSC 3.0 (Next-Gen Digital TV)               │
├────────────────────────────────────────────────────────────┤
│ Broadcast Range:    30-50 km (covers entire metropolitan) │
│ Bandwidth:          6 MHz (plenty of room)                │
│ Spectrum:           Free (already licensed for TV)        │
│ Data Capacity:      5-57 Mbps depending on robustness    │
│ Latency:            5-10 seconds (acceptable for RTK)     │
│ Reach:              50+ million vehicles in US alone      │
│                                                            │
│ Compare to:                                                │
│ • Cellular (LTE/5G): $$$, limited spectrum               │
│ • WiFi: <1 km range, need constant network               │
│ • Satellite: 1 second latency, coverage gaps             │
│                                                            │
│ Why ATSC 3.0?                                              │
│ • It's BROADCAST (one transmitter → thousands of cars)    │
│ • Free spectrum (no licensing fees)                       │
│ • Mobile reception (works at highway speeds)              │
│ • Robust (handles urban multipath, fading)                │
└────────────────────────────────────────────────────────────┘
```

---

## 1.3 Why AI is Needed (The Smart Part)

### Without AI: Static Broadcasting
```
Traditional Approach:
  Base Station → RTCM Corrections → ATSC 3.0 Transmitter → Always ON
  
Problems:
  ❌ Transmitter wastes power (always broadcasting at full strength)
  ❌ No adaptation to environment changes
  ❌ Same settings for open sky AND tunnels (inefficient)
  ❌ Cannot help vehicles in dense urban areas
  ❌ Bandwidth wasted when not needed
```

### With AI: Intelligent Adaptive Broadcasting
```
Our Approach:
  
  Base Station → RTCM Corrections
                      ↓
                    Vehicles
                      ↓
            Collect Performance Data
                      ↓
                  AI Agent
                      ↓
         Decide: "How to broadcast?"
                      ↓
    Adjust: Redundancy, Update Rate, Tile Resolution, FEC Level
                      ↓
            New ATSC 3.0 Broadcast
                      ↓
            Helps MORE vehicles achieve RTK FIX
                      ↓
            Save bandwidth, save power, save cost

Benefits:
  ✅ Adapts to real-time conditions
  ✅ Optimization based on fleet performance
  ✅ Saves bandwidth during good conditions
  ✅ Increases robustness during tunnel/blockage
  ✅ Feedback loop improves over time
```

---

## 1.4 Key Problem Statements

### Problem 1: GPS Signal Loss in Tunnels & Canyons
**What:** Vehicles lose centimeter-level positioning for 20-120 seconds
**Current Cost:** Estimated $5-10 billion annually in autonomous vehicle safety incidents
**Our Solution:** Pre-broadcast high-quality corrections + intelligent redundancy + coverage maps

### Problem 2: Uncertainty in Urban Environments  
**What:** GNSS signal degrades from multipath reflections (bounces off buildings)
**Current Cost:** Reduced autonomy capabilities, human intervention required
**Our Solution:** AI learns which areas have multipath, pre-adjusts broadcast robustness

### Problem 3: Wasted Broadcast Resources
**What:** Broadcasting same signal strength everywhere, regardless of need
**Current Cost:** Unnecessary power consumption, spectrum inefficiency
**Our Solution:** AI scales transmission up/down based on fleet needs

### Problem 4: No Feedback Loop
**What:** Traditional systems don't know if corrections actually help vehicles
**Current Cost:** Inefficient infrastructure, missed optimization opportunities
**Our Solution:** Real-time telemetry from vehicles feeds AI decision engine

---

---

# PART 2: TECHNICAL DEEP DIVE - COMPLETE ARCHITECTURE

## 2.1 System Overview: The Three Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENTIRE SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐        ┌──────────────────┐                      │
│  │  BASE STATION    │───────▶│  RTCM GENERATOR  │                      │
│  │ (Known Location) │        │  (Error Vectors) │                      │
│  └──────────────────┘        └──────────────────┘                      │
│           │                           │                                │
│           │                           ▼                                │
│           │                  ┌──────────────────┐                      │
│           │                  │ COVERAGE MAP GEN │                      │
│           │                  │ (Signal Quality) │                      │
│           │                  └──────────────────┘                      │
│           │                           │                                │
│           └──────────────────────────────┘                             │
│                                │                                       │
│                                ▼                                       │
│                  ┌──────────────────────────┐                          │
│                  │  DATA AGGREGATOR & AI    │                          │
│                  │  FEEDBACK CONTROLLER     │                          │
│                  │  (Makes Decisions)       │                          │
│                  └──────────────────────────┘                          │
│                                │                                       │
│                                ▼                                       │
│                  ┌──────────────────────────┐                          │
│                  │ ATSC 3.0 ENCODER        │                          │
│                  │ OFDM TRANSMITTER        │                          │
│                  │ (Broadcasts RF Signal)  │                          │
│                  └──────────────────────────┘                          │
│                                │                                       │
│                ┌───────────────▼───────────────┐                       │
│                │                               │                       │
│                ▼ (RF Signal Travels)           ▼                       │
│                                                                          │
│    ┌────────────────────────────────────────────────┐                  │
│    │              VEHICLE / CLIENT SIDE             │                  │
│    │                                                │                  │
│    │  ┌──────────────┐  ┌──────────────────┐       │                  │
│    │  │ ATSC 3.0     │  │ GNSS RECEIVER    │       │                  │
│    │  │ TUNER        │─▶│ (GPS/GLONASS)    │       │                  │
│    │  │ (Receive RF) │  │                  │       │                  │
│    │  └──────────────┘  └──────────────────┘       │                  │
│    │                           │                   │                  │
│    │                           ▼                   │                  │
│    │                  ┌──────────────────┐        │                  │
│    │                  │ RTK ENGINE       │        │                  │
│    │                  │ (RTKLIB)         │        │                  │
│    │                  │ Fuses corrections│        │                  │
│    │                  │ + satellite data │        │                  │
│    │                  └──────────────────┘        │                  │
│    │                           │                   │                  │
│    │                           ▼                   │                  │
│    │                  ┌──────────────────┐        │                  │
│    │                  │ CENTIMETER-LEVEL │        │                  │
│    │                  │ POSITION         │        │                  │
│    │                  │ (±1.5 cm)        │        │                  │
│    │                  └──────────────────┘        │                  │
│    │                           │                   │                  │
│    │                           ▼                   │                  │
│    │                  ┌──────────────────┐        │                  │
│    │                  │ TELEMETRY SENT   │        │                  │
│    │                  │ (via cellular)   │        │                  │
│    │                  │ Back to AI Agent │        │                  │
│    │                  └──────────────────┘        │                  │
│    └────────────────────────────────────────────────┘                  │
│                                │                                       │
│                                ▼ (LTE/5G)                             │
│                                                                          │
│                    ┌──────────────────────────┐                        │
│                    │  FEEDBACK LOOP (AI)      │                        │
│                    │  "How many vehicles got  │                        │
│                    │   FIX? How fast? Where?  │                        │
│                    │   Adjust broadcast next" │                        │
│                    └──────────────────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Detailed Component Breakdown

### COMPONENT 1: Base Station & Reference System

**What It Does:**
```
The base station is a GNSS receiver at a KNOWN location
(latitude, longitude, altitude measured to millimeter precision)

It continuously:
  1. Receives GPS signals from satellites
  2. Measures: "Where did the signals say I am?"
  3. Compares: Measured_Position vs Known_Position
  4. Calculates: ERROR VECTOR (the difference)
  
Example Error Vector:
  ΔX = -0.847 m (East-West error)
  ΔY = +0.423 m (North-South error)
  ΔZ = -0.911 m (Vertical error)
  
This error is caused by:
  • Ionosphere delays (free electrons bending signals)
  • Troposphere delays (water vapor bending signals)
  • Satellite orbit errors (ephemeris not perfect)
  • Clock errors (satellite clocks not synchronized perfectly)
```

**Technical Specs:**
```
Receiver Type:    Multi-constellation GNSS
  • GPS (USA)
  • GLONASS (Russia)
  • Galileo (Europe)
  • BeiDou (China)

Measurement Rate: 1 Hz (one measurement per second)

Accuracy of Base Position:
  • Horizontal: ±5 mm
  • Vertical: ±10 mm
  • Must be surveyed with precision equipment

Output:
  • Error vectors per satellite
  • Per-satellite atmospheric delays
  • Clock offset of system
  • Signal quality metrics
```

---

### COMPONENT 2: RTCM Correction Generator

**What It Does:**
```
Takes raw error vectors from base station
Encodes them into RTCM 3.x binary format
This format is the STANDARD for GPS corrections (international spec)

RTCM Frame Structure (Binary):
┌─────────────────────────────────┐
│ Preamble (1 byte): 0xD3         │  Marker: "RTCM data starts here"
├─────────────────────────────────┤
│ Reserved (6 bits): 0            │  Future use
├─────────────────────────────────┤
│ Frame Length (10 bits)          │  Size of this message
├─────────────────────────────────┤
│ Message Type (12 bits)          │  Type 1004 = RTK base data
│                                 │  Type 1005 = Base position
│                                 │  Type 1012 = GLONASS observations
├─────────────────────────────────┤
│ Payload (Variable)              │  Actual correction data
│  • Error vectors                │  • Ionosphere corrections
│  • Satellite information        │  • Troposphere model
│  • System state                 │  • Ambiguity hints
├─────────────────────────────────┤
│ CRC-24 Checksum (3 bytes)       │  Error detection
└─────────────────────────────────┘

Total Frame Size: 100-300 bytes (highly compressed!)
Update Rate: 1 Hz (one frame every second)
Transmission Rate: 100-300 bytes/sec = 0.8-2.4 kilobits/sec (tiny!)
```

**Why RTCM?**
```
✓ International standard (receivers everywhere support it)
✓ Highly compressed binary format (tiny data size)
✓ Error-corrected (CRC checksums ensure integrity)
✓ Hierarchical message types (flexible scaling)
✓ Industry-proven (used in professional surveying for 20+ years)
```

---

### COMPONENT 3: Coverage Map Generator (Optional Enhancement)

**What It Does:**
```
Creates visual representation of where GPS signals are available
Useful for predicting signal loss ahead of time

Format: 100×100 pixel bitmap tiles
  White (255) = Good signal (open sky)
  Black (0)   = Blocked signal (building, tunnel)
  Gray (128)  = Degraded signal (partial blockage)

Example Tile: Downtown Seoul, Building 15
┌──────────────────────────────────────┐
│ WWWWWWWWWWWWWWWWWBBWWWWWWWWWWWWW Row 1
│ WWWWWBBBBBBBWWWWWBBWWWWWWWWWWWWW Row 2
│ WWWWWBBBBBBBWWWWWBBWWWWWWWWWWWWW Row 3
│ ...
│ BBBBBWWWWWWWWWWWWWWBBBBBBBBBBBBB Row 100
└──────────────────────────────────────┘
(Black squares = large buildings, tunnel segments)

Size:
  • Uncompressed: 10 KB per tile
  • Compressed: 200-800 bytes per tile
  • Can be cached at edge (same geographic area reused)

How Vehicles Use It:
  1. Receive bitmap as vehicle approaches area
  2. Check current position on bitmap
  3. If white = expect good signal
  4. If black = prepare for signal loss (switch to fallback)
  5. If gray = be cautious, multipath likely
```

---

### COMPONENT 4: Data Aggregator & AI Feedback Controller

**What It Does:**
```
Collects telemetry from all vehicles in fleet
Analyzes real-time performance metrics
Makes DECISIONS on how to adjust the broadcast

Decision Process:
┌─────────────────────────────────────────┐
│ GATHER DATA                             │
│ • How many vehicles in FIX mode?        │
│ • How many in FLOAT mode?               │
│ • Average convergence time?             │
│ • Signal strength across fleet?         │
│ • Geographic patterns (tunnel ahead?)   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ NEURAL NETWORK INFERENCE                │
│ (PyTorch model running on broadcaster)  │
│                                         │
│ Input: Fleet telemetry + environmental │
│ Output: Broadcast parameters            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ CONFIDENCE CHECK                        │
│ • Is model confident? (>75%?)          │
│ • Or fallback to rule-based logic       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ BROADCAST COMMAND GENERATED             │
│ • Redundancy: 1.0x to 2.0x             │
│ • Update Frequency: 0.5 to 5.0 Hz      │
│ • Tile Resolution: low/medium/high     │
│ • FEC Level: 10-40% overhead           │
└─────────────────────────────────────────┘
```

**Example Decisions:**

```
Scenario 1: OPEN SKY CONDITIONS
  Fleet Status: 95% in FIX mode, avg 18s convergence
  Decision: "System working great. Reduce resource usage"
  Broadcast Command: {
    redundancy: 1.0,
    update_frequency: 0.8 Hz,
    tile_resolution: "medium",
    fec_overhead: 15%
  }
  Effect: Save bandwidth, save power

Scenario 2: TUNNEL AHEAD (Fleet reporting degradation)
  Fleet Status: 40% FIX, 35% FLOAT, 25% STAND-ALONE
  Decision: "Emergency mode. Maximize support"
  Broadcast Command: {
    redundancy: 2.0,          # Send corrections TWICE
    update_frequency: 2.0 Hz, # Send 2.5x more often
    tile_resolution: "high",  # Maximum detail maps
    fec_overhead: 40%         # Maximum error correction
  }
  Effect: Help vehicles survive tunnel with minimal accuracy loss

Scenario 3: RECOVERY FROM TUNNEL
  Fleet Status: 70% FLOAT, improving, convergence time 35s
  Decision: "Recovery in progress. Maintain elevated support"
  Broadcast Command: {
    redundancy: 1.5,
    update_frequency: 1.5 Hz,
    tile_resolution: "high",
    fec_overhead: 30%
  }
  Effect: Accelerate FIX recovery, reduce to normal within 30s
```

---

### COMPONENT 5: ATSC 3.0 Broadcast Encoder

**What It Does:**
```
Takes RTCM frames from base station
Takes bitmap tiles from coverage map
Packages them according to AI decision
Broadcasts everything over ATSC 3.0 RF channel

Encoding Process:

Step 1: Apply AI Decision
  Read broadcast command from AI agent
  Extract parameters: redundancy, update_freq, tile_resolution

Step 2: Create ALP Packets (ATSC Link Layer Protocol)
  Packet Format:
  ┌──────────────────────────┐
  │ ALP Header (metadata)    │
  ├──────────────────────────┤
  │ Payload (RTCM/Tile data) │
  ├──────────────────────────┤
  │ Checksum (error detect)  │
  └──────────────────────────┘
  
  With Redundancy (if AI says 2.0x):
  • Create PRIMARY packet
  • Create DUPLICATE packet
  • Send both (only one needed)

Step 3: Apply FEC (Forward Error Correction)
  Encoding Type: LDPC (Low-Density Parity-Check)
  • Add parity bits: create additional error-correction info
  • If 40% FEC overhead:
    - Original data: 100 bits
    - FEC adds: 40 bits
    - Total transmitted: 140 bits
  • Receiver can recover from loss of 30-40% of packets
  
  Why LDPC?
  • Can recover from many bit errors without retransmission
  • Proven in satellite/cellular standards
  • Parallel decoding (fast)

Step 4: OFDM Modulation
  OFDM = Orthogonal Frequency Division Multiplexing
  
  What: Break signal into many small carriers
  Why: Resistant to multipath (fading) and interference
  How:
  ┌─────────────────────────────┐
  │ Data to transmit: 0110...   │
  ├─────────────────────────────┤
  │ Split into subcarriers:     │
  │ • 1 Hz: 0                   │
  │ • 2 Hz: 1                   │
  │ • 3 Hz: 1                   │
  │ • 4 Hz: 0                   │
  │ (8000-16000 simultaneous)   │
  ├─────────────────────────────┤
  │ Each modulated independently│
  │ Sent on single RF channel   │
  └─────────────────────────────┘
  
  Guard Interval: 1/12 or 1/32 of symbol time
  • Protects against multipath
  • Mobile: 1/12 (more protection)
  • Fixed: 1/32 (more capacity)

Step 5: Multiple PLPs (Physical Layer Pipes)
  
  PLP = Virtual channel within one RF frequency
  Purpose: Different content, different protection levels
  
  ┌──────────────────────────────────────────┐
  │ Physical Layer Pipes (Within 6 MHz RF)  │
  ├──────────────────────────────────────────┤
  │ PLP_0: Mobile (Most Robust)              │
  │  • RTCM corrections only                │
  │  • QPSK modulation (very robust)        │
  │  • High FEC overhead (40%)              │
  │  • Lower data rate (1-5 Mbps)           │
  │  • Can receive at highway speeds        │
  │                                          │
  │ PLP_1: Hybrid (Balanced)                 │
  │  • RTCM + Bitmap tiles                  │
  │  • 16-QAM modulation                    │
  │  • Medium FEC overhead (25%)            │
  │  • Medium data rate (10-20 Mbps)        │
  │  • For vehicles and fixed points        │
  │                                          │
  │ PLP_2: Fixed (High Capacity)             │
  │  • Tiles + supplementary data           │
  │  • 256-QAM or 1024-QAM modulation      │
  │  • Low FEC overhead (15%)               │
  │  • High data rate (30-50 Mbps)          │
  │  • For stationary/slow-moving receivers│
  └──────────────────────────────────────────┘
  
  Receiver selects which PLP based on situation:
  • Driving on highway? Select PLP_0 (most robust)
  • City driving? Select PLP_1 (balanced)
  • Parked? Select PLP_2 (high data)

Step 6: Physical Transmission
  ┌──────────────────────────────┐
  │ RF Parameters                │
  ├──────────────────────────────┤
  │ Frequency: 471-700 MHz       │
  │ (VHF/UHF broadcast band)     │
  │                              │
  │ Channel Width: 6 MHz         │
  │ (standard TV channel width)  │
  │                              │
  │ Effective Bitrate: 5-57 Mbps │
  │ (depends on PLP config)      │
  │                              │
  │ Transmission Power: 1-10 kW  │
  │ (typical broadcast station)  │
  │                              │
  │ Range: 30-50 km line-of-sight│
  │                              │
  │ Mobile Reception: Yes        │
  │ Works at 200+ km/h          │
  │ (verified in ATSC 3.0 trials)│
  └──────────────────────────────┘
```

---

### COMPONENT 6: Vehicle Receiver & RTK Processing

**What It Does:**
```
Vehicle has two antennas:
  1. ATSC 3.0 receiver (gets broadcast corrections)
  2. GNSS antenna (gets satellite signals)

Process:
┌────────────────────────────────────┐
│ STEP 1: RF RECEPTION               │
│ • ATSC 3.0 tuner locks to signal  │
│ • Demodulates OFDM constellation  │
│ • Applies FEC decoding (LDPC)     │
│ • De-packetizes ALP packets       │
│ • Verifies checksums              │
│ • Extracts RTCM frames + tiles    │
└────────────────────────────────────┘
         │
         ▼ (~5-10 second latency)
┌────────────────────────────────────┐
│ STEP 2: GNSS SIGNAL RECEPTION      │
│ • Receives raw GPS/GLONASS signals │
│ • Tracks satellites (usually 10-15)│
│ • Measures:                        │
│   - Pseudoranges (code phase)      │
│   - Carrier phases (1000x precise) │
│   - Doppler (velocity info)        │
│   - Signal strength (C/N0 ratio)   │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ STEP 3: APPLY CORRECTIONS          │
│ • RTK processor reads RTCM messages│
│ • For each satellite:              │
│   Corrected_Range = Raw - RTCM_Err │
│ • Use corrections to refine:       │
│   - Ionosphere delay              │
│   - Troposphere delay             │
│   - Satellite clock/orbit         │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ STEP 4: AMBIGUITY RESOLUTION       │
│ • Carrier phase is "wrapped"       │
│ • Find integer number of cycles    │
│ • Takes 10-60 seconds normally     │
│ • With good corrections: faster    │
│ • With bitmap tiles: faster (fewer │
│   ambiguities to search)           │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ STEP 5: POSITION SOLUTION          │
│                                    │
│ Mode 1: STAND-ALONE (No RTK)      │
│ • Only raw GPS available          │
│ • Accuracy: ±5-10 meters          │
│ • Status: Emergency mode           │
│                                    │
│ Mode 2: FLOAT (Partial RTK)       │
│ • Corrections received             │
│ • Ambiguities not yet fixed        │
│ • Accuracy: ±10-100 cm            │
│ • Status: Converging              │
│                                    │
│ Mode 3: FIX (Full RTK - GOAL!)    │
│ • Corrections received             │
│ • All ambiguities resolved         │
│ • Accuracy: ±1.5 cm               │
│ • Status: Ready for autonomy       │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ STEP 6: TELEMETRY COLLECTION       │
│ Gather metrics:                    │
│ • RTK mode (STAND-ALONE/FLOAT/FIX)│
│ • Position accuracy (cm)           │
│ • Num satellites tracked           │
│ • Signal strength per satellite    │
│ • Convergence time to FIX          │
│ • Current location                 │
│ • Urban density estimate           │
│ • Environment (tunnel? canyon?)    │
│ • Vehicle speed/heading            │
│ • Confidence in solution           │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ STEP 7: SEND FEEDBACK              │
│ Via cellular (LTE/5G):            │
│ • Send telemetry to broadcaster    │
│ • Every 10 seconds (or on-demand)  │
│ • Privacy: Can be anonymized       │
│ • Aggregated from many vehicles    │
└────────────────────────────────────┘
```

**RTK Modes Explained:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     RTK MODE PROGRESSION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ WITHOUT BROADCAST CORRECTIONS:                                  │
│                                                                 │
│ T=0s   STAND-ALONE     ±5-10 meters   (all we have is GPS)    │
│   │                                                            │
│   └─→ (No improvement, stuck at ±5-10m)                       │
│                                                                 │
│                                                                 │
│ WITH BROADCAST CORRECTIONS (Our System):                        │
│                                                                 │
│ T=0s   STAND-ALONE     ±5-10 meters   (first, waiting for data)│
│         │                                                       │
│ T=3s   │─→ FLOAT        ±30-100 cm    (corrections arrived)   │
│         │   • Started using RTCM                               │
│         │   • Pseudoranges corrected                           │
│         │   • Carrier phase not yet locked                     │
│         │   • Accuracy jumped 50x!                             │
│         │                                                       │
│ T=8s   │─→ FLOAT        ±20-50 cm     (better geometry)       │
│         │   • More satellites visible                          │
│         │   • Corrections refined                              │
│         │   • Still searching for ambiguities                  │
│         │                                                       │
│ T=18s  │─→ FIX          ±1.5-2.0 cm   (GOAL REACHED!)        │
│         │   ✅ Ambiguities locked                             │
│         │   ✅ Integer solution found                         │
│         │   ✅ Ready for autonomous operation                 │
│         │   ✅ Centimeter-level accuracy!                     │
│         │                                                       │
│ T=18s+ │─→ FIX (Continuous) for as long as signal lasts      │
│         │   • Can hold FIX even briefly losing 1-2 sats      │
│         │   • Typical availability: 99%+                      │
│         │                                                       │
│                                                                 │
│ WHAT AFFECTS CONVERGENCE TIME:                                 │
│ • Satellite geometry (dilution of precision)                  │
│ • Signal quality                                              │
│ • Baseline length (distance from base)                        │
│ • Correction quality                                          │
│ • Update frequency of corrections                             │
│ • Bitmap tiles (helps search space)                           │
│ • Environmental multipath                                     │
│                                                                 │
│ TYPICAL TIMES (With Our System):                               │
│ • Open sky: 10-20 seconds to FIX                              │
│ • Urban: 20-40 seconds to FIX                                 │
│ • Dense urban: 30-60 seconds to FIX                           │
│ • Coming out of tunnel: 25-35 seconds to FIX                  │
│                                                                 │
│ WITHOUT BROADCAST SYSTEM:                                      │
│ • Open sky: 60-120 seconds (much slower!)                     │
│ • Urban: 2-5 minutes or never                                 │
│ • Dense urban: Can't achieve FIX at all                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.3 Real-World Scenario: Complete Tunnel Walkthrough

### Timeline: Vehicle Entering Tunnel (120 Seconds)

```
═══════════════════════════════════════════════════════════════════════

T = -60 SECONDS: BEFORE TUNNEL (Open Sky)

Location:        Highway, 60 meters before tunnel
GPS Satellites:  14 visible (excellent)
Signal Quality:  40-42 dB-Hz (very strong)
RTK Mode:        FIX (centimeter accuracy)
Vehicle Speed:   80 km/h
Position Error:  ±1.5 cm

WHAT'S HAPPENING:
• GNSS receiver getting excellent satellite signals
• RTK processor locked and producing precise positions
• Broadcast system: Normal mode (1.0x redundancy, 0.8 Hz)
• AI Agent observing: "85% of fleet in FIX mode - great conditions"

VEHICLE STATE:
  Position: [37.580746° N, 126.892210° E, 106.950 m elev]
  Accuracy: ±1.5 cm
  Status: ✅ AUTONOMOUS MODE READY
  Lane-keeping: Working perfectly
  Obstacle detection: Full precision available

BROADCAST TRANSMITTED:
  Update rate: 0.8 Hz (0.8 frames per second)
  Redundancy: 1.0x (normal, no extra copies)
  Tile resolution: Medium
  FEC overhead: 15%
  Effect: Efficient, minimal bandwidth usage

═══════════════════════════════════════════════════════════════════════

T = 0 SECONDS: ENTERING TUNNEL (Critical Moment)

Location:        Tunnel entrance (front of car enters)
GPS Satellites:  14 → 8 (signal degrading rapidly)
Signal Quality:  41 dB-Hz → 25 dB-Hz
RTK Mode:        FIX → FLOAT (transitioning)
Position Error:  ±1.5 cm → ±45 cm

WHAT'S HAPPENING:
• Concrete tunnel ceiling blocking direct satellite signals
• Multipath: signals bouncing off tunnel walls (creating errors)
• Some satellites still partially visible (above tunnel opening)
• GNSS receiver under stress - ambiguities becoming uncertain

SIGNAL LOSS PATTERN:
  GPS G02: Signal lost (was strong, now blocked by rock)
  GPS G05: Signal lost (same reason)
  GPS G08: Signal lost
  GPS G11: Still visible but weak (marginal signal)
  GPS G14: Still visible but multipath detected
  
  Net result: 6 satellites lost, only 8 remaining

VEHICLE STATE CHANGE:
  Position: [37.5808° N, 126.8922° E] (same, but accuracy worse!)
  Accuracy: ±45 cm (30x worse than before!)
  Status: ⚠️ FLOAT MODE (not ideal)
  Lane-keeping: Works but with reduced confidence
  Autonomous operation: Still safe but degraded

RTK ENGINE ANALYSIS:
  Satellites available: 8
  For solution: Need minimum 4, but quality matters
  Ambiguity resolution: Becoming difficult
  Confidence: Drop from 0.99 to 0.45

TELEMETRY SENT TO BROADCASTER:
  {
    rtk_mode: "FLOAT",
    position_error_cm: 45,
    num_satellites: 8,
    signal_strength_db: 28.5,
    is_in_blockage: true,
    alert: "ENTERING_TUNNEL_SIGNAL_LOSS"
  }

AI AGENT RECEIVES ALERT:
  • Detects 30 vehicles reporting tunnel entry
  • Fleet performance degrading
  • FIX mode availability: 95% → 45%
  • Convergence time: 18s → 35s+

AI DECISION: "EMERGENCY MODE - MAXIMIZE BROADCAST ROBUSTNESS"

NEW BROADCAST COMMAND GENERATED:
  {
    redundancy: 2.0,           # DOUBLED - send corrections 2x
    update_frequency_hz: 2.0,  # DOUBLED - from 0.8 to 2.0 Hz
    tile_resolution: "high",   # Maximize detail maps
    fec_overhead: 40%,         # Maximum error correction
    plp_mode: "mobile"         # Most robust configuration
  }

BROADCASTER REACTS (within 100ms):
  • Immediately switches to emergency broadcast
  • RTCM frames now sent at 2 Hz (instead of 0.8 Hz)
  • Each frame sent TWICE (redundancy = 2.0x)
  • High-resolution tunnel maps transmitted
  • Increased FEC: 40% error correction overhead

LATENCY TO RECEIVER:
  Old setup: ~8 seconds for corrections to reach vehicle
  New setup: ~8 seconds (same, but more frequent + robust)
  Difference: More recent corrections, backup copies

═══════════════════════════════════════════════════════════════════════

T = 30 SECONDS: DEEPER INTO TUNNEL (Getting Worse)

Location:        Inside tunnel (150 meters in)
GPS Satellites:  5 (continuing to degrade)
Signal Quality:  18 dB-Hz (barely usable)
RTK Mode:        FLOAT → STAND-ALONE (losing corrections benefit)
Position Error:  ±45 cm → ±200 cm (±2 meters!)

WHAT'S HAPPENING:
• Tunnel depth now significant
• Direct sky contact nearly gone
• Only 5 satellites with extremely weak, multipath-corrupted signals
• Ambiguity resolution essentially impossible

VEHICLE STATE:
  Position: [37.5820° N, 126.8930° E] (dead-reckoning now)
  Accuracy: ±200 cm (±2 meters!)
  Status: ⚠️ DEGRADED
  Lane-keeping: Still works (road-constrained)
  Autonomous: Limited to pre-mapped routes only

FALLBACK SYSTEMS ACTIVATED:
  • Dead-reckoning: Using wheel odometry
  • Inertial Measurement: IMU tracking motion
  • Map-matching: Constraining to known tunnel path
  • Saved bitmap tiles: Showing tunnel geometry
  
  Combination effect: Position estimate stays reasonable
  (error grows, but road constraints help)

BROADCAST AT THIS POINT:
  • Still sending at emergency rates (2 Hz, 2x redundancy)
  • High-res maps crucial: Shows tunnel walls, lane markings
  • Corrections becoming less useful (can't get good satellites anyway)
  • But maintaining them for the moment signal comes back

═══════════════════════════════════════════════════════════════════════

T = 60 SECONDS: MIDWAY THROUGH TUNNEL (Deepest Point)

Location:        Deep tunnel (300 meters in, midway)
GPS Satellites:  0-2 (essentially gone)
Signal Quality:  <15 dB-Hz (noise level)
RTK Mode:        STAND-ALONE (completely autonomous)
Position Error:  ±5-10 meters (dead-reckoning only)

WHAT'S HAPPENING:
• Complete GNSS signal loss from direct paths
• Only scattered multipath reflections (useless for positioning)
• Vehicle now entirely dependent on:
  - Dead-reckoning (wheel speed, steering angle)
  - Inertial navigation (gyros, accelerometers)
  - Map matching (knowing tunnel path from pre-loaded map)

VEHICLE STATE:
  Position: Estimated based on motion since tunnel entry
  Accuracy: ±5-10 meters (not great, but good enough with constraints)
  Status: ⚠️ GNSS DENIED MODE
  Lane-keeping: Using map + IMU only (still works!)
  Autonomous: Fully functional on pre-mapped route

WHAT THE VEHICLE "KNOWS":
  • Started tunnel 60 seconds ago at position [37.5808, 126.8922]
  • Driving northeast at 80 km/h = 22.2 m/s
  • In 60 seconds, traveled: 22.2 × 60 = 1333 meters
  • Current position estimate: [37.5858, 126.8945] (± large error)
  
  Error Growth:
  • At 0s (tunnel entry): 0 error (from FIX solution)
  • At 10s: ~2-3 meter error (accumulated)
  • At 30s: ~5-8 meter error
  • At 60s: ~8-15 meter error
  
  Without broadcast system's maps:
  • Vehicle would be much more uncertain
  • Might slow down or stop for safety
  
  With our system:
  • Pre-loaded tunnel map shows location to ±0.5 meter
  • Vehicle confident can maintain lane at ±2 meter error
  • Continues autonomously

AI AGENT AT THIS POINT:
  • Observing: 85% of vehicles in STAND-ALONE mode
  • Fleet-wide statistics showing:
    - Average accuracy degraded to ±8 meters
    - Relying on fallback systems
    - Convergence times undefined (no GNSS signals)
  
  Decision:
    "Maintain emergency broadcast. Vehicles relying on fallback +
     pre-loaded maps. System working as designed. Predict exit at
     T+120s, prepare for re-convergence burst."

═══════════════════════════════════════════════════════════════════════

T = 120 SECONDS: EXITING TUNNEL (Critical Recovery Phase)

Location:        Tunnel exit (500 meters in, now exiting)
GPS Satellites:  0 → 5 → 10 (rapidly re-acquiring)
Signal Quality:  8 dB-Hz → 20 dB-Hz → 35 dB-Hz
RTK Mode:        STAND-ALONE → FLOAT → FIX
Position Error:  ±5m → ±50cm → ±1.5cm

WHAT'S HAPPENING:
• Front of vehicle clears tunnel ceiling
• Satellite signals suddenly reappearing
• Multipath still strong from tunnel walls (rear antenna affected)
• CRITICAL WINDOW: Must re-establish fix quickly

RE-ACQUISITION SEQUENCE:
┌─────────────────────────┐
│ T+120s+0s: First signal │
│ • GPS G08 detected      │
│ • GPS G14 detected      │
│ • 5 satellites acquired │
│ • Mode: STAND-ALONE     │
└─────────────────────────┘
         │
         ▼ (corrections arrive)
┌─────────────────────────┐
│ T+120s+8s: Corrections  │
│ • RTCM messages flowing │
│ • Pseudoranges improved │
│ • 8 satellites now      │
│ • Mode: FLOAT           │
│ • Accuracy: ±50 cm     │
│ • Confidence: 0.35      │
└─────────────────────────┘
         │
         ▼ (carrier tracking)
┌─────────────────────────┐
│ T+120s+15s: Converging  │
│ • Multipath fading      │
│ • 10 satellites         │
│ • Better geometry       │
│ • Mode: FLOAT (still)   │
│ • Accuracy: ±30 cm     │
│ • Confidence: 0.55      │
└─────────────────────────┘
         │
         ▼ (ambiguity search)
┌─────────────────────────┐
│ T+120s+25s: Locking     │
│ • Ambiguities narrowing │
│ • 12 satellites         │
│ • Excellent geometry    │
│ • Carrier-phase clean   │
│ • Mode: FLOAT → FIX!    │
│ • Accuracy: ±1.5 cm!!! │
│ • Confidence: 0.95      │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ T+120s+28s: RECOVERY    │
│ • Full FIX established  │
│ • 13 satellites tracked │
│ • High confidence (0.98)│
│ • Position locked       │
│ • Ready for autonomy    │
└─────────────────────────┘

TOTAL RECOVERY TIME: 28 SECONDS (from STAND-ALONE to FIX)

Why So Fast?
  ✓ Emergency broadcast (2 Hz corrections)
  ✓ Redundancy (2x copies, less packet loss)
  ✓ Coverage maps (showed tunnel location, radio pre-convergence search)
  ✓ Good signal quality re-emerging
  ✓ AI predictions (broadcaster prepared for exit)

Without Broadcast System:
  • Typical recovery: 60-120 seconds
  • Even then: risky, might lose FIX again

With Our System:
  • Recovery: 28 seconds
  • High confidence throughout
  • Smooth handover back to full autonomy

VEHICLE STATE DURING RECOVERY:
  T+120s+8s:  Mode FLOAT, ⚠️ Warning: "Recovering from signal loss"
  T+120s+15s: Mode FLOAT, ⏳ Status: "Re-acquiring fix..."
  T+120s+25s: Mode FIX, ✅ Alert: "Position lock established!"
  T+120s+30s: Mode FIX, ✅ Ready: "Autonomous mode available"

═══════════════════════════════════════════════════════════════════════

T = 150 SECONDS: FULLY RECOVERED (Normal Operation Resumed)

Location:        30 seconds past tunnel exit
GPS Satellites:  14 (back to normal)
Signal Quality:  39 dB-Hz (strong)
RTK Mode:        FIX (solid)
Position Error:  ±1.5 cm
Vehicle Speed:   80 km/h

VEHICLE STATE:
  Position: [37.5858° N, 126.8945° E, 106.95 m] (±1.5cm!)
  Status: ✅ AUTONOMOUS MODE ACTIVE
  Lane-keeping: Full precision available
  Obstacle detection: Operating normally
  Lidar/Radar: Integrated with RTK positioning

BROADCAST WINDING DOWN:
  AI Agent decision: "Exit zone confirmed. Fleet recovering to FIX.
                     Return to normal efficient broadcast."
  
  New Broadcast Command:
  {
    redundancy: 1.0,        # Back to normal (no extra copies)
    update_frequency: 0.8 Hz, # Reduced from emergency 2 Hz
    tile_resolution: "medium", # Reduced from high
    fec_overhead: 15%       # Reduced from emergency 40%
  }
  
  Broadcaster applies immediately:
  • RTCM frames: 0.8 Hz (back to normal rate)
  • No redundancy: Single transmission per frame
  • Tile resolution: Medium detail
  • FEC: Standard 15% overhead

FLEET-WIDE STATUS:
  Before tunnel:  95% FIX mode, avg convergence 18s
  In tunnel:      5% FIX, 25% FLOAT, 70% STAND-ALONE
  Exiting:        70% FLOAT, 20% FIX, 10% STAND-ALONE
  After tunnel:   95% FIX mode, avg convergence 25s

SYSTEM PERFORMANCE SUMMARY:
  ✅ Vehicles maintained centimeter accuracy through tunnel exit
  ✅ Recovery time: 28 seconds (vs 60-120s without system)
  ✅ Fallback systems kept vehicles on road during GNSS denial
  ✅ AI successfully predicted and responded to event
  ✅ Broadcast resources dynamically scaled
  ✅ No accidents, no safety incidents
  ✅ Smooth handover to full autonomy

═══════════════════════════════════════════════════════════════════════
```

---

## 2.4 Data Formats & Communication Protocols

### RTCM Frame Format (Binary)
```
┌──────┬──────┬──────┬──────┬───────┬──────┐
│ PRE  │ RSV  │ LEN  │ TYPE │PAYLOAD│ CRC  │
├──────┼──────┼──────┼──────┼───────┼──────┤
│ 1B   │ 6b   │ 10b  │ 12b  │ Var   │ 3B   │
└──────┴──────┴──────┴──────┴───────┴──────┘

Preamble (0xD3):    Synchronization marker
Reserved:           Future expansion
Length:             Size of message in bytes
Type:               Message type (1004, 1005, 1012, etc.)
Payload:            Actual correction data
CRC-24:             Error detection

Common Message Types:
  1004 = RTK Base Station Observations
  1005 = Base Station Coordinates
  1012 = GLONASS Observations
  1013 = GLONASS Satellite/Signal Index
  1033 = Receiver/Antenna Description
```

### Vehicle Telemetry JSON Format
```json
{
  "timestamp": 1705094400,
  "vehicle_id": "vehicle_001",
  "location": {
    "latitude": 37.580746,
    "longitude": 126.892210,
    "height_m": 106.950
  },
  "rtk_metrics": {
    "mode": "FIX",
    "position_error_cm": 1.5,
    "num_satellites_used": 12,
    "signal_strength_db_hz": 39.8,
    "convergence_time_sec": 18.3
  },
  "environment": {
    "urban_density": 0.2,
    "is_in_tunnel": false,
    "is_in_canyon": false
  },
  "vehicle_state": {
    "speed_kmh": 80,
    "heading_deg": 45,
    "confidence": 0.98
  }
}
```

### AI Broadcast Command JSON Format
```json
{
  "timestamp": 1705094500,
  "broadcast_config": {
    "redundancy": 1.2,
    "tile_resolution": "high",
    "update_frequency_hz": 2.0,
    "plp_mode": "mobile",
    "fec_overhead_pct": 25
  },
  "reasoning": {
    "current_fix_pct": 85,
    "avg_convergence_sec": 22,
    "tunnel_ahead": true,
    "decision": "INCREASE_BROADCAST_ROBUSTNESS"
  },
  "confidence": 0.92
}
```

---

---

# PART 3: DETAILED TEAM ORGANIZATION & WORK DISTRIBUTION

## 3.1 Three-Person Team Structure

```
┌────────────────────────────────────────────────────────────┐
│              3-PERSON TEAM ORGANIZATION                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐ │
│  │   TEAM 1: GNSS │  │ TEAM 2:BROADCAST│  │ TEAM 3:AI │ │
│  │   Positioning  │  │   Systems       │  │/ML        │ │
│  │   Engineer     │  │   Engineer      │  │Engineer   │ │
│  │                │  │                 │  │           │ │
│  │ Domain:        │  │ Domain:         │  │ Domain:   │ │
│  │ • GPS/GNSS     │  │ • RF/Telecom    │  │ • AI/ML   │ │
│  │ • RTK          │  │ • ATSC 3.0      │  │ • Neural  │ │
│  │ • Positioning  │  │ • FEC/Encoding  │  │   Networks│ │
│  │ • RTKLIB       │  │ • Signal proc   │  │ • PyTorch │ │
│  │                │  │ • Modulation    │  │ • ONNX    │ │
│  └─────────────────┘  └─────────────────┘  └───────────┘ │
│           │                   │                    │       │
│           └───────────────────┴────────────────────┘       │
│                        │                                   │
│                    SYNC POINTS:                            │
│                   • API handoffs between modules           │
│                   • Phase completion gates                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 3.2 TEAM 1: GNSS Positioning Engineer - ANIRUDH

### Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│          GNSS ENGINEER: COMPLETE SCOPE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ PHASE 0 (Weeks 1-2): FOUNDATION & SETUP                    │
│ ├─ Install RTKLIB, verify compilation                     │
│ ├─ Download sample GNSS data (RINEX format)               │
│ ├─ Basic positioning tests (single solution, no RTK)      │
│ ├─ Understand RTCM format specification                   │
│ ├─ Design error vector data structure                     │
│ └─ Deliverable: Baseline positioning system working       │
│                                                             │
│ PHASE 1 (Weeks 3-5): RTCM GENERATION                       │
│ ├─ Build RTCM frame generator (Python)                    │
│ │  └─ Message Type 1004 (RTK base observations)          │
│ │  └─ Message Type 1005 (Base station coordinates)        │
│ │  └─ Message Type 1012 (GLONASS observations)           │
│ ├─ Implement CRC-24 checksum calculation                  │
│ ├─ Create synthetic error vectors from real GNSS data    │
│ ├─ Generate 1000+ test RTCM frames                        │
│ ├─ Validate frames against specification                  │
│ └─ Deliverable: rtcm_generator.py (production-ready)      │
│                                                             │
│ PHASE 1 (Weeks 4-5): COVERAGE MAP GENERATION              │
│ ├─ Build coverage map generator (Python + PIL)           │
│ ├─ Algorithm: Ray-tracing or empirical model              │
│ ├─ Input: Base station location + map data               │
│ ├─ Output: 100×100 pixel bitmaps (white/black/gray)      │
│ ├─ Create test tiles for multiple urban scenarios         │
│ ├─ Compression: Test simple compression (PNG/JPEG)        │
│ └─ Deliverable: coverage_map_generator.py                 │
│                                                             │
│ PHASE 1 (Weeks 4-6): VEHICLE SCENARIO SIMULATOR           │
│ ├─ Build realistic vehicle state generator                │
│ ├─ Simulate RTK processing outcomes                       │
│ ├─ Generate fleet telemetry (JSON format)                 │
│ ├─ Create 10,000 scenarios covering:                      │
│ │  • Open sky (perfect conditions)                        │
│ │  • Urban canyon (signal degradation)                    │
│ │  • Tunnel entry/exit (signal loss recovery)            │
│ │  • Power fluctuations (satellite quality changes)       │
│ │  • False positives (benign signal spikes)              │
│ ├─ Output: training_data/telemetry_phase1.jsonl          │
│ └─ Deliverable: scenario_simulator.py + 50 MB dataset    │
│                                                             │
│ PHASE 2 (Weeks 7-9): INTEGRATION & VALIDATION             │
│ ├─ Integrate with Broadcast Team's ATSC encoder          │
│ ├─ Verify RTCM frames received correctly                  │
│ ├─ Test end-to-end: RTCM → Broadcast → Vehicle RTK       │
│ ├─ Validate coverage maps used by broadcast              │
│ ├─ Measure latencies and packet loss                      │
│ ├─ Performance testing: throughput, accuracy              │
│ └─ Deliverable: Integration tests passing 95%+           │
│                                                             │
│ PHASE 2 (Weeks 8-9): TUNNEL SCENARIO DEEP DIVE            │
│ ├─ Create detailed tunnel simulation                      │
│ ├─ Model signal degradation vs tunnel depth              │
│ ├─ Simulate multipath effects                             │
│ ├─ Test fallback systems (dead-reckoning + IMU)          │
│ ├─ Generate 100+ tunnel scenarios                         │
│ └─ Deliverable: tunnel_simulation_dataset.h5              │
│                                                             │
│ PHASE 3 (Weeks 10-12): PRODUCTION OPTIMIZATION            │
│ ├─ Profile code for performance bottlenecks              │
│ ├─ Optimize RTCM encoding (vectorize with NumPy)         │
│ ├─ Memory optimization (reduce allocations)               │
│ ├─ Real-time performance: Must generate frames < 5ms     │
│ ├─ Documentation: API reference, examples                 │
│ ├─ Edge deployment: Export lightweight version            │
│ └─ Deliverable: Production GNSS module + docs            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Deliverables

```
Week 2:
  ✓ RTKLIB compiled and working
  ✓ Sample GNSS data downloaded (10+ hours)
  ✓ Basic positioning test scripts

Week 5:
  ✓ rtcm_generator.py (1000+ test frames)
  ✓ coverage_map_generator.py (20+ sample tiles)
  ✓ scenario_simulator.py (10K vehicle scenarios)
  ✓ Training dataset: 50 MB JSONL telemetry

Week 9:
  ✓ Integration tests with Broadcast Team (95% pass rate)
  ✓ Tunnel simulation complete (100+ scenarios)
  ✓ End-to-end validation document

Week 12:
  ✓ Production code ready
  ✓ API documentation (30+ pages)
  ✓ Jupyter notebooks with examples
  ✓ Performance benchmarks (latency, throughput)
```

### Technical Skills Required

```
✓ GPS/GNSS fundamentals (RTK, carrier phase, ambiguity resolution)
✓ Python 3.9+ (NumPy, SciPy, H5PY)
✓ RTKLIB API usage
✓ Binary data handling (struct module)
✓ CRC algorithms
✓ Time series data handling
✓ JSON/JSONL file formats
✓ Debugging satellite data issues

Nice to have:
✓ Image processing (PIL for bitmaps)
✓ Ray-tracing or visibility algorithms
✓ Kalman filtering basics
```

---

## 3.3 TEAM 2: Broadcast Systems Engineer - RISHI

### Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│        BROADCAST ENGINEER: COMPLETE SCOPE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ PHASE 0 (Weeks 1-2): FOUNDATION & SETUP                    │
│ ├─ Study ATSC 3.0 specification                           │
│ ├─ Study OFDM fundamentals                                │
│ ├─ Study FEC concepts (LDPC, Reed-Solomon)               │
│ ├─ Install CommPy (LDPC library)                          │
│ ├─ Install reedsolo (Reed-Solomon library)               │
│ ├─ Verify libraries with unit tests                       │
│ └─ Deliverable: Technical reference document              │
│                                                             │
│ PHASE 1 (Weeks 3-4): ATSC 3.0 ENCODER                     │
│ ├─ Build ALP packet creator (ATSC Link Layer)            │
│ ├─ Format specification:                                  │
│ │  [Header: 1B][Length: 2B][Payload: Var][Checksum: 1B] │
│ ├─ Handle RTCM frames as payload                         │
│ ├─ Implement redundancy (send duplicate packets)          │
│ ├─ Unit test with 1000 RTCM frames                       │
│ └─ Deliverable: alp_encoder.py (handles any input data)  │
│                                                             │
│ PHASE 1 (Weeks 4-5): FEC IMPLEMENTATION                   │
│ ├─ Implement LDPC encoding (CommPy)                       │
│ │  • Input: Binary payload                               │
│ │  • Output: Encoded + parity bits                       │
│ │  • Configurable overhead: 10-40%                       │
│ ├─ Implement Reed-Solomon encoding (reedsolo)            │
│ ├─ Build hybrid FEC system                                │
│ ├─ Test robustness with packet loss simulation           │
│ ├─ Verify: Can recover from 30% packet loss              │
│ └─ Deliverable: fec_encoder.py (production-ready)        │
│                                                             │
│ PHASE 1 (Weeks 5-6): OFDM MODULATION                      │
│ ├─ OFDM subcarrier generation (NumPy/SciPy)             │
│ ├─ Configurable parameters:                              │
│ │  • Subcarriers: 4096, 8192, 16384                     │
│ │  • Guard interval: 1/12 or 1/32                       │
│ │  • Modulation: QPSK, 16-QAM, 256-QAM, 1024-QAM       │
│ ├─ IFFT-based implementation                             │
│ ├─ Cyclic prefix insertion                               │
│ ├─ Unit tests for modulation accuracy                    │
│ └─ Deliverable: ofdm_modulator.py                        │
│                                                             │
│ PHASE 1 (Weeks 5-6): RF CHANNEL SIMULATION                │
│ ├─ Build channel simulator (NumPy)                        │
│ ├─ Model: Rayleigh/Rician fading                         │
│ ├─ Fading types:                                          │
│ │  • Rayleigh: Severe multipath (urban)                  │
│ │  • Rician: One strong path + multipath                 │
│ │  • AWGN: Additive white Gaussian noise                 │
│ ├─ Configurable parameters:                              │
│ │  • SNR (signal-to-noise ratio)                         │
│ │  • Fading speed                                        │
│ │  • Doppler frequency                                   │
│ ├─ Generate realistic channel responses                   │
│ └─ Deliverable: rf_channel_simulator.py                  │
│                                                             │
│ PHASE 1 (Weeks 6-7): MULTIPLE PLP SYSTEM                  │
│ ├─ Design 3 Physical Layer Pipes:                        │
│ │  PLP_0: Mobile (most robust, low bitrate)             │
│ │  PLP_1: Hybrid (balanced)                              │
│ │  PLP_2: Fixed (high capacity, less robust)            │
│ ├─ Implement separate encoding per PLP                   │
│ ├─ Route data: RTCM → PLP_0, Tiles → PLP_1              │
│ ├─ Test receiver can decode selected PLPs               │
│ └─ Deliverable: plp_system.py                            │
│                                                             │
│ PHASE 2 (Weeks 7-9): BROADCAST SCHEDULER                  │
│ ├─ Build packet scheduler                                │
│ ├─ Input: RTCM frames + tile data + redundancy factor   │
│ ├─ Output: Scheduled transmission queue                  │
│ ├─ Features:                                              │
│ │  • Priority queuing (RTCM high, tiles medium)         │
│ │  • Redundancy insertion (send duplicates if factor>1)  │
│ │  • Rate limiting (0.5-5 Hz)                           │
│ │  • Load balancing across PLPs                         │
│ ├─ Test: 100+ scheduling scenarios                       │
│ └─ Deliverable: broadcast_scheduler.py                   │
│                                                             │
│ PHASE 2 (Weeks 7-8): AI INTERFACE & DECISION ENGINE       │
│ ├─ Build controller that receives AI commands            │
│ ├─ Parse JSON broadcast configuration                     │
│ ├─ Apply dynamic parameter adjustments:                  │
│ │  • Redundancy (1.0 → 2.0x)                             │
│ │  • Update frequency (0.5 → 5.0 Hz)                    │
│ │  • FEC overhead (10 → 40%)                             │
│ │  • PLP distribution                                    │
│ ├─ Real-time parameter updates                           │
│ ├─ Fallback logic if AI confidence < 0.5                 │
│ └─ Deliverable: broadcast_controller.py                  │
│                                                             │
│ PHASE 2 (Weeks 8-9): CHANNEL METRICS COLLECTION           │
│ ├─ Track broadcast performance:                          │
│ │  • Packet loss rate (with/without FEC)                │
│ │  • Latency end-to-end                                 │
│ │  • Fading characteristics per scenario                │
│ │  • SNR distribution                                    │
│ │  • Bit error rates                                     │
│ ├─ Output: CSV file (channel_metrics.csv)               │
│ ├─ Generate 1000+ scenarios                              │
│ ├─ Feed to AI team for training                          │
│ └─ Deliverable: metrics_collector.py + 500KB CSV         │
│                                                             │
│ PHASE 3 (Weeks 10-12): PRODUCTION OPTIMIZATION            │
│ ├─ Performance profiling                                  │
│ ├─ Bottleneck elimination                                │
│ ├─ Real-time constraints: < 50ms per broadcast frame    │
│ ├─ Memory efficiency                                      │
│ ├─ Parallelization (if needed)                           │
│ ├─ Documentation & API reference                         │
│ ├─ Edge deployment optimization                          │
│ └─ Deliverable: Production broadcast module + docs      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Deliverables

```
Week 2:
  ✓ ATSC 3.0 fundamentals document
  ✓ OFDM reference implementation
  ✓ CommPy/reedsolo installed & tested

Week 6:
  ✓ alp_encoder.py (tested with 1000 frames)
  ✓ fec_encoder.py (verified 30% loss recovery)
  ✓ ofdm_modulator.py (verified accuracy)
  ✓ rf_channel_simulator.py (3 fading models)

Week 9:
  ✓ broadcast_scheduler.py (100+ test cases)
  ✓ broadcast_controller.py (receives AI commands)
  ✓ channel_metrics.csv (1000 scenarios)
  ✓ Integration tests with GNSS Team (90%+ pass)

Week 12:
  ✓ Production broadcast module
  ✓ API documentation (40+ pages)
  ✓ Performance benchmarks
  ✓ Jupyter notebooks with examples
```

### Technical Skills Required

```
✓ ATSC 3.0 specification knowledge
✓ OFDM fundamentals
✓ FEC/Error correction theory (LDPC, Reed-Solomon)
✓ RF/Signal processing (NumPy/SciPy)
✓ Python 3.9+ (advanced)
✓ Digital communication theory
✓ Modulation schemes
✓ Channel modeling

Nice to have:
✓ CommPy library expertise
✓ Multiprocessing/parallel programming
✓ Real-time systems experience
```

---

## 3.4 TEAM 3: AI/ML Systems Engineer :- TARUNIKA

### Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│          AI/ML ENGINEER: COMPLETE SCOPE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ PHASE 0-1 (Weeks 1-2): FUNDAMENTALS & DATA PREP            │
│ ├─ Study RTK positioning metrics                          │
│ ├─ Study broadcast optimization concepts                  │
│ ├─ Setup PyTorch environment (CPU/GPU)                   │
│ ├─ Data preprocessing pipeline                           │
│ │  • Load JSONL telemetry from GNSS Team                │
│ │  • Load CSV channel metrics from Broadcast Team        │
│ │  • Data cleaning & normalization                       │
│ │  • Feature engineering                                 │
│ ├─ Exploratory Data Analysis (EDA)                        │
│ │  • Visualize fleet RTK mode distributions             │
│ │  • Analyze convergence time patterns                   │
│ │  • Identify correlations with environment             │
│ ├─ Train/test split (70/20/10)                           │
│ └─ Deliverable: data_preprocessor.py + EDA notebook     │
│                                                             │
│ PHASE 1 (Weeks 3-5): NEURAL NETWORK DESIGN                │
│ ├─ Architecture design (PyTorch)                         │
│ ├─ Input features (dimension: 50+):                      │
│ │  • Fleet RTK modes (% in FIX/FLOAT/STAND-ALONE)       │
│ │  • Average convergence times                           │
│ │  • Signal strength statistics                         │
│ │  • Satellite count distribution                       │
│ │  • Environmental factors (urban_density, tunnel?)     │
│ │  • Geographic coordinates (lat/lon)                   │
│ │  • Time of day, vehicle speeds                        │
│ │  • Historical channel conditions                      │
│ │                                                        │
│ ├─ Output (dimension: 5):                                │
│ │  • Redundancy factor (1.0-2.0x)                       │
│ │  • Update frequency (0.5-5.0 Hz)                      │
│ │  • Tile resolution (0-1 scale)                        │
│ │  • FEC overhead (10-40%)                              │
│ │  • PLP distribution (0-1 scale)                       │
│ │                                                        │
│ ├─ Model architecture:                                   │
│ │  • Input layer: 50 neurons                             │
│ │  • Hidden layer 1: 128 neurons (ReLU)                │
│ │  • Hidden layer 2: 64 neurons (ReLU)                 │
│ │  • Hidden layer 3: 32 neurons (ReLU)                 │
│ │  • Output layer: 5 neurons (various activations)     │
│ │  • Dropout: 0.2 for regularization                   │
│ │  • Batch normalization between layers                │
│ │                                                        │
│ ├─ Loss function: MSE (mean squared error)              │
│ ├─ Optimizer: Adam (learning rate 0.001)                │
│ └─ Deliverable: broadcast_decision_model.py (PyTorch)  │
│                                                             │
│ PHASE 1 (Weeks 4-6): TRAINING & VALIDATION                │
│ ├─ Training loop implementation:                         │
│ │  • 10,000 samples from GNSS dataset                   │
│ │  • Batch size: 32                                     │
│ │  • Epochs: 100-200 (early stopping at 50)            │
│ │  • Validation every 10 epochs                         │
│ │  • TensorBoard logging                                │
│ │                                                        │
│ ├─ Performance metrics:                                  │
│ │  • Training loss: Target < 0.01                       │
│ │  • Validation loss: Target < 0.015                    │
│ │  • MSE per output: Individual tracking                │
│ │                                                        │
│ ├─ Hyperparameter tuning:                               │
│ │  • Learning rate sweep: 0.0001-0.01                  │
│ │  • Layer depths: 1-5 hidden layers                    │
│ │  • Batch sizes: 16-128                                │
│ │  • Dropout rates: 0.1-0.5                             │
│ │                                                        │
│ ├─ Cross-validation:                                     │
│ │  • K-fold validation (k=5)                            │
│ │  • Verify consistency across folds                    │
│ │  • Report std deviation of accuracy                   │
│ │                                                        │
│ ├─ Error analysis:                                       │
│ │  • Confusion on which scenarios?                      │
│ │  • Outlier detection                                  │
│ │  • Failure mode analysis                              │
│ │                                                        │
│ └─ Deliverable: Trained model (model_v1.pth)            │
│                Model validation report (20 pages)        │
│                TensorBoard logs                          │
│                                                             │
│ PHASE 2 (Weeks 7-8): MODEL INFERENCE & DEPLOYMENT         │
│ ├─ Build inference wrapper:                              │
│ │  • Load trained model                                 │
│ │  • Preprocess input telemetry                         │
│ │  • Run forward pass                                   │
│ │  • Post-process outputs (constrain to valid ranges)  │
│ │  • Return broadcast command JSON                      │
│ │                                                        │
│ ├─ Real-time performance:                               │
│ │  • Latency target: < 50ms per inference              │
│ │  • Throughput: 100+ inferences/sec                    │
│ │  • Memory: < 100 MB                                   │
│ │                                                        │
│ ├─ Confidence scoring:                                  │
│ │  • Predict confidence for each decision              │
│ │  • Threshold: 0.75 (if below, use rule-based)        │
│ │  • Provide explanation of decision                    │
│ │                                                        │
│ ├─ Fallback logic:                                       │
│ │  • Rule-based system for confidence < 0.75           │
│ │  • Conservative defaults (1.0x redundancy, etc)      │
│ │  • Graceful degradation                               │
│ │                                                        │
│ └─ Deliverable: inference_engine.py                      │
│                                                             │
│ PHASE 2 (Weeks 8-9): MODEL EXPORT & OPTIMIZATION           │
│ ├─ Export to ONNX format:                                │
│ │  • PyTorch → ONNX conversion                          │
│ │  • Verify outputs match                               │
│ │  • Test on CPU and GPU (if available)                │
│ │                                                        │
│ ├─ Model compression techniques:                        │
│ │  • Quantization (FP32 → INT8 if speed needed)        │
│ │  • Pruning (remove small-weight connections)         │
│ │  • Distillation (smaller model mimicking large)       │
│ │  • Trade-off: Accuracy vs. speed/size                │
│ │                                                        │
│ ├─ Edge deployment optimization:                        │
│ │  • Target: Qualcomm Snapdragon (final phase)         │
│ │  • ONNX Runtime support verification                  │
│ │  • Benchmark on development hardware                  │
│ │  • Memory profile for embedded systems                │
│ │                                                        │
│ ├─ Testing on real scenarios:                           │
│ │  • Load 100 unseen test scenarios                    │
│ │  • Verify model outputs are reasonable                │
│ │  • Compare to rule-based baseline                     │
│ │  • Document any surprising behaviors                  │
│ │                                                        │
│ └─ Deliverable: model_v1.onnx + optimization report    │
│                                                             │
│ PHASE 2 (Weeks 9): FEEDBACK LOOP INTEGRATION              │
│ ├─ Build feedback receiver:                              │
│ │  • Listen for vehicle telemetry via message queue    │
│ │  • Parse and validate incoming data                   │
│ │  • Store in database (HDF5)                           │
│ │                                                        │
│ ├─ Online learning preparation:                         │
│ │  • Design periodic retraining schedule (daily)       │
│ │  • Plan model versioning system                       │
│ │  • Safety checks (don't degrade performance)         │
│ │  • Rollback mechanism if new model worse              │
│ │                                                        │
│ ├─ Performance monitoring:                              │
│ │  • Track decision effectiveness                       │
│ │  • Monitor prediction confidence                      │
│ │  • Detect model drift                                 │
│ │  • Alert if performance degrades                      │
│ │                                                        │
│ └─ Deliverable: feedback_integration.py                  │
│                Retraining pipeline (documentation)       │
│                                                             │
│ PHASE 3 (Weeks 10-12): PRODUCTION POLISH & DEPLOYMENT     │
│ ├─ Documentation & refactoring:                          │
│ │  • Code review & optimization                        │
│ │  • Type hints (mypy compliance)                       │
│ │  • Docstrings & API reference                        │
│ │  • 95%+ unit test coverage                            │
│ │                                                        │
│ ├─ API finalization:                                     │
│ │  • RESTful endpoint for inference                     │
│ │  • Model serving (Flask/FastAPI)                      │
│ │  • Input validation & error handling                  │
│ │  • Rate limiting & monitoring                         │
│ │                                                        │
│ ├─ Edge deployment (Qualcomm AI Hub):                    │
│ │  • Export ONNX to Qualcomm Hub format                │
│ │  • Test on Snapdragon emulator                        │
│ │  • Verify latency & accuracy                          │
│ │  • Measure power consumption                          │
│ │                                                        │
│ ├─ Integration testing with all teams:                   │
│ │  • End-to-end system tests                            │
│ │  • Scenario: tunnel crossing (simulated)              │
│ │  • Verify decisions affect broadcast correctly        │
│ │  • Verify vehicles respond to broadcast changes       │
│ │  • Performance under load (100+ vehicles)             │
│ │                                                        │
│ ├─ Final documentation:                                  │
│ │  • AI model architecture whitepaper                   │
│ │  • Training dataset analysis                          │
│ │  • Performance benchmarks                             │
│ │  • Limitations & future work                          │
│ │                                                        │
│ └─ Deliverable: Production AI module                     │
│                Model documentation (50+ pages)           │
│                Jupyter notebooks                         │
│                Qualcomm Hub export                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Deliverables

```
Week 2:
  ✓ data_preprocessor.py (JSONL + CSV handling)
  ✓ EDA notebook showing data patterns
  ✓ Cleaned training/test datasets

Week 6:
  ✓ broadcast_decision_model.py (PyTorch architecture)
  ✓ Training logs & loss curves
  ✓ Model validation report (20 pages)
  ✓ model_v1.pth (trained weights)

Week 9:
  ✓ inference_engine.py (real-time inference)
  ✓ model_v1.onnx (ONNX export)
  ✓ feedback_integration.py (telemetry receiver)
  ✓ Integration tests (90%+ pass rate)

Week 12:
  ✓ Production AI module (fully tested)
  ✓ API documentation (30+ pages)
  ✓ Model documentation (50+ pages)
  ✓ Qualcomm Hub export package
  ✓ Jupyter notebooks with examples
  ✓ Performance benchmarks (latency, accuracy)
```

### Technical Skills Required

```
✓ PyTorch fundamentals (model building, training)
✓ Neural network design & architecture
✓ Data preprocessing (NumPy, Pandas)
✓ Hyperparameter tuning
✓ Model evaluation & validation
✓ ONNX export & optimization
✓ Python 3.9+ (advanced)
✓ TensorBoard for monitoring

Nice to have:
✓ Quantization & model compression
✓ Time series prediction
✓ Feature engineering
✓ Bayesian optimization
✓ Distributed training
✓ FastAPI/Flask for serving
```

---

## 3.5 12-Week Project Timeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                      12-WEEK PROJECT TIMELINE                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ PHASE 0 (Weeks 1-2): FOUNDATION & SETUP                             │
│                                                                       │
│ Week 1:                                                              │
│   Mon-Tue:   Team meetings, repository setup, branch creation        │
│   Wed-Fri:   All teams: Install tools, verify imports               │
│                         Read specifications & technical docs          │
│   Deliverable: Development environment ready for all 3 members      │
│                                                                       │
│ Week 2:                                                              │
│   Mon-Wed:   GNSS Team: Download GNSS sample data                   │
│             Broadcast Team: Study ATSC 3.0 / OFDM theory            │
│             AI Team: Study RTK metrics & data structures             │
│   Thu-Fri:   Sync meeting: Discuss data formats & APIs              │
│   Deliverable: Technical reference documents ready                  │
│                Individual foundational work completed                 │
│                                                                       │
│ ───────────────────────────────────────────────────────────────────  │
│                                                                       │
│ PHASE 1 (Weeks 3-6): CORE MODULE DEVELOPMENT                        │
│                                                                       │
│ Week 3:                                                              │
│   GNSS Team:      Start RTCM generator (Week 3-4 focus)              │
│   Broadcast Team: Study FEC libraries, start ALP encoder             │
│   AI Team:        Data preprocessing pipeline                        │
│                                                                       │
│ Week 4:                                                              │
│   GNSS Team:      RTCM generator complete, 1000 test frames         │
│   Broadcast Team: ALP encoder + FEC encoder design                  │
│   AI Team:        EDA notebook, feature engineering                  │
│   Sync: Share RTCM sample frames with Broadcast Team               │
│                                                                       │
│ Week 5:                                                              │
│   GNSS Team:      Coverage map generator, scenario simulator        │
│   Broadcast Team: OFDM modulator, RF channel simulator              │
│   AI Team:        Neural network architecture design                │
│   Sync: Broadcast Team receives GNSS training dataset               │
│                                                                       │
│ Week 6:                                                              │
│   GNSS Team:      10K scenarios complete, dataset 50 MB             │
│   Broadcast Team: Multiple PLP system, broadcast scheduler          │
│   AI Team:        Start model training (PyTorch)                    │
│   Sync: All core modules completed, integration planning            │
│                                                                       │
│ ───────────────────────────────────────────────────────────────────  │
│                                                                       │
│ PHASE 2 (Weeks 7-9): INTEGRATION & AI TRAINING                      │
│                                                                       │
│ Week 7:                                                              │
│   GNSS Team:      Integration tests with Broadcast (start)          │
│   Broadcast Team: AI interface module, decision controller           │
│   AI Team:        Model training (epochs 1-50)                      │
│   Parallel: Channel metrics collection                               │
│                                                                       │
│ Week 8:                                                              │
│   GNSS Team:      E2E validation (end-to-end tests)                 │
│   Broadcast Team: Scheduler optimization, metrics complete          │
│   AI Team:        Model training complete, validation               │
│             Inference engine development                             │
│   Sync: AI model ready, broadcast can apply decisions               │
│                                                                       │
│ Week 9:                                                              │
│   GNSS Team:      Tunnel scenarios deep dive (100+ cases)           │
│   Broadcast Team: Performance optimization, profiling                │
│   AI Team:        ONNX export, feedback loop design                 │
│   Integration:    Full system tests (all 3 teams together)          │
│             Tunnel crossing simulation                              │
│   Sync: Phase 2 completion, readiness for Phase 3                   │
│                                                                       │
│ ───────────────────────────────────────────────────────────────────  │
│                                                                       │
│ PHASE 3 (Weeks 10-12): PRODUCTION & DEPLOYMENT                      │
│                                                                       │
│ Week 10:                                                             │
│   All Teams:      Code refactoring & optimization                   │
│             Unit test coverage >95%                                 │
│             Documentation (API references)                           │
│   AI Team:        Model compression research (quantization)          │
│   Broadcast Team: Edge optimization                                  │
│                                                                       │
│ Week 11:                                                             │
│   All Teams:      Final integration testing                          │
│             Performance benchmarking                                 │
│             Load testing (100+ vehicles)                             │
│   AI Team:        Qualcomm AI Hub export                             │
│   Broadcast Team: Real-time constraints verification                │
│   GNSS Team:      Final validation                                   │
│                                                                       │
│ Week 12:                                                             │
│   All Teams:      Production readiness review                        │
│             Final documentation                                     │
│             Jupyter notebooks & examples                            │
│             Performance reports & benchmarks                        │
│   Demo:           Full system demonstration                          │
│             Tunnel scenario walkthrough                             │
│             Broadcast adaptation shown live                         │
│   Deliverable:    Production-ready PoC                              │
│                                                                       │
│ ───────────────────────────────────────────────────────────────────  │
│                                                                       │
│ WEEKLY SYNC SCHEDULE:                                                │
│ • Monday 10:00 AM:  Problem-solving sync (30 min)                   │
│ • Wednesday 3:00 PM: Technical deep-dive (1 hour)                   │
│ • Friday 11:00 AM:  Integration checkpoint (45 min)                 │
│                                                                       │
│ PHASE GATES (Must Pass):                                             │
│ • Phase 0 → 1: All dev environments ready ✓                         │
│ • Phase 1 → 2: All core modules functioning ✓                       │
│ • Phase 2 → 3: Full system integration tests 90%+ ✓                 │
│ • Phase 3 Complete: Production release ready ✓                      │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3.6 Inter-Team Communication & APIs

### Data Handoff: GNSS Team → Broadcast Team

```
INTERFACE 1: RTCM Frames
┌────────────────────────────────┐
│ GNSS Team Produces             │
│ • Binary RTCM 3.x frames       │
│ • Size: 100-300 bytes each     │
│ • Rate: 1 per second           │
│ • Format: Binary with CRC      │
└────────────────────────────────┘
         │
         ▼ (Python File I/O)
┌────────────────────────────────┐
│ Location: data/rtcm_frames/    │
│ Filename: rtcm_YYYYMMDD.bin    │
│ Checksum: MD5 included         │
│ Size: ~100 KB per hour         │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Broadcast Team Consumes        │
│ • Parses frames in sequence    │
│ • Verifies CRC checksums       │
│ • Wraps in ALP packets         │
│ • Schedules for transmission   │
└────────────────────────────────┘

API Function (Python):
  frame = rtcm_gen.generate_frame(gps_err, glonass_err)
  # frame is bytes, 100-300 long
  # broadcast_team.queue_frame(frame, timestamp)
```

### Data Handoff: GNSS Team → AI Team

```
INTERFACE 2: Training Telemetry Dataset
┌────────────────────────────────┐
│ GNSS Team Produces             │
│ • JSON-Lines (.jsonl) format   │
│ • One scenario per line        │
│ • 10,000+ scenarios covering:  │
│   - Open sky                   │
│   - Urban canyon               │
│   - Tunnel scenarios           │
│   - Signal degradation         │
│   - False positives            │
└────────────────────────────────┘
         │
         ▼ (File System)
┌────────────────────────────────┐
│ Location: data/training_data/  │
│ Filename: telemetry_phase1.jsonl
│ Size: ~50 MB for 10K scenarios │
│ Checksum: Included             │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ AI Team Consumes               │
│ • Loads with jsonlines library │
│ • Parses JSON objects          │
│ • Feature extraction           │
│ • Training data preparation    │
└────────────────────────────────┘

Schema (JSON):
  {
    "scenario_id": "tunnel_001",
    "rtk_mode": "FIX",
    "position_error_cm": 1.5,
    "num_satellites": 12,
    ... (50+ fields)
  }
```

### Data Handoff: Broadcast Team → AI Team

```
INTERFACE 3: Channel Performance Metrics
┌────────────────────────────────┐
│ Broadcast Team Produces        │
│ • CSV format (standard)        │
│ • One scenario per row         │
│ • 1000+ RF channel simulations │
│ • Metrics:                     │
│   - Packet loss (no/with FEC)  │
│   - Latency                    │
│   - SNR, BER                   │
│   - Fading model               │
└────────────────────────────────┘
         │
         ▼ (File System)
┌────────────────────────────────┐
│ Location: data/results/        │
│ Filename: channel_metrics.csv  │
│ Size: ~500 KB for 1000 rows    │
│ Headers: Included              │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ AI Team Consumes               │
│ • Loads with pandas            │
│ • Analyzes correlations        │
│ • Feature engineering          │
│ • Training data enrichment     │
└────────────────────────────────┘

CSV Schema:
  scenario_id,packet_loss_no_fec_pct,packet_loss_with_fec_pct,...
  scenario_001,5.2,0.8,...
  scenario_002,4.9,0.6,...
```

### AI Command: AI Team → Broadcast Team

```
INTERFACE 4: Real-Time Broadcast Decisions
┌────────────────────────────────┐
│ AI Team Produces               │
│ • JSON format (real-time)      │
│ • Broadcast configuration      │
│ • Sent via message queue       │
│ • Update frequency: 1-10 sec   │
└────────────────────────────────┘
         │
         ▼ (Message Queue / RabbitMQ / Redis)
┌────────────────────────────────┐
│ Topic: broadcast_commands      │
│ Format: JSON (3-4 KB)         │
│ Latency: <100ms               │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Broadcast Team Consumes        │
│ • Parses JSON command          │
│ • Validates parameters         │
│ • Applies to transmission      │
│ • Takes effect immediately    │
└────────────────────────────────┘

JSON Schema:
  {
    "timestamp": 1705094500,
    "broadcast_config": {
      "redundancy": 1.2,
      "tile_resolution": "high",
      "update_frequency_hz": 2.0,
      "plp_mode": "mobile",
      "fec_overhead_pct": 25
    },
    "confidence": 0.92
  }
```

---

## 3.7 Repository Structure & Git Workflow

```
ai-positioning-poc/
├── README.md                    # Project overview
├── requirements.txt             # Python dependencies (18 tools)
├── setup.sh                     # First-time setup script
│
├── src/                         # Source code (by team)
│   ├── gnss/
│   │   ├── __init__.py
│   │   ├── rtcm_generator.py    # TEAM 1
│   │   ├── coverage_map_gen.py  # TEAM 1
│   │   ├── scenario_simulator.py # TEAM 1
│   │   └── tests/
│   │       ├── test_rtcm.py
│   │       ├── test_coverage.py
│   │       └── test_scenarios.py
│   │
│   ├── broadcast/
│   │   ├── __init__.py
│   │   ├── alp_encoder.py       # TEAM 2
│   │   ├── fec_encoder.py       # TEAM 2
│   │   ├── ofdm_modulator.py    # TEAM 2
│   │   ├── rf_channel_sim.py    # TEAM 2
│   │   ├── plp_system.py        # TEAM 2
│   │   ├── scheduler.py         # TEAM 2
│   │   ├── controller.py        # TEAM 2
│   │   └── tests/
│   │       ├── test_alp.py
│   │       ├── test_fec.py
│   │       ├── test_ofdm.py
│   │       └── test_scheduler.py
│   │
│   └── ai_ml/
│       ├── __init__.py
│       ├── data_preprocessor.py # TEAM 3
│       ├── model_builder.py     # TEAM 3
│       ├── training.py          # TEAM 3
│       ├── inference.py         # TEAM 3
│       ├── feedback_loop.py     # TEAM 3
│       └── tests/
│           ├── test_preprocessing.py
│           ├── test_model.py
│           └── test_inference.py
│
├── data/
│   ├── gnss_logs/               # Sample GNSS data
│   │   └── sample_data_*.rinex
│   │
│   ├── training_data/           # AI training datasets
│   │   ├── telemetry_phase1.jsonl     (from GNSS Team)
│   │   └── channel_metrics.csv        (from Broadcast Team)
│   │
│   ├── results/                 # Output & results
│   │   ├── models/
│   │   │   └── model_v1.pth
│   │   │   └── model_v1.onnx
│   │   └── metrics/
│   │       └── performance_report.csv
│   │
│   └── rtcm_frames/             # RTCM test frames
│       └── rtcm_YYYYMMDD.bin
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      (TEAM 3)
│   ├── 02_model_training.ipynb        (TEAM 3)
│   ├── 03_rtcm_analysis.ipynb         (TEAM 1)
│   ├── 04_broadcast_performance.ipynb (TEAM 2)
│   └── 05_end_to_end_demo.ipynb       (All Teams)
│
├── docs/
│   ├── PHASE0_SETUP.md          # Environment setup
│   ├── PHASE1_ARCHITECTURE.md   # Technical architecture
│   ├── PHASE2_INTEGRATION.md    # Integration guide
│   ├── PHASE3_DEPLOYMENT.md     # Deployment guide
│   ├── API_REFERENCE.md         # API documentation
│   ├── DATA_FORMATS.md          # Data schema reference
│   └── TROUBLESHOOTING.md       # Common issues
│
├── tests/                       # Integration tests
│   ├── test_gnss_broadcast_integration.py
│   ├── test_broadcast_ai_integration.py
│   ├── test_full_system.py
│   └── test_tunnel_scenario.py
│
├── CI_CD/                       # Continuous integration
│   ├── .github/
│   │   └── workflows/
│   │       ├── unit_tests.yml   # Run on every commit
│   │       ├── integration_tests.yml # Run nightly
│   │       └── benchmark.yml    # Run weekly
│   │
│   └── Dockerfile               # For containerization (Phase 3)
│
└── .gitignore
```

### Git Branching Strategy

```
main (master)
  ├─ develop (integration branch)
  │   ├─ feature/gnss-rtcm-generator (TEAM 1)
  │   ├─ feature/gnss-coverage-maps (TEAM 1)
  │   ├─ feature/broadcast-alp-encoder (TEAM 2)
  │   ├─ feature/broadcast-fec (TEAM 2)
  │   ├─ feature/ai-model-v1 (TEAM 3)
  │   ├─ feature/ai-inference (TEAM 3)
  │   ├─ integration/gnss-broadcast (Phase 2)
  │   ├─ integration/broadcast-ai (Phase 2)
  │   └─ integration/full-system (Phase 3)
  │
  └─ release/v1.0.0 (when ready)

Commit Discipline:
  ✓ One logical unit per commit
  ✓ Never commit broken code
  ✓ Descriptive commit messages
  ✗ No "WIP" or "fix" commits to main/develop
  ✓ Pull request required for all changes
  ✓ Code review by another team member
```

---

## 3.8 Success Metrics & Acceptance Criteria

```
PHASE 0 ACCEPTANCE:
✓ All 18 tools installed and verified
✓ All 3 team members can run existing code
✓ Repository configured with CI/CD
✓ First week of development started
Measurement: "Are the lights on?"

PHASE 1 ACCEPTANCE:
✓ GNSS: 10,000 scenarios generated, 50 MB dataset ✓
✓ Broadcast: All core modules completed ✓
✓ AI: Data preprocessed, model architecture designed ✓
✓ Unit test coverage: >90% for each module
✓ Code review: All PRs approved
Measurement: "Do individual parts work?"

PHASE 2 ACCEPTANCE:
✓ Integration tests: 90%+ pass rate ✓
✓ E2E validation: RTCM → Broadcast → RTK proven ✓
✓ AI model: Training loss < 0.015, validation complete ✓
✓ Tunnel scenarios: 100+ cases simulated ✓
✓ Channel metrics: 1000 RF scenarios analyzed ✓
✓ No critical bugs in integrated system
Measurement: "Do parts work together?"

PHASE 3 ACCEPTANCE:
✓ Performance benchmarks: All targets met
✓ Production code quality: 95%+ unit test coverage
✓ Documentation: Complete (API ref, tutorials, examples)
✓ Qualcomm Hub export: ONNX model verified
✓ Real-time latency: <50ms per broadcast frame
✓ Memory: <100 MB for AI inference
✓ Scalability: Tested with 100+ simulated vehicles
✓ Demo ready: Full tunnel scenario walkthrough
Measurement: "Is it production-ready?"
```
---