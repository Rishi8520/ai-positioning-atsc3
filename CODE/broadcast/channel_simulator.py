#!/usr/bin/env python3
# filepath: /media/rishi/New SSD/PROJECT_AND_RESEARCH/BUILD-A-THON_4/CODE/broadcast/channel_simulator.py
"""
ATSC 3.0 Channel Simulator

PURPOSE:
Simulates real-world RF channel effects for testing broadcast robustness.
Models various impairments that affect signal reception in different environments.

CHANNEL MODELS SUPPORTED:
1. AWGN (Additive White Gaussian Noise) - Thermal noise
2. Multipath (Echoes) - Reflections from buildings, terrain
3. Rayleigh Fading - Dense urban, no line-of-sight
4. Rician Fading - Suburban, dominant line-of-sight
5. Doppler Shift - Motion-induced frequency shift
6. Impulse Noise - Electrical interference, lightning
7. Path Loss - Distance-based attenuation
8. Shadowing - Slow fading from obstacles

ENVIRONMENT PRESETS:
- RURAL: Open sky, minimal multipath, low noise
- SUBURBAN: Moderate multipath, some fading
- URBAN: Severe multipath, Rayleigh fading, high noise
- HIGHWAY: High Doppler, moderate multipath
- INDOOR: Severe multipath, shadowing

USAGE:
    from broadcast.channel_simulator import ChannelSimulator, ChannelConfig
    
    # Quick simulation with preset
    simulator = ChannelSimulator.from_preset("urban")
    degraded_signal = simulator.apply(ofdm_signal)
    
    # Custom configuration
    config = ChannelConfig(
        snr_db=15.0,
        multipath_enabled=True,
        num_paths=6,
        max_delay_us=20.0,
        doppler_hz=50.0
    )
    simulator = ChannelSimulator(config)
    result = simulator.apply(signal, return_metrics=True)

INTEGRATION:
- Used after: broadcast/pipeline.py (OFDM output)
- Used before: broadcast/decoder.py (reception)
- Called by: gnss/rtcm_to_broadcast_handoff.py (robustness testing)
- Configured by: broadcast/broadcast_controller.py

NOTES:
- All effects can be individually enabled/disabled
- Reproducible results via random seed
- Metrics provided for analysis
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import special
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class EnvironmentType(Enum):
    """Predefined environment types with typical channel characteristics."""
    RURAL = auto()          # Open area, clear sky
    SUBURBAN = auto()       # Residential area, some obstructions
    URBAN = auto()          # Dense buildings, severe multipath
    URBAN_CANYON = auto()   # Street between tall buildings
    HIGHWAY = auto()        # High-speed vehicle on highway
    INDOOR = auto()         # Inside building
    MOUNTAINOUS = auto()    # Terrain-induced effects
    MARITIME = auto()       # Over water
    

class FadingType(Enum):
    """Types of fading models."""
    NONE = auto()           # No fading
    RAYLEIGH = auto()       # No dominant path (dense urban)
    RICIAN = auto()         # Dominant LOS path (suburban/rural)
    NAKAGAMI = auto()       # Generalized fading model
    LOGNORMAL = auto()      # Shadowing/slow fading


class NoiseType(Enum):
    """Types of noise models."""
    AWGN = auto()           # Additive White Gaussian Noise
    COLORED = auto()        # Colored/filtered noise
    IMPULSE = auto()        # Impulse noise (interference)
    PHASE = auto()          # Phase noise


class MultipathProfile(Enum):
    """Predefined multipath delay profiles."""
    FLAT = auto()           # Single path, no delay spread
    PEDESTRIAN_A = auto()   # ITU Pedestrian A
    PEDESTRIAN_B = auto()   # ITU Pedestrian B
    VEHICULAR_A = auto()    # ITU Vehicular A
    VEHICULAR_B = auto()    # ITU Vehicular B
    URBAN_MACRO = auto()    # 3GPP Urban Macro
    INDOOR_OFFICE = auto()  # Indoor office environment
    CUSTOM = auto()         # User-defined profile


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class MultipathTap:
    """Single tap in multipath channel model."""
    delay_us: float          # Delay in microseconds
    amplitude: float         # Relative amplitude (linear)
    phase_deg: float = 0.0   # Phase shift in degrees
    doppler_hz: float = 0.0  # Doppler shift for this path


@dataclass
class ChannelConfig:
    """Configuration for channel simulator."""
    
    # ========================================
    # Noise Configuration
    # ========================================
    snr_db: float = 20.0              # Signal-to-noise ratio
    noise_enabled: bool = True
    noise_type: NoiseType = NoiseType.AWGN
    
    # Impulse noise parameters
    impulse_rate: float = 0.001       # Probability per sample
    impulse_amplitude: float = 5.0    # Relative to signal
    
    # ========================================
    # Multipath Configuration
    # ========================================
    multipath_enabled: bool = True
    multipath_profile: MultipathProfile = MultipathProfile.PEDESTRIAN_A
    num_paths: int = 4                # Number of multipath taps
    max_delay_us: float = 10.0        # Maximum delay spread
    custom_taps: List[MultipathTap] = field(default_factory=list)
    
    # ========================================
    # Fading Configuration
    # ========================================
    fading_enabled: bool = False
    fading_type: FadingType = FadingType.RAYLEIGH
    rician_k_factor: float = 6.0      # Rician K-factor (dB)
    nakagami_m: float = 1.0           # Nakagami-m parameter
    coherence_time_ms: float = 10.0   # Fading coherence time
    
    # ========================================
    # Doppler Configuration
    # ========================================
    doppler_enabled: bool = False
    doppler_hz: float = 0.0           # Maximum Doppler frequency
    velocity_kmh: float = 0.0         # Alternatively, specify velocity
    carrier_freq_mhz: float = 600.0   # Carrier frequency for Doppler calc
    
    # ========================================
    # Path Loss Configuration
    # ========================================
    path_loss_enabled: bool = False
    path_loss_db: float = 0.0         # Fixed path loss
    distance_km: float = 0.0          # Distance for path loss calculation
    path_loss_exponent: float = 3.5   # Path loss exponent (2=free space, 4=urban)
    
    # ========================================
    # Shadowing Configuration
    # ========================================
    shadowing_enabled: bool = False
    shadowing_std_db: float = 8.0     # Log-normal standard deviation
    shadowing_correlation: float = 0.5  # Correlation distance
    
    # ========================================
    # Simulation Parameters
    # ========================================
    sample_rate_hz: float = 6.144e6   # ATSC 3.0 typical sample rate
    seed: Optional[int] = None        # Random seed for reproducibility
    
    # ========================================
    # Derived Parameters
    # ========================================
    def get_doppler_from_velocity(self) -> float:
        """Calculate Doppler frequency from velocity."""
        if self.velocity_kmh > 0:
            c = 3e8  # Speed of light
            v = self.velocity_kmh * 1000 / 3600  # m/s
            fc = self.carrier_freq_mhz * 1e6  # Hz
            return v * fc / c
        return self.doppler_hz
    
    def get_coherence_bandwidth_hz(self) -> float:
        """Estimate coherence bandwidth from delay spread."""
        if self.max_delay_us > 0:
            delay_spread_s = self.max_delay_us * 1e-6
            return 1 / (5 * delay_spread_s)  # Approximate formula
        return float('inf')


@dataclass
class ChannelMetrics:
    """Metrics from channel simulation."""
    
    # Applied effects
    effects_applied: List[str] = field(default_factory=list)
    
    # Signal quality
    input_power_db: float = 0.0
    output_power_db: float = 0.0
    effective_snr_db: float = 0.0
    
    # Multipath
    delay_spread_us: float = 0.0
    num_significant_paths: int = 0
    
    # Fading
    fading_margin_db: float = 0.0
    doppler_spread_hz: float = 0.0
    
    # Error estimation
    estimated_ber: float = 0.0
    estimated_packet_error_rate: float = 0.0
    
    # Timing
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "effects_applied": self.effects_applied,
            "signal": {
                "input_power_db": self.input_power_db,
                "output_power_db": self.output_power_db,
                "effective_snr_db": self.effective_snr_db
            },
            "multipath": {
                "delay_spread_us": self.delay_spread_us,
                "num_paths": self.num_significant_paths
            },
            "fading": {
                "margin_db": self.fading_margin_db,
                "doppler_hz": self.doppler_spread_hz
            },
            "error_estimation": {
                "ber": self.estimated_ber,
                "per": self.estimated_packet_error_rate
            },
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class ChannelSimulatorResult:
    """Result from channel simulation."""
    
    signal: np.ndarray                 # Degraded signal
    metrics: ChannelMetrics            # Simulation metrics
    config_used: ChannelConfig         # Configuration used
    
    # Optional detailed outputs
    channel_impulse_response: Optional[np.ndarray] = None
    fading_coefficients: Optional[np.ndarray] = None
    noise_samples: Optional[np.ndarray] = None


# ============================================================================
# MULTIPATH CHANNEL
# ============================================================================


class MultipathChannel:
    """
    Multipath channel model.
    
    Implements tapped delay line (TDL) model with configurable
    delay profile and path amplitudes.
    """
    
    # ITU/3GPP Standard Delay Profiles (delays in us, powers in dB)
    STANDARD_PROFILES = {
        MultipathProfile.PEDESTRIAN_A: [
            (0.0, 0.0),
            (0.11, -9.7),
            (0.19, -19.2),
            (0.41, -22.8)
        ],
        MultipathProfile.PEDESTRIAN_B: [
            (0.0, 0.0),
            (0.2, -0.9),
            (0.8, -4.9),
            (1.2, -8.0),
            (2.3, -7.8),
            (3.7, -23.9)
        ],
        MultipathProfile.VEHICULAR_A: [
            (0.0, 0.0),
            (0.31, -1.0),
            (0.71, -9.0),
            (1.09, -10.0),
            (1.73, -15.0),
            (2.51, -20.0)
        ],
        MultipathProfile.VEHICULAR_B: [
            (0.0, -2.5),
            (0.3, 0.0),
            (8.9, -12.8),
            (12.9, -10.0),
            (17.1, -25.2),
            (20.0, -16.0)
        ],
        MultipathProfile.URBAN_MACRO: [
            (0.0, 0.0),
            (0.2, -1.0),
            (0.4, -2.0),
            (0.6, -3.0),
            (0.8, -8.0),
            (1.2, -17.2),
            (1.4, -20.8)
        ],
        MultipathProfile.INDOOR_OFFICE: [
            (0.0, 0.0),
            (0.05, -3.0),
            (0.11, -10.0),
            (0.17, -18.0),
            (0.23, -26.0)
        ]
    }
    
    def __init__(self, config: ChannelConfig):
        """Initialize multipath channel model."""
        self.config = config
        self.sample_rate = config.sample_rate_hz
        self.taps = self._create_taps()
        
        # Pre-compute filter coefficients
        self.filter_coeffs = self._compute_filter_coefficients()
    
    def _create_taps(self) -> List[MultipathTap]:
        """Create multipath taps from configuration."""
        if self.config.custom_taps:
            return self.config.custom_taps
        
        profile = self.config.multipath_profile
        
        if profile == MultipathProfile.FLAT:
            return [MultipathTap(delay_us=0.0, amplitude=1.0)]
        
        if profile == MultipathProfile.CUSTOM:
            # Generate random taps
            return self._generate_random_taps()
        
        if profile in self.STANDARD_PROFILES:
            profile_data = self.STANDARD_PROFILES[profile]
            taps = []
            for delay_us, power_db in profile_data:
                amplitude = 10 ** (power_db / 20)  # Convert dB to linear
                phase = np.random.uniform(0, 360)  # Random phase
                taps.append(MultipathTap(
                    delay_us=delay_us,
                    amplitude=amplitude,
                    phase_deg=phase
                ))
            return taps
        
        # Default: generate based on num_paths and max_delay
        return self._generate_random_taps()
    
    def _generate_random_taps(self) -> List[MultipathTap]:
        """Generate random multipath taps."""
        taps = []
        
        # First tap is always at 0 delay with strongest power
        taps.append(MultipathTap(
            delay_us=0.0,
            amplitude=1.0,
            phase_deg=0.0
        ))
        
        # Generate remaining taps with exponentially decaying power
        for i in range(1, self.config.num_paths):
            delay = (i / self.config.num_paths) * self.config.max_delay_us
            # Exponential decay
            power_decay_db = -3.0 * i - np.random.uniform(0, 5)
            amplitude = 10 ** (power_decay_db / 20)
            phase = np.random.uniform(0, 360)
            
            taps.append(MultipathTap(
                delay_us=delay,
                amplitude=amplitude,
                phase_deg=phase
            ))
        
        return taps
    
    def _compute_filter_coefficients(self) -> np.ndarray:
        """Compute FIR filter coefficients for multipath."""
        if not self.taps:
            return np.array([1.0])
        
        # Find maximum delay in samples
        max_delay_samples = int(
            self.config.max_delay_us * 1e-6 * self.sample_rate
        ) + 1
        max_delay_samples = max(max_delay_samples, 1)
        
        # Create filter
        h = np.zeros(max_delay_samples, dtype=np.complex128)
        
        for tap in self.taps:
            delay_samples = int(tap.delay_us * 1e-6 * self.sample_rate)
            if 0 <= delay_samples < len(h):
                phase_rad = np.deg2rad(tap.phase_deg)
                h[delay_samples] += tap.amplitude * np.exp(1j * phase_rad)
        
        # Normalize to preserve power
        h = h / np.sqrt(np.sum(np.abs(h) ** 2))
        
        return h
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply multipath channel to signal."""
        if len(self.filter_coeffs) <= 1:
            return signal * self.filter_coeffs[0]
        
        # Convolve signal with channel impulse response
        output = np.convolve(signal, self.filter_coeffs, mode='same')
        
        return output
    
    def get_delay_spread(self) -> float:
        """Calculate RMS delay spread in microseconds."""
        if len(self.taps) <= 1:
            return 0.0
        
        powers = np.array([tap.amplitude ** 2 for tap in self.taps])
        delays = np.array([tap.delay_us for tap in self.taps])
        
        # Normalize powers
        powers = powers / np.sum(powers)
        
        # Mean delay
        mean_delay = np.sum(powers * delays)
        
        # RMS delay spread
        rms_delay = np.sqrt(np.sum(powers * (delays - mean_delay) ** 2))
        
        return rms_delay
    
    def get_impulse_response(self) -> np.ndarray:
        """Get channel impulse response."""
        return self.filter_coeffs.copy()


# ============================================================================
# FADING CHANNEL
# ============================================================================


class FadingChannel:
    """
    Fading channel model.
    
    Implements Rayleigh, Rician, and Nakagami fading.
    """
    
    def __init__(self, config: ChannelConfig, rng: np.random.Generator):
        """Initialize fading channel."""
        self.config = config
        self.rng = rng
        self.sample_rate = config.sample_rate_hz
        
        # Calculate coherence samples
        coherence_time_s = config.coherence_time_ms * 1e-3
        self.coherence_samples = max(1, int(coherence_time_s * self.sample_rate))
    
    def apply(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fading to signal.
        
        Returns:
            Tuple of (faded_signal, fading_coefficients)
        """
        num_samples = len(signal)
        
        if self.config.fading_type == FadingType.NONE:
            return signal, np.ones(num_samples)
        
        # Generate fading coefficients
        if self.config.fading_type == FadingType.RAYLEIGH:
            fading = self._generate_rayleigh(num_samples)
        elif self.config.fading_type == FadingType.RICIAN:
            fading = self._generate_rician(num_samples)
        elif self.config.fading_type == FadingType.NAKAGAMI:
            fading = self._generate_nakagami(num_samples)
        elif self.config.fading_type == FadingType.LOGNORMAL:
            fading = self._generate_lognormal(num_samples)
        else:
            fading = np.ones(num_samples)
        
        # Apply fading
        faded_signal = signal * fading
        
        return faded_signal, fading
    
    def _generate_rayleigh(self, num_samples: int) -> np.ndarray:
        """Generate Rayleigh fading coefficients."""
        # Jakes' model approximation
        num_blocks = (num_samples + self.coherence_samples - 1) // self.coherence_samples
        
        # Generate complex Gaussian samples
        block_fading = (
            self.rng.standard_normal(num_blocks) + 
            1j * self.rng.standard_normal(num_blocks)
        ) / np.sqrt(2)
        
        # Interpolate to sample rate
        if num_blocks > 1:
            x_blocks = np.linspace(0, 1, num_blocks)
            x_samples = np.linspace(0, 1, num_samples)
            
            real_interp = interp1d(x_blocks, np.real(block_fading), kind='cubic')
            imag_interp = interp1d(x_blocks, np.imag(block_fading), kind='cubic')
            
            fading = real_interp(x_samples) + 1j * imag_interp(x_samples)
        else:
            fading = np.full(num_samples, block_fading[0])
        
        return fading
    
    def _generate_rician(self, num_samples: int) -> np.ndarray:
        """Generate Rician fading coefficients."""
        k_linear = 10 ** (self.config.rician_k_factor / 10)
        
        # LOS component
        los = np.sqrt(k_linear / (k_linear + 1))
        
        # Scattered component (Rayleigh)
        scattered_scale = 1 / np.sqrt(k_linear + 1)
        rayleigh = self._generate_rayleigh(num_samples)
        
        fading = los + scattered_scale * rayleigh
        
        return fading
    
    def _generate_nakagami(self, num_samples: int) -> np.ndarray:
        """Generate Nakagami-m fading coefficients."""
        m = max(0.5, self.config.nakagami_m)
        
        # Nakagami can be generated from chi distribution
        num_blocks = (num_samples + self.coherence_samples - 1) // self.coherence_samples
        
        # Chi distribution with 2m degrees of freedom
        block_fading = np.sqrt(self.rng.gamma(m, 1/m, num_blocks))
        
        # Add random phase
        phases = self.rng.uniform(0, 2 * np.pi, num_blocks)
        block_fading = block_fading * np.exp(1j * phases)
        
        # Interpolate
        if num_blocks > 1:
            x_blocks = np.linspace(0, 1, num_blocks)
            x_samples = np.linspace(0, 1, num_samples)
            
            mag_interp = interp1d(x_blocks, np.abs(block_fading), kind='cubic')
            phase_interp = interp1d(x_blocks, np.unwrap(np.angle(block_fading)), kind='cubic')
            
            fading = mag_interp(x_samples) * np.exp(1j * phase_interp(x_samples))
        else:
            fading = np.full(num_samples, block_fading[0])
        
        return fading
    
    def _generate_lognormal(self, num_samples: int) -> np.ndarray:
        """Generate log-normal (shadowing) fading."""
        std_linear = self.config.shadowing_std_db / 8.686  # Convert dB to natural log scale
        
        num_blocks = (num_samples + self.coherence_samples - 1) // self.coherence_samples
        
        # Log-normal samples
        block_fading = np.exp(self.rng.normal(0, std_linear, num_blocks))
        
        # Interpolate
        if num_blocks > 1:
            x_blocks = np.linspace(0, 1, num_blocks)
            x_samples = np.linspace(0, 1, num_samples)
            
            interp_func = interp1d(x_blocks, block_fading, kind='cubic')
            fading = interp_func(x_samples)
        else:
            fading = np.full(num_samples, block_fading[0])
        
        return fading.astype(np.complex128)


# ============================================================================
# DOPPLER SIMULATOR
# ============================================================================


class DopplerSimulator:
    """
    Doppler shift simulator.
    
    Models frequency shift and spreading due to motion.
    """
    
    def __init__(self, config: ChannelConfig, rng: np.random.Generator):
        """Initialize Doppler simulator."""
        self.config = config
        self.rng = rng
        self.sample_rate = config.sample_rate_hz
        self.doppler_hz = config.get_doppler_from_velocity()
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply Doppler shift to signal."""
        if self.doppler_hz == 0:
            return signal
        
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        # Simple frequency shift
        # For more realistic simulation, could add time-varying Doppler
        doppler_shift = np.exp(2j * np.pi * self.doppler_hz * t)
        
        return signal * doppler_shift
    
    def apply_jakes_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Jakes' Doppler spectrum (classical model).
        
        This creates the characteristic U-shaped spectrum.
        """
        if self.doppler_hz == 0:
            return signal
        
        num_samples = len(signal)
        
        # Generate Doppler-filtered noise
        # Number of sinusoids for Jakes' model
        N = 8  # Number of oscillators
        
        doppler_process = np.zeros(num_samples, dtype=np.complex128)
        t = np.arange(num_samples) / self.sample_rate
        
        for n in range(N):
            # Arrival angle
            alpha_n = 2 * np.pi * n / N + self.rng.uniform(0, np.pi/N)
            # Doppler frequency for this ray
            f_n = self.doppler_hz * np.cos(alpha_n)
            # Random phase
            phi_n = self.rng.uniform(0, 2 * np.pi)
            
            doppler_process += np.exp(1j * (2 * np.pi * f_n * t + phi_n))
        
        doppler_process /= np.sqrt(N)
        
        return signal * doppler_process


# ============================================================================
# NOISE GENERATOR
# ============================================================================


class NoiseGenerator:
    """
    Noise generator for various noise types.
    """
    
    def __init__(self, config: ChannelConfig, rng: np.random.Generator):
        """Initialize noise generator."""
        self.config = config
        self.rng = rng
    
    def generate_awgn(
        self,
        signal: np.ndarray,
        snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and add AWGN noise.
        
        Returns:
            Tuple of (noisy_signal, noise_samples)
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(signal) ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex
        noise = noise_std * (
            self.rng.standard_normal(len(signal)) +
            1j * self.rng.standard_normal(len(signal))
        )
        
        return signal + noise, noise
    
    def generate_colored_noise(
        self,
        signal: np.ndarray,
        snr_db: float,
        noise_bandwidth_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate colored (filtered) noise.
        """
        # Generate white noise first
        _, white_noise = self.generate_awgn(signal, snr_db)
        
        # Design lowpass filter for coloring
        cutoff = noise_bandwidth_ratio
        b, a = scipy_signal.butter(4, cutoff, btype='low')
        
        # Apply filter to real and imaginary parts
        colored_noise = (
            scipy_signal.filtfilt(b, a, np.real(white_noise)) +
            1j * scipy_signal.filtfilt(b, a, np.imag(white_noise))
        )
        
        # Adjust power to match target SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear
        current_noise_power = np.mean(np.abs(colored_noise) ** 2)
        
        if current_noise_power > 0:
            colored_noise *= np.sqrt(target_noise_power / current_noise_power)
        
        return signal + colored_noise, colored_noise
    
    def generate_impulse_noise(
        self,
        signal: np.ndarray,
        snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and add impulse noise.
        """
        # Start with AWGN
        noisy_signal, awgn = self.generate_awgn(signal, snr_db)
        
        # Add impulse noise
        num_samples = len(signal)
        impulse_mask = self.rng.random(num_samples) < self.config.impulse_rate
        
        # Signal RMS for impulse amplitude
        signal_rms = np.sqrt(np.mean(np.abs(signal) ** 2))
        impulse_amp = signal_rms * self.config.impulse_amplitude
        
        # Random impulse values
        impulses = np.zeros(num_samples, dtype=np.complex128)
        num_impulses = np.sum(impulse_mask)
        
        if num_impulses > 0:
            impulses[impulse_mask] = impulse_amp * (
                self.rng.standard_normal(num_impulses) +
                1j * self.rng.standard_normal(num_impulses)
            )
        
        combined_noise = awgn + impulses
        
        return signal + combined_noise, combined_noise
    
    def generate_phase_noise(
        self,
        signal: np.ndarray,
        phase_noise_std_deg: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and add phase noise.
        """
        num_samples = len(signal)
        
        # Generate phase noise (correlated)
        phase_noise_std_rad = np.deg2rad(phase_noise_std_deg)
        phase_innovations = self.rng.normal(0, phase_noise_std_rad, num_samples)
        
        # Random walk for phase
        phase_noise = np.cumsum(phase_innovations)
        
        # Limit phase drift
        phase_noise = phase_noise - np.mean(phase_noise)
        
        # Apply phase noise
        phase_modulation = np.exp(1j * phase_noise)
        noisy_signal = signal * phase_modulation
        
        return noisy_signal, phase_noise


# ============================================================================
# PATH LOSS CALCULATOR
# ============================================================================


class PathLossCalculator:
    """
    Calculate path loss for various propagation models.
    """
    
    # Speed of light
    C = 3e8
    
    def __init__(self, config: ChannelConfig):
        """Initialize path loss calculator."""
        self.config = config
        self.carrier_freq_hz = config.carrier_freq_mhz * 1e6
    
    def free_space_loss_db(self, distance_km: float) -> float:
        """Calculate free-space path loss (Friis equation)."""
        if distance_km <= 0:
            return 0.0
        
        distance_m = distance_km * 1000
        wavelength = self.C / self.carrier_freq_hz
        
        fspl = (4 * np.pi * distance_m / wavelength) ** 2
        return 10 * np.log10(fspl)
    
    def log_distance_loss_db(
        self,
        distance_km: float,
        reference_distance_km: float = 0.001,  # 1 meter
        reference_loss_db: Optional[float] = None
    ) -> float:
        """
        Calculate log-distance path loss.
        
        PL(d) = PL(d0) + 10 * n * log10(d/d0)
        """
        if distance_km <= 0:
            return 0.0
        
        if reference_loss_db is None:
            reference_loss_db = self.free_space_loss_db(reference_distance_km)
        
        if distance_km <= reference_distance_km:
            return reference_loss_db
        
        path_loss = reference_loss_db + 10 * self.config.path_loss_exponent * np.log10(
            distance_km / reference_distance_km
        )
        
        return path_loss
    
    def cost231_hata_db(
        self,
        distance_km: float,
        base_height_m: float = 30.0,
        mobile_height_m: float = 1.5,
        urban: bool = True
    ) -> float:
        """
        COST-231 Hata model for path loss.
        
        Valid for:
        - Frequency: 1500-2000 MHz
        - Distance: 1-20 km
        """
        f_mhz = self.config.carrier_freq_mhz
        d_km = max(1.0, min(distance_km, 20.0))  # Clamp to valid range
        hb = base_height_m
        hm = mobile_height_m
        
        # Mobile antenna height correction
        if urban:
            a_hm = 3.2 * (np.log10(11.75 * hm)) ** 2 - 4.97
        else:
            a_hm = (1.1 * np.log10(f_mhz) - 0.7) * hm - (1.56 * np.log10(f_mhz) - 0.8)
        
        # Urban correction
        C_m = 3.0 if urban else 0.0
        
        # Path loss
        L = (46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(hb) -
             a_hm + (44.9 - 6.55 * np.log10(hb)) * np.log10(d_km) + C_m)
        
        return L
    
    def apply_path_loss(
        self,
        signal: np.ndarray,
        path_loss_db: Optional[float] = None
    ) -> np.ndarray:
        """Apply path loss attenuation to signal."""
        if path_loss_db is None:
            if self.config.path_loss_db > 0:
                path_loss_db = self.config.path_loss_db
            elif self.config.distance_km > 0:
                path_loss_db = self.log_distance_loss_db(self.config.distance_km)
            else:
                return signal
        
        # Convert dB to linear
        attenuation = 10 ** (-path_loss_db / 20)
        
        return signal * attenuation


# ============================================================================
# MAIN CHANNEL SIMULATOR
# ============================================================================


class ChannelSimulator:
    """
    Complete channel simulator combining all impairment models.
    """
    
    # Environment presets
    ENVIRONMENT_PRESETS = {
        EnvironmentType.RURAL: ChannelConfig(
            snr_db=25.0,
            noise_enabled=True,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.FLAT,
            num_paths=2,
            max_delay_us=1.0,
            fading_enabled=False,
            doppler_enabled=False,
            path_loss_enabled=True,
            path_loss_exponent=2.5
        ),
        EnvironmentType.SUBURBAN: ChannelConfig(
            snr_db=20.0,
            noise_enabled=True,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.PEDESTRIAN_A,
            num_paths=4,
            max_delay_us=5.0,
            fading_enabled=True,
            fading_type=FadingType.RICIAN,
            rician_k_factor=6.0,
            doppler_enabled=False,
            path_loss_enabled=True,
            path_loss_exponent=3.0
        ),
        EnvironmentType.URBAN: ChannelConfig(
            snr_db=15.0,
            noise_enabled=True,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.VEHICULAR_A,
            num_paths=6,
            max_delay_us=10.0,
            fading_enabled=True,
            fading_type=FadingType.RAYLEIGH,
            doppler_enabled=False,
            path_loss_enabled=True,
            path_loss_exponent=3.5
        ),
        EnvironmentType.URBAN_CANYON: ChannelConfig(
            snr_db=10.0,
            noise_enabled=True,
            noise_type=NoiseType.AWGN,
            impulse_rate=0.001,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.VEHICULAR_B,
            num_paths=8,
            max_delay_us=20.0,
            fading_enabled=True,
            fading_type=FadingType.RAYLEIGH,
            coherence_time_ms=5.0,
            doppler_enabled=True,
            doppler_hz=20.0,
            path_loss_enabled=True,
            path_loss_exponent=4.0,
            shadowing_enabled=True,
            shadowing_std_db=10.0
        ),
        EnvironmentType.HIGHWAY: ChannelConfig(
            snr_db=18.0,
            noise_enabled=True,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.VEHICULAR_A,
            num_paths=5,
            max_delay_us=8.0,
            fading_enabled=True,
            fading_type=FadingType.RICIAN,
            rician_k_factor=3.0,
            doppler_enabled=True,
            velocity_kmh=120.0,
            carrier_freq_mhz=600.0,
            path_loss_enabled=True,
            path_loss_exponent=3.2
        ),
        EnvironmentType.INDOOR: ChannelConfig(
            snr_db=22.0,
            noise_enabled=True,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.INDOOR_OFFICE,
            num_paths=5,
            max_delay_us=0.5,
            fading_enabled=True,
            fading_type=FadingType.RAYLEIGH,
            coherence_time_ms=50.0,
            doppler_enabled=False,
            path_loss_enabled=True,
            path_loss_exponent=3.0,
            shadowing_enabled=True,
            shadowing_std_db=6.0
        ),
        EnvironmentType.MOUNTAINOUS: ChannelConfig(
            snr_db=16.0,
            noise_enabled=True,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.VEHICULAR_B,
            num_paths=6,
            max_delay_us=15.0,
            fading_enabled=True,
            fading_type=FadingType.LOGNORMAL,
            shadowing_std_db=12.0,
            doppler_enabled=False,
            path_loss_enabled=True,
            path_loss_exponent=4.0
        ),
        EnvironmentType.MARITIME: ChannelConfig(
            snr_db=20.0,
            noise_enabled=True,
            multipath_enabled=True,
            multipath_profile=MultipathProfile.FLAT,
            num_paths=2,
            max_delay_us=3.0,
            fading_enabled=True,
            fading_type=FadingType.RICIAN,
            rician_k_factor=10.0,
            doppler_enabled=True,
            velocity_kmh=30.0,
            path_loss_enabled=True,
            path_loss_exponent=2.0
        )
    }
    
    def __init__(
        self,
        config: Optional[ChannelConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize channel simulator.
        
        Args:
            config: Channel configuration (uses default if None)
            seed: Random seed for reproducibility
        """
        self.config = config or ChannelConfig()
        
        # Use seed from config if not provided
        actual_seed = seed if seed is not None else self.config.seed
        self.rng = np.random.default_rng(actual_seed)
        
        # Initialize components
        self.multipath = MultipathChannel(self.config) if self.config.multipath_enabled else None
        self.fading = FadingChannel(self.config, self.rng) if self.config.fading_enabled else None
        self.doppler = DopplerSimulator(self.config, self.rng) if self.config.doppler_enabled else None
        self.noise_gen = NoiseGenerator(self.config, self.rng)
        self.path_loss_calc = PathLossCalculator(self.config)
        
        logger.info(f"ChannelSimulator initialized with SNR={self.config.snr_db}dB")
    
    @classmethod
    def from_preset(
        cls,
        environment: Union[EnvironmentType, str],
        snr_override: Optional[float] = None,
        seed: Optional[int] = None
    ) -> 'ChannelSimulator':
        """
        Create channel simulator from environment preset.
        
        Args:
            environment: Environment type or name
            snr_override: Override SNR value
            seed: Random seed
        
        Returns:
            Configured ChannelSimulator
        """
        if isinstance(environment, str):
            environment = EnvironmentType[environment.upper()]
        
        config = cls.ENVIRONMENT_PRESETS.get(environment, ChannelConfig())
        
        # Make a copy to avoid modifying preset
        config = ChannelConfig(
            snr_db=snr_override if snr_override is not None else config.snr_db,
            noise_enabled=config.noise_enabled,
            noise_type=config.noise_type,
            impulse_rate=config.impulse_rate,
            impulse_amplitude=config.impulse_amplitude,
            multipath_enabled=config.multipath_enabled,
            multipath_profile=config.multipath_profile,
            num_paths=config.num_paths,
            max_delay_us=config.max_delay_us,
            fading_enabled=config.fading_enabled,
            fading_type=config.fading_type,
            rician_k_factor=config.rician_k_factor,
            coherence_time_ms=config.coherence_time_ms,
            doppler_enabled=config.doppler_enabled,
            doppler_hz=config.doppler_hz,
            velocity_kmh=config.velocity_kmh,
            carrier_freq_mhz=config.carrier_freq_mhz,
            path_loss_enabled=config.path_loss_enabled,
            path_loss_db=config.path_loss_db,
            path_loss_exponent=config.path_loss_exponent,
            shadowing_enabled=config.shadowing_enabled,
            shadowing_std_db=config.shadowing_std_db,
            sample_rate_hz=config.sample_rate_hz,
            seed=seed
        )
        
        return cls(config, seed)
    
    @classmethod
    def from_scenario_profile(
        cls,
        profile: Dict[str, Any],
        seed: Optional[int] = None
    ) -> 'ChannelSimulator':
        """
        Create channel simulator from scenario profile.
        
        Maps scenario environment settings to channel configuration.
        """
        env = profile.get("environment", {})
        
        # Map scenario environment to channel config
        sky_view = env.get("sky_view", "open")
        multipath = env.get("multipath", "low")
        signal_quality = env.get("signal_quality", "good")
        
        # Select base preset based on sky view
        if sky_view == "open":
            base_env = EnvironmentType.RURAL
        elif sky_view == "partial":
            base_env = EnvironmentType.SUBURBAN
        else:
            base_env = EnvironmentType.URBAN
        
        # Get base config
        config = cls.ENVIRONMENT_PRESETS.get(base_env, ChannelConfig())
        
        # Adjust based on multipath level
        if multipath == "severe":
            config.num_paths = 8
            config.max_delay_us = 20.0
            config.multipath_profile = MultipathProfile.VEHICULAR_B
        elif multipath == "moderate":
            config.num_paths = 5
            config.max_delay_us = 10.0
        elif multipath == "low":
            config.num_paths = 2
            config.max_delay_us = 2.0
        
        # Adjust SNR based on signal quality
        snr_map = {
            "excellent": 25.0,
            "good": 20.0,
            "moderate": 15.0,
            "degraded": 10.0,
            "poor": 5.0
        }
        config.snr_db = snr_map.get(signal_quality, 15.0)
        
        # Apply SNR margin if specified
        snr_margin = env.get("snr_margin_db", 0)
        config.snr_db += snr_margin
        
        # Enable Doppler for vehicle scenarios
        dynamics = env.get("dynamics", "low")
        if dynamics == "high":
            config.doppler_enabled = True
            config.velocity_kmh = 100.0
        elif dynamics == "medium":
            config.doppler_enabled = True
            config.velocity_kmh = 50.0
        
        config.seed = seed
        
        return cls(config, seed)
    
    def apply(
        self,
        signal: np.ndarray,
        return_metrics: bool = True
    ) -> Union[np.ndarray, ChannelSimulatorResult]:
        """
        Apply all channel effects to signal.
        
        Args:
            signal: Input signal (complex samples)
            return_metrics: If True, return full result with metrics
        
        Returns:
            Degraded signal or ChannelSimulatorResult
        """
        start_time = time.time()
        metrics = ChannelMetrics()
        
        # Ensure complex type
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)
        
        # Record input power
        input_power = np.mean(np.abs(signal) ** 2)
        metrics.input_power_db = 10 * np.log10(input_power + 1e-12)
        
        output = signal.copy()
        channel_ir = None
        fading_coeffs = None
        noise_samples = None
        
        # ========================================
        # 1. Path Loss
        # ========================================
        if self.config.path_loss_enabled:
            output = self.path_loss_calc.apply_path_loss(output)
            metrics.effects_applied.append("path_loss")
        
        # ========================================
        # 2. Multipath
        # ========================================
        if self.config.multipath_enabled and self.multipath:
            output = self.multipath.apply(output)
            channel_ir = self.multipath.get_impulse_response()
            metrics.delay_spread_us = self.multipath.get_delay_spread()
            metrics.num_significant_paths = len(self.multipath.taps)
            metrics.effects_applied.append("multipath")
        
        # ========================================
        # 3. Fading
        # ========================================
        if self.config.fading_enabled and self.fading:
            output, fading_coeffs = self.fading.apply(output)
            metrics.fading_margin_db = -10 * np.log10(
                np.min(np.abs(fading_coeffs) ** 2) + 1e-12
            )
            metrics.effects_applied.append(f"fading_{self.config.fading_type.name.lower()}")
        
        # ========================================
        # 4. Doppler
        # ========================================
        if self.config.doppler_enabled and self.doppler:
            output = self.doppler.apply(output)
            metrics.doppler_spread_hz = self.config.get_doppler_from_velocity()
            metrics.effects_applied.append("doppler")
        
        # ========================================
        # 5. Shadowing (slow fading)
        # ========================================
        if self.config.shadowing_enabled:
            # Apply log-normal shadowing
            shadow_fading = FadingChannel(
                ChannelConfig(
                    fading_type=FadingType.LOGNORMAL,
                    shadowing_std_db=self.config.shadowing_std_db,
                    coherence_time_ms=100.0,  # Slow fading
                    sample_rate_hz=self.config.sample_rate_hz
                ),
                self.rng
            )
            output, _ = shadow_fading.apply(output)
            metrics.effects_applied.append("shadowing")
        
        # ========================================
        # 6. Noise
        # ========================================
        if self.config.noise_enabled:
            noise_type = self.config.noise_type
            
            if noise_type == NoiseType.AWGN:
                output, noise_samples = self.noise_gen.generate_awgn(
                    output, self.config.snr_db
                )
            elif noise_type == NoiseType.COLORED:
                output, noise_samples = self.noise_gen.generate_colored_noise(
                    output, self.config.snr_db
                )
            elif noise_type == NoiseType.IMPULSE:
                output, noise_samples = self.noise_gen.generate_impulse_noise(
                    output, self.config.snr_db
                )
            elif noise_type == NoiseType.PHASE:
                output, noise_samples = self.noise_gen.generate_phase_noise(
                    output
                )
            
            metrics.effects_applied.append(f"noise_{noise_type.name.lower()}")
            metrics.effective_snr_db = self.config.snr_db
        
        # ========================================
        # Calculate output metrics
        # ========================================
        output_power = np.mean(np.abs(output) ** 2)
        metrics.output_power_db = 10 * np.log10(output_power + 1e-12)
        
        # Estimate BER from SNR (rough approximation for QPSK)
        if self.config.noise_enabled and self.config.snr_db > 0:
            snr_linear = 10 ** (self.config.snr_db / 10)
            # QPSK BER approximation
            metrics.estimated_ber = 0.5 * special.erfc(np.sqrt(snr_linear))
            # Packet error rate for 1000-byte packet
            metrics.estimated_packet_error_rate = 1 - (1 - metrics.estimated_ber) ** 8000
        
        metrics.processing_time_ms = (time.time() - start_time) * 1000
        
        if return_metrics:
            return ChannelSimulatorResult(
                signal=output,
                metrics=metrics,
                config_used=self.config,
                channel_impulse_response=channel_ir,
                fading_coefficients=fading_coeffs,
                noise_samples=noise_samples
            )
        else:
            return output
    
    def sweep_snr(
        self,
        signal: np.ndarray,
        snr_range: List[float],
        metric_callback: Optional[callable] = None
    ) -> Dict[float, ChannelMetrics]:
        """
        Apply channel at multiple SNR values.
        
        Useful for generating SNR vs BER curves.
        
        Args:
            signal: Input signal
            snr_range: List of SNR values to test
            metric_callback: Optional callback for each result
        
        Returns:
            Dictionary mapping SNR to metrics
        """
        results = {}
        
        for snr in snr_range:
            # Create new config with different SNR
            test_config = ChannelConfig(
                snr_db=snr,
                noise_enabled=self.config.noise_enabled,
                noise_type=self.config.noise_type,
                multipath_enabled=self.config.multipath_enabled,
                multipath_profile=self.config.multipath_profile,
                num_paths=self.config.num_paths,
                max_delay_us=self.config.max_delay_us,
                fading_enabled=self.config.fading_enabled,
                fading_type=self.config.fading_type,
                doppler_enabled=self.config.doppler_enabled,
                doppler_hz=self.config.doppler_hz,
                sample_rate_hz=self.config.sample_rate_hz,
                seed=self.config.seed
            )
            
            test_simulator = ChannelSimulator(test_config)
            result = test_simulator.apply(signal, return_metrics=True)
            results[snr] = result.metrics
            
            if metric_callback:
                metric_callback(snr, result)
        
        return results
    
    def get_channel_response(self, num_points: int = 1024) -> np.ndarray:
        """
        Get frequency response of multipath channel.
        
        Args:
            num_points: Number of frequency points
        
        Returns:
            Complex frequency response
        """
        if self.multipath:
            h = self.multipath.get_impulse_response()
            H = np.fft.fft(h, n=num_points)
            return H
        else:
            return np.ones(num_points, dtype=np.complex128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current simulator status."""
        return {
            "snr_db": self.config.snr_db,
            "effects_enabled": {
                "noise": self.config.noise_enabled,
                "multipath": self.config.multipath_enabled,
                "fading": self.config.fading_enabled,
                "doppler": self.config.doppler_enabled,
                "path_loss": self.config.path_loss_enabled,
                "shadowing": self.config.shadowing_enabled
            },
            "multipath": {
                "profile": self.config.multipath_profile.name,
                "num_paths": self.config.num_paths,
                "max_delay_us": self.config.max_delay_us
            },
            "fading": {
                "type": self.config.fading_type.name,
                "rician_k": self.config.rician_k_factor
            },
            "doppler_hz": self.config.get_doppler_from_velocity()
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def apply_awgn(
    signal: np.ndarray,
    snr_db: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Quick function to add AWGN to signal.
    
    Args:
        signal: Input signal
        snr_db: Desired SNR in dB
        seed: Random seed
    
    Returns:
        Noisy signal
    """
    config = ChannelConfig(
        snr_db=snr_db,
        noise_enabled=True,
        multipath_enabled=False,
        fading_enabled=False,
        doppler_enabled=False,
        path_loss_enabled=False,
        seed=seed
    )
    
    simulator = ChannelSimulator(config)
    return simulator.apply(signal, return_metrics=False)


def apply_multipath(
    signal: np.ndarray,
    num_paths: int = 4,
    max_delay_us: float = 10.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Quick function to apply multipath to signal.
    
    Args:
        signal: Input signal
        num_paths: Number of multipath taps
        max_delay_us: Maximum delay spread
        seed: Random seed
    
    Returns:
        Signal with multipath
    """
    config = ChannelConfig(
        noise_enabled=False,
        multipath_enabled=True,
        multipath_profile=MultipathProfile.CUSTOM,
        num_paths=num_paths,
        max_delay_us=max_delay_us,
        fading_enabled=False,
        doppler_enabled=False,
        path_loss_enabled=False,
        seed=seed
    )
    
    simulator = ChannelSimulator(config)
    return simulator.apply(signal, return_metrics=False)


def simulate_environment(
    signal: np.ndarray,
    environment: str,
    snr_db: Optional[float] = None,
    seed: Optional[int] = None
) -> ChannelSimulatorResult:
    """
    Apply environment-specific channel model.
    
    Args:
        signal: Input signal
        environment: Environment name (rural, suburban, urban, etc.)
        snr_db: Optional SNR override
        seed: Random seed
    
    Returns:
        ChannelSimulatorResult with degraded signal and metrics
    """
    simulator = ChannelSimulator.from_preset(environment, snr_db, seed)
    return simulator.apply(signal, return_metrics=True)


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Command-line interface for channel simulator."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="ATSC 3.0 Channel Simulator - Test signal degradation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ========================================
    # Simulate command
    # ========================================
    sim_parser = subparsers.add_parser("simulate", help="Apply channel to signal")
    sim_parser.add_argument(
        "input_file",
        help="Input signal file (complex64 binary)"
    )
    sim_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file for degraded signal"
    )
    sim_parser.add_argument(
        "--environment",
        choices=[e.name.lower() for e in EnvironmentType],
        default="suburban",
        help="Environment preset"
    )
    sim_parser.add_argument(
        "--snr",
        type=float,
        help="SNR in dB (overrides preset)"
    )
    sim_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    sim_parser.add_argument(
        "--metrics-out",
        help="Output file for metrics JSON"
    )
    
    # ========================================
    # Sweep command
    # ========================================
    sweep_parser = subparsers.add_parser("sweep", help="Sweep SNR range")
    sweep_parser.add_argument(
        "input_file",
        help="Input signal file"
    )
    sweep_parser.add_argument(
        "--snr-min",
        type=float,
        default=0.0,
        help="Minimum SNR"
    )
    sweep_parser.add_argument(
        "--snr-max",
        type=float,
        default=30.0,
        help="Maximum SNR"
    )
    sweep_parser.add_argument(
        "--snr-step",
        type=float,
        default=2.0,
        help="SNR step"
    )
    sweep_parser.add_argument(
        "--environment",
        choices=[e.name.lower() for e in EnvironmentType],
        default="suburban",
        help="Environment preset"
    )
    sweep_parser.add_argument(
        "-o", "--output",
        help="Output CSV file for results"
    )
    
    # ========================================
    # Presets command
    # ========================================
    presets_parser = subparsers.add_parser("presets", help="Show environment presets")
    presets_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    # ========================================
    # Demo command
    # ========================================
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    demo_parser.add_argument(
        "--scenario",
        choices=["all_environments", "snr_sweep", "multipath_profiles"],
        default="all_environments",
        help="Demo scenario"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    if args.command == "simulate":
        # Read input signal
        signal = np.fromfile(args.input_file, dtype=np.complex64)
        print(f"Read {len(signal)} samples from {args.input_file}")
        
        # Create simulator
        simulator = ChannelSimulator.from_preset(
            args.environment,
            args.snr,
            args.seed
        )
        
        # Apply channel
        result = simulator.apply(signal, return_metrics=True)
        
        # Save output
        result.signal.astype(np.complex64).tofile(args.output)
        print(f"Saved degraded signal to {args.output}")
        
        # Print metrics
        print("\nChannel Effects Applied:")
        for effect in result.metrics.effects_applied:
            print(f"  - {effect}")
        
        print(f"\nSignal Quality:")
        print(f"  Input power:  {result.metrics.input_power_db:.1f} dB")
        print(f"  Output power: {result.metrics.output_power_db:.1f} dB")
        print(f"  Effective SNR: {result.metrics.effective_snr_db:.1f} dB")
        print(f"  Estimated BER: {result.metrics.estimated_ber:.2e}")
        
        if result.metrics.delay_spread_us > 0:
            print(f"\nMultipath:")
            print(f"  RMS delay spread: {result.metrics.delay_spread_us:.2f} µs")
            print(f"  Number of paths: {result.metrics.num_significant_paths}")
        
        # Save metrics
        if args.metrics_out:
            with open(args.metrics_out, "w") as f:
                json.dump(result.metrics.to_dict(), f, indent=2)
            print(f"\nMetrics saved to {args.metrics_out}")
    
    elif args.command == "sweep":
        # Read input signal
        signal = np.fromfile(args.input_file, dtype=np.complex64)
        print(f"Read {len(signal)} samples")
        
        # Create SNR range
        snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)
        
        # Create simulator
        simulator = ChannelSimulator.from_preset(args.environment)
        
        print(f"Sweeping SNR from {args.snr_min} to {args.snr_max} dB...")
        print("-" * 50)
        print(f"{'SNR (dB)':<10} {'BER':<12} {'PER':<12}")
        print("-" * 50)
        
        results = []
        for snr in snr_range:
            metrics = simulator.sweep_snr(signal, [snr])[snr]
            print(f"{snr:<10.1f} {metrics.estimated_ber:<12.2e} {metrics.estimated_packet_error_rate:<12.2e}")
            results.append({
                "snr_db": snr,
                "ber": metrics.estimated_ber,
                "per": metrics.estimated_packet_error_rate
            })
        
        if args.output:
            import csv
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["snr_db", "ber", "per"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResults saved to {args.output}")
    
    elif args.command == "presets":
        presets_info = {}
        
        for env_type, config in ChannelSimulator.ENVIRONMENT_PRESETS.items():
            presets_info[env_type.name] = {
                "snr_db": config.snr_db,
                "multipath": config.multipath_profile.name if config.multipath_enabled else "disabled",
                "num_paths": config.num_paths,
                "fading": config.fading_type.name if config.fading_enabled else "disabled",
                "doppler": config.doppler_enabled,
                "path_loss_exponent": config.path_loss_exponent
            }
        
        if args.format == "json":
            print(json.dumps(presets_info, indent=2))
        else:
            print("Environment Presets")
            print("=" * 70)
            for name, info in presets_info.items():
                print(f"\n{name}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
    
    elif args.command == "demo":
        # Generate test signal
        print("Generating test OFDM-like signal...")
        num_samples = 100000
        t = np.arange(num_samples) / 6.144e6
        
        # Multi-carrier signal (simplified OFDM)
        signal = np.zeros(num_samples, dtype=np.complex128)
        for f in range(100, 3000, 100):
            phase = np.random.uniform(0, 2 * np.pi)
            signal += np.exp(1j * (2 * np.pi * f * t + phase))
        signal /= np.sqrt(np.mean(np.abs(signal) ** 2))  # Normalize
        
        if args.scenario == "all_environments":
            print("\nTesting all environments:")
            print("-" * 70)
            print(f"{'Environment':<15} {'SNR':<8} {'Paths':<8} {'Delay':<10} {'BER':<12}")
            print("-" * 70)
            
            for env_type in EnvironmentType:
                simulator = ChannelSimulator.from_preset(env_type, seed=42)
                result = simulator.apply(signal)
                m = result.metrics
                
                print(f"{env_type.name:<15} {simulator.config.snr_db:<8.1f} "
                      f"{m.num_significant_paths:<8} {m.delay_spread_us:<10.2f} "
                      f"{m.estimated_ber:<12.2e}")
        
        elif args.scenario == "multipath_profiles":
            print("\nTesting multipath profiles:")
            print("-" * 60)
            print(f"{'Profile':<20} {'Paths':<10} {'Delay Spread (µs)':<20}")
            print("-" * 60)
            
            for profile in MultipathProfile:
                if profile == MultipathProfile.CUSTOM:
                    continue
                
                config = ChannelConfig(
                    noise_enabled=False,
                    multipath_enabled=True,
                    multipath_profile=profile,
                    fading_enabled=False
                )
                simulator = ChannelSimulator(config)
                result = simulator.apply(signal)
                m = result.metrics
                
                print(f"{profile.name:<20} {m.num_significant_paths:<10} {m.delay_spread_us:<20.3f}")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())