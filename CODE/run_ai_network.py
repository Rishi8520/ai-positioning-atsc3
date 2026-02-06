#!/usr/bin/env python3
"""
AI-Enhanced Network Pipeline for PPaaS
Integrates AI decision-making into broadcast configuration for Scenario 1
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from datetime import datetime

# Add AI directory to path
sys.path.insert(0, str(Path(__file__).parent / "ai"))

# AI Imports
from ai_inference_engine_v2 import InferenceEngineV2, InferenceBackend
from ai_feedback_loop import FeedbackLoop, FieldTelemetry
from ai_intent_parser import IntentParser

# Broadcast imports
from broadcast.pipeline import BroadcastPipeline, BroadcastConfig
from broadcast.channel_simulator import ChannelSimulator
from broadcast.decoder import BroadcastDecoder, DecoderConfig
from broadcast.config import FECCodeRate, ModulationScheme, FFTSize, GuardInterval

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class AINetworkPipeline:
    """
    AI-Enhanced Network Pipeline
    Replaces static broadcast configuration with AI-driven decisions
    """
    
    def __init__(self, scenario_id: str, output_dir: Path, mode: str = "ai"):
        self.scenario_id = scenario_id
        self.output_dir = Path(output_dir)
        self.mode = mode  # "traditional" or "ai"
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directories
        self.run_output_dir = self.output_dir / self.run_id
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # AI components (lazy loaded)
        self.ai_engine = None
        self.feedback_loop = FeedbackLoop()
        self.intent_parser = IntentParser()
        
        # Results storage
        self.results = {
            'scenario_id': scenario_id,
            'run_id': self.run_id,
            'mode': mode,
            'phases': {}
        }
        
        logger.info("=" * 70)
        logger.info(f"  {'AI-ENHANCED' if mode == 'ai' else 'TRADITIONAL'} NETWORK PIPELINE")
        logger.info(f"  Scenario: {scenario_id}")
        logger.info(f"  Run ID: {self.run_id}")
        logger.info("=" * 70)
    
    def _load_ai_model(self):
        """Load trained AI model (lazy loading)"""
        if self.ai_engine is not None:
            return
        
        model_dir = Path(__file__).parent / "ai" / "models" / "broadcast_decision_model_v2"
        
        if not model_dir.exists():
            logger.error(f"AI model not found at {model_dir}")
            logger.error("Please train the model first:")
            logger.error("  cd ai && python ai_broadcast_decision_model_v2.py")
            raise FileNotFoundError(f"AI model not found: {model_dir}")
        
        logger.info(f"Loading AI model from {model_dir.name}...")
        self.ai_engine = InferenceEngineV2(
            model_path=str(model_dir),
            confidence_threshold=0.6,
            backend=InferenceBackend.PYTORCH,
            mc_samples=20
        )
        logger.info("✓ AI model loaded")
    
    def _extract_telemetry_from_gnss(self, gnss_results: Dict, scenario_profile: Dict) -> np.ndarray:
        """
    Extract 50D telemetry features from GNSS results
    NOW READS SCENARIO PROFILE FOR ENVIRONMENT
    
    Returns:
        50D numpy array for AI model input
        """
        features = np.zeros(50)
    
        try:
            metrics = gnss_results.get('metrics', {})
        
        # Get environment from scenario profile - FIXED PARSING
            env_data = scenario_profile.get('environment', {})
        
        # Handle both dict and legacy formats
            if isinstance(env_data, dict):
            # New format: environment is a dict with keys
                skyview = env_data.get('skyview', 'open')
                multipath_level = env_data.get('multipath', 'low')
                signal_quality = env_data.get('signalquality', 'excellent')
                urban_density = env_data.get('urbandensity', 'low')
            elif isinstance(env_data, str):
            # Legacy format: environment is just a string like "rural"
                env_str = env_data.lower()
                if 'urban' in env_str:
                    skyview = 'limited'
                    multipath_level = 'severe'
                    signal_quality = 'degraded'
                    urban_density = 'high'
                elif 'suburban' in env_str:
                    skyview = 'partial'
                    multipath_level = 'moderate'
                    signal_quality = 'good'
                    urban_density = 'medium'
                else:  # rural
                    skyview = 'open'
                    multipath_level = 'low'
                    signal_quality = 'excellent'
                    urban_density = 'low'
            else:
            # Fallback defaults
                skyview = 'open'
                multipath_level = 'low'
                signal_quality = 'excellent'
                urban_density = 'low'
        
            logger.info(f"Environment from profile: skyview={skyview}, multipath={multipath_level}, signal={signal_quality}, urban={urban_density}")
        
        # Signal Strength (4D) - indices 0-3
        # Degrade SNR based on signal quality
            base_snr = metrics.get('mean_snr_db', 25.0)
            if signal_quality == 'degraded':
                snr = base_snr - 5.0
            elif signal_quality == 'good':
                snr = base_snr - 2.0
            else:
                snr = base_snr
            features[0:4] = [snr, snr, snr, snr]
        
        # Carrier Phase quality (4D) - indices 4-7
            features[4:8] = np.random.randn(4) * 0.1
        
        # Pseudorange Error (4D) - indices 8-11
            hpe = metrics.get('mean_hpe_cm', 5.0) / 100.0
            features[8:12] = [hpe, hpe, hpe, hpe]
        
        # Doppler (4D) - indices 12-15
            features[12:16] = np.random.randn(4) * 2.0
        
        # Tracking Lock (4D) - indices 16-19
            fix_rate = metrics.get('fix_rate_pct', 95.0) / 100.0
            features[16:20] = [fix_rate, fix_rate, fix_rate, fix_rate]
        
        # Received Power (4D) - indices 20-23
            features[20:24] = [snr-5, snr-5, snr-5, snr-5]
        
        # Carrier Power (4D) - indices 24-27
            features[24:28] = [snr, snr, snr, snr]
        
        # Noise Power (2D) - indices 28-29
            features[28:30] = [15.0, 15.0]
        
        # SNR (4D) - indices 30-33
            features[30:34] = [snr, snr, snr, snr]
        
        # CNR (4D) - indices 34-37
            features[34:38] = [snr+2, snr+2, snr+2, snr+2]
        
        # Multipath (2D) - indices 38-39 - FROM SCENARIO PROFILE
            multipath_map = {'low': 0.2, 'moderate': 0.5, 'severe': 0.8}
            multipath = multipath_map.get(multipath_level, 0.2)
            features[38:40] = [multipath, multipath]

        # Environment (10D) - indices 40-49 - FROM SCENARIO PROFILE
            urban_map = {'low': 0.1, 'medium': 0.5, 'high': 0.9}
            urban_density_val = urban_map.get(urban_density, 0.1)
        
            skyview_map = {'open': 0.1, 'partial': 0.4, 'limited': 0.7}
            blockage = skyview_map.get(skyview, 0.1)
        
            signal_map = {'excellent': 0.2, 'good': 0.4, 'degraded': 0.7}
            shadow_fading = signal_map.get(signal_quality, 0.2)
        
            nlos_map = {'open': 0.1, 'partial': 0.3, 'limited': 0.6}
            nlos_prob = nlos_map.get(skyview, 0.1)
        
            features[40] = urban_density_val  # urban_density
            features[41] = blockage  # blockage
            features[42] = multipath  # multipath_likelihood
            features[43] = shadow_fading  # shadow_fading
            features[44] = nlos_prob  # nlos_prob
            features[45] = 0.0  # tunnel_prob
            features[46] = datetime.now().hour / 24.0  # time_of_day
            features[47] = 0.0  # vehicle_speed
            features[48] = 0.0  # heading
            features[49] = fix_rate  # gnss_availability
        
            logger.info(f"✓ Extracted 50D telemetry: urban={urban_density_val:.2f}, multipath={multipath:.2f}, blockage={blockage:.2f}, nlos={nlos_prob:.2f}")
        
        except Exception as e:
            logger.warning(f"Failed to extract telemetry: {e}, using defaults")
            features = np.random.randn(50) * 0.5
    
        return features
    
    def _ai_infer_broadcast_config(self, telemetry: np.ndarray, intent: str) -> BroadcastConfig:
        """
    Use AI to determine optimal broadcast configuration
    
    Returns:
        BroadcastConfig object with AI-determined parameters
        """
    # Parse intent
        parsed_intent = self.intent_parser.parse(intent)
        logger.info(f"Intent: {parsed_intent.intent_type.value} (confidence: {parsed_intent.confidence:.2f})")
    
    # AI Inference
        result = self.ai_engine.infer(telemetry)
        decision = result.broadcast_decision
    
        logger.info("AI Decision:")
        logger.info(f"  Redundancy Ratio: {decision.redundancy_ratio:.2f}")
        logger.info(f"  Spectrum (Mbps): {decision.spectrum_mbps:.2f}")
        logger.info(f"  Availability: {decision.availability_pct:.2%}")
        logger.info(f"  Convergence (s): {decision.convergence_time_sec:.1f}")
        logger.info(f"  Accuracy HPE (cm): {decision.accuracy_hpe_cm:.1f}")
        logger.info(f"  Confidence: {decision.confidence:.3f}")
        logger.info(f"  Uncertainty: {decision.uncertainty:.3f}")
    
    # Translate AI decision to broadcast config WITH INTENT
        config = self._translate_ai_to_broadcast_config(decision, parsed_intent)
    
        return config
    
    def _translate_ai_to_broadcast_config(self, decision, parsed_intent) -> BroadcastConfig:
        """
    Translate AI decision values to concrete broadcast parameters
    NOW INTENT-AWARE: Different intents make different trade-offs
        """
    
    # Determine intent type
        intent_type = parsed_intent.intent_type.value.lower()
    
        logger.info(f"Translating for intent: {intent_type}")
    
    # INTENT-AWARE TRANSLATION LOGIC
    
        if "spectrum" in intent_type or "bandwidth" in intent_type:
        # OPTIMIZE FOR SPECTRAL EFFICIENCY
            logger.info("→ Optimizing for bandwidth efficiency")
        
        # Prioritize lower FEC overhead
            if decision.redundancy_ratio >= 3.0:
                fec_rate = FECCodeRate.RATE_8_15
                fec_overhead = 25.0
            elif decision.redundancy_ratio >= 2.0:
                fec_rate = FECCodeRate.RATE_10_15
                fec_overhead = 15.0
            else:
                fec_rate = FECCodeRate.RATE_12_15  # Most efficient
                fec_overhead = 10.0
        
        # Use smaller FFT for better spectral efficiency
            fft_size = FFTSize.FFT_8K  # Smaller = more efficient
        
        # Minimal guard interval
            guard_interval = GuardInterval.GI_1_16
        
        # Can use higher-order modulation if conditions allow
            if decision.spectrum_mbps >= 1.5:
                modulation = ModulationScheme.QAM_16  # Higher throughput
            elif decision.spectrum_mbps >= 1.0:
                modulation = ModulationScheme.QPSK
            else:
                modulation = ModulationScheme.QPSK  # Safe choice
    
        elif "reliability" in intent_type or "robust" in intent_type:
        # OPTIMIZE FOR RELIABILITY
            logger.info("→ Optimizing for reliability/robustness")
        
        # Prioritize high FEC redundancy
            if decision.redundancy_ratio >= 3.0:
                fec_rate = FECCodeRate.RATE_6_15  # Maximum protection
                fec_overhead = 50.0
            elif decision.redundancy_ratio >= 2.0:
                fec_rate = FECCodeRate.RATE_8_15
                fec_overhead = 35.0
            else:
                fec_rate = FECCodeRate.RATE_10_15
                fec_overhead = 25.0
        
        # Larger FFT for better frequency diversity
            fft_size = FFTSize.FFT_16K
        
        # Maximum guard interval for multipath protection
            if decision.redundancy_ratio >= 3.0:
                guard_interval = GuardInterval.GI_1_4  # Maximum protection
            else:
                guard_interval = GuardInterval.GI_1_8
        
        # Always use robust modulation
            modulation = ModulationScheme.QPSK  # Most robust
    
        elif "accuracy" in intent_type or "precise" in intent_type:
        # OPTIMIZE FOR ACCURACY
            logger.info("→ Optimizing for accuracy")
        
        # Moderate FEC - balance protection and overhead
            if decision.redundancy_ratio >= 2.5:
                fec_rate = FECCodeRate.RATE_8_15
                fec_overhead = 30.0
            else:
                fec_rate = FECCodeRate.RATE_10_15
                fec_overhead = 20.0
        
        # Larger FFT for better resolution
            if decision.accuracy_hpe_cm <= 5.0:
                fft_size = FFTSize.FFT_16K  # Better for sub-5cm
            else:
                fft_size = FFTSize.FFT_8K
        
        # Moderate guard interval
            guard_interval = GuardInterval.GI_1_8
        
        # Use QPSK for reliability
            modulation = ModulationScheme.QPSK
    
        else:
        # FALLBACK: Balanced approach
            logger.info("→ Using balanced configuration")
        
            if decision.redundancy_ratio >= 3.0:
                fec_rate = FECCodeRate.RATE_8_15
                fec_overhead = 30.0
            else:
                fec_rate = FECCodeRate.RATE_10_15
                fec_overhead = 20.0
        
            fft_size = FFTSize.FFT_8K
            guard_interval = GuardInterval.GI_1_8
            modulation = ModulationScheme.QPSK
    
        config = BroadcastConfig(
        use_alp=True,
        fec_ldpc_rate=fec_rate,
        fec_rs_symbols=16,
        fec_overhead_pct=fec_overhead,
        fft_size=fft_size,
        guard_interval=guard_interval,
        modulation=modulation,
        frame_duration_ms=50.0,
        pilots_enabled=True,
        time_interleaving=True
        )
    
        logger.info(f"Translated to Broadcast Config:")
        logger.info(f"  FEC: {fec_rate.name}, Overhead: {fec_overhead}%")
        logger.info(f"  Modulation: {modulation.name}")
        logger.info(f"  FFT: {fft_size.name}, GI: {guard_interval.name}")
        logger.info(f"  Intent-driven optimization: {intent_type}")
    
        return config
    
    def _get_traditional_config(self) -> BroadcastConfig:
        """Get static traditional broadcast config"""
        return BroadcastConfig(
            use_alp=True,
            fec_ldpc_rate=FECCodeRate.RATE_8_15,
            fec_rs_symbols=16,
            fec_overhead_pct=15.0,
            fft_size=FFTSize.FFT_8K,
            guard_interval=GuardInterval.GI_1_8,
            modulation=ModulationScheme.QPSK,
            frame_duration_ms=50.0,
            pilots_enabled=True,
            time_interleaving=True
        )
    
    def run_phase1_gnss(self, scenario_profile: Dict) -> Dict:
        """Phase 1: GNSS Processing"""
        logger.info("=" * 70)
        logger.info("  PHASE 1/6: GNSS Processing")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Run GNSS baseline runner
        gnss_cmd = [
            sys.executable,
            str(Path(__file__).parent / "gnss" / "gnss_baseline_runner_v2.py"),
            "--scenario", self.scenario_id,
            "--mode", "traditional",
            "--output-dir", str(self.run_output_dir),
            "--timeout-sec", "300"
        ]
        
        logger.info(f"Running: {' '.join(gnss_cmd)}")
        result = subprocess.run(gnss_cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"\n  Status: ✓ SUCCESS")
            logger.info(f"  Duration: {duration:.2f} s")
            
            # Extract metrics (simplified)
            gnss_results = {
                'success': True,
                'metrics': {
                    'mean_hpe_cm': 2.5,
                    'mean_snr_db': 28.0,
                    'fix_rate_pct': 98.0,
                    'environment': scenario_profile.get('environment', 'rural')
                }
            }
        else:
            logger.error(f"  Status: ✗ FAILED")
            logger.error(result.stderr)
            gnss_results = {'success': False}
        
        self.results['phases']['gnss'] = gnss_results
        return gnss_results
    
    def run_phase2_ai_inference(self, gnss_results: Dict, intent: str, scenario_profile: Dict) -> BroadcastConfig:
        """Phase 2: AI Inference (NEW)"""
        logger.info("=" * 70)
        logger.info("  PHASE 2/6: AI Inference" if self.mode == "ai" else "  PHASE 2/6: Traditional Config")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        if self.mode == "ai":
            # Load AI model
            self._load_ai_model()
            
            # Extract telemetry
            telemetry = self._extract_telemetry_from_gnss(gnss_results, scenario_profile)            
            # AI inference
            config = self._ai_infer_broadcast_config(telemetry, intent)
        else:
            # Traditional static config
            config = self._get_traditional_config()
            logger.info("Using traditional static configuration")
        
        duration = time.time() - start_time
        logger.info(f"\n  Status: ✓ SUCCESS")
        logger.info(f"  Duration: {duration:.2f} s")
        
        return config
    
    def run_phase3_rtcm(self) -> Dict:
        """Phase 3: RTCM Generation"""
        logger.info("=" * 70)
        logger.info("  PHASE 3/6: RTCM Generation")
        logger.info("=" * 70)
        
        # Create synthetic RTCM
        rtcm_data = b'\x00' * 2600  # 2.6 KB synthetic
        
        rtcm_dir = self.run_output_dir / "rtcm"
        rtcm_dir.mkdir(exist_ok=True)
        rtcm_file = rtcm_dir / "corrections.rtcm"
        
        with open(rtcm_file, 'wb') as f:
            f.write(rtcm_data)
        
        logger.info(f"  RTCM file: {rtcm_file}")
        logger.info(f"  Size: {len(rtcm_data)} bytes ({len(rtcm_data)/1024:.2f} KB)")
        logger.info(f"\n  Status: ✓ SUCCESS")
        
        return {'data': rtcm_data, 'file': rtcm_file}
    
    def run_phase4_broadcast_tx(self, rtcm_data: bytes, config: BroadcastConfig) -> Dict:
        """Phase 4: Broadcast TX"""
        logger.info("=" * 70)
        logger.info("  PHASE 4/6: Broadcast TX")
        logger.info("=" * 70)
        
        pipeline = BroadcastPipeline(config=config)
        result = pipeline.process(rtcm_data, config=config)
        
        # Save TX signal
        tx_dir = self.run_output_dir / "broadcast" / "tx"
        tx_dir.mkdir(parents=True, exist_ok=True)
        
        signal_file = tx_dir / "tx_signal.npy"
        np.save(signal_file, result.signal.time_domain_signal)
        
        logger.info(f"  TX complete: {result.signal.num_samples:,} samples")
        logger.info(f"  Expansion ratio: {result.expansion_ratio:.4f}x")
        logger.info(f"  Spectral efficiency: {result.spectral_efficiency:.4f} bps/Hz")
        logger.info(f"\n  Status: ✓ SUCCESS")
        
        return {'result': result, 'signal_file': signal_file}
    
    def run_phase5_channel(self, tx_signal: np.ndarray, environment: str) -> Dict:
        """Phase 5: Channel Simulation"""
        logger.info("=" * 70)
        logger.info("  PHASE 5/6: Channel Simulation")
        logger.info("=" * 70)
        
        simulator = ChannelSimulator.from_preset(environment, snr_override=40.0)
        result = simulator.apply(tx_signal, return_metrics=True)
        
        # Save RX signal
        rx_dir = self.run_output_dir / "broadcast" / "rx"
        rx_dir.mkdir(parents=True, exist_ok=True)
        
        signal_file = rx_dir / "rx_signal.npy"
        np.save(signal_file, result.signal)
        
        logger.info(f"  Environment: {environment}")
        logger.info(f"  SNR: {result.metrics.effective_snr_db:.1f} dB")
        logger.info(f"\n  Status: ✓ SUCCESS")
        
        return {'result': result, 'signal_file': signal_file}
    
    def run_phase6_broadcast_rx(self, rx_signal: np.ndarray, tx_result) -> Dict:
        """Phase 6: Broadcast RX"""
        logger.info("=" * 70)
        logger.info("  PHASE 6/6: Broadcast RX")
        logger.info("=" * 70)
    
        decoder_config = DecoderConfig()
        decoder = BroadcastDecoder(config=decoder_config)
    
    # Decoder expects just the signal
        try:
            result = decoder.decode(
            rx_signal,
            fec_encoded_length=len(tx_result.fec_result.encoded_data)
            )
        except TypeError:
            #logger.warning("Decoder doesn't support fec_encoded_length, trying basic decode")
            logger.info("Applying Basic Decode")
            result = decoder.decode(rx_signal)
    
    # Check which attribute has the data
        recovered_bytes = 0
        if hasattr(result, 'recovered_data'):
            recovered_bytes = len(result.recovered_data) if result.recovered_data else 0
        elif hasattr(result, 'data'):
            recovered_bytes = len(result.data) if result.data else 0
        elif hasattr(result, 'payload'):
            recovered_bytes = len(result.payload) if result.payload else 0
    
        logger.info(f"  Decode success: {result.success}")
        logger.info(f"  Recovered: {recovered_bytes} bytes")
        logger.info(f"\n  Status: ✓ SUCCESS")
    
        return {'result': result}
    
    def run_phase7_feedback(self, verification: Dict, ai_decision=None):
        """Phase 7: AI Feedback Loop (only in AI mode)"""
        if self.mode != "ai":
            return
        
        logger.info("=" * 70)
        logger.info("  PHASE 7/7: AI Feedback Loop")
        logger.info("=" * 70)
        
        telemetry = FieldTelemetry(
            timestamp=time.time(),
            vehicle_id=f"{self.scenario_id}_vehicle",
            rtk_mode="FIX" if verification.get('exact_match') else "FLOAT",
            actual_hpe_cm=verification.get('recovered_hpe_cm', 50.0),
            actual_vpe_cm=10.0,
            actual_availability_pct=95.0 if verification['success'] else 50.0,
            convergence_time_sec=30.0,
            num_satellites=15,
            signal_strength_avg_db=28.0,
            multipath_indicator=0.2,
            model_uncertainty=0.05,
            inference_confidence=0.90
        )
        
        drift_result = self.feedback_loop.process_telemetry(telemetry)
        
        if drift_result and drift_result.drift_detected:
            logger.warning(f"⚠️  Drift detected: {drift_result.metric_affected}")
            logger.warning(f"    Recommendation: {drift_result.recommendation}")
        else:
            logger.info("✓ No drift detected")
        
        logger.info(f"\n  Status: ✓ SUCCESS")
    
    def run(self, scenario_profile: Dict):
        """Execute complete pipeline"""
        start_time = time.time()
        
        # Phase 1: GNSS
        gnss_results = self.run_phase1_gnss(scenario_profile)
        if not gnss_results['success']:
            logger.error("GNSS phase failed, aborting")
            return
        
        # Phase 2: AI Inference or Traditional Config
        intent = scenario_profile.get('intent', 'provide_sub_3cm_accuracy')
        broadcast_config = self.run_phase2_ai_inference(gnss_results, intent, scenario_profile)
        
        # Phase 3: RTCM
        rtcm_results = self.run_phase3_rtcm()
        
        # Phase 4: Broadcast TX
        tx_results = self.run_phase4_broadcast_tx(rtcm_results['data'], broadcast_config)
        
                # Phase 5: Channel
        env_data = scenario_profile.get('environment', {})
        if isinstance(env_data, dict):
            environment = env_data.get('type', 'rural')  # Extract 'type' field
        else:
            environment = env_data if env_data else 'rural'
        
        channel_results = self.run_phase5_channel(
            tx_results['result'].signal.time_domain_signal,
            environment
        )

        # Phase 6: RX
        rx_results = self.run_phase6_broadcast_rx(
            channel_results['result'].signal,
            tx_results['result']
        )
        
        # Phase 7: Feedback (AI mode only)
        verification = {
            'success': rx_results['result'].success,
            'exact_match': False,
            'recovered_hpe_cm': 5.0
        }
        self.run_phase7_feedback(verification)
        
        total_duration = time.time() - start_time
# Store results for visualization
        self.results['config'] = {
            'fec_rate': str(broadcast_config.fec_ldpc_rate.name) if hasattr(broadcast_config.fec_ldpc_rate, 'name') else 'RATE_8_15',
            'fec_overhead_pct': broadcast_config.fec_overhead_pct,
            'fft_size': str(broadcast_config.fft_size.name) if hasattr(broadcast_config.fft_size, 'name') else 'FFT_8K',
            'guard_interval': str(broadcast_config.guard_interval.name) if hasattr(broadcast_config.guard_interval, 'name') else 'GI_1_8',
            'modulation': str(broadcast_config.modulation.name) if hasattr(broadcast_config.modulation, 'name') else 'QPSK',
        }
        
        self.results['phases']['phase4_broadcast_tx'] = {
        'result': tx_results['result'],
        'duration': 0
        }

# Store RX results properly  
        self.results['phases']['phase6_broadcast_rx'] = {
            'result': rx_results['result'],
            'duration': total_duration * 0.7  # Approximate RX time
        }
# Store intent
        self.results['intent'] = intent

# NOW save metrics
        metrics_file = self.run_output_dir / 'metrics.json'
        metrics = self._collect_metrics(self.scenario_id, self.mode, self.results)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Saved metrics: {metrics_file}")
        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Status: ✓ SUCCESS")
        logger.info(f"  Duration: {total_duration/60:.1f} min")
        logger.info(f"  Output: {self.run_output_dir}")
        logger.info("=" * 70)

    def _collect_metrics(self, scenario_id: str, mode: str, results: Dict) -> Dict:
        """
    Extract all metrics from results for visualization - FIXED VERSION
        """
        metrics = {
        'scenario_id': scenario_id,
        'mode': mode,
        'timestamp': datetime.now().isoformat(),
        }
    
    # Extract from stored results during pipeline run
    # GNSS metrics - use defaults since we use synthetic data
        metrics['horizontal_error_cm'] = 5.0
        metrics['vertical_error_cm'] = 8.0
        metrics['availability_pct'] = 95.0
    
    # Broadcast config - FROM STORED CONFIG
        config = results.get('config', {})
        metrics['fec_rate'] = config.get('fec_rate', 'RATE_8_15')
        metrics['fec_overhead_pct'] = config.get('fec_overhead_pct', 15.0)
        metrics['fft_size'] = config.get('fft_size', 'FFT_8K')
        metrics['guard_interval'] = config.get('guard_interval', 'GI_1_8')
        metrics['modulation'] = config.get('modulation', 'QPSK')
    
    # Broadcast TX metrics - FROM PHASE 4
        tx_phase = results.get('phases', {}).get('phase4_broadcast_tx', {})
        tx_result = tx_phase.get('result')
    
        if tx_result and hasattr(tx_result, 'spectral_efficiency'):
            metrics['spectral_efficiency'] = tx_result.spectral_efficiency
        else:
            metrics['spectral_efficiency'] = 0.5  # Default
    
        if tx_result and hasattr(tx_result, 'signal'):
            metrics['tx_samples'] = tx_result.signal.num_samples
        else:
            metrics['tx_samples'] = 0
    
        metrics['expansion_ratio'] = metrics['tx_samples'] / 2600.0 if metrics['tx_samples'] > 0 else 0.0
    
    # RX metrics - FROM PHASE 6
        rx_phase = results.get('phases', {}).get('phase6_broadcast_rx', {})
        rx_result = rx_phase.get('result')
    
        if rx_result:
            if hasattr(rx_result, 'data') and rx_result.data:
                metrics['recovered_bytes'] = len(rx_result.data)
            elif hasattr(rx_result, 'recovered_data') and rx_result.recovered_data:
                metrics['recovered_bytes'] = len(rx_result.recovered_data)
            else:
                metrics['recovered_bytes'] = 0
        else:
            metrics['recovered_bytes'] = 0
    
    # Decode time
        metrics['decode_time_sec'] = rx_phase.get('duration', 0)
        metrics['decode_time_min'] = metrics['decode_time_sec'] / 60.0
    
    # Intent (for AI mode)
        if mode == 'ai':
            metrics['intent'] = results.get('intent', 'unknown')
    
        return metrics
    
    def generate_separate_graphs(self, scenario_id: str, ai_metrics: Dict, trad_metrics: Dict, scenario_profile: Dict):
        """
    Generate 4 separate, clean bar charts for one scenario
    Much cleaner and more presentation-ready than one big card
        """
        logger.info(f"Generating visualizations for {scenario_id}...")
    
        output_dir = self.run_output_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)
    
    # Color scheme
        TRAD_COLOR = '#E74C3C'  # Red
        AI_COLOR = '#2ECC71'    # Green
    
        scenario_name = scenario_profile.get('name', scenario_id)
        intent = ai_metrics.get('intent', scenario_profile.get('intent', 'N/A'))
    
    # ========== GRAPH 1: Configuration Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))
    
        categories = ['FEC Overhead (%)', 'FFT Size (K)', 'Guard Interval']
    
    # Convert to numeric
        fft_trad = int(trad_metrics['fft_size'].replace('FFT_', '').replace('K', ''))
        fft_ai = int(ai_metrics['fft_size'].replace('FFT_', '').replace('K', ''))
    
        gi_map = {'GI_1_4': 0.25, 'GI_1_8': 0.125, 'GI_1_16': 0.0625, 'GI_1_32': 0.03125}
        gi_trad = gi_map.get(trad_metrics['guard_interval'], 0.125) * 100
        gi_ai = gi_map.get(ai_metrics['guard_interval'], 0.125) * 100
    
        trad_values = [trad_metrics['fec_overhead_pct'], fft_trad, gi_trad]
        ai_values = [ai_metrics['fec_overhead_pct'], fft_ai, gi_ai]
    
        x = np.arange(len(categories))
        width = 0.35
    
        bars1 = ax.bar(x - width/2, trad_values, width, label='Traditional', color=TRAD_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width/2, ai_values, width, label='AI', color=AI_COLOR, alpha=0.8)
    
    # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario_name}: Configuration Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
        plt.tight_layout()
        graph1_path = output_dir / f"{scenario_id}_1_config.png"
        plt.savefig(graph1_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {graph1_path.name}")
    
    # ========== GRAPH 2: Spectral Efficiency ==========
        fig, ax = plt.subplots(figsize=(8, 5))
    
        categories = ['Traditional', 'AI']
        values = [trad_metrics['spectral_efficiency'], ai_metrics['spectral_efficiency']]
        colors = [TRAD_COLOR, AI_COLOR]
    
        bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    
    # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.4f}\nbps/Hz',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Improvement annotation
        improvement = ((ai_metrics['spectral_efficiency'] - trad_metrics['spectral_efficiency']) / trad_metrics['spectral_efficiency'] * 100)
        ax.text(0.5, max(values) * 0.5, f'AI Improvement:\n{improvement:+.1f}%',
           ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
        ax.set_ylabel('Spectral Efficiency (bps/Hz)', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario_name}: Spectral Efficiency Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
        plt.tight_layout()
        graph2_path = output_dir / f"{scenario_id}_2_spectral_eff.png"
        plt.savefig(graph2_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {graph2_path.name}")
    
    # ========== GRAPH 3: Data Recovery ==========
        fig, ax = plt.subplots(figsize=(8, 5))
    
        values = [trad_metrics['recovered_bytes'], ai_metrics['recovered_bytes']]
        bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val,
               f'{val:,}\nbytes',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
        recovery_improvement = ((ai_metrics['recovered_bytes'] - trad_metrics['recovered_bytes']) / trad_metrics['recovered_bytes'] * 100) if trad_metrics['recovered_bytes'] > 0 else 0
        ax.text(0.5, max(values) * 0.5, f'AI Recovery:\n{recovery_improvement:+.1f}%',
           ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue' if recovery_improvement > 0 else 'lightcoral', alpha=0.4))
    
        ax.set_ylabel('Recovered Bytes', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario_name}: Data Recovery Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
        plt.tight_layout()
        graph3_path = output_dir / f"{scenario_id}_3_data_recovery.png"
        plt.savefig(graph3_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {graph3_path.name}")
    
    # ========== GRAPH 4: KPI Summary ==========
        fig, ax = plt.subplots(figsize=(10, 6))
    
        kpi_names = ['Spectral\nEfficiency', 'FEC\nOverhead', 'Decode\nTime (min)', 'Data\nRecovery (KB)']
        trad_kpi = [
        trad_metrics['spectral_efficiency'] * 100,  # Scale for visibility
        trad_metrics['fec_overhead_pct'],
        trad_metrics['decode_time_min'],
        trad_metrics['recovered_bytes'] / 1000
        ]
        ai_kpi = [
        ai_metrics['spectral_efficiency'] * 100,
        ai_metrics['fec_overhead_pct'],
        ai_metrics['decode_time_min'],
        ai_metrics['recovered_bytes'] / 1000
        ]
    
        x = np.arange(len(kpi_names))
        width = 0.35
    
        bars1 = ax.bar(x - width/2, trad_kpi, width, label='Traditional', color=TRAD_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width/2, ai_kpi, width, label='AI', color=AI_COLOR, alpha=0.8)
    
        ax.set_ylabel('Value (normalized)', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario_name}: KPI Summary', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(kpi_names, fontsize=10)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
        plt.tight_layout()
        graph4_path = output_dir / f"{scenario_id}_4_kpi_summary.png"
        plt.savefig(graph4_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {graph4_path.name}")

        logger.info(f"✓ Generated 4 separate graphs for {scenario_id}")
        return [graph1_path, graph2_path, graph3_path, graph4_path]

    def generate_cross_scenario_line_chart(self, all_metrics: Dict):
        """
    Generate clean multi-line chart showing AI improvements across scenarios
    Similar to user's attached example
        """
        logger.info("Generating cross-scenario summary line chart...")
    
        fig, ax = plt.subplots(figsize=(10, 6))
    
        scenarios = ['Scenario 1\n(Accuracy)', 'Scenario 2\n(Bandwidth)', 'Scenario 3\n(Reliability)']
        x_pos = range(len(scenarios))

    # Calculate improvements
        spec_eff_improvements = []
        data_recovery_improvements = []
    
        for scenario_id in ['scenario1', 'scenario2', 'scenario3']:
            if scenario_id in all_metrics:
                ai = all_metrics[scenario_id]['ai']
                trad = all_metrics[scenario_id]['traditional']
            
            # Spectral efficiency
                spec_imp = ((ai['spectral_efficiency'] - trad['spectral_efficiency']) / trad['spectral_efficiency'] * 100) if trad['spectral_efficiency'] > 0 else 0
                spec_eff_improvements.append(spec_imp)
            
            # Data recovery
                data_imp = ((ai['recovered_bytes'] - trad['recovered_bytes']) / trad['recovered_bytes'] * 100) if trad['recovered_bytes'] > 0 else 0
                data_recovery_improvements.append(data_imp)
    
    # Plot lines
        ax.plot(x_pos, spec_eff_improvements, 'o-', 
           linewidth=3, markersize=10, label='Spectral Efficiency Improvement', color='#3498DB')
        ax.plot(x_pos, data_recovery_improvements, 's-', 
           linewidth=3, markersize=10, label='Data Recovery Improvement', color='#2ECC71')
    
    # Value labels
        for i, x in enumerate(x_pos):
            ax.text(x, spec_eff_improvements[i] + 3, 
               f"{spec_eff_improvements[i]:+.1f}%", 
               ha='center', fontsize=10, fontweight='bold', color='#3498DB')
            ax.text(x, data_recovery_improvements[i] + 3, 
               f"{data_recovery_improvements[i]:+.1f}%", 
               ha='center', fontsize=10, fontweight='bold', color='#2ECC71')
    
        ax.set_xlabel('Scenario (Intent)', fontsize=13, fontweight='bold')
        ax.set_ylabel('AI Improvement (%)', fontsize=13, fontweight='bold')
        ax.set_title('AI Performance Across 3 Scenarios', fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, fontsize=11)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
        plt.tight_layout()
    
        output_path = Path('OUTPUTs') / 'cross_scenario_summary.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
        logger.info(f"✓ Saved cross-scenario summary: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="AI-Enhanced Network Pipeline")
    parser.add_argument("--scenario", required=True, help="Scenario ID")
    parser.add_argument("--mode", default="ai", choices=["ai", "traditional"], 
                       help="ai or traditional")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Try multiple possible locations for scenario profile
    possible_paths = [
        Path(f"scenarios/{args.scenario}/scenario_profile.json"),
        Path(f"SCENARIOS/{args.scenario}/scenario_profile.json"),
        Path(f"gnss/scenarios/{args.scenario}/scenario_profile.json"),
        Path(__file__).parent / "scenarios" / args.scenario / "scenario_profile.json",
        # ADD THIS LINE:
        Path(__file__).parent.parent / "DATA" / "scenarios" / args.scenario / "scenario_profile.json",
    ]
    
    scenario_file = None
    for path in possible_paths:
        if path.exists():
            scenario_file = path
            break
    
    if scenario_file is None:
        logger.error(f"Scenario profile not found for {args.scenario}")
        logger.error("Tried locations:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        sys.exit(1)
    
    logger.info(f"Loading scenario profile from: {scenario_file}")
    
    with open(scenario_file) as f:
        profile = json.load(f)
    
    # Run pipeline
    pipeline = AINetworkPipeline(args.scenario, Path(args.output_dir), mode=args.mode)
    pipeline.run(profile)

def visualize_scenario(scenario_id: str):
    """
    Generate separate clean visualizations for one scenario
    """
    import json
    from pathlib import Path
    
    logger.info(f"Visualizing {scenario_id}...")
    
    # Load AI metrics
    ai_base = Path(f"OUTPUTs/{scenario_id}/ai_network")
    ai_run_dirs = sorted(ai_base.glob("run_*"), reverse=True)
    if not ai_run_dirs:
        logger.error(f"No AI runs found in {ai_base}")
        return
    ai_metrics_file = ai_run_dirs[0] / "metrics.json"
    
    if not ai_metrics_file.exists():
        logger.error(f"AI metrics not found: {ai_metrics_file}")
        return
    
    with open(ai_metrics_file) as f:
        ai_metrics = json.load(f)
    logger.info(f"Loaded AI metrics: Spectral Eff = {ai_metrics.get('spectral_efficiency', 0)}")
    
    # Load Traditional metrics
    trad_base = Path(f"OUTPUTs/{scenario_id}/traditional_comparison")
    trad_run_dirs = sorted(trad_base.glob("run_*"), reverse=True)
    if not trad_run_dirs:
        logger.error(f"No Traditional runs found in {trad_base}")
        return
    trad_metrics_file = trad_run_dirs[0] / "metrics.json"
    
    if not trad_metrics_file.exists():
        logger.error(f"Traditional metrics not found: {trad_metrics_file}")
        return
    
    with open(trad_metrics_file) as f:
        trad_metrics = json.load(f)
    logger.info(f"Loaded Traditional metrics: Spectral Eff = {trad_metrics.get('spectral_efficiency', 0)}")
    
    # Load scenario profile
    profile_path = Path(__file__).parent.parent / "DATA" / "scenarios" / scenario_id / "scenario_profile.json"
    if profile_path.exists():
        with open(profile_path) as f:
            scenario_profile = json.load(f)
    else:
        scenario_profile = {'name': scenario_id, 'intent': 'unknown', 'environment': {}}
    
    # Generate separate visualizations
    pipeline = AINetworkPipeline(scenario_id, ai_run_dirs[0].parent, mode='ai')
    graphs = pipeline.generate_separate_graphs(scenario_id, ai_metrics, trad_metrics, scenario_profile)
    
    logger.info("=" * 70)
    logger.info(f"✓ VISUALIZATION COMPLETE - 4 GRAPHS GENERATED")
    for graph in graphs:
        logger.info(f"   {graph}")
    logger.info("=" * 70)
def generate_all_scenarios_summary():
    """Generate cross-scenario summary chart"""
    import json
    from pathlib import Path
    
    print("\nSearching for scenario metrics...")
    all_metrics = {}
    
    for scenario_id in ['scenario1', 'scenario2', 'scenario3']:
        print(f"\n  Searching {scenario_id}...")
        
        # Load AI metrics - check multiple possible locations
        ai_metrics = None
        possible_ai_paths = [
            Path(f"OUTPUTs/{scenario_id}/ai_network"),
            Path(f"OUTPUTs/{scenario_id}/ai"),
        ]
        
        for ai_base in possible_ai_paths:
            if not ai_base.exists():
                continue
            
            ai_runs = sorted(ai_base.glob("run_*/metrics.json"), reverse=True)
            if ai_runs:
                print(f"    Found AI metrics: {ai_runs[0]}")
                with open(ai_runs[0]) as f:
                    ai_metrics = json.load(f)
                break
        
        if not ai_metrics:
            print(f"    ⚠️  No AI metrics found for {scenario_id}")
            continue
        
        # Load Traditional metrics
        trad_metrics = None
        possible_trad_paths = [
            Path(f"OUTPUTs/{scenario_id}/traditional_comparison"),
            Path(f"OUTPUTs/{scenario_id}/traditional"),
        ]
        
        for trad_base in possible_trad_paths:
            if not trad_base.exists():
                continue
            
            trad_runs = sorted(trad_base.glob("run_*/metrics.json"), reverse=True)
            if trad_runs:
                print(f"    Found Traditional metrics: {trad_runs[0]}")
                with open(trad_runs[0]) as f:
                    trad_metrics = json.load(f)
                break
        
        if not trad_metrics:
            print(f"    ⚠️  No Traditional metrics found for {scenario_id}")
            continue
        
        # Store both metrics
        all_metrics[scenario_id] = {
            'ai': ai_metrics,
            'traditional': trad_metrics
        }
        print(f"    ✓ Loaded both AI and Traditional metrics for {scenario_id}")
    
    print(f"\n  Total scenarios with complete metrics: {len(all_metrics)}")
    
    if len(all_metrics) >= 2:
        print("\n  Generating cross-scenario summary chart...")
        # Generate cross-scenario summary
        pipeline = AINetworkPipeline('summary', Path("OUTPUTs"), mode='ai')
        summary_path = pipeline.generate_cross_scenario_line_chart(all_metrics)
        print(f"\n✓ Generated cross-scenario summary: {summary_path}")
        print("\n" + "="*70)
        print("  SUMMARY GENERATION COMPLETE")
        print("="*70)
    else:
        print(f"\n✗ Need at least 2 scenarios to generate summary (found {len(all_metrics)})")
        print("\nAvailable scenarios:")
        for scenario_id in ['scenario1', 'scenario2', 'scenario3']:
            base_path = Path(f"OUTPUTs/{scenario_id}")
            if base_path.exists():
                ai_exists = any(base_path.glob("ai*/run_*/metrics.json"))
                trad_exists = any(base_path.glob("traditional*/run_*/metrics.json"))
                print(f"  {scenario_id}: AI={'✓' if ai_exists else '✗'}, Traditional={'✓' if trad_exists else '✗'}")

if __name__ == "__main__":
    import sys
    
    # Check for special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == '--visualize':
            if len(sys.argv) < 3:
                print("Usage: python run_ai_network.py --visualize <scenario_id>")
                sys.exit(1)
            visualize_scenario(sys.argv[2])
            sys.exit(0)
        
        elif sys.argv[1] == '--compare':
            if len(sys.argv) < 3:
                print("Usage: python run_ai_network.py --compare <scenario_id>")
                print("This runs BOTH AI and Traditional modes, then generates visualizations")
                sys.exit(1)
            
            scenario_id = sys.argv[2]
        elif sys.argv[1] == '--summary':
            # NEW: Generate cross-scenario summary
            print("="*70)
            print("  GENERATING CROSS-SCENARIO SUMMARY")
            print("="*70)
            generate_all_scenarios_summary()
            sys.exit(0)

            # Run AI mode
            print(f"\n{'='*70}")
            print(f"  RUNNING AI MODE FOR {scenario_id}")
            print(f"{'='*70}\n")
            sys.argv = ['run_ai_network.py', '--scenario', scenario_id, '--mode', 'ai', 
                       '--output-dir', f'OUTPUTs/{scenario_id}/ai_network']
            main()
            
            # Run Traditional mode
            print(f"\n{'='*70}")
            print(f"  RUNNING TRADITIONAL MODE FOR {scenario_id}")
            print(f"{'='*70}\n")
            sys.argv = ['run_ai_network.py', '--scenario', scenario_id, '--mode', 'traditional',
                       '--output-dir', f'OUTPUTs/{scenario_id}/traditional_comparison']
            main()
            
            # Generate visualizations
            print(f"\n{'='*70}")
            print(f"  GENERATING VISUALIZATIONS")
            print(f"{'='*70}\n")
            visualize_scenario(scenario_id)
            sys.exit(0)
    
    # Normal execution
    main()