#!/usr/bin/env python3
"""
Tests for scenario_simulator.py

Tests the synthetic scenario data generation for scenario2 and scenario3.
"""

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_simulator import (
    # Utilities
    meters_to_deg_lat,
    meters_to_deg_lon,
    normalize_heading,
    mode_name,
    # Data structures
    GroundTruthEpoch,
    RTCMScheduleEntry,
    MultipathEntry,
    ModeTransitionEntry,
    SimulationResult,
    # Generation functions
    generate_ground_truth,
    generate_corrections_schedule,
    generate_multipath_profile,
    generate_mode_transitions,
    # I/O functions
    write_ground_truth_csv,
    write_corrections_schedule_json,
    write_multipath_profile_json,
    write_mode_transitions_csv,
    update_scenario_profile,
    # Main simulation
    simulate_scenario,
    # Constants
    MODE_FIX,
    MODE_FLOAT,
    MODE_STANDALONE,
    VALID_SCENARIOS,
)


# ----------------------------
# Utility Function Tests
# ----------------------------

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_meters_to_deg_lat(self):
        """Test latitude conversion."""
        # ~111 km per degree at equator
        result = meters_to_deg_lat(111000)
        assert abs(result - 1.0) < 0.01
        
        # 1 meter
        result = meters_to_deg_lat(1.0)
        assert result == pytest.approx(1.0 / 111000.0, rel=1e-6)
    
    def test_meters_to_deg_lon(self):
        """Test longitude conversion at various latitudes."""
        # At equator
        result = meters_to_deg_lon(111000, 0.0)
        assert abs(result - 1.0) < 0.01
        
        # At 60 degrees (cos(60) = 0.5, so ~2 degrees per 111km)
        result = meters_to_deg_lon(111000, 60.0)
        assert abs(result - 2.0) < 0.05
    
    def test_normalize_heading(self):
        """Test heading normalization."""
        assert normalize_heading(0) == 0
        assert normalize_heading(90) == 90
        assert normalize_heading(360) == 0
        assert normalize_heading(450) == 90
        assert normalize_heading(-90) == 270
        assert normalize_heading(-360) == 0
    
    def test_mode_name(self):
        """Test mode name conversion."""
        assert mode_name(1) == "FIX"
        assert mode_name(2) == "FLOAT"
        assert mode_name(5) == "STANDALONE"
        assert mode_name(99) == "Q99"


# ----------------------------
# Ground Truth Generation Tests
# ----------------------------

class TestGroundTruthGeneration:
    """Tests for ground truth trajectory generation."""
    
    def test_generate_ground_truth_basic(self):
        """Test basic ground truth generation."""
        gt = generate_ground_truth(
            n_epochs=100,
            seed=42,
            scenario="scenario2",
        )
        
        assert len(gt) == 100
        assert all(isinstance(e, GroundTruthEpoch) for e in gt)
    
    def test_generate_ground_truth_deterministic(self):
        """Test that same seed produces same output."""
        gt1 = generate_ground_truth(n_epochs=50, seed=123, scenario="scenario2")
        gt2 = generate_ground_truth(n_epochs=50, seed=123, scenario="scenario2")
        
        for e1, e2 in zip(gt1, gt2):
            assert e1.lat == e2.lat
            assert e1.lon == e2.lon
            assert e1.height == e2.height
    
    def test_generate_ground_truth_different_seeds(self):
        """Test that different seeds produce different output."""
        gt1 = generate_ground_truth(n_epochs=50, seed=1, scenario="scenario2")
        gt2 = generate_ground_truth(n_epochs=50, seed=2, scenario="scenario2")
        
        # At least some positions should differ
        differences = sum(1 for e1, e2 in zip(gt1, gt2) if e1.lat != e2.lat)
        assert differences > 0
    
    def test_generate_ground_truth_timestamps(self):
        """Test that timestamps are sequential."""
        gt = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario2")
        
        for i, e in enumerate(gt):
            assert e.epoch == i
            # Timestamps should be sequential (1 second apart)
            if i > 0:
                assert e.timestamp == gt[i - 1].timestamp + 1
    
    def test_generate_ground_truth_valid_coordinates(self):
        """Test that coordinates are valid."""
        gt = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario2")
        
        for e in gt:
            assert -90 <= e.lat <= 90
            assert -180 <= e.lon <= 180
            assert e.height >= 0
            assert e.velocity_mps >= 0
            assert 0 <= e.heading_deg < 360
    
    def test_generate_ground_truth_scenario_differences(self):
        """Test that scenario2 and scenario3 have different dynamics."""
        gt2 = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario2")
        gt3 = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario3")
        
        # Scenario2 (highway) should have higher average velocity
        avg_v2 = sum(e.velocity_mps for e in gt2) / len(gt2)
        avg_v3 = sum(e.velocity_mps for e in gt3) / len(gt3)
        assert avg_v2 > avg_v3  # Highway faster than urban


# ----------------------------
# RTCM Schedule Generation Tests
# ----------------------------

class TestRTCMScheduleGeneration:
    """Tests for RTCM corrections schedule generation."""
    
    def test_generate_corrections_schedule_basic(self):
        """Test basic schedule generation."""
        schedule = generate_corrections_schedule(n_epochs=300, seed=42)
        
        assert len(schedule) > 0
        assert all(isinstance(s, RTCMScheduleEntry) for s in schedule)
    
    def test_generate_corrections_schedule_coverage(self):
        """Test that schedule covers all epochs."""
        schedule = generate_corrections_schedule(n_epochs=300, seed=42)
        
        # Check coverage
        covered = set()
        for s in schedule:
            for epoch in range(s.epoch_start, s.epoch_end + 1):
                covered.add(epoch)
        
        assert covered == set(range(300))
    
    def test_generate_corrections_schedule_deterministic(self):
        """Test that same seed produces same schedule."""
        s1 = generate_corrections_schedule(n_epochs=100, seed=42)
        s2 = generate_corrections_schedule(n_epochs=100, seed=42)
        
        assert len(s1) == len(s2)
        for e1, e2 in zip(s1, s2):
            assert e1.epoch_start == e2.epoch_start
            assert e1.available == e2.available
    
    def test_generate_corrections_schedule_has_dropouts(self):
        """Test that schedule includes dropout intervals."""
        # Use large n_epochs to ensure statistical coverage
        schedule = generate_corrections_schedule(n_epochs=1000, seed=42)
        
        # Should have at least some dropouts
        dropouts = sum(1 for s in schedule if not s.available)
        assert dropouts > 0
    
    def test_generate_corrections_schedule_interval_bounds(self):
        """Test that intervals are within expected range."""
        schedule = generate_corrections_schedule(n_epochs=300, seed=42)
        
        for s in schedule:
            interval = s.epoch_end - s.epoch_start + 1
            assert 1 <= interval <= 10  # Max interval is 10s


# ----------------------------
# Multipath Profile Generation Tests
# ----------------------------

class TestMultipathProfileGeneration:
    """Tests for multipath profile generation."""
    
    def test_generate_multipath_profile_basic(self):
        """Test basic multipath profile generation."""
        gt = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario3")
        profile = generate_multipath_profile(n_epochs=100, seed=42, ground_truth=gt)
        
        assert len(profile) == 100
        assert all(isinstance(p, MultipathEntry) for p in profile)
    
    def test_generate_multipath_profile_valid_values(self):
        """Test that multipath values are in valid ranges."""
        gt = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario3")
        profile = generate_multipath_profile(n_epochs=100, seed=42, ground_truth=gt)
        
        for p in profile:
            assert p.multipath_bias_m >= 0
            assert p.multipath_noise_m >= 0
            assert isinstance(p.nlos_affected, bool)
            assert 0 < p.sky_view_pct <= 100
            assert p.num_visible_sats >= 4
    
    def test_generate_multipath_profile_deterministic(self):
        """Test deterministic generation."""
        gt = generate_ground_truth(n_epochs=50, seed=42, scenario="scenario3")
        p1 = generate_multipath_profile(n_epochs=50, seed=42, ground_truth=gt)
        p2 = generate_multipath_profile(n_epochs=50, seed=42, ground_truth=gt)
        
        for e1, e2 in zip(p1, p2):
            assert e1.multipath_bias_m == e2.multipath_bias_m
            assert e1.nlos_affected == e2.nlos_affected


# ----------------------------
# Mode Transitions Tests
# ----------------------------

class TestModeTransitionsGeneration:
    """Tests for mode transitions generation."""
    
    def test_generate_mode_transitions_basic(self):
        """Test basic mode transitions generation."""
        gt = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario3")
        mp = generate_multipath_profile(n_epochs=100, seed=42, ground_truth=gt)
        transitions = generate_mode_transitions(
            n_epochs=100, seed=42, multipath_profile=mp, ground_truth=gt
        )
        
        assert len(transitions) == 100
        assert all(isinstance(t, ModeTransitionEntry) for t in transitions)
    
    def test_generate_mode_transitions_valid_modes(self):
        """Test that modes are valid."""
        gt = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario3")
        mp = generate_multipath_profile(n_epochs=100, seed=42, ground_truth=gt)
        transitions = generate_mode_transitions(
            n_epochs=100, seed=42, multipath_profile=mp, ground_truth=gt
        )
        
        valid_modes = {MODE_FIX, MODE_FLOAT, MODE_STANDALONE}
        for t in transitions:
            assert t.mode in valid_modes
            assert t.mode_name in ["FIX", "FLOAT", "STANDALONE"]
    
    def test_generate_mode_transitions_deterministic(self):
        """Test deterministic generation."""
        gt = generate_ground_truth(n_epochs=50, seed=42, scenario="scenario3")
        mp = generate_multipath_profile(n_epochs=50, seed=42, ground_truth=gt)
        
        t1 = generate_mode_transitions(n_epochs=50, seed=42, multipath_profile=mp, ground_truth=gt)
        t2 = generate_mode_transitions(n_epochs=50, seed=42, multipath_profile=mp, ground_truth=gt)
        
        for e1, e2 in zip(t1, t2):
            assert e1.mode == e2.mode
    
    def test_generate_mode_transitions_duration_tracking(self):
        """Test that duration tracking works correctly."""
        gt = generate_ground_truth(n_epochs=100, seed=42, scenario="scenario3")
        mp = generate_multipath_profile(n_epochs=100, seed=42, ground_truth=gt)
        transitions = generate_mode_transitions(
            n_epochs=100, seed=42, multipath_profile=mp, ground_truth=gt
        )
        
        # Duration should reset on mode change
        for i in range(1, len(transitions)):
            if transitions[i].mode != transitions[i - 1].mode:
                assert transitions[i].duration_in_mode == 1


# ----------------------------
# File I/O Tests
# ----------------------------

class TestFileIO:
    """Tests for file I/O functions."""
    
    def test_write_ground_truth_csv(self, tmp_path: Path):
        """Test writing ground truth CSV."""
        gt = generate_ground_truth(n_epochs=10, seed=42, scenario="scenario2")
        out_path = tmp_path / "gt.csv"
        
        write_ground_truth_csv(gt, out_path)
        
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        
        # Verify CSV structure
        with out_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 10
            assert "lat_deg" in reader.fieldnames
            assert "lon_deg" in reader.fieldnames
    
    def test_write_corrections_schedule_json(self, tmp_path: Path):
        """Test writing corrections schedule JSON."""
        schedule = generate_corrections_schedule(n_epochs=100, seed=42)
        out_path = tmp_path / "corrections_schedule.json"
        
        write_corrections_schedule_json(schedule, out_path)
        
        assert out_path.exists()
        
        with out_path.open() as f:
            data = json.load(f)
        
        assert data["synthetic"] is True
        assert "schedule" in data
        assert len(data["schedule"]) == len(schedule)
    
    def test_write_multipath_profile_json(self, tmp_path: Path):
        """Test writing multipath profile JSON."""
        gt = generate_ground_truth(n_epochs=50, seed=42, scenario="scenario3")
        profile = generate_multipath_profile(n_epochs=50, seed=42, ground_truth=gt)
        out_path = tmp_path / "multipath_profile.json"
        
        write_multipath_profile_json(profile, out_path)
        
        assert out_path.exists()
        
        with out_path.open() as f:
            data = json.load(f)
        
        assert data["synthetic"] is True
        assert "summary" in data
        assert "epochs" in data
        assert len(data["epochs"]) == 50
    
    def test_write_mode_transitions_csv(self, tmp_path: Path):
        """Test writing mode transitions CSV."""
        gt = generate_ground_truth(n_epochs=50, seed=42, scenario="scenario3")
        mp = generate_multipath_profile(n_epochs=50, seed=42, ground_truth=gt)
        transitions = generate_mode_transitions(
            n_epochs=50, seed=42, multipath_profile=mp, ground_truth=gt
        )
        out_path = tmp_path / "mode_transitions.csv"
        
        write_mode_transitions_csv(transitions, out_path)
        
        assert out_path.exists()
        
        with out_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 50
            assert "mode" in reader.fieldnames
            assert "mode_name" in reader.fieldnames


# ----------------------------
# Scenario Profile Update Tests
# ----------------------------

class TestScenarioProfileUpdate:
    """Tests for scenario profile update function."""
    
    def test_update_scenario_profile(self, tmp_path: Path):
        """Test updating scenario profile with synthetic metadata."""
        # Create a minimal profile
        profile_path = tmp_path / "scenario_profile.json"
        initial_profile = {
            "name": "test_scenario",
            "files": {
                "rover_obs": "rover.obs",
                "nav_file": "nav.nav",
            }
        }
        with profile_path.open("w") as f:
            json.dump(initial_profile, f)
        
        # Update it
        updated = update_scenario_profile(
            profile_path=profile_path,
            scenario="scenario2",
            n_epochs=100,
            seed=42,
        )
        
        assert "synthetic_metadata" in updated
        assert updated["synthetic_metadata"]["synthetic"] is True
        assert updated["synthetic_metadata"]["seed"] == 42
        assert updated["synthetic_metadata"]["n_epochs"] == 100
    
    def test_update_scenario_profile_with_comments(self, tmp_path: Path):
        """Test updating profile that has // comments."""
        profile_path = tmp_path / "scenario_profile.json"
        content = """// This is a comment
{
  "name": "test_scenario",
  "files": {
    "rover_obs": "rover.obs",
    "nav_file": "nav.nav"
  }
}"""
        profile_path.write_text(content)
        
        # Should not raise
        updated = update_scenario_profile(
            profile_path=profile_path,
            scenario="scenario2",
            n_epochs=50,
            seed=1,
        )
        
        assert updated["name"] == "test_scenario"
        assert updated["synthetic_metadata"]["synthetic"] is True


# ----------------------------
# Integration Tests
# ----------------------------

class TestSimulateScenario:
    """Integration tests for full scenario simulation."""
    
    @pytest.fixture
    def mock_scenario_dir(self, tmp_path: Path) -> Path:
        """Create a mock scenario directory structure."""
        scen_dir = tmp_path / "DATA" / "scenarios" / "scenario2"
        scen_dir.mkdir(parents=True)
        
        # Create placeholder files
        (scen_dir / "rover.obs").touch()
        (scen_dir / "nav.nav").touch()
        (scen_dir / "base.obs").touch()
        (scen_dir / "gt.csv").touch()
        
        # Create scenario profile
        profile = {
            "name": "scenario2_test",
            "files": {
                "rover_obs": "rover.obs",
                "nav_file": "nav.nav",
                "base_obs": "base.obs",
                "ground_truth": "gt.csv",
            },
            "station_coordinates": {
                "x_ecef": -2694500.00,
                "y_ecef": -4297800.00,
                "z_ecef": 3854500.00,
            },
        }
        with (scen_dir / "scenario_profile.json").open("w") as f:
            json.dump(profile, f)
        
        return tmp_path / "DATA" / "scenarios"
    
    def test_simulate_scenario2(self, mock_scenario_dir: Path):
        """Test full scenario2 simulation."""
        result = simulate_scenario(
            scenario="scenario2",
            seed=42,
            n_epochs=100,
            scenario_root=mock_scenario_dir,
        )
        
        assert result.success
        assert len(result.ground_truth) == 100
        assert result.corrections_schedule is not None
        assert len(result.corrections_schedule) > 0
        
        # Check files were created
        scen_dir = mock_scenario_dir / "scenario2"
        assert (scen_dir / "gt.csv").stat().st_size > 0
        assert (scen_dir / "corrections_schedule.json").exists()
    
    @pytest.fixture
    def mock_scenario3_dir(self, tmp_path: Path) -> Path:
        """Create a mock scenario3 directory structure."""
        scen_dir = tmp_path / "DATA" / "scenarios" / "scenario3"
        scen_dir.mkdir(parents=True)
        
        # Create placeholder files
        (scen_dir / "rover.obs").touch()
        (scen_dir / "nav.nav").touch()
        (scen_dir / "base.obs").touch()
        (scen_dir / "gt.csv").touch()
        
        # Create scenario profile
        profile = {
            "name": "scenario3_test",
            "files": {
                "rover_obs": "rover.obs",
                "nav_file": "nav.nav",
                "base_obs": "base.obs",
                "ground_truth": "gt.csv",
            },
            "station_coordinates": {
                "x_ecef": -2694200.00,
                "y_ecef": -4298000.00,
                "z_ecef": 3854200.00,
            },
        }
        with (scen_dir / "scenario_profile.json").open("w") as f:
            json.dump(profile, f)
        
        return tmp_path / "DATA" / "scenarios"
    
    def test_simulate_scenario3(self, mock_scenario3_dir: Path):
        """Test full scenario3 simulation."""
        result = simulate_scenario(
            scenario="scenario3",
            seed=42,
            n_epochs=100,
            scenario_root=mock_scenario3_dir,
        )
        
        assert result.success
        assert len(result.ground_truth) == 100
        assert result.multipath_profile is not None
        assert result.mode_transitions is not None
        assert len(result.multipath_profile) == 100
        assert len(result.mode_transitions) == 100
        
        # Check files were created
        scen_dir = mock_scenario3_dir / "scenario3"
        assert (scen_dir / "gt.csv").stat().st_size > 0
        assert (scen_dir / "multipath_profile.json").exists()
        assert (scen_dir / "mode_transitions.csv").exists()
    
    def test_simulate_nonexistent_scenario(self, tmp_path: Path):
        """Test simulation with nonexistent scenario directory."""
        result = simulate_scenario(
            scenario="scenario2",
            seed=42,
            n_epochs=100,
            scenario_root=tmp_path / "nonexistent",
        )
        
        assert not result.success
        assert "not found" in result.error_message.lower()
    
    def test_simulate_deterministic(self, mock_scenario_dir: Path):
        """Test that simulation is deterministic."""
        r1 = simulate_scenario(
            scenario="scenario2",
            seed=42,
            n_epochs=50,
            scenario_root=mock_scenario_dir,
        )
        
        # Reset files by recreating
        scen_dir = mock_scenario_dir / "scenario2"
        (scen_dir / "gt.csv").write_text("")
        
        r2 = simulate_scenario(
            scenario="scenario2",
            seed=42,
            n_epochs=50,
            scenario_root=mock_scenario_dir,
        )
        
        assert r1.success and r2.success
        for e1, e2 in zip(r1.ground_truth, r2.ground_truth):
            assert e1.lat == e2.lat
            assert e1.lon == e2.lon


# ----------------------------
# Constants Validation Tests
# ----------------------------

class TestConstants:
    """Tests for module constants."""
    
    def test_valid_scenarios(self):
        """Test that valid scenarios set is correct."""
        assert "scenario2" in VALID_SCENARIOS
        assert "scenario3" in VALID_SCENARIOS
        assert "scenario1" not in VALID_SCENARIOS
    
    def test_mode_constants(self):
        """Test mode constant values."""
        assert MODE_FIX == 1
        assert MODE_FLOAT == 2
        assert MODE_STANDALONE == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
