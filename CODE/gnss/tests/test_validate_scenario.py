#!/usr/bin/env python3
"""
Regression tests for validate_scenario.py

Tests validation logic for:
- scenario1: real RINEX with gt.csv (not .pos)
- scenario2/3: synthetic scenarios with relaxed RINEX checks
"""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from validate_scenario import (
    validate_scenario,
    load_json_with_comments,
    is_synthetic_scenario,
    find_ground_truth_file,
    MIN_FILE_SIZES,
)


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def mock_scenario1(tmp_path: Path) -> Path:
    """
    Create a mock scenario1 directory with real RINEX files.
    
    Has:
    - Valid rover.obs (>1KB)
    - Valid nav.nav (>500B)
    - Valid base.obs (>1KB)
    - gt.csv (NOT ground_truth.pos)
    """
    scen_dir = tmp_path / "scenarios" / "scenario1"
    scen_dir.mkdir(parents=True)
    
    # Create scenario profile (references ground_truth.pos but gt.csv exists)
    profile = {
        "name": "scenario1_test",
        "files": {
            "rover_obs": "rover.obs",
            "nav_file": "nav.nav",
            "base_obs": "base.obs",
            "ground_truth": "ground_truth.pos",  # Note: doesn't exist
        }
    }
    with (scen_dir / "scenario_profile.json").open("w") as f:
        json.dump(profile, f)
    
    # Create valid RINEX files (>min size)
    (scen_dir / "rover.obs").write_bytes(b"x" * 2000)  # > 1000
    (scen_dir / "nav.nav").write_bytes(b"x" * 1000)    # > 500
    (scen_dir / "base.obs").write_bytes(b"x" * 2000)   # > 1000
    
    # Create gt.csv (not ground_truth.pos)
    (scen_dir / "gt.csv").write_text("epoch,lat,lon,height\n0,37.5,126.9,50\n")
    
    return tmp_path / "scenarios"


@pytest.fixture
def mock_scenario1_no_gt(tmp_path: Path) -> Path:
    """
    Create scenario1 without any ground truth file.
    Should still pass validation (ground truth is optional).
    """
    scen_dir = tmp_path / "scenarios" / "scenario1"
    scen_dir.mkdir(parents=True)
    
    profile = {
        "name": "scenario1_no_gt",
        "files": {
            "rover_obs": "rover.obs",
            "nav_file": "nav.nav",
        }
    }
    with (scen_dir / "scenario_profile.json").open("w") as f:
        json.dump(profile, f)
    
    (scen_dir / "rover.obs").write_bytes(b"x" * 2000)
    (scen_dir / "nav.nav").write_bytes(b"x" * 1000)
    
    return tmp_path / "scenarios"


@pytest.fixture
def mock_scenario1_empty_rover(tmp_path: Path) -> Path:
    """
    Create scenario1 with empty rover.obs file.
    Should FAIL validation.
    """
    scen_dir = tmp_path / "scenarios" / "scenario1"
    scen_dir.mkdir(parents=True)
    
    profile = {
        "name": "scenario1_empty_rover",
        "files": {
            "rover_obs": "rover.obs",
            "nav_file": "nav.nav",
        }
    }
    with (scen_dir / "scenario_profile.json").open("w") as f:
        json.dump(profile, f)
    
    # Empty rover.obs - should fail
    (scen_dir / "rover.obs").write_bytes(b"")
    (scen_dir / "nav.nav").write_bytes(b"x" * 1000)
    
    return tmp_path / "scenarios"


@pytest.fixture
def mock_scenario1_missing_nav(tmp_path: Path) -> Path:
    """
    Create scenario1 with missing nav.nav file.
    Should FAIL validation.
    """
    scen_dir = tmp_path / "scenarios" / "scenario1"
    scen_dir.mkdir(parents=True)
    
    profile = {
        "name": "scenario1_missing_nav",
        "files": {
            "rover_obs": "rover.obs",
            "nav_file": "nav.nav",  # Specified but not created
        }
    }
    with (scen_dir / "scenario_profile.json").open("w") as f:
        json.dump(profile, f)
    
    (scen_dir / "rover.obs").write_bytes(b"x" * 2000)
    # nav.nav NOT created
    
    return tmp_path / "scenarios"


@pytest.fixture
def mock_scenario2_synthetic(tmp_path: Path) -> Path:
    """
    Create synthetic scenario2 with proper synthetic files.
    """
    scen_dir = tmp_path / "scenarios" / "scenario2"
    scen_dir.mkdir(parents=True)
    
    profile = {
        "name": "scenario2_synthetic",
        "files": {
            "rover_obs": "rover.obs",
            "nav_file": "nav.nav",
            "base_obs": "base.obs",
            "ground_truth": "gt.csv",
        },
        "synthetic_metadata": {
            "synthetic": True,
            "generator": "scenario_simulator.py",
            "scenario_type": "scenario2",
        }
    }
    with (scen_dir / "scenario_profile.json").open("w") as f:
        json.dump(profile, f)
    
    # Synthetic scenarios can have empty/placeholder RINEX
    (scen_dir / "rover.obs").write_bytes(b"")
    (scen_dir / "nav.nav").write_bytes(b"")
    (scen_dir / "base.obs").write_bytes(b"")
    
    # Ground truth
    (scen_dir / "gt.csv").write_text("epoch,lat,lon\n")
    
    # Required synthetic file
    (scen_dir / "corrections_schedule.json").write_text('{"synthetic": true}')
    
    return tmp_path / "scenarios"


@pytest.fixture
def mock_scenario3_synthetic(tmp_path: Path) -> Path:
    """
    Create synthetic scenario3 with proper synthetic files.
    """
    scen_dir = tmp_path / "scenarios" / "scenario3"
    scen_dir.mkdir(parents=True)
    
    profile = {
        "name": "scenario3_synthetic",
        "files": {
            "rover_obs": "rover.obs",
            "nav_file": "nav.nav",
            "ground_truth": "gt.csv",
        },
        "synthetic_metadata": {
            "synthetic": True,
            "generator": "scenario_simulator.py",
            "scenario_type": "scenario3",
        }
    }
    with (scen_dir / "scenario_profile.json").open("w") as f:
        json.dump(profile, f)
    
    # Synthetic can have empty RINEX
    (scen_dir / "rover.obs").write_bytes(b"")
    (scen_dir / "nav.nav").write_bytes(b"")
    
    # Ground truth
    (scen_dir / "gt.csv").write_text("epoch,lat,lon\n")
    
    # Required synthetic files for scenario3
    (scen_dir / "multipath_profile.json").write_text('{"synthetic": true}')
    (scen_dir / "mode_transitions.csv").write_text("epoch,mode\n0,1\n")
    
    return tmp_path / "scenarios"


@pytest.fixture
def mock_scenario2_synthetic_missing_schedule(tmp_path: Path) -> Path:
    """
    Create synthetic scenario2 WITHOUT corrections_schedule.json.
    Should FAIL validation.
    """
    scen_dir = tmp_path / "scenarios" / "scenario2"
    scen_dir.mkdir(parents=True)
    
    profile = {
        "name": "scenario2_synthetic_incomplete",
        "files": {
            "rover_obs": "rover.obs",
            "nav_file": "nav.nav",
        },
        "synthetic_metadata": {
            "synthetic": True,
            "scenario_type": "scenario2",
        }
    }
    with (scen_dir / "scenario_profile.json").open("w") as f:
        json.dump(profile, f)
    
    (scen_dir / "rover.obs").write_bytes(b"")
    (scen_dir / "nav.nav").write_bytes(b"")
    # corrections_schedule.json NOT created
    
    return tmp_path / "scenarios"


# ----------------------------
# Regression Tests
# ----------------------------

class TestScenario1Validation:
    """Regression tests for scenario1 (non-synthetic) validation."""
    
    def test_scenario1_passes_with_gt_csv(self, mock_scenario1: Path):
        """
        REGRESSION: scenario1 should pass when gt.csv exists,
        even if profile specifies ground_truth.pos.
        """
        is_valid, issues = validate_scenario(mock_scenario1, "scenario1")
        
        assert is_valid, f"Expected valid, got issues: {issues}"
        assert len(issues) == 0
    
    def test_scenario1_passes_without_ground_truth(self, mock_scenario1_no_gt: Path):
        """
        scenario1 should pass when no ground truth file exists.
        Ground truth is optional.
        """
        is_valid, issues = validate_scenario(mock_scenario1_no_gt, "scenario1")
        
        assert is_valid, f"Expected valid, got issues: {issues}"
        assert len(issues) == 0
    
    def test_scenario1_fails_with_empty_rover(self, mock_scenario1_empty_rover: Path):
        """
        REGRESSION: scenario1 should FAIL if rover.obs is empty.
        Non-synthetic scenarios require valid RINEX files.
        """
        is_valid, issues = validate_scenario(mock_scenario1_empty_rover, "scenario1")
        
        assert not is_valid, "Expected validation to fail with empty rover.obs"
        assert any("empty" in issue.lower() or "placeholder" in issue.lower() 
                   for issue in issues)
    
    def test_scenario1_fails_with_missing_nav(self, mock_scenario1_missing_nav: Path):
        """
        REGRESSION: scenario1 should FAIL if nav.nav is missing.
        """
        is_valid, issues = validate_scenario(mock_scenario1_missing_nav, "scenario1")
        
        assert not is_valid, "Expected validation to fail with missing nav.nav"
        assert any("not found" in issue.lower() for issue in issues)


class TestSyntheticScenarioValidation:
    """Tests for synthetic scenario validation."""
    
    def test_scenario2_synthetic_passes(self, mock_scenario2_synthetic: Path):
        """
        REGRESSION: scenario2 synthetic should pass with proper files.
        """
        is_valid, issues = validate_scenario(mock_scenario2_synthetic, "scenario2")
        
        assert is_valid, f"Expected valid, got issues: {issues}"
    
    def test_scenario3_synthetic_passes(self, mock_scenario3_synthetic: Path):
        """
        REGRESSION: scenario3 synthetic should pass with proper files.
        """
        is_valid, issues = validate_scenario(mock_scenario3_synthetic, "scenario3")
        
        assert is_valid, f"Expected valid, got issues: {issues}"
    
    def test_synthetic_allows_empty_rinex(self, mock_scenario2_synthetic: Path):
        """
        Synthetic scenarios should allow empty RINEX placeholder files.
        """
        is_valid, issues = validate_scenario(mock_scenario2_synthetic, "scenario2")
        
        assert is_valid, f"Synthetic should allow empty RINEX, got: {issues}"
        # No issues about empty/placeholder files
        assert not any("empty" in issue.lower() for issue in issues)
    
    def test_scenario2_synthetic_fails_without_schedule(
        self, mock_scenario2_synthetic_missing_schedule: Path
    ):
        """
        scenario2 synthetic should FAIL without corrections_schedule.json.
        """
        is_valid, issues = validate_scenario(
            mock_scenario2_synthetic_missing_schedule, "scenario2"
        )
        
        assert not is_valid, "Expected validation to fail without schedule"
        assert any("corrections_schedule.json" in issue for issue in issues)


class TestFindGroundTruthFile:
    """Tests for find_ground_truth_file function."""
    
    def test_finds_specified_file(self, tmp_path: Path):
        """Finds the exact file specified."""
        (tmp_path / "ground_truth.pos").write_text("data")
        
        found, path = find_ground_truth_file(tmp_path, "ground_truth.pos")
        
        assert found
        assert path.name == "ground_truth.pos"
    
    def test_finds_gt_csv_alternative(self, tmp_path: Path):
        """Finds gt.csv when ground_truth.pos doesn't exist."""
        (tmp_path / "gt.csv").write_text("data")
        
        found, path = find_ground_truth_file(tmp_path, "ground_truth.pos")
        
        assert found
        assert path.name == "gt.csv"
    
    def test_finds_ground_truth_csv_alternative(self, tmp_path: Path):
        """Finds ground_truth.csv as alternative."""
        (tmp_path / "ground_truth.csv").write_text("data")
        
        found, path = find_ground_truth_file(tmp_path, "ground_truth.pos")
        
        assert found
        assert path.name == "ground_truth.csv"
    
    def test_returns_false_when_no_file(self, tmp_path: Path):
        """Returns (False, specified_path) when no file found."""
        found, path = find_ground_truth_file(tmp_path, "ground_truth.pos")
        
        assert not found
        assert path.name == "ground_truth.pos"


class TestIsSyntheticScenario:
    """Tests for is_synthetic_scenario function."""
    
    def test_detects_synthetic_true(self):
        """Detects synthetic_metadata.synthetic = true."""
        cfg = {"synthetic_metadata": {"synthetic": True}}
        assert is_synthetic_scenario(cfg) is True
    
    def test_detects_non_synthetic(self):
        """Returns False when no synthetic_metadata."""
        cfg = {"name": "regular_scenario"}
        assert is_synthetic_scenario(cfg) is False
    
    def test_detects_explicit_false(self):
        """Returns False when synthetic is explicitly False."""
        cfg = {"synthetic_metadata": {"synthetic": False}}
        assert is_synthetic_scenario(cfg) is False


class TestLoadJsonWithComments:
    """Tests for JSON loading with comment stripping."""
    
    def test_loads_clean_json(self, tmp_path: Path):
        """Loads clean JSON without comments."""
        json_path = tmp_path / "test.json"
        json_path.write_text('{"key": "value"}')
        
        data = load_json_with_comments(json_path)
        
        assert data == {"key": "value"}
    
    def test_strips_line_comments(self, tmp_path: Path):
        """Strips // line comments."""
        json_path = tmp_path / "test.json"
        json_path.write_text('// This is a comment\n{"key": "value"}')
        
        data = load_json_with_comments(json_path)
        
        assert data == {"key": "value"}


# ----------------------------
# Integration with real scenarios
# ----------------------------

class TestRealScenarioValidation:
    """
    Integration tests with actual scenario directories.
    These tests use the real DATA/scenarios/* directories.
    """
    
    @pytest.fixture
    def real_scenario_root(self) -> Path:
        """Path to real scenarios, skips if not available."""
        root = Path(__file__).parent.parent.parent.parent / "DATA" / "scenarios"
        if not root.exists():
            pytest.skip("Real scenarios not available")
        return root
    
    def test_real_scenario1_validation(self, real_scenario_root: Path):
        """
        Integration: validate real scenario1.
        Should pass with existing gt.csv even if profile says ground_truth.pos.
        """
        scen_dir = real_scenario_root / "scenario1"
        if not scen_dir.exists():
            pytest.skip("scenario1 not available")
        
        is_valid, issues = validate_scenario(real_scenario_root, "scenario1")
        
        # Should pass - gt.csv exists and RINEX files are valid
        assert is_valid, f"Real scenario1 should validate, got: {issues}"
    
    def test_real_scenario2_validation(self, real_scenario_root: Path):
        """Integration: validate real scenario2 (may be synthetic)."""
        scen_dir = real_scenario_root / "scenario2"
        if not scen_dir.exists():
            pytest.skip("scenario2 not available")
        
        is_valid, issues = validate_scenario(real_scenario_root, "scenario2")
        
        # Should pass (synthetic or not)
        assert is_valid, f"Real scenario2 should validate, got: {issues}"
    
    def test_real_scenario3_validation(self, real_scenario_root: Path):
        """Integration: validate real scenario3 (may be synthetic)."""
        scen_dir = real_scenario_root / "scenario3"
        if not scen_dir.exists():
            pytest.skip("scenario3 not available")
        
        is_valid, issues = validate_scenario(real_scenario_root, "scenario3")
        
        # Should pass (synthetic or not)
        assert is_valid, f"Real scenario3 should validate, got: {issues}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
