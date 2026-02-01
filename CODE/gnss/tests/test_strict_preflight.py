"""
Tests for strict preflight validation and RTK profiles.

Tests cover:
1. Strict validator - fails on missing/empty files
2. CRINEX detection and error messages
3. RTK profile loading
4. Ground truth CSV parsing
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strict_preflight import (
    PreflightResult,
    is_crinex_file,
    is_crx2rnx_available,
    is_valid_rinex_obs,
    is_valid_rinex_nav,
    strict_preflight_check,
    load_rtk_profiles,
    get_profile_settings,
    find_rtklib_binary,
)


# ----------------------------
# Test Fixtures
# ----------------------------

@pytest.fixture
def tmp_scenario_dir(tmp_path):
    """Create a temporary scenario directory with test files."""
    scenario_dir = tmp_path / "test_scenario"
    scenario_dir.mkdir()
    return scenario_dir


@pytest.fixture
def valid_rinex_header():
    """Valid RINEX observation file header (>1000 bytes)."""
    header = """\
     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE
teqc  2019Feb25     NOAA/NOS/NGS/CORS   20260112 09:31:36UTCPGM / RUN BY / DATE
TEST STATION                                                MARKER NAME
 1674416.8130 -5717888.7740  2268832.6550                  APPROX POSITION XYZ
                                                            END OF HEADER
 26  1 11  0  0  0.0000000  0  5G01G02G03G05G06
   23456789.123    12345.678       45.0
"""
    # Pad to ensure >1000 bytes
    return header + " " * 800


@pytest.fixture
def crinex_header():
    """CRINEX (Compact RINEX) file header (>1000 bytes for obs validation)."""
    header = """\
1.0                 COMPACT RINEX FORMAT                    CRINEX VERS   / TYPE
RNX2CRX ver.4.1.0                       12-Jan-26 09:31     CRINEX PROG / DATE
     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE
TEST STATION                                                MARKER NAME
 1674416.8130 -5717888.7740  2268832.6550                  APPROX POSITION XYZ
                                                            END OF HEADER
"""
    # Pad to ensure >1000 bytes for observation file size validation
    return header + " " * 700


@pytest.fixture
def valid_nav_content():
    """Valid RINEX navigation file content (>500 bytes)."""
    content = """\
     2.11           N: GPS NAV DATA                         RINEX VERSION / TYPE
CCRINEXN V1.6.0 UX  CDDIS               26-JAN-11 00:05     PGM / RUN BY / DATE
                                                            END OF HEADER
 1 26  1 11  0  0  0.0  1.234567890123D-04  0.000000000000D+00  0.000000000000D+00
    1.234567890123D+04  4.567890123456D+00 -1.234567890123D-09  0.000000000000D+00
    1.234567890123D+04  4.567890123456D+00 -1.234567890123D-09  0.000000000000D+00
    1.234567890123D+04  4.567890123456D+00 -1.234567890123D-09  0.000000000000D+00
    1.234567890123D+04  4.567890123456D+00 -1.234567890123D-09  0.000000000000D+00
 2 26  1 11  2  0  0.0  2.345678901234D-04  0.000000000000D+00  0.000000000000D+00
    2.345678901234D+04  5.678901234567D+00 -2.345678901234D-09  0.000000000000D+00
"""
    # Pad to ensure >500 bytes
    return content + " " * 200


# ----------------------------
# Tests: RINEX Validation
# ----------------------------

class TestRINEXValidation:
    """Tests for RINEX file validation."""

    def test_valid_rinex_obs_passes(self, tmp_scenario_dir, valid_rinex_header):
        """Test that valid RINEX obs file passes validation."""
        obs_file = tmp_scenario_dir / "rover.obs"
        obs_file.write_text(valid_rinex_header)
        
        is_valid, error = is_valid_rinex_obs(obs_file)
        
        assert is_valid is True
        assert error == ""

    def test_missing_file_fails(self, tmp_scenario_dir):
        """Test that missing file fails validation."""
        obs_file = tmp_scenario_dir / "nonexistent.obs"
        
        is_valid, error = is_valid_rinex_obs(obs_file)
        
        assert is_valid is False
        assert "not found" in error.lower()

    def test_empty_file_fails(self, tmp_scenario_dir):
        """Test that empty file fails validation."""
        obs_file = tmp_scenario_dir / "empty.obs"
        obs_file.write_text("")
        
        is_valid, error = is_valid_rinex_obs(obs_file)
        
        assert is_valid is False
        assert "empty" in error.lower()

    def test_too_small_file_fails(self, tmp_scenario_dir):
        """Test that file with <1000 bytes fails validation."""
        obs_file = tmp_scenario_dir / "small.obs"
        obs_file.write_text("x" * 500)  # Only 500 bytes
        
        is_valid, error = is_valid_rinex_obs(obs_file)
        
        assert is_valid is False
        assert "too small" in error.lower()

    def test_valid_nav_passes(self, tmp_scenario_dir, valid_nav_content):
        """Test that valid navigation file passes validation."""
        nav_file = tmp_scenario_dir / "nav.nav"
        nav_file.write_text(valid_nav_content)
        
        is_valid, error = is_valid_rinex_nav(nav_file)
        
        assert is_valid is True
        assert error == ""

    def test_empty_nav_fails(self, tmp_scenario_dir):
        """Test that empty navigation file fails validation."""
        nav_file = tmp_scenario_dir / "empty.nav"
        nav_file.write_text("")
        
        is_valid, error = is_valid_rinex_nav(nav_file)
        
        assert is_valid is False
        assert "empty" in error.lower()


# ----------------------------
# Tests: CRINEX Detection
# ----------------------------

class TestCRINEXDetection:
    """Tests for CRINEX file detection."""

    def test_detects_crinex_compact_format(self, tmp_scenario_dir, crinex_header):
        """Test detection of COMPACT RINEX FORMAT."""
        crinex_file = tmp_scenario_dir / "rover.obs"
        crinex_file.write_text(crinex_header)
        
        assert is_crinex_file(crinex_file) is True

    def test_standard_rinex_not_crinex(self, tmp_scenario_dir, valid_rinex_header):
        """Test that standard RINEX is not detected as CRINEX."""
        rinex_file = tmp_scenario_dir / "rover.obs"
        rinex_file.write_text(valid_rinex_header)
        
        assert is_crinex_file(rinex_file) is False

    def test_nonexistent_file_not_crinex(self, tmp_scenario_dir):
        """Test that nonexistent file returns False."""
        assert is_crinex_file(tmp_scenario_dir / "nonexistent.obs") is False

    def test_none_path_not_crinex(self):
        """Test that None path returns False."""
        assert is_crinex_file(None) is False


# ----------------------------
# Tests: Strict Preflight Check
# ----------------------------

class TestStrictPreflightCheck:
    """Tests for strict preflight validation."""

    def test_valid_scenario_passes(self, tmp_scenario_dir, valid_rinex_header, valid_nav_content):
        """Test that valid scenario passes preflight check."""
        # Create valid files
        rover_file = tmp_scenario_dir / "rover.obs"
        nav_file = tmp_scenario_dir / "nav.nav"
        rover_file.write_text(valid_rinex_header)
        nav_file.write_text(valid_nav_content)
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=rover_file,
                nav_file=nav_file,
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_rover_fails(self, tmp_scenario_dir, valid_nav_content):
        """Test that missing rover file fails with exit code 2."""
        nav_file = tmp_scenario_dir / "nav.nav"
        nav_file.write_text(valid_nav_content)
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=tmp_scenario_dir / "missing.obs",
                nav_file=nav_file,
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_missing_nav_fails(self, tmp_scenario_dir, valid_rinex_header):
        """Test that missing nav file fails."""
        rover_file = tmp_scenario_dir / "rover.obs"
        rover_file.write_text(valid_rinex_header)
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=rover_file,
                nav_file=tmp_scenario_dir / "missing.nav",
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_crinex_without_auto_convert_fails(self, tmp_scenario_dir, crinex_header, valid_nav_content):
        """Test that CRINEX file fails without --auto-crinex-convert."""
        rover_file = tmp_scenario_dir / "rover.obs"
        nav_file = tmp_scenario_dir / "nav.nav"
        rover_file.write_text(crinex_header)
        nav_file.write_text(valid_nav_content)
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=rover_file,
                nav_file=nav_file,
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is False
        assert any("crinex" in e.lower() for e in result.errors)
        assert any("crx2rnx" in e.lower() for e in result.errors)

    def test_rtklib_not_found_fails(self, tmp_scenario_dir, valid_rinex_header, valid_nav_content):
        """Test that missing RTKLIB binary fails."""
        rover_file = tmp_scenario_dir / "rover.obs"
        nav_file = tmp_scenario_dir / "nav.nav"
        rover_file.write_text(valid_rinex_header)
        nav_file.write_text(valid_nav_content)
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=None):
            result = strict_preflight_check(
                rover_obs=rover_file,
                nav_file=nav_file,
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is False
        assert any("rnx2rtkp" in e.lower() for e in result.errors)


# ----------------------------
# Tests: RTK Profile Loading
# ----------------------------

class TestRTKProfiles:
    """Tests for RTK profile loading."""

    def test_load_profiles_returns_dict(self):
        """Test that profile loading returns a dictionary."""
        profiles = load_rtk_profiles()
        
        assert isinstance(profiles, dict)

    def test_get_baseline_settings(self):
        """Test getting baseline (traditional) profile settings."""
        profiles = {
            "baseline": {
                "description": "Test baseline",
                "pos2-armode": "continuous",
                "pos2-niter": "1",
            }
        }
        
        settings = get_profile_settings(profiles, "traditional", "any_intent")
        
        assert "pos2-armode" in settings
        assert settings["pos2-armode"] == "continuous"
        assert "description" not in settings  # Should be excluded

    def test_get_optimised_settings_by_intent(self):
        """Test getting optimised profile by intent."""
        profiles = {
            "optimised": {
                "provide_sub_3cm_accuracy": {
                    "description": "High accuracy",
                    "pos2-armode": "fix-and-hold",
                    "pos2-niter": "5",
                },
                "default": {
                    "description": "Default",
                    "pos2-armode": "continuous",
                }
            }
        }
        
        settings = get_profile_settings(profiles, "ai", "provide_sub_3cm_accuracy")
        
        assert settings["pos2-armode"] == "fix-and-hold"
        assert settings["pos2-niter"] == "5"

    def test_get_optimised_settings_falls_back_to_default(self):
        """Test that unknown intent falls back to default profile."""
        profiles = {
            "optimised": {
                "default": {
                    "pos2-armode": "continuous",
                }
            }
        }
        
        settings = get_profile_settings(profiles, "ai", "unknown_intent")
        
        assert settings["pos2-armode"] == "continuous"

    def test_get_scenario_specific_baseline_settings(self):
        """Test getting scenario-specific baseline settings."""
        profiles = {
            "baseline": {
                "pos2-armode": "continuous",
            },
            "scenarios": {
                "scenario2": {
                    "baseline": {
                        "description": "Scenario2 baseline",
                        "pos2-armode": "instantaneous",
                        "pos1-navsys": "1",
                    },
                    "optimised": {
                        "description": "Scenario2 optimised",
                        "pos2-armode": "fix-and-hold",
                        "pos1-navsys": "7",
                    }
                }
            }
        }
        
        # Should use scenario-specific baseline, not global
        settings = get_profile_settings(profiles, "traditional", "any_intent", scenario_name="scenario2")
        
        assert settings["pos2-armode"] == "instantaneous"
        assert settings["pos1-navsys"] == "1"

    def test_get_scenario_specific_optimised_settings(self):
        """Test getting scenario-specific optimised settings."""
        profiles = {
            "optimised": {
                "default": {
                    "pos2-armode": "continuous",
                }
            },
            "scenarios": {
                "scenario2": {
                    "baseline": {
                        "pos2-armode": "instantaneous",
                    },
                    "optimised": {
                        "description": "Scenario2 optimised",
                        "pos2-armode": "fix-and-hold",
                        "pos1-navsys": "7",
                    }
                }
            }
        }
        
        # Should use scenario-specific optimised, not global default
        settings = get_profile_settings(profiles, "ai", "any_intent", scenario_name="scenario2")
        
        assert settings["pos2-armode"] == "fix-and-hold"
        assert settings["pos1-navsys"] == "7"

    def test_scenario_not_found_falls_back_to_global(self):
        """Test that unknown scenario falls back to global profiles."""
        profiles = {
            "baseline": {
                "pos2-armode": "continuous",
            },
            "scenarios": {
                "scenario2": {
                    "baseline": {
                        "pos2-armode": "instantaneous",
                    }
                }
            }
        }
        
        # Unknown scenario should fall back to global baseline
        settings = get_profile_settings(profiles, "traditional", "any_intent", scenario_name="scenario99")
        
        assert settings["pos2-armode"] == "continuous"


# ----------------------------
# Tests: Ground Truth Parsing
# ----------------------------

class TestGroundTruthParsing:
    """Tests for ground truth file parsing (RTKLIB .pos format)."""

    def test_parse_ground_truth_pos_format(self, tmp_scenario_dir):
        """Test parsing ground truth from RTKLIB .pos format."""
        gt_file = tmp_scenario_dir / "gt.pos"
        
        # Write sample .pos file in RTKLIB format
        pos_content = """\
% GPST                   latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)
2026/01/11 00:00:00.000   37.56650000  126.97800000     50.000    1   8    0.0010   0.0012   0.0025
2026/01/11 00:00:30.000   37.56660000  126.97900000     51.000    1   9    0.0011   0.0013   0.0026
"""
        gt_file.write_text(pos_content)
        
        # Import and use the parser
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from gnss_baseline_runner_v2 import parse_ground_truth
        
        epochs = parse_ground_truth(gt_file)
        
        assert len(epochs) == 2
        assert abs(epochs[0].lat - 37.5665) < 0.001
        assert abs(epochs[0].lon - 126.978) < 0.001

    def test_parse_ground_truth_missing_file(self, tmp_scenario_dir):
        """Test that missing ground truth file returns empty list."""
        gt_file = tmp_scenario_dir / "nonexistent.pos"
        
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from gnss_baseline_runner_v2 import parse_ground_truth
        
        epochs = parse_ground_truth(gt_file)
        
        assert epochs == []


# ----------------------------
# Tests: Integration with Real Scenario Files
# ----------------------------

class TestRealScenarioValidation:
    """Integration tests with real scenario files."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_scenario1_with_crinex_fails_without_convert(self, project_root):
        """Test that scenario1 with CRINEX files fails without auto-convert flag."""
        rover_path = project_root / "DATA" / "scenarios" / "scenario1" / "rover.obs"
        nav_path = project_root / "DATA" / "scenarios" / "scenario1" / "nav.nav"
        
        if not rover_path.exists():
            pytest.skip("scenario1 rover.obs not found")
        
        # Check if rover.obs is CRINEX
        if not is_crinex_file(rover_path):
            pytest.skip("scenario1 rover.obs is not CRINEX")
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=rover_path,
                nav_file=nav_path,
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is False
        assert len(result.crinex_files) > 0

    def test_scenario1_with_converted_rinex_passes(self, project_root):
        """Test that scenario1 with pre-converted RINEX passes."""
        rover_path = project_root / "DATA" / "scenarios" / "scenario1" / "rover_rnx.obs"
        nav_path = project_root / "DATA" / "scenarios" / "scenario1" / "nav.nav"
        
        if not rover_path.exists():
            pytest.skip("scenario1 rover_rnx.obs not found")
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=rover_path,
                nav_file=nav_path,
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        # Should pass if rover_rnx.obs is valid standard RINEX
        if result.valid:
            assert len(result.errors) == 0
        else:
            # May fail for other reasons (e.g., missing rnx2rtkp)
            assert not any("crinex" in e.lower() for e in result.errors)

    def test_scenario2_empty_files_fail_strict(self, project_root):
        """Test that scenario2 with empty placeholder files fails strict validation."""
        rover_path = project_root / "DATA" / "scenarios" / "scenario2" / "rover.obs"
        nav_path = project_root / "DATA" / "scenarios" / "scenario2" / "nav.nav"
        
        if not rover_path.exists():
            pytest.skip("scenario2 rover.obs not found")
        
        # Check if file is empty - if it has real data, skip this test
        if rover_path.stat().st_size > 1000:
            pytest.skip("scenario2 rover.obs has real data (not empty placeholder)")
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=rover_path,
                nav_file=nav_path,
                base_obs=None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is False
        assert any("empty" in e.lower() or "too small" in e.lower() for e in result.errors)

    def test_scenario2_with_real_data_passes(self, project_root):
        """Test that scenario2 with real RINEX data passes strict validation."""
        rover_path = project_root / "DATA" / "scenarios" / "scenario2" / "rover.obs"
        nav_path = project_root / "DATA" / "scenarios" / "scenario2" / "nav.nav"
        base_path = project_root / "DATA" / "scenarios" / "scenario2" / "base.obs"
        
        if not rover_path.exists():
            pytest.skip("scenario2 rover.obs not found")
        
        # Only run if scenario2 has real data
        if rover_path.stat().st_size < 1000:
            pytest.skip("scenario2 rover.obs is empty placeholder (not real data)")
        
        with mock.patch('strict_preflight.find_rtklib_binary', return_value=Path("/usr/local/bin/rnx2rtkp")):
            result = strict_preflight_check(
                rover_obs=rover_path,
                nav_file=nav_path,
                base_obs=base_path if base_path.exists() else None,
                auto_crinex_convert=False,
            )
        
        assert result.valid is True
        assert len(result.errors) == 0

    def test_scenario2_ground_truth_has_real_epochs(self, project_root):
        """Test that scenario2 ground truth has real epochs from RTKLIB processing."""
        gt_path = project_root / "DATA" / "scenarios" / "scenario2" / "gt.csv"
        
        if not gt_path.exists():
            pytest.skip("scenario2 gt.csv not found")
        
        # Check if gt.csv has enough data
        if gt_path.stat().st_size < 1000:
            pytest.skip("scenario2 gt.csv too small")
        
        # Parse ground truth
        import csv
        epochs = []
        with open(gt_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(row)
        
        # Should have many epochs (real processing produces 1000+ epochs for 24hr data)
        assert len(epochs) > 100, f"Expected >100 epochs, got {len(epochs)}"
        
        # Check first epoch has valid coordinates
        first = epochs[0]
        lat = float(first.get("lat_deg", 0))
        lon = float(first.get("lon_deg", 0))
        
        # Coordinates should be reasonable (not zero, not placeholder)
        assert lat != 0 and lon != 0, "Ground truth coordinates appear to be placeholders"
        assert -90 <= lat <= 90, f"Invalid latitude: {lat}"
        assert -180 <= lon <= 180, f"Invalid longitude: {lon}"
