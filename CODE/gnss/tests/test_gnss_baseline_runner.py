"""
Regression tests for gnss_baseline_runner_v2.py

STRICT REAL-DATA MODE:
- All tests validate against real RTKLIB processing paths only
- Synthetic scenario tests have been removed (no synthetic epochs allowed)
- CRINEX functions are imported from strict_preflight module

Tests cover:
1. RTKLIB .pos file parsing (various formats)
2. Run metrics computation
3. Mode transition detection
4. CRINEX (Compact RINEX) file detection (via strict_preflight)
5. RTKLIB configuration building
6. Synthetic scenario detection (via validate_scenario)
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnss_baseline_runner_v2 import (
    PositionEpoch,
    RunManifest,
    RunMetrics,
    parse_rtk_pos_file,
    parse_ground_truth,
    compute_run_metrics,
    compute_mode_transitions,
    build_rtklib_conf,
    ScenarioConfig,
    Q_FIX,
    Q_FLOAT,
    Q_SINGLE,
    MODE_TRADITIONAL,
    MODE_AI,
)

# CRINEX functions are now in strict_preflight module
from strict_preflight import (
    is_crinex_file,
    is_crx2rnx_available,
    is_valid_rinex_obs,
    is_valid_rinex_nav,
    strict_preflight_check,
)

# Import from validate_scenario for synthetic detection
from validate_scenario import is_synthetic_scenario


# ----------------------------
# Sample RTKLIB .pos snippets
# ----------------------------

SAMPLE_POS_DATE_TIME_FORMAT = """\
% program   : RTKLIB ver.2.4.3
% inp file  : rover.obs
% inp file  : base.obs
% inp file  : nav.nav
% pos mode  : kinematic
%
% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float,3:sbas,4:dgps,5:single,6:ppp,ns=# of satellites)
%  GPST          latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
2024/01/15 12:00:00.000   37.417679820 -122.090811920   825.7036   1  12   0.0023   0.0018   0.0045   0.0001   0.0002   0.0001    0.0   45.2
2024/01/15 12:00:01.000   37.417679825 -122.090811925   825.7038   1  12   0.0024   0.0019   0.0046   0.0001   0.0002   0.0001    0.0   42.1
2024/01/15 12:00:02.000   37.417679830 -122.090811930   825.7040   2  10   0.0523   0.0419   0.0946   0.0001   0.0002   0.0001    0.5   1.8
2024/01/15 12:00:03.000   37.417679835 -122.090811935   825.7042   5   8   0.5230   0.4190   0.9460   0.0001   0.0002   0.0001    1.0   0.0
2024/01/15 12:00:04.000   37.417679840 -122.090811940   825.7044   1  11   0.0025   0.0020   0.0047   0.0001   0.0002   0.0001    0.0   38.5
"""

SAMPLE_POS_GPS_WEEK_TOW_FORMAT = """\
% RTKLIB output
%  week     tow         lat           lon         height    Q  ns  sdn     sde     sdu     age   ratio
  2345  432000.000   37.417679820 -122.090811920   825.7036  1  12  0.0023  0.0018  0.0045  0.0   45.2
  2345  432001.000   37.417679825 -122.090811925   825.7038  1  12  0.0024  0.0019  0.0046  0.0   42.1
  2345  432002.000   37.417679830 -122.090811930   825.7040  2  10  0.0523  0.0419  0.0946  0.5   1.8
"""

SAMPLE_POS_NUMERIC_TOW_FORMAT = """\
% Simplified format
% timestamp lat lon height Q ns sdn sde sdu age ratio
32400.000   37.417679820 -122.090811920   825.7036  1  12  0.0023  0.0018  0.0045  0.0   45.2
32401.000   37.417679825 -122.090811925   825.7038  1  12  0.0024  0.0019  0.0046  0.0   42.1
32402.000   37.417679830 -122.090811930   825.7040  2  10  0.0523  0.0419  0.0946  0.5   1.8
32403.000   37.417679835 -122.090811935   825.7042  5   8  0.5230  0.4190  0.9460  1.0   0.0
"""

SAMPLE_POS_EMPTY = """\
% program   : RTKLIB ver.2.4.2
% inp file  : rover.obs
% obs start : 1970/01/01 00:00:00.0 GPST (week-522 -259200.0s)
% obs end   : 1970/01/01 00:00:00.0 GPST (week-522 -259200.0s)
%
% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float,3:sbas,4:dgps,5:single,6:ppp,ns=# of satellites)
%  GPST          latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
%
"""


class TestParsePosFile:
    """Tests for parsing RTKLIB .pos output files."""

    def test_parse_date_time_format(self, tmp_path):
        """Test parsing .pos file with YYYY/MM/DD HH:MM:SS.SSS format."""
        pos_file = tmp_path / "solution.pos"
        pos_file.write_text(SAMPLE_POS_DATE_TIME_FORMAT)
        
        epochs = parse_rtk_pos_file(pos_file)
        
        assert len(epochs) == 5
        assert epochs[0].quality == Q_FIX
        assert epochs[2].quality == Q_FLOAT
        assert epochs[3].quality == Q_SINGLE
        assert epochs[0].num_sats == 12

    def test_parse_gps_week_tow_format(self, tmp_path):
        """Test parsing .pos file with GPS week/TOW format."""
        pos_file = tmp_path / "solution.pos"
        pos_file.write_text(SAMPLE_POS_GPS_WEEK_TOW_FORMAT)
        
        epochs = parse_rtk_pos_file(pos_file)
        
        assert len(epochs) == 3
        assert epochs[0].timestamp == 432000.0
        assert epochs[1].timestamp == 432001.0

    def test_parse_numeric_tow_format(self, tmp_path):
        """Test parsing .pos file with simple numeric TOW format."""
        pos_file = tmp_path / "solution.pos"
        pos_file.write_text(SAMPLE_POS_NUMERIC_TOW_FORMAT)
        
        epochs = parse_rtk_pos_file(pos_file)
        
        assert len(epochs) == 4
        assert epochs[0].timestamp == 32400.0
        assert epochs[3].timestamp == 32403.0
        assert epochs[3].quality == Q_SINGLE

    def test_parse_empty_pos_file(self, tmp_path):
        """Test parsing .pos file with no data epochs."""
        pos_file = tmp_path / "solution.pos"
        pos_file.write_text(SAMPLE_POS_EMPTY)
        
        epochs = parse_rtk_pos_file(pos_file)
        
        assert len(epochs) == 0

    def test_parse_nonexistent_file(self, tmp_path):
        """Test parsing non-existent file returns empty list."""
        pos_file = tmp_path / "nonexistent.pos"
        
        epochs = parse_rtk_pos_file(pos_file)
        
        assert len(epochs) == 0

    def test_parse_extracts_ar_ratio(self, tmp_path):
        """Test that AR ratio is correctly extracted."""
        pos_file = tmp_path / "solution.pos"
        pos_file.write_text(SAMPLE_POS_DATE_TIME_FORMAT)
        
        epochs = parse_rtk_pos_file(pos_file)
        
        assert abs(epochs[0].ratio - 45.2) < 0.1
        assert abs(epochs[3].ratio - 0.0) < 0.1  # SINGLE has no ratio


class TestSyntheticScenarioDetection:
    """Tests for synthetic scenario detection."""

    def test_detects_synthetic_metadata_flag(self, tmp_path):
        """Test detection via synthetic_metadata.synthetic."""
        scenario_dir = tmp_path / "scenario2"
        scenario_dir.mkdir()
        
        profile = {
            "name": "scenario2",
            "synthetic_metadata": {
                "synthetic": True,
                "generator": "scenario_simulator.py"
            }
        }
        (scenario_dir / "scenario_profile.json").write_text(json.dumps(profile))
        
        assert is_synthetic_scenario(scenario_dir) is True

    def test_detects_top_level_synthetic_flag(self, tmp_path):
        """Test detection via top-level synthetic: true."""
        scenario_dir = tmp_path / "scenario_synth"
        scenario_dir.mkdir()
        
        profile = {
            "name": "scenario_synth",
            "synthetic": True
        }
        (scenario_dir / "scenario_profile.json").write_text(json.dumps(profile))
        
        assert is_synthetic_scenario(scenario_dir) is True

    def test_non_synthetic_scenario(self, tmp_path):
        """Test non-synthetic scenario returns False."""
        scenario_dir = tmp_path / "scenario1"
        scenario_dir.mkdir()
        
        profile = {
            "name": "scenario1",
            "description": "Real RINEX data"
        }
        (scenario_dir / "scenario_profile.json").write_text(json.dumps(profile))
        
        assert is_synthetic_scenario(scenario_dir) is False

    def test_missing_profile_returns_false(self, tmp_path):
        """Test missing scenario_profile.json returns False."""
        scenario_dir = tmp_path / "empty_scenario"
        scenario_dir.mkdir()
        
        assert is_synthetic_scenario(scenario_dir) is False


class TestRunManifestFields:
    """Tests for RunManifest dataclass fields."""

    def test_manifest_basic_fields(self):
        """Test that RunManifest has basic required fields."""
        manifest = RunManifest(
            run_id="test_run",
            scenario_name="scenario1",
            mode="traditional",
            intent="test",
            timestamp="2024-01-15T12:00:00",
        )
        
        assert manifest.run_id == "test_run"
        assert manifest.scenario_name == "scenario1"
        assert manifest.mode == "traditional"
        assert manifest.error_message == ""
        assert manifest.success is False

    def test_manifest_with_error(self):
        """Test RunManifest with error state."""
        manifest = RunManifest(
            run_id="test",
            scenario_name="test",
            mode="traditional",
            intent="test",
            timestamp="2026-01-01T00:00:00",
            success=False,
            error_message="Test error",
        )
        
        assert manifest.success is False
        assert "Test error" in manifest.error_message


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_compute_run_metrics_basic(self):
        """Test basic metrics computation."""
        epochs = [
            PositionEpoch(timestamp=i, lat=37.0, lon=-122.0, height=100.0, 
                         quality=Q_FIX if i < 7 else Q_FLOAT, num_sats=10)
            for i in range(10)
        ]
        
        metrics = compute_run_metrics(epochs)
        
        assert metrics.total_epochs == 10
        assert metrics.fix_epochs == 7
        assert metrics.float_epochs == 3
        assert metrics.fix_pct == 70.0
        assert metrics.availability_pct == 100.0  # FIX + FLOAT

    def test_compute_mode_transitions(self):
        """Test mode transition detection."""
        epochs = [
            PositionEpoch(timestamp=0, lat=37.0, lon=-122.0, height=100.0, quality=Q_FIX, num_sats=10),
            PositionEpoch(timestamp=1, lat=37.0, lon=-122.0, height=100.0, quality=Q_FIX, num_sats=10),
            PositionEpoch(timestamp=2, lat=37.0, lon=-122.0, height=100.0, quality=Q_FLOAT, num_sats=8),
            PositionEpoch(timestamp=3, lat=37.0, lon=-122.0, height=100.0, quality=Q_SINGLE, num_sats=6),
            PositionEpoch(timestamp=4, lat=37.0, lon=-122.0, height=100.0, quality=Q_FIX, num_sats=10),
        ]
        
        transitions = compute_mode_transitions(epochs)
        
        assert len(transitions) == 3  # FIX->FLOAT, FLOAT->SINGLE, SINGLE->FIX
        assert transitions[0].from_mode == "FIX"
        assert transitions[0].to_mode == "FLOAT"


class TestRealScenarioIntegration:
    """Integration tests with real scenario directories."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_scenario1_is_not_synthetic(self, project_root):
        """Test that scenario1 is detected as non-synthetic."""
        scenario_dir = project_root / "DATA" / "scenarios" / "scenario1"
        if not scenario_dir.exists():
            pytest.skip("scenario1 directory not found")
        
        assert is_synthetic_scenario(scenario_dir) is False

    def test_scenario2_is_synthetic(self, project_root):
        """Test that scenario2 is detected as synthetic."""
        scenario_dir = project_root / "DATA" / "scenarios" / "scenario2"
        if not scenario_dir.exists():
            pytest.skip("scenario2 directory not found")
        
        assert is_synthetic_scenario(scenario_dir) is True

    def test_scenario3_is_synthetic(self, project_root):
        """Test that scenario3 is detected as synthetic."""
        scenario_dir = project_root / "DATA" / "scenarios" / "scenario3"
        if not scenario_dir.exists():
            pytest.skip("scenario3 directory not found")
        
        assert is_synthetic_scenario(scenario_dir) is True


class TestCrinexDetection:
    """Tests for CRINEX (Compact RINEX) file detection."""

    def test_is_crinex_file_detects_compact_rinex_format(self, tmp_path):
        """Test detection of CRINEX file with COMPACT RINEX FORMAT header."""
        crinex_file = tmp_path / "test.obs"
        crinex_file.write_text(
            "1.0                 COMPACT RINEX FORMAT                    CRINEX VERS   / TYPE\n"
            "Some other content\n"
        )
        
        assert is_crinex_file(crinex_file) is True

    def test_is_crinex_file_detects_crinex_keyword(self, tmp_path):
        """Test detection of CRINEX file with CRINEX keyword."""
        crinex_file = tmp_path / "test.obs"
        crinex_file.write_text(
            "3.0                                                         CRINEX VERS / TYPE\n"
            "Some other content\n"
        )
        
        assert is_crinex_file(crinex_file) is True

    def test_is_crinex_file_returns_false_for_regular_rinex(self, tmp_path):
        """Test that regular RINEX files are not detected as CRINEX."""
        rinex_file = tmp_path / "test.obs"
        rinex_file.write_text(
            "     3.04           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n"
            "Some other content\n"
        )
        
        assert is_crinex_file(rinex_file) is False

    def test_is_crinex_file_returns_false_for_nonexistent_file(self, tmp_path):
        """Test that non-existent files return False."""
        nonexistent = tmp_path / "nonexistent.obs"
        
        assert is_crinex_file(nonexistent) is False

    def test_is_crinex_file_returns_false_for_none(self):
        """Test that None path returns False."""
        assert is_crinex_file(None) is False


class TestCrinexDetectionIntegration:
    """Integration tests for CRINEX detection with real scenario files."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_scenario1_rover_obs_is_crinex(self, project_root):
        """Test that scenario1's rover.obs is correctly detected as CRINEX."""
        rover_path = project_root / "DATA" / "scenarios" / "scenario1" / "rover.obs"
        if not rover_path.exists():
            pytest.skip("scenario1 rover.obs not found")
        
        assert is_crinex_file(rover_path) is True

    def test_scenario1_base_obs_is_crinex(self, project_root):
        """Test that scenario1's base.obs is correctly detected as CRINEX."""
        base_path = project_root / "DATA" / "scenarios" / "scenario1" / "base.obs"
        if not base_path.exists():
            pytest.skip("scenario1 base.obs not found")
        
        assert is_crinex_file(base_path) is True


class TestStrictPreflightValidation:
    """Tests for strict preflight validation."""

    def test_valid_rinex_passes(self, tmp_path):
        """Test that valid RINEX files pass validation."""
        rover = tmp_path / "rover.obs"
        nav = tmp_path / "nav.nav"
        
        # Create valid RINEX content (>1000 bytes with header)
        rover_content = "     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n"
        rover_content += "TEST                                                        MARKER NAME\n"
        rover_content += " " * 1000 + "\nEND OF HEADER\n"
        rover.write_text(rover_content)
        
        nav_content = "     2.11           N: GPS NAV DATA                         RINEX VERSION / TYPE\n"
        nav_content += " " * 500 + "\nEND OF HEADER\n"
        nav.write_text(nav_content)
        
        valid, error = is_valid_rinex_obs(rover)
        assert valid is True, f"Rover validation failed: {error}"
        
        valid, error = is_valid_rinex_nav(nav)
        assert valid is True, f"Nav validation failed: {error}"

    def test_empty_file_fails(self, tmp_path):
        """Test that empty file fails validation."""
        empty_file = tmp_path / "empty.obs"
        empty_file.write_text("")
        
        valid, error = is_valid_rinex_obs(empty_file)
        
        assert valid is False
        assert "empty" in error.lower()

    def test_missing_file_fails(self, tmp_path):
        """Test that missing file fails validation."""
        missing = tmp_path / "missing.obs"
        
        valid, error = is_valid_rinex_obs(missing)
        
        assert valid is False
        assert "not found" in error.lower()


class TestRTKLIBConfSettings:
    """Tests for RTKLIB configuration file generation."""

    def test_ar_mode_is_continuous_with_base(self, tmp_path):
        """Test AR mode is 'continuous' when base station is used."""
        conf_path = tmp_path / "rtklib.conf"
        
        build_rtklib_conf(
            out_path=conf_path,
            use_base=True,
            rtklib_settings={},
            station_coords=None,
        )
        
        conf_content = conf_path.read_text()
        conf_lines = {
            line.split("=")[0]: line.split("=")[1]
            for line in conf_content.strip().split("\n")
            if "=" in line
        }
        
        assert conf_lines.get("pos2-armode") == "continuous"

    def test_ar_mode_is_off_without_base(self, tmp_path):
        """Test AR mode is 'off' when no base station is used."""
        conf_path = tmp_path / "rtklib.conf"
        
        build_rtklib_conf(
            out_path=conf_path,
            use_base=False,
            rtklib_settings={},
            station_coords=None,
        )
        
        conf_content = conf_path.read_text()
        conf_lines = {
            line.split("=")[0]: line.split("=")[1]
            for line in conf_content.strip().split("\n")
            if "=" in line
        }
        
        assert conf_lines.get("pos2-armode") == "off"

    def test_base_station_coords_written_as_xyz(self, tmp_path):
        """Test base station coordinates are written in ECEF XYZ format."""
        conf_path = tmp_path / "rtklib.conf"
        
        station_coords = (1674416.8130, -5717888.7740, 2268832.6550)
        
        build_rtklib_conf(
            out_path=conf_path,
            use_base=True,
            rtklib_settings={},
            station_coords=station_coords,
        )
        
        conf_content = conf_path.read_text()
        conf_lines = {
            line.split("=")[0]: line.split("=")[1]
            for line in conf_content.strip().split("\n")
            if "=" in line
        }
        
        assert conf_lines.get("ant2-postype") == "xyz"
        assert "1674416.8130" in conf_lines.get("ant2-pos1", "")
        assert "-5717888.7740" in conf_lines.get("ant2-pos2", "")
        assert "2268832.6550" in conf_lines.get("ant2-pos3", "")

    def test_custom_settings_override_defaults(self, tmp_path):
        """Test that custom settings override defaults."""
        conf_path = tmp_path / "rtklib.conf"
        
        custom_settings = {
            "pos1-elmask": "20",
            "pos2-armode": "fix-and-hold",
        }
        
        build_rtklib_conf(
            out_path=conf_path,
            use_base=True,
            rtklib_settings=custom_settings,
            station_coords=None,
        )
        
        conf_content = conf_path.read_text()
        conf_lines = {
            line.split("=")[0]: line.split("=")[1]
            for line in conf_content.strip().split("\n")
            if "=" in line
        }
        
        assert conf_lines.get("pos1-elmask") == "20"
        assert conf_lines.get("pos2-armode") == "fix-and-hold"
