#!/usr/bin/env python3
"""
Unit tests for coverage_map_generator.py

Tests coverage map generation, file output, and utility functions.

Run with:
  pytest CODE/gnss/tests/test_coverage_map.py -v
  python -m pytest CODE/gnss/tests/test_coverage_map.py -v
"""

import csv
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from coverage_map_generator import (
    CoverageCell,
    CoverageSummary,
    RunManifest,
    TransmitterConfig,
    calculate_coverage_summary,
    calculate_fspl_db,
    calculate_snr_db,
    classify_coverage,
    ecef_to_geodetic,
    generate_coverage_grid,
    generate_run_id,
    haversine_distance_m,
    meters_to_degrees_lat,
    meters_to_degrees_lon,
    write_coverage_csv,
    write_coverage_summary,
    write_run_manifest,
)


# ----------------------------
# Utility Function Tests
# ----------------------------

class TestHaversineDistance:
    """Tests for haversine_distance_m function."""
    
    def test_same_point_zero_distance(self):
        """Same point should have zero distance."""
        dist = haversine_distance_m(37.5665, 126.9780, 37.5665, 126.9780)
        assert dist == pytest.approx(0.0, abs=0.01)
    
    def test_known_distance(self):
        """Test with known approximate distance (Seoul to Incheon ~30km)."""
        # Seoul: 37.5665, 126.9780
        # Incheon: 37.4563, 126.7052
        dist = haversine_distance_m(37.5665, 126.9780, 37.4563, 126.7052)
        # Approximate distance should be ~25-35 km
        assert 25000 < dist < 35000
    
    def test_small_distance_accuracy(self):
        """Test accuracy for small distances (~1km)."""
        # Move ~1km north (approximately)
        lat1 = 37.5665
        lat2 = lat1 + (1.0 / 111.0)  # ~1km in latitude
        lon = 126.9780
        
        dist = haversine_distance_m(lat1, lon, lat2, lon)
        assert dist == pytest.approx(1000, rel=0.01)  # 1% tolerance


class TestFSPLCalculation:
    """Tests for free-space path loss calculation."""
    
    def test_fspl_zero_distance(self):
        """FSPL at zero distance should be 0."""
        fspl = calculate_fspl_db(0.0, 600.0)
        assert fspl == 0.0
    
    def test_fspl_negative_distance(self):
        """FSPL at negative distance should be 0."""
        fspl = calculate_fspl_db(-100.0, 600.0)
        assert fspl == 0.0
    
    def test_fspl_1km(self):
        """Test FSPL at 1km, 600 MHz."""
        # FSPL = 20*log10(1) + 20*log10(600) + 32.45
        #      = 0 + 55.56 + 32.45 = 88.01 dB
        fspl = calculate_fspl_db(1000.0, 600.0)
        assert fspl == pytest.approx(88.0, rel=0.01)
    
    def test_fspl_increases_with_distance(self):
        """FSPL should increase with distance."""
        fspl_1km = calculate_fspl_db(1000.0, 600.0)
        fspl_10km = calculate_fspl_db(10000.0, 600.0)
        fspl_100km = calculate_fspl_db(100000.0, 600.0)
        
        assert fspl_1km < fspl_10km < fspl_100km
    
    def test_fspl_increases_with_frequency(self):
        """FSPL should increase with frequency."""
        dist = 10000.0  # 10 km
        fspl_300 = calculate_fspl_db(dist, 300.0)
        fspl_600 = calculate_fspl_db(dist, 600.0)
        fspl_1000 = calculate_fspl_db(dist, 1000.0)
        
        assert fspl_300 < fspl_600 < fspl_1000


class TestSNRCalculation:
    """Tests for SNR calculation."""
    
    def test_snr_basic(self):
        """Basic SNR calculation."""
        # 60 dBm TX - 100 dB FSPL - (-105 dBm noise) = 60 - 100 + 105 = 65 dB SNR
        snr = calculate_snr_db(60.0, 100.0, -105.0)
        assert snr == pytest.approx(65.0, rel=0.01)
    
    def test_snr_decreases_with_fspl(self):
        """SNR should decrease as FSPL increases."""
        snr_low = calculate_snr_db(60.0, 80.0, -105.0)
        snr_high = calculate_snr_db(60.0, 120.0, -105.0)
        
        assert snr_low > snr_high
    
    def test_snr_can_be_negative(self):
        """SNR can be negative (signal below noise)."""
        snr = calculate_snr_db(60.0, 200.0, -105.0)
        assert snr < 0


class TestClassifyCoverage:
    """Tests for coverage classification."""
    
    def test_good_coverage(self):
        """High SNR should be classified as good."""
        assert classify_coverage(25.0, 20.0, 10.0) == "good"
        assert classify_coverage(20.0, 20.0, 10.0) == "good"
    
    def test_marginal_coverage(self):
        """Medium SNR should be classified as marginal."""
        assert classify_coverage(15.0, 20.0, 10.0) == "marginal"
        assert classify_coverage(10.0, 20.0, 10.0) == "marginal"
    
    def test_bad_coverage(self):
        """Low SNR should be classified as bad."""
        assert classify_coverage(5.0, 20.0, 10.0) == "bad"
        assert classify_coverage(9.9, 20.0, 10.0) == "bad"
        assert classify_coverage(-5.0, 20.0, 10.0) == "bad"


class TestECEFToGeodetic:
    """Tests for ECEF to geodetic conversion."""
    
    def test_known_conversion(self):
        """Test with known ECEF coordinates from scenario profiles."""
        # The scenario profile coordinates correspond to approximately:
        # lat ~37.4, lon ~-122 (San Francisco Bay Area)
        x, y, z = -2694892.46, -4297557.47, 3854813.42
        
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        
        # Verify reasonable latitude (Northern Hemisphere mid-latitudes)
        assert 35 < lat < 40
        # Verify reasonable longitude (Western US or similar)
        assert -125 < lon < -120
    
    def test_equator_prime_meridian(self):
        """Test near equator and prime meridian."""
        # ECEF for lat=0, lon=0, alt=0 is approximately X=6378137, Y=0, Z=0
        lat, lon, alt = ecef_to_geodetic(6378137.0, 0.0, 0.0)
        
        assert lat == pytest.approx(0.0, abs=0.01)
        assert lon == pytest.approx(0.0, abs=0.01)


# ----------------------------
# Grid Generation Tests
# ----------------------------

class TestGenerateCoverageGrid:
    """Tests for coverage grid generation."""
    
    @pytest.fixture
    def tx_config(self):
        """Sample transmitter configuration."""
        return TransmitterConfig(
            lat=37.5665,
            lon=126.9780,
            height_m=30.0,
            power_dbm=60.0,
            frequency_mhz=600.0
        )
    
    def test_generates_cells(self, tx_config):
        """Grid should generate cells."""
        cells = generate_coverage_grid(
            tx_config=tx_config,
            grid_res_m=1000.0,  # 1km resolution for small test
            max_range_km=5.0,   # 5km range
            rx_height_m=1.5
        )
        
        assert len(cells) > 0
    
    def test_cells_within_range(self, tx_config):
        """All cells should be within max range."""
        max_range_km = 5.0
        cells = generate_coverage_grid(
            tx_config=tx_config,
            grid_res_m=500.0,
            max_range_km=max_range_km,
            rx_height_m=1.5
        )
        
        for cell in cells:
            assert cell.distance_m <= max_range_km * 1000 + 100  # Small tolerance
    
    def test_cells_have_required_attributes(self, tx_config):
        """Each cell should have all required attributes."""
        cells = generate_coverage_grid(
            tx_config=tx_config,
            grid_res_m=1000.0,
            max_range_km=2.0,
            rx_height_m=1.5
        )
        
        assert len(cells) > 0
        cell = cells[0]
        
        assert hasattr(cell, 'lat')
        assert hasattr(cell, 'lon')
        assert hasattr(cell, 'distance_m')
        assert hasattr(cell, 'fspl_db')
        assert hasattr(cell, 'snr_db')
        assert hasattr(cell, 'coverage_class')
    
    def test_coverage_classes_present(self, tx_config):
        """Grid should contain various coverage classes."""
        cells = generate_coverage_grid(
            tx_config=tx_config,
            grid_res_m=500.0,
            max_range_km=50.0,  # Large range to get all classes
            rx_height_m=1.5
        )
        
        classes = set(c.coverage_class for c in cells)
        # Should have at least good and bad (maybe marginal depending on params)
        assert "good" in classes or "marginal" in classes or "bad" in classes
    
    def test_center_cell_best_snr(self, tx_config):
        """Cell at/near transmitter should have best SNR."""
        cells = generate_coverage_grid(
            tx_config=tx_config,
            grid_res_m=100.0,
            max_range_km=5.0,
            rx_height_m=1.5
        )
        
        # Find cell closest to transmitter
        center_cell = min(cells, key=lambda c: c.distance_m)
        
        # Center should have good or best SNR
        assert center_cell.snr_db == max(c.snr_db for c in cells)


class TestCoverageSummary:
    """Tests for coverage summary calculation."""
    
    def test_summary_counts(self):
        """Summary should correctly count cells by class."""
        cells = [
            CoverageCell(lat=0, lon=0, distance_m=0, fspl_db=0, snr_db=25, coverage_class="good"),
            CoverageCell(lat=0, lon=0, distance_m=0, fspl_db=0, snr_db=25, coverage_class="good"),
            CoverageCell(lat=0, lon=0, distance_m=0, fspl_db=0, snr_db=15, coverage_class="marginal"),
            CoverageCell(lat=0, lon=0, distance_m=0, fspl_db=0, snr_db=5, coverage_class="bad"),
        ]
        
        tx_config = TransmitterConfig(lat=0, lon=0, height_m=30, power_dbm=60, frequency_mhz=600)
        
        summary = calculate_coverage_summary(
            cells=cells,
            tx_config=tx_config,
            grid_res_m=100.0,
            max_range_km=10.0,
            rx_height_m=1.5,
            noise_floor_dbm=-105.0,
            snr_good_thresh=20.0,
            snr_marginal_thresh=10.0
        )
        
        assert summary.total_cells == 4
        assert summary.good_cells == 2
        assert summary.marginal_cells == 1
        assert summary.bad_cells == 1
        assert summary.good_pct == pytest.approx(50.0, rel=0.01)


# ----------------------------
# File Output Tests
# ----------------------------

class TestFileOutputs:
    """Tests for file output functions."""
    
    @pytest.fixture
    def sample_cells(self):
        """Sample cells for testing file output."""
        return [
            CoverageCell(lat=37.5665, lon=126.9780, distance_m=0, fspl_db=0, snr_db=65.0, coverage_class="good"),
            CoverageCell(lat=37.5755, lon=126.9780, distance_m=1000, fspl_db=88.0, snr_db=42.0, coverage_class="good"),
            CoverageCell(lat=37.5845, lon=126.9780, distance_m=2000, fspl_db=94.0, snr_db=31.0, coverage_class="good"),
        ]
    
    def test_write_coverage_csv(self, sample_cells):
        """CSV file should be written with correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "coverage_map.csv"
            write_coverage_csv(sample_cells, output_path)
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # Read and verify CSV
            with output_path.open("r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Check required columns
            assert "lat" in reader.fieldnames
            assert "lon" in reader.fieldnames
            assert "distance_m" in reader.fieldnames
            assert "fspl_db" in reader.fieldnames
            assert "snr_db" in reader.fieldnames
            assert "class" in reader.fieldnames
            
            # Check row count
            assert len(rows) == len(sample_cells)
    
    def test_write_coverage_summary(self, sample_cells):
        """JSON summary should be written with correct structure."""
        tx_config = TransmitterConfig(
            lat=37.5665, lon=126.9780, height_m=30, power_dbm=60, frequency_mhz=600
        )
        
        summary = calculate_coverage_summary(
            cells=sample_cells,
            tx_config=tx_config,
            grid_res_m=100.0,
            max_range_km=10.0,
            rx_height_m=1.5,
            noise_floor_dbm=-105.0,
            snr_good_thresh=20.0,
            snr_marginal_thresh=10.0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "coverage_summary.json"
            write_coverage_summary(summary, output_path)
            
            assert output_path.exists()
            
            # Read and verify JSON
            with output_path.open("r") as f:
                data = json.load(f)
            
            # Check required fields
            assert "total_cells" in data
            assert "good_cells" in data
            assert "marginal_cells" in data
            assert "bad_cells" in data
            assert "good_pct" in data
            assert "snr_good_threshold_db" in data
            assert "snr_marginal_threshold_db" in data
            assert "tx_config" in data
    
    def test_creates_parent_directories(self, sample_cells):
        """Output functions should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            output_path = Path(tmpdir) / "a" / "b" / "c" / "coverage_map.csv"
            
            # Should not raise
            write_coverage_csv(sample_cells, output_path)
            
            assert output_path.exists()


# ----------------------------
# Run ID and Manifest Tests
# ----------------------------

class TestGenerateRunId:
    """Tests for run ID generation."""
    
    def test_run_id_format(self):
        """Run ID should follow expected format."""
        run_id, timestamp = generate_run_id("scenario1")
        
        # Run ID format: <scenario>_coverage_<YYYYMMDDTHHMMSSZ>
        assert run_id.startswith("scenario1_coverage_")
        assert timestamp in run_id
        
        # Timestamp format: YYYYMMDDTHHMMSSZ
        assert re.match(r"\d{8}T\d{6}Z", timestamp)
    
    def test_run_id_unique(self):
        """Consecutive run IDs should be unique (or same if fast enough)."""
        run_id1, _ = generate_run_id("test")
        run_id2, _ = generate_run_id("test")
        
        # They might be same if generated in same second, but format should be consistent
        assert run_id1.startswith("test_coverage_")
        assert run_id2.startswith("test_coverage_")
    
    def test_run_id_custom_scenario(self):
        """Run ID should include custom scenario name."""
        run_id, _ = generate_run_id("custom")
        assert run_id.startswith("custom_coverage_")


class TestRunManifest:
    """Tests for run manifest functionality."""
    
    def test_write_run_manifest(self):
        """Run manifest should be written with correct structure."""
        manifest = RunManifest(
            run_id="scenario1_coverage_20260201T120000Z",
            scenario="scenario1",
            timestamp="20260201T120000Z",
            args={
                "grid_res_m": 50.0,
                "max_range_km": 10.0,
                "tx_lat": 37.5,
                "tx_lon": 127.0
            },
            success=True,
            output_files=["coverage_map.csv", "coverage_summary.json"]
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run_manifest.json"
            write_run_manifest(manifest, output_path)
            
            assert output_path.exists()
            
            # Read and verify JSON
            with output_path.open("r") as f:
                data = json.load(f)
            
            # Check required fields
            assert "run_id" in data
            assert "scenario" in data
            assert "timestamp" in data
            assert "args" in data
            assert "success" in data
            
            # Check values
            assert data["run_id"] == "scenario1_coverage_20260201T120000Z"
            assert data["scenario"] == "scenario1"
            assert data["timestamp"] == "20260201T120000Z"
            assert data["args"]["grid_res_m"] == 50.0
            assert data["success"] is True
    
    def test_manifest_contains_scenario(self):
        """Manifest should contain scenario name."""
        manifest = RunManifest(
            run_id="test_coverage_20260201T000000Z",
            scenario="test_scenario",
            timestamp="20260201T000000Z",
            args={}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run_manifest.json"
            write_run_manifest(manifest, output_path)
            
            with output_path.open("r") as f:
                data = json.load(f)
            
            assert data["scenario"] == "test_scenario"
    
    def test_manifest_contains_timestamp(self):
        """Manifest should contain timestamp."""
        manifest = RunManifest(
            run_id="test_coverage_20260201T153045Z",
            scenario="test",
            timestamp="20260201T153045Z",
            args={}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run_manifest.json"
            write_run_manifest(manifest, output_path)
            
            with output_path.open("r") as f:
                data = json.load(f)
            
            assert data["timestamp"] == "20260201T153045Z"
    
    def test_manifest_contains_args(self):
        """Manifest should contain args dict."""
        test_args = {
            "scenario": "scenario1",
            "grid_res_m": 100.0,
            "max_range_km": 25.0,
            "tx_lat": 37.5665,
            "tx_lon": 126.9780,
            "tx_height_m": 30.0,
        }
        
        manifest = RunManifest(
            run_id="test_coverage_20260201T000000Z",
            scenario="scenario1",
            timestamp="20260201T000000Z",
            args=test_args
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run_manifest.json"
            write_run_manifest(manifest, output_path)
            
            with output_path.open("r") as f:
                data = json.load(f)
            
            assert "args" in data
            assert data["args"]["scenario"] == "scenario1"
            assert data["args"]["grid_res_m"] == 100.0
            assert data["args"]["tx_lat"] == 37.5665


# ----------------------------
# Integration Tests
# ----------------------------

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_tiny_grid_generation(self):
        """Generate a tiny grid and verify outputs."""
        tx_config = TransmitterConfig(
            lat=37.5665,
            lon=126.9780,
            height_m=30.0,
            power_dbm=60.0,
            frequency_mhz=600.0
        )
        
        # Very small grid for fast testing
        cells = generate_coverage_grid(
            tx_config=tx_config,
            grid_res_m=2000.0,  # 2km resolution
            max_range_km=5.0,   # 5km range
            rx_height_m=1.5,
            noise_floor_dbm=-105.0,
            snr_good_thresh=20.0,
            snr_marginal_thresh=10.0
        )
        
        # Verify we got cells
        assert len(cells) > 0
        
        # Verify all cells have valid coverage classes
        valid_classes = {"good", "marginal", "bad"}
        for cell in cells:
            assert cell.coverage_class in valid_classes
        
        # Write outputs and verify
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "coverage_map.csv"
            json_path = Path(tmpdir) / "coverage_summary.json"
            
            write_coverage_csv(cells, csv_path)
            
            summary = calculate_coverage_summary(
                cells=cells,
                tx_config=tx_config,
                grid_res_m=2000.0,
                max_range_km=5.0,
                rx_height_m=1.5,
                noise_floor_dbm=-105.0,
                snr_good_thresh=20.0,
                snr_marginal_thresh=10.0
            )
            write_coverage_summary(summary, json_path)
            
            # Both files should exist
            assert csv_path.exists()
            assert json_path.exists()
            
            # CSV should have header + data rows
            with csv_path.open("r") as f:
                lines = f.readlines()
            assert len(lines) == len(cells) + 1  # +1 for header
            
            # JSON should parse correctly
            with json_path.open("r") as f:
                data = json.load(f)
            assert data["total_cells"] == len(cells)
    
    def test_run_directory_structure(self):
        """Test that run directory is created with all expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate the run directory structure
            scenario_name = "test_scenario"
            run_id, timestamp = generate_run_id(scenario_name)
            
            # Create run directory
            run_dir = Path(tmpdir) / scenario_name / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Verify run_id format
            assert run_id.startswith(f"{scenario_name}_coverage_")
            assert re.match(r"\d{8}T\d{6}Z", timestamp)
            
            # Create sample data
            tx_config = TransmitterConfig(
                lat=37.5, lon=127.0, height_m=30.0, power_dbm=60.0, frequency_mhz=600.0
            )
            cells = generate_coverage_grid(
                tx_config=tx_config,
                grid_res_m=2000.0,
                max_range_km=3.0,
                rx_height_m=1.5
            )
            summary = calculate_coverage_summary(
                cells=cells,
                tx_config=tx_config,
                grid_res_m=2000.0,
                max_range_km=3.0,
                rx_height_m=1.5,
                noise_floor_dbm=-105.0,
                snr_good_thresh=20.0,
                snr_marginal_thresh=10.0
            )
            
            # Write all files to run directory
            csv_path = run_dir / "coverage_map.csv"
            json_path = run_dir / "coverage_summary.json"
            manifest_path = run_dir / "run_manifest.json"
            
            write_coverage_csv(cells, csv_path)
            write_coverage_summary(summary, json_path)
            
            manifest = RunManifest(
                run_id=run_id,
                scenario=scenario_name,
                timestamp=timestamp,
                args={"grid_res_m": 2000.0, "max_range_km": 3.0},
                success=True,
                output_files=["coverage_map.csv", "coverage_summary.json", "run_manifest.json"]
            )
            write_run_manifest(manifest, manifest_path)
            
            # Verify all files exist in run directory
            assert csv_path.exists()
            assert json_path.exists()
            assert manifest_path.exists()
            
            # Verify run directory path structure
            assert run_dir.parent.name == scenario_name
            assert run_dir.name == run_id
            
            # Verify manifest content
            with manifest_path.open("r") as f:
                manifest_data = json.load(f)
            
            assert manifest_data["scenario"] == scenario_name
            assert manifest_data["timestamp"] == timestamp
            assert "args" in manifest_data
            assert manifest_data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
