"""
Tests for rag/config.py

Covers:
  - ParameterGrid.snap() to nearest allowed value
  - ParameterGrid.validate_and_snap() with in-grid and out-of-grid values
  - RAGConfig.from_tuner_dict()
  - RAGConfig.default() values
"""

import pytest

from rag.config import ParameterGrid, RAGConfig


@pytest.fixture
def grid() -> ParameterGrid:
    return ParameterGrid.default()


class TestParameterGridSnap:
    def test_snap_exact_match(self, grid):
        assert grid.snap(1024, grid.chunk_sizes) == 1024

    def test_snap_rounds_down(self, grid):
        # 600 is between 512 and 768 — closer to 512
        assert grid.snap(600, grid.chunk_sizes) == 512

    def test_snap_rounds_up(self, grid):
        # 700 is between 512 and 768 — closer to 768
        assert grid.snap(700, grid.chunk_sizes) == 768

    def test_snap_exact_boundary(self, grid):
        assert grid.snap(512, grid.chunk_sizes) == 512
        assert grid.snap(768, grid.chunk_sizes) == 768

    def test_snap_overlap_ratio(self, grid):
        # 0.15 is between 0.10 and 0.20 — equidistant, min picks first
        result = grid.snap(0.15, grid.overlap_ratios)
        assert result in grid.overlap_ratios

    def test_snap_k_value(self, grid):
        assert grid.snap(4, grid.k_values) == 4
        assert grid.snap(7, grid.k_values) == 6 or grid.snap(7, grid.k_values) == 8


class TestParameterGridValidateAndSnap:
    def test_valid_config_unchanged(self, grid):
        config = {
            "chunk_size": 1024,
            "overlap_ratio": 0.20,
            "k": 5,
            "rationale": "Dense prose.",
        }
        result = grid.validate_and_snap(config)
        assert result.chunk_size == 1024
        assert result.default_k == 5
        assert result.rationale == "Dense prose."

    def test_out_of_grid_chunk_size_snapped(self, grid):
        config = {"chunk_size": 900, "overlap_ratio": 0.20, "k": 5}
        result = grid.validate_and_snap(config)
        assert result.chunk_size in grid.chunk_sizes

    def test_out_of_grid_k_snapped(self, grid):
        config = {"chunk_size": 1024, "overlap_ratio": 0.20, "k": 7}
        result = grid.validate_and_snap(config)
        assert result.default_k in grid.k_values

    def test_chunk_overlap_computed_from_ratio(self, grid):
        config = {"chunk_size": 1024, "overlap_ratio": 0.20, "k": 5}
        result = grid.validate_and_snap(config)
        expected_overlap = round(1024 * 0.20)
        assert result.chunk_overlap == expected_overlap

    def test_missing_keys_use_defaults(self, grid):
        result = grid.validate_and_snap({})
        assert result.chunk_size in grid.chunk_sizes
        assert result.default_k in grid.k_values


class TestRAGConfigFromTunerDict:
    def test_from_tuner_dict_maps_correctly(self):
        d = {
            "chunk_size": 768,
            "chunk_overlap": 150,
            "k": 4,
            "rationale": "Short paragraphs.",
        }
        config = RAGConfig.from_tuner_dict(d)
        assert config.chunk_size == 768
        assert config.chunk_overlap == 150
        assert config.default_k == 4
        assert config.rationale == "Short paragraphs."

    def test_from_tuner_dict_missing_keys_use_defaults(self):
        config = RAGConfig.from_tuner_dict({})
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 256
        assert config.default_k == 5


class TestRAGConfigDefault:
    def test_default_values(self):
        config = RAGConfig.default()
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 256
        assert config.default_k == 5
        assert "all-mpnet-base-v2" in config.embeddings_model

    def test_frozen(self):
        config = RAGConfig.default()
        with pytest.raises(Exception):
            config.chunk_size = 999  # type: ignore[misc]
