"""Offline CLI smoke tests for PMPD argument parsing."""

import sys
import pytest
from evaluation.cli import parse_eval_args


def test_pmpd_cli_accepts_indexes(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--data_file", "data/datasets/xstest_180.json",
            "--indexes", "1",
        ],
    )
    args = parse_eval_args()

    assert args.mode == "pmpd"
    assert args.indexes == "1"


def test_pmpd_cli_accepts_custom_input_fields(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--data_file", "data/datasets/vandalism.json",
            "--indexes", "0-5",
            "--input-fields", "oldtext,newtext,diff",
            "--max-rounds", "3",
            "--inspect-pmpd",
        ],
    )
    args = parse_eval_args()

    assert args.input_fields == "oldtext,newtext,diff"
    assert args.max_rounds == 3
    assert args.inspect_pmpd is True


def test_pmpd_cli_accepts_output_label_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--data_file", "data/datasets/vandalism.json",
            "--indexes", "1",
            "--output-labels", "allow,review,block",
            "--default-output-label", "allow",
            "--error-output-label", "block",
        ],
    )
    args = parse_eval_args()

    assert args.output_labels == "allow,review,block"
    assert args.default_output_label == "allow"
    assert args.error_output_label == "block"


def test_cli_accepts_keys_file_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--data_file", "data/datasets/xstest_180.json",
            "--indexes", "0",
            "--keys-file", "custom_keys.txt",
        ],
    )
    args = parse_eval_args()
    assert args.keys_file == "custom_keys.txt"


def test_cli_has_keys_file_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--data_file", "data/datasets/xstest_180.json",
            "--indexes", "0",
        ],
    )
    args = parse_eval_args()
    assert hasattr(args, "keys_file")
    assert args.keys_file == "api_keys.txt"


def test_cli_allows_parse_only_without_data_file(monkeypatch):
    """Admin commands like --parse-only should not require a dataset."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--parse-only",
        ],
    )
    args = parse_eval_args()
    assert args.parse_only is True
    assert args.data_file is None


def test_cli_allows_force_bootstrap_without_data_file(monkeypatch):
    """Admin commands like --force-bootstrap should not require a dataset."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--force-bootstrap",
        ],
    )
    args = parse_eval_args()
    assert args.force_bootstrap is True
    assert args.data_file is None


def test_cli_rejects_missing_index_selection(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--mode", "pmpd",
            "--data_file", "data/datasets/xstest_180.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_eval_args()
