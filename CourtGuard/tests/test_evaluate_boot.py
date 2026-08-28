"""
Boot tests for evaluate.py.

These tests actually call main() with various flags to ensure that the
scoping, imports, and CLI-to-Runtime handoff logic is sound. We mock
the key manager and orchestrator to avoid real network/API calls.
"""

import sys
from unittest.mock import MagicMock, patch
import pytest
from evaluate import main


def test_evaluate_main_clean_all_boots(monkeypatch):
    """Verify that --clean-all mode executes correctly without UnboundLocalError."""
    # 1. Mock CLI args
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--mode", "pmpd", "--clean-all"],
    )

    # 2. Mock Orchestrator and KeyManager to avoid real cleanup
    with patch("evaluate.BootstrapOrchestrator") as mock_orc_cls:
        with patch("evaluate.APIKeyManager") as mock_key_cls:
            mock_orc = mock_orc_cls.return_value
            
            # 3. Call main() -- it should exit(0) on success
            with pytest.raises(SystemExit) as excinfo:
                main()
            
            assert excinfo.value.code == 0
            assert mock_orc.clean_all.called


def test_evaluate_main_parse_only_boots(monkeypatch):
    """Verify that --parse-only reaches the key-selection stage without UnboundLocalError."""
    # 1. Mock CLI args (using we now have our sandbox, so we don't need a real dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--mode", "pmpd", "--parse-only"],
    )

    # 2. Mock the interactive parts of main()
    # We want to see if it reaches the KeyManager instantiation
    with patch("evaluate.APIKeyManager") as mock_key_cls:
        # Mock input() to avoid hanging the test
        monkeypatch.setattr("builtins.input", lambda _: "1")
        
        # Mock Orchestrator.run() to return a success dummy
        with patch("evaluate.BootstrapOrchestrator") as mock_orc_cls:
            mock_orc = mock_orc_cls.return_value
            mock_result = MagicMock()
            mock_result.success = True
            mock_orc.run.return_value = mock_result
            
            # 3. Call main() -- it should exit(0) after bootstrap because of --parse-only
            with pytest.raises(SystemExit) as excinfo:
                main()
            
            assert excinfo.value.code == 0
            assert mock_key_cls.called
            assert mock_orc.run.called
