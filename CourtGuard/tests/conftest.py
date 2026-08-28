"""Test-wide dependency stubs for offline collection."""

from __future__ import annotations

import os
import sys
import types
import pytest
from dataclasses import fields
from infrastructure.config import PathConfig, ModelConfig


# ---------------------------------------------------------------------------
# OpenAI Stub (Existing)
# ---------------------------------------------------------------------------

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - simple import stub
        def __init__(self, *args, **kwargs) -> None:
            self.api_key = kwargs.get("api_key", "")

    openai_stub.OpenAI = OpenAI
    sys.modules["openai"] = openai_stub


# ---------------------------------------------------------------------------
# Global Sandbox: PathConfig Redirection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def sandbox_paths(tmp_path_factory, monkeypatch_session):
    """
    PROTECTIVE SANDBOX:
    Redirects all default PathConfig attributes to a temporary directory
    during any test run. This prevents tests from cluttering the project root
    with bootstrap_stats.json, .bootstrap_state.json, etc.
    """
    tmp_root = tmp_path_factory.mktemp("courtguard_test_sandbox")
    
    # Pre-create standard subdirs in the sandbox
    (tmp_root / "policy").mkdir()
    (tmp_root / "data" / "results").mkdir(parents=True)

    # 1. Build a fully-isolated PathConfig
    sandbox_kwargs = {}
    for f in fields(PathConfig):
        # Join the default filename (e.g. "pmpd_store.json") with our temp root
        sandbox_kwargs[f.name] = str(tmp_root / f.default)
    
    sandboxed_cfg = PathConfig(**sandbox_kwargs)

    # 2. Monkeypatch the .default() classmethod
    # Using lambda to ensure it always returns a FRESH isolated config
    monkeypatch_session.setattr(PathConfig, "default", lambda: sandboxed_cfg)
    
    # Also sandbox ModelConfig to avoid accidentally reading local .env during unit tests
    _test_cfg = ModelConfig(
        bootstrap_model="test/bootstrap-model",
        debate_model="test/debate-model",
    )
    monkeypatch_session.setattr(ModelConfig, "default", lambda: _test_cfg)
    
    # Forcefully sandbox os.environ so classes that bypass ModelConfig (like PolicyDebate)
    # never crash when looking for models, ensuring tests are 100% offline-safe globally.
    monkeypatch_session.setenv("COURTGUARD_DEBATE_MODEL", "offline-debate-model")
    monkeypatch_session.setenv("COURTGUARD_BOOTSTRAP_MODEL", "offline-bootstrap-model")

    return sandboxed_cfg


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch helper."""
    from _pytest.monkeypatch import MonkeyPatch
    m = MonkeyPatch()
    yield m
    m.undo()

