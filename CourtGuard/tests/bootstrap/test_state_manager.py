"""
Tests for bootstrap/state_manager.py

Covers:
  - BootstrapState.reset()
  - BootstrapState.to_dict() / from_dict() roundtrip
  - BootstrapStateManager.load_state() — missing file, corrupt file, valid
  - BootstrapStateManager.save_state() / load_state() roundtrip
  - BootstrapStateManager.load_rag_config() / save_rag_config() roundtrip
  - BootstrapStateManager.detect_policy_pdf()
"""

import os

import pytest

from bootstrap.state_manager import BootstrapState, BootstrapStateManager
from infrastructure.config import PathConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path) -> PathConfig:
    """PathConfig pointing entirely into a temp directory."""
    policy_dir = str(tmp_path / "policy")
    os.makedirs(policy_dir, exist_ok=True)
    return PathConfig(
        policy_input_dir=policy_dir,
        markdown_tree_dir=str(tmp_path / "policy" / "md_tree"),
        rag_config_file=str(tmp_path / "rag_config.json"),
        generated_prompts=str(tmp_path / "generated_prompts.json"),
        bootstrap_state=str(tmp_path / ".bootstrap_state.json"),
        pmpd_db_path=str(tmp_path / "pmpd_store.json"),
        api_keys_file=str(tmp_path / "api_keys.txt"),
    )


@pytest.fixture
def manager(paths) -> BootstrapStateManager:
    return BootstrapStateManager(paths)


# ---------------------------------------------------------------------------
# BootstrapState
# ---------------------------------------------------------------------------


class TestBootstrapState:
    def test_default_all_false(self):
        s = BootstrapState()
        assert s.ingested is False
        assert s.rag_tuned is False
        assert s.prompts_generated is False
        assert s.pmpd_parsed is False
        assert s.policy_file is None

    def test_reset_clears_all(self):
        s = BootstrapState(
            ingested=True,
            rag_tuned=True,
            prompts_generated=True,
            pmpd_parsed=True,
            policy_file="/some/path.pdf",
        )
        s.reset()
        assert s.ingested is False
        assert s.policy_file is None

    def test_roundtrip_to_from_dict(self):
        s = BootstrapState(
            ingested=True,
            rag_tuned=False,
            prompts_generated=True,
            pmpd_parsed=False,
            policy_file="/tmp/policy.pdf",
        )
        d = s.to_dict()
        back = BootstrapState.from_dict(d)
        assert back.ingested is True
        assert back.rag_tuned is False
        assert back.prompts_generated is True
        assert back.policy_file == "/tmp/policy.pdf"

    def test_from_dict_handles_missing_keys(self):
        """Partial dicts (e.g. old format missing pmpd_parsed) should not crash."""
        s = BootstrapState.from_dict({"ingested": True})
        assert s.ingested is True
        assert s.pmpd_parsed is False


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStateManagerPersistence:
    def test_load_missing_file_returns_default(self, manager):
        state = manager.load_state()
        assert state.ingested is False

    def test_save_and_load_roundtrip(self, manager):
        state = BootstrapState()
        state.ingested = True
        state.policy_file = "/path/to/policy.pdf"
        manager.save_state(state)

        loaded = manager.load_state()
        assert loaded.ingested is True
        assert loaded.policy_file == "/path/to/policy.pdf"

    def test_corrupt_file_returns_default(self, manager, paths):
        with open(paths.bootstrap_state, "w") as f:
            f.write("not valid json {{{")
        state = manager.load_state()
        assert state.ingested is False

    def test_all_fields_persisted(self, manager):
        state = BootstrapState(
            ingested=True,
            rag_tuned=True,
            prompts_generated=True,
            pmpd_parsed=True,
            policy_file="/p/policy.pdf",
        )
        manager.save_state(state)
        loaded = manager.load_state()
        assert loaded.rag_tuned is True
        assert loaded.prompts_generated is True
        assert loaded.pmpd_parsed is True


# ---------------------------------------------------------------------------
# RAG config persistence
# ---------------------------------------------------------------------------


class TestRAGConfigPersistence:
    def test_load_missing_returns_none(self, manager):
        assert manager.load_rag_config() is None

    def test_save_and_load_roundtrip(self, manager):
        config = {
            "chunk_size": 1024,
            "chunk_overlap": 256,
            "k": 5,
            "rationale": "Dense legal prose.",
        }
        manager.save_rag_config(config)
        loaded = manager.load_rag_config()
        assert loaded["chunk_size"] == 1024
        assert loaded["k"] == 5
        assert loaded["rationale"] == "Dense legal prose."


# ---------------------------------------------------------------------------
# Document detection
# ---------------------------------------------------------------------------


class TestDetectPolicyDocument:
    def test_no_document_returns_none(self, manager):
        assert manager.detect_policy_document() is None

    def test_detects_document_in_directory(self, manager, paths):
        doc_path = os.path.join(paths.policy_input_dir, "policy.docx")
        open(doc_path, "w").close()  # create empty file
        detected = manager.detect_policy_document()
        assert detected == doc_path

    def test_returns_first_alphabetically(self, manager, paths):
        for name in ("z_policy.pdf", "a_policy.docx", "m_policy.md"):
            open(os.path.join(paths.policy_input_dir, name), "w").close()
        detected = manager.detect_policy_document()
        assert os.path.basename(detected) == "a_policy.docx"

    def test_ignores_unsupported_files(self, manager, paths):
        open(os.path.join(paths.policy_input_dir, "readme.txt"), "w").close()
        assert manager.detect_policy_document() is None
