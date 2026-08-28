"""
Tests for BootstrapOrchestrator cache teardown methods.

Verifies that:
  - clean_pmpd() deletes pmpd_store.json, generated_prompts.json, AND policy/md_tree/
  - clean_rag()  deletes any directory matching *_faiss AND policy/md_tree/
  - The correct BootstrapState flags are reset in every case
  - Teardown logic does not crash if files are missing (idempotency)
"""
import os
import shutil
import re
from unittest.mock import MagicMock, patch
from bootstrap.orchestrator import BootstrapOrchestrator
from bootstrap.state_manager import BootstrapState
from infrastructure.api_key_manager import APIKeyManager
from infrastructure.config import PathConfig


def _make_orchestrator(tmp_path, extra_attrs: dict) -> BootstrapOrchestrator:
    """Helper: build an orchestrator with a custom PathConfig pointing at tmp_path."""
    key_mgr = MagicMock(spec=APIKeyManager)
    paths = PathConfig.default()
    object.__setattr__(paths, "bootstrap_archive_dir", str(tmp_path / "archive"))
    for attr, value in extra_attrs.items():
        object.__setattr__(paths, attr, str(value))
    orchestrator = BootstrapOrchestrator(key_manager=key_mgr, paths=paths)
    return orchestrator


class TestCleanPMPD:
    """clean_pmpd() must wipe pmpd_store, generated_prompts, AND md_tree."""

    def test_deletes_pmpd_db(self, tmp_path):
        pmpd_db = tmp_path / "pmpd_store.json"
        pmpd_db.touch()
        md_tree = tmp_path / "md_tree"
        md_tree.mkdir()

        orc = _make_orchestrator(tmp_path, {
            "pmpd_db_path": pmpd_db,
            "generated_prompts": tmp_path / "generated_prompts.json",
            "markdown_tree_dir": md_tree,
        })
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc.clean_pmpd()
        assert not pmpd_db.exists(), "pmpd_store.json should be deleted"

    def test_deletes_generated_prompts(self, tmp_path):
        prompts = tmp_path / "generated_prompts.json"
        prompts.touch()
        md_tree = tmp_path / "md_tree"
        md_tree.mkdir()

        orc = _make_orchestrator(tmp_path, {
            "pmpd_db_path": tmp_path / "pmpd_store.json",
            "generated_prompts": prompts,
            "markdown_tree_dir": md_tree,
        })
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc.clean_pmpd()
        assert not prompts.exists(), "generated_prompts.json should be deleted"

    def test_deletes_md_tree_in_pmpd_clean(self, tmp_path):
        md_tree = tmp_path / "md_tree"
        md_tree.mkdir()
        (md_tree / "section.md").touch()

        orc = _make_orchestrator(tmp_path, {
            "pmpd_db_path": tmp_path / "pmpd_store.json",
            "generated_prompts": tmp_path / "generated_prompts.json",
            "markdown_tree_dir": md_tree,
        })
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc.clean_pmpd()
        assert not md_tree.exists(), "md_tree should be deleted by clean_pmpd"

    def test_resets_ingested_flag_in_pmpd_clean(self, tmp_path):
        state = BootstrapState(ingested=True)
        orc = _make_orchestrator(tmp_path, {})
        orc._state_manager.load_state = MagicMock(return_value=state)
        orc.clean_pmpd()
        assert state.ingested is False

    def test_resets_pmpd_parsed_flag_in_pmpd_clean(self, tmp_path):
        state = BootstrapState(pmpd_parsed=True)
        orc = _make_orchestrator(tmp_path, {})
        orc._state_manager.load_state = MagicMock(return_value=state)
        orc.clean_pmpd()
        assert state.pmpd_parsed is False

    def test_resets_prompts_generated_flag_in_pmpd_clean(self, tmp_path):
        state = BootstrapState(prompts_generated=True)
        orc = _make_orchestrator(tmp_path, {})
        orc._state_manager.load_state = MagicMock(return_value=state)
        orc.clean_pmpd()
        assert state.prompts_generated is False

    def test_clean_pmpd_tolerates_missing_files(self, tmp_path):
        """Should not crash if files are already gone."""
        orc = _make_orchestrator(tmp_path, {
            "pmpd_db_path": tmp_path / "missing_db.json",
            "generated_prompts": tmp_path / "missing_prompts.json",
            "markdown_tree_dir": tmp_path / "missing_md",
        })
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc.clean_pmpd()  # Success if no raise

    def test_clean_pmpd_archives_existing_artifacts_before_delete(self, tmp_path):
        pmpd_db = tmp_path / "pmpd_store.json"
        pmpd_db.write_text("{}", encoding="utf-8")
        prompts = tmp_path / "generated_prompts.json"
        prompts.write_text("{}", encoding="utf-8")
        md_tree = tmp_path / "md_tree"
        md_tree.mkdir()
        (md_tree / "section.md").write_text("hello", encoding="utf-8")

        orc = _make_orchestrator(tmp_path, {
            "pmpd_db_path": pmpd_db,
            "generated_prompts": prompts,
            "markdown_tree_dir": md_tree,
        })
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        snapshot = orc.clean_pmpd()

        assert snapshot is not None
        archived = [artifact.artifact_type for artifact in snapshot.artifacts]
        assert "pmpd_store" in archived
        assert "generated_prompts" in archived
        assert "markdown_tree" in archived


class TestCleanRAG:
    """clean_rag() must wipe *_faiss directories AND md_tree/."""

    def test_deletes_globbed_faiss_indices(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        faiss_dir = tmp_path / "test_policy_faiss"
        faiss_dir.mkdir()
        
        orc = _make_orchestrator(tmp_path, {"markdown_tree_dir": tmp_path / "md"})
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc._state_manager.detect_policy_document = MagicMock(return_value=None)
        
        orc.clean_rag()
        assert not faiss_dir.exists()

    def test_deletes_specific_policy_index(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        doc_path = tmp_path / "Policy V1.pdf"
        doc_path.touch()
        # safe name for "Policy V1" is "policy_v1_faiss"
        specific_faiss = tmp_path / "policy_v1_faiss"
        specific_faiss.mkdir()

        orc = _make_orchestrator(tmp_path, {"markdown_tree_dir": tmp_path / "md"})
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc._state_manager.detect_policy_document = MagicMock(return_value=str(doc_path))

        orc.clean_rag()
        assert not specific_faiss.exists()

    def test_deletes_md_tree_in_rag_clean(self, tmp_path):
        md_tree = tmp_path / "md_tree"
        md_tree.mkdir()
        orc = _make_orchestrator(tmp_path, {"markdown_tree_dir": md_tree})
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc.clean_rag()
        assert not md_tree.exists()

    def test_resets_ingested_flag_in_rag_clean(self, tmp_path):
        state = BootstrapState(ingested=True)
        orc = _make_orchestrator(tmp_path, {})
        orc._state_manager.load_state = MagicMock(return_value=state)
        orc.clean_rag()
        assert state.ingested is False

    def test_resets_rag_tuned_flag_in_rag_clean(self, tmp_path):
        state = BootstrapState(rag_tuned=True)
        orc = _make_orchestrator(tmp_path, {})
        orc._state_manager.load_state = MagicMock(return_value=state)
        orc.clean_rag()
        assert state.rag_tuned is False

    def test_clean_rag_tolerates_missing_dirs(self, tmp_path):
        orc = _make_orchestrator(tmp_path, {"markdown_tree_dir": tmp_path / "missing_md"})
        orc._state_manager.load_state = MagicMock(return_value=BootstrapState())
        orc.clean_rag()  # Success if no raise

    def test_clean_all_archives_once_then_removes_outputs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pmpd_db = tmp_path / "pmpd_store.json"
        pmpd_db.write_text("{}", encoding="utf-8")
        prompts = tmp_path / "generated_prompts.json"
        prompts.write_text("{}", encoding="utf-8")
        md_tree = tmp_path / "md_tree"
        md_tree.mkdir()
        (md_tree / "section.md").write_text("hello", encoding="utf-8")
        faiss_dir = tmp_path / "policy_v1_faiss"
        faiss_dir.mkdir()

        orc = _make_orchestrator(tmp_path, {
            "pmpd_db_path": pmpd_db,
            "generated_prompts": prompts,
            "markdown_tree_dir": md_tree,
            "rag_config_file": tmp_path / "rag_config.json",
        })
        state = BootstrapState(ingested=True, rag_tuned=True, prompts_generated=True, pmpd_parsed=True)
        orc._state_manager.load_state = MagicMock(return_value=state)
        orc._state_manager.detect_policy_document = MagicMock(return_value=None)

        snapshot = orc.clean_all()

        assert snapshot is not None
        assert not pmpd_db.exists()
        assert not prompts.exists()
        assert not md_tree.exists()
        assert not faiss_dir.exists()
