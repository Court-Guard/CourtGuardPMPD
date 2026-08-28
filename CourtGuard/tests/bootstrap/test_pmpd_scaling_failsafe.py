import pytest
from core.exceptions import PMPDScalabilityError
from bootstrap.orchestrator import BootstrapOrchestrator
from infrastructure.api_key_manager import APIKeyManager
from core.pmpd import PMPD
from core.models import CategoryModule
from unittest.mock import MagicMock

class TestPMPDScalabilityFailsafe:
    def test_pmpd_mode_fails_when_categories_exceed_limit(self, tmp_path):
        key_mgr = MagicMock(spec=APIKeyManager)
        key_mgr.current_key = "test_key"
        
        db = PMPD()
        # Add 6 categories to exceed limit of 5
        for i in range(6):
            db.categories[f"ID_{i}"] = CategoryModule(
                category_id=f"ID_{i}", 
                name=f"Cat_{i}",
                general_rules="",
                exceptions=[],
                one_shots=[],
                citations=[]
            )
            
        from infrastructure.config import PathConfig
        paths = PathConfig.default()
        object.__setattr__(paths, 'markdown_tree_dir', str(tmp_path))
        
        orchestrator = BootstrapOrchestrator(key_manager=key_mgr, paths=paths)
        orchestrator._state_manager = MagicMock()
        orchestrator._state_manager.detect_policy_document.return_value = "dummy.pdf"
        
        # Override Stage 4 parsing completely so we just load our giant DB mock
        orchestrator._run_pmpd_parsing = MagicMock()
        orchestrator._load_pmpd_or_empty = MagicMock(return_value=db)
        
        # Mock compute_needs to enforce PMPD parsing checks
        orchestrator._compute_needs = MagicMock(return_value={
            "ingestion": False,
            "rag_tuning": False,
            "prompts": False,
            "pmpd": False  # Force _load_pmpd_or_empty
        })

        with pytest.raises(PMPDScalabilityError) as exc_info:
            orchestrator.run(eval_mode="pmpd", pmpd_k=5)
            
        # Ensure it halted because of the limit
        assert "Policy contains 6 categories (max configured limit is 5)" in str(exc_info.value)
        assert exc_info.value.requires_faiss is True

    def test_rag_mode_bypasses_limit(self, tmp_path):
        key_mgr = MagicMock(spec=APIKeyManager)
        key_mgr.current_key = "test_key"
        
        db = PMPD()
        for i in range(20):
            db.categories[f"ID_{i}"] = CategoryModule(
                category_id=f"ID_{i}", 
                name=f"Cat_{i}",
                general_rules="",
                exceptions=[],
                one_shots=[],
                citations=[]
            )
            
        from infrastructure.config import PathConfig
        paths = PathConfig.default()
        object.__setattr__(paths, 'markdown_tree_dir', str(tmp_path))
        
        orchestrator = BootstrapOrchestrator(key_manager=key_mgr, paths=paths)
        orchestrator._state_manager = MagicMock()
        orchestrator._state_manager.detect_policy_document.return_value = "dummy.pdf"
        
        orchestrator._run_pmpd_parsing = MagicMock()
        orchestrator._load_pmpd_or_empty = MagicMock(return_value=db)
        orchestrator._run_index_build = MagicMock(return_value=True)
        orchestrator._run_rag_tuning_or_load = MagicMock(return_value={"k": 5})
        
        orchestrator._compute_needs = MagicMock(return_value={
            "ingestion": False,
            "rag_tuning": False,
            "prompts": False,
            "pmpd": False
        })
        
        # Should gracefully return success without throwing PMPDScalabilityError
        result = orchestrator.run(eval_mode="rag", pmpd_k=5)
        assert result.success is True

    def test_parse_only_bypasses_scalability_check(self, tmp_path):
        """--parse-only must never raise PMPDScalabilityError regardless of category count."""
        key_mgr = MagicMock(spec=APIKeyManager)
        key_mgr.current_key = "test_key"

        db = PMPD()
        # Add 12 categories — well over the default limit of 5
        for i in range(12):
            db.categories[f"ID_{i}"] = CategoryModule(
                category_id=f"ID_{i}",
                name=f"Cat_{i}",
                general_rules="",
                exceptions=[],
                one_shots=[],
                citations=[]
            )

        from infrastructure.config import PathConfig
        paths = PathConfig.default()
        object.__setattr__(paths, 'markdown_tree_dir', str(tmp_path))

        orchestrator = BootstrapOrchestrator(key_manager=key_mgr, paths=paths)
        orchestrator._state_manager = MagicMock()
        orchestrator._state_manager.detect_policy_document.return_value = "dummy.pdf"

        orchestrator._run_pmpd_parsing = MagicMock()
        orchestrator._load_pmpd_or_empty = MagicMock(return_value=db)

        orchestrator._compute_needs = MagicMock(return_value={
            "ingestion": False,
            "rag_tuning": False,
            "prompts": False,
            "pmpd": False
        })

        # Must NOT raise — parse_only=True bypasses the check
        result = orchestrator.run(eval_mode="pmpd", pmpd_k=5, parse_only=True)
        assert result.success is True

    def test_skip_scalability_check_bypasses_limit_for_runtime_specific_modes(self, tmp_path):
        key_mgr = MagicMock(spec=APIKeyManager)
        key_mgr.current_key = "test_key"

        db = PMPD()
        for i in range(12):
            db.categories[f"ID_{i}"] = CategoryModule(
                category_id=f"ID_{i}",
                name=f"Cat_{i}",
                general_rules="",
                exceptions=[],
                one_shots=[],
                citations=[]
            )

        from infrastructure.config import PathConfig
        paths = PathConfig.default()
        object.__setattr__(paths, 'markdown_tree_dir', str(tmp_path))

        orchestrator = BootstrapOrchestrator(key_manager=key_mgr, paths=paths)
        orchestrator._state_manager = MagicMock()
        orchestrator._state_manager.detect_policy_document.return_value = "dummy.pdf"

        orchestrator._run_pmpd_parsing = MagicMock()
        orchestrator._load_pmpd_or_empty = MagicMock(return_value=db)

        orchestrator._compute_needs = MagicMock(return_value={
            "ingestion": False,
            "rag_tuning": False,
            "prompts": False,
            "pmpd": False
        })

        result = orchestrator.run(
            eval_mode="pmpd",
            pmpd_k=6,
            skip_pmpd_scalability_check=True,
        )
        assert result.success is True
