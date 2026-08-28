"""
Bootstrap artifact archiver.

Creates timestamped snapshots of current bootstrap artifacts before cleanup or
re-bootstrap so PMPD / RAG state is never deleted without an archive copy.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime

from infrastructure.config import PathConfig


@dataclass
class ArchivedArtifact:
    source: str
    archived_as: str
    artifact_type: str


@dataclass
class ArchiveSnapshot:
    snapshot_dir: str
    label: str
    created_at: str
    artifacts: list[ArchivedArtifact] = field(default_factory=list)

    @property
    def archived_sources(self) -> list[str]:
        return [artifact.source for artifact in self.artifacts]


class BootstrapArtifactArchiver:
    """Archive bootstrap artifacts into a timestamped snapshot directory."""

    def __init__(self, paths: PathConfig | None = None) -> None:
        self._paths = paths or PathConfig.default()

    def archive(
        self,
        label: str = "manual",
        include_pmpd: bool = True,
        include_rag: bool = True,
    ) -> ArchiveSnapshot | None:
        artifacts = self._collect_artifacts(
            include_pmpd=include_pmpd,
            include_rag=include_rag,
        )
        if not artifacts:
            return None

        snapshot = ArchiveSnapshot(
            snapshot_dir=self._make_snapshot_dir(label),
            label=label,
            created_at=datetime.now().isoformat(),
        )
        os.makedirs(snapshot.snapshot_dir, exist_ok=True)

        for artifact_type, source_path, relative_dest in artifacts:
            destination = os.path.join(snapshot.snapshot_dir, relative_dest)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, destination)
            else:
                shutil.copy2(source_path, destination)
            snapshot.artifacts.append(
                ArchivedArtifact(
                    source=os.path.abspath(source_path),
                    archived_as=os.path.abspath(destination),
                    artifact_type=artifact_type,
                )
            )

        manifest_path = os.path.join(snapshot.snapshot_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "label": snapshot.label,
                    "created_at": snapshot.created_at,
                    "snapshot_dir": os.path.abspath(snapshot.snapshot_dir),
                    "artifacts": [asdict(artifact) for artifact in snapshot.artifacts],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print(f"  Archived bootstrap artifacts -> {snapshot.snapshot_dir}")
        print(f"  Archived items: {len(snapshot.artifacts)}")
        return snapshot

    def _collect_artifacts(
        self,
        include_pmpd: bool,
        include_rag: bool,
    ) -> list[tuple[str, str, str]]:
        items: list[tuple[str, str, str]] = []

        def add_file(path: str, artifact_type: str, relative_dest: str) -> None:
            if os.path.isfile(path):
                items.append((artifact_type, path, relative_dest))

        def add_dir(path: str, artifact_type: str, relative_dest: str) -> None:
            if os.path.isdir(path):
                items.append((artifact_type, path, relative_dest))

        def add_with_optional_txt(path: str, artifact_type: str, relative_dest: str) -> None:
            add_file(path, artifact_type, relative_dest)
            txt_path = f"{path}.txt"
            add_file(txt_path, f"{artifact_type}_text", f"{relative_dest}.txt")

        if include_pmpd:
            add_with_optional_txt(self._paths.pmpd_db_path, "pmpd_store", "pmpd/pmpd_store.json")
            add_with_optional_txt(
                self._paths.generated_prompts,
                "generated_prompts",
                "pmpd/generated_prompts.json",
            )
            add_dir(self._paths.markdown_tree_dir, "markdown_tree", "policy/md_tree")

        add_file(self._paths.bootstrap_state, "bootstrap_state", "bootstrap/bootstrap_state.json")
        add_file(
            self._paths.bootstrap_stats_path,
            "bootstrap_stats",
            "bootstrap/bootstrap_stats.json",
        )
        add_file("stage1_inspection.txt", "stage1_inspection", "bootstrap/stage1_inspection.txt")
        add_file("stage2_inspection.txt", "stage2_inspection", "bootstrap/stage2_inspection.txt")

        if include_rag:
            add_file(self._paths.rag_config_file, "rag_config", "rag/rag_config.json")
            for faiss_dir in sorted(glob.glob("*_faiss")):
                if os.path.isdir(faiss_dir):
                    add_dir(faiss_dir, "faiss_index", os.path.join("rag", "indices", faiss_dir))

        deduped: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for artifact_type, source_path, relative_dest in items:
            abs_source = os.path.abspath(source_path)
            if abs_source in seen:
                continue
            seen.add(abs_source)
            deduped.append((artifact_type, source_path, relative_dest))
        return deduped

    def _make_snapshot_dir(self, label: str) -> str:
        safe_label = self._slugify(label or "manual")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.abspath(self._paths.bootstrap_archive_dir)
        return os.path.join(root, f"{timestamp}_{safe_label}")

    @staticmethod
    def _slugify(text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
        return cleaned.strip("_") or "snapshot"
