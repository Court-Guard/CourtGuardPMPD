"""
Markdown Tree Reader

Single implementation of policy Markdown tree traversal that was previously
duplicated across three modules:

  pmpd_parser.py       — _collect_hazard_files()   (categories/ only)
  prompt_generator.py  — _read_policy_meta()        (categories/ + meta/overview.md)
  rag_pipeline.py      — _sample_tree()             (full walk, char-limited)
                       — _load_markdown_files()      (full walk, Document objects)

All four use-cases are covered by this class.

Tree structure expected
───────────────────────
  <tree_path>/
    meta/
      overview.md      ← document title, domain
    definitions.md     ← glossary
    examples.md        ← illustrative cases
    categories/
      <category>.md    ← one file per category (×N)

The category ID (S1, S2, …) is assigned sequentially by sorted
filename — matching the original logic in pmpd_parser._collect_hazard_files().
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HazardEntry:
    """
    Represents one category file in the Markdown tree.

    Note: Class name kept as HazardEntry for backward compatibility
    with pmpd_parser.py and other consumers.

    Attributes
    ----------
    category_id   : Sequential ID assigned by sort order, e.g. "S1"
    category_name : Human-readable name derived from filename stem,
                    e.g. "defamation.md" → "Defamation"
    file_path     : Absolute path to the Markdown file
    """

    category_id: str
    category_name: str
    file_path: str


@dataclass
class PolicyMeta:
    """
    Metadata extracted from the overview.md file in the tree.

    Attributes
    ----------
    title           : Document title from the first H1 heading
    domain          : Domain string from _Domain: ..._ line
    overview_text   : Full text of overview.md
    """

    title: str = "Policy Document"
    domain: str = "General policy compliance"
    overview_text: str = ""


@dataclass
class MarkdownFile:
    """
    A single Markdown file from the tree, ready for embedding.

    Attributes
    ----------
    content       : Full text content of the file
    relative_path : Path relative to tree root, e.g. "categories/defamation.md"
    category      : Top-level directory name, e.g. "categories", "meta", "root"
    filename      : Bare filename, e.g. "defamation.md"
    full_path     : Absolute filesystem path
    """

    content: str
    relative_path: str
    category: str
    filename: str
    full_path: str


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class MarkdownTreeReader:
    """
    Reads and traverses a PolicyIngester Markdown tree.

    All reading is lazy — files are read on demand, not at construction.

    Usage
    -----
        reader  = MarkdownTreeReader("policy/md_tree")
        meta    = reader.read_meta()
        hazards = reader.list_hazard_files()
        files   = reader.load_all_files()
        sample  = reader.sample_tree(max_chars_per_file=1500)
    """

    def __init__(self, tree_path: str) -> None:
        """
        Args:
            tree_path: Root directory of the PolicyIngester Markdown tree.
        """
        self.tree_path = tree_path

    # ------------------------------------------------------------------
    # Policy metadata
    # ------------------------------------------------------------------

    def read_meta(self) -> PolicyMeta:
        """
        Read document metadata from meta/overview.md.

        Returns PolicyMeta with default values if the file does not exist.
        """
        overview_path = os.path.join(self.tree_path, "meta", "overview.md")
        if not os.path.exists(overview_path):
            return PolicyMeta()

        try:
            with open(overview_path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            print(f"  ⚠ Could not read overview.md: {exc}")
            return PolicyMeta()

        title = self._extract_h1(text) or "Policy Document"
        domain = self._extract_domain(text) or "General policy compliance"

        return PolicyMeta(title=title, domain=domain, overview_text=text)

    def read_definitions(self, max_chars: int = 8000) -> str:
        """
        Read the definitions.md file.

        Args:
            max_chars: Character limit.

        Returns:
            File content, or empty string if not found.
        """
        return self._read_file(
            os.path.join(self.tree_path, "definitions.md"),
            max_chars=max_chars,
        )

    def read_examples(self, max_chars: int = 16000) -> str:
        """
        Read the examples.md file.

        Args:
            max_chars: Character limit.

        Returns:
            File content, or empty string if not found.
        """
        return self._read_file(
            os.path.join(self.tree_path, "examples.md"),
            max_chars=max_chars,
        )

    # ------------------------------------------------------------------
    # Hazard file enumeration
    # ------------------------------------------------------------------

    def list_hazard_files(self) -> list[HazardEntry]:
        """
        Return a sorted list of HazardEntry objects for all hazard .md files.

        Category IDs are assigned sequentially (S1, S2, …) by sorted
        filename — preserving the original pmpd_parser behaviour.

        Returns:
            List of HazardEntry, sorted by filename.
        """
        categories_dir = os.path.join(self.tree_path, "categories")
        if not os.path.exists(categories_dir):
            # Backward compat: also check legacy "hazards" folder
            categories_dir = os.path.join(self.tree_path, "hazards")
            if not os.path.exists(categories_dir):
                return []

        entries: list[HazardEntry] = []
        for idx, fname in enumerate(sorted(os.listdir(categories_dir)), start=1):
            if not fname.endswith(".md"):
                continue
            entries.append(
                HazardEntry(
                    category_id=f"S{idx}",
                    category_name=self._stem_to_name(fname),
                    file_path=os.path.join(categories_dir, fname),
                )
            )
        return entries

    def read_hazard_summaries(self, max_chars: int = 3000) -> dict[str, str]:
        """
        Return a dict mapping category_name → file_content (truncated).

        Used by PromptGenerator to build the generation prompt.

        Args:
            max_chars: Character limit per hazard file.

        Returns:
            Dict of {category_name: content}.
        """
        summaries: dict[str, str] = {}
        for entry in self.list_hazard_files():
            content = self._read_file(entry.file_path, max_chars=max_chars)
            summaries[entry.category_name] = content
        return summaries

    # ------------------------------------------------------------------
    # Full tree loading
    # ------------------------------------------------------------------

    def load_all_files(self) -> list[MarkdownFile]:
        """
        Recursively load all Markdown files from the tree.

        Used by RAGPipeline to build the FAISS index.

        Returns:
            List of MarkdownFile objects, sorted by relative path.
        """
        results: list[MarkdownFile] = []

        for root, _, files in os.walk(self.tree_path):
            for fname in sorted(files):
                if not fname.endswith(".md"):
                    continue

                full_path = os.path.join(root, fname)
                try:
                    with open(full_path, encoding="utf-8") as f:
                        content = f.read()
                except OSError as exc:
                    print(f"  ⚠ Could not read {full_path}: {exc}")
                    continue

                rel_path = os.path.relpath(full_path, self.tree_path)
                path_parts = rel_path.split(os.sep)
                category = path_parts[0] if len(path_parts) > 1 else "root"

                results.append(
                    MarkdownFile(
                        content=content,
                        relative_path=rel_path,
                        category=category,
                        filename=fname,
                        full_path=full_path,
                    )
                )

        return results

    def sample_tree(self, max_chars_per_file: int = 1500) -> str:
        """
        Read a character-limited sample from every Markdown file in the tree.

        Used by RAGTuner to analyse document characteristics without loading
        the full content.

        Args:
            max_chars_per_file: Character limit applied per file.

        Returns:
            Concatenated sample string with file labels.
        """
        parts: list[str] = []

        for root, _, files in os.walk(self.tree_path):
            for fname in sorted(files):
                if not fname.endswith(".md"):
                    continue
                full_path = os.path.join(root, fname)
                rel = os.path.relpath(full_path, self.tree_path)
                content = self._read_file(full_path, max_chars=max_chars_per_file)
                parts.append(f"### FILE: {rel}\n{content}\n")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: str, max_chars: int = 8000) -> str:
        """Read a file up to max_chars, returning empty string on error."""
        try:
            with open(path, encoding="utf-8") as f:
                return f.read(max_chars)
        except OSError as exc:
            print(f"  ⚠ Could not read {path}: {exc}")
            return ""

    @staticmethod
    def _extract_h1(text: str) -> str | None:
        """Extract the first H1 heading from Markdown text."""
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_domain(text: str) -> str | None:
        """Extract the _Domain: ..._ line from overview.md."""
        match = re.search(r"_Domain:\s*(.+?)_", text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _stem_to_name(filename: str) -> str:
        """Convert a filename stem to a human-readable category name.

        e.g. "defamation.md" → "Defamation"
             "hate_speech.md" → "Hate Speech"
        """
        return os.path.splitext(filename)[0].replace("_", " ").title()
