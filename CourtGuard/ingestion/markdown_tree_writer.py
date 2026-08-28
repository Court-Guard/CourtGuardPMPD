"""
Markdown Tree Writer

Writes a section file map to the Markdown directory tree on disk.

Extracted from _write_markdown_tree() in policy_ingester.py.

Responsibilities
────────────────
  • Create output directories
  • Write each section file with a standard header
  • Handle the special overview.md header (title + domain)
  • Fill empty sections with a placeholder rather than empty files

This class has zero LLM calls and zero knowledge of how the content
was produced — it only knows how to write it.
"""

from __future__ import annotations

import os

from ingestion.structure_planner import StructurePlan


class MarkdownTreeWriter:
    """
    Writes a section file map to a Markdown directory tree on disk.

    Usage
    -----
        writer = MarkdownTreeWriter()
        writer.write(file_map, plan, "policy/md_tree")
    """

    def write(
        self,
        file_map: dict[str, str],
        plan: StructurePlan,
        output_dir: str,
    ) -> None:
        """
        Write all section files to disk.

        Args:
            file_map:   Dict mapping relative .md paths to content strings.
                        e.g. {"categories/defamation.md": "...", ...}
            plan:       StructurePlan supplying title and domain for headers.
            output_dir: Root output directory for the Markdown tree.
        """
        os.makedirs(output_dir, exist_ok=True)

        for rel_path, content in file_map.items():
            self._write_file(rel_path, content, plan, output_dir)

        print(f"  💾 Markdown tree written to: {output_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_file(
        self,
        rel_path: str,
        content: str,
        plan: StructurePlan,
        output_dir: str,
    ) -> None:
        """
        Write a single section file with the appropriate header.

        Args:
            rel_path:   Relative path within the tree, e.g. "categories/defamation.md".
            content:    Section content to write.
            plan:       StructurePlan for title/domain in overview header.
            output_dir: Root of the Markdown tree.
        """
        full_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        header = self._build_header(rel_path, plan)
        body = (
            content.strip() if content.strip() else ("_No content identified for this section._")
        )

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(body + "\n")

    @staticmethod
    def _build_header(rel_path: str, plan: StructurePlan) -> str:
        """
        Build the Markdown header for a section file.

        The overview file gets a special header with the document title
        and domain.  All other files get a simple H1 derived from their
        filename stem.

        Args:
            rel_path: Relative .md path within the tree.
            plan:     StructurePlan for title and domain.

        Returns:
            Header string (including trailing newlines).
        """
        if rel_path == "meta/overview.md":
            return f"# {plan.title}\n\n_Domain: {plan.domain}_\n\n"

        section_name = os.path.splitext(os.path.basename(rel_path))[0].replace("_", " ").title()
        return f"# {section_name}\n\n"
