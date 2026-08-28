"""
PDF Extractor

Converts a raw policy PDF into a single plain-text string.

Responsibilities
────────────────
  • Text extraction via pdfplumber
  • Table flattening to prose
  • Repeated footer deduplication

This class contains zero LLM calls and zero knowledge of the downstream
pipeline.  It is the only place in the project that imports pdfplumber,
resolving the deferred-import smell from policy_ingester.py:

    # old (deferred import buried in free function):
    def _extract_raw_text(pdf_path):
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install pdfplumber")

    # new (fail fast at construction, clear dependency):
    from ingestion.pdf_extractor import PDFExtractor
"""

from __future__ import annotations

import os

try:
    import pdfplumber
except ImportError as exc:
    raise ImportError(
        "pdfplumber is required for PDF ingestion. " "Install it with: pip install pdfplumber"
    ) from exc


class PDFExtractor:
    """
    Extracts all text and tables from a policy PDF.

    Tables are flattened to key: value prose so the downstream LLM can
    reason about them without special table-parsing logic.

    Repeated short lines (footers, page numbers) are deduplicated across
    pages to prevent them polluting the LLM context.

    Usage
    -----
        extractor = PDFExtractor()
        raw_text  = extractor.extract("policy/document.pdf")
    """

    def extract(self, pdf_path: str) -> str:
        """
        Extract all text and tables from a PDF file.

        Args:
            pdf_path: Absolute or relative path to the PDF.

        Returns:
            Full extracted text as a single string, with page markers.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError:      If pdfplumber fails to open the file.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages_text: list[str] = []
        seen_footers: set[str] = set()

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total = len(pdf.pages)
                print(f"  📄 Extracting {total} pages...")

                for i, page in enumerate(pdf.pages, start=1):
                    page_text = self._extract_page(page, i, seen_footers)
                    pages_text.append(page_text)

        except Exception as exc:
            raise RuntimeError(f"pdfplumber failed to process {pdf_path}: {exc}") from exc

        return "\n".join(pages_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_page(
        self,
        page,
        page_number: int,
        seen_footers: set[str],
    ) -> str:
        """
        Extract text and tables from a single PDF page.

        Args:
            page:         pdfplumber Page object.
            page_number:  1-based page index for the page marker.
            seen_footers: Mutable set of short lines already seen —
                          updated in place for cross-page deduplication.

        Returns:
            Extracted page content as a string.
        """
        parts: list[str] = [f"\n\n--- PAGE {page_number} ---\n"]

        # Flatten tables to prose
        for table in page.extract_tables() or []:
            table_text = self._flatten_table(table)
            if table_text:
                parts.append(table_text)

        # Plain text with footer deduplication
        plain = page.extract_text()
        if plain:
            parts.append(self._deduplicate_lines(plain, seen_footers))

        return "".join(parts)

    @staticmethod
    def _flatten_table(table: list) -> str:
        """
        Flatten a pdfplumber table (list of rows) to key: value prose.

        Args:
            table: Nested list from pdfplumber.extract_tables().

        Returns:
            Prose string, or empty string if the table is trivial.
        """
        if not table:
            return ""

        rows = [r for r in table if any(c for c in r)]
        if len(rows) < 2:
            return ""

        header = rows[0]
        lines: list[str] = ["\n"]

        for row in rows[1:]:
            cells: list[str] = []
            for h, c in zip(header, row):
                h = str(h).strip() if h else ""
                c = str(c).strip() if c else ""
                if h and c:
                    cells.append(f"{h}: {c}")
                elif c:
                    cells.append(c)
            if cells:
                lines.append(". ".join(cells) + ".\n")

        return "".join(lines)

    @staticmethod
    def _deduplicate_lines(plain_text: str, seen_footers: set[str]) -> str:
        """
        Remove repeated short lines (footers, page numbers) from page text.

        A line is treated as a potential footer if it is shorter than 80
        characters.  The first occurrence is kept; subsequent duplicates
        across pages are dropped.

        Args:
            plain_text:   Raw text from pdfplumber.extract_text().
            seen_footers: Mutable set updated in place.

        Returns:
            Cleaned text string.
        """
        cleaned: list[str] = []
        for line in plain_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) < 80 and stripped in seen_footers:
                continue
            if len(stripped) < 80:
                seen_footers.add(stripped)
            cleaned.append(stripped)
        return "\n".join(cleaned)
