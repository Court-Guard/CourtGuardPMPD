import os
from core.exceptions import ExtractionFailedError
from ingestion.extractors.document_extractor import DocumentExtractor

class MarkdownExtractor(DocumentExtractor):
    """
    Extracts raw text directly from a Markdown file.
    Since the target output is Markdown, this is a native pass-through.
    """
    def extract_text(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise ExtractionFailedError(f"Markdown file not found: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise ExtractionFailedError(f"Failed to extract markdown: {str(e)}")
