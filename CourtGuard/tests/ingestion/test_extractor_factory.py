import pytest
from ingestion.extractors.extractor_factory import ExtractorFactory
from ingestion.extractors.markdown_extractor import MarkdownExtractor
from ingestion.extractors.word_extractor import WordExtractor
from ingestion.extractors.extractor_factory import PdfRoutingExtractor

class TestExtractorFactory:
    def test_routes_markdown_to_markdown_extractor(self):
        extractor = ExtractorFactory.get_extractor("dummy.md")
        assert isinstance(extractor, MarkdownExtractor)
        
    def test_routes_docx_to_word_extractor(self):
        extractor = ExtractorFactory.get_extractor("dummy.DOCX")
        assert isinstance(extractor, WordExtractor)
        
    def test_routes_pdf_to_pdf_router(self):
        extractor = ExtractorFactory.get_extractor("dummy.pdf")
        assert isinstance(extractor, PdfRoutingExtractor)
        
    def test_unsupported_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            ExtractorFactory.get_extractor("dummy.txt")
