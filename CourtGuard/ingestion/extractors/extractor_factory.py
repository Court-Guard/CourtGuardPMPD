import os
from typing import Dict, Type
from ingestion.extractors.document_extractor import DocumentExtractor
from ingestion.extractors.markdown_extractor import MarkdownExtractor
from ingestion.extractors.word_extractor import WordExtractor
from ingestion.extractors.llama_parse_extractor import LlamaParseExtractor
from ingestion.extractors.advanced_table_pdf_extractor import AdvancedTablePdfExtractor

class ExtractorFactory:
    """
    Routes document extraction based on file extension.
    For `.pdf`, it attempts the LlamaParseExtractor first. If that fails,
    it falls back to the AdvancedTablePdfExtractor gracefully.
    """
    
    @staticmethod
    def get_extractor(file_path: str) -> DocumentExtractor:
        """
        Produce the appropriate configured extractor strategy.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.md':
            return MarkdownExtractor()
            
        elif ext == '.docx':
            return WordExtractor()
            
        elif ext == '.pdf':
            # For PDF, we return a hybrid class or just try to 
            # instantiate LlamaParse Extractor directly. 
            # To handle the fallback loop elegantly, we create a hybrid router inline.
            return PdfRoutingExtractor()
            
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Permitted formats are .pdf, .docx, .md.")

class PdfRoutingExtractor(DocumentExtractor):
    """
    A smart router that tries LlamaParse, and catches its ExtractionFailedError 
    to automatically fallback to the offline pipeline without crashing the host Orchestrator.
    """
    def extract_text(self, file_path: str) -> str:
        llama_extractor = LlamaParseExtractor()
        
        # If no API key is present, don't even try the 10x loop.
        if not llama_extractor.api_key:
            print("  ⚠ LLAMA_CLOUD_API_KEY missing. Defaulting directly to offline Advanced Table PDF Extractor.")
            return AdvancedTablePdfExtractor().extract_text(file_path)
            
        try:
            print("  🚀 Intercepting PDF via LlamaCloud Vision Engine...")
            return llama_extractor.extract_text(file_path)
        except Exception as e:
            from core.exceptions import ExtractionFailedError
            if isinstance(e, ExtractionFailedError):
                print(f"  ⚠ LlamaParse Engine Failed: {e}")
                print("  ⚠ Falling back to offline Advanced Table PDF Extractor...")
                return AdvancedTablePdfExtractor().extract_text(file_path)
            raise e
