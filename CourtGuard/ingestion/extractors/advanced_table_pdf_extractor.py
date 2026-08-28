import os
from core.exceptions import ExtractionFailedError
from ingestion.extractors.document_extractor import DocumentExtractor

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

class AdvancedTablePdfExtractor(DocumentExtractor):
    """
    Offline PDF extraction fallback.
    Uses pdfplumber but with advanced table extraction settings to isolate
    grid cells, maintaining logical key-value integrity on poorly formatted documents.
    """
    def extract_text(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise ExtractionFailedError(f"PDF document not found: {file_path}")
            
        if pdfplumber is None:
            raise ExtractionFailedError("pdfplumber library is not installed.")

        try:
            full_text = []
            
            # Use advanced table heuristics
            table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_y_tolerance": 5,
                "intersection_x_tolerance": 5,
            }
            
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=3)
                    if page_text:
                        full_text.append(page_text)
                        
                    tables = page.extract_tables(table_settings)
                    
                    if tables:
                        for table in tables:
                            for row in table:
                                # Clean None cells
                                clean_row = [cell.strip().replace('\n', ' ') if cell else "" for cell in row]
                                # Discard empty rows
                                if any(clean_row):
                                    full_text.append(" | ".join(clean_row))
                                    
            return "\n\n".join(full_text)
            
        except Exception as e:
            raise ExtractionFailedError(f"Offline Advanced PDF Extraction failed: {str(e)}")
