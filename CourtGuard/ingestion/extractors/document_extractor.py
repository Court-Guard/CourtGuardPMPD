from typing import Protocol, runtime_checkable

@runtime_checkable
class DocumentExtractor(Protocol):
    """
    Protocol for universal document extraction.
    
    All extraction strategies (PDF, DOCX, MD) must implement this interface
    to guarantee uniform processing by the PolicyIngester.
    """
    def extract_text(self, file_path: str) -> str:
        """
        Extracts raw textual and structural data from the document.
        
        Args:
            file_path: Absolute or relative path to the physical file.
            
        Returns:
            A completely extracted markdown-formatted or raw string representation
            of the document.
            
        Raises:
            ExtractionFailedError: If extraction fatally fails (corrupt file, API outage).
        """
        ...
