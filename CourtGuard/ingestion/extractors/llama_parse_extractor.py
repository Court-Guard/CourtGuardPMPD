import os
import time
from core.exceptions import ExtractionFailedError
from ingestion.extractors.document_extractor import DocumentExtractor

try:
    from llama_cloud import LlamaCloud
except ImportError:
    LlamaCloud = None

class LlamaParseExtractor(DocumentExtractor):
    """
    Extracts high-fidelity structural data from PDFs using LlamaCloud Vision Models.
    Implements a rigorous 10-attempt exponential backoff for HTTP 429/Server Overloads.
    Migrated to new llama-cloud SDK (>=1.0.0).
    """
    MAX_RETRIES = 10
    BASE_DELAY = 2.0

    def __init__(self):
        from infrastructure.config import LlamaCloudConfig
        config = LlamaCloudConfig.from_env()
        self.api_key = config.api_key.strip() if config.api_key else None
        self.base_url = config.base_url

    def extract_text(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise ExtractionFailedError(f"PDF document not found: {file_path}")
            
        if LlamaCloud is None:
            raise ExtractionFailedError("llama-cloud library is not installed. Run: pip install llama-cloud>=1.0")
            
        if not self.api_key:
            raise ExtractionFailedError("LLAMA_CLOUD_API_KEY not found in environment for LlamaParse Extractor.")

        client = LlamaCloud(
            api_key=self.api_key,
            base_url=self.base_url
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # The unified parse method uploads, polls, and returns the result in one go
                # We need to supply the file as a tuple or Path. Let's use tuple 
                file_name = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    file_contents = f.read()
                
                result = client.parsing.parse(
                    upload_file=(file_name, file_contents, "application/pdf"),
                    tier="agentic",
                    version="latest",
                    expand=["markdown"],
                    verbose=False
                )
                
                # In LlamaCloud SDK v2, result.markdown is a structured object, not a string
                full_text = None
                if hasattr(result, "markdown_full") and result.markdown_full:
                    full_text = result.markdown_full
                elif hasattr(result, "markdown") and hasattr(result.markdown, "pages"):
                    full_text = "\n\n".join(page.markdown for page in result.markdown.pages if page.markdown)
                
                if not full_text:
                    raise ExtractionFailedError("LlamaCloud returned empty markdown content. Check your API key or Region.")
                    
                return full_text
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limits or server errors
                is_throttle_error = any(code in error_msg for code in ["429", "503", "502", "504", "rate limit", "timeout"])
                
                if is_throttle_error:
                    if attempt < self.MAX_RETRIES:
                        sleep_time = self.BASE_DELAY * (2 ** (attempt - 1))
                        print(f"  ⚠ LlamaCloud Server Overload ({e}). Retrying in {sleep_time} seconds... (Attempt {attempt}/{self.MAX_RETRIES})")
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise ExtractionFailedError(f"LlamaCloud extraction failed: Maximum retries exceeded. Last error: {e}")
                else:
                    # Provide specific help for Auth/Region errors
                    if "401" in error_msg or "403" in error_msg or "invalid" in error_msg or "key" in error_msg:
                        print(f"\n✖ LlamaCloud API Error: {e}")
                        print("  TIP: If your key is correct, your project might be in the EU region.")
                        print("  Set LLAMA_CLOUD_BASE_URL=XXXX in your .env file.")
                    
                    raise ExtractionFailedError(f"LlamaCloud failed irrecoverably after {attempt} attempts. Error: {e}")
                    
        raise ExtractionFailedError("LlamaCloud extraction failed: Maximum retries exceeded.")
