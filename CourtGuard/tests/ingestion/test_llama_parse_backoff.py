import pytest
import unittest
from unittest.mock import MagicMock, patch
from core.exceptions import ExtractionFailedError
from ingestion.extractors.llama_parse_extractor import LlamaParseExtractor

class DummyLlamaCloudResult:
    def __init__(self, text):
        self.markdown_full = text
        self.markdown = None

class TestLlamaParseBackoff:
    @patch("ingestion.extractors.llama_parse_extractor.time.sleep")
    @patch("ingestion.extractors.llama_parse_extractor.LlamaCloud")
    @patch("os.path.exists", return_value=True)
    def test_backoff_on_429_success_eventually(self, mock_exists, mock_llama, mock_sleep):
        import os
        os.environ["LLAMA_CLOUD_API_KEY"] = "fake_key"
        
        # Instantiate extractor
        extractor = LlamaParseExtractor()
        
        mock_client = MagicMock()
        mock_llama.return_value = mock_client
        
        # Make parsing.parse fail with 429 3 times, then succeed on the 4th
        mock_client.parsing.parse.side_effect = [
            Exception("HTTP 429 Rate Limit Exceeded"),
            Exception("HTTP 503 Server Overload"),
            Exception("timeout error occurred"),
            DummyLlamaCloudResult("Successful parsed content")
        ]
        
        # Write dummy file to pass open() check in extract_text
        with patch("builtins.open", unittest.mock.mock_open(read_data=b"dummy")):
            result = extractor.extract_text("dummy.pdf")
        
        assert result == "Successful parsed content"
        assert mock_client.parsing.parse.call_count == 4
        assert mock_sleep.call_count == 3
        
        # Check backoff timing
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)
        mock_sleep.assert_any_call(8.0)

    @patch("ingestion.extractors.llama_parse_extractor.time.sleep")
    @patch("ingestion.extractors.llama_parse_extractor.LlamaCloud")
    @patch("os.path.exists", return_value=True)
    def test_backoff_gives_up_after_max_retries(self, mock_exists, mock_llama, mock_sleep):
        import os
        os.environ["LLAMA_CLOUD_API_KEY"] = "fake_key"
        
        extractor = LlamaParseExtractor()
        
        mock_client = MagicMock()
        mock_llama.return_value = mock_client
        mock_client.parsing.parse.side_effect = Exception("HTTP 429 Rate Limit Exceeded")
        
        with patch("builtins.open", unittest.mock.mock_open(read_data=b"dummy")):
            with pytest.raises(ExtractionFailedError, match="Maximum retries exceeded"):
                extractor.extract_text("dummy.pdf")
            
        assert mock_client.parsing.parse.call_count == 10
        assert mock_sleep.call_count == 9
        
    @patch("ingestion.extractors.llama_parse_extractor.LlamaCloud")
    @patch("os.path.exists", return_value=True)
    def test_irrecoverable_error_does_not_retry(self, mock_exists, mock_llama):
        import os
        os.environ["LLAMA_CLOUD_API_KEY"] = "fake_key"
        
        extractor = LlamaParseExtractor()
        
        mock_client = MagicMock()
        mock_llama.return_value = mock_client
        mock_client.parsing.parse.side_effect = Exception("Invalid API Key")
        
        with patch("builtins.open", unittest.mock.mock_open(read_data=b"dummy")):
            with pytest.raises(ExtractionFailedError, match="irrecoverably"):
                extractor.extract_text("dummy.pdf")
            
        assert mock_client.parsing.parse.call_count == 1
