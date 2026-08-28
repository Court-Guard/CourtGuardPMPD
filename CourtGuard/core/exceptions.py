"""
Domain Exceptions

Custom exceptions for CourtGuard.
These allow higher-level orchestrators (like FastAPI servers or CLIs)
to catch specific errors gracefully rather than relying on sys.exit().
"""

class CourtGuardError(Exception):
    """Base exception for all CourtGuard domain errors."""
    pass

class APIKeyError(CourtGuardError):
    """Raised when all available API keys are exhausted or invalid."""
    pass

class ExtractionFailedError(CourtGuardError):
    """Raised when the document extractor fails to parse the incoming policy."""
    pass

class PMPDScalabilityError(CourtGuardError):
    """
    Raised when the policy database has too many categories to pass 
    in their entirety to the Attacker's context window.
    
    Requires running `--mode rag` to construct the FAISS index to 
    dynamically pre-filter categories.
    """
    def __init__(self, message: str, requires_faiss: bool = True):
        super().__init__(message)
        self.requires_faiss = requires_faiss


class SectionRoutingError(CourtGuardError):
    """
    Raised when a text chunk cannot be parsed into valid JSON after
    exhausting all token-limit escalation attempts.

    This is a FATAL error — the ingestion pipeline must crash rather
    than silently discard data. A partial Markdown tree is worse than
    no tree at all for scientific evaluation purposes.

    Attributes
    ----------
    chunk_index : 1-based index of the chunk that failed.
    attempts    : Number of token-limit escalations attempted.
    last_response : The raw LLM response from the final attempt.
    """
    def __init__(
        self,
        message: str,
        chunk_index: int,
        attempts: int,
        last_response: str = "",
    ):
        super().__init__(message)
        self.chunk_index   = chunk_index
        self.attempts      = attempts
        self.last_response = last_response
