"""
API Client Module for OpenRouter Integration

Handles API communication with OpenRouter, including error handling,
retry logic, and usage tracking.

Changes from original api_client.py
─────────────────────────────────────
  • OpenAI SDK instance is now private (_client) — eliminates the
    client.client.api_key Law of Demeter chain seen in main.py/pmpd_main.py.
  • api_key exposed as a read-only property.
  • Error classification strings promoted to class-level constants
    (UPTIME_ERROR_INDICATORS, CRITICAL_ERROR_INDICATORS) — Open/Closed.
  • Usage tracking extracted into UsageTracker dataclass — SRP.
  • X-Title header updated from old project name to "CourtGuard".
  • hasattr(completion, 'usage') guard replaced with getattr(..., None).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some shells
    find_dotenv = None
    load_dotenv = None
from openai import OpenAI

if find_dotenv and load_dotenv:
    load_dotenv(find_dotenv(usecwd=True))


# ---------------------------------------------------------------------------
# Exceptions  (unchanged — imported by all consumers)
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Exception for critical API errors requiring key rotation or process halt."""

    pass


class UptimeError(Exception):
    """Exception for temporary provider/uptime errors requiring retry with backoff."""

    pass


# ---------------------------------------------------------------------------
# Usage tracker
# ---------------------------------------------------------------------------


@dataclass
class UsageTracker:
    """
    Tracks API usage statistics across calls.

    Extracted from APIClient to satisfy SRP — the client handles
    communication; the tracker handles observability.

    Attributes
    ----------
    total_requests : Total number of calls made through the client.
    current_model  : Model identifier used in the most recent call.
    default_model  : Fallback model when no model is specified per-call.
    """

    default_model: str
    total_requests: int = field(default=0)
    current_model: str = field(default="")

    def record(self, model: str) -> None:
        """Increment request count and update current model."""
        self.total_requests += 1
        self.current_model = model

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of current usage statistics."""
        return {
            "total_requests": self.total_requests,
            "model": self.current_model or self.default_model,
        }


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


class APIClient:
    """
    Client for interacting with OpenRouter API.

    Provides robust error handling with distinction between transient
    provider errors (UptimeError — retry) and critical errors
    (APIError — key rotation or halt).

    The internal OpenAI SDK instance is private.  Use the api_key property
    to read the key without reaching into internals.
    """

    DEFAULT_MODEL: str = os.getenv("COURTGUARD_DEBATE_MODEL", "")
    BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "XXXX")

    # ------------------------------------------------------------------
    # Error classification — class-level constants (OCP)
    # Adding a new indicator = adding to the list, not modifying logic.
    # ------------------------------------------------------------------

    #: Transient provider errors — should be retried with backoff.
    UPTIME_ERROR_INDICATORS: tuple[str, ...] = (
        "provider returned error",
        "temporarily rate-limited upstream",
        "please retry shortly",
        "upstream",
        "provider error",
        "temporarily unavailable",
        # Network-level errors (retryable, not critical)
        "connection error",
        "timeout",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "readtimeout",
        "connectionreset",
        "remotedisconnected",
        "sslerror",
        "recv(",
        "getaddrinfo",
        "connecterror",
        "readerror",
    )

    #: Critical errors — require key rotation or process halt.
    CRITICAL_ERROR_INDICATORS: tuple[str, ...] = (
        "rate limit",
        "rate_limit",
        "ratelimit",
        "rate-limited",
        "error code: 429",
        "quota",
        "exceeded",
        "too many requests",
        "api key",
        "authentication",
        "unauthorized",
        "forbidden",
    )

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
    ) -> None:
        """
        Initialize API client.

        Args:
            api_key:  OpenRouter API key.
            base_url: Base URL for OpenRouter API.
        """
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._tracker = UsageTracker(default_model=self.DEFAULT_MODEL)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str:
        """
        The API key used by this client.

        Exposed as a first-class property to eliminate the
        client.client.api_key Law of Demeter violation in callers.
        """
        return self._client.api_key

    @property
    def default_model(self) -> str:
        """The default model used when no model is specified per-call."""
        return self._tracker.default_model

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def call_model(
        self,
        prompt:        str,
        system_msg:    str         = "You are a helpful AI assistant...",
        developer_msg: str | None  = None,
        model:       str | None = None,
        temperature: float      = 0.7,
        max_tokens:  int | None = None,
    ) -> dict[str, Any]:
        """
        Make API call to language model.

        Args:
            prompt:      User prompt.
            system_msg:  System message for model context.
            model:       Model identifier (uses DEFAULT_MODEL if None).
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens in response.

        Returns:
            Dictionary containing response content, success status,
            timing, and metadata.

        Raises:
            APIError:    For critical errors requiring key rotation.
            UptimeError: For temporary provider errors requiring retry.
        """
        start_time = time.time()
        model_to_use = model or self.DEFAULT_MODEL
        self._tracker.record(model_to_use)

        try:
            completion = self._client.chat.completions.create(
                model=model_to_use,
                messages=self._build_messages(system_msg, developer_msg, prompt),
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "XXXX",
                    "X-Title": "CourtGuard",
                },
            )

            raw_content = completion.choices[0].message.content
            if not raw_content:
                raise UptimeError("Model returned null or empty content. This is likely a transient provider error.")

            content = raw_content.strip()
            usage = getattr(completion, "usage", None)

            return {
                "content":        content,
                "success":        True,
                "response_time":  time.time() - start_time,
                "model_used":     model_to_use,
                "tokens_used":    getattr(usage, "total_tokens",       None),
                "input_tokens":   getattr(usage, "prompt_tokens",      None),
                "output_tokens":  getattr(usage, "completion_tokens",  None),
            }

        except Exception as exc:
            return self._handle_exception(exc, model_to_use, start_time)

    # ------------------------------------------------------------------
    # Usage statistics
    # ------------------------------------------------------------------

    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get API usage statistics.

        Returns:
            Dictionary with total_requests and current model.
        """
        return self._tracker.stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_exception(
        self,
        exc: Exception,
        model: str,
        start_time: float,
    ) -> dict[str, Any]:
        """
        Classify an exception and raise the appropriate typed error,
        or return a non-critical error result dict.

        Critical and uptime errors are raised so retry logic upstream
        can handle them.  All other errors are returned as a result dict
        so the caller can decide how to proceed.
        """
        error_msg = str(exc)
        error_msg_lower = error_msg.lower()

        print(f"\nAPI Error (Request #{self._tracker.total_requests}): {error_msg}")

        # Type-based catch: network-level exceptions are always retryable
        # regardless of error message content.
        try:
            import httpx
            import httpcore
            if isinstance(exc, (
                httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                httpcore.ReadTimeout, httpcore.ConnectTimeout,
                ConnectionError, OSError,
            )):
                raise UptimeError(f"Network error (retryable): {error_msg}") from exc
        except ImportError:
            pass  # httpx/httpcore not installed — fall through to string matching

        # Uptime errors take priority — they are a subset of "provider errors"
        # and should be retried rather than cause key rotation.
        if any(ind in error_msg_lower for ind in self.UPTIME_ERROR_INDICATORS):
            raise UptimeError(f"Provider/Uptime Error: {error_msg}")

        if any(ind in error_msg_lower for ind in self.CRITICAL_ERROR_INDICATORS):
            raise APIError(f"Critical API Error: {error_msg}")

        # Non-critical — return error result so caller can handle gracefully.
        return {
            "content": "",
            "success": False,
            "error": error_msg,
            "response_time": time.time() - start_time,
            "model_used": model,
        }
    

    def _build_messages(
        self,
        system_msg:    str,
        developer_msg: str | None,
        prompt:        str,
    ) -> list[dict[str, str]]:
        """
        Build the message list for an API call.
    
        When developer_msg is provided AND use_harmony_roles is enabled,
        sends system + developer + user as separate messages.
        Otherwise collapses into system + user for full compatibility.
    
        The harmony roles (system=meta, developer=instructions) follow
        OpenAI gpt-oss Harmony format. Falls back gracefully to standard
        format for all other models or when harmony roles are disabled.
        """
        import os
        use_harmony = os.getenv(
            "COURTGUARD_USE_HARMONY_ROLES", "false"
        ).lower() == "true"
    
        if developer_msg and use_harmony:
            return [
                {"role": "system",    "content": system_msg},
                {"role": "developer", "content": developer_msg},
                {"role": "user",      "content": prompt},
            ]
    
        # Fallback: collapse developer instructions into system
        if developer_msg:
            combined = f"{system_msg}\n\n{developer_msg}"
            return [
                {"role": "system", "content": combined},
                {"role": "user",   "content": prompt},
            ]
    
        return [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ]
