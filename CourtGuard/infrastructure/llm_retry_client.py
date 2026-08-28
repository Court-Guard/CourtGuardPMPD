"""
LLM Retry Client

Single implementation of the retry-with-backoff wrapper that was previously
duplicated across four modules:

  pmpd_parser.py      — free function, exponential backoff, 5 retries, 10 s base
  prompt_generator.py — free function, exponential backoff, 5 retries, 10 s base
  policy_ingester.py  — free function, exponential backoff, 5 retries, 10 s base
  debate_logic.py     — instance method, fixed long waits, 10 retries, 300 s / 180 s

Both behaviours are unified here via RetryConfig.  The two canonical presets
(BOOTSTRAP and DEBATE) reproduce the exact timing from the original code.

Design
──────
  LLMRetryClient wraps an APIClient and exposes a single call() method.
  It catches UptimeError and retries with the configured strategy.
  APIError is always re-raised immediately — it signals a key-rotation event,
  not a transient failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from infrastructure.api_client import APIClient, APIError, UptimeError

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryConfig:
    """
    Immutable configuration for the retry strategy.

    Attributes
    ----------
    max_retries        : Maximum number of retry attempts after the first failure.
    base_delay         : Seconds to wait before the first retry.
    exponential        : If True, delay doubles on each attempt (base * 2^attempt).
                         If False, delay is fixed at base_delay for all retries
                         except the first, which uses initial_delay.
    initial_delay      : Delay for the first retry when exponential=False.
                         Ignored when exponential=True.
    countdown_display  : If True, print a live countdown while waiting.
                         Used by the debate loop for long waits.
    """

    max_retries: int = 5
    base_delay: float = 10.0
    exponential: bool = True
    initial_delay: float = 300.0  # only used when exponential=False
    countdown_display: bool = False

    # ------------------------------------------------------------------
    # Canonical presets
    # ------------------------------------------------------------------

    @classmethod
    def bootstrap(cls) -> RetryConfig:
        """
        Preset for bootstrap pipeline modules (ingestion, parsing, generation).

        Exponential backoff: 10 s, 20 s, 40 s, 80 s, 160 s.
        Matches the original _call_with_retry() in:
          policy_ingester.py, prompt_generator.py, pmpd_parser.py
        """
        return cls(
            max_retries=5,
            base_delay=10.0,
            exponential=True,
        )

    @classmethod
    def debate(cls) -> RetryConfig:
        """
        Preset for the debate loop.

        First retry waits 5 minutes (300 s); subsequent retries wait 3 minutes
        (180 s).  A live countdown is printed so the operator can see progress.
        Matches the original _call_with_retry() in debate_logic.py.
        """
        return cls(
            max_retries=10,
            base_delay=180.0,
            exponential=False,
            initial_delay=300.0,
            countdown_display=True,
        )

    @classmethod
    def multivote(cls) -> RetryConfig:
        """
        Preset for compact multi-vote runs.

        Multi-vote makes several short independent calls per record, so the
        very long debate backoff is too expensive here. Use a short
        exponential retry instead: 10 s, 20 s, 40 s.
        """
        return cls(
            max_retries=3,
            base_delay=10.0,
            exponential=True,
            countdown_display=False,
        )


# ---------------------------------------------------------------------------
# LLM Retry Client
# ---------------------------------------------------------------------------


class LLMRetryClient:
    """
    Wraps APIClient with configurable retry logic for transient UptimeErrors.

    Usage — bootstrap context
    ─────────────────────────
        retry_client = LLMRetryClient(api_client, RetryConfig.bootstrap())
        content      = retry_client.call(
            prompt     = "...",
            system_msg = "...",
            model      = BootstrapModel.LLAMA_70B,
            max_tokens = 1024,
        )

    Usage — debate context
    ──────────────────────
        retry_client = LLMRetryClient(api_client, RetryConfig.debate())
        result       = retry_client.call_raw(
            prompt     = "...",
            system_msg = "...",
            model      = BootstrapModel.GPT_OSS_20B,
        )
        # call_raw() returns the full result dict from APIClient.call_model()
    """

    def __init__(
        self,
        api_client: APIClient,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """
        Args:
            api_client:   Initialized APIClient instance.
            retry_config: Retry strategy configuration.
                          Defaults to RetryConfig.bootstrap() if not provided.
        """
        self._client = api_client
        self._config = retry_config or RetryConfig.bootstrap()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        prompt:        str,
        system_msg:    str,
        model:         str,
        developer_msg: str | None = None,
        max_tokens:    int        = 1024,
        temperature:   float      = 0.1,
    ) -> str:
        """
        Make an LLM call, retrying on UptimeError.

        Returns the response content string on success.
        Raises RuntimeError if all retries are exhausted.
        Raises APIError immediately — never retried.

        Args:
            prompt:      User-turn prompt string.
            system_msg:  System message for the LLM.
            model:       Model identifier string.
            max_tokens:  Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            Response content string.

        Raises:
            APIError:    Critical error — key rotation required.
            RuntimeError: All retry attempts exhausted.
        """
        result = self.call_raw(
            prompt=        prompt,
            system_msg=    system_msg,
            model=         model,
            developer_msg= developer_msg,
            max_tokens=    max_tokens,
            temperature=   temperature,
        )

        if result.get("success"):
            return result["content"]

        raise RuntimeError(f"API call failed after retries: {result.get('error', 'unknown')}")

    def call_raw(
        self,
        prompt: str,
        system_msg: str,
        model: str,
        developer_msg: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Make an LLM call and return the full result dict from APIClient.

        Used by debate_logic where response_time and model_used are needed.
        Retries on UptimeError using the configured strategy.

        Args:
            prompt:      User-turn prompt string.
            system_msg:  System message for the LLM.
            model:       Model identifier string.
            max_tokens:  Maximum tokens, or None for model default.
            temperature: Sampling temperature.

        Returns:
            Full result dict from APIClient.call_model().

        Raises:
            APIError:    Critical error — never retried.
            RuntimeError: All retries exhausted.
        """
        cfg = self._config

        for attempt in range(cfg.max_retries + 1):
            try:
                return self._client.call_model(
                    prompt=        prompt,
                    system_msg=    system_msg,
                    developer_msg= developer_msg,
                    model=         model,
                    temperature=   temperature,
                    max_tokens=    max_tokens,
                )

            except UptimeError as exc:
                if attempt >= cfg.max_retries:
                    raise APIError(
                        f"Provider unavailable after {cfg.max_retries} retries: {exc}"
                    ) from exc

                delay = self._compute_delay(attempt, cfg)
                self._wait(delay, attempt, cfg)

            except APIError:
                raise  # critical — never retry

        # Unreachable, but satisfies type checkers
        raise RuntimeError("All retry attempts exhausted.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delay(attempt: int, cfg: RetryConfig) -> float:
        """Compute wait time for this attempt based on the retry strategy."""
        if cfg.exponential:
            return cfg.base_delay * (2**attempt)
        # Fixed strategy: first retry uses initial_delay, subsequent use base_delay
        return cfg.initial_delay if attempt == 0 else cfg.base_delay

    @staticmethod
    def _wait(delay: float, attempt: int, cfg: RetryConfig) -> None:
        """Wait for `delay` seconds, optionally printing a countdown."""
        delay_int = int(delay)
        wait_minutes = delay_int // 60

        print(
            f"\nProvider temporarily unavailable "
            f"(retry {attempt + 1}/{cfg.max_retries}) — "
            f"waiting {wait_minutes}m {delay_int % 60}s..."
        )

        if cfg.countdown_display:
            for remaining in range(delay_int, 0, -30):
                mins = remaining // 60
                secs = remaining % 60
                print(f"   {mins}m {secs:02d}s remaining...", end="\r")
                time.sleep(min(30, remaining))
            print()  # newline after countdown
        else:
            time.sleep(delay)
