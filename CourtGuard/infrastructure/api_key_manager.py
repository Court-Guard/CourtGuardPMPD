"""
API Key Manager Module

Manages multiple API keys with automatic rotation upon rate limit exhaustion.

Changes from original api_key_manager.py
─────────────────────────────────────────
  • _api_keys dict is now private — external mutation of key state prevented.
  • select_starting_key() removed — stdin blocking does not belong in a
    domain class.  Key selection is now the responsibility of the CLI layer
    (CourtGuardApp / app.py).  Use set_key() to activate a key by number.
  • display_available_keys() no longer prints — returns data only.
    Callers decide how to display it.
  • get_next_key() no longer prints — returns a RotationResult so the
    caller can log/display as appropriate.
  • from_dict() classmethod factory added — enables unit testing without
    a real file on disk.
  • KeyFileParser extracted as a private static method — OCP: adding a new
    file format (e.g. .env) requires adding a new parser, not modifying load().
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotationResult:
    """
    Result of a key rotation attempt.

    Attributes
    ----------
    success    : True if a next key was available and activated.
    key_number : The new key number, or None if exhausted.
    key_value  : The new key value, or None if exhausted.
    """

    success: bool
    key_number: int | None
    key_value: str | None

    @classmethod
    def exhausted(cls) -> RotationResult:
        """Convenience factory for the all-keys-exhausted case."""
        return cls(success=False, key_number=None, key_value=None)


# ---------------------------------------------------------------------------
# API Key Manager
# ---------------------------------------------------------------------------


class APIKeyManager:
    """
    Manages multiple API keys from a configuration file with rotation support.

    Keys are stored in a text file with format: api_N = sk-or-v1-...
    The manager handles key selection, rotation, and exhaustion tracking.

    The manager is pure key-management logic — it performs no I/O to stdin
    or stdout.  Logging and user interaction are the caller's responsibility.

    Usage
    -----
        manager = APIKeyManager("api_keys.txt")
        manager.set_key(1)                        # activate key #1

        client = APIClient(api_key=manager.current_key)

        # On rate-limit hit:
        result = manager.rotate()
        if result.success:
            client = APIClient(api_key=result.key_value)
        else:
            # all keys exhausted
    """

    def __init__(self, keys_file: str = "api_keys.txt") -> None:
        """
        Initialize API Key Manager by loading keys from file.

        Args:
            keys_file: Path to file containing API keys.

        Raises:
            FileNotFoundError: If keys file doesn't exist.
            ValueError:        If no valid keys found in file.
        """
        self.keys_file = keys_file
        self._api_keys: dict[int, str] = {}
        self._current_index: int | None = None
        self._load()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, keys: dict[int, str]) -> APIKeyManager:
        """
        Create an APIKeyManager from an in-memory dict.

        Useful for unit testing — no file required.

        Args:
            keys: Dict mapping key number → key value, e.g. {1: "sk-...", 2: "sk-..."}.

        Returns:
            Configured APIKeyManager instance.

        Raises:
            ValueError: If keys dict is empty.
        """
        if not keys:
            raise ValueError("Cannot create APIKeyManager from empty keys dict.")
        instance = cls.__new__(cls)
        instance.keys_file = "<in-memory>"
        instance._api_keys = dict(keys)
        instance._current_index = None
        return instance

    # ------------------------------------------------------------------
    # Key activation
    # ------------------------------------------------------------------

    def set_key(self, key_number: int) -> str:
        """
        Activate the key with the given number.

        Args:
            key_number: The key number to activate.

        Returns:
            The key value.

        Raises:
            KeyError: If key_number is not in the loaded keys.
        """
        if key_number not in self._api_keys:
            available = sorted(self._api_keys.keys())
            raise KeyError(f"Key #{key_number} not found. Available: {available}")
        self._current_index = key_number
        return self._api_keys[key_number]

    # ------------------------------------------------------------------
    # Key access
    # ------------------------------------------------------------------

    @property
    def current_key(self) -> str | None:
        """
        The currently active API key value, or None if no key is set.

        Replaces get_current_key() — exposed as a property for cleaner access.
        """
        if self._current_index is None:
            return None
        return self._api_keys.get(self._current_index)

    @property
    def current_key_number(self) -> int | None:
        """The currently active key number, or None if no key is set."""
        return self._current_index

    @property
    def available_key_numbers(self) -> list[int]:
        """
        Sorted list of all loaded key numbers.

        Replaces display_available_keys() — returns data only, no printing.
        """
        return sorted(self._api_keys.keys())

    @property
    def remaining_keys_count(self) -> int:
        """
        Number of keys not yet used (i.e. after the current key in sequence).

        Returns len(all keys) if no key is currently active.
        """
        if self._current_index is None:
            return len(self._api_keys)
        available = self.available_key_numbers
        try:
            pos = available.index(self._current_index)
            return len(available) - pos - 1
        except ValueError:
            return 0

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate(self) -> RotationResult:
        """
        Advance to the next available key in sequence.

        Replaces get_next_key() — returns a typed RotationResult instead of
        printing directly.  The caller decides how to log the rotation event.

        Returns:
            RotationResult with success=True and the new key if available,
            or RotationResult.exhausted() if no more keys remain.
        """
        if self._current_index is None:
            return RotationResult.exhausted()

        available = self.available_key_numbers
        try:
            pos = available.index(self._current_index)
        except ValueError:
            return RotationResult.exhausted()

        if pos < len(available) - 1:
            next_index = available[pos + 1]
            self._current_index = next_index
            return RotationResult(
                success=True,
                key_number=next_index,
                key_value=self._api_keys[next_index],
            )

        return RotationResult.exhausted()

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load API keys from the configuration file.

        Expected format: api_N = sk-or-v1-xxxxx
        Lines starting with # are treated as comments.

        Raises:
            FileNotFoundError: If keys file doesn't exist.
            ValueError:        If no valid keys found.
        """
        if not os.path.exists(self.keys_file):
            raise FileNotFoundError(f"API keys file not found: {self.keys_file}")

        self._api_keys = self._parse_key_file(self.keys_file)

        if not self._api_keys:
            raise ValueError("No valid API keys found in the keys file.")

    @staticmethod
    def _parse_key_file(path: str) -> dict[int, str]:
        """
        Parse a key file and return a dict of {key_number: key_value}.

        Expected line format: api_N = sk-or-v1-...
        Blank lines and lines starting with # are ignored.

        Args:
            path: Path to the key file.

        Returns:
            Dict mapping key number to key value.
        """
        keys: dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"api_(\d+)\s*=\s*(.+)", line)
                if match:
                    keys[int(match.group(1))] = match.group(2).strip()
        return keys
