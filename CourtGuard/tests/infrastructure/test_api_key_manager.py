"""
Tests for infrastructure/api_key_manager.py

Uses from_dict() factory — no real api_keys.txt needed.

Covers:
  - set_key() and current_key property
  - rotate() happy path and exhaustion
  - available_key_numbers
  - remaining_keys_count
  - Error cases: empty dict, bad key number
"""

import pytest

from infrastructure.api_key_manager import APIKeyManager, RotationResult


@pytest.fixture
def manager() -> APIKeyManager:
    """Three-key manager starting with no key selected."""
    return APIKeyManager.from_dict({1: "key-one", 2: "key-two", 3: "key-three"})


@pytest.fixture
def single_manager() -> APIKeyManager:
    return APIKeyManager.from_dict({1: "only-key"})


class TestFromDict:
    def test_creates_manager(self, manager):
        assert manager is not None

    def test_raises_on_empty_dict(self):
        with pytest.raises(ValueError, match="empty"):
            APIKeyManager.from_dict({})

    def test_no_key_selected_initially(self, manager):
        assert manager.current_key is None
        assert manager.current_key_number is None


class TestSetKey:
    def test_set_key_activates_key(self, manager):
        manager.set_key(1)
        assert manager.current_key == "key-one"
        assert manager.current_key_number == 1

    def test_set_key_returns_value(self, manager):
        value = manager.set_key(2)
        assert value == "key-two"

    def test_set_key_invalid_raises(self, manager):
        with pytest.raises(KeyError):
            manager.set_key(99)


class TestAvailableKeyNumbers:
    def test_sorted_ascending(self, manager):
        assert manager.available_key_numbers == [1, 2, 3]

    def test_single_key(self, single_manager):
        assert single_manager.available_key_numbers == [1]


class TestRemainingKeysCount:
    def test_remaining_before_selection(self, manager):
        """All keys remaining when none selected."""
        assert manager.remaining_keys_count == 3

    def test_remaining_on_first_key(self, manager):
        manager.set_key(1)
        assert manager.remaining_keys_count == 2

    def test_remaining_on_last_key(self, manager):
        manager.set_key(3)
        assert manager.remaining_keys_count == 0

    def test_remaining_on_middle_key(self, manager):
        manager.set_key(2)
        assert manager.remaining_keys_count == 1


class TestRotate:
    def test_rotate_advances_to_next_key(self, manager):
        manager.set_key(1)
        result = manager.rotate()
        assert result.success is True
        assert result.key_number == 2
        assert result.key_value == "key-two"
        assert manager.current_key == "key-two"

    def test_rotate_successive_calls(self, manager):
        manager.set_key(1)
        manager.rotate()  # → 2
        result = manager.rotate()  # → 3
        assert result.key_number == 3
        assert manager.current_key == "key-three"

    def test_rotate_returns_exhausted_on_last_key(self, manager):
        manager.set_key(3)
        result = manager.rotate()
        assert result.success is False
        assert result.key_number is None
        assert result.key_value is None

    def test_rotate_returns_exhausted_when_no_key_set(self, manager):
        result = manager.rotate()
        assert result.success is False

    def test_single_key_exhausted_immediately(self, single_manager):
        single_manager.set_key(1)
        result = single_manager.rotate()
        assert result.success is False

    def test_rotation_result_exhausted_factory(self):
        result = RotationResult.exhausted()
        assert result.success is False
        assert result.key_number is None
        assert result.key_value is None


class TestPrivateKeysNotMutable:
    def test_api_keys_not_directly_accessible(self, manager):
        """_api_keys is private — external code cannot mutate key state."""
        assert not hasattr(manager, "api_keys")
        assert hasattr(manager, "_api_keys")
