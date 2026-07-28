"""Regression tests for configuration loading.

These tests protect the rule that the application starts with a complete built-in
configuration when no file is supplied, while a user config.json may add new keys
or override existing behavior such as SDAT lookup."""

from pathlib import Path

from metadata_extraction import (
    DEFAULT_CONFIG,
    load_config,
)


def test_load_config_nonetype():
    """Protect the application's no-config fallback.

    Passing None should return the complete built-in DEFAULT_CONFIG, allowing the app
    to scan even when the reviewer has not selected a custom config.json."""
    assert load_config(None) == DEFAULT_CONFIG


def test_load_config_newkey():
    """Confirm that a custom configuration may introduce a new setting.

    The fixture contains Add-Check, so this test proves load_config() merges keys that
    do not exist in DEFAULT_CONFIG instead of silently discarding them."""
    config = load_config(
        Path(f"{Path(__file__).resolve().parent}/test_config.json")
    )
    assert config.get("Add-Check") == True


def test_load_config_changekey():
    """Confirm that a custom configuration can override a built-in setting.

    The fixture disables sdat_lookup. The assertion protects the reviewer's ability to
    turn an existing application feature off through config.json."""
    config = load_config(
        Path(f"{Path(__file__).resolve().parent}/test_config.json")
    )
    assert config.get("sdat_lookup") == False
