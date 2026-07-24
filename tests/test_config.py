from pathlib import Path

from metadata_extraction import (
    DEFAULT_CONFIG,
    load_config,
)


def test_load_config_nonetype():
    assert load_config(None) == DEFAULT_CONFIG


def test_load_config_newkey():
    config = load_config(
        Path(f"{Path(__file__).resolve().parent}/test_config.json")
    )
    assert config.get("Add-Check") == True


def test_load_config_changekey():
    config = load_config(
        Path(f"{Path(__file__).resolve().parent}/test_config.json")
    )
    assert config.get("sdat_lookup") == False