from pathlib import Path

from metadata_extraction import (
    DEFAULT_CONFIG,
    load_config,
    normalize_ocr_numbers,
)


def test_normalize_ocr_numbers():
    test_text = ["1o0", "IOB", "1Ss", "|liBB"]
    changed_text = []
    for text in test_text:
        changed_text.append(normalize_ocr_numbers(text))

    assert changed_text == ["100", "108", "155", "11188"]


def test_load_config_nonetype():
    assert load_config(None) == DEFAULT_CONFIG


def test_load_config_newkey():
    config = load_config(
        Path(f"{Path(__file__).resolve().parent}/test_config.json")
    )
    assert config.get("Add-Check") == True  # noqa: E712


def test_load_config_changekey():
    config = load_config(
        Path(f"{Path(__file__).resolve().parent}/test_config.json")
    )
    assert config.get("sdat_lookup") == False  # noqa: E712

    