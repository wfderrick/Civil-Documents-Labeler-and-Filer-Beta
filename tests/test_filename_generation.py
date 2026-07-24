from metadata_extraction import (
    safe_path_part,
)


def test_safe_path_part_empty():
    assert safe_path_part("", "test") == "test"

def test_safe_path_part_normal():
    assert safe_path_part("pass", "fail") == "pass"

def test_safe_path_part_invalid_chars():
    assert safe_path_part("<>:\"/\\|?*\x00\x1fpass .", "fail") == "pass"

def test_safe_path_part_length():
    assert len(safe_path_part("federalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalism", "fail")) == 140