"""Regression tests for Windows-safe folder and PDF name components.

Filing ultimately uses these strings as paths. The tests ensure blank input receives
a fallback, normal text is preserved, Windows-forbidden characters are removed, and
extremely long OCR values are capped before a path is created."""

from metadata_extraction import (
    safe_path_part,
)


def test_safe_path_part_empty():
    """Verify that an empty OCR value is replaced with the supplied path fallback.

    Windows cannot file a document under a blank component, so the function must return
    a usable default rather than an empty string."""
    assert safe_path_part("", "test") == "test"

def test_safe_path_part_normal():
    """Verify that already-safe text is not changed.

    Sanitization should protect paths without unnecessarily rewriting valid project or
    document names that reviewers expect to recognize."""
    assert safe_path_part("pass", "fail") == "pass"

def test_safe_path_part_invalid_chars():
    """Verify removal of characters Windows forbids in file and folder names.

    The test includes reserved punctuation, control characters, and trailing space/dot
    content, while confirming the meaningful word remains."""
    assert safe_path_part("<>:\"/\\|?*\x00\x1fpass .", "fail") == "pass"

def test_safe_path_part_length():
    """Protect the 140-character cap for one generated path component.

    OCR can return very long title strings. Capping each component reduces the chance
    that the final Windows destination exceeds filesystem path limits."""
    assert len(safe_path_part("federalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalismfederalism", "fail")) == 140
