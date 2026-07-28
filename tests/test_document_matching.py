"""Regression tests for document classification and batch metadata voting.

The cases in this file protect important COABarrett naming decisions: Site Plans
must outrank easement wording, OCR digit confusions must be repaired, the fast exact
classifier must preserve legacy tie order, and unknown values must not win a batch vote."""

from metadata_extraction import (
    DEFAULT_CONFIG,
    normalize_ocr_numbers,
    regex_document_type,
)
from pipeline import vote_for_value


def test_site_plan_and_easement_plat_prefers_site_plan():
    """Protect classification precedence for a combined title.

    A drawing titled “SITE PLAN AND EASEMENT PLAT” is filed as a Site Plan. This case
    ensures the more useful primary drawing type wins over secondary easement wording."""
    text = "SITE PLAN AND EASEMENT PLAT FOR LOT 12"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Site Plan"


def test_site_plan_then_sewage_easement_plat_prefers_site_plan():
    """Protect Site Plan classification when easement language appears on another line.

    OCR line breaks must not make “SEWAGE EASEMENT PLAT” override the drawing's primary
    “SITE PLAN” title."""
    text = "SITE PLAN, LOT 3\nSEWAGE EASEMENT PLAT"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Site Plan"


def test_forest_conservation_plat_stays_plat():
    """Protect a genuine plat title from the Site Plan precedence rules.

    “FOREST CONSERVATION AMENDMENT PLAT” is a Plat/Replat, so the classifier must not
    apply the broader Site Plan rule merely because the word “plan/plat” is similar."""
    text = "FOREST CONSERVATION AMENDMENT PLAT"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Plat/Replat"
def test_vote_for_value_empty_values():
    """Verify that an empty batch vote returns the caller's fallback.

    This represents a packet where no document supplied a candidate for the shared
    field; vote_for_value() must not index an empty counter or invent metadata."""
    values = []
    fallback = "test"
    assert vote_for_value((val.test for val in values), fallback) == fallback


def test_vote_for_value_all_unknown():
    """Verify that OCR placeholder values do not count as evidence.

    Even many “Unknown” votes must lose to the fallback because placeholders describe
    missing data, not agreement among drawings."""
    values = [
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
    ]
    fallback = "test"
    assert vote_for_value((val for val in values), fallback) == fallback

def test_vote_for_value_one_known():
    """Verify that one real metadata value beats any number of placeholders.

    A single readable sheet may be the only source for a lot or address, and Batch mode
    must preserve that useful value."""
    values = [
            "Unknown",
            "Unknown",
            "test",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
        ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"

def test_vote_for_value_tie():
    """Protect deterministic behavior when two known values receive equal votes.

    The first value to reach the counter remains the display winner. Keeping this stable
    prevents equal evidence from changing filenames unpredictably between versions."""
    values = [
            "Unknown",
            "Unknown",
            "test",
            "wrong",
            "test",
            "wrong",
            "Unknown",
            ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"

def test_vote_for_value_first_val():
    """Verify that a known value at the beginning of the packet is retained.

    This guards against iterator or placeholder filtering errors that could accidentally
    skip the first drawing's valid metadata."""
    values = [
            "test",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown"   
        ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"

def test_vote_for_value_last_val():
    """Verify that a known value at the end of the packet is still discovered.

    The vote loop must consume the complete iterable rather than stopping after earlier
    unknown placeholders."""
    values = [
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "test",
        ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"

def test_normalize_ocr_numbers():
    """Protect corrections for characters OCR commonly confuses with digits.

    Examples such as o/0, I/1, S/5, and vertical bars are normalized before Tax ID and
    other numeric patterns are extracted."""
    test_text = ["1o0", "IOB", "1Ss", "|liBB"]
    changed_text = []
    for text in test_text:
        changed_text.append(normalize_ocr_numbers(text))

    assert changed_text == ["100", "108", "155", "11188"]

def test_fuzzy_document_type_exact_fast_path_preserves_first_configured_winner():
    """Protect behavior while using the fast exact-keyword optimization.

    When two configured types contain the same exact keyword, the first configured type
    must still win. This prevents a performance change from altering legacy filenames."""
    from metadata_extraction import fuzzy_document_type

    keywords = {
        "First": ["site plan"],
        "Second": ["site plan"],
    }
    match = fuzzy_document_type("TITLE: SITE PLAN", keywords)
    assert match is not None
    assert match.label == "First"
    assert match.score == 1.0


def test_fuzzy_document_type_still_accepts_ocr_typo():
    """Confirm that the optimized classifier still falls back to fuzzy matching.

    A misspelled OCR title such as “WALL CHEGK” should still classify as Wall Check even
    though the fast exact phrase path cannot match it."""
    from metadata_extraction import fuzzy_document_type

    match = fuzzy_document_type("TITLE: WALL CHEGK", {"Wall Check": ["wall check"]})
    assert match is not None
    assert match.label == "Wall Check"
    assert match.score >= 0.75
