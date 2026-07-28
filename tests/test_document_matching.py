from metadata_extraction import (
    DEFAULT_CONFIG,
    normalize_ocr_numbers,
    regex_document_type,
)
from pipeline import vote_for_value


def test_site_plan_and_easement_plat_prefers_site_plan():
    """Test site plan and easement plat prefers site plan.
    """
    text = "SITE PLAN AND EASEMENT PLAT FOR LOT 12"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Site Plan"


def test_site_plan_then_sewage_easement_plat_prefers_site_plan():
    """Test site plan then sewage easement plat prefers site plan.
    """
    text = "SITE PLAN, LOT 3\nSEWAGE EASEMENT PLAT"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Site Plan"


def test_forest_conservation_plat_stays_plat():
    """Test forest conservation plat stays plat.
    """
    text = "FOREST CONSERVATION AMENDMENT PLAT"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Plat/Replat"
def test_vote_for_value_empty_values():
    values = []
    fallback = "test"
    assert vote_for_value((val.test for val in values), fallback) == fallback


def test_vote_for_value_all_unknown():
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
    test_text = ["1o0", "IOB", "1Ss", "|liBB"]
    changed_text = []
    for text in test_text:
        changed_text.append(normalize_ocr_numbers(text))

    assert changed_text == ["100", "108", "155", "11188"]

def test_fuzzy_document_type_exact_fast_path_preserves_first_configured_winner():
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
    from metadata_extraction import fuzzy_document_type

    match = fuzzy_document_type("TITLE: WALL CHEGK", {"Wall Check": ["wall check"]})
    assert match is not None
    assert match.label == "Wall Check"
    assert match.score >= 0.75
