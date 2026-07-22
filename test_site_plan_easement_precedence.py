from metadata_extraction import regex_document_type, DEFAULT_CONFIG


def test_site_plan_and_easement_plat_prefers_site_plan():
    text = "SITE PLAN AND EASEMENT PLAT FOR LOT 12"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Site Plan"


def test_site_plan_then_sewage_easement_plat_prefers_site_plan():
    text = "SITE PLAN, LOT 3\nSEWAGE EASEMENT PLAT"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Site Plan"


def test_forest_conservation_plat_stays_plat():
    text = "FOREST CONSERVATION AMENDMENT PLAT"
    match = regex_document_type(text, DEFAULT_CONFIG["document_type_regex_rules"])
    assert match is not None
    assert match.label == "Plat/Replat"
