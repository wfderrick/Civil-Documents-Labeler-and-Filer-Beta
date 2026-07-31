"""Unit tests for Mass Scan's single-document metadata pipeline.

These tests do not import Flask. They verify that independent jobs use only their
own OCR evidence and conservative SDAT lookups.
"""

from dataclasses import replace

import pipeline
from metadata_extraction import ExtractedMetadata
from sdat import LOOKUP_DOCUMENT_TYPE


def metadata(**changes) -> ExtractedMetadata:
    """Return a complete, predictable metadata object for pipeline tests."""
    base = ExtractedMetadata(
        lot="104",
        address="123 MAIN RD",
        project_code="OCR-CODE",
        document_type="Site Plan",
        tax_map="12",
        parcel="34",
        tax_id="01-123456",
    )
    return replace(base, **changes)


def test_independent_metadata_skips_all_sdat_when_disabled(monkeypatch):
    """Disabling SDAT should return OCR metadata without making network lookups."""
    monkeypatch.setattr(pipeline, "extract_metadata", lambda *_a, **_k: metadata())
    monkeypatch.setattr(
        pipeline,
        "_lookup_by_tax_id",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Tax ID lookup ran")),
    )
    monkeypatch.setattr(
        pipeline,
        "lookup_maryland_property_by_address",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Address lookup ran")),
    )

    result = pipeline.extract_independent_document_metadata(
        {"ocr_text": "text", "ocr_pages": []},
        {"sdat_lookup": False},
        "MASS-PROJECT",
        "Document",
    )

    assert result.project_code == "MASS-PROJECT"
    assert result.address == "123 MAIN RD"


def test_independent_metadata_prefers_direct_tax_id_record(monkeypatch):
    """A successful Tax ID lookup should enrich the PDF and avoid address lookup."""
    original = metadata()
    authoritative = metadata(address="123 MAIN ROAD", tax_id="01-999999")
    record = {"record": "tax-id"}

    monkeypatch.setattr(pipeline, "extract_metadata", lambda *_a, **_k: original)
    monkeypatch.setattr(pipeline, "_lookup_by_tax_id", lambda *_a, **_k: [record])
    monkeypatch.setattr(
        pipeline,
        "metadata_from_sdat_record",
        lambda seed, selected: authoritative if selected is record else seed,
    )
    monkeypatch.setattr(
        pipeline,
        "lookup_maryland_property_by_address",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("Address lookup should not run after Tax ID success")
        ),
    )

    result = pipeline.extract_independent_document_metadata(
        {"ocr_text": "text", "ocr_pages": []},
        {"sdat_lookup": True, "default_county": "Calvert"},
        "AA",
        "Document",
    )

    assert result.address == "123 MAIN ROAD"
    assert result.tax_id == "01-999999"
    assert result.project_code == "AA"


def test_independent_metadata_uses_address_after_tax_id_miss(monkeypatch):
    """A failed direct lookup should fall back to one confidently matched address."""
    original = metadata()
    selected_record = {"record": "address"}
    authoritative = metadata(address="123 MAIN ROAD", tax_id="01-777777")

    monkeypatch.setattr(pipeline, "extract_metadata", lambda *_a, **_k: original)
    monkeypatch.setattr(pipeline, "_lookup_by_tax_id", lambda *_a, **_k: [])
    monkeypatch.setattr(
        pipeline, "lookup_maryland_property_by_address", lambda *_a, **_k: [selected_record]
    )
    monkeypatch.setattr(
        pipeline, "_confident_unique_address_record", lambda *_a, **_k: selected_record
    )
    monkeypatch.setattr(
        pipeline, "metadata_from_sdat_record", lambda *_a, **_k: authoritative
    )

    result = pipeline.extract_independent_document_metadata(
        {"ocr_text": "text", "ocr_pages": []},
        {"sdat_lookup": True, "default_county": "Calvert"},
        "AA",
        "Document",
    )

    assert result.tax_id == "01-777777"
    assert result.project_code == "AA"


def test_independent_metadata_rejects_ambiguous_address_and_never_uses_map_parcel(
    monkeypatch,
):
    """Mass mode must keep OCR values when address results are ambiguous."""
    original = metadata(tax_id="")
    ambiguous_records = [
        {"address": "123 MAIN RD", "account": "one"},
        {"address": "123 MAIN RD", "account": "two"},
    ]

    monkeypatch.setattr(pipeline, "extract_metadata", lambda *_a, **_k: original)
    monkeypatch.setattr(
        pipeline, "lookup_maryland_property_by_address", lambda *_a, **_k: ambiguous_records
    )
    monkeypatch.setattr(
        pipeline, "_confident_unique_address_record", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        pipeline,
        "lookup_maryland_property_records",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("Mass Scan must not use map/parcel fallback")
        ),
    )

    result = pipeline.extract_independent_document_metadata(
        {"ocr_text": "text", "ocr_pages": []},
        {"sdat_lookup": True, "default_county": "Calvert"},
        "AA",
        "Document",
    )

    assert result == replace(original, project_code="AA")


def test_independent_lookup_document_bypasses_sdat_and_project_override(monkeypatch):
    """SDAT printouts remain reference records instead of normal filing documents."""
    lookup = metadata(document_type=LOOKUP_DOCUMENT_TYPE)
    monkeypatch.setattr(pipeline, "extract_metadata", lambda *_a, **_k: lookup)
    monkeypatch.setattr(
        pipeline,
        "_lookup_by_tax_id",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Lookup PDF was enriched")),
    )

    result = pipeline.extract_independent_document_metadata(
        {"ocr_text": "SDAT", "ocr_pages": []},
        {"sdat_lookup": True},
        "MASS-PROJECT",
        "Document",
    )

    assert result is lookup
    assert result.project_code == "OCR-CODE"
