"""Tests for the separate Batch and Mass Scan metadata workflows.

These tests focus on the architectural boundary that matters most: Mass Scan must
process each PDF as an independent job and must never use packet voting, duplicate
classification repair, or shared metadata merging.
"""

from dataclasses import replace
import pytest

pytest.importorskip("flask")

import app as app_module
from metadata_extraction import ExtractedMetadata
from sdat import LOOKUP_DOCUMENT_TYPE


def metadata(**changes) -> ExtractedMetadata:
    """Return a complete, predictable metadata object for scan workflow tests."""
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


def mass_settings(**changes) -> dict:
    """Build the small settings dictionary required by ``scan_mass``."""
    settings = {
        "dpi": 300,
        "lang": "en",
        "ocr_threads_per_worker": 4,
        "ocr_device": "cpu",
        "gpu_device_id": 0,
        "project_code": "AA",
        "document_type": "Document",
        "section": "",
    }
    settings.update(changes)
    return settings


def test_scan_mass_returns_empty_when_folder_has_no_pdfs(tmp_path, monkeypatch):
    """An empty folder should finish immediately without starting OCR."""
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("OCR must not run when no PDFs exist")

    monkeypatch.setattr(app_module, "ocr_pdf_batch", fail_if_called)

    assert app_module.scan_mass(tmp_path, object(), {}, mass_settings()) == []
    assert called is False


def test_scan_mass_processes_every_pdf_as_an_independent_job(tmp_path, monkeypatch):
    """Each PDF should be OCRed and sent separately to the independent pipeline."""
    (tmp_path / "b.pdf").write_bytes(b"pdf")
    (tmp_path / "a.pdf").write_bytes(b"pdf")
    (tmp_path / "ignore.txt").write_text("not a PDF", encoding="utf-8")

    ocr_calls: list[str] = []
    extraction_calls: list[str] = []

    def fake_ocr(paths, **_kwargs):
        assert len(paths) == 1
        path = paths[0]
        ocr_calls.append(path.name)
        return [{
            "source_path": str(path),
            "source_name": path.name,
            "ocr_text": f"OCR FOR {path.name}",
            "ocr_pages": [],
        }]

    def fake_independent(scanned_document, **_kwargs):
        extraction_calls.append(scanned_document["source_name"])
        return metadata(address=f"{len(extraction_calls)} MAIN RD", tax_id="")

    monkeypatch.setattr(app_module, "ocr_pdf_batch", fake_ocr)
    monkeypatch.setattr(
        app_module, "extract_independent_document_metadata", fake_independent
    )

    documents = app_module.scan_mass(tmp_path, object(), {}, mass_settings())

    assert ocr_calls == ["a.pdf", "b.pdf"]
    assert extraction_calls == ["a.pdf", "b.pdf"]
    assert [doc["source_name"] for doc in documents] == ["a.pdf", "b.pdf"]
    assert documents[0]["metadata"]["address"] == "1 MAIN RD"
    assert documents[1]["metadata"]["address"] == "2 MAIN RD"


def test_scan_mass_does_not_call_batch_vote_or_merge(tmp_path, monkeypatch):
    """Regression guard: Mass Scan must not re-enter either Batch-only helper."""
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"pdf")

    monkeypatch.setattr(
        app_module,
        "ocr_pdf_batch",
        lambda *_args, **_kwargs: [{
            "source_path": str(pdf),
            "source_name": pdf.name,
            "ocr_text": "SITE PLAN 123 MAIN RD",
            "ocr_pages": [],
        }],
    )
    monkeypatch.setattr(
        app_module,
        "extract_independent_document_metadata",
        lambda **_kwargs: metadata(tax_id=""),
    )
    monkeypatch.setattr(
        app_module,
        "choose_batch_metadata_by_vote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Mass Scan called batch voting")
        ),
    )
    monkeypatch.setattr(
        app_module,
        "merge_batch_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Mass Scan called batch merging")
        ),
    )

    documents = app_module.scan_mass(tmp_path, object(), {}, mass_settings())
    assert len(documents) == 1
    assert documents[0]["metadata"]["address"] == "123 MAIN RD"


def test_scan_mass_publishes_completed_documents_in_order(tmp_path, monkeypatch):
    """The browser callback should receive each completed record before the next result."""
    for name in ("first.pdf", "second.pdf"):
        (tmp_path / name).write_bytes(b"pdf")

    monkeypatch.setattr(
        app_module,
        "ocr_pdf_batch",
        lambda paths, **_kwargs: [{
            "source_path": str(paths[0]),
            "source_name": paths[0].name,
            "ocr_text": paths[0].name,
            "ocr_pages": [],
        }],
    )
    monkeypatch.setattr(
        app_module,
        "extract_independent_document_metadata",
        lambda **_kwargs: metadata(tax_id=""),
    )

    published: list[str] = []
    messages: list[str] = []
    documents = app_module.scan_mass(
        tmp_path,
        object(),
        {},
        mass_settings(),
        progress_callback=messages.append,
        document_ready_callback=lambda doc: published.append(doc["source_name"]),
    )

    assert published == [doc["source_name"] for doc in documents]
    assert published == ["first.pdf", "second.pdf"]
    assert messages[-1] == "Ready for review: second.pdf"


def test_scan_mass_lookup_document_is_not_given_normal_filing_names(
    tmp_path, monkeypatch
):
    """Lookup reference PDFs should remain lookup-only and skip normal naming logic."""
    pdf = tmp_path / "sdat.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(
        app_module,
        "ocr_pdf_batch",
        lambda *_args, **_kwargs: [{
            "source_path": str(pdf),
            "source_name": pdf.name,
            "ocr_text": "SDAT",
            "ocr_pages": [],
        }],
    )
    monkeypatch.setattr(
        app_module,
        "extract_independent_document_metadata",
        lambda **_kwargs: metadata(document_type=LOOKUP_DOCUMENT_TYPE),
    )

    document = app_module.scan_mass(tmp_path, object(), {}, mass_settings())[0]

    assert document["is_lookup_document"] is True
    assert document["status"] == "lookup_only"


