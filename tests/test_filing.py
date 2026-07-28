"""Integration-style tests for the final PDF filing service.

These tests exercise the decisions that can affect original company files: In-Place
mode must not rename or move them, copy mode must preserve the source, PDF metadata
and optional OCR text must be written, and duplicate destination names must be made
unique rather than overwritten."""

from pathlib import Path

import pikepdf

import document_service
from document_service import file_document_to_output


def test_in_place_filing_preserves_path_and_name(tmp_path, monkeypatch):
    """Protect the defining safety promise of In-Place mode.

    The original PDF must stay at the same path and keep its name while metadata is
    written into it. Optional OCR text is saved beside the source, and no output folder
    should be created because normal filing is bypassed."""
    source = tmp_path / "Original Name.pdf"
    source.write_bytes(b"original-pdf")
    document = {
        "source_path": str(source),
        "source_name": source.name,
        "ocr_text": "searchable text",
        "metadata": {"address": "2432 COMPTROLLERS CT"},
        "folder_name": "Generated Folder",
        "file_name": "Generated Name.pdf",
    }

    def fake_write_pdf_metadata(path: Path, _document):
        """Stand in for the real PDF metadata writer during this filesystem test.

        Appending marker bytes makes the write observable without requiring a valid PDF,
        allowing the test to focus on In-Place path and naming behavior."""
        path.write_bytes(path.read_bytes() + b"-with-metadata")

    monkeypatch.setattr(
        document_service, "write_pdf_metadata", fake_write_pdf_metadata
    )

    filed = file_document_to_output(
        document,
        tmp_path / "unused-output",
        in_place=True,
        save_text=True,
    )

    assert source.exists()
    assert source.name == "Original Name.pdf"
    assert source.read_bytes() == b"original-pdf-with-metadata"
    assert filed["filed_path"] == str(source)
    assert filed["status"] == "filed"
    assert (
        source.with_suffix(".txt").read_text(encoding="utf-8")
        == "searchable text"
    )
    assert not (tmp_path / "unused-output").exists()


def test_regular_filing_saves_ocr_text_and_copies(tmp_path, monkeypatch):
    """Verify normal copy-mode filing from source to a chosen destination.

    The source must remain unchanged, the copied PDF must receive metadata, and the OCR
    text sidecar must be written beside the destination using the selected folder/name."""
    source = tmp_path / "Original Name.pdf"
    source.write_bytes(b"original-pdf")
    document = {
        "source_path": str(source),
        "source_name": source.name,
        "ocr_text": "searchable text",
        "metadata": {"address": "2432 COMPTROLLERS CT"},
        "folder_name": "Generated Folder",
        "file_name": "Generated Name.pdf",
    }

    def fake_write_pdf_metadata(path: Path, _document):
        """Replace PDF metadata writing with an observable byte marker for copy-mode testing.

        This isolates path, copy, and text-sidecar behavior from pikepdf internals while still
        proving that metadata writing targets the new destination rather than the source."""
        path.write_bytes(path.read_bytes() + b"-with-metadata")

    monkeypatch.setattr(
        document_service, "write_pdf_metadata", fake_write_pdf_metadata
    )

    filed = file_document_to_output(
        document,
        tmp_path,
        in_place=False,
        save_text=True,
        copy_file=True,
        folder_name="Selected Folder",
        file_name="Selected File",
    )
    assert (
        source.exists()
        and (tmp_path / "Selected Folder\\Selected File.pdf").exists()
    )
    assert source.name == "Original Name.pdf"
    assert (
        source.read_bytes() == b"original-pdf"
        and (tmp_path / "Selected Folder\\Selected File.pdf").read_bytes()
        == b"original-pdf-with-metadata"
    )
    assert filed["filed_path"] == str(
        tmp_path / "Selected Folder\\Selected File.pdf"
    )
    assert filed["status"] == "filed"
    assert (tmp_path / "Selected Folder\\Selected File.pdf").with_suffix(
        ".txt"
    ).read_text(encoding="utf-8") == "searchable text"


def test_updated_xml_metadata(tmp_path, monkeypatch):
    """Inspect a filed PDF to confirm application metadata reaches its XMP packet.

    After copying a real fixture PDF, the test reopens it with pikepdf and checks that the
    reviewed address is present alongside the expected destination and OCR text file."""
    source = Path(
        "C:\\Users\\wderrick\\Documents\\GitHub\\COABarrett File Identifier and Sorter - Version 2.4\\tests\\Site Plan - Lot 104.pdf"
    )
    document = {
        "source_path": str(source),
        "source_name": source.name,
        "ocr_text": "searchable text",
        "metadata": {"address": "2432 COMPTROLLERS CT"},
        "folder_name": "Generated Folder",
        "file_name": "Generated Name.pdf",
    }

    filed = file_document_to_output(
        document,
        tmp_path,
        in_place=False,
        save_text=True,
        copy_file=True,
        folder_name="Selected Folder",
        file_name="Selected File",
    )

    pdf = pikepdf.Pdf.open(
        str(tmp_path / "Selected Folder\\Selected File.pdf")
    )
    test_dict = {}
    with pdf.open_metadata() as meta:
        for key, value in meta.items():
            print(f"{key},{value}")
            test_dict[key] = value

    assert (
        source.exists()
        and (tmp_path / "Selected Folder\\Selected File.pdf").exists()
    )
    assert source.name == "Site Plan - Lot 104.pdf"
    assert filed["filed_path"] == str(
        tmp_path / "Selected Folder\\Selected File.pdf"
    )
    assert filed["status"] == "filed"
    assert (tmp_path / "Selected Folder\\Selected File.pdf").with_suffix(
        ".txt"
    ).read_text(encoding="utf-8") == "searchable text"
    assert test_dict.get("Address") == "2432 COMPTROLLERS CT"


def test_filing_with_duplicate_doc_name(tmp_path, monkeypatch):
    """Protect existing output files when two documents request the same destination name.

    The first receives the normal name and the second receives “(2)”. Both source files
    are moved, both outputs contain their own data, and neither result is overwritten."""
    source1 = tmp_path / "Original Name1.pdf"
    source2 = tmp_path / "Original Name2.pdf"
    source1.write_bytes(b"original1-pdf")
    source2.write_bytes(b"original2-pdf")
    document1 = {
        "source_path": str(source1),
        "source_name": source1.name,
        "ocr_text": "searchable text",
        "metadata": {"address": "2432 COMPTROLLERS CT", "lot": "67"},
        "folder_name": "Generated Folder",
        "file_name": "Generated Name.pdf",
    }

    document2 = {
        "source_path": str(source2),
        "source_name": source1.name,
        "ocr_text": "searchable text",
        "metadata": {"address": "2432 COMPTROLLERS CT", "lot": "67"},
        "folder_name": "Generated Folder",
        "file_name": "Generated Name.pdf",
    }

    def fake_write_pdf_metadata(path: Path, _document):
        """Add a visible metadata marker to each duplicate-name test PDF.

        Using the same lightweight stand-in for both documents makes it possible to verify
        that unique-name handling preserves each source's separate contents."""
        path.write_bytes(path.read_bytes() + b"-with-metadata")

    monkeypatch.setattr(
        document_service, "write_pdf_metadata", fake_write_pdf_metadata
    )

    filed1 = file_document_to_output(
        document1,
        tmp_path,
        in_place=False,
        save_text=False,
    )
    filed2 = file_document_to_output(
        document2,
        tmp_path,
        in_place=False,
        save_text=False,
    )

    assert not source1.exists() and not source2.exists()
    assert (tmp_path / "Generated Folder\\Generated Name.pdf").exists() and (
        tmp_path / "Generated Folder\\Generated Name (2).pdf"
    ).exists()
    assert (
        tmp_path / "Generated Folder\\Generated Name.pdf"
    ).read_bytes() == b"original1-pdf-with-metadata" and (
        tmp_path / "Generated Folder\\Generated Name (2).pdf"
    ).read_bytes() == b"original2-pdf-with-metadata"
    assert filed1["filed_path"] == str(
        tmp_path / "Generated Folder\\Generated Name.pdf"
    ) and filed2["filed_path"] == str(
        tmp_path / "Generated Folder\\Generated Name (2).pdf"
    )
    assert filed1["status"] == "filed" and filed2["status"] == "filed"
    assert (
        not (tmp_path / "Original Name1.pdf").exists()
        and not (tmp_path / "Original Name2.pdf").exists()
    )
