from pathlib import Path

import document_service


def test_in_place_filing_preserves_path_and_name(tmp_path, monkeypatch):
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
        path.write_bytes(path.read_bytes() + b"-with-metadata")

    monkeypatch.setattr(document_service, "write_pdf_metadata", fake_write_pdf_metadata)

    filed = document_service.file_document_to_output(
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
    assert source.with_suffix(".txt").read_text(encoding="utf-8") == "searchable text"
    assert not (tmp_path / "unused-output").exists()
