"""Document-domain operations used after OCR. The functions in this module merge metadata, synchronize batch fields, perform SDAT-assisted updates, create suggested filenames, and apply user edits without coupling those operations to Flask routes.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from metadata_extraction import ExtractedMetadata, safe_path_part, unique_path
from pdf_processing import write_pdf_metadata
from sdat import (
    LOOKUP_DOCUMENT_TYPE,
    SDAT_METADATA_FIELDS,
    enrich_metadata_with_sdat,
    lookup_by_tax_id,
    lookup_maryland_property_by_address,
    metadata_from_sdat_record,
)
from state_store import load_config_from_state

REQUIRED_METADATA_FIELDS = ("lot", "address", "project_code", "document_type")
OPTIONAL_METADATA_FIELDS = ("tax_map", "parcel", "tax_id", "section")


def is_unknown(value: str) -> bool:
    """The is_unknown() function returns True if the value parameter is unknown
    and False otherwise. It checks if it doesn't have a value at all first, then
    if the value begins with the string unknown, and finally if the value is
    Project or Document. If any of those are true it returns True."""
    return (
        not value
        or value.lower().startswith("unknown")
        or value in {"Project", "Document"}
    )


def suggested_folder(metadata: dict[str, str]) -> str:
    """The suggested_folder() function returns a string with a suggested folder name
    based on the metadata parameter. The folder name follows the naming conventions
    Lot # - Address(ex: Lot 1 - 34 Jibsail Street). After the lot and address
    information are pulled from the metadata parameter they are passed into the
    safe_path_part() function imported from pipeline.py to ensure they contain only
    allowed characters and remove extra spaces."""
    return safe_path_part(
        f"Lot {metadata.get('lot', '')} - {metadata.get('address', '')}",
        "Unknown Lot - Unknown Address",
    )


def suggested_filename(metadata: dict[str, str], source_name: str) -> str:
    """The suggested_filename() function returns a string with a suggested file name
    based on the metadata parameter. The file name follows the naming conventions
    Document Type - Lot #(ex: Site Plan - Lot 1). After the lot and address
    information are pulled from the metadata parameter they are passed into the
    safe_path_part() function imported from pipeline.py to ensure they contain only
    allowed characters and remove extra spaces."""
    stem = f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}"
    return safe_path_part(stem, Path(source_name).stem) + ".pdf"


def document_status(metadata: dict[str, str]) -> str:
    """The document_status() function returns either needs_review or ready based
    on the metadata parameter. If any of the fields in REQUIRED_METADATA_FIELDS
    are empty in the metadata parameter the function returns needs_review.
    Otherwise ready is returned.
    """
    return (
        "needs_review"
        if any(
            is_unknown(metadata.get(field, "")) for field in REQUIRED_METADATA_FIELDS
        )
        else "ready"
    )


def sync_document_metadata(
    document: dict[str, Any],
    auto_folder: bool = False,
    auto_file_name: bool = False,
) -> dict[str, Any]:
    """The sync_document_metadata() function returns a document with updated
    folder name, file name, and status. The metadata for the document is stored
    in the metadata variable. If the auto_folder parameter is True or the
    document parameter doesn't contain a folder_name key the document folder_name key is created or changed to match the output of the suggested_folder() function called on the metadata variable. The same is done for file name except it is based on the auto_file_name parameter the file_name key and the suggested_filename() function. The status key in document is set everytime using the document_status() function unless it is a lookup only document.
    """
    metadata = document.setdefault("metadata", {})
    source_name = str(document.get("source_name", "document.pdf"))

    if auto_folder or "folder_name" not in document:
        document["folder_name"] = suggested_folder(metadata)

    if auto_file_name or "file_name" not in document:
        document["file_name"] = suggested_filename(metadata, source_name)

    document["status"] = (
        "lookup_only"
        if document.get("is_lookup_document")
        else document_status(metadata)
    )
    return document


def find_document(state: dict[str, Any], document_id: str) -> dict[str, Any] | None:
    """The find_document() function returns the document in the state parameter
    with id matching the document_id parameter or None if there aren't any
    documents in the state parameter with an id that matches the document_id
    parameter."""
    return next(
        (doc for doc in state.get("documents", []) if doc.get("id") == document_id),
        None,
    )


def metadata_from_dict(metadata: dict[str, Any]) -> ExtractedMetadata:
    """Metadata from dict.
    
    Args:
        metadata: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    return ExtractedMetadata(
        lot=str(metadata.get("lot", "Unknown Lot")),
        address=str(metadata.get("address", "Unknown Address")),
        project_code=str(metadata.get("project_code", "Project")),
        document_type=str(metadata.get("document_type", "Document")),
        tax_map=str(metadata.get("tax_map", "")),
        parcel=str(metadata.get("parcel", "")),
        tax_id=str(metadata.get("tax_id", "")),
        section=str(metadata.get("section", "")),
        **{
            field: str(metadata.get(field, "") or "")
            for field in SDAT_METADATA_FIELDS
        },
    )


def refresh_property_fields_from_sdat(
    state: dict[str, Any],
    documents: list[dict[str, Any]],
    changed_field: str,
) -> dict[str, str] | None:
    """Validate one property edit with SDAT and update the supplied documents.

    The caller chooses the synchronization scope. Batch scanning passes every
    permanent document because those files represent one property. Mass
    scanning passes only the document being edited so separate jobs can never
    overwrite one another.
    """
    documents = [doc for doc in documents if not doc.get("is_lookup_document")]
    if not documents:
        return None

    config = load_config_from_state(state)
    if not config.get("sdat_lookup", True):
        return None

    seed = metadata_from_dict(documents[0].get("metadata", {}))
    county = str(config.get("default_county", "") or "")
    records: list[dict[str, Any]] = []

    if changed_field == "tax_id":
        records = lookup_by_tax_id(seed.tax_id, county)
    elif changed_field == "address":
        records = lookup_maryland_property_by_address(
            seed.address, county=county, limit=25
        )
    else:
        query_seed = replace(seed, tax_id="", address="Unknown Address")
        enriched = enrich_metadata_with_sdat(query_seed, "", config)
        if enriched == query_seed:
            return None

        values = {
            field: getattr(enriched, field)
            for field in (
                "lot",
                "address",
                "tax_map",
                "parcel",
                "tax_id",
                "section",
                *SDAT_METADATA_FIELDS,
            )
        }
        for target_document in documents:
            target_document["metadata"].update(values)
            sync_document_metadata(
                target_document, auto_folder=True, auto_file_name=True
            )
        return values

    if not records:
        return None

    enriched = metadata_from_sdat_record(seed, records[0])
    values = {
        field: getattr(enriched, field)
        for field in (
            "lot",
            "address",
            "tax_map",
            "parcel",
            "tax_id",
            "section",
            *SDAT_METADATA_FIELDS,
        )
    }
    for target_document in documents:
        target_document["metadata"].update(values)
        sync_document_metadata(
            target_document, auto_folder=True, auto_file_name=True
        )
    return values


def refresh_batch_property_fields_from_sdat(
    state: dict[str, Any], changed_field: str
) -> dict[str, str] | None:
    """Validate a property edit and synchronize all permanent batch files."""
    return refresh_property_fields_from_sdat(
        state,
        list(state.get("documents", [])),
        changed_field,
    )


def apply_document_update(
    state: dict[str, Any], document: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Apply edits using the synchronization rules for the active scan mode.

    Batch mode intentionally shares property-level metadata across the batch.
    Mass mode treats every PDF as a separate job, so edits and SDAT results are
    restricted to the selected document.
    """
    metadata = document["metadata"]
    scan_mode = str(state.get("settings", {}).get("scan_mode", "batch")).lower()
    mass_mode = scan_mode == "mass"

    property_field_names = (
        "lot",
        "address",
        "tax_map",
        "parcel",
        "tax_id",
        "section",
        "project_code",
    )
    property_updates = {
        field: payload[field] for field in property_field_names if field in payload
    }
    changed_field = payload.get("changed_field", "")

    if property_updates:
        update_targets = (
            [document]
            if mass_mode
            else list(state.get("documents", []))
        )
        for target_document in update_targets:
            target_document["metadata"].update(property_updates)
            sync_document_metadata(
                target_document, auto_folder=True, auto_file_name=True
            )

    if changed_field in {"tax_map", "parcel", "tax_id", "address", "section"}:
        if mass_mode:
            refresh_property_fields_from_sdat(state, [document], changed_field)
        else:
            refresh_batch_property_fields_from_sdat(state, changed_field)

    # Non-property fields, including document type, always belong only to the
    # selected document in both scan modes.
    for field in (*REQUIRED_METADATA_FIELDS, *OPTIONAL_METADATA_FIELDS):
        if field in payload and field not in property_updates:
            metadata[field] = payload[field]

    document["is_lookup_document"] = (
        metadata.get("document_type") == LOOKUP_DOCUMENT_TYPE
    )

    if payload.get("auto_folder"):
        document["folder_name"] = suggested_folder(metadata)
    elif "folder_name" in payload:
        document["folder_name"] = safe_path_part(
            payload["folder_name"], suggested_folder(metadata)
        )

    if payload.get("auto_file_name"):
        document["file_name"] = suggested_filename(metadata, document["source_name"])
    elif "file_name" in payload:
        stem = Path(payload["file_name"]).stem
        document["file_name"] = (
            safe_path_part(stem, Path(document["source_name"]).stem) + ".pdf"
        )

    return sync_document_metadata(document)


def file_document_to_output(
    document: dict[str, Any],
    output_folder: Path,
    copy_file: bool = False,
    save_text: bool = False,
    folder_name: str | None = None,
    file_name: str | None = None,
    in_place: bool = False,
) -> dict[str, Any]:
    """The file_document_to_output() function places the given document in the
    given output_folder, writes standard and XMP metadata,
    adds a text layer onto the pdf, and updates the document parameter and 
    returns it. The function begins by checking that the source PDF still
    exists by checking that the source_path key in the document parameter is a
    valid path. Then the function diverges into in-place saving and saving to
    the given output.

    In-place:  Build the updated PDF beside the source, then atomically replace
    the original. A metadata-writing failure therefore leaves the source file
    untouched instead of partially rewriting it.
    """
    source_path = Path(document["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF no longer exists: {source_path}")

    if in_place:
        temp_handle, temp_name = tempfile.mkstemp(
            prefix=f".{source_path.stem}_metadata_",
            suffix=source_path.suffix,
            dir=source_path.parent,
        )
        os.close(temp_handle)
        temp_path = Path(temp_name)
        try:
            shutil.copy2(source_path, temp_path)
            write_pdf_metadata(temp_path, document)
            os.replace(temp_path, source_path)
        finally:
            temp_path.unlink(missing_ok=True)

        if save_text:
            source_path.with_suffix(".txt").write_text(
                document.get("ocr_text", ""), encoding="utf-8"
            )

        document.update(
            {
                "filed_path": str(source_path),
                "status": "filed",
            }
        )
        return document

    resolved_folder = safe_path_part(
        folder_name or document.get("folder_name", ""),
        "Unknown Lot - Unknown Address",
    )
    file_stem = Path(file_name or document.get("file_name", source_path.name)).stem
    resolved_file_name = safe_path_part(file_stem, source_path.stem) + ".pdf"

    destination_folder = output_folder / resolved_folder
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = unique_path(destination_folder / resolved_file_name)

    if copy_file:
        shutil.copy2(source_path, destination)
    else:
        shutil.move(str(source_path), destination)

    write_pdf_metadata(destination, document)

    if save_text:
        destination.with_suffix(".txt").write_text(
            document.get("ocr_text", ""), encoding="utf-8"
        )

    document.update(
        {
            "folder_name": resolved_folder,
            "file_name": resolved_file_name,
            "filed_path": str(destination),
            "status": "filed",
        }
    )
    return document
