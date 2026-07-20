from __future__ import annotations
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any
from metadata_extraction import ExtractedMetadata, safe_path_part, unique_path
from pdf_processing import write_pdf_metadata
from sdat import (
    LOOKUP_DOCUMENT_TYPE,
    lookup_by_tax_id,
    enrich_metadata_with_sdat,
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
    stem = (
        f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}"
    )
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
            is_unknown(metadata.get(field, ""))
            for field in REQUIRED_METADATA_FIELDS
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

def find_document(
    state: dict[str, Any], document_id: str
) -> dict[str, Any] | None:
    """The find_document() function returns the document in the state parameter
    with id matching the document_id parameter or None if there aren't any
    documents in the state parameter with an id that matches the document_id
    parameter."""
    return next(
        (
            doc
            for doc in state.get("documents", [])
            if doc.get("id") == document_id
        ),
        None,
    )

def metadata_from_dict(metadata: dict[str, Any]) -> ExtractedMetadata:
    return ExtractedMetadata(
        lot=str(metadata.get("lot", "Unknown Lot")),
        address=str(metadata.get("address", "Unknown Address")),
        project_code=str(metadata.get("project_code", "Project")),
        document_type=str(metadata.get("document_type", "Document")),
        tax_map=str(metadata.get("tax_map", "")),
        parcel=str(metadata.get("parcel", "")),
        tax_id=str(metadata.get("tax_id", "")),
        section=str(metadata.get("section", "")),
    )


def refresh_batch_property_fields_from_sdat(
    state: dict[str, Any], changed_field: str
) -> dict[str, str] | None:
    """Validate an edited property field with SDAT and synchronize the batch.

    This business-layer function intentionally lives outside ``app.py`` so that
    reusable document logic never imports the Flask entry point.  Keeping the
    dependency direction one-way prevents circular imports during application
    startup.
    """
    documents = [
        doc for doc in state.get("documents", []) if not doc.get("is_lookup_document")
    ]
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
            )
        }
        for batch_document in documents:
            batch_document["metadata"].update(values)
            sync_document_metadata(
                batch_document, auto_folder=True, auto_file_name=True
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
        )
    }
    for batch_document in documents:
        batch_document["metadata"].update(values)
        sync_document_metadata(batch_document, auto_folder=True, auto_file_name=True)
    return values

def apply_document_update(
    state: dict[str, Any], document: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    metadata = document["metadata"]

    # These are batch-level values. If the user corrects one file, apply the
    # correction to every document in the current batch.
    shared_field_names = (
        "lot",
        "address",
        "tax_map",
        "parcel",
        "tax_id",
        "section",
        "project_code",
    )
    shared_updates = {
        field: payload[field]
        for field in shared_field_names
        if field in payload
    }
    changed_field = payload.get("changed_field", "")

    if shared_updates:
        for batch_document in state.get("documents", []):
            batch_document["metadata"].update(shared_updates)
            sync_document_metadata(
                batch_document, auto_folder=True, auto_file_name=True
            )

    # If the user edits a property identifier, refresh the official SDAT address
    # once and apply that address to every document in the batch.
    if changed_field in {"tax_map", "parcel", "tax_id", "address", "section"}:
        refresh_batch_property_fields_from_sdat(state, changed_field)

    # Keep non-shared fields document-specific if they are ever posted by older UI/state.
    for field in (*REQUIRED_METADATA_FIELDS, *OPTIONAL_METADATA_FIELDS):
        if field in payload and field not in shared_updates:
            metadata[field] = payload[field]

    # Keep the lookup-only behavior synchronized with the editable document type.
    # Lookup-only documents stay visible in the review queue, but they are not
    # filed and are removed only after the permanent batch files successfully.
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
        document["file_name"] = suggested_filename(
            metadata, document["source_name"]
        )
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
) -> dict[str, Any]:
    source_path = Path(document["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF no longer exists: {source_path}")

    resolved_folder = safe_path_part(
        folder_name or document.get("folder_name", ""),
        "Unknown Lot - Unknown Address",
    )
    file_stem = Path(
        file_name or document.get("file_name", source_path.name)
    ).stem
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
