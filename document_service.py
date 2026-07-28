"""Business rules for review records after OCR has finished.

A "document" in this module is a dictionary stored in the review-state
JSON file. It contains the source PDF path, OCR text, extracted metadata,
suggested names, review status, and eventual filed path. The functions
here keep that record internally consistent when metadata changes.

This module is responsible for three app-level jobs:
    1. Build suggested folder and file names from metadata.
    2. Apply browser edits and optionally re-check property data in SDAT.
    3. Write metadata into a PDF and move/copy it to its final location.

Keeping these rules outside ``app.py`` is important. Flask routes remain
small, while tests and other callers can use the same update and filing
behavior without pretending to be a web request."""

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
    """Decide whether a metadata value is still a placeholder.
    
    The review interface uses labels such as ``Unknown Lot``, ``Project``, and
    ``Document`` when extraction has no trustworthy answer. Treating those as
    missing prevents a document from being marked ready merely because the
    placeholder string is non-empty."""
    return (
        not value
        or value.lower().startswith("unknown")
        or value in {"Project", "Document"}
    )


def suggested_folder(metadata: dict[str, str]) -> str:
    """Build the default property-folder name shown in the review form.
    
    The convention is ``Lot <lot> - <address>``. ``safe_path_part`` removes
    characters Windows forbids in folder names and supplies a readable fallback
    when OCR did not identify either value. This function suggests a name; it
    does not create the directory."""
    return safe_path_part(
        f"Lot {metadata.get('lot', '')} - {metadata.get('address', '')}",
        "Unknown Lot - Unknown Address",
    )


def suggested_filename(metadata: dict[str, str], source_name: str) -> str:
    """Build the default PDF filename from document type and lot.
    
    A result such as ``Site Plan - Lot 104.pdf`` is easier to search and sort
    than the scanner's original filename. If required metadata is unavailable,
    the original source stem becomes the fallback. Path sanitization happens
    before the ``.pdf`` suffix is restored."""
    stem = (
        f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}"
    )
    return safe_path_part(stem, Path(source_name).stem) + ".pdf"


def document_status(metadata: dict[str, str]) -> str:
    """Translate metadata completeness into the review status displayed by the UI.
    
    Lot, address, project code, and document type are required for normal
    documents. If any one is blank or a placeholder, the result is
    ``needs_review``; otherwise it is ``ready``. Optional SDAT fields do not
    block filing."""
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
    """Recalculate all values derived from a document's metadata.
    
    Browser edits and SDAT refreshes can change lot, address, or document type.
    Those changes may make the old suggested folder, filename, or status stale.
    This function updates the derived fields together so the record cannot show
    a new lot with an old filename.
    
    ``auto_folder`` and ``auto_file_name`` control whether a user's manual name
    is replaced. Lookup-only records receive a special status because they are
    reference material rather than files to be permanently filed."""
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
    """Locate one review record by its generated stable ID.
    
    Filenames and paths can change during review, so routes identify records
    with a UUID stored in the ``id`` field. ``next(..., None)`` returns the first
    match without raising an exception when the record has already been filed."""
    return next(
        (
            doc
            for doc in state.get("documents", [])
            if doc.get("id") == document_id
        ),
        None,
    )


def metadata_from_dict(metadata: dict[str, Any]) -> ExtractedMetadata:
    """Convert the JSON-style metadata stored in state into ``ExtractedMetadata``.
    
    SDAT helpers use an immutable dataclass, while the browser and state file use
    dictionaries. This adapter fills visible defaults and copies every hidden
    SDAT field so no property information is lost during a browser edit or
    re-lookup."""
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
    """Revalidate edited property identifiers and synchronize authoritative fields.
    
    App role:
        When a user changes Tax ID, address, map, parcel, or lot, the old SDAT
        values may no longer describe the selected property. This function
        performs a new lookup and updates the chosen document scope.
    
    How it chooses a lookup:
        * Tax ID change -> direct district/account search.
        * Address change -> address search.
        * Other property field -> general SDAT search with the edited identifiers.
    
    Batch mode passes every permanent drawing because they share one property.
    Mass mode passes only the edited PDF. Suggested names and status are rebuilt
    after the authoritative values are copied into each target record."""
    documents = [doc for doc in documents if not doc.get("is_lookup_document")]
    if not documents:
        return None

    config = load_config_from_state(state)
    if not config.get("sdat_lookup", True):
        return None

    seed = metadata_from_dict(documents[0].get("metadata", {}))
    county = str(config.get("default_county", "") or "")
    records: list[dict[str, Any]] = []

    # Choose the lookup that best matches what the user corrected. Reusing an
    # old address after a Tax ID edit, for example, could restore the wrong parcel.
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
    """Apply the SDAT refresh workflow to all permanent documents in Batch mode.
    
    This small wrapper selects the batch-wide synchronization scope and delegates
    the lookup logic to ``refresh_property_fields_from_sdat``. Keeping the wrapper
    makes the caller's intent explicit and protects lookup-only helper records."""
    return refresh_property_fields_from_sdat(
        state,
        list(state.get("documents", [])),
        changed_field,
    )


def apply_document_update(
    state: dict[str, Any], document: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Apply browser edits while enforcing Batch-versus-Mass synchronization rules.
    
    The payload may change metadata, suggested names, or both. Shared property
    fields are copied across all permanent documents only in Batch mode; document
    type remains specific to the selected PDF. In Mass mode no neighboring record
    is modified because each PDF can represent a different property.
    
    When a property identifier changes, the function asks SDAT to refresh related
    fields. It then runs ``sync_document_metadata`` so status and automatic names
    match the final values. The selected document is returned for an immediate UI
    update, while the caller persists the whole state atomically."""
    # The selected record is mutable because it belongs to the locked live
    # state transaction. Batch peers may also be changed below when a shared
    # property field is edited.
    metadata = document["metadata"]
    scan_mode = str(
        state.get("settings", {}).get("scan_mode", "batch")
    ).lower()
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
        field: payload[field]
        for field in property_field_names
        if field in payload
    }
    changed_field = payload.get("changed_field", "")

    if property_updates:
        update_targets = (
            [document] if mass_mode else list(state.get("documents", []))
        )
        for target_document in update_targets:
            target_document["metadata"].update(property_updates)
            sync_document_metadata(
                target_document, auto_folder=True, auto_file_name=True
            )

    if changed_field in {"tax_map", "parcel", "tax_id", "address"}:
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
    in_place: bool = False,
) -> dict[str, Any]:
    """Finish one reviewed document: choose its destination, write metadata, and move it.
    
    Filing sequence:
        1. Confirm the source PDF still exists.
        2. Rebuild names from metadata unless the user supplied overrides.
        3. Choose either the source folder (In-Place) or the configured property
           subfolder under the output root.
        4. Avoid overwriting an existing PDF by creating a numbered path.
        5. Work on a temporary copy, write standard/XMP metadata, then atomically
           place the completed file at its destination.
        6. Copy or move the source according to the user's option and optionally
           save the OCR text beside the PDF.
    
    The returned dictionary is the same review record with ``filed_path`` updated.
    Temporary-file cleanup in ``finally`` prevents abandoned partial PDFs after an
    exception."""
    # Never begin naming or metadata work until the source is confirmed. A
    # stale review record should fail clearly rather than create an empty file.
    source_path = Path(document["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF no longer exists: {source_path}")

    if in_place:
        # Write metadata to a temporary sibling first. The final destination only
        # appears after a complete PDF exists, avoiding half-written project files.
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

        document["filed_path"] = str(source_path)
        document["status"] = "filed"

        return document

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
