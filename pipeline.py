"""Coordinate metadata extraction across a related batch of PDFs.

Individual drawings in one project packet often repeat the same lot,
address, map, parcel, and Tax ID. OCR may read each repetition slightly
differently. This module asks every drawing for metadata, lets the
documents vote on shared property values, and then uses SDAT to replace
uncertain OCR values with authoritative property data when possible.

Batch pipeline:
    OCR records
        -> per-document metadata extraction
        -> remove lookup-only helper PDFs from normal voting
        -> visually correct suspicious duplicate document types
        -> vote on shared property fields
        -> SDAT lookup in Tax ID / address / map-parcel priority order
        -> merge shared fields back into each permanent document

Document type and project code remain document-specific. Shared property
fields are synchronized because the drawings in Batch mode represent one
property. Mass mode uses ``extract_independent_document_metadata`` instead, so
unrelated jobs never enter the voting or shared-value merge stages."""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from metadata_extraction import (
    Config,
    ExtractedMetadata,
    extract_metadata,
    is_known_value,
    normalize_for_fuzzy,
    prefer_known,
    safe_path_part,
)
from sdat import (
    LOOKUP_DOCUMENT_TYPE,
    SDAT_METADATA_FIELDS,
    SdatSearchTerms,
    format_sdat_address,
    lookup_by_tax_id,
    lookup_maryland_property_by_address,
    lookup_maryland_property_records,
    metadata_from_sdat_record,
)
from visual_classifier import (
    fix_duplicate_document_types_with_visual_classifier,
)


def vote_key(value: str) -> str:
    """Convert a metadata value into the key used for batch majority voting.
    
    Case, punctuation, spacing, and common OCR confusions are removed so small formatting
    differences such as ``Lot-104`` and ``LOT 104`` count as the same answer. The original
    display form is retained separately by ``vote_for_value``."""
    return re.sub(r"[^a-z0-9]", "", normalize_for_fuzzy(value))


def vote_for_value(values: Iterable[str], fallback: str) -> str:
    """Choose the most frequently supported known value from several drawings.
    
    Placeholders are ignored. Each usable value is grouped by ``vote_key`` and counted, while
    the first human-readable spelling is remembered for display. If no drawing supplies a
    known value, the caller's fallback is returned."""
    seen_display: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for value in values:
        if not is_known_value(value):
            continue
        key = vote_key(value)
        if key:
            counts[key] += 1
            seen_display.setdefault(key, value)
    return seen_display[counts.most_common(1)[0][0]] if counts else fallback


def extract_document_metadata_votes(
    scanned_documents: Iterable[dict[str, Any]],
    config: Config,
    default_project_code: str,
    default_document_type: str,
) -> list[ExtractedMetadata]:
    """Run per-document extraction for an iterable of OCR scan records.
    
    Each result is an independent opinion about the property and document type. The function
    passes OCR page coordinates as well as text so address extraction can use title-block
    layout. Batch voting happens later; no values are shared in this step."""
    return [
        extract_metadata(
            document.get("ocr_text", ""),
            config,
            default_project_code,
            default_document_type,
            document.get("ocr_pages", []),
        )
        for document in scanned_documents
    ]


# Backward-compatible alias for older callers.
_lookup_by_tax_id = lookup_by_tax_id


def _apply_sdat_record_to_shared(
    shared: dict[str, str],
    seed: ExtractedMetadata,
    record: dict[str, Any],
) -> None:
    """Copy authoritative values from one selected SDAT record into the shared batch dictionary.
    
    ``metadata_from_sdat_record`` performs field mapping and normalization. Only known values
    overwrite the voted dictionary, so an empty SDAT column cannot erase useful OCR evidence.
    Hidden SDAT fields are copied together with the visible property identifiers."""
    resolved = metadata_from_sdat_record(seed, record)
    # SDAT is authoritative for the visible property fields and for the hidden
    # values that are persisted only in the document record and XMP packet.
    for field in (
        "lot",
        "address",
        "tax_map",
        "parcel",
        "tax_id",
        "section",
        *SDAT_METADATA_FIELDS,
    ):
        value = getattr(resolved, field)
        if is_known_value(value):
            shared[field] = value




def _normalized_address(value: str) -> str:
    """Create a strict address key for equality and containment checks.
    
    Uppercasing and removing all non-alphanumeric characters allows ``12 Main St.`` to match
    ``12 MAIN STREET`` closely enough for uniqueness checks without changing the displayed
    address stored in metadata."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _confident_unique_address_record(
    address: str, records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Accept an address lookup only when exactly one returned SDAT record clearly matches.
    
    This guard is used by Mass Scan, where neighboring PDFs are unrelated. Candidate addresses
    are normalized and considered when equal or when one fully contains the other. Zero or
    multiple convincing records returns ``None`` rather than guessing the first parcel."""
    target = _normalized_address(address)
    if not target:
        return None

    exact_or_containing: list[dict[str, Any]] = []
    for record in records:
        candidate = _normalized_address(format_sdat_address(record))
        if not candidate:
            continue
        if candidate == target or target in candidate or candidate in target:
            exact_or_containing.append(record)

    return exact_or_containing[0] if len(exact_or_containing) == 1 else None


def extract_independent_document_metadata(
    scanned_document: Mapping[str, Any],
    config: Config,
    default_project_code: str,
    default_document_type: str,
    performance_callback=None,
) -> ExtractedMetadata:
    """Extract and safely enrich metadata for one unrelated Mass Scan PDF.

    App role:
        Mass mode treats every PDF as a separate job. This function therefore
        performs only single-document work: it extracts values from that PDF,
        optionally verifies the property through SDAT, and returns the finished
        metadata for that one review record. It never votes with another PDF,
        repairs duplicate document types across a set, or applies packet-level
        shared values.

    How it works:
        1. Run the normal OCR-text metadata extractor for this document.
        2. Return lookup-only SDAT printouts unchanged because they are reference
           material rather than engineering drawings to be enriched and filed.
        3. If SDAT is enabled, try a direct Tax ID search first.
        4. If Tax ID does not resolve, search by address and accept the result only
           when exactly one returned record clearly matches that address.
        5. Do not use the broader map/parcel fallback used by Batch mode. A broad
           match is unsafe when neighboring files may describe unrelated jobs.
        6. Force the configured project code onto the result so the filing location
           follows the user's Mass Scan settings rather than a stray OCR reading.

    Returns:
        One immutable ``ExtractedMetadata`` object ready to be converted into the
        JSON-style document dictionary stored in the review queue.
    """
    emit = performance_callback or (lambda _message: None)
    started = time.perf_counter()

    metadata = extract_metadata(
        scanned_document.get("ocr_text", ""),
        config,
        default_project_code,
        default_document_type,
        scanned_document.get("ocr_pages", []),
        performance_callback=emit,
        profile_label="mass_document",
    )
    emit(
        f"[PERF] mass.metadata.extract: {time.perf_counter() - started:.4f}s | "
        f"chars={len(scanned_document.get('ocr_text', ''))}; "
        f"type={metadata.document_type}"
    )

    # A downloaded SDAT sheet is kept as evidence for review. Enriching it is
    # unnecessary, and changing its extracted values could hide what the source
    # reference document actually contained.
    if metadata.document_type == LOOKUP_DOCUMENT_TYPE:
        return metadata

    result = replace(
        metadata,
        project_code=safe_path_part(default_project_code, "Project"),
    )
    if not config.get("sdat_lookup", True):
        return result

    county = str(config.get("default_county", "") or "")

    # A Tax ID identifies a property more precisely than an address, so it is
    # always attempted first. Only the first record is used because the direct
    # district/account query is expected to identify one parcel.
    if is_known_value(result.tax_id):
        lookup_started = time.perf_counter()
        records = _lookup_by_tax_id(result.tax_id, county)
        emit(
            f"[PERF] mass.sdat.lookup_by_tax_id: "
            f"{time.perf_counter() - lookup_started:.4f}s | records={len(records)}"
        )
        if records:
            return replace(
                metadata_from_sdat_record(result, records[0]),
                project_code=safe_path_part(default_project_code, "Project"),
            )

    # Address lookup is deliberately conservative. Maryland data can return
    # multiple parcels for one street address, and Mass mode must never guess
    # which unrelated PDF owns which Tax ID.
    if is_known_value(result.address):
        lookup_started = time.perf_counter()
        records = lookup_maryland_property_by_address(
            result.address, county=county, limit=25
        )
        emit(
            f"[PERF] mass.sdat.lookup_by_address: "
            f"{time.perf_counter() - lookup_started:.4f}s | records={len(records)}"
        )
        selected = _confident_unique_address_record(result.address, records)
        if selected is not None:
            return replace(
                metadata_from_sdat_record(result, selected),
                project_code=safe_path_part(default_project_code, "Project"),
            )

    return result

def choose_batch_metadata_by_vote(
    scanned_documents: list[dict[str, Any]],
    config: Config,
    default_project_code: str,
    default_document_type: str,
    resolve_duplicate_document_types: bool = True,
    strict_independent_lookup: bool = False,
    performance_callback=None,
) -> tuple[dict[str, str], list[ExtractedMetadata]]:
    """Produce shared property metadata and per-document classifications for a scan set.
    
    Detailed workflow:
        1. Extract metadata from each OCR record and collect stage timings.
        2. Separate SDAT printouts from permanent engineering documents.
        3. Optionally use the visual classifier to repair suspicious duplicate plan types.
        4. Vote on lot, address, map, parcel, Tax ID, and section across normal drawings.
        5. Build an SDAT seed and try the most reliable lookup available: Tax ID first,
           address second, and map/parcel last.
        6. Copy authoritative SDAT values into the shared dictionary and emit a ranked
           performance report.
    
    ``strict_independent_lookup`` changes safety rules for Mass Scan: address results must be
    uniquely convincing and broad map/parcel fallback is disabled. The function returns the
    shared dictionary plus the original per-document metadata list, whose document types stay
    independent."""
    emit = performance_callback or (lambda _message: None)
    stage_rows: list[tuple[str, float, str]] = []
    total_started = time.perf_counter()

    def timed(label: str, started: float, detail: str = "") -> float:
        """Measure, store, and immediately report one batch-pipeline stage.
        
        The saved rows later become a ranked performance report. The optional detail text records
        context such as document count, record count, or which lookup resolved the property."""
        elapsed = time.perf_counter() - started
        stage_rows.append((label, elapsed, detail))
        emit(f"[PERF] {label}: {elapsed:.4f}s" + (f" | {detail}" if detail else ""))
        return elapsed

    # PHASE 1 - Ask each PDF independently before sharing any information.
    # Keeping these raw votes is essential for document-specific type labels
    # and for diagnosing which sheet supplied a questionable value.
    extraction_started = time.perf_counter()
    votes: list[ExtractedMetadata] = []
    for index, document in enumerate(scanned_documents, start=1):
        item_started = time.perf_counter()
        vote = extract_metadata(
            document.get("ocr_text", ""),
            config,
            default_project_code,
            default_document_type,
            document.get("ocr_pages", []),
            performance_callback=emit,
            profile_label=f"document_{index}",
        )
        votes.append(vote)
        elapsed = time.perf_counter() - item_started
        emit(
            f"[PERF] metadata.extract.document_{index}: {elapsed:.4f}s | "
            f"chars={len(document.get('ocr_text', ''))}; type={vote.document_type}"
        )
    timed("metadata.extract.all_documents", extraction_started, f"documents={len(votes)}")

    partition_started = time.perf_counter()
    # PHASE 2 - SDAT printouts are evidence, not permanent drawings. Separate
    # them so their placeholder lot/address values cannot weaken the vote.
    lookup_indexes = [
        i for i, vote in enumerate(votes) if vote.document_type == LOOKUP_DOCUMENT_TYPE
    ]
    lookup_index_set = set(lookup_indexes)
    normal_indexes = [i for i in range(len(votes)) if i not in lookup_index_set]
    normal_votes = [votes[i] for i in normal_indexes]
    normal_docs = [scanned_documents[i] for i in normal_indexes]
    timed(
        "metadata.partition_lookup_documents",
        partition_started,
        f"normal={len(normal_votes)}; lookup={len(lookup_indexes)}",
    )

    visual_started = time.perf_counter()
    if normal_votes and resolve_duplicate_document_types:
        fixed = fix_duplicate_document_types_with_visual_classifier(
            normal_votes, normal_docs, config
        )
        for index, vote in zip(normal_indexes, fixed):
            votes[index] = vote
        normal_votes = fixed
    timed("metadata.visual_duplicate_resolution", visual_started)

    voting_started = time.perf_counter()
    lookup_tax_ids = [
        votes[i].tax_id for i in lookup_indexes if is_known_value(votes[i].tax_id)
    ]
    # PHASE 4 - Create the packet-level property record. A lookup PDF Tax ID
    # outranks OCR votes because it comes from the government printout itself.
    shared = {
        "lot": vote_for_value((vote.lot for vote in normal_votes), "Unknown Lot"),
        "address": vote_for_value(
            (vote.address for vote in normal_votes), "Unknown Address"
        ),
        "tax_map": vote_for_value((vote.tax_map for vote in normal_votes), ""),
        "parcel": vote_for_value((vote.parcel for vote in normal_votes), ""),
        "tax_id": (
            lookup_tax_ids[0]
            if lookup_tax_ids
            else vote_for_value((vote.tax_id for vote in normal_votes), "")
        ),
        "section": vote_for_value((vote.section for vote in normal_votes), ""),
    }
    timed("metadata.vote_shared_fields", voting_started)

    if not config.get("sdat_lookup", True):
        timed("metadata_and_sdat.total", total_started, "SDAT disabled")
        _emit_performance_summary(emit, stage_rows)
        return shared, votes

    seed_started = time.perf_counter()
    seed_source = (
        normal_votes[0]
        if normal_votes
        else ExtractedMetadata(
            "Unknown Lot",
            "Unknown Address",
            default_project_code,
            default_document_type,
        )
    )
    seed = replace(seed_source, **shared)
    county = str(config.get("default_county", "") or "")
    timed("sdat.prepare_seed", seed_started, f"county={county or 'none'}")

    # PHASE 5 - Enrich only after voting. A Tax ID is the strongest key, an
    # address is next, and map/parcel are broader fallbacks. Returning as soon
    # as a stronger lookup succeeds avoids slower and less reliable queries.
    # Priority 1: explicit Tax ID.
    if is_known_value(shared["tax_id"]):
        lookup_started = time.perf_counter()
        records = _lookup_by_tax_id(shared["tax_id"], county)
        timed(
            "sdat.lookup_by_tax_id",
            lookup_started,
            f"records={len(records)}; tax_id={shared['tax_id']}",
        )
        if records:
            apply_started = time.perf_counter()
            _apply_sdat_record_to_shared(shared, seed, records[0])
            timed("sdat.apply_tax_id_record", apply_started)
            timed("metadata_and_sdat.total", total_started, "resolved_by=tax_id")
            _emit_performance_summary(emit, stage_rows)
            return shared, votes
        shared["tax_id"] = ""
        seed = replace(seed, tax_id="")

    # Priority 2: address.
    if is_known_value(shared["address"]):
        lookup_started = time.perf_counter()
        records = lookup_maryland_property_by_address(
            shared["address"], county=county, limit=25
        )
        timed(
            "sdat.lookup_by_address",
            lookup_started,
            f"records={len(records)}; address={shared['address']}",
        )
        if records:
            select_started = time.perf_counter()
            selected_record = (
                _confident_unique_address_record(shared["address"], records)
                if strict_independent_lookup
                else records[0]
            )
            timed("sdat.select_address_record", select_started)
            if selected_record is not None:
                apply_started = time.perf_counter()
                _apply_sdat_record_to_shared(shared, seed, selected_record)
                timed("sdat.apply_address_record", apply_started)
                timed("metadata_and_sdat.total", total_started, "resolved_by=address")
                _emit_performance_summary(emit, stage_rows)
                return shared, votes

    # Priority 3: map/parcel fallback.
    if (shared["tax_map"] or shared["parcel"]) and not strict_independent_lookup:
        terms_started = time.perf_counter()
        terms = SdatSearchTerms(
            county=county,
            lot=(
                ""
                if str(shared["lot"]).lower().startswith("unknown")
                else shared["lot"]
            ),
            tax_map=shared["tax_map"],
            parcel=shared["parcel"],
        )
        timed("sdat.prepare_map_parcel_terms", terms_started)
        lookup_started = time.perf_counter()
        records = lookup_maryland_property_records(terms)
        timed(
            "sdat.lookup_by_map_parcel",
            lookup_started,
            f"records={len(records)}; map={shared['tax_map']}; parcel={shared['parcel']}",
        )
        if records:
            apply_started = time.perf_counter()
            _apply_sdat_record_to_shared(shared, seed, records[0])
            timed("sdat.apply_map_parcel_record", apply_started)

    timed("metadata_and_sdat.total", total_started, "resolved_by=none_or_map_parcel")
    _emit_performance_summary(emit, stage_rows)
    return shared, votes


def _emit_performance_summary(emit, rows: list[tuple[str, float, str]]) -> None:
    """Print a copy-and-paste-friendly ranking of metadata and SDAT stage durations.
    
    Rows are sorted slowest first and expressed as seconds plus percentage of total time. The
    explicit START/END markers let a user paste exactly one report back into a debugging chat
    or issue without unrelated Flask access-log lines."""
    total = next((seconds for label, seconds, _ in reversed(rows) if label == "metadata_and_sdat.total"), 0.0)
    emit("=== COAB PERFORMANCE REPORT START ===")
    emit(f"report_version=3.0; metadata_sdat_total_seconds={total:.4f}")
    for rank, (label, seconds, detail) in enumerate(
        sorted(rows, key=lambda row: row[1], reverse=True), start=1
    ):
        percent = (seconds / total * 100.0) if total else 0.0
        suffix = f"; detail={detail}" if detail else ""
        emit(
            f"rank={rank}; stage={label}; seconds={seconds:.4f}; "
            f"percent_of_total={percent:.1f}{suffix}"
        )
    emit("=== COAB PERFORMANCE REPORT END ===")


def merge_batch_metadata(
    document_text: str,
    config: Config,
    default_project_code: str,
    default_document_type: str,
    shared_metadata: Mapping[str, str],
    document_metadata: ExtractedMetadata | None = None,
) -> ExtractedMetadata:
    """Combine one drawing's own classification with the batch's shared property values.
    
    The document keeps its own document type and extracted details, while known shared lot,
    address, map, parcel, Tax ID, section, and hidden SDAT fields take precedence. Unknown
    shared values fall back to the individual drawing rather than erasing it.
    
    The project code is deliberately taken from the scan setting/folder convention so every
    document in the packet is filed under the same project even if OCR reads a stray code."""
    document_metadata = document_metadata or extract_metadata(
        document_text, config, default_project_code, default_document_type
    )
    return replace(
        document_metadata,
        lot=prefer_known(shared_metadata.get("lot", ""), document_metadata.lot),
        address=prefer_known(
            shared_metadata.get("address", ""), document_metadata.address
        ),
        tax_map=prefer_known(
            shared_metadata.get("tax_map", ""), document_metadata.tax_map
        ),
        parcel=prefer_known(
            shared_metadata.get("parcel", ""), document_metadata.parcel
        ),
        tax_id=prefer_known(
            shared_metadata.get("tax_id", ""), document_metadata.tax_id
        ),
        section=prefer_known(
            shared_metadata.get("section", ""), document_metadata.section
        ),
        project_code=safe_path_part(default_project_code, "Project"),
        **{
            field: prefer_known(
                shared_metadata.get(field, ""), getattr(document_metadata, field)
            )
            for field in SDAT_METADATA_FIELDS
        },
    )
