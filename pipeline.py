from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any
import time

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

"""High-level scan pipeline. This module orchestrates PDF discovery, page OCR, metadata extraction, batch voting, SDAT enrichment, output naming, and creation of the document records consumed by the review interface.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""
def vote_key(value: str) -> str:
    """Vote key.
    
    Args:
        value: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    return re.sub(r"[^a-z0-9]", "", normalize_for_fuzzy(value))


def vote_for_value(values: Iterable[str], fallback: str) -> str:
    """Vote for value.
    
    Args:
        values: Input used by this operation.
        fallback: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
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
    """Extract document metadata votes.
    
    Args:
        scanned_documents: Input used by this operation.
        config: Input used by this operation.
        default_project_code: Input used by this operation.
        default_document_type: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
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
    """Apply sdat record to shared.
    
    Args:
        shared: Input used by this operation.
        seed: Input used by this operation.
        record: Input used by this operation.
    """
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
    """Return a comparison-only address key with punctuation and spacing removed."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _confident_unique_address_record(
    address: str, records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return one SDAT record only when the OCR address identifies it uniquely.

    Mass Scan documents are independent.  This guard prevents an ambiguous
    address lookup (for example, a shared street address with several parcels)
    from silently assigning the first SDAT Tax ID to multiple PDFs.
    """
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

def choose_batch_metadata_by_vote(
    scanned_documents: list[dict[str, Any]],
    config: Config,
    default_project_code: str,
    default_document_type: str,
    resolve_duplicate_document_types: bool = True,
    strict_independent_lookup: bool = False,
    performance_callback=None,
) -> tuple[dict[str, str], list[ExtractedMetadata]]:
    """Choose shared metadata while reporting detailed performance timings.

    ``performance_callback`` receives concise terminal-safe lines. It is optional
    so existing callers and tests remain compatible.
    """
    emit = performance_callback or (lambda _message: None)
    stage_rows: list[tuple[str, float, str]] = []
    total_started = time.perf_counter()

    def timed(label: str, started: float, detail: str = "") -> float:
        elapsed = time.perf_counter() - started
        stage_rows.append((label, elapsed, detail))
        emit(f"[PERF] {label}: {elapsed:.4f}s" + (f" | {detail}" if detail else ""))
        return elapsed

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
    """Emit a copy/paste-friendly performance block sorted by elapsed time."""
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
    """Merge batch metadata.
    
    Args:
        document_text: Input used by this operation.
        config: Input used by this operation.
        default_project_code: Input used by this operation.
        default_document_type: Input used by this operation.
        shared_metadata: Input used by this operation.
        document_metadata: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
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
