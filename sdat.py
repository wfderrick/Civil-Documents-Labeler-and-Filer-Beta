"""Maryland SDAT lookup client and record-selection logic. It normalizes addresses and parcel identifiers, queries available SDAT endpoints, ranks candidate records, and returns property metadata suitable for review.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import requests

from metadata_extraction import (
    Config,
    ExtractedMetadata,
    first_match,
    identifier_options,
    normalize_for_fuzzy,
    normalize_identifier,
    normalize_ocr_numbers,
    normalize_value,
    safe_path_part,
)
from tax_id_utils import extract_tax_id_parts

LOOKUP_DOCUMENT_TYPE = "Lookup Only"

SDAT_LOOKUP_ANCHORS = (
    "department of assessments and taxation",
    "real property data search",
    "account identifier",
    "account number",
    "premises address",
)

SDAT_API_URL = "https://opendata.maryland.gov/resource/ed4q-f8tm.json"

SDAT_FIELDS = {
    "county": "county_name_mdp_field_cntyname",
    "account_id": "account_id_mdp_field_acctid",
    "district": "record_key_district_ward_sdat_field_2",
    "account_number": "record_key_account_number_sdat_field_3",
    "lot": "lot_mdp_field_lot_sdat_field_41",
    "map": "map_mdp_field_map_sdat_field_42",
    "parcel": "parcel_mdp_field_parcel_sdat_field_44",
    "section": "section_mdp_field_section_sdat_field_39",
    "premise_number": "premise_address_number_mdp_field_premsnum_sdat_field_20",
    "premise_name": "premise_address_name_mdp_field_premsnam_sdat_field_23",
    "premise_type": "premise_address_type_mdp_field_premstyp_sdat_field_24",
    "premise_city": "premise_address_city_mdp_field_premcity_sdat_field_25",
    "premise_zip": "premise_address_zip_code_mdp_field_premzip_sdat_field_26",
    "mdp_address": "mdp_street_address_mdp_field_address",
    "mdp_city": "mdp_street_address_city_mdp_field_city",
    "mdp_zip": "mdp_street_address_zip_code_mdp_field_zipcode",
    "link": "real_property_search_link",
    "jurisdiction_code_mdp_field_jurscode": "jurisdiction_code_mdp_field_jurscode",
    "finder_online_link": "finder_online_link",
    "mdp_longitude_mdp_field_digxcord_converted_to_wgs84": "mdp_longitude_mdp_field_digxcord_converted_to_wgs84",
    "mdp_latitude_mdp_field_digycord_converted_to_wgs84": "mdp_latitude_mdp_field_digycord_converted_to_wgs84",
    "mappable_latitude_and_longitude": "mappable_latitude_and_longitude",
    "legal_description_line_1_mdp_field_legal1_sdat_field_17": "legal_description_line_1_mdp_field_legal1_sdat_field_17",
    "legal_description_line_2_mdp_field_legal2_sdat_field_18": "legal_description_line_2_mdp_field_legal2_sdat_field_18",
    "deed_reference_1_liber_mdp_field_dr1liber_sdat_field_30": "deed_reference_1_liber_mdp_field_dr1liber_sdat_field_30",
    "deed_reference_1_folio_mdp_field_dr1folio_sdat_field_31": "deed_reference_1_folio_mdp_field_dr1folio_sdat_field_31",
    "subdivision_code_mdp_field_subdivsn_sdat_field_37": "subdivision_code_mdp_field_subdivsn_sdat_field_37",
    "grid_mdp_field_grid_sdat_field_43": "grid_mdp_field_grid_sdat_field_43",
    "zoning_code_mdp_field_zoning_sdat_field_45": "zoning_code_mdp_field_zoning_sdat_field_45",
    "land_use_code_mdp_field_lu_desclu_sdat_field_50": "land_use_code_mdp_field_lu_desclu_sdat_field_50",
    "property_factors_utilities_water_mdp_field_pfuw_sdat_field_63": "property_factors_utilities_water_mdp_field_pfuw_sdat_field_63",
    "property_factors_utilities_sewer_mdp_field_pfus_sdat_field_64": "property_factors_utilities_sewer_mdp_field_pfus_sdat_field_64",
    "property_factors_location_waterfront_mdp_field_pflw_sdat_field_65": "property_factors_location_waterfront_mdp_field_pflw_sdat_field_65",
    "property_factors_street_paved_mdp_field_pfsp_sdat_field_67": "property_factors_street_paved_mdp_field_pfsp_sdat_field_67",
    "property_factors_street_unpaved_mdp_field_pfsu_sdat_field_68": "property_factors_street_unpaved_mdp_field_pfsu_sdat_field_68",
}

# Fields below are retained as hidden document metadata and embedded in XMP.
# They are intentionally separate from the review UI's editable property fields.
SDAT_METADATA_FIELDS = (
    "jurisdiction_code_mdp_field_jurscode",
    "finder_online_link",
    "mdp_longitude_mdp_field_digxcord_converted_to_wgs84",
    "mdp_latitude_mdp_field_digycord_converted_to_wgs84",
    "mappable_latitude_and_longitude",
    "legal_description_line_1_mdp_field_legal1_sdat_field_17",
    "legal_description_line_2_mdp_field_legal2_sdat_field_18",
    "deed_reference_1_liber_mdp_field_dr1liber_sdat_field_30",
    "deed_reference_1_folio_mdp_field_dr1folio_sdat_field_31",
    "subdivision_code_mdp_field_subdivsn_sdat_field_37",
    "grid_mdp_field_grid_sdat_field_43",
    "zoning_code_mdp_field_zoning_sdat_field_45",
    "land_use_code_mdp_field_lu_desclu_sdat_field_50",
    "property_factors_utilities_water_mdp_field_pfuw_sdat_field_63",
    "property_factors_utilities_sewer_mdp_field_pfus_sdat_field_64",
    "property_factors_location_waterfront_mdp_field_pflw_sdat_field_65",
    "property_factors_street_paved_mdp_field_pfsp_sdat_field_67",
    "property_factors_street_unpaved_mdp_field_pfsu_sdat_field_68",
)


@dataclass(frozen=True)
class SdatSearchTerms:
    """Represent SdatSearchTerms behavior and related state."""

    county: str = ""
    lot: str = ""
    tax_map: str = ""
    parcel: str = ""
    tax_id: str = ""
    district: str = ""
    account_number: str = ""


def lookup_by_tax_id(tax_id: str, county: str = "") -> list[dict[str, Any]]:
    """Perform the fastest, most specific SDAT lookup for a Tax ID."""
    district, account_number = extract_tax_id_parts(tax_id)
    if not district or not account_number:
        return []
    return lookup_maryland_property_records(
        SdatSearchTerms(
            county=county,
            tax_id=tax_id,
            district=district,
            account_number=account_number,
        )
    )


def is_sdat_lookup_document(text: str) -> bool:
    """Identify an SDAT printout using several stable page anchors."""
    normalized = normalize_for_fuzzy(text)
    hits = sum(
        normalize_for_fuzzy(anchor) in normalized
        for anchor in SDAT_LOOKUP_ANCHORS
    )
    strong_header = (
        normalize_for_fuzzy("department of assessments and taxation")
        in normalized
        and normalize_for_fuzzy("real property data search") in normalized
    )
    account_block = (
        normalize_for_fuzzy("account identifier") in normalized
        and normalize_for_fuzzy("account number") in normalized
    )
    return (strong_header and account_block) or hits >= 4


def extract_sdat_lookup_tax_id(text: str) -> tuple[str, str, str] | None:
    """Extract district/account from an SDAT printout and build its Tax ID."""
    patterns = (
        r"\bdistrict\s*[-:#]?\s*([0-9Oo]{1,2})\s+account\s+(?:number|no\.?|#)\s*[-:#]?\s*([0-9OoIl]{4,10})\b",
        r"\baccount\s+identifier.{0,80}?district\s*[-:#]?\s*([0-9Oo]{1,2}).{0,80}?account\s+(?:number|no\.?|#)\s*[-:#]?\s*([0-9OoIl]{4,10})\b",
    )
    for pattern in patterns:
        match = re.search(
            pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        if not match:
            continue
        district = re.sub(
            r"\D", "", normalize_ocr_numbers(match.group(1))
        ).zfill(2)
        account = re.sub(
            r"\D", "", normalize_ocr_numbers(match.group(2))
        ).zfill(6)
        if district and account:
            return district, account, f"{district}-{account}"
    return None


def soql_escape(value: str) -> str:
    """Soql escape.

    Args:
        value: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    return str(value or "").replace("'", "''").strip()


def or_equals(
    field: str, value: str, widths: Iterable[int] = (2, 3, 4, 6, 8)
) -> str:
    """Or equals.

    Args:
        field: Input used by this operation.
        value: Input used by this operation.
        widths: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    options = identifier_options(value, widths)
    return (
        "("
        + " OR ".join(
            f"{field} = '{soql_escape(option)}'" for option in options
        )
        + ")"
    )


def extract_sdat_search_terms(
    text: str, metadata: ExtractedMetadata, config: Config
) -> SdatSearchTerms:
    """Extract sdat search terms.

    Args:
        text: Input used by this operation.
        metadata: Input used by this operation.
        config: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    county = first_match(
        text, config.get("county_patterns", [])
    ) or config.get("default_county", "")
    county = re.sub(
        r"\bcounty\b", "", str(county), flags=re.IGNORECASE
    ).strip()
    tax_map = (
        metadata.tax_map
        or first_match(
            text, config.get("map_patterns", []), normalize_numbers=False
        )
        or ""
    )
    parcel = (
        metadata.parcel
        or first_match(
            text, config.get("parcel_patterns", []), normalize_numbers=False
        )
        or ""
    )
    tax_id = (
        metadata.tax_id
        or first_match(
            text, config.get("tax_id_patterns", []), normalize_numbers=True
        )
        or ""
    )
    district, account_number = extract_tax_id_parts(tax_id)
    district = (
        district
        or first_match(
            text, config.get("district_patterns", []), normalize_numbers=True
        )
        or ""
    )
    account_number = (
        account_number
        or first_match(
            text, config.get("account_patterns", []), normalize_numbers=True
        )
        or ""
    )
    lot = "" if metadata.lot.lower().startswith("unknown") else metadata.lot
    return SdatSearchTerms(
        county=safe_path_part(county, "") if county else "",
        lot=safe_path_part(lot, "") if lot else "",
        tax_map=safe_path_part(tax_map, "") if tax_map else "",
        parcel=safe_path_part(parcel, "") if parcel else "",
        tax_id=safe_path_part(tax_id, "") if tax_id else "",
        district=safe_path_part(district, "") if district else "",
        account_number=safe_path_part(account_number, "")
        if account_number
        else "",
    )


def selected_sdat_fields() -> list[str]:
    """Return every Socrata column needed for matching and PDF metadata."""
    core_fields = [
        SDAT_FIELDS["county"],
        SDAT_FIELDS["account_id"],
        SDAT_FIELDS["district"],
        SDAT_FIELDS["account_number"],
        SDAT_FIELDS["lot"],
        SDAT_FIELDS["map"],
        SDAT_FIELDS["parcel"],
        SDAT_FIELDS["section"],
        SDAT_FIELDS["premise_number"],
        SDAT_FIELDS["premise_name"],
        SDAT_FIELDS["premise_type"],
        SDAT_FIELDS["premise_city"],
        SDAT_FIELDS["premise_zip"],
        SDAT_FIELDS["mdp_address"],
        SDAT_FIELDS["mdp_city"],
        SDAT_FIELDS["mdp_zip"],
        SDAT_FIELDS["link"],
    ]
    return core_fields + [SDAT_FIELDS[field] for field in SDAT_METADATA_FIELDS]


def sdat_get(where_parts: list[str], limit: int = 200) -> list[dict[str, Any]]:
    """Query the Maryland Open Data SDAT endpoint with the selected columns.

    Args:
        where_parts: SoQL predicates joined with ``AND``.
        limit: Maximum number of property records to request.

    Returns:
        Decoded SDAT records returned by the Socrata API.
    """
    if not where_parts:
        return []
    response = requests.get(
        SDAT_API_URL,
        params={
            "$limit": limit,
            "$select": ",".join(selected_sdat_fields()),
            "$where": " AND ".join(where_parts),
        },
        timeout=20,
    )
    if not response.ok:
        print(response.url, file=sys.stderr)
        print(response.text, file=sys.stderr)
        response.raise_for_status()
    return response.json()


def record_identifier_matches(
    record: dict[str, Any], key: str, target: str
) -> bool:
    """Record identifier matches.

    Args:
        record: Input used by this operation.
        key: Input used by this operation.
        target: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    if not target:
        return True
    field = {"tax_map": "map", "account_number": "account_number"}.get(
        key, key
    )
    return normalize_identifier(
        record.get(SDAT_FIELDS[field], "")
    ) == normalize_identifier(target)


def filter_sdat_records(
    records: list[dict[str, Any]], terms: SdatSearchTerms
) -> list[dict[str, Any]]:
    """Filter sdat records.

    Args:
        records: Input used by this operation.
        terms: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    filtered = []
    for record in records:
        if terms.tax_map and not record_identifier_matches(
            record, "tax_map", terms.tax_map
        ):
            continue
        if terms.parcel and not record_identifier_matches(
            record, "parcel", terms.parcel
        ):
            continue
        if terms.lot and not record_identifier_matches(
            record, "lot", terms.lot
        ):
            continue
        if terms.district and not record_identifier_matches(
            record, "district", terms.district
        ):
            continue
        if terms.account_number and not record_identifier_matches(
            record, "account_number", terms.account_number
        ):
            continue
        filtered.append(record)
    return filtered or records


def lookup_maryland_property_records(
    terms: SdatSearchTerms,
) -> list[dict[str, Any]]:
    """Lookup maryland property records.

    Args:
        terms: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    county_filter = (
        f"upper({SDAT_FIELDS['county']}) like upper('%{soql_escape(terms.county)}%')"
        if terms.county
        else ""
    )

    strategies: list[tuple[list[str], bool]] = []

    # 1. Best: district + account + county
    if terms.account_number and terms.district and county_filter:
        strategies.append(
            (
                [
                    county_filter,
                    or_equals(
                        SDAT_FIELDS["account_number"],
                        terms.account_number,
                        (6, 8),
                    ),
                    or_equals(SDAT_FIELDS["district"], terms.district, (2,)),
                ],
                False,  # do NOT filter by lot/map/parcel after this
            )
        )

    # 2. Tax ID without county, useful when county OCR fails
    if terms.account_number and terms.district:
        strategies.append(
            (
                [
                    or_equals(
                        SDAT_FIELDS["account_number"],
                        terms.account_number,
                        (6, 8),
                    ),
                    or_equals(SDAT_FIELDS["district"], terms.district, (2,)),
                ],
                False,
            )
        )

    # 3. Map/parcel fallback
    if county_filter and terms.tax_map:
        strategies.append(
            (
                [
                    county_filter,
                    or_equals(SDAT_FIELDS["map"], terms.tax_map, (3, 4)),
                ],
                True,
            )
        )

    if county_filter and terms.parcel:
        strategies.append(
            (
                [
                    county_filter,
                    or_equals(SDAT_FIELDS["parcel"], terms.parcel, (3, 4)),
                ],
                True,
            )
        )

    for where_parts, should_filter in strategies:
        records = sdat_get(where_parts)
        if records:
            return (
                filter_sdat_records(records, terms)
                if should_filter
                else records
            )

    return []


def format_sdat_address(record: dict[str, Any]) -> str:
    """Format sdat address.

    Args:
        record: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    number = normalize_value(record.get(SDAT_FIELDS["premise_number"], ""))
    street = normalize_value(record.get(SDAT_FIELDS["premise_name"], ""))
    street_type = normalize_value(record.get(SDAT_FIELDS["premise_type"], ""))
    city = normalize_value(record.get(SDAT_FIELDS["premise_city"], ""))
    zip_code = normalize_value(record.get(SDAT_FIELDS["premise_zip"], ""))
    street_address = " ".join(
        part for part in [number, street, street_type] if part
    ).strip()
    if not street_address:
        street_address = normalize_value(
            record.get(SDAT_FIELDS["mdp_address"], "")
        )
        city = city or normalize_value(record.get(SDAT_FIELDS["mdp_city"], ""))
        zip_code = zip_code or normalize_value(
            record.get(SDAT_FIELDS["mdp_zip"], "")
        )
    return (
        " ".join(
            part for part in [street_address, city, "MD", zip_code] if part
        ).strip()
        if street_address
        else ""
    )


def tax_id_from_sdat_record(record: dict[str, Any]) -> str:
    """Tax id from sdat record.

    Args:
        record: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    district = normalize_value(record.get(SDAT_FIELDS["district"], ""))
    account = normalize_value(record.get(SDAT_FIELDS["account_number"], ""))
    if district and account:
        return f"{district.zfill(2)}-{account.zfill(6)}"
    return ""


def normalize_sdat_metadata_value(value: Any) -> str:
    """Convert an SDAT scalar or location object into stable metadata text."""
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_value(str(value))


def metadata_from_sdat_record(
    metadata: ExtractedMetadata, record: dict[str, Any]
) -> ExtractedMetadata:
    """Merge one authoritative SDAT record into extracted document metadata.

    Existing OCR values are preserved when SDAT omits a field. File-naming
    fields are sanitized, while descriptive values and links are retained as
    ordinary text for later XMP serialization.
    """
    address = format_sdat_address(record)
    lot = normalize_value(record.get(SDAT_FIELDS["lot"], ""))
    tax_map = normalize_value(record.get(SDAT_FIELDS["map"], ""))
    parcel = normalize_value(record.get(SDAT_FIELDS["parcel"], ""))
    section = normalize_value(record.get(SDAT_FIELDS["section"], ""))
    tax_id = tax_id_from_sdat_record(record)
    return replace(
        metadata,
        lot=safe_path_part(lot, metadata.lot) if lot else metadata.lot,
        address=(
            safe_path_part(address, metadata.address)
            if address
            else metadata.address
        ),
        tax_map=safe_path_part(tax_map, "") if tax_map else metadata.tax_map,
        parcel=safe_path_part(parcel, "") if parcel else metadata.parcel,
        tax_id=safe_path_part(tax_id, "") if tax_id else metadata.tax_id,
        section=safe_path_part(section, "") if section else metadata.section,
        **{
            field: normalize_sdat_metadata_value(record.get(SDAT_FIELDS[field], ""))
            or getattr(metadata, field)
            for field in SDAT_METADATA_FIELDS
        },
    )


def _address_tokens(address: str) -> tuple[str, list[str]]:
    """Address tokens.

    Args:
        address: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    cleaned = re.sub(r"[^0-9A-Za-z ]", " ", str(address or "")).upper()
    parts = [part for part in cleaned.split() if part]
    number = parts[0] if parts and parts[0].isdigit() else ""
    stop = {
        "MD",
        "MARYLAND",
        "ST",
        "STREET",
        "RD",
        "ROAD",
        "DR",
        "DRIVE",
        "LN",
        "LANE",
        "CT",
        "COURT",
        "AVE",
        "AVENUE",
        "BLVD",
        "BOULEVARD",
        "WAY",
        "PL",
        "PLACE",
        "CIR",
        "CIRCLE",
    }
    words = [
        part for part in parts[1:] if part not in stop and not part.isdigit()
    ]
    return number, words[:3]


def lookup_maryland_property_by_address(
    address: str, county: str = "", limit: int = 100
) -> list[dict[str, Any]]:
    """Lookup maryland property by address.

    Args:
        address: Input used by this operation.
        county: Input used by this operation.
        limit: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    number, words = _address_tokens(address)
    if not number or not words:
        return []
    str_number = soql_escape(number)
    add_zero = 5 - len(str_number)
    for i in range(add_zero):
        str_number = "0" + str_number
    where = [f"{SDAT_FIELDS['premise_number']} = '{str_number}'"]
    where.append(
        f"upper({SDAT_FIELDS['mdp_address']}) like upper('%{soql_escape(words[0])}%')"
    )
    if county:
        where.append(
            f"upper({SDAT_FIELDS['county']}) like upper('%{soql_escape(county)}%')"
        )
    records = sdat_get(where, limit=limit)
    if not records:
        return []
    target = re.sub(r"[^A-Z0-9]", "", address.upper())

    def score(record: dict[str, Any]) -> int:
        """Score.

        Args:
            record: Input used by this operation.

        Returns:
            The computed result for the caller. See the function body and type hints for the exact shape.
        """
        candidate = re.sub(
            r"[^A-Z0-9]", "", format_sdat_address(record).upper()
        )
        return sum(
            1 for token in [number, *words] if token and token in candidate
        ) + (5 if candidate == target else 0)

    return sorted(records, key=score, reverse=True)


def enrich_metadata_with_sdat(
    metadata: ExtractedMetadata, text: str, config: Config
) -> ExtractedMetadata:
    """Enrich metadata with sdat.

    Args:
        metadata: Input used by this operation.
        text: Input used by this operation.
        config: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    if not config.get("sdat_lookup", True):
        return metadata
    records = lookup_maryland_property_records(
        extract_sdat_search_terms(text, metadata, config)
    )
    return (
        metadata_from_sdat_record(metadata, records[0])
        if records
        else metadata
    )
