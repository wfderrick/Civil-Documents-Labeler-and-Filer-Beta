"""Query Maryland SDAT property records and merge authoritative results.

OCR can discover identifiers printed on a plan, but it cannot prove that
a value is correct. This module converts the available identifiers into
Socrata/SoQL queries against Maryland's Real Property dataset, ranks or
filters the returned records, and maps SDAT columns into the application's
``ExtractedMetadata`` object.

Lookup priority elsewhere in the app is deliberate:
    1. District + account number (Tax ID) is most specific.
    2. Street address is useful but can match several parcels.
    3. Tax map / parcel are fallbacks when stronger identifiers are absent.

Network results are treated cautiously. Mass Scan requires a uniquely
convincing address match because every PDF may represent a different job."""

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
    """Normalized identifiers available for one Maryland property lookup.
    
    The object separates county, lot, map, parcel, district, and account number so lookup code
    can build strategies from strongest to weakest. ``tax_id`` is retained for context, while
    district/account are the components actually used by the most specific query."""

    county: str = ""
    lot: str = ""
    tax_map: str = ""
    parcel: str = ""
    tax_id: str = ""
    district: str = ""
    account_number: str = ""


def lookup_by_tax_id(tax_id: str, county: str = "") -> list[dict[str, Any]]:
    """Perform the application's most specific SDAT lookup from a Tax ID.
    
    The canonical Tax ID is split into district and account components. Invalid or incomplete
    values return no records immediately; valid parts are wrapped in ``SdatSearchTerms`` and
    delegated to the general strategy engine."""
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
    """Distinguish an SDAT property printout from an engineering drawing.
    
    The check uses several stable government-page phrases rather than one fragile title. A
    strong department/search header plus account block is sufficient, or at least four known
    anchors may identify the page. Lookup printouts supply property evidence but are not filed
    as permanent plan documents."""
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
    """Read district and account number directly from an SDAT printout's OCR text.
    
    Ordered patterns handle common page layouts and OCR punctuation. Letter-shaped digit errors
    are repaired, district is padded to two digits, and account is padded to six. The function
    returns both components plus the canonical combined Tax ID for downstream voting."""
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
    """Escape a value before inserting it into a quoted SoQL filter.
    
    Socrata uses two apostrophes to represent one literal apostrophe. Applying that rule here
    prevents an address or county name containing an apostrophe from breaking the query string."""
    return str(value or "").replace("'", "''").strip()


def or_equals(
    field: str, value: str, widths: Iterable[int] = (2, 3, 4, 6, 8)
) -> str:
    """Build a parenthesized SoQL condition covering equivalent identifier formats.
    
    SDAT may store the same number with different leading-zero widths. ``identifier_options``
    generates those forms, and this helper joins exact comparisons with ``OR`` so one request
    can match any valid representation."""
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
    """Assemble the strongest available property identifiers from OCR, metadata, and configuration.
    
    Existing extracted metadata is preferred; missing values are searched directly from text.
    County falls back to the configured default, and a Tax ID is split into district/account.
    Unknown placeholders are converted to empty terms so they cannot become query filters.
    
    Values are sanitized into stable strings before being placed in ``SdatSearchTerms``."""
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
    """Return the exact Maryland dataset columns needed by matching and PDF metadata.
    
    Restricting ``$select`` reduces response size while still including core identifiers,
    formatted-address components, links, and every hidden SDAT field written to XMP. The field
    list is built centrally so query and mapping code cannot drift apart."""
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
    """Send one filtered request to Maryland's Socrata endpoint and decode the records.
    
    Caller-supplied predicates are joined with ``AND`` and only selected columns are requested.
    A 20-second timeout prevents a stalled network call from hanging a scan indefinitely.
    Unsuccessful responses print the request and body for diagnosis before raising the HTTP error."""
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
    """Compare one returned SDAT identifier with a target independent of padding or punctuation.
    
    Human-facing names such as ``tax_map`` are translated to dataset field keys, then both sides
    pass through ``normalize_identifier``. An empty target imposes no restriction and therefore
    matches every record."""
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
    """Narrow a broad SDAT result set with every confident identifier available.
    
    Map, parcel, lot, district, and account filters are applied one record at a time. If filtering
    removes everything, the original records are returned rather than pretending the network
    lookup failed; the caller can still inspect or rank the broader result."""
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
    """Try SDAT query strategies from most specific to progressively broader fallbacks.
    
    Strategy order matters:
        1. District + account + county.
        2. District + account without county when county OCR may be wrong.
        3. County + map.
        4. County + parcel.
    
    The first strategy that returns records wins. Exact Tax ID strategies are trusted directly;
    broader map/parcel results are post-filtered with the remaining identifiers. This ordering
    reduces false matches and unnecessary network requests."""
    county_filter = (
        f"upper({SDAT_FIELDS['county']}) like upper('%{soql_escape(terms.county)}%')"
        if terms.county
        else ""
    )

    # Each tuple contains query predicates plus a flag indicating whether broad
    # results need local identifier filtering. Strategy order is the safety policy.
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

    # Stop after the first successful strategy. Running weaker fallbacks after a
    # precise account match would waste time and could introduce ambiguity.
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
    """Build the display address used for review, naming, and address comparison.
    
    Separate premise number/name/type/city/ZIP columns are preferred. If they do not produce a
    street address, the function falls back to Maryland's combined address fields. Empty pieces
    are omitted so the result does not contain repeated spaces or placeholder punctuation."""
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
    """Construct the canonical district-account Tax ID from one SDAT record.
    
    District and account values are normalized and padded to the widths expected by the app.
    Missing either component returns an empty string, preventing a partial identifier from being
    treated as authoritative."""
    district = normalize_value(record.get(SDAT_FIELDS["district"], ""))
    account = normalize_value(record.get(SDAT_FIELDS["account_number"], ""))
    if district and account:
        return f"{district.zfill(2)}-{account.zfill(6)}"
    return ""


def normalize_sdat_metadata_value(value: Any) -> str:
    """Convert any SDAT field value into stable text suitable for JSON and XMP.
    
    Most columns are scalars, while location fields may be dictionaries or lists. Structured
    values are serialized with sorted keys for repeatable output; scalars use ordinary metadata
    normalization. Null and empty values remain empty."""
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_value(str(value))


def metadata_from_sdat_record(
    metadata: ExtractedMetadata, record: dict[str, Any]
) -> ExtractedMetadata:
    """Merge one authoritative SDAT record into an immutable metadata result.
    
    Visible property fields are formatted and path-sanitized because they influence folder and
    file names. Hidden dataset fields are converted to stable text for state and XMP. Any SDAT
    column that is empty leaves the existing OCR value unchanged, so enrichment never replaces
    useful evidence with a blank."""
    address = format_sdat_address(record)
    lot = normalize_value(record.get(SDAT_FIELDS["lot"], ""))
    tax_map = normalize_value(record.get(SDAT_FIELDS["map"], ""))
    parcel = normalize_value(record.get(SDAT_FIELDS["parcel"], ""))
    section = normalize_value(record.get(SDAT_FIELDS["section"], ""))
    tax_id = tax_id_from_sdat_record(record)
    # ``replace`` creates a new frozen dataclass. Every field falls back to the
    # prior OCR value when SDAT omitted it, so enrichment is additive rather
    # than destructive.
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
    """Reduce a user/OCR address to a street number and a few strong street-name words.
    
    Punctuation, state names, street-type abbreviations, and other weak tokens are removed. The
    remaining number and first three words are sufficient to create a selective Socrata query
    without requiring exact spelling of the entire formatted address."""
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
    """Query SDAT by street number/name and rank the returned properties by similarity.
    
    SDAT stores premise numbers as five-character strings, so the number is zero-padded before
    querying. The first strong street word and optional county narrow the network result. Each
    record is then scored by how many target tokens appear in its formatted address, with a large
    bonus for an exact normalized match.
    
    The ordered list lets Batch mode use the best candidate and lets Mass mode apply an additional
    uniqueness check before accepting it."""
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
        """Assign one address candidate a simple, explainable match score.
        
        The score counts the street number and strong street words present in the candidate and adds
        an exact-address bonus. It is used only for sorting records returned by an already narrowed
        SDAT query."""
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
    """Perform the general one-document SDAT enrichment path.
    
    When SDAT lookup is disabled, metadata passes through unchanged. Otherwise search terms are
    extracted, the strategy engine is called, and the first successful record is merged. Batch
    mode normally uses its more deliberate shared-field lookup in ``pipeline.py`` instead."""
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
