from __future__ import annotations
import re
import sys
from dataclasses import dataclass, replace
from typing import Any, Iterable
import requests
from tax_id_utils import extract_tax_id_parts
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
}


@dataclass(frozen=True)
class SdatSearchTerms:
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
        normalize_for_fuzzy(anchor) in normalized for anchor in SDAT_LOOKUP_ANCHORS
    )
    strong_header = (
        normalize_for_fuzzy("department of assessments and taxation") in normalized
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
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if not match:
            continue
        district = re.sub(r"\D", "", normalize_ocr_numbers(match.group(1))).zfill(2)
        account = re.sub(r"\D", "", normalize_ocr_numbers(match.group(2))).zfill(6)
        if district and account:
            return district, account, f"{district}-{account}"
    return None


def soql_escape(value: str) -> str:
    return str(value or "").replace("'", "''").strip()


def or_equals(field: str, value: str, widths: Iterable[int] = (2, 3, 4, 6, 8)) -> str:
    options = identifier_options(value, widths)
    return (
        "("
        + " OR ".join(f"{field} = '{soql_escape(option)}'" for option in options)
        + ")"
    )


def extract_sdat_search_terms(
    text: str, metadata: ExtractedMetadata, config: Config
) -> SdatSearchTerms:
    county = first_match(text, config.get("county_patterns", [])) or config.get(
        "default_county", ""
    )
    county = re.sub(r"\bcounty\b", "", str(county), flags=re.IGNORECASE).strip()
    tax_map = (
        metadata.tax_map
        or first_match(text, config.get("map_patterns", []), normalize_numbers=False)
        or ""
    )
    parcel = (
        metadata.parcel
        or first_match(text, config.get("parcel_patterns", []), normalize_numbers=False)
        or ""
    )
    tax_id = (
        metadata.tax_id
        or first_match(text, config.get("tax_id_patterns", []), normalize_numbers=True)
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
        or first_match(text, config.get("account_patterns", []), normalize_numbers=True)
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
        account_number=safe_path_part(account_number, "") if account_number else "",
    )


def selected_sdat_fields() -> list[str]:
    return [
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


def sdat_get(where_parts: list[str], limit: int = 200) -> list[dict[str, Any]]:
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


def record_identifier_matches(record: dict[str, Any], key: str, target: str) -> bool:
    if not target:
        return True
    field = {"tax_map": "map", "account_number": "account_number"}.get(key, key)
    return normalize_identifier(
        record.get(SDAT_FIELDS[field], "")
    ) == normalize_identifier(target)


def filter_sdat_records(
    records: list[dict[str, Any]], terms: SdatSearchTerms
) -> list[dict[str, Any]]:
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
        if terms.lot and not record_identifier_matches(record, "lot", terms.lot):
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


def lookup_maryland_property_records(terms: SdatSearchTerms) -> list[dict[str, Any]]:
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
                        SDAT_FIELDS["account_number"], terms.account_number, (6, 8)
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
                        SDAT_FIELDS["account_number"], terms.account_number, (6, 8)
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
                [county_filter, or_equals(SDAT_FIELDS["map"], terms.tax_map, (3, 4))],
                True,
            )
        )

    if county_filter and terms.parcel:
        strategies.append(
            (
                [county_filter, or_equals(SDAT_FIELDS["parcel"], terms.parcel, (3, 4))],
                True,
            )
        )

    for where_parts, should_filter in strategies:
        records = sdat_get(where_parts)
        if records:
            return filter_sdat_records(records, terms) if should_filter else records

    return []


def format_sdat_address(record: dict[str, Any]) -> str:
    number = normalize_value(record.get(SDAT_FIELDS["premise_number"], ""))
    street = normalize_value(record.get(SDAT_FIELDS["premise_name"], ""))
    street_type = normalize_value(record.get(SDAT_FIELDS["premise_type"], ""))
    city = normalize_value(record.get(SDAT_FIELDS["premise_city"], ""))
    zip_code = normalize_value(record.get(SDAT_FIELDS["premise_zip"], ""))
    street_address = " ".join(
        part for part in [number, street, street_type] if part
    ).strip()
    if not street_address:
        street_address = normalize_value(record.get(SDAT_FIELDS["mdp_address"], ""))
        city = city or normalize_value(record.get(SDAT_FIELDS["mdp_city"], ""))
        zip_code = zip_code or normalize_value(record.get(SDAT_FIELDS["mdp_zip"], ""))
    return (
        " ".join(
            part for part in [street_address, city, "MD", zip_code] if part
        ).strip()
        if street_address
        else ""
    )


def tax_id_from_sdat_record(record: dict[str, Any]) -> str:
    district = normalize_value(record.get(SDAT_FIELDS["district"], ""))
    account = normalize_value(record.get(SDAT_FIELDS["account_number"], ""))
    if district and account:
        return f"{district.zfill(2)}-{account.zfill(6)}"
    return ""


def metadata_from_sdat_record(
    metadata: ExtractedMetadata, record: dict[str, Any]
) -> ExtractedMetadata:
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
            safe_path_part(address, metadata.address) if address else metadata.address
        ),
        tax_map=safe_path_part(tax_map, "") if tax_map else metadata.tax_map,
        parcel=safe_path_part(parcel, "") if parcel else metadata.parcel,
        tax_id=safe_path_part(tax_id, "") if tax_id else metadata.tax_id,
        section=safe_path_part(section, "") if section else metadata.section,
    )


def _address_tokens(address: str) -> tuple[str, list[str]]:
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
    words = [part for part in parts[1:] if part not in stop and not part.isdigit()]
    return number, words[:3]


def lookup_maryland_property_by_address(
    address: str, county: str = "", limit: int = 100
) -> list[dict[str, Any]]:
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
        candidate = re.sub(r"[^A-Z0-9]", "", format_sdat_address(record).upper())
        return sum(1 for token in [number, *words] if token and token in candidate) + (
            5 if candidate == target else 0
        )

    return sorted(records, key=score, reverse=True)


def enrich_metadata_with_sdat(
    metadata: ExtractedMetadata, text: str, config: Config
) -> ExtractedMetadata:
    if not config.get("sdat_lookup", True):
        return metadata
    records = lookup_maryland_property_records(
        extract_sdat_search_terms(text, metadata, config)
    )
    return metadata_from_sdat_record(metadata, records[0]) if records else metadata
