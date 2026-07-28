"""Regression tests for SDAT field mapping and safe address resolution.

The app stores many SDAT values in PDF metadata even when they are not shown in the
review form. This file checks that those fields are requested and retained, and that
Mass Scan accepts an address only when it identifies one unambiguous parcel."""

from dataclasses import asdict

from metadata_extraction import ExtractedMetadata
from pipeline import _confident_unique_address_record
from sdat import (
    SDAT_FIELDS,
    SDAT_METADATA_FIELDS,
    metadata_from_sdat_record,
    selected_sdat_fields,
)


def base_metadata() -> ExtractedMetadata:
    """Create a minimal ExtractedMetadata seed for SDAT mapping tests.

    Unknown property placeholders imitate OCR that found a document type/project code but
    needs authoritative SDAT values to fill the remaining fields."""
    return ExtractedMetadata(
        lot="Unknown Lot",
        address="Unknown Address",
        project_code="AA",
        document_type="Site Plan",
    )


def test_requested_sdat_fields_are_selected_and_mapped():
    """Verify that every app-level SDAT metadata field maps to a requested API column.

    Without this relationship, metadata_from_sdat_record() could expect a value that the
    Socrata query never selected, silently leaving PDF metadata incomplete."""
    selected = set(selected_sdat_fields())
    assert set(SDAT_METADATA_FIELDS) <= set(SDAT_FIELDS)
    assert {SDAT_FIELDS[field] for field in SDAT_METADATA_FIELDS} <= selected


def test_sdat_metadata_is_retained_in_document_metadata():
    """Verify that all selected SDAT values survive conversion into document metadata.

    The latitude/longitude object is also checked after serialization because structured
    Socrata values must remain available for later XMP writing."""
    record = {SDAT_FIELDS[field]: f"value-{index}" for index, field in enumerate(SDAT_METADATA_FIELDS)}
    record[SDAT_FIELDS["mappable_latitude_and_longitude"]] = { # type: ignore
        "latitude": "38.5",
        "longitude": "-76.5",
    }

    resolved = metadata_from_sdat_record(base_metadata(), record)
    values = asdict(resolved)

    for field in SDAT_METADATA_FIELDS:
        assert values[field]
    assert '"latitude": "38.5"' in values["mappable_latitude_and_longitude"]

def record(number: str, street: str, account: str) -> dict[str, str]:
    """Build one synthetic SDAT parcel record for address-uniqueness checks.

    The premise number is zero-padded to mimic the Maryland dataset, while street, city,
    and account fields provide enough data for format_sdat_address() and parcel identity.

    Args:
        number: Street number before SDAT-style zero padding.
        street: Street name without the fixed RD suffix.
        account: Distinct parcel account number.

    Returns:
        Dictionary shaped like one relevant SDAT API record."""
    return {
        SDAT_FIELDS["premise_number"]: number.zfill(5),
        SDAT_FIELDS["premise_name"]: street,
        SDAT_FIELDS["premise_type"]: "RD",
        SDAT_FIELDS["premise_city"]: "PRINCE FREDERICK",
        SDAT_FIELDS["premise_zip"]: "20678",
        SDAT_FIELDS["district"]: "01",
        SDAT_FIELDS["account_number"]: account,
    }


def main() -> None:
    """Exercise Mass Scan's conservative address-record selection rules.

    One matching parcel is accepted. Two parcels at the same address are rejected as
    ambiguous, and an unrelated address is rejected. These checks prevent a Tax ID from
    being guessed and copied into an independent PDF."""
    unique = [record("123", "MAIN", "111111")]
    assert _confident_unique_address_record("123 Main Rd", unique) is unique[0]

    # Two parcels can share one mailing/premise address. Mass Scan must not pick
    # the first record and give that Tax ID to every PDF.
    ambiguous = [
        record("123", "MAIN", "111111"),
        record("123", "MAIN", "222222"),
    ]
    assert _confident_unique_address_record("123 Main Rd", ambiguous) is None

    unrelated = [record("987", "OTHER", "333333")]
    assert _confident_unique_address_record("123 Main Rd", unrelated) is None
