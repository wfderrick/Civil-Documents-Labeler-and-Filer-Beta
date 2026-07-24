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
    return ExtractedMetadata(
        lot="Unknown Lot",
        address="Unknown Address",
        project_code="AA",
        document_type="Site Plan",
    )


def test_requested_sdat_fields_are_selected_and_mapped():
    selected = set(selected_sdat_fields())
    assert set(SDAT_METADATA_FIELDS) <= set(SDAT_FIELDS)
    assert {SDAT_FIELDS[field] for field in SDAT_METADATA_FIELDS} <= selected


def test_sdat_metadata_is_retained_in_document_metadata():
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
    """Record.

    Args:
        number: Input used by this operation.
        street: Input used by this operation.
        account: Input used by this operation.

    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
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
    """Run the module as a command-line entry point."""
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