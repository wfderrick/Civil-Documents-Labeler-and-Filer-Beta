from dataclasses import asdict

from metadata_extraction import ExtractedMetadata
from sdat import SDAT_FIELDS, SDAT_METADATA_FIELDS, metadata_from_sdat_record, selected_sdat_fields


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
    record[SDAT_FIELDS["mappable_latitude_and_longitude"]] = {
        "latitude": "38.5",
        "longitude": "-76.5",
    }

    resolved = metadata_from_sdat_record(base_metadata(), record)
    values = asdict(resolved)

    for field in SDAT_METADATA_FIELDS:
        assert values[field]
    assert '"latitude": "38.5"' in values["mappable_latitude_and_longitude"]
