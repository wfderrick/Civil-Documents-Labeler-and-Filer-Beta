"""Regression checks for Mass Scan Tax ID isolation."""

from pipeline import _confident_unique_address_record
from sdat import SDAT_FIELDS


def record(number: str, street: str, account: str) -> dict[str, str]:
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

    print("Mass Scan Tax ID isolation tests passed.")


if __name__ == "__main__":
    main()
