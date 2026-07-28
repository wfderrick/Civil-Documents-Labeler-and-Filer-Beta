"""Normalize and compare Maryland property Tax IDs consistently.

Tax IDs appear in OCR text, browser edits, SDAT responses, and query
filters. OCR may confuse O/0, I/1, l/1, S/5, or use different dash
characters. These helpers repair those common forms and produce one
canonical ``DD-ACCOUNT`` representation before validation or comparison.

Centralizing the rules prevents each module from interpreting the same
property identifier differently."""

from __future__ import annotations

import re

_TAX_ID_RE = re.compile(r"^(\d{1,2})-(\d{4,8})$")
_OCR_DIGITS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    }
)


def normalize_tax_id(tax_id: str) -> str:
    """Convert a loosely formatted or OCR-damaged Tax ID into ``DD-ACCOUNT`` form.
    
    Common letter/digit confusions and Unicode dashes are repaired first. With an explicit dash,
    the two sides are cleaned separately; without one, a long digit string is split after the first
    two digits. Values too short to identify both parts return an empty string."""
    value = str(tax_id or "").strip().translate(_OCR_DIGITS)
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    digits = re.sub(r"\D", "", value)
    if not digits:
        return ""
    if "-" in value:
        left, right = value.split("-", 1)
        district = re.sub(r"\D", "", left).zfill(2)
        account = re.sub(r"\D", "", right)
    elif len(digits) >= 6:
        district, account = digits[:2], digits[2:]
    else:
        return ""
    return f"{district}-{account}"


def is_valid_tax_id(tax_id: str) -> bool:
    """Check whether a value becomes a complete Tax ID after normalization.
    
    Validation uses the shared canonical form, so harmless OCR spacing or dash variations do not
    make an otherwise usable identifier fail. The district must contain one or two digits before
    padding, and the account portion must contain four to eight digits."""
    return bool(_TAX_ID_RE.fullmatch(normalize_tax_id(tax_id)))


def extract_tax_id_parts(tax_id: str) -> tuple[str, str]:
    """Return normalized district and account components for SDAT querying.
    
    Invalid values return two empty strings rather than raising, allowing lookup callers to stop
    cleanly. Valid output is already repaired and ready for zero-padding or query construction."""
    normalized = normalize_tax_id(tax_id)
    match = _TAX_ID_RE.fullmatch(normalized)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def format_tax_id(district: str, account_number: str) -> str:
    """Combine separate district and account inputs through the canonical normalizer.
    
    Using ``normalize_tax_id`` here guarantees browser/SDAT components receive the same OCR repair,
    padding, and dash rules as a Tax ID read from plan text."""
    return normalize_tax_id(f"{district}-{account_number}")


def tax_id_matches(first: str, second: str) -> bool:
    """Compare two Tax IDs by their canonical forms.
    
    Both sides must normalize to non-empty values, preventing two invalid inputs from comparing as
    equal merely because they both became an empty string."""
    left, right = normalize_tax_id(first), normalize_tax_id(second)
    return bool(left and right and left == right)
