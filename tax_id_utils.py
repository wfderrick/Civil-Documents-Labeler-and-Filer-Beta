"""Tax-account identifier normalization, validation, formatting, and comparison helpers shared by OCR extraction, SDAT lookup, and document editing.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""

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
    """Normalize tax id.
    
    Args:
        tax_id: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
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
    """Is valid tax id.
    
    Args:
        tax_id: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    return bool(_TAX_ID_RE.fullmatch(normalize_tax_id(tax_id)))


def extract_tax_id_parts(tax_id: str) -> tuple[str, str]:
    """Extract tax id parts.
    
    Args:
        tax_id: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    normalized = normalize_tax_id(tax_id)
    match = _TAX_ID_RE.fullmatch(normalized)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def format_tax_id(district: str, account_number: str) -> str:
    """Format tax id.
    
    Args:
        district: Input used by this operation.
        account_number: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    return normalize_tax_id(f"{district}-{account_number}")


def tax_id_matches(first: str, second: str) -> bool:
    """Tax id matches.
    
    Args:
        first: Input used by this operation.
        second: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    left, right = normalize_tax_id(first), normalize_tax_id(second)
    return bool(left and right and left == right)
