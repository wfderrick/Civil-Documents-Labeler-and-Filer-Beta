from __future__ import annotations
import re

_TAX_ID_RE = re.compile(r"^(\d{1,2})-(\d{4,8})$")
_OCR_DIGITS = str.maketrans({"O":"0","o":"0","I":"1","i":"1","l":"1","|":"1","S":"5","s":"5","B":"8"})

def normalize_tax_id(tax_id: str) -> str:
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
    return bool(_TAX_ID_RE.fullmatch(normalize_tax_id(tax_id)))

def extract_tax_id_parts(tax_id: str) -> tuple[str, str]:
    normalized = normalize_tax_id(tax_id)
    match = _TAX_ID_RE.fullmatch(normalized)
    if not match:
        return "", ""
    return match.group(1), match.group(2)

def format_tax_id(district: str, account_number: str) -> str:
    return normalize_tax_id(f"{district}-{account_number}")

def tax_id_matches(first: str, second: str) -> bool:
    left, right = normalize_tax_id(first), normalize_tax_id(second)
    return bool(left and right and left == right)
