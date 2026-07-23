"""Text-to-metadata extraction and document classification. This module converts OCR text into structured project, property, document-type, and title-block fields while preserving confidence and source information used by later review steps.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

DOCUMENT_TYPE_THRESHOLD = 0.75

INVALID_PATH_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

DEFAULT_CONFIG: dict[str, Any] = {
    "sdat_lookup": True,
    "ocr_device": "auto",
    "gpu_device_id": 0,
    "parallel_ocr": False,
    "ocr_workers": 1,
    "ocr_threads_per_worker": 4,
    "visual_field_notes_classifier": True,
    "visual_field_notes_threshold": 0.75,
    "default_county": "Calvert",
    "lot_pattern": [r"\blot\s*[:#-]?\s*([0-9]+R?)\b"],
    "county_patterns": [r"\b([A-Za-z]+)\s+County\b"],
    "map_patterns": [
        r"\btax\s+map\s*[:#-]?\s*([0-9]+[A-Za-z]*)\b",
    ],
    "parcel_patterns": [
        r"\bparcel\s*[:#-]?\s*([0-9A-Za-z]+)\b",
        r"\bmap\s*/\s*parcel\s*[:#-]?\s*[0-9A-Za-z]+\s*/\s*([0-9A-Za-z]+)\b",
    ],
    "tax_id_patterns": [
        r"\btax\s*(?:id|i\.?d\.?|1\.?d\.?)\s*[:#.-]?\s*"
        r"([0-9Oo]{1,2})\s*[- ]\s*([0-9OoIl]{4,8})\b",
    ],
    "district_patterns": [
        r"\bdistrict\s*[:#-]?\s*([0-9A-Za-z]+)\b",
        r"\bdist\.?\s*[:#-]?\s*([0-9A-Za-z]+)\b",
    ],
    "account_patterns": [
        r"\baccount\s*(?:number|no\.?|#)?\s*[:#-]?\s*([0-9A-Za-z]+)\b",
        r"\bacct\.?\s*(?:no\.?|#)?\s*[:#-]?\s*([0-9A-Za-z]+)\b",
    ],
    "address_patterns": [
        r"\s(?:property|site|project)\s+address\s*[:#-]?\s*(.+)",
        r"\saddress\s*[:#-]?\s*(.+)",
        r"(?<!\w)([1-9]\d{0,5}\s+[A-Za-z][A-Za-z0-9 .'-]*\s+(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|court|ct\.?|circle|cir\.?|way|place|pl\.?)\b[^\n]*)",
    ],
    "bbox_address_bottom_fraction": 0.65,
    "bbox_address_line_tolerance": 0.75,
    "ignored_address_keywords": [
        "phone",
        "fax",
        "www",
        ".com",
        "@",
        "survey",
        "surveyor",
        "surveying",
        "engineer",
        "engineering",
    ],
    "ignored_addresses": [],
    "project_code_patterns": [r"\s(aa|cc|ch|nav|pg|sm|usaf[0-9]{4})\s"],
    "document_type_keywords": {
        "House Location": [
            "house location",
            "houselocation",
            "house loc",
            "hse location",
            "location drawing",
        ],
        "Site Plan": ["site plan", "siteplan", "plot plan", "sitemap"],
        "Wall Check": ["wall check", "wallcheck", "wall chk", "foundation check"],
        "Plat/Replat": [
            "forest conservation amendment plat",
            "replat",
            "re plat",
        ],
        "Construction Permit": ["septic construction permit", "construction permit"],
        "Field Notes": ["field notes", "fieldnotes", "field note", "notes"],
    },
    "document_type_regex_rules": {
        "Site Plan": [
            r"\bsite\s+plan\b[\s\S]{0,160}?\beasement\s+plat\b",
            r"\b(?:sewage|drainage|utility|access|ingress|egress|storm\s*drain|water|sanitary)\s+easement\s+plat\b",
        ],
        "Plat/Replat": [
            r"\bforest\s+conservation\s+amendment\s+plat\b",
        ],
        "Construction Permit": [r"\bseptic\s+construction\s+permit\b"],
    },
    "document_type_patterns": [
        r"\s(wall check|site plan|field notes|replat|house location|construction permit)\s"
    ],
}

OCR_CONFUSION_MAP = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "I": "l",
        "|": "l",
        "!": "l",
        "5": "s",
        "$": "s",
        "3": "e",
        "@": "a",
        "8": "b",
        "6": "g",
        "2": "z",
        "+": "t",
    }
)

OCR_NUMBER_MAP = str.maketrans(
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


def normalize_ocr_numbers(text: str) -> str:
    """Normalize OCR mistakes that commonly appear inside numeric identifiers."""
    return str(text or "").translate(OCR_NUMBER_MAP)


@dataclass(frozen=True)
class ExtractedMetadata:
    """Structured OCR and SDAT metadata carried through the scan pipeline.

    The first eight fields are used by the review interface and output naming.
    The remaining fields are SDAT-only property details that are retained in the
    document record and written to XMP without adding controls to the UI.
    """

    lot: str
    address: str
    project_code: str
    document_type: str
    tax_map: str = ""
    parcel: str = ""
    tax_id: str = ""
    section: str = ""
    jurisdiction_code_mdp_field_jurscode: str = ""
    finder_online_link: str = ""
    mdp_longitude_mdp_field_digxcord_converted_to_wgs84: str = ""
    mdp_latitude_mdp_field_digycord_converted_to_wgs84: str = ""
    mappable_latitude_and_longitude: str = ""
    legal_description_line_1_mdp_field_legal1_sdat_field_17: str = ""
    legal_description_line_2_mdp_field_legal2_sdat_field_18: str = ""
    deed_reference_1_liber_mdp_field_dr1liber_sdat_field_30: str = ""
    deed_reference_1_folio_mdp_field_dr1folio_sdat_field_31: str = ""
    subdivision_code_mdp_field_subdivsn_sdat_field_37: str = ""
    grid_mdp_field_grid_sdat_field_43: str = ""
    zoning_code_mdp_field_zoning_sdat_field_45: str = ""
    land_use_code_mdp_field_lu_desclu_sdat_field_50: str = ""
    property_factors_utilities_water_mdp_field_pfuw_sdat_field_63: str = ""
    property_factors_utilities_sewer_mdp_field_pfus_sdat_field_64: str = ""
    property_factors_location_waterfront_mdp_field_pflw_sdat_field_65: str = ""
    property_factors_street_paved_mdp_field_pfsp_sdat_field_67: str = ""
    property_factors_street_unpaved_mdp_field_pfsu_sdat_field_68: str = ""


@dataclass(frozen=True)
class FuzzyMatch:
    """Represent FuzzyMatch behavior and related state.
    """
    label: str
    score: float
    start: int
    end: int
    matched_text: str
    keyword: str


Config = dict[str, Any]


def load_config(path: Path | None) -> Config:
    """The load_config() function returns a dictionary with settings and regex
    patterns which determine how the app functions. If the path parameter is
    None the DEFAULT_CONFIG is returned. Otherwise the file pointed to by the
    path parameter is opened and saved in config_file. config_file is then
    loaded as a json to convert it into a python object which is saved into
    user_config. Next, the default config is updated with the updated settings
    from user_config, so settings are changed or added depending on whether they
    already exist. Finally config is returned."""
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with path.open("r", encoding="utf-8") as config_file:
        user_config = json.load(config_file)
    config.update(
        {
            key: value
            for key, value in user_config.items()
            if value not in (None, "", [])
        }
    )
    return config


def normalize_value(value: str) -> str:
    """The normalize_value() function returns a cleaned version of the value
    parameter. To begin it calls the built in str() class on either value if it is
    not None or an empty string to convert that result to a string. The split()
    function is called on the result of that in order to separate the non whitespace
    characters into groups and then rejoined using the join() function called on " "
    so that each group in the list generated from split() is separated by a single
    space in the singel string generated from the join() function.
    Then strip() is called on the result of join() to remove any leading or
    trailing spaces and/or bad characters like :, -, #, ., ,,and/or ;."""
    return " ".join(str(value or "").split()).strip(" :-#.,;")


def first_match(
    text: str, patterns: Iterable[str], *, normalize_numbers: bool = False
) -> str | None:
    """Return the first regex capture, optionally fixing OCR digit/letter mistakes first."""
    search_text = normalize_ocr_numbers(text) if normalize_numbers else str(text or "")

    for pattern in patterns:
        match = re.search(pattern, search_text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue

        if match.lastindex and match.lastindex > 1:
            return "-".join(
                normalize_value(match.group(i)) for i in range(1, match.lastindex + 1)
            )

        return normalize_value(match.group(1))

    return None


def all_matches(
    text: str, patterns: Iterable[str], *, normalize_numbers: bool = False
) -> list[str]:
    """Return all regex captures, optionally fixing OCR digit/letter mistakes first."""
    search_text = normalize_ocr_numbers(text) if normalize_numbers else str(text or "")
    values: list[str] = []

    for pattern in patterns:
        for match in re.finditer(
            pattern, search_text, flags=re.IGNORECASE | re.MULTILINE
        ):
            if match.lastindex and match.lastindex > 1:
                values.append(
                    "-".join(
                        normalize_value(match.group(i))
                        for i in range(1, match.lastindex + 1)
                    )
                )
            else:
                values.append(normalize_value(match.group(1)))

    return values


def safe_path_part(value: str, fallback: str) -> str:
    """The safe_path_part() function returns a string which represents a file or
    folder name which is allowed. To begin the normalize_value() function is called
    to remove large groups of spaces and some invalid characters. Next sub() is
    called to substitute more invalid characters out of the string. strip() is
    called as a final check to remove any spaces or periods at the end of a folder
    or file name since it's not allowed in Windows. Finally a truncated version of
    the cleaned value string is returned or if its a None type the fallback
    parameter is returned."""
    value = normalize_value(value) or fallback
    value = INVALID_PATH_RE.sub("", value)
    value = value.strip(" .")
    return value[:140] or fallback


def unique_path(path: Path) -> Path:
    """Unique path.
    
    Args:
        path: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    
    Notes:
        Errors are handled or propagated according to the surrounding scan/API workflow.
    """
    if not path.exists():
        return path
    for counter in range(2, 10000):
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique path for {path}")


def normalize_for_fuzzy(value: str) -> str:
    """Normalize for fuzzy.
    
    Args:
        value: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    return str(value or "").lower().translate(OCR_CONFUSION_MAP)


def keyword_groups(raw_keywords: Any) -> dict[str, list[str]]:
    """Keyword groups.
    
    Args:
        raw_keywords: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    if isinstance(raw_keywords, Mapping):
        return {
            str(label): (
                [str(item) for item in keywords]
                if isinstance(keywords, list)
                else [str(keywords)]
            )
            for label, keywords in raw_keywords.items()
        }
    if isinstance(raw_keywords, list):
        return {str(label): [str(label)] for label in raw_keywords}
    return {}


def best_keyword_window(keyword: str, normalized_text: str) -> tuple[float, int, int]:
    """Return the best fuzzy keyword window while preserving legacy tie behavior.

    Window lengths remain in ascending order and scores still use a strict ``>``
    comparison. ``real_quick_ratio`` and ``quick_ratio`` are upper-bound pruning
    checks, so skipping a window when either bound is less than or equal to the
    current best cannot change the winning score or the first tie-winner.
    """
    if not keyword or not normalized_text:
        return 0.0, -1, -1
    exact_start = normalized_text.find(keyword)
    if exact_start >= 0:
        return 1.0, exact_start, exact_start + len(keyword)

    keyword_length = len(keyword)
    min_window = max(3, keyword_length - 3)
    max_window = min(len(normalized_text), keyword_length + 4)
    best_score, best_start, best_end = 0.0, -1, -1

    matcher = SequenceMatcher(None, keyword, autojunk=False)
    for window_length in range(min_window, max_window + 1):
        for start in range(0, len(normalized_text) - window_length + 1):
            end = start + window_length
            matcher.set_seq2(normalized_text[start:end])

            # Both methods are documented upper bounds for ratio(). Because the
            # legacy code only updates on a strict improvement, <= is safe and
            # preserves the original first-match tie behavior exactly.
            if matcher.real_quick_ratio() <= best_score:
                continue
            if matcher.quick_ratio() <= best_score:
                continue

            score = matcher.ratio()
            if score > best_score:
                best_score, best_start, best_end = score, start, end
                if best_score >= 0.98:
                    return best_score, best_start, best_end
    return best_score, best_start, best_end


def regex_document_type(text: str, rules: Any) -> FuzzyMatch | None:
    """Return a normalized document type for an explicit regex rule match.

    Site Plan takes precedence when the title contains both ``site plan`` and
    ``easement plat``. Without this guard, the earlier Plat/Replat rule for an
    easement plat can win before the Site Plan rule is evaluated.
    """
    if not isinstance(rules, Mapping):
        return None

    site_plan_easement = re.search(
        r"\bsite\s+plan\b[\s\S]{0,160}?\beasement\s+plat\b"
        r"|\beasement\s+plat\b[\s\S]{0,160}?\bsite\s+plan\b",
        text or "",
        flags=re.IGNORECASE,
    )
    if site_plan_easement:
        return FuzzyMatch(
            label="Site Plan",
            score=1.0,
            start=site_plan_easement.start(),
            end=site_plan_easement.end(),
            matched_text=site_plan_easement.group(0),
            keyword="site plan + easement plat precedence",
        )

    for label, patterns in rules.items():
        for pattern in patterns or []:
            match = re.search(str(pattern), text or "", flags=re.IGNORECASE)
            if match:
                return FuzzyMatch(
                    label=str(label),
                    score=1.0,
                    start=match.start(),
                    end=match.end(),
                    matched_text=match.group(0),
                    keyword=str(pattern),
                )
    return None


def fuzzy_document_type(
    text: str, keywords: Any, threshold: float = DOCUMENT_TYPE_THRESHOLD
) -> FuzzyMatch | None:
    """Return the highest-scoring configured document type found in OCR text.

    Each configured keyword is normalized and compared with the most similar
    text window. The best match is returned only when its score meets
    ``threshold``; otherwise the function returns ``None``.

    Args:
        text: OCR text to classify.
        keywords: Document-type labels and their candidate phrases.
        threshold: Minimum similarity score required to accept a match.

    Returns:
        The strongest accepted match, or ``None`` when no candidate is reliable.
    """
    normalized_text = normalize_for_fuzzy(text)
    best: FuzzyMatch | None = None
    for label, candidates in keyword_groups(keywords).items():
        for keyword in candidates:
            score, start, end = best_keyword_window(
                normalize_for_fuzzy(keyword), normalized_text
            )
            if start < 0:
                continue
            match = FuzzyMatch(
                label=label,
                score=score,
                start=start,
                end=end,
                matched_text=text[start:end],
                keyword=keyword,
            )
            if best is None or match.score > best.score:
                best = match
    return best if best and best.score >= threshold else None


def is_ignored_address(address: str, config: Config) -> bool:
    """Is ignored address.
    
    Args:
        address: Input used by this operation.
        config: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    cleaned = normalize_for_fuzzy(address)
    compact = re.sub(r"[^a-z0-9]", "", cleaned)
    for blocked in config.get("ignored_addresses", []):
        blocked_clean = normalize_for_fuzzy(str(blocked))
        if blocked_clean and (
            blocked_clean in cleaned
            or re.sub(r"[^a-z0-9]", "", blocked_clean) in compact
        ):
            return True
    return any(
        normalize_for_fuzzy(str(keyword)) in cleaned
        for keyword in config.get("ignored_address_keywords", [])
        if normalize_for_fuzzy(str(keyword))
    )


def _ocr_item_rect(item: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """Return an OCR item's rectangle as x0, y0, x1, y1."""
    raw = first_nonempty_value(item.get("bbox"), item.get("polygon"))
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    points = _points_from_any(raw)
    if not points:
        return None
    x0, y0, x1, y1 = _bbox_from_points(points)
    return x0, y0, x1, y1


def _layout_address_lines(
    ocr_pages: Iterable[Mapping[str, Any]],
    *,
    bottom_fraction: float,
    line_tolerance: float,
) -> list[str]:
    """Rebuild bottom-page OCR lines from bounding boxes.

    Keeping tokens on their physical line prevents an isolated OCR token from a
    neighboring line (for example ``0``) from being prepended to a real address.
    """
    lines: list[str] = []
    for page in ocr_pages or []:
        try:
            image_height = float(page.get("image_height") or 0)
        except (TypeError, ValueError):
            image_height = 0.0
        if image_height <= 0:
            continue

        positioned: list[tuple[float, float, float, float, str]] = []
        cutoff = image_height * max(0.0, min(1.0, bottom_fraction))
        for item in page.get("items", []) or []:
            text = normalize_value(item.get("text", ""))
            rect = _ocr_item_rect(item)
            if not text or rect is None:
                continue
            x0, y0, x1, y1 = rect
            if (y0 + y1) / 2 < cutoff:
                continue
            positioned.append((x0, y0, x1, y1, text))

        if not positioned:
            continue

        heights = sorted(max(1.0, y1 - y0) for _, y0, _, y1, _ in positioned)
        median_height = heights[len(heights) // 2]
        tolerance = max(3.0, median_height * max(0.25, line_tolerance))

        rows: list[dict[str, Any]] = []
        for x0, y0, x1, y1, text in sorted(
            positioned, key=lambda value: ((value[1] + value[3]) / 2, value[0])
        ):
            center_y = (y0 + y1) / 2
            best_row = None
            best_distance = float("inf")
            for row in rows:
                distance = abs(center_y - row["center_y"])
                if distance <= tolerance and distance < best_distance:
                    best_row = row
                    best_distance = distance
            if best_row is None:
                rows.append({"center_y": center_y, "tokens": [(x0, text)]})
            else:
                best_row["tokens"].append((x0, text))
                count = len(best_row["tokens"])
                best_row["center_y"] = (
                    (best_row["center_y"] * (count - 1)) + center_y
                ) / count

        for row in sorted(rows, key=lambda value: value["center_y"]):
            line = " ".join(
                text for _, text in sorted(row["tokens"], key=lambda token: token[0])
            )
            line = normalize_value(line)
            if line:
                lines.append(line)
    return lines


def first_valid_address(
    text: str,
    config: Config,
    ocr_pages: Iterable[Mapping[str, Any]] | None = None,
) -> str | None:
    """Return a plausible address, preferring bounding-box reconstructed lines."""
    if ocr_pages:
        try:
            bottom_fraction = float(config.get("bbox_address_bottom_fraction", 0.65))
        except (TypeError, ValueError):
            bottom_fraction = 0.65
        try:
            line_tolerance = float(config.get("bbox_address_line_tolerance", 0.75))
        except (TypeError, ValueError):
            line_tolerance = 0.75

        for line in _layout_address_lines(
            ocr_pages,
            bottom_fraction=bottom_fraction,
            line_tolerance=line_tolerance,
        ):
            for address in all_matches(line, config.get("address_patterns", [])):
                if address and not is_ignored_address(address, config):
                    return address

    # Compatibility fallback for PDFs/results without usable bounding boxes.
    for address in all_matches(text, config.get("address_patterns", [])):
        if address and not is_ignored_address(address, config):
            return address
    return None


def _points_from_any(value: Any) -> list[list[float]]:
    """Points from any.
    
    Args:
        value: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    
    Notes:
        Errors are handled or propagated according to the surrounding scan/API workflow.
    """
    if not isinstance(value, (list, tuple)):
        return []
    points: list[list[float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    return points


def _bbox_from_points(points: list[list[float]]) -> list[float]:
    """Bbox from points.
    
    Args:
        points: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def first_nonempty_value(*values: Any) -> Any:
    """First nonempty value.
    
    Args:
        *values: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def is_known_value(value: str) -> bool:
    """Is known value.
    
    Args:
        value: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    value = str(value or "").strip()
    return (
        bool(value)
        and not value.lower().startswith("unknown")
        and value not in {"Project", "Document"}
    )


def extract_metadata(
    text: str,
    config: Config,
    default_project_code: str,
    default_document_type: str,
    ocr_pages: Iterable[Mapping[str, Any]] | None = None,
) -> ExtractedMetadata:
    """Extract metadata.
    
    Args:
        text: Input used by this operation.
        config: Input used by this operation.
        default_project_code: Input used by this operation.
        default_document_type: Input used by this operation.
        ocr_pages: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    from sdat import (
        LOOKUP_DOCUMENT_TYPE,
        extract_sdat_lookup_tax_id,
        is_sdat_lookup_document,
    )

    if is_sdat_lookup_document(text):
        lookup = extract_sdat_lookup_tax_id(text)
        tax_id = lookup[2] if lookup else ""
        return ExtractedMetadata(
            lot="Unknown Lot",
            address="Unknown Address",
            project_code=safe_path_part(default_project_code, "Project"),
            document_type=LOOKUP_DOCUMENT_TYPE,
            tax_id=tax_id,
        )

    doc_match = regex_document_type(text, config.get("document_type_regex_rules"))
    if doc_match is None:
        doc_match = fuzzy_document_type(text, config.get("document_type_keywords"))
    document_type = (
        doc_match.label
        if doc_match
        else first_match(text, config.get("document_type_patterns", []))
        or default_document_type
        or "Field Notes"
    )
    # Preserve the original lot technique: search only after the detected document type.
    lot_search_text = text[doc_match.start :] if doc_match else text
    lot = first_match(lot_search_text, config.get("lot_pattern", [])) or "Unknown Lot"
    tax_map = (
        first_match(text, config.get("map_patterns", []), normalize_numbers=False) or ""
    )
    parcel = (
        first_match(text, config.get("parcel_patterns", []), normalize_numbers=True)
        or ""
    )
    tax_id = (
        first_match(text, config.get("tax_id_patterns", []), normalize_numbers=True)
        or ""
    )
    return ExtractedMetadata(
        lot=safe_path_part(lot, "Unknown Lot"),
        address=safe_path_part(
            first_valid_address(text, config, ocr_pages) or "Unknown Address",
            "Unknown Address",
        ),
        project_code=safe_path_part(
            first_match(text, config.get("project_code_patterns", []))
            or default_project_code,
            "Project",
        ),
        # Keep the UI classification label intact. Filename creation sanitizes
        # path-invalid characters separately in document_service.suggested_filename().
        document_type=normalize_value(document_type) or "Field Notes",
        tax_map=safe_path_part(tax_map, "") if tax_map else "",
        parcel=safe_path_part(parcel, "") if parcel else "",
        tax_id=safe_path_part(tax_id, "") if tax_id else "",
    )


def prefer_known(value: str, fallback: str) -> str:
    """Prefer known.
    
    Args:
        value: Input used by this operation.
        fallback: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    return value if is_known_value(value) else fallback


def normalize_identifier(value: Any) -> str:
    """Normalize identifier.
    
    Args:
        value: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    cleaned = re.sub(r"[^0-9A-Za-z]", "", str(value or "")).upper()
    return cleaned.lstrip("0") or cleaned


def identifier_options(
    value: str, widths: Iterable[int] = (2, 3, 4, 6, 8)
) -> list[str]:
    """Identifier options.
    
    Args:
        value: Input used by this operation.
        widths: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
    compact = re.sub(r"[^0-9A-Za-z]", "", str(value or "")).upper()
    if not compact:
        return []
    options = {compact, compact.lstrip("0") or "0"}
    if compact.isdigit():
        options.update(
            compact.zfill(width) for width in widths if len(compact) <= width
        )
    return sorted(options)
