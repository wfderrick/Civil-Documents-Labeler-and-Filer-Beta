"""Convert OCR text and OCR layout information into structured metadata.

OCR returns mostly unstructured text. This module turns that text into
values the application can use: document type, lot, address, project
code, tax map, parcel, and Tax ID. It also contains the document-type
classifier and the layout-aware address finder.

The extraction sequence used by ``extract_metadata`` is:
    1. Detect whether the PDF is an SDAT lookup printout.
    2. Classify the engineering document type.
    3. Extract identifiers with ordered regular expressions.
    4. Reconstruct likely title-block address lines from OCR coordinates.
    5. Sanitize values before they are used in Windows paths.

``ExtractedMetadata`` is the common data object passed from this module
into the batch pipeline, SDAT enrichment, review interface, and PDF
metadata writer. Changes to its fields therefore affect several files."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

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
        r"\btax\s*(?:id|i\.?d\.?|1\.?d\.?)\s*[:#.-]?\s*([0-9Oo]{1,2})\s*[- ]\s*([0-9OoIl]{4,8})\b",
    ],
    "district_patterns": [
        r"\bdistrict\s*[:#-]?\s*([0-9A-Za-z]+)\b\bdist\.?\s*[:#-]?\s*([0-9A-Za-z]+)\b",
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
        "Wall Check": [
            "wall check",
            "wallcheck",
            "wall chk",
            "foundation check",
        ],
        "Plat/Replat": [
            "forest conservation amendment plat",
            "replat",
            "re plat",
        ],
        "Construction Permit": [
            "septic construction permit",
            "construction permit",
        ],
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
    """Repair letter-shaped OCR errors before parsing numeric identifiers.
    
    Tax IDs and parcel numbers frequently contain ``O`` read for zero or ``I``
    and ``l`` read for one. Translation is intentionally used only in numeric
    extraction paths; applying it to all OCR text would damage ordinary words."""
    return str(text or "").translate(OCR_NUMBER_MAP)


@dataclass(frozen=True)
class ExtractedMetadata:
    """Immutable container for every value carried through the metadata pipeline.
    
    The first fields drive review, naming, and SDAT searches. The longer fields
    mirror selected Maryland dataset columns and are written into XMP metadata
    even though they are not individually editable in the browser.
    
    ``frozen=True`` prevents accidental in-place changes. Callers use
    ``dataclasses.replace`` to create an updated copy, making it clearer which
    stage supplied a new value."""
    
    lot: str
    address: str
    project_code: str
    document_type: str
    county: str = ""
    account_id: str = ""
    tax_map: str = ""
    parcel: str = ""
    tax_id: str = ""
    section: str = ""
    legal_description_line_1_mdp_field_legal1_sdat_field_17: str = ""
    legal_description_line_2_mdp_field_legal2_sdat_field_18: str = ""
    deed_reference_1_liber_mdp_field_dr1liber_sdat_field_30: str = ""
    deed_reference_1_folio_mdp_field_dr1folio_sdat_field_31: str = ""
    grid_mdp_field_grid_sdat_field_43: str = ""
    zoning_code_mdp_field_zoning_sdat_field_45: str = ""
    land_use_code_mdp_field_lu_desclu_sdat_field_50: str = ""
    property_factors_utilities_water_mdp_field_pfuw_sdat_field_63: str = ""
    property_factors_utilities_sewer_mdp_field_pfus_sdat_field_64: str = ""
    plat_reference_liber_mdp_field_pltliber_sdat_field_267: str = ""
    plat_reference_folio_mdp_field_pltfolio_sdat_field_268: str = ""
    


@dataclass(frozen=True)
class FuzzyMatch:
    """Describe why the document-type classifier selected one label.
    
    Besides the winning label and similarity score, the object stores the text
    window and keyword that produced the score. ``start`` is also used to begin
    the lot search after the detected title, preserving the behavior of earlier
    versions of the application."""

    label: str
    score: float
    start: int
    end: int
    matched_text: str
    keyword: str


Config = dict[str, Any]

_CONFIG_CACHE: dict[tuple[str, int, int], Config] = {}


def load_config(path: Path | None) -> Config:
    """Build the effective configuration used by scanning and extraction.
    
    The built-in defaults guarantee every expected key exists. Non-empty values
    from ``config.json`` then override those defaults. A cache key containing the
    absolute path, modification time, and file size allows repeated scans to skip
    JSON parsing while automatically invalidating the cache after an edit.
    
    A copy is returned so a caller can safely add request-specific values such as
    the selected county without mutating the cached configuration."""
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config

    resolved = path.resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    with resolved.open("r", encoding="utf-8") as config_file:
        user_config = json.load(config_file)
    config.update(
        {
            key: value
            for key, value in user_config.items()
            if value not in (None, "", [])
        }
    )
    _CONFIG_CACHE.clear()
    _CONFIG_CACHE[cache_key] = dict(config)
    return config


def normalize_value(value: str) -> str:
    """Clean one regex capture without changing its meaningful internal text.
    
    OCR often inserts repeated spaces or leaves labels surrounded by punctuation.
    Splitting and rejoining collapses whitespace to single spaces; ``strip`` then
    removes common separators from the ends. This normalization is used before
    values are compared, displayed, or converted into path components."""
    return " ".join(str(value or "").split()).strip(" :-#.,;")


def first_match(
    text: str, patterns: Iterable[str], *, normalize_numbers: bool = False
) -> str | None:
    """Run ordered extraction patterns and return the first successful capture.
    
    Configuration order expresses priority: a specific title-block pattern can
    appear before a broad fallback. Multi-group patterns are joined with a dash,
    which is how district and account captures become one Tax ID. Optional number
    normalization repairs OCR digit errors before matching."""
    search_text = (
        normalize_ocr_numbers(text) if normalize_numbers else str(text or "")
    )

    for pattern in patterns:
        match = re.search(
            pattern, search_text, flags=re.IGNORECASE | re.MULTILINE
        )
        if not match:
            continue

        if match.lastindex and match.lastindex > 1:
            return "-".join(
                normalize_value(match.group(i))
                for i in range(1, match.lastindex + 1)
            )

        return normalize_value(match.group(1))

    return None


def all_matches(
    text: str, patterns: Iterable[str], *, normalize_numbers: bool = False
) -> list[str]:
    """Collect every captured value from every configured pattern.
    
    Address extraction needs more than the first regex hit because company contact
    information may appear before the project address. This helper preserves pattern
    and page-text order so the caller can reject bad candidates and accept the first
    plausible one."""
    search_text = (
        normalize_ocr_numbers(text) if normalize_numbers else str(text or "")
    )
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
    """Make extracted text safe for use inside a Windows filename or folder name.
    
    The function normalizes spacing, removes reserved path characters and control
    codes, removes trailing periods/spaces forbidden by Windows, and limits length
    to avoid unwieldy paths. The fallback ensures callers never receive an empty
    path component."""
    value = normalize_value(value) or fallback
    value = INVALID_PATH_RE.sub("", value)
    value = value.strip(" .")
    return value[:140] or fallback


def unique_path(path: Path) -> Path:
    """Select a destination path without overwriting an existing file.
    
    The requested path is returned unchanged when free. Otherwise the function
    tries ``name (2).pdf``, ``name (3).pdf``, and so on. Filing code calls this
    before the final move so a previous project record is never silently replaced."""
    if not path.exists():
        return path
    for counter in range(2, 10000):
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique path for {path}")


def normalize_for_fuzzy(value: str) -> str:
    """Create the comparison form used by OCR-tolerant document classification.
    
    Text is lowercased and common visual confusions are translated so, for example,
    a zero in ``site plan`` is less damaging to similarity. This representation is
    for matching only; the original OCR text remains available for display and
    metadata extraction."""
    return str(value or "").lower().translate(OCR_CONFUSION_MAP)


def keyword_groups(raw_keywords: Any) -> dict[str, list[str]]:
    """Convert supported configuration styles into ``label -> list of phrases``.
    
    Modern configuration maps each document type to several OCR variants. Older
    files may contain only a list of labels. Normalizing both shapes here lets the
    classifier use one loop and keeps backward compatibility with earlier projects."""
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


def best_keyword_window(
    keyword: str, normalized_text: str
) -> tuple[float, int, int]:
    """Find the OCR substring most similar to one document-type keyword.
    
    The function slides windows near the keyword's length across normalized OCR
    text and scores each with ``SequenceMatcher``. Cheap upper-bound checks skip
    windows that cannot beat the current best, which provides the Version 3 speedup.
    
    Window lengths remain ascending and the score comparison remains strictly ``>``.
    Those details preserve the legacy first-winner behavior when two locations tie,
    which matters because the winning start position influences the later lot search."""
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
        for start in range(len(normalized_text) - window_length + 1):
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
    """Apply high-confidence document-type rules before fuzzy matching.
    
    Some phrases must outrank broader keywords—for example, an easement plat may
    contain words that resemble another plan type. The function checks configured
    rules in insertion order and returns a perfect-score ``FuzzyMatch`` at the first
    rule hit. Returning the match location keeps the downstream interface identical
    to the fuzzy classifier."""
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
    """Classify a document from configured title phrases while avoiding unnecessary work.
    
    Classification occurs in two stages. First, every normalized phrase is searched
    as an exact substring; most clear title blocks finish here in milliseconds. Only
    when no exact phrase exists does the function call ``best_keyword_window`` for
    OCR-tolerant similarity scoring.
    
    The highest score above the threshold wins. Configuration order and strict score
    comparisons preserve previous tie behavior, so the optimization changes speed
    without intentionally changing classifications."""
    # Build the comparison text once. Repeating lowercase/translation inside
    # every keyword loop was unnecessary work in earlier versions.
    normalized_text = normalize_for_fuzzy(text)
    groups = keyword_groups(keywords)

    # FAST PATH - Most title blocks contain at least one configured phrase
    # exactly after OCR normalization. Returning here avoids thousands of
    # SequenceMatcher window comparisons on ordinary documents.
    # Fast path: preserve the legacy winner exactly when a configured keyword
    # occurs verbatim after OCR normalization. The legacy loop uses a strict
    # ``>`` comparison, so the first exact match (score 1.0) can never be
    # displaced by a later candidate. Returning it immediately avoids running
    # thousands of SequenceMatcher comparisons for every nonmatching keyword.
    for label, candidates in groups.items():
        for keyword in candidates:
            normalized_keyword = normalize_for_fuzzy(keyword)
            if not normalized_keyword:
                continue
            start = normalized_text.find(normalized_keyword)
            if start >= 0:
                end = start + len(normalized_keyword)
                return FuzzyMatch(
                    label=label,
                    score=1.0,
                    start=start,
                    end=end,
                    matched_text=text[start:end],
                    keyword=keyword,
                )

    # Compatibility fallback: when OCR contains no exact configured phrase,
    # retain the existing exhaustive fuzzy algorithm and tie behavior. This
    # keeps typo-tolerance unchanged while making ordinary scans much faster.
    best: FuzzyMatch | None = None
    for label, candidates in groups.items():
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
                if best.score >= 1.0:
                    return best
    return best if best and best.score >= threshold else None


def is_ignored_address(address: str, config: Config) -> bool:
    """Reject a regex address candidate that is probably not the project property.
    
    Engineering sheets often print the surveyor's office address, phone number, URL,
    or email in the title block. The function compares normalized candidates against
    explicitly ignored addresses and configured warning keywords so those contacts do
    not become the destination folder name."""
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


def _ocr_item_rect(
    item: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    """Convert one OCR token's possible geometry formats into a simple rectangle.
    
    Paddle versions may provide ``bbox``, ``polygon``, or other point arrays. This
    helper accepts the available form and returns ``left, top, right, bottom`` values.
    Layout-aware address reconstruction uses the rectangle to group tokens into lines."""
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
    """Reconstruct readable text lines from OCR tokens near page bottoms.
    
    Most survey title blocks are in the lower part of a page. For each OCR page, the
    function filters tokens by vertical position, estimates a typical token height,
    groups nearby tokens into the same physical line, then sorts each line left-to-right.
    
    The resulting strings preserve visual layout better than Paddle's global text order.
    ``first_valid_address`` searches these lines before falling back to plain OCR text."""
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
                text
                for _, text in sorted(
                    row["tokens"], key=lambda token: token[0]
                )
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
    """Return the first plausible project address found in OCR output.
    
    Coordinate-aware title-block lines are searched first because they are less likely
    to confuse the engineering company's contact address with the property address.
    Every regex candidate is passed through ``is_ignored_address``. If geometry is
    unavailable, the function repeats the same validation against plain OCR text."""
    if ocr_pages:
        try:
            bottom_fraction = float(
                config.get("bbox_address_bottom_fraction", 0.65)
            )
        except (TypeError, ValueError):
            bottom_fraction = 0.65
        try:
            line_tolerance = float(
                config.get("bbox_address_line_tolerance", 0.75)
            )
        except (TypeError, ValueError):
            line_tolerance = 0.75

        for line in _layout_address_lines(
            ocr_pages,
            bottom_fraction=bottom_fraction,
            line_tolerance=line_tolerance,
        ):
            for address in all_matches(
                line, config.get("address_patterns", [])
            ):
                if address and not is_ignored_address(address, config):
                    return address

    # FALLBACK - Old saved OCR results or unusual Paddle versions may not
    # contain geometry. Plain-text scanning is less precise, but retaining it
    # prevents layout improvements from breaking those documents.
    # Compatibility fallback for PDFs/results without usable bounding boxes.
    for address in all_matches(text, config.get("address_patterns", [])):
        if address and not is_ignored_address(address, config):
            return address
    return None


def _points_from_any(value: Any) -> list[list[float]]:
    """Safely convert a loose collection of OCR points into numeric ``[x, y]`` pairs.
    
    Invalid items are skipped rather than failing the entire document. This defensive
    conversion is needed because OCR libraries may return lists, tuples, NumPy values,
    or partially missing geometry depending on version and page content."""
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
    """Compute the smallest axis-aligned rectangle enclosing polygon points.
    
    The minimum and maximum x/y values make later layout calculations simple. An empty
    point list returns a zero rectangle, which callers can treat as unusable geometry."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def first_nonempty_value(*values: Any) -> Any:
    """Choose the first supplied OCR field that contains usable data.
    
    Different PaddleOCR versions store equivalent geometry under different keys. This
    helper lets compatibility code try those keys in preferred order without a chain of
    repeated ``if`` statements."""
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def is_known_value(value: str) -> bool:
    """Test whether a value is real metadata rather than a UI placeholder.
    
    Empty strings, ``Unknown ...`` labels, ``Project``, and ``Document`` must not win
    batch votes or overwrite an existing known value. This shared definition keeps
    voting, merging, and SDAT application consistent."""
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
    *,
    performance_callback: Any | None = None,
    profile_label: str = "document",
) -> ExtractedMetadata:
    """Run the complete classification and metadata extraction process for one PDF.
    
    App role:
        This is the main public function of the module. OCR services supply text and
        page geometry; the batch pipeline receives one ``ExtractedMetadata`` result.
    
    Processing order:
        1. Recognize SDAT lookup printouts and extract only their Tax ID.
        2. Try high-priority regex document rules, then the fast exact/fuzzy classifier.
        3. Search for lot after the matched title and search the full text for map,
           parcel, Tax ID, address, and project code.
        4. Sanitize values needed by Windows paths and build the immutable result.
        5. Emit per-stage timings when profiling is enabled.
    
    The function intentionally does not call the SDAT network API. Network enrichment
    occurs later, after Batch mode has combined evidence from all related drawings."""
    from sdat import (
        LOOKUP_DOCUMENT_TYPE,
        extract_sdat_lookup_tax_id,
        is_sdat_lookup_document,
    )

    emit = performance_callback or (lambda _message: None)
    profile_started = time.perf_counter()
    stage_rows: list[tuple[str, float]] = []

    def stage(label: str, started: float) -> float:
        """Record one named sub-stage in the per-document performance report.
        
        The nested helper measures from the caller's supplied start time, stores the row for
        later ranking, and emits a line immediately so a slow or stalled stage is visible in
        the terminal and progress panel."""
        elapsed = time.perf_counter() - started
        stage_rows.append((label, elapsed))
        emit(f"[META-PERF] {profile_label}.{label}: {elapsed:.4f}s")
        return elapsed

    started = time.perf_counter()
    # STAGE 1 - Lookup PDFs follow a deliberately short path. They contribute
    # a Tax ID to the packet but should not be classified or filed as plans.
    lookup_document = is_sdat_lookup_document(text)
    stage("lookup_document_detection", started)
    if lookup_document:
        started = time.perf_counter()
        lookup = extract_sdat_lookup_tax_id(text)
        stage("lookup_tax_id_extraction", started)
        tax_id = lookup[2] if lookup else ""
        result = ExtractedMetadata(
            lot="Unknown Lot",
            address="Unknown Address",
            project_code=safe_path_part(default_project_code, "Project"),
            document_type=LOOKUP_DOCUMENT_TYPE,
            tax_id=tax_id,
        )
        total = time.perf_counter() - profile_started
        emit(f"[META-PERF] {profile_label}.total: {total:.4f}s | chars={len(text)}; type={result.document_type}")
        return result

    started = time.perf_counter()
    # STAGE 2 - Rules for unambiguous phrases run before fuzzy matching. This
    # both improves precedence and lets clear documents skip expensive scoring.
    doc_match = regex_document_type(text, config.get("document_type_regex_rules"))
    stage("document_type_regex", started)

    if doc_match is None:
        started = time.perf_counter()
        doc_match = fuzzy_document_type(text, config.get("document_type_keywords"))
        stage("document_type_fuzzy", started)
    else:
        stage_rows.append(("document_type_fuzzy", 0.0))
        emit(f"[META-PERF] {profile_label}.document_type_fuzzy: 0.0000s | skipped=regex_match")

    started = time.perf_counter()
    document_type = (
        doc_match.label
        if doc_match
        else first_match(text, config.get("document_type_patterns", []))
        or default_document_type
        or "Field Notes"
    )
    stage("document_type_fallback", started)

    started = time.perf_counter()
    # Search for the lot after the detected title when possible. This reduces
    # the chance that a revision note or unrelated header number is chosen.
    lot_search_text = text[doc_match.start :] if doc_match else text
    lot = first_match(lot_search_text, config.get("lot_pattern", [])) or "Unknown Lot"
    stage("lot_extraction", started)

    started = time.perf_counter()
    tax_map = first_match(text, config.get("map_patterns", []), normalize_numbers=False) or ""
    stage("tax_map_extraction", started)

    started = time.perf_counter()
    parcel = first_match(text, config.get("parcel_patterns", []), normalize_numbers=True) or ""
    stage("parcel_extraction", started)

    started = time.perf_counter()
    tax_id = first_match(text, config.get("tax_id_patterns", []), normalize_numbers=True) or ""
    stage("tax_id_extraction", started)

    started = time.perf_counter()
    address = first_valid_address(text, config, ocr_pages) or "Unknown Address"
    stage("address_extraction", started)

    started = time.perf_counter()
    project_code = first_match(text, config.get("project_code_patterns", [])) or default_project_code
    stage("project_code_extraction", started)

    started = time.perf_counter()
    result = ExtractedMetadata(
        lot=safe_path_part(lot, "Unknown Lot"),
        address=safe_path_part(address, "Unknown Address"),
        project_code=safe_path_part(project_code, "Project"),
        document_type=normalize_value(document_type) or "Field Notes",
        tax_map=safe_path_part(tax_map, "") if tax_map else "",
        parcel=safe_path_part(parcel, "") if parcel else "",
        tax_id=safe_path_part(tax_id, "") if tax_id else "",
    )
    stage("result_sanitization", started)

    total = time.perf_counter() - profile_started
    emit(f"[META-PERF] {profile_label}.total: {total:.4f}s | chars={len(text)}; type={result.document_type}")
    ranked = sorted(stage_rows, key=lambda row: row[1], reverse=True)
    emit(f"=== METADATA EXTRACTION HOTSPOTS {profile_label} START ===")
    for rank, (label, seconds) in enumerate(ranked, start=1):
        percent = (seconds / total * 100.0) if total else 0.0
        emit(f"rank={rank}; stage={label}; seconds={seconds:.4f}; percent={percent:.1f}")
    emit(f"=== METADATA EXTRACTION HOTSPOTS {profile_label} END ===")
    return result


def prefer_known(value: str, fallback: str) -> str:
    """Use a proposed metadata value only when it is genuinely known.
    
    Batch merging calls this for every shared property field. A blank or placeholder
    shared value must not erase a valid value extracted from the individual drawing."""
    return value if is_known_value(value) else fallback


def normalize_identifier(value: Any) -> str:
    """Reduce an identifier to a comparison key independent of formatting and zero padding.
    
    Punctuation is removed, letters are uppercased, and leading zeros are ignored. This
    allows values such as ``0012`` and ``12`` to compare equal when SDAT stores a fixed
    width but the plan prints a shorter form."""
    cleaned = re.sub(r"[^0-9A-Za-z]", "", str(value or "")).upper()
    return cleaned.lstrip("0") or cleaned


def identifier_options(
    value: str, widths: Iterable[int] = (2, 3, 4, 6, 8)
) -> list[str]:
    """Generate the padded and unpadded forms that SDAT may use for one identifier.
    
    Socrata queries compare text exactly. For a numeric map, parcel, district, or account
    value, this helper returns the original compact form, the form without leading zeros,
    and allowed zero-padded widths. ``sdat.or_equals`` turns those options into one query."""
    compact = re.sub(r"[^0-9A-Za-z]", "", str(value or "")).upper()
    if not compact:
        return []
    options = {compact, compact.lstrip("0") or "0"}
    if compact.isdigit():
        options.update(
            compact.zfill(width) for width in widths if len(compact) <= width
        )
    return sorted(options)
