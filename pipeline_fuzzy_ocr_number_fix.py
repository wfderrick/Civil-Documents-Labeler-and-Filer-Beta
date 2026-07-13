from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz
import requests

try:
    import paddle  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional runtime check
    paddle = None

from paddleocr import PaddleOCR  # pyright: ignore[reportMissingImports]

from visual_classifier import FIELD_NOTES_LABEL, classify_pdf_visual

DEFAULT_TITLE_BLOCK_CROP = (0.55, 0.65, 1.0, 1.0)
DOCUMENT_TYPE_THRESHOLD = 0.75
SDAT_API_URL = "https://opendata.maryland.gov/resource/ed4q-f8tm.json"

SDAT_FIELDS = {
    "county": "county_name_mdp_field_cntyname",
    "account_id": "account_id_mdp_field_acctid",
    "district": "record_key_district_ward_sdat_field_2",
    "account_number": "record_key_account_number_sdat_field_3",
    "lot": "lot_mdp_field_lot_sdat_field_41",
    "map": "map_mdp_field_map_sdat_field_42",
    "parcel": "parcel_mdp_field_parcel_sdat_field_44",
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

DEFAULT_CONFIG: dict[str, Any] = {
    "sdat_lookup": True,
    "ocr_device": "auto",
    "gpu_device_id": 0,
    "parallel_ocr": False,
    "ocr_workers": 1,
    "ocr_threads_per_worker": 4,
    "visual_field_notes_classifier": True,
    "visual_field_notes_threshold": 0.70,
    "default_county": "",
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
        r"\btax\s*(?:id|i\.?d\.?)\s*[:#.-]?\s*([0-9]{1,2})\s*[- ]\s*([0-9]{4,8})\b",
        r"\b([0-9]{1,2})\s*[- ]\s*([0-9]{6,8})\b",
    ],
    "district_patterns": [r"\bdistrict\s*[:#-]?\s*([0-9A-Za-z]+)\b", r"\bdist\.?\s*[:#-]?\s*([0-9A-Za-z]+)\b"],
    "account_patterns": [
        r"\baccount\s*(?:number|no\.?|#)?\s*[:#-]?\s*([0-9A-Za-z]+)\b",
        r"\bacct\.?\s*(?:no\.?|#)?\s*[:#-]?\s*([0-9A-Za-z]+)\b",
    ],
    "address_patterns": [
        r"\s(?:property|site|project)\s+address\s*[:#-]?\s*(.+)",
        r"\saddress\s*[:#-]?\s*(.+)",
        r"\s(\d{1,6}\s+[A-Za-z0-9 .'-]+\s+(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|court|ct\.?|circle|cir\.?|way|place|pl\.?)\b[^\n]*)",
    ],
    "ignored_address_keywords": ["phone", "fax", "www", ".com", "@", "survey", "surveyor", "surveying", "engineer", "engineering"],
    "ignored_addresses": [],
    "project_code_patterns": [r"\s(aa|cc|ch|nav|pg|sm|usaf[0-9]{4})\s"],
    "document_type_keywords": {
        "House Location": ["house location", "houselocation", "house loc", "hse location", "location drawing"],
        "Site Plan": ["site plan", "siteplan", "plot plan", "sitemap"],
        "Wall Check": ["wall check", "wallcheck", "wall chk", "foundation check"],
        "Field Notes": ["field notes", "fieldnotes", "field note", "notes"],
        "Replat": ["replat", "re plat"],
    },
    "document_type_patterns": [r"\s(wall check|site plan|field notes|replat|house location)\s"],
}

OCR_CONFUSION_MAP = str.maketrans({"0": "o", "1": "l", "I": "l", "|": "l", "!": "l", "5": "s", "$": "s", "3": "e", "@": "a", "8": "b", "6": "g", "2": "z", "+": "t"})

# Used for metadata identifiers like Tax ID, tax map, parcel, district, and account number.
# This intentionally maps common letter-like OCR mistakes back to digits.
OCR_NUMBER_MAP = str.maketrans({
    "O": "0",
    "o": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "|": "1",
    "S": "5",
    "s": "5",
    "B": "8",
})


def normalize_ocr_numbers(text: str) -> str:
    """Normalize OCR mistakes that commonly appear inside numeric identifiers."""
    return str(text or "").translate(OCR_NUMBER_MAP)


@dataclass(frozen=True)
class ExtractedMetadata:
    lot: str
    address: str
    project_code: str
    document_type: str
    tax_map: str = ""
    parcel: str = ""
    tax_id: str = ""
    section: str = ""


@dataclass(frozen=True)
class FuzzyMatch:
    label: str
    score: float
    start: int
    end: int
    matched_text: str
    keyword: str


@dataclass(frozen=True)
class SdatSearchTerms:
    county: str = ""
    lot: str = ""
    tax_map: str = ""
    parcel: str = ""
    tax_id: str = ""
    district: str = ""
    account_number: str = ""


Config = dict[str, Any]


def load_config(path: Path | None) -> Config:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with path.open("r", encoding="utf-8") as config_file:
        user_config = json.load(config_file)
    config.update({key: value for key, value in user_config.items() if value not in (None, "", [])})
    return config


def normalize_value(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n:-#")
    return value.rstrip(".,;")


def first_match(text: str, patterns: Iterable[str], *, normalize_numbers: bool = False) -> str | None:
    """Return the first regex capture, optionally fixing OCR digit/letter mistakes first."""
    search_text = normalize_ocr_numbers(text) if normalize_numbers else str(text or "")

    for pattern in patterns:
        match = re.search(pattern, search_text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue

        if match.lastindex and match.lastindex > 1:
            return "-".join(normalize_value(match.group(i)) for i in range(1, match.lastindex + 1))

        return normalize_value(match.group(1))

    return None


def all_matches(text: str, patterns: Iterable[str], *, normalize_numbers: bool = False) -> list[str]:
    """Return all regex captures, optionally fixing OCR digit/letter mistakes first."""
    search_text = normalize_ocr_numbers(text) if normalize_numbers else str(text or "")
    values: list[str] = []

    for pattern in patterns:
        for match in re.finditer(pattern, search_text, flags=re.IGNORECASE | re.MULTILINE):
            if match.lastindex and match.lastindex > 1:
                values.append("-".join(normalize_value(match.group(i)) for i in range(1, match.lastindex + 1)))
            else:
                values.append(normalize_value(match.group(1)))

    return values


def safe_path_part(value: str, fallback: str) -> str:
    value = normalize_value(value) or fallback
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:140] or fallback


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for counter in range(2, 10000):
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique path for {path}")


def normalize_for_fuzzy(value: str) -> str:
    return str(value or "").lower().translate(OCR_CONFUSION_MAP)


def fuzzy_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def keyword_groups(raw_keywords: Any) -> dict[str, list[str]]:
    if isinstance(raw_keywords, Mapping):
        return {str(label): ([str(item) for item in keywords] if isinstance(keywords, list) else [str(keywords)]) for label, keywords in raw_keywords.items()}
    if isinstance(raw_keywords, list):
        return {str(label): [str(label)] for label in raw_keywords}
    return {}


def best_keyword_window(keyword: str, normalized_text: str) -> tuple[float, int, int]:
    if not keyword or not normalized_text:
        return 0.0, -1, -1
    exact_start = normalized_text.find(keyword)
    if exact_start >= 0:
        return 1.0, exact_start, exact_start + len(keyword)

    keyword_length = len(keyword)
    min_window = max(3, keyword_length - 3)
    max_window = min(len(normalized_text), keyword_length + 4)
    best_score, best_start, best_end = 0.0, -1, -1
    for window_length in range(min_window, max_window + 1):
        for start in range(0, len(normalized_text) - window_length + 1):
            end = start + window_length
            score = fuzzy_ratio(keyword, normalized_text[start:end])
            if score > best_score:
                best_score, best_start, best_end = score, start, end
                if best_score >= 0.98:
                    return best_score, best_start, best_end
    return best_score, best_start, best_end


def fuzzy_document_type(text: str, keywords: Any, threshold: float = DOCUMENT_TYPE_THRESHOLD) -> FuzzyMatch | None:
    normalized_text = normalize_for_fuzzy(text)
    best: FuzzyMatch | None = None
    for label, candidates in keyword_groups(keywords).items():
        for keyword in candidates:
            score, start, end = best_keyword_window(normalize_for_fuzzy(keyword), normalized_text)
            if start < 0:
                continue
            match = FuzzyMatch(label=label, score=score, start=start, end=end, matched_text=text[start:end], keyword=keyword)
            if best is None or match.score > best.score:
                best = match
    return best if best and best.score >= threshold else None


def is_ignored_address(address: str, config: Config) -> bool:
    cleaned = normalize_for_fuzzy(address)
    compact = re.sub(r"[^a-z0-9]", "", cleaned)
    for blocked in config.get("ignored_addresses", []):
        blocked_clean = normalize_for_fuzzy(str(blocked))
        if blocked_clean and (blocked_clean in cleaned or re.sub(r"[^a-z0-9]", "", blocked_clean) in compact):
            return True
    return any(normalize_for_fuzzy(str(keyword)) in cleaned for keyword in config.get("ignored_address_keywords", []) if normalize_for_fuzzy(str(keyword)))


def first_valid_address(text: str, config: Config) -> str | None:
    for address in all_matches(text, config.get("address_patterns", [])):
        if address and not is_ignored_address(address, config):
            return address
    return None


def render_pdf_pages(pdf_path: Path, image_dir: Path, dpi: int) -> list[Path]:
    image_paths: list[Path] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):  # pyright: ignore[reportArgumentType]
            image_path = image_dir / f"page-{page_index + 1:04d}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(image_path)
            image_paths.append(image_path)
    return image_paths


def render_pdf_page_crop(pdf_path: Path, image_dir: Path, dpi: int, page_index: int = 0, crop_box: tuple[float, float, float, float] = DEFAULT_TITLE_BLOCK_CROP) -> Path:
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as document:
        page = document[page_index]
        rect = page.rect
        clip = fitz.Rect(rect.width * crop_box[0], rect.height * crop_box[1], rect.width * crop_box[2], rect.height * crop_box[3])
        image_path = image_dir / "title-block-crop.png"
        page.get_pixmap(matrix=matrix, alpha=False, clip=clip).save(image_path)
        return image_path



def _as_float_pair(value: Any) -> list[float] | None:
    """Convert a PaddleOCR point-like value to [x, y]."""
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    except Exception:
        return None
    return None


def _points_from_any(value: Any) -> list[list[float]]:
    """Extract polygon points from PaddleOCR box/polygon formats."""
    if value is None:
        return []

    # Rect format: [x0, y0, x1, y1]
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        x0, y0, x1, y1 = [float(v) for v in value]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    if isinstance(value, (list, tuple)):
        points: list[list[float]] = []
        for item in value:
            point = _as_float_pair(item)
            if point:
                points.append(point)
        return points

    return []


def _bbox_from_points(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def first_nonempty_value(*values):
    for value in values:
        if value is None:
            continue
        try:
            if hasattr(value, "size") and value.size == 0:
                continue
            if len(value) == 0:
                continue
        except TypeError:
            pass
        return value
    return None


def extract_ocr_items(ocr_result: Any) -> list[dict[str, Any]]:
    """Return PaddleOCR text, confidence, and bounding boxes in image pixels.

    Supports both PaddleOCR 3.x dictionary results and older list-style results.
    """
    items: list[dict[str, Any]] = []
    
    for page_result in ocr_result or []:
        if isinstance(page_result, dict):
            texts = page_result.get("rec_texts") or []
            scores = page_result.get("rec_scores") or []
            boxes = first_nonempty_value(
            page_result.get("rec_polys"),
            page_result.get("rec_boxes"),
            page_result.get("dt_polys"),
            page_result.get("boxes"),
        )

            for i, text in enumerate(texts):
                item = {"text": str(text)}
                item: dict[str, Any] = {
                "text": str(text),
                }
                if i < len(scores):
                    item["score"] = float(scores[i])

                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    if hasattr(box, "tolist"):
                        box = box.tolist()
                    item["bbox"] = box

                items.append(item)

        elif isinstance(page_result, list):
            for raw_item in page_result:
                try:
                    points = _points_from_any(raw_item[0])
                    text = str(raw_item[1][0]).strip()
                    confidence = float(raw_item[1][1])
                except Exception:
                    continue
                if text:
                    items.append({
                        "text": text,
                        "confidence": confidence,
                        "polygon": points,
                        "bbox": _bbox_from_points(points),
                    })

    return items


def extract_line_text(ocr_result: Any) -> list[str]:
    return [item["text"] for item in extract_ocr_items(ocr_result) if item.get("text")]


def ocr_images(image_paths: Iterable[Path], ocr: PaddleOCR) -> str:
    lines: list[str] = []
    for image_path in image_paths:
        lines.extend(extract_line_text(ocr.predict(str(image_path))))
    return "\n".join(lines)


def render_pdf_pages_with_info(pdf_path: Path, image_dir: Path, dpi: int) -> list[dict[str, Any]]:
    """Render every page and keep enough geometry to map OCR pixels back to PDF points."""
    pages: list[dict[str, Any]] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):  # pyright: ignore[reportArgumentType]
            image_path = image_dir / f"page-{page_index + 1:04d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(image_path)
            pages.append({
                "page_index": page_index,
                "image_path": image_path,
                "image_width": pixmap.width,
                "image_height": pixmap.height,
                "page_width": float(page.rect.width),
                "page_height": float(page.rect.height),
                "dpi": dpi,
            })
    return pages


def ocr_pdf_with_layout(pdf_path: Path, ocr: PaddleOCR, dpi: int) -> dict[str, Any]:
    """OCR a PDF and preserve page-level bounding boxes for searchable text layers."""
    lines: list[str] = []
    ocr_pages: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="paddleocr_pdf_") as temp_dir:
        rendered_pages = render_pdf_pages_with_info(pdf_path, Path(temp_dir), dpi)
        for page_info in rendered_pages:
            result = ocr.predict(str(page_info["image_path"]))
            items = extract_ocr_items(result)
            lines.extend(item["text"] for item in items if item.get("text"))
            ocr_pages.append({
                "page_index": page_info["page_index"],
                "image_width": page_info["image_width"],
                "image_height": page_info["image_height"],
                "page_width": page_info["page_width"],
                "page_height": page_info["page_height"],
                "dpi": page_info["dpi"],
                "items": items,
            })

    return {"text": "\n".join(lines), "pages": ocr_pages}


def ocr_pdf(pdf_path: Path, ocr: PaddleOCR, dpi: int) -> str:
    return ocr_pdf_with_layout(pdf_path, ocr, dpi)["text"]


def ocr_pdf_title_block(pdf_path: Path, ocr: PaddleOCR, dpi: int) -> str:
    with tempfile.TemporaryDirectory(prefix="paddleocr_title_block_") as temp_dir:
        return ocr_images([render_pdf_page_crop(pdf_path, Path(temp_dir), dpi)], ocr)


_WORKER_OCR: PaddleOCR | None = None


def gpu_is_available() -> bool:
    """Return True when the installed Paddle package can see a CUDA GPU."""
    if paddle is None:
        return False
    try:
        return bool(paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
    except Exception:
        return False


def resolve_ocr_device(ocr_device: str = "auto", gpu_device_id: int = 0) -> str:
    """Resolve auto/gpu/cpu into the device string PaddleOCR should use."""
    requested = str(ocr_device or "auto").lower().strip()
    if requested == "gpu":
        return f"gpu:{int(gpu_device_id or 0)}"
    if requested == "cpu":
        return "cpu"
    return f"gpu:{int(gpu_device_id or 0)}" if gpu_is_available() else "cpu"


def make_ocr(
    lang: str = "en",
    cpu_threads: int | None = None,
    ocr_device: str = "auto",
    gpu_device_id: int = 0,
) -> PaddleOCR:
    """Create one PaddleOCR engine optimized for either GPU or CPU.

    GPU mode should use a single OCR engine. CPU mode may use worker processes.
    The PaddleOCR API has changed between versions, so this tries the newer
    `device=` argument first, then falls back to older constructor styles.
    """
    resolved_device = resolve_ocr_device(ocr_device, gpu_device_id)
    base_kwargs: dict[str, Any] = {"lang": lang}

    if resolved_device == "cpu" and cpu_threads:
        base_kwargs["cpu_threads"] = int(cpu_threads)

    attempts: list[dict[str, Any]] = []

    # PaddleOCR 3.x
    attempts.append({**base_kwargs, "device": resolved_device})

    # Older PaddleOCR versions sometimes used use_gpu instead of device.
    if resolved_device.startswith("gpu"):
        attempts.append({**base_kwargs, "use_gpu": True, "gpu_id": int(gpu_device_id or 0)})
    else:
        attempts.append({**base_kwargs, "use_gpu": False})

    # Last-resort default constructor.
    attempts.append(base_kwargs)

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            if resolved_device == "cpu":
                try:
                    return PaddleOCR(**kwargs, enable_mkldnn=False)  # type: ignore[arg-type]
                except TypeError:
                    return PaddleOCR(**kwargs)
            return PaddleOCR(**kwargs)
        except TypeError as error:
            last_error = error
            continue

    if last_error:
        raise last_error
    return PaddleOCR(**base_kwargs)


def _init_ocr_worker(lang: str, cpu_threads: int, ocr_device: str, gpu_device_id: int) -> None:
    global _WORKER_OCR
    _WORKER_OCR = make_ocr(
        lang=lang,
        cpu_threads=cpu_threads,
        ocr_device=ocr_device,
        gpu_device_id=gpu_device_id,
    )


def _ocr_one_pdf_worker(index: int, pdf_path_text: str, dpi: int) -> tuple[int, dict[str, Any]]:
    if _WORKER_OCR is None:
        raise RuntimeError("OCR worker was not initialized.")
    pdf_path = Path(pdf_path_text)
    title_text = ocr_pdf_title_block(pdf_path, _WORKER_OCR, dpi)
    full_ocr = ocr_pdf_with_layout(pdf_path, _WORKER_OCR, dpi)
    full_text = full_ocr["text"]
    return index, {
        "source_path": str(pdf_path),
        "source_name": pdf_path.name,
        "ocr_text": f"{pdf_path.stem}\n{title_text}\n{full_text}",
        "ocr_pages": full_ocr["pages"],
    }


def ocr_pdf_batch(
    pdf_paths: list[Path],
    *,
    dpi: int,
    lang: str = "en",
    workers: int = 1,
    threads_per_worker: int = 4,
    existing_ocr: PaddleOCR | None = None,
    ocr_device: str = "auto",
    gpu_device_id: int = 0,
) -> list[dict[str, Any]]:
    """OCR a batch of PDFs, optionally in parallel.

    Use process-level parallelism because PaddleOCR engines should not be shared
    across processes. Results are returned in the same order as pdf_paths.
    """
    if not pdf_paths:
        return []

    resolved_device = resolve_ocr_device(ocr_device, gpu_device_id)

    # One GPU should use one OCR engine. Multiple GPU worker processes usually
    # fight over the same VRAM and are slower/less stable than one GPU engine.
    if resolved_device.startswith("gpu"):
        workers = 1
    else:
        workers = max(1, int(workers or 1))

    if workers == 1:
        ocr = existing_ocr or make_ocr(
            lang=lang,
            cpu_threads=threads_per_worker,
            ocr_device=resolved_device,
            gpu_device_id=gpu_device_id,
        )
        results: list[dict[str, Any]] = []
        for pdf_path in pdf_paths:
            title_text = ocr_pdf_title_block(pdf_path, ocr, dpi)
            full_ocr = ocr_pdf_with_layout(pdf_path, ocr, dpi)
            full_text = full_ocr["text"]
            results.append({
                "source_path": str(pdf_path),
                "source_name": pdf_path.name,
                "ocr_text": f"{pdf_path.stem}\n{title_text}\n{full_text}",
                "ocr_pages": full_ocr["pages"],
            })
        return results

    indexed_results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_ocr_worker,
        initargs=(lang, int(threads_per_worker or 4), resolved_device, int(gpu_device_id or 0)),
    ) as executor:
        futures = {
            executor.submit(_ocr_one_pdf_worker, index, str(pdf_path), dpi): index
            for index, pdf_path in enumerate(pdf_paths)
        }
        for future in as_completed(futures):
            index, result = future.result()
            indexed_results[index] = result

    return [indexed_results[index] for index in range(len(pdf_paths))]


def extract_tax_id_parts(tax_id: str) -> tuple[str, str]:
    """Split a Tax ID like 02-078384 into SDAT district/account parts."""
    numbers = re.findall(r"\d+", normalize_ocr_numbers(tax_id))
    if len(numbers) >= 2:
        return numbers[0].zfill(2), numbers[1].zfill(6)
    return "", ""


def extract_metadata(text: str, config: Config, default_project_code: str, default_document_type: str) -> ExtractedMetadata:
    doc_match = fuzzy_document_type(text, config.get("document_type_keywords"))
    document_type = doc_match.label if doc_match else first_match(text, config.get("document_type_patterns", [])) or default_document_type

    # Use ORIGINAL text for lot search so "Lot" is not changed to "10t"
    lot_search_text = text[doc_match.start:] if doc_match else text
    lot = first_match(lot_search_text, config.get("lot_pattern", [])) or "Unknown Lot"

    # Use OCR-number normalization for numeric identifiers only.
    tax_map = first_match(text, config.get("map_patterns", []), normalize_numbers=False) or ""
    parcel = first_match(text, config.get("parcel_patterns", []), normalize_numbers=True) or ""
    tax_id = first_match(text, config.get("tax_id_patterns", []), normalize_numbers=True) or ""

    return ExtractedMetadata(
        lot=safe_path_part(lot, "Unknown Lot"),
        address=safe_path_part(first_valid_address(text, config) or "Unknown Address", "Unknown Address"),
        project_code=safe_path_part(first_match(text, config.get("project_code_patterns", [])) or default_project_code, "Project"),
        document_type=safe_path_part(document_type, "Document"),
        tax_map=safe_path_part(tax_map, "") if tax_map else "",
        parcel=safe_path_part(parcel, "") if parcel else "",
        tax_id=safe_path_part(tax_id, "") if tax_id else "",
    )

def extract_project_code_from_output_folder(output_folder: Path | str, config: Config, fallback: str = "Project") -> str:
    path = Path(output_folder)
    patterns = config.get("project_code_patterns", [])
    for part in reversed(path.parts):
        for pattern in patterns:
            match = re.search(pattern, f" {part} ", flags=re.IGNORECASE)
            if match:
                return safe_path_part(match.group(1).upper(), fallback)
    return safe_path_part(path.name or fallback, fallback)


def prefer_known(value: str, fallback: str) -> str:
    return value if is_known_value(value) else fallback


def normalize_identifier(value: Any) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]", "", str(value or "")).upper()
    return cleaned.lstrip("0") or cleaned


def identifier_options(value: str, widths: Iterable[int] = (2, 3, 4, 6, 8)) -> list[str]:
    compact = re.sub(r"[^0-9A-Za-z]", "", str(value or "")).upper()
    if not compact:
        return []
    options = {compact, compact.lstrip("0") or "0"}
    if compact.isdigit():
        options.update(compact.zfill(width) for width in widths if len(compact) <= width)
    return sorted(options)


def soql_escape(value: str) -> str:
    return str(value or "").replace("'", "''").strip()


def or_equals(field: str, value: str, widths: Iterable[int] = (2, 3, 4, 6, 8)) -> str:
    options = identifier_options(value, widths)
    return "(" + " OR ".join(f"{field} = '{soql_escape(option)}'" for option in options) + ")"


def extract_sdat_search_terms(text: str, metadata: ExtractedMetadata, config: Config) -> SdatSearchTerms:
    county = first_match(text, config.get("county_patterns", [])) or config.get("default_county", "")
    county = re.sub(r"\bcounty\b", "", str(county), flags=re.IGNORECASE).strip()
    tax_map = metadata.tax_map or first_match(text, config.get("map_patterns", []), normalize_numbers=True) or ""
    parcel = metadata.parcel or first_match(text, config.get("parcel_patterns", []), normalize_numbers=True) or ""
    tax_id = metadata.tax_id or first_match(text, config.get("tax_id_patterns", []), normalize_numbers=True) or ""
    district, account_number = extract_tax_id_parts(tax_id)
    district = district or first_match(text, config.get("district_patterns", []), normalize_numbers=True) or ""
    account_number = account_number or first_match(text, config.get("account_patterns", []), normalize_numbers=True) or ""
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
        SDAT_FIELDS["county"], SDAT_FIELDS["account_id"], SDAT_FIELDS["district"], SDAT_FIELDS["account_number"],
        SDAT_FIELDS["lot"], SDAT_FIELDS["map"], SDAT_FIELDS["parcel"], SDAT_FIELDS["premise_number"],
        SDAT_FIELDS["premise_name"], SDAT_FIELDS["premise_type"], SDAT_FIELDS["premise_city"], SDAT_FIELDS["premise_zip"],
        SDAT_FIELDS["mdp_address"], SDAT_FIELDS["mdp_city"], SDAT_FIELDS["mdp_zip"], SDAT_FIELDS["link"],
    ]


def sdat_get(where_parts: list[str], limit: int = 200) -> list[dict[str, Any]]:
    if not where_parts:
        return []
    response = requests.get(
        SDAT_API_URL,
        params={"$limit": limit, "$select": ",".join(selected_sdat_fields()), "$where": " AND ".join(where_parts)},
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
    return normalize_identifier(record.get(SDAT_FIELDS[field], "")) == normalize_identifier(target)


def filter_sdat_records(records: list[dict[str, Any]], terms: SdatSearchTerms) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        if terms.tax_map and not record_identifier_matches(record, "tax_map", terms.tax_map):
            continue
        if terms.parcel and not record_identifier_matches(record, "parcel", terms.parcel):
            continue
        if terms.lot and not record_identifier_matches(record, "lot", terms.lot):
            continue
        if terms.district and not record_identifier_matches(record, "district", terms.district):
            continue
        if terms.account_number and not record_identifier_matches(record, "account_number", terms.account_number):
            continue
        filtered.append(record)
    return filtered or records


def lookup_maryland_property_records(terms: SdatSearchTerms) -> list[dict[str, Any]]:
    county_filter = (
        f"upper({SDAT_FIELDS['county']}) like upper('%{soql_escape(terms.county)}%')"
        if terms.county else ""
    )

    strategies: list[tuple[list[str], bool]] = []

    # 1. Best: district + account + county
    if terms.account_number and terms.district and county_filter:
        strategies.append((
            [
                county_filter,
                or_equals(SDAT_FIELDS["account_number"], terms.account_number, (6, 8)),
                or_equals(SDAT_FIELDS["district"], terms.district, (2,)),
            ],
            False,  # do NOT filter by lot/map/parcel after this
        ))

    # 2. Tax ID without county, useful when county OCR fails
    if terms.account_number and terms.district:
        strategies.append((
            [
                or_equals(SDAT_FIELDS["account_number"], terms.account_number, (6, 8)),
                or_equals(SDAT_FIELDS["district"], terms.district, (2,)),
            ],
            False,
        ))

    # 3. Map/parcel fallback
    if county_filter and terms.tax_map:
        strategies.append((
            [county_filter, or_equals(SDAT_FIELDS["map"], terms.tax_map, (3, 4))],
            True,
        ))

    if county_filter and terms.parcel:
        strategies.append((
            [county_filter, or_equals(SDAT_FIELDS["parcel"], terms.parcel, (3, 4))],
            True,
        ))

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
    street_address = " ".join(part for part in [number, street, street_type] if part).strip()
    if not street_address:
        street_address = normalize_value(record.get(SDAT_FIELDS["mdp_address"], ""))
        city = city or normalize_value(record.get(SDAT_FIELDS["mdp_city"], ""))
        zip_code = zip_code or normalize_value(record.get(SDAT_FIELDS["mdp_zip"], ""))
    return " ".join(part for part in [street_address, city, "MD", zip_code] if part).strip() if street_address else ""


def tax_id_from_sdat_record(record: dict[str, Any]) -> str:
    district = normalize_value(record.get(SDAT_FIELDS["district"], ""))
    account = normalize_value(record.get(SDAT_FIELDS["account_number"], ""))
    if district and account:
        return f"{district.zfill(2)}-{account.zfill(6)}"
    return ""


def metadata_from_sdat_record(metadata: ExtractedMetadata, record: dict[str, Any]) -> ExtractedMetadata:
    address = format_sdat_address(record)
    tax_map = normalize_value(record.get(SDAT_FIELDS["map"], ""))
    parcel = normalize_value(record.get(SDAT_FIELDS["parcel"], ""))
    tax_id = tax_id_from_sdat_record(record)
    return replace(
        metadata,
        address=safe_path_part(address, metadata.address) if address else metadata.address,
        tax_map=safe_path_part(tax_map, "") if tax_map else metadata.tax_map,
        parcel=safe_path_part(parcel, "") if parcel else metadata.parcel,
        tax_id=safe_path_part(tax_id, "") if tax_id else metadata.tax_id,
    )


def _address_tokens(address: str) -> tuple[str, list[str]]:
    cleaned = re.sub(r"[^0-9A-Za-z ]", " ", str(address or "")).upper()
    parts = [part for part in cleaned.split() if part]
    number = parts[0] if parts and parts[0].isdigit() else ""
    stop = {"MD", "MARYLAND", "ST", "STREET", "RD", "ROAD", "DR", "DRIVE", "LN", "LANE", "CT", "COURT", "AVE", "AVENUE", "BLVD", "BOULEVARD", "WAY", "PL", "PLACE", "CIR", "CIRCLE"}
    words = [part for part in parts[1:] if part not in stop and not part.isdigit()]
    return number, words[:3]


def lookup_maryland_property_by_address(address: str, county: str = "", limit: int = 100) -> list[dict[str, Any]]:
    number, words = _address_tokens(address)
    if not number or not words:
        return []
    where = [f"{SDAT_FIELDS['premise_number']} = '{soql_escape(number)}'"]
    where.append(f"upper({SDAT_FIELDS['mdp_address']}) like upper('%{soql_escape(words[0])}%')")
    if county:
        where.append(f"upper({SDAT_FIELDS['county']}) like upper('%{soql_escape(county)}%')")
    records = sdat_get(where, limit=limit)
    if not records:
        return []
    target = re.sub(r"[^A-Z0-9]", "", address.upper())
    def score(record: dict[str, Any]) -> int:
        candidate = re.sub(r"[^A-Z0-9]", "", format_sdat_address(record).upper())
        return sum(1 for token in [number, *words] if token and token in candidate) + (5 if candidate == target else 0)
    return sorted(records, key=score, reverse=True)


def enrich_metadata_with_sdat(metadata: ExtractedMetadata, text: str, config: Config) -> ExtractedMetadata:
    if not config.get("sdat_lookup", True):
        return metadata
    records = lookup_maryland_property_records(extract_sdat_search_terms(text, metadata, config))
    return metadata_from_sdat_record(metadata, records[0]) if records else metadata


def is_known_value(value: str) -> bool:
    value = str(value or "").strip()
    return bool(value) and not value.lower().startswith("unknown") and value not in {"Project", "Document"}


def vote_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_for_fuzzy(value))


def vote_for_value(values: Iterable[str], fallback: str) -> str:
    seen_display: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for value in values:
        if not is_known_value(value):
            continue
        key = vote_key(value)
        if key:
            counts[key] += 1
            seen_display.setdefault(key, value)
    return seen_display[counts.most_common(1)[0][0]] if counts else fallback


def extract_document_metadata_votes(scanned_documents: Iterable[dict[str, Any]], config: Config, default_project_code: str, default_document_type: str) -> list[ExtractedMetadata]:
    return [extract_metadata(document.get("ocr_text", ""), config, default_project_code, default_document_type) for document in scanned_documents]



PLAN_DOCUMENT_TYPES = {"Site Plan", "House Location", "Wall Check"}


def _field_notes_visual_threshold(config: Config) -> float:
    try:
        return float(config.get("visual_field_notes_threshold", 0.70))
    except Exception:
        return 0.70


def fix_duplicate_document_types_with_visual_classifier(
    votes: list[ExtractedMetadata],
    scanned_documents: list[dict[str, Any]],
    config: Config,
) -> list[ExtractedMetadata]:
    """Use visual classification to catch field notes mislabeled as plan documents.

    This is intentionally a post-processing safety net. It only runs when there
    are duplicate plan document types in the same batch, because that is the
    common signal that Field Notes were OCR-labeled as Site Plan/Wall Check/etc.
    It does not use OCR text; it renders the PDF and classifies the page visuals.
    """
    if not config.get("visual_field_notes_classifier", True):
        return votes

    by_type: dict[str, list[int]] = {}
    for index, metadata in enumerate(votes):
        by_type.setdefault(metadata.document_type, []).append(index)

    updated = list(votes)
    threshold = _field_notes_visual_threshold(config)

    for document_type, indexes in by_type.items():
        if document_type == "Field Notes" or document_type not in PLAN_DOCUMENT_TYPES or len(indexes) < 2:
            continue

        scored: list[tuple[float, int, str]] = []
        for index in indexes:
            source_path = scanned_documents[index].get("source_path") or scanned_documents[index].get("path")
            if not source_path:
                continue
            label, confidence = classify_pdf_visual(source_path)
            if label == FIELD_NOTES_LABEL:
                scored.append((confidence, index, label))

        # Convert visually confirmed field notes. Keep at least one original plan type.
        scored.sort(reverse=True)
        for confidence, index, _label in scored:
            remaining_same_type = sum(1 for item in updated if item.document_type == document_type)
            if confidence >= threshold and remaining_same_type > 1:
                updated[index] = replace(updated[index], document_type="Field Notes")

    return updated


def choose_batch_metadata_by_vote(
    scanned_documents: list[dict[str, Any]],
    config: Config,
    default_project_code: str,
    default_document_type: str,
) -> tuple[dict[str, str], list[ExtractedMetadata]]:
    """Vote once across the batch for shared metadata.

    Lot selection keeps the original technique inside extract_metadata(): the lot
    search starts at the detected document-type index, so random surrounding lot
    labels are less likely to win. Tax map, parcel, and tax ID are also shared
    across the batch so the best document can supply them for all related files.
    """
    votes = extract_document_metadata_votes(scanned_documents, config, default_project_code, default_document_type)
    votes = fix_duplicate_document_types_with_visual_classifier(votes, scanned_documents, config)

    shared = {
        "lot": vote_for_value((vote.lot for vote in votes), "Unknown Lot"),
        "address": vote_for_value((vote.address for vote in votes), "Unknown Address"),
        "tax_map": vote_for_value((vote.tax_map for vote in votes), ""),
        "parcel": vote_for_value((vote.parcel for vote in votes), ""),
        "tax_id": vote_for_value((vote.tax_id for vote in votes), ""),
        "section": vote_for_value((vote.section for vote in votes), ""),
    }

    # Do one SDAT lookup after voting instead of one lookup per document.
    if config.get("sdat_lookup", True):
        seed = replace(
            votes[0] if votes else ExtractedMetadata("Unknown Lot", "Unknown Address", default_project_code, default_document_type),
            lot=shared["lot"],
            address=shared["address"],
            tax_map=shared["tax_map"],
            parcel=shared["parcel"],
            tax_id=shared["tax_id"],
            section=shared.get("section", ""),
        )
        batch_text = "\n".join(document.get("ocr_text", "") for document in scanned_documents)
        enriched = enrich_metadata_with_sdat(seed, batch_text, config)
        shared["address"] = prefer_known(enriched.address, shared["address"])
        shared["tax_map"] = prefer_known(enriched.tax_map, shared["tax_map"])
        shared["parcel"] = prefer_known(enriched.parcel, shared["parcel"])
        shared["tax_id"] = prefer_known(enriched.tax_id, shared["tax_id"])

    return shared, votes


def merge_batch_metadata(
    document_text: str,
    config: Config,
    default_project_code: str,
    default_document_type: str,
    shared_metadata: Mapping[str, str],
    document_metadata: ExtractedMetadata | None = None,
) -> ExtractedMetadata:
    document_metadata = document_metadata or extract_metadata(document_text, config, default_project_code, default_document_type)
    return replace(
        document_metadata,
        lot=prefer_known(shared_metadata.get("lot", ""), document_metadata.lot),
        address=prefer_known(shared_metadata.get("address", ""), document_metadata.address),
        tax_map=prefer_known(shared_metadata.get("tax_map", ""), document_metadata.tax_map),
        parcel=prefer_known(shared_metadata.get("parcel", ""), document_metadata.parcel),
        tax_id=prefer_known(shared_metadata.get("tax_id", ""), document_metadata.tax_id),
        section=prefer_known(shared_metadata.get("section", ""), document_metadata.section),
        project_code=safe_path_part(default_project_code, "Project"),
    )


def file_pdf(pdf_path: Path, text: str, metadata: ExtractedMetadata, output_root: Path, copy_file: bool, save_text: bool) -> Path:
    folder_name = safe_path_part(f"Lot {metadata.lot} - {metadata.address}", "Unknown Lot - Unknown Address")
    destination_folder = output_root / folder_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    file_stem = safe_path_part(f"{metadata.document_type} - Lot {metadata.lot}", pdf_path.stem)
    destination_pdf = unique_path(destination_folder / f"{file_stem}.pdf")
    shutil.copy2(pdf_path, destination_pdf) if copy_file else shutil.move(str(pdf_path), destination_pdf)
    if save_text:
        destination_pdf.with_suffix(".txt").write_text(text, encoding="utf-8")
    return destination_pdf


def iter_pdfs(input_folder: Path) -> Iterable[Path]:
    return sorted(path for path in input_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR and file related PDFs as one lot packet.")
    parser.add_argument("--input", required=True, type=Path, help="Folder containing incoming PDF files.")
    parser.add_argument("--output", required=True, type=Path, help="Root folder for filed PDFs.")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="JSON config file.")
    parser.add_argument("--project-code", default="Project", help="Fallback project code.")
    parser.add_argument("--document-type", default="Document", help="Fallback document type.")
    parser.add_argument("--copy", action="store_true", help="Copy PDFs instead of moving them.")
    parser.add_argument("--save-text", action="store_true", help="Save OCR text beside the filed PDF.")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI before OCR.")
    parser.add_argument("--lang", default="en", help="PaddleOCR language code.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_folder = args.input.resolve()
    output_root = args.output.resolve()
    if not input_folder.is_dir():
        print(f"Input folder does not exist or is not a folder: {input_folder}", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)
    pdfs = list(iter_pdfs(input_folder))
    if not pdfs:
        print(f"No PDFs found in {input_folder}")
        return 0
    config = load_config(args.config if args.config.exists() else None)
    project_code = extract_project_code_from_output_folder(output_root, config, args.project_code)
    scanned = [
        {"path": Path(item["source_path"]), "ocr_text": item["ocr_text"]}
        for item in ocr_pdf_batch(
            pdfs,
            dpi=args.dpi,
            lang=args.lang,
            workers=1,
            threads_per_worker=4,
            ocr_device="auto",
            gpu_device_id=0,
        )
    ]
    shared_metadata, _votes = choose_batch_metadata_by_vote(scanned, config, project_code, args.document_type)
    for item in scanned:
        metadata = merge_batch_metadata(item["ocr_text"], config, project_code, args.document_type, shared_metadata)
        print(f"FILED: {file_pdf(item['path'], item['ocr_text'], metadata, output_root, args.copy, args.save_text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
