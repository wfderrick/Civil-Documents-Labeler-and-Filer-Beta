"""Render PDFs, run PaddleOCR, and normalize version-dependent OCR output.

The rest of the application expects one stable result shape even though
PaddleOCR 2.x and 3.x return different Python structures. This module is
the compatibility boundary: it converts each PDF page to an image, asks
PaddleOCR to recognize it, and returns both plain text and coordinates.

Why coordinates are retained:
    Plain text is enough for most regex searches, but an address printed
    in a title block can be confused with the survey company's address.
    Page coordinates let metadata_extraction rebuild text lines near the
    bottom of the page and prefer the actual project title block.

The module also caches expensive OCR engines and supports process-based
CPU parallelism. GPU scans deliberately use one engine because multiple
processes competing for one GPU usually reduce reliability and speed."""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fitz

try:
    import paddle
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    paddle = None



@contextmanager
def time_block(name: str, progress_callback: Callable[[str], None] | None = None):
    """Measure one OCR operation and report its elapsed time when the block exits.
    
    This context manager surrounds rendering and page recognition calls. Code inside the
    ``with`` block runs normally; afterward the helper formats a message for either the
    browser progress callback or the terminal. It keeps timing logic out of the OCR loops."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    message = f"{name} in {elapsed:.2f} seconds."
    if progress_callback:
        progress_callback(message)
    else:
        print(message)


def _as_float_pair(value: Any) -> list[float] | None:
    """Convert one point-like OCR value into a numeric x/y pair.
    
    Geometry from third-party libraries is not always a plain Python list. Conversion is
    attempted defensively, and invalid values return ``None`` so one malformed point does
    not cancel OCR for the entire page."""
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    except Exception:  # noqa: BLE001
        return None
    return None


def _points_from_any(value: Any) -> list[list[float]]:
    """Normalize rectangles or polygons returned by different PaddleOCR versions.
    
    A four-number rectangle is expanded into its four corners. A list of point-like values
    is converted one item at a time. The rest of the app can therefore work with one
    polygon representation regardless of the OCR package version."""
    if value is None:
        return []

    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    ):
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
    """Collapse polygon points into ``left, top, right, bottom`` coordinates.
    
    Metadata extraction and PDF text-layer logic need simple extents rather than arbitrary
    polygons. A zero rectangle represents missing geometry without raising an exception."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def first_nonempty_value(*values):
    """Select the first non-empty candidate from several OCR result fields.
    
    NumPy arrays cannot always be tested with ordinary truth-value rules, so the helper
    checks ``size`` and ``len`` carefully. It is mainly used to locate whichever box field
    exists in the installed PaddleOCR version."""
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
    """Convert raw PaddleOCR output into the application's stable token format.
    
    PaddleOCR 3.x often returns dictionaries containing parallel text, score, and polygon
    arrays; older versions return nested lists. This function understands both shapes and
    creates dictionaries containing text, confidence, polygon, and bounding box.
    
    Downstream code never needs to know which Paddle version produced the result. Invalid
    individual tokens are skipped, while valid tokens on the same page are preserved."""
    items: list[dict[str, Any]] = []

    # PaddleOCR 3.x returns dictionary-based pages; 2.x commonly returns
    # nested lists. Both branches below produce the exact same item schema.
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
                item: dict[str, Any] = {"text": str(text)}
                if i < len(scores):
                    item["score"] = float(scores[i])

                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    if hasattr(box, "tolist"):
                        box = box.tolist()

                    points = _points_from_any(box)
                    if points:
                        item["polygon"] = points
                        item["bbox"] = _bbox_from_points(points)

                items.append(item)

        elif isinstance(page_result, list):
            for raw_item in page_result:
                try:
                    points = _points_from_any(raw_item[0])
                    text = str(raw_item[1][0]).strip()
                    confidence = float(raw_item[1][1])
                except Exception:  # noqa: BLE001, S112
                    continue
                if text:
                    items.append(
                        {
                            "text": text,
                            "confidence": confidence,
                            "polygon": points,
                            "bbox": _bbox_from_points(points),
                        }
                    )

    return items


MAX_OCR_IMAGE_SIDE = 3999


def _page_ocr_matrix(
    page: fitz.Page, requested_dpi: int, max_side: int = MAX_OCR_IMAGE_SIDE
) -> tuple[fitz.Matrix, float]:
    """Choose a PDF render scale that reaches the requested DPI without oversized OCR images.
    
    Paddle would resize an image whose longest side exceeds its detection limit. Rendering
    directly at that effective maximum avoids creating a much larger temporary PNG only
    for Paddle to shrink it again. The returned effective DPI is saved for coordinate
    conversion later."""
    requested_scale = max(float(requested_dpi), 72.0) / 72.0
    projected_max = max(
        float(page.rect.width) * requested_scale,
        float(page.rect.height) * requested_scale,
    )
    if projected_max > max_side:
        scale = requested_scale * (max_side / projected_max)
    else:
        scale = requested_scale
    return fitz.Matrix(scale, scale), scale * 72.0


def render_pdf_pages_with_info(
    pdf_path: Path, image_dir: Path, dpi: int
) -> list[dict[str, Any]]:
    """Render every PDF page to a temporary PNG and retain coordinate-conversion information.
    
    Each returned record contains the image path and dimensions together with the original
    PDF page dimensions, rotation, and effective DPI. OCR operates on the PNG; later code
    uses these values to map recognized token positions back to the PDF page accurately."""
    pages: list[dict[str, Any]] = []
    image_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):  # pyright: ignore[reportArgumentType]
            matrix, effective_dpi = _page_ocr_matrix(page, dpi)
            image_path = image_dir / f"page-{page_index + 1:04d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False, annots=False)
            pixmap.save(image_path)
            pages.append(
                {
                    "page_index": page_index,
                    "image_path": image_path,
                    "image_width": pixmap.width,
                    "image_height": pixmap.height,
                    "page_width": float(page.rect.width),
                    "page_height": float(page.rect.height),
                    "dpi": effective_dpi,
                    "page_rotation": int(page.rotation),
                }
            )
    return pages


def ocr_pdf_with_layout(
    pdf_path: Path,
    ocr: PaddleOCR,
    dpi: int,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """OCR one PDF and return both combined text and page-level layout data.
    
    The PDF is rendered inside a temporary directory that is automatically deleted. Each
    page is sent to PaddleOCR, normalized through ``extract_ocr_items``, and added to a page
    record with its dimensions. Token text is also joined into one newline-separated string
    for regex and fuzzy metadata extraction.
    
    Progress callbacks report rendering, recognition, and normalization separately, making
    it possible to distinguish a slow PDF render from a slow OCR model prediction."""
    lines: list[str] = []
    ocr_pages: list[dict[str, Any]] = []

    # Rendered page images are intermediate OCR inputs, not project records.
    # A temporary directory guarantees cleanup even when one page raises.
    with tempfile.TemporaryDirectory(prefix="paddleocr_pdf_") as temp_dir:
        with time_block(f"{pdf_path.name}: rendered pages", progress_callback):
            rendered_pages = render_pdf_pages_with_info(pdf_path, Path(temp_dir), dpi)
        total = len(rendered_pages)
        num = 1
        for page_info in rendered_pages:
            with time_block(
                f"{pdf_path.name}: OCR page {num} of {total}", progress_callback
            ):
                result = ocr.predict(str(page_info["image_path"]))
            num += 1
            extract_started = time.perf_counter()
            items = extract_ocr_items(result)
            extract_elapsed = time.perf_counter() - extract_started
            if progress_callback:
                progress_callback(
                    f"{pdf_path.name}: normalized OCR page {num - 1} of {total} "
                    f"in {extract_elapsed:.2f} seconds."
                )
            lines.extend(item["text"] for item in items if item.get("text"))
            ocr_pages.append(
                {
                    "page_index": page_info["page_index"],
                    "image_width": page_info["image_width"],
                    "image_height": page_info["image_height"],
                    "page_width": page_info["page_width"],
                    "page_height": page_info["page_height"],
                    "dpi": page_info["dpi"],
                    "items": items,
                }
            )

    return {"text": "\n".join(lines), "pages": ocr_pages}


def gpu_is_available() -> bool:
    """Check whether the installed Paddle build and computer can actually use CUDA.
    
    Merely requesting GPU mode is not proof that Paddle was compiled with CUDA or that a
    visible device exists. Exceptions are treated as unavailable so ``auto`` mode can fall
    back to CPU instead of failing before a scan begins."""
    if paddle is None:
        return False
    try:
        return bool(
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:  # noqa: BLE001
        return False


def resolve_ocr_device(ocr_device: str = "auto", gpu_device_id: int = 0) -> str:
    """Turn the user's ``auto``, ``cpu``, or ``gpu`` choice into Paddle's device string.
    
    Explicit choices are honored; ``auto`` selects the requested GPU index only when
    ``gpu_is_available`` succeeds. This resolved value is also part of the OCR-engine cache
    key, preventing a CPU engine from being reused for a later GPU request."""
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
    """Construct a PaddleOCR engine compatible with several PaddleOCR releases.
    
    The function disables orientation and unwarping models because they add startup cost and
    transform token coordinates away from the original PDF geometry. It then attempts the
    modern ``device=`` constructor, the older ``use_gpu`` form, and finally a minimal form.
    
    CPU thread settings are applied only to CPU mode. A constructor ``TypeError`` means a
    particular API style is unsupported and triggers the next attempt; other failures still
    surface to the caller."""
    resolved_device = resolve_ocr_device(ocr_device, gpu_device_id)

    # Keep PaddleOCR geometry in the same orientation and coordinate system as
    # the rendered PDF page. The document orientation / unwarping pipelines
    # transform the image before detection, which makes their returned boxes
    # unsuitable for writing directly back onto the original PDF. Disabling
    # them also avoids loading three unnecessary preprocessing models.
    base_kwargs: dict[str, Any] = {
        "lang": lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_det_limit_side_len": 4000,
        "text_det_limit_type": "max",
    }

    if resolved_device == "cpu" and cpu_threads:
        base_kwargs["cpu_threads"] = int(cpu_threads)

    attempts: list[dict[str, Any]] = []

    # PaddleOCR 3.x
    attempts.append({**base_kwargs, "device": resolved_device})

    # Older PaddleOCR versions sometimes used use_gpu instead of device.
    if resolved_device.startswith("gpu"):
        attempts.append(
            {**base_kwargs, "use_gpu": True, "gpu_id": int(gpu_device_id or 0)}
        )
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


_OCR_ENGINE_CACHE: dict[tuple[str, str, int, int], PaddleOCR] = {}
_OCR_ENGINE_CACHE_LOCK = threading.Lock()


def get_cached_ocr(
    lang: str = "en",
    cpu_threads: int | None = None,
    ocr_device: str = "auto",
    gpu_device_id: int = 0,
) -> PaddleOCR:
    """Reuse a process-lifetime OCR engine for an identical configuration.
    
    Loading Paddle models is expensive compared with scanning a small packet. The cache key
    includes language, resolved device, GPU index, and CPU thread count. A lock prevents two
    simultaneous requests from constructing the same engine twice.
    
    Worker processes do not share this cache because their memory is isolated and Paddle
    predictor objects should not be transferred between processes."""
    resolved_device = resolve_ocr_device(ocr_device, gpu_device_id)
    key = (
        str(lang or "en"),
        resolved_device,
        int(gpu_device_id or 0),
        int(cpu_threads or 0),
    )
    with _OCR_ENGINE_CACHE_LOCK:
        engine = _OCR_ENGINE_CACHE.get(key)
        if engine is None:
            engine = make_ocr(
                lang=key[0],
                cpu_threads=(key[3] or None),
                ocr_device=resolved_device,
                gpu_device_id=key[2],
            )
            _OCR_ENGINE_CACHE[key] = engine
        return engine


def _init_ocr_worker(
    lang: str, cpu_threads: int, ocr_device: str, gpu_device_id: int
) -> None:
    """Create the private OCR engine used inside one CPU worker process.
    
    ``ProcessPoolExecutor`` calls this initializer once per worker. The engine is stored in a
    process-global variable so every PDF assigned to that worker can reuse the loaded models
    instead of rebuilding them for each task."""
    global _WORKER_OCR
    _WORKER_OCR = make_ocr(
        lang=lang,
        cpu_threads=cpu_threads,
        ocr_device=ocr_device,
        gpu_device_id=gpu_device_id,
    )


def _ocr_one_pdf_worker(
    index: int, pdf_path_text: str, dpi: int
) -> tuple[int, dict[str, Any]]:
    """OCR one PDF inside a worker process and preserve its original batch position.
    
    The worker uses its process-local engine, builds the standard scan record, and returns
    the input index alongside the result. The parent process uses that index to restore the
    original filename order even though workers finish out of order."""
    if _WORKER_OCR is None:
        raise RuntimeError("OCR worker was not initialized.")
    pdf_path = Path(pdf_path_text)
    full_ocr = ocr_pdf_with_layout(pdf_path, _WORKER_OCR, dpi)
    full_text = full_ocr["text"]
    return index, {
        "source_path": str(pdf_path),
        "source_name": pdf_path.name,
        "ocr_text": f"{pdf_path.stem}\n{full_text}",
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
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """OCR an ordered list of PDFs using the safest available execution strategy.
    
    With one worker, the function reuses the supplied or cached engine and reports detailed
    page progress. With multiple CPU workers, it creates a process pool whose workers each
    initialize a private engine. GPU requests are forced to one worker to avoid competing
    model instances on the same device.
    
    Every returned item has ``source_path``, ``source_name``, combined ``ocr_text``, and
    coordinate-rich ``ocr_pages``. Parallel results are reordered to match the input list so
    batch voting and UI display remain deterministic."""
    if not pdf_paths:
        return []

    resolved_device = resolve_ocr_device(ocr_device, gpu_device_id)

    if resolved_device.startswith("gpu"):
        workers = 1
    else:
        workers = max(1, int(workers or 1))

    # Sequential mode gives the richest progress messages and can reuse the
    # main process engine. The process-pool branch below trades that detail for
    # parallel CPU throughput on larger batches.
    if workers == 1:
        ocr = existing_ocr or make_ocr(
            lang=lang,
            cpu_threads=threads_per_worker,
            ocr_device=resolved_device,
            gpu_device_id=gpu_device_id,
        )
        results: list[dict[str, Any]] = []
        total = len(pdf_paths)
        for num, pdf_path in enumerate(pdf_paths, start=1):
            if progress_callback:
                if total > 1:
                    progress_callback(f"Document {num} of {total}: {pdf_path.name}")
                else:
                    progress_callback(f"Document: {pdf_path.name}")
            full_ocr = ocr_pdf_with_layout(
                pdf_path, ocr, dpi, progress_callback=progress_callback
            )
            full_text = full_ocr["text"]
            results.append(
                {
                    "source_path": str(pdf_path),
                    "source_name": pdf_path.name,
                    "ocr_text": f"{full_text}",
                    "ocr_pages": full_ocr["pages"],
                }
            )
        return results

    indexed_results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_ocr_worker,
        initargs=(
            lang,
            int(threads_per_worker or 4),
            resolved_device,
            int(gpu_device_id or 0),
        ),
    ) as executor:
        futures = {
            executor.submit(_ocr_one_pdf_worker, index, str(pdf_path), dpi): index
            for index, pdf_path in enumerate(pdf_paths)
        }
        completed = 0
        for future in as_completed(futures):
            index, result = future.result()
            indexed_results[index] = result
            completed += 1  # noqa: SIM113
            if progress_callback:
                progress_callback(
                    f"Completed document {completed} of {len(pdf_paths)}: "
                    f"{result['source_name']}"
                )

    return [indexed_results[index] for index in range(len(pdf_paths))]
