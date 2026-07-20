from __future__ import annotations
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping
import fitz
try:
    import paddle
except Exception:
    paddle = None
from paddleocr import PaddleOCR
from metadata_extraction import Config

@contextmanager
def time_block(name: str, progress_callback: Callable[[str], None] | None = None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    message = f"{name} in {elapsed:.2f} seconds."
    if progress_callback:
        progress_callback(message)
    else:
        print(message)

def _as_float_pair(value: Any) -> list[float] | None:
    """Convert a PaddleOCR point-like value to [x, y]."""
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    except Exception:
        return None
    return None

def _points_from_any(value: Any) -> list[list[float]]:
    """The _points_from_any() function returns points for a polygon based on the
    value parameter. Converts bounding boxes and polygons in polygon point
    format."""
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
    """The _bbox_from_points() function returns the bounding box representation
    of the points parameter entered. It returns the leftmost x, the lowest y,
    the rightmost x, and the highest y as a list."""
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
    """The extract_ocr_items() function returns a list of dictionaries
    containing words, the confidence associated with them, and bounding boxes to
    show their location on the page.
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
                except Exception:
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
    """Return the fastest render matrix that does not trigger Paddle resizing."""
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
    """The render_pdf_page_with_info() function converts the pdf at the location
    specified in the pdf_path parameter into images and places them at the
    location specified in the image_dir parameter with the image resolution
    controlled with the dpi parameter. The function the returns a list of
    dictionaries with information on each page contained in the pdf. This data
    includes index, path, pdf width, pdf height, image width, image height, and
    dpi. This allows for conversions between location of text on the rendered
    image vs location of text on the pdf."""
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
    """The ocr_pdf_with_layout() function returns a dictionary with text from
    the pdf located at the pdf_path parameter by the ocr pointed to by the ocr
    parameter after being converted to images with resolution determined by the
    dpi parameter. First a temporary directory is created to house the rendered
    image form of the pdfs. Then the pdfs are rendered by the
    render_pdf_pages_with_info() function and placed into the directory and the
    function returns data on all of the images. Then each page is ocred with
    the predict() function and timed to send an update to the user.
    The extract_ocr_items() function takes the result of the predict() function
    and converts it to a list of tokens with each token having the characters
    contained, the ocr's confidence in the classification, and the location.
    lines has each text item added to it. The info taken from the
    render_pdf_with_info() function and items from the extract_ocr_items() are
    put into the ocr_pages list. Finally all of it is returned with the lines
    list of text being combined into a single string and the ocr_pages list."""
    lines: list[str] = []
    ocr_pages: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="paddleocr_pdf_") as temp_dir:
        rendered_pages = render_pdf_pages_with_info(pdf_path, Path(temp_dir), dpi)
        total = len(rendered_pages)
        num = 1
        for page_info in rendered_pages:
            with time_block(
                f"{pdf_path.name}: OCR page {num} of {total}", progress_callback
            ):
                result = ocr.predict(str(page_info["image_path"]))
            num += 1
            items = extract_ocr_items(result)
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
    """Return True when the installed Paddle package can see a CUDA GPU. If both
    paddle's built in is_compiled_with_cuda() function to checks that the paddle
    installed can run on a gpu and device_count() to checks that a gpu that can
    run cuda are available the function returns True.
    Otherwise it returns False."""
    if paddle is None:
        return False
    try:
        return bool(
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False

def resolve_ocr_device(ocr_device: str = "auto", gpu_device_id: int = 0) -> str:
    """Resolve auto/gpu/cpu into the device string PaddleOCR should use. Both
    gpu and cpu are selected when the ocr_device parameter is set to them
    respectively. When the ocr_device parameter is set to auto the function
    returns gpu if the gpu_is_available() function returns true otherwise cpu is
    returned."""
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
    Multiple attempts are made with different arguments inputted into the
    PaddleOCR constructor to ensure a constructor is returned.
    """
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

def _init_ocr_worker(
    lang: str, cpu_threads: int, ocr_device: str, gpu_device_id: int
) -> None:
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
    """The ocr_pdf_batch() function ocrs all of the pdfs in the pdf_paths
    parameter and returns a list of dictionaries holding the path
    """
    if not pdf_paths:
        return []

    resolved_device = resolve_ocr_device(ocr_device, gpu_device_id)

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
        total = len(pdf_paths)
        for num, pdf_path in enumerate(pdf_paths, start=1):
            if progress_callback:
                progress_callback(f"Document {num} of {total}: {pdf_path.name}")
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
            completed += 1
            if progress_callback:
                progress_callback(
                    f"Completed document {completed} of {len(pdf_paths)}: "
                    f"{result['source_name']}"
                )

    return [indexed_results[index] for index in range(len(pdf_paths))]
