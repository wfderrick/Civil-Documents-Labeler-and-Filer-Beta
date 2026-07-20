from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz
import numpy as np

try:
    import joblib  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    joblib = None

MODEL_PATH = Path(__file__).resolve().parent / "visual_field_notes_classifier.joblib"
FIELD_NOTES_LABEL = "field_notes"
NOT_FIELD_NOTES_LABEL = "not_field_notes"


@lru_cache(maxsize=4)
def _load_model_cached(model_path: str, modified_ns: int) -> Any:
    """Load a joblib model once per path/version.

    ``modified_ns`` is part of the cache key, so replacing or retraining the
    model invalidates the old cached entry automatically.
    """
    if joblib is None:
        return None
    return joblib.load(model_path)


def clear_model_cache() -> None:
    """Clear the in-process visual classifier model cache."""
    _load_model_cached.cache_clear()


def _get_cached_model(model_file: Path) -> Any:
    stat = model_file.stat()
    return _load_model_cached(str(model_file.resolve()), stat.st_mtime_ns)


def render_page_gray(pdf_path: Path, page_index: int = 0, dpi: int = 72) -> np.ndarray | None:
    """Render one PDF page as a small grayscale numpy image for visual classification."""
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count <= page_index:
                return None
            pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            return image.mean(axis=2).astype(np.uint8)
    except Exception:
        return None


def pdf_page_count(pdf_path: Path) -> int:
    try:
        with fitz.open(pdf_path) as doc:
            return int(doc.page_count)
    except Exception:
        return 0


def _resize_sample(gray: np.ndarray, target_h: int = 256, target_w: int = 192) -> np.ndarray:
    """Fast dependency-free resize by sampling. Good enough for classification features."""
    if gray.size == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    y_idx = np.linspace(0, gray.shape[0] - 1, target_h).astype(int)
    x_idx = np.linspace(0, gray.shape[1] - 1, target_w).astype(int)
    return gray[np.ix_(y_idx, x_idx)]


def visual_features(pdf_path: Path) -> np.ndarray:
    """Extract visual features without OCR.

    Binary classifier target:
      - field_notes
      - not_field_notes

    These features separate notebook/field-note pages from all non-field-note
    documents by page count, page aspect, darkness, edge texture, margins, and
    title-block-like density in the lower-right area.
    """
    page_count = pdf_page_count(pdf_path)
    gray = render_page_gray(pdf_path)
    if gray is None:
        return np.zeros(30, dtype=float)

    small = _resize_sample(gray)
    dark = small < 185
    very_dark = small < 90

    dx = np.abs(np.diff(small.astype(float), axis=1))
    dy = np.abs(np.diff(small.astype(float), axis=0))
    edge_ratio = float((dx > 30).mean() + (dy > 30).mean()) / 2.0

    h, w = small.shape
    lower_right = small[int(h * 0.62):, int(w * 0.55):]
    top_half = small[: int(h * 0.50), :]
    left_half = small[:, : int(w * 0.50)]

    zone_features: list[float] = []
    for y0 in np.linspace(0, h, 5, dtype=int)[:-1]:
        y1 = min(h, y0 + h // 4)
        for x0 in np.linspace(0, w, 5, dtype=int)[:-1]:
            x1 = min(w, x0 + w // 4)
            zone = small[y0:y1, x0:x1]
            zone_features.append(float((zone < 185).mean()) if zone.size else 0.0)

    return np.array([
        float(page_count),
        float(w) / float(h or 1),
        float(dark.mean()),
        float(very_dark.mean()),
        edge_ratio,
        float((lower_right < 185).mean()) if lower_right.size else 0.0,
        float((top_half < 185).mean()) if top_half.size else 0.0,
        float((left_half < 185).mean()) if left_half.size else 0.0,
        *zone_features,
    ], dtype=float)


def heuristic_field_notes_probability(pdf_path: Path) -> float:
    """Fallback visual score when no trained model is available."""
    features = visual_features(pdf_path)
    page_count = features[0]
    aspect = features[1]
    dark_ratio = features[2]
    edge_ratio = features[4]
    lower_right_density = features[5]

    score = 0.0
    if page_count >= 2:
        score += 0.55
    if page_count >= 4:
        score += 0.15
    if aspect < 0.85:
        score += 0.10
    if lower_right_density < 0.08:
        score += 0.08
    if 0.015 <= edge_ratio <= 0.13:
        score += 0.07
    if 0.005 <= dark_ratio <= 0.20:
        score += 0.05

    return max(0.0, min(0.98, score))


def classify_pdf_visual(pdf_path: str | Path, model_path: str | Path | None = None) -> tuple[str, float]:
    """Classify a PDF visually as field_notes or not_field_notes.

    If visual_field_notes_classifier.joblib exists and joblib is installed, it is used.
    Otherwise a lightweight heuristic is used. This function does not use OCR text.
    """
    path = Path(pdf_path)
    model_file = Path(model_path) if model_path else MODEL_PATH

    if joblib is not None and model_file.exists():
        model = _get_cached_model(model_file)
        features = visual_features(path).reshape(1, -1)
        label = str(model.predict(features)[0])
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        return label, confidence

    probability = heuristic_field_notes_probability(path)
    if probability >= 0.70:
        return FIELD_NOTES_LABEL, probability
    return NOT_FIELD_NOTES_LABEL, 1.0 - probability


def train_visual_classifier(training_root: str | Path, output_model: str | Path = MODEL_PATH) -> None:
    """Train binary visual classifier from folders: field_notes and not_field_notes."""
    if joblib is None:
        raise RuntimeError("Install joblib and scikit-learn to train the visual classifier.")
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[reportMissingImports]
    from sklearn.model_selection import train_test_split  # type: ignore[reportMissingImports]
    from sklearn.metrics import classification_report  # type: ignore[reportMissingImports]

    root = Path(training_root)
    expected = {FIELD_NOTES_LABEL, NOT_FIELD_NOTES_LABEL}
    x: list[np.ndarray] = []
    y: list[str] = []

    for label in sorted(expected):
        label_dir = root / label
        if not label_dir.is_dir():
            raise RuntimeError(f"Missing training folder: {label_dir}")
        for pdf_path in sorted(label_dir.glob("*.pdf")):
            x.append(visual_features(pdf_path))
            y.append(label)

    if len(set(y)) < 2:
        raise RuntimeError("Training data must include both field_notes and not_field_notes PDFs.")
    if len(y) < 6:
        raise RuntimeError("Add more training PDFs before training. Aim for at least 20 per folder.")

    X = np.vstack(x)
    if len(y) >= 10 and min(y.count(FIELD_NOTES_LABEL), y.count(NOT_FIELD_NOTES_LABEL)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump(model, output_model)
    clear_model_cache()

    print(f"Saved binary visual field-notes classifier to {output_model}")
    print("\nTraining summary:")
    print(classification_report(y_test, model.predict(X_test)))


# Batch post-processing helpers moved from pipeline.py.
from dataclasses import replace
from typing import Any
from metadata_extraction import Config, ExtractedMetadata

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
        if (
            document_type == "Field Notes"
            or document_type not in PLAN_DOCUMENT_TYPES
            or len(indexes) < 2
        ):
            continue

        scored: list[tuple[float, int, str]] = []
        for index in indexes:
            source_path = scanned_documents[index].get(
                "source_path"
            ) or scanned_documents[index].get("path")
            if not source_path:
                continue
            label, confidence = classify_pdf_visual(source_path)
            if label == FIELD_NOTES_LABEL:
                scored.append((confidence, index, label))

        # Convert visually confirmed field notes. Keep at least one original plan type.
        scored.sort(reverse=True)
        for confidence, index, _label in scored:
            remaining_same_type = sum(
                1 for item in updated if item.document_type == document_type
            )
            if confidence >= threshold and remaining_same_type > 1:
                updated[index] = replace(updated[index], document_type="Field Notes")

    return updated
