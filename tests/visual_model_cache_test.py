"""Self-contained regression check for visual-classifier model caching.

Loading a joblib model for every PDF would waste scan time. This script confirms
repeated reads reuse one object and that replacing the model file invalidates the
cache through its modification timestamp."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import joblib

from visual_classifier import _get_cached_model, clear_model_cache


def main() -> None:
    """Verify both reuse and invalidation of the cached joblib visual model.

    A temporary model is loaded twice and must return the same object identity. After the
    file is replaced with a newer modification time, the next load must return the new
    contents and a different object."""
    with tempfile.TemporaryDirectory(prefix="visual_model_cache_test_") as temp_dir:
        model_path = Path(temp_dir) / "synthetic.joblib"

        first_object = {"version": 1, "weights": [1, 2, 3]}
        joblib.dump(first_object, model_path)
        clear_model_cache()

        loaded_a = _get_cached_model(model_path)
        loaded_b = _get_cached_model(model_path)
        assert loaded_a is loaded_b, "Repeated reads should reuse the cached object."
        assert loaded_a == first_object

        # Ensure a distinct modification timestamp before replacing the model.
        time.sleep(0.01)
        second_object = {"version": 2, "weights": [4, 5, 6]}
        joblib.dump(second_object, model_path)

        loaded_c = _get_cached_model(model_path)
        assert loaded_c == second_object, (
            "Replacing the model should invalidate by mtime."
        )
        assert loaded_c is not loaded_a


if __name__ == "__main__":
    main()
