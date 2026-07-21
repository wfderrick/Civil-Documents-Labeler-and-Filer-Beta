"""Small self-contained test for the visual-classifier joblib cache."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import joblib

from visual_classifier import _get_cached_model, clear_model_cache


def main() -> None:
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

    print("Visual classifier model-cache test passed.")


if __name__ == "__main__":
    main()
