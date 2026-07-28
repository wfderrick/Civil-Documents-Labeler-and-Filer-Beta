"""Command-line wrapper for training the optional Field Notes classifier.

This file is not used during an ordinary browser scan. It lets a developer
point the project at labeled ``field_notes`` and ``not_field_notes`` PDF
folders, then delegates the actual feature extraction and model training
to ``visual_classifier.train_visual_classifier``."""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_classifier import train_visual_classifier


def main() -> int:
    """Parse command-line training options and launch visual-model training.
    
    The required argument points to a folder containing ``field_notes`` and ``not_field_notes``
    subfolders. ``--output`` controls where the joblib model is saved. Returning zero lets
    ``SystemExit`` communicate a successful command-line run to scripts and shells."""
    parser = argparse.ArgumentParser(
        description="Train the binary visual Field Notes classifier."
    )
    parser.add_argument(
        "training_root",
        type=Path,
        help="Folder containing field_notes and not_field_notes subfolders.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("visual_field_notes_classifier.joblib")
    )
    args = parser.parse_args()
    train_visual_classifier(args.training_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
