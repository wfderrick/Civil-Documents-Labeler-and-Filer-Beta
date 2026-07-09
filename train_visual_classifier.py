from __future__ import annotations

import argparse
from pathlib import Path

from visual_classifier import train_visual_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the binary visual Field Notes classifier.")
    parser.add_argument("training_root", type=Path, help="Folder containing field_notes and not_field_notes subfolders.")
    parser.add_argument("--output", type=Path, default=Path("visual_field_notes_classifier.joblib"))
    args = parser.parse_args()
    train_visual_classifier(args.training_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
