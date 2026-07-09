from __future__ import annotations

import argparse
from pathlib import Path

from visual_classifier import classify_pdf_visual


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the binary visual Field Notes classifier on PDFs.")
    parser.add_argument("path", type=Path, help="PDF file or folder of PDFs to classify.")
    args = parser.parse_args()

    paths = [args.path] if args.path.is_file() else sorted(args.path.glob("*.pdf"))
    for pdf_path in paths:
        label, confidence = classify_pdf_visual(pdf_path)
        print(f"{pdf_path.name} -> {label} ({confidence:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
