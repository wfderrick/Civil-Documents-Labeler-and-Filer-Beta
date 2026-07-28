"""Command-line inspection tool for the binary visual document classifier.

This is not a normal pytest test. A developer gives it one PDF or a folder, and it
prints the model's Field Notes/Other decision and confidence so model behavior can
be checked against real drawings."""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_classifier import classify_pdf_visual


def main() -> int:
    """Run the visual classifier manually against one PDF or every PDF in a folder.

    ArgumentParser collects the path, each PDF is passed to classify_pdf_visual(), and the
    label/confidence are printed for human inspection. Returning zero signals that the
    inspection command completed normally."""
    parser = argparse.ArgumentParser(
        description="Test the binary visual Field Notes classifier on PDFs."
    )
    parser.add_argument(
        "path", type=Path, help="PDF file or folder of PDFs to classify."
    )
    args = parser.parse_args()

    paths = [args.path] if args.path.is_file() else sorted(args.path.glob("*.pdf"))
    for pdf_path in paths:
        label, confidence = classify_pdf_visual(pdf_path)
        print(f"{pdf_path.name} -> {label} ({confidence:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
