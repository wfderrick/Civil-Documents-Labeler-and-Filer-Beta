"""Implementation module for the COABarrett File Identifier and Sorter. It groups related application behavior and is documented to help future maintainers trace data flow and side effects.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_classifier import classify_pdf_visual


def main() -> int:
    """Run the module as a command-line entry point.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
    """
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
