"""Append a human-readable audit row after a batch is filed.

The main application state describes work still being reviewed. Once a
batch leaves that queue, this CSV provides a separate historical record
of what property was filed, where it was filed, when it happened, and
which filenames were created."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

TRACKER_DIR = Path(r"C:\ocr tracker")
TRACKER_FILE = TRACKER_DIR / "filed_batches.csv"


def append_batch_tracker(
    documents: list[dict[str, Any]],
    output_folder: Path,
    filed_documents: list[dict[str, Any]],
) -> None:
    """Append one audit row describing a successfully filed batch.
    
    The first permanent document supplies shared property/project values, while ``filed_documents``
    supplies the actual destination filenames. The function creates ``C:\\ocr tracker`` when
    needed, writes a header for a new/empty CSV, and appends rather than replacing prior history.
    
    Empty inputs return immediately because there is no meaningful completed batch to record."""
    if not documents or not filed_documents:
        return

    metadata = documents[0].get("metadata", {})
    destination_folder = Path(filed_documents[0]["filed_path"]).parent
    row = {
        "lot_number": metadata.get("lot", ""),
        "address": metadata.get("address", ""),
        "location_filed": str(destination_folder),
        "time_filed": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "project_code": metadata.get("project_code", ""),
        "section": metadata.get("section", ""),
        "file_count": len(filed_documents),
        "files_filed": "|".join(
            Path(doc["filed_path"]).name for doc in filed_documents
        ),
    }

    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    needs_header = (
        not TRACKER_FILE.exists() or TRACKER_FILE.stat().st_size == 0
    )
    with TRACKER_FILE.open("a", newline="", encoding="utf-8") as tracker:
        writer = csv.DictWriter(tracker, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
