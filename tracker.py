"""Output-change tracking helpers used to record files created, moved, or renamed by the application.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""

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
    """The append_batch_tracker() function adds a new row to the ocr tracker
    file when a batch of documents is filed with lot number,
    address, location filed, time filed, project code, section, file count, and
    files filed. It consolidates that information into a python dictionary and
    uses the DictWriter() function to write a new line to the csv file. If the t
    tracker hasn't been created yet a header is added at the start of the file
    when it is created."""
    if not documents or not filed_documents:
        return

    metadata = documents[0].get("metadata", {})
    destination_folder = Path(filed_documents[0]["filed_path"]).parent
    row = {
        "lot_number": metadata.get("lot", ""),
        "address": metadata.get("address", ""),
        "location_filed": str(destination_folder),
        "time_filed": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_code": metadata.get("project_code", ""),
        "section": metadata.get("section", ""),
        "file_count": len(filed_documents),
        "files_filed": "|".join(
            Path(doc["filed_path"]).name for doc in filed_documents
        ),
    }

    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    needs_header = not TRACKER_FILE.exists() or TRACKER_FILE.stat().st_size == 0
    with TRACKER_FILE.open("a", newline="", encoding="utf-8") as tracker:
        writer = csv.DictWriter(tracker, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
