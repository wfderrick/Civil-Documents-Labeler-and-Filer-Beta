from __future__ import annotations
import threading
import time
from typing import Any

_SCAN_PROGRESS_LOCK = threading.Lock()
_SCAN_PROGRESS: dict[str, Any] = {
    "active": False,
    "finished": False,
    "failed": False,
    "started_at": 0.0,
    "messages": [],
}


def reset_scan_progress() -> None:
    """The reset_scan_progress() function clears previous information from the
    _SCAN_PROGRESS dictionary to start fresh when a new batch of documents is
    being scanned."""
    with _SCAN_PROGRESS_LOCK:
        _SCAN_PROGRESS.update(
            {
                "active": True,
                "finished": False,
                "failed": False,
                "started_at": time.perf_counter(),
                "messages": [],
            }
        )


def add_scan_progress(message: str) -> None:
    """The add_scan_progress() function takes in the message parameter and adds
    it to the messages key in the _SCAN_PROGRESS dictionary."""
    text = str(message or "").strip()
    if not text:
        return
    with _SCAN_PROGRESS_LOCK:
        elapsed = max(
            0.0,
            time.perf_counter() - float(_SCAN_PROGRESS.get("started_at") or 0.0),
        )
        _SCAN_PROGRESS.setdefault("messages", []).append(
            {
                "text": text,
                "elapsed": round(elapsed, 2),
            }
        )


def finish_scan_progress(*, failed: bool = False, message: str = "") -> None:
    """The finish_scan_progress() function sets the _SCAN_PROGRESS dictionary to
    its finished state after a batch has finished scanning."""
    if message:
        add_scan_progress(message)
    with _SCAN_PROGRESS_LOCK:
        _SCAN_PROGRESS["active"] = False
        _SCAN_PROGRESS["finished"] = True
        _SCAN_PROGRESS["failed"] = failed


def scan_progress_snapshot() -> dict[str, Any]:
    """The scan_progress_snapshot() function returns total time, active,
    finished, and failed scans and messages in the _SCAN_PROGRESS dictionary."""
    with _SCAN_PROGRESS_LOCK:
        started_at = float(_SCAN_PROGRESS.get("started_at") or 0.0)
        elapsed = max(0.0, time.perf_counter() - started_at) if started_at else 0.0
        return {
            "active": bool(_SCAN_PROGRESS.get("active")),
            "finished": bool(_SCAN_PROGRESS.get("finished")),
            "failed": bool(_SCAN_PROGRESS.get("failed")),
            "elapsed": round(elapsed, 3),
            "messages": list(_SCAN_PROGRESS.get("messages", [])),
        }
