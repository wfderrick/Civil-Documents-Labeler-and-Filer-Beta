"""Thread-safe progress channel between a long scan and the browser.

OCR may take minutes, while an HTTP request normally returns only when
its work is complete. The application stores progress messages and stage
timings in one protected dictionary. ``static/app.js`` polls the
``/api/scan-progress`` route and redraws the progress panel from snapshots
produced here.

Every public function acquires the same lock before reading or changing
the dictionary. That prevents a browser poll from seeing a half-updated
message list while the scan thread is writing to it."""

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
    """Initialize a clean progress record at the start of a scan request.
    
    The function marks the scan active, records the start time, clears old messages and
    timings, and resets finished/failed flags under the shared lock. The browser can begin
    polling immediately after this call."""
    with _SCAN_PROGRESS_LOCK:
        _SCAN_PROGRESS.update(
            {
                "active": True,
                "finished": False,
                "failed": False,
                "started_at": time.perf_counter(),
                "messages": [],
                "timings": {},
            }
        )


def add_scan_progress(message: str) -> None:
    """Append one timestamped human-readable message to the active scan.
    
    Messages describe document/page progress and performance stages. The list is updated under
    the lock so a simultaneous browser snapshot cannot observe a partially appended entry."""
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



def set_scan_timing(name: str, elapsed: float) -> None:
    """Store the final elapsed time for one named performance stage.
    
    Repeated calls with the same name replace the previous value. This is appropriate for
    whole-scan stages such as configuration load, OCR engine startup, or total scan time."""
    key = str(name or "").strip()
    if not key:
        return
    with _SCAN_PROGRESS_LOCK:
        _SCAN_PROGRESS.setdefault("timings", {})[key] = round(max(0.0, float(elapsed)), 3)


def add_scan_timing(name: str, elapsed: float) -> None:
    """Accumulate elapsed time into a named performance total.
    
    This variant is useful when the same stage occurs repeatedly—for example, adding several
    page-level operations into one category—without requiring the caller to read the current
    value first."""
    key = str(name or "").strip()
    if not key:
        return
    with _SCAN_PROGRESS_LOCK:
        timings = _SCAN_PROGRESS.setdefault("timings", {})
        timings[key] = round(float(timings.get(key, 0.0)) + max(0.0, float(elapsed)), 3)


def finish_scan_progress(*, failed: bool = False, message: str = "") -> None:
    """Mark the current scan complete or failed and optionally append a final message.
    
    The active flag is cleared so the browser stops treating the timer as live. ``failed``
    distinguishes a successful completion from an exception while preserving all messages and
    timings for inspection."""
    if message:
        add_scan_progress(message)
    with _SCAN_PROGRESS_LOCK:
        _SCAN_PROGRESS["active"] = False
        _SCAN_PROGRESS["finished"] = True
        _SCAN_PROGRESS["failed"] = failed


def scan_progress_snapshot() -> dict[str, Any]:
    """Return a detached JSON-safe view of the current progress record.
    
    A deep copy is made while holding the lock, then elapsed time is calculated from the saved
    start time. The browser receives a consistent snapshot and cannot mutate the server's live
    dictionary through a shared reference."""
    with _SCAN_PROGRESS_LOCK:
        started_at = float(_SCAN_PROGRESS.get("started_at") or 0.0)
        elapsed = max(0.0, time.perf_counter() - started_at) if started_at else 0.0
        return {
            "active": bool(_SCAN_PROGRESS.get("active")),
            "finished": bool(_SCAN_PROGRESS.get("finished")),
            "failed": bool(_SCAN_PROGRESS.get("failed")),
            "elapsed": round(elapsed, 3),
            "messages": list(_SCAN_PROGRESS.get("messages", [])),
            "timings": dict(_SCAN_PROGRESS.get("timings", {})),
        }
