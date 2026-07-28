"""Persistent repository for settings and the live review queue.

The browser and scan process share ``.review_state/documents.json``.
Mass Scan may append a completed PDF at the same time the user edits or
files an earlier one. A simple "read, change, write" sequence in separate
functions could lose one of those changes.

This module solves that problem with a re-entrant lock and atomic writes:
    1. Acquire the lock.
    2. Read the newest JSON from disk.
    3. Apply one caller-supplied mutation.
    4. Write a temporary file and replace the real file in one operation.

Callers should prefer the narrow helpers such as ``append_document`` or
``update_document`` instead of editing the state file directly."""

from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from metadata_extraction import load_config

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / ".review_state" / "documents.json"
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_STATE: dict[str, Any] = {"settings": {}, "documents": []}
_STATE_FILE_LOCK = threading.RLock()

T = TypeVar("T")
StateMutator = Callable[[dict[str, Any]], T]


def _default_state() -> dict[str, Any]:
    """Create a fresh empty state object for a first run or unreadable state file.
    
    A deep copy is required because nested ``settings`` and ``documents`` containers must not be
    shared between callers; mutating one request's default should not alter the module constant."""
    return copy.deepcopy(DEFAULT_STATE)


def _read_state_unlocked() -> dict[str, Any]:
    """Read and normalize the state file while the caller already owns the lock.
    
    Missing files return a fresh default. Existing JSON is checked so ``settings`` is a dictionary
    and ``documents`` is a list, preventing malformed or older state shapes from causing failures
    throughout the UI. This private helper never acquires the lock itself."""
    if not STATE_FILE.exists():
        return _default_state()
    with STATE_FILE.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    state.setdefault("settings", {})
    state.setdefault("documents", [])
    return state


def _write_state_unlocked(state: dict[str, Any]) -> None:
    """Persist state safely through a temporary file and atomic replacement.
    
    JSON is written completely to a sibling temporary path, flushed to disk, and then moved over
    the real state file with ``os.replace``. Readers therefore see either the previous complete
    JSON or the new complete JSON, never a half-written file. The caller must hold the lock."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = STATE_FILE.with_suffix(".json.tmp")
    with temporary_file.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)
        state_file.flush()
        os.fsync(state_file.fileno())
    temporary_file.replace(STATE_FILE)


def read_state() -> dict[str, Any]:
    """Return a detached snapshot of the latest persisted settings and documents.
    
    The lock protects the disk read, and a deep copy prevents the caller from accidentally
    modifying in-memory data that another operation assumes is stable. Use a mutation helper to
    make persistent changes."""
    with _STATE_FILE_LOCK:
        return _read_state_unlocked()


def write_state(state: dict[str, Any]) -> None:
    """Replace the entire state under the repository lock.
    
    This broad operation is mainly useful for deliberate full replacements. Request-time code
    should prefer ``mutate_state`` and its narrower wrappers, which re-read the newest state before
    changing it and are less likely to overwrite concurrent Mass Scan or browser updates."""
    with _STATE_FILE_LOCK:
        _write_state_unlocked(state)


def mutate_state(mutator: StateMutator[T]) -> tuple[dict[str, Any], T]:
    """Execute one read-modify-write transaction against the newest state.
    
    The lock remains held while the current JSON is read, the caller's callback changes it, and
    the updated JSON is atomically written. The function returns both a deep-copied final state and
    the callback's result. This is the central concurrency safeguard for the review queue."""
    # The callback runs while the lock is held. This is deliberate: releasing
    # between read and write would allow a second request to create a lost update.
    with _STATE_FILE_LOCK:
        state = _read_state_unlocked()
        result = mutator(state)
        _write_state_unlocked(state)
        return state, result


def replace_state(
    *, settings: dict[str, Any] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Intentionally replace settings and the document queue at a scan boundary.
    
    Batch scan calls this after producing its complete set. Mass scan calls it first with an empty
    queue, then appends documents incrementally. Arguments are deep-copied so later caller changes
    cannot silently alter what was persisted."""
    new_state = {
        "settings": copy.deepcopy(settings or {}),
        "documents": copy.deepcopy(documents or []),
    }
    write_state(new_state)
    return new_state


def append_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Add one completed scan result without duplicating an existing record ID.
    
    The duplicate check and append happen inside one locked mutation. This matters during Mass
    Scan because the browser may be reading, editing, or filing records while new PDFs finish OCR."""
    document_copy = copy.deepcopy(document)

    def append(latest_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Perform the duplicate-ID check and append while ``mutate_state`` owns the lock.
        
        Returning ``False`` tells the caller the record was already present; returning ``True`` means
        a deep-copied document was added to the live queue."""
        documents = latest_state.setdefault("documents", [])
        document_id = document_copy.get("id")
        if not any(item.get("id") == document_id for item in documents):
            documents.append(document_copy)
        return copy.deepcopy(documents)

    _, documents = mutate_state(append)
    return documents


def remove_document(document_id: str) -> tuple[dict[str, Any], bool]:
    """Remove one review record by ID without disturbing documents added concurrently.
    
    Single-document filing uses this after the PDF succeeds. The locked mutation re-reads the
    newest queue, filters only the target ID, and reports whether a record was actually removed."""
    def remove(latest_state: dict[str, Any]) -> bool:
        """Filter the live document list and report whether its length changed.
        
        This nested function runs inside the repository transaction, so the list includes any Mass
        Scan results appended immediately before the removal began."""
        documents = latest_state.setdefault("documents", [])
        original_count = len(documents)
        latest_state["documents"] = [
            item for item in documents if item.get("id") != document_id
        ]
        return len(latest_state["documents"]) != original_count

    return mutate_state(remove)


def clear_documents() -> dict[str, Any]:
    """Empty the active review queue while preserving the latest settings.
    
    File-All calls this only after permanent PDFs have been filed successfully. Configuration and
    folder choices remain available for the next scan."""
    state, _ = mutate_state(lambda latest_state: latest_state.__setitem__("documents", []))
    return state


def update_settings(values: dict[str, Any]) -> dict[str, Any]:
    """Merge selected setting values into the newest persisted settings dictionary.
    
    Existing keys not mentioned by the caller are preserved. The merge happens under the state
    transaction lock so it cannot erase documents appended by a concurrent scan."""
    values_copy = copy.deepcopy(values)

    def update(latest_state: dict[str, Any]) -> None:
        """Apply a shallow settings merge inside the locked state transaction.
        
        A shallow merge is appropriate because the current settings are simple scalar values rather
        than nested configuration trees."""
        latest_state.setdefault("settings", {}).update(values_copy)

    state, _ = mutate_state(update)
    return state


def update_document(
    document_id: str,
    updater: Callable[[dict[str, Any], dict[str, Any]], T],
) -> tuple[dict[str, Any], T]:
    """Find one live document and update it atomically with caller-supplied business logic.
    
    The callback receives both the newest full state and the matching mutable document. That lets
    ``apply_document_update`` synchronize batch peers while still completing one disk transaction.
    A missing ID raises ``KeyError`` so the Flask route can return a 404."""
    def update(latest_state: dict[str, Any]) -> T:
        """Locate the requested record and run the supplied updater under the file lock.
        
        The updater may edit the selected record and other records in the same state. Its return value
        is passed back through ``mutate_state`` for the API response."""
        document = next(
            (
                item
                for item in latest_state.setdefault("documents", [])
                if item.get("id") == document_id
            ),
            None,
        )
        if document is None:
            raise KeyError(document_id)
        return updater(latest_state, document)

    return mutate_state(update)


def update_output_folder(raw_value: str) -> tuple[dict[str, Any], Path]:
    """Validate/create an output folder and save it in the latest settings atomically.
    
    This wrapper packages ``update_output_folder_setting`` as a repository mutation, returning
    both the updated state and resolved ``Path`` to the Flask route."""
    def update(latest_state: dict[str, Any]) -> Path:
        """Apply output-folder validation while the state repository lock is held.
        
        The nested callback changes only the in-memory state supplied by ``mutate_state``; the outer
        transaction performs the atomic disk write afterward."""
        return update_output_folder_setting(latest_state, raw_value)

    return mutate_state(update)


def update_output_folder_setting(state: dict[str, Any], raw_value: str) -> Path:
    """Resolve, create, and store an output directory on an in-memory state object.
    
    Blank values are rejected because normal filing requires a destination. Parent folders are
    created as needed, and the absolute path is stored as text for JSON serialization. This helper
    does not lock or write the state file by itself."""
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("Output folder is required.")
    output_folder = Path(value).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    state.setdefault("settings", {})["output_folder"] = str(output_folder)
    return output_folder


def load_config_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Load the effective extraction configuration associated with a saved review state.
    
    Browser edits that trigger SDAT refresh must use the same config path and county as the scan
    that created the documents. The function resolves that path, falls back to built-in defaults
    when absent, and overlays the saved county for the new lookup."""
    settings = state.get("settings", {})
    config_path = Path(settings.get("config_path") or DEFAULT_CONFIG_PATH).resolve()
    config = load_config(config_path if config_path.exists() else None)
    if settings.get("county"):
        config["default_county"] = settings["county"]
    return config
