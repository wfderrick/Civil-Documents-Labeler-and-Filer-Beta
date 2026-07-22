from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

from metadata_extraction import load_config

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / ".review_state" / "documents.json"
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_STATE: dict[str, Any] = {"settings": {}, "documents": []}
_STATE_FILE_LOCK = threading.RLock()

T = TypeVar("T")
StateMutator = Callable[[dict[str, Any]], T]


def _default_state() -> dict[str, Any]:
    """Return a fresh default state so callers cannot mutate a shared object."""
    return copy.deepcopy(DEFAULT_STATE)


def _read_state_unlocked() -> dict[str, Any]:
    """Read state while the caller already holds ``_STATE_FILE_LOCK``."""
    if not STATE_FILE.exists():
        return _default_state()
    with STATE_FILE.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    state.setdefault("settings", {})
    state.setdefault("documents", [])
    return state


def _write_state_unlocked(state: dict[str, Any]) -> None:
    """Atomically write state while the caller holds ``_STATE_FILE_LOCK``."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = STATE_FILE.with_suffix(".json.tmp")
    with temporary_file.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)
        state_file.flush()
        os.fsync(state_file.fileno())
    temporary_file.replace(STATE_FILE)


def read_state() -> dict[str, Any]:
    """Return the latest persisted application settings and review documents."""
    with _STATE_FILE_LOCK:
        return _read_state_unlocked()


def write_state(state: dict[str, Any]) -> None:
    """Atomically replace the persisted state.

    Prefer the narrower mutation helpers below for request-time updates. They
    keep the read/modify/write sequence under one lock and therefore avoid
    overwriting edits made by another request.
    """
    with _STATE_FILE_LOCK:
        _write_state_unlocked(state)


def mutate_state(mutator: StateMutator[T]) -> tuple[dict[str, Any], T]:
    """Apply one mutation to the newest state and persist it atomically.

    The lock covers the complete read/modify/write transaction. The callback
    may mutate the supplied state and optionally return a useful result.
    """
    with _STATE_FILE_LOCK:
        state = _read_state_unlocked()
        result = mutator(state)
        _write_state_unlocked(state)
        return state, result


def replace_state(
    *, settings: dict[str, Any] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replace the scan state intentionally, such as when starting a new scan."""
    new_state = {
        "settings": copy.deepcopy(settings or {}),
        "documents": copy.deepcopy(documents or []),
    }
    write_state(new_state)
    return new_state


def append_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Append a document if its id is not already in the active review queue."""
    document_copy = copy.deepcopy(document)

    def append(latest_state: dict[str, Any]) -> list[dict[str, Any]]:
        documents = latest_state.setdefault("documents", [])
        document_id = document_copy.get("id")
        if not any(item.get("id") == document_id for item in documents):
            documents.append(document_copy)
        return copy.deepcopy(documents)

    _, documents = mutate_state(append)
    return documents


def remove_document(document_id: str) -> tuple[dict[str, Any], bool]:
    """Remove one document from the active review queue by id."""
    def remove(latest_state: dict[str, Any]) -> bool:
        documents = latest_state.setdefault("documents", [])
        original_count = len(documents)
        latest_state["documents"] = [
            item for item in documents if item.get("id") != document_id
        ]
        return len(latest_state["documents"]) != original_count

    return mutate_state(remove)


def clear_documents() -> dict[str, Any]:
    """Clear the active review queue while preserving the latest settings."""
    state, _ = mutate_state(lambda latest_state: latest_state.__setitem__("documents", []))
    return state


def update_settings(values: dict[str, Any]) -> dict[str, Any]:
    """Merge supplied values into the latest persisted settings."""
    values_copy = copy.deepcopy(values)

    def update(latest_state: dict[str, Any]) -> None:
        latest_state.setdefault("settings", {}).update(values_copy)

    state, _ = mutate_state(update)
    return state


def update_document(
    document_id: str,
    updater: Callable[[dict[str, Any], dict[str, Any]], T],
) -> tuple[dict[str, Any], T]:
    """Update one document against the latest state in a locked transaction. It
    has a child function named ``update()`` which takes in the current state and then
    uses the ``updater`` and ``document_id`` parameters to update the state and the
    document with document_id. It returns the result of calling 
    ``mutate_state()`` with the ``update()`` function as a parameter.

    ``updater`` receives ``(state, document)`` and may mutate either. A
    ``KeyError`` is raised when the document no longer exists.
    """
    def update(latest_state: dict[str, Any]) -> T:
        """Updates the current state and document matching the document id taken
        from the parent function ``update_document()``."""
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
    """Validate/create an output folder and persist it in the latest settings."""
    def update(latest_state: dict[str, Any]) -> Path:
        return update_output_folder_setting(latest_state, raw_value)

    return mutate_state(update)


def update_output_folder_setting(state: dict[str, Any], raw_value: str) -> Path:
    """Set a validated output-folder value on an in-memory state object."""
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("Output folder is required.")
    output_folder = Path(value).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    state.setdefault("settings", {})["output_folder"] = str(output_folder)
    return output_folder


def load_config_from_state(state: dict[str, Any]) -> dict[str, Any]:
    settings = state.get("settings", {})
    config_path = Path(settings.get("config_path") or DEFAULT_CONFIG_PATH).resolve()
    config = load_config(config_path if config_path.exists() else None)
    if settings.get("county"):
        config["default_county"] = settings["county"]
    return config
