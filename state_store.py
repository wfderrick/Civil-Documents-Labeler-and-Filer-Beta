from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from metadata_extraction import load_config

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / ".review_state" / "documents.json"
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_STATE: dict[str, Any] = {"settings": {}, "documents": []}


def read_state() -> dict[str, Any]:
    """The read_state() function returns all of the current settings and
    document metadata stored in the documents.json file which is in the
    .review_state folder in the project directory. If that file has not
    been created yet it returns a default dictionary with empty settings
    and documents."""
    if not STATE_FILE.exists():
        return dict(DEFAULT_STATE)
    with STATE_FILE.open("r", encoding="utf-8") as state_file:
        return json.load(state_file)


def write_state(state: dict[str, Any]) -> None:
    """The write_state() function updates the documents.json file with current
    settings and document metadata. It begins by ensuring the parent directory
    exists for the STATE_FILE. Then it opens the state file to either be created
    if it doesn't exist or overwritten if it does. Finally dump() writes itself
    onto the file."""
    print(str(STATE_FILE.parent))
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)


def update_output_folder_setting(state: dict[str, Any], raw_value: str) -> Path:
    """The update_output_folder_setting() function sets the output folder in the
    parameter to the new output folder given in the raw_value parameter after
    trimming and validating it."""
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
    return load_config(config_path if config_path.exists() else None)
