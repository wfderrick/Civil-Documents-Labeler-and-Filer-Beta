from pathlib import Path
from app import _folder_project_and_section


def test_folder_project_and_section_empty():
    assert _folder_project_and_section(Path("")) == ("", "")


def test_folder_project_and_section_no_section():
    assert _folder_project_and_section(Path("CC6767")) == ("CC6767", "")


def test_folder_project_and_section_section_no_extra():
    assert _folder_project_and_section(Path("CC6767.67")) == ("CC6767", "67")


def test_folder_project_and_section_extra():
    assert _folder_project_and_section(Path("CC6767.67 - werwfwfwg")) == (
        "CC6767",
        "67",
    )


def test_folder_project_and_section_full():
    assert _folder_project_and_section(
        Path("C:/wderrickDocuments/CC6767.67 - werwfwfwg")
    ) == ("CC6767", "67")
