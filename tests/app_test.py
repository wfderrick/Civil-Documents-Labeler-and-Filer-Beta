from pathlib import Path

from app import _folder_project_and_section

"""Can't import app.py to test function inside because the paddlepaddle-gpu
version being used is not stored in pip yet. """

"""def _folder_project_and_section(output_folder: Path) -> tuple[str, str]:
    
    Extract the project code and section name from an output folder.

    Expected folder format:

        PROJECT.SECTION-Description

    Examples:
        12345.A-Drainage
            -> ("12345", "A")

        12345.B
            -> ("12345", "B")

        12345
            -> ("12345", "")

    Any text after the first '-' is ignored because it is treated as a
    descriptive suffix rather than part of the section identifier.
    
    name = output_folder.name.strip()

    if "." not in name:
        return name, ""

    project_code, section = name.split(".", 1)
    section = section.split("-", 1)[0]

    return project_code.strip(), section.strip()"""


def test_folder_project_and_section_empty():
    assert _folder_project_and_section(Path("")) == ("", "")


def test_folder_project_and_section_no_section():
    assert _folder_project_and_section(Path("CC6767")) == ("CC6767", "")


def test_folder_project_and_section_section_no_extra():
    assert _folder_project_and_section(Path("CC6767.67")) == ("CC6767", "67")


def test_folder_project_and_section_extra():
    assert _folder_project_and_section(Path("CC6767.67 - werwfwfwg")) == (
        "CC6767",
        "67"
    )


def test_folder_project_and_section_full():
    assert _folder_project_and_section(
        Path("C:/wderrickDocuments/CC6767.67 - werwfwfwg")
    ) == ("CC6767", "67")
