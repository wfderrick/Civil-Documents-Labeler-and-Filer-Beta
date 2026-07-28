"""Regression tests for deriving project code and section from folder names.

Mass Scan uses the parent folder name as fallback metadata. These examples protect
accepted forms such as CC6767, CC6767.67, and names with a descriptive suffix."""

from pathlib import Path

from app import _folder_project_and_section


def test_folder_project_and_section_empty():
    """Verify that an empty folder name yields no fallback project or section.

    This keeps missing folder context from being converted into misleading metadata."""
    assert _folder_project_and_section(Path("")) == ("", "")


def test_folder_project_and_section_no_section():
    """Verify parsing of a folder that contains only a project code.

    The full name becomes the project code and section remains blank."""
    assert _folder_project_and_section(Path("CC6767")) == ("CC6767", "")


def test_folder_project_and_section_section_no_extra():
    """Verify parsing of the compact `PROJECT.SECTION` folder convention.

    The period separates CC6767 from section 67 without requiring a descriptive suffix."""
    assert _folder_project_and_section(Path("CC6767.67")) == ("CC6767", "67")


def test_folder_project_and_section_extra():
    """Verify that descriptive text after ` - ` does not become project metadata.

    Only the project and section prefix should influence generated filenames and shared
    batch values."""
    assert _folder_project_and_section(Path("CC6767.67 - werwfwfwg")) == (
        "CC6767",
        "67"
    )


def test_folder_project_and_section_full():
    """Verify the parser uses the final folder name rather than the complete path.

    Parent directories must not contaminate a project code when Mass Scan derives fallback
    metadata from a nested Windows folder."""
    assert _folder_project_and_section(
        Path("C:/wderrickDocuments/CC6767.67 - werwfwfwg")
    ) == ("CC6767", "67")
