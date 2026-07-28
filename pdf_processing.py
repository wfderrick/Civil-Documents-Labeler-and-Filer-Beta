"""PDF rendering and text-layer utilities. These helpers rasterize pages for OCR, inspect existing text, build review PDFs, and handle temporary files used by the scan pipeline.

Maintenance notes:
    Keep this module focused on its current responsibility. When changing behavior,
    update the relevant tests and the project README so scan and review workflows
    remain understandable to future maintainers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

try:
    import pikepdf
except Exception:  # noqa: BLE001
    pikepdf = None


def metadata_keyword_text(document: dict[str, Any]) -> str:
    """The ``metadata_keyword_text()`` function returns a string of important
    keywords to be placed as metadata with the document given by the document
    parameter. Each of the items is separated by a semicolon.
    """
    metadata = document.get("metadata", {})
    custom_text = {
        "lot": metadata.get("lot", ""),
        "address": metadata.get("address", ""),
        "project_code": metadata.get("project_code", ""),
        "document_type": metadata.get("document_type", ""),
        "tax_map": metadata.get("tax_map", ""),
        "parcel": metadata.get("parcel", ""),
        "tax_id": metadata.get("tax_id", ""),
        "section": metadata.get("section", ""),
        "source_name": document.get("source_name", ""),
        "filed_at": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
    }
    return "; ".join(
        f"{key}={value}" for key, value in custom_text.items() if value
    )


def write_standard_pdf_metadata(
    pdf_path: Path, document: dict[str, Any]
) -> None:
    """The ``write_standard_pdf_metadata()`` function updates the Windows
    metadata which is visible in the file explorer for the pdf at the path
    specified by the pdf_path parameter. First the metadata is saved from the
    document parameter. Then using the ``Pymupdf`` library's ``open()`` and
    ``set_metadata()`` functions, title, subject, keywords, and creator are
    set to match the values gathered from the metadata field contained in the
    document parameter. The pdf is then saved with the new metadata.
    """
    metadata = document.get("metadata", {})
    with fitz.open(pdf_path) as pdf:
        pdf.set_metadata(
            {
                **pdf.metadata,  # type: ignore
                "title": f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}",
                "subject": metadata.get("address", ""),
                "keywords": metadata_keyword_text(document),
                "creator": "COA Barrett File Identifier and Sorter",
            }
        )
        pdf.saveIncr()


def write_xmp_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    """Write structured XMP metadata with a custom COA namespace."""
    if pikepdf is None:
        return

    metadata = document.get("metadata", {})
    namespace = "https://coabarrett.local/ns/ocr-file-sorter/1.0/"

    try:
        with pikepdf.Pdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            with pdf.open_metadata(set_pikepdf_as_editor=True) as meta:
                try:
                    meta.register_xml_namespace("coa", namespace)
                except Exception:  # noqa: BLE001, S110
                    pass

                title = f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}".strip(
                    " -"
                )
                if title:
                    meta["dc:title"] = title
                if metadata.get("address"):
                    meta["dc:description"] = metadata.get("address", "")
                meta["pdf:Keywords"] = metadata_keyword_text(document)

                custom_fields = {
                    "coa:Lot": metadata.get("lot", ""),
                    "coa:Address": metadata.get("address", ""),
                    "coa:ProjectCode": metadata.get("project_code", ""),
                    "coa:DocumentType": metadata.get("document_type", ""),
                    "coa:TaxMap": metadata.get("tax_map", ""),
                    "coa:Parcel": metadata.get("parcel", ""),
                    "coa:TaxID": metadata.get("tax_id", ""),
                    "coa:Section": metadata.get("section", ""),
                    "coa:JurisdictionCode": metadata.get(
                        "jurisdiction_code_mdp_field_jurscode", ""
                    ),
                    "coa:FinderOnlineLink": metadata.get(
                        "finder_online_link", ""
                    ),
                    "coa:LongitudeWGS84": metadata.get(
                        "mdp_longitude_mdp_field_digxcord_converted_to_wgs84",
                        "",
                    ),
                    "coa:LatitudeWGS84": metadata.get(
                        "mdp_latitude_mdp_field_digycord_converted_to_wgs84",
                        "",
                    ),
                    "coa:MappableLatitudeAndLongitude": metadata.get(
                        "mappable_latitude_and_longitude", ""
                    ),
                    "coa:LegalDescriptionLine1": metadata.get(
                        "legal_description_line_1_mdp_field_legal1_sdat_field_17",
                        "",
                    ),
                    "coa:LegalDescriptionLine2": metadata.get(
                        "legal_description_line_2_mdp_field_legal2_sdat_field_18",
                        "",
                    ),
                    "coa:DeedReference1Liber": metadata.get(
                        "deed_reference_1_liber_mdp_field_dr1liber_sdat_field_30",
                        "",
                    ),
                    "coa:DeedReference1Folio": metadata.get(
                        "deed_reference_1_folio_mdp_field_dr1folio_sdat_field_31",
                        "",
                    ),
                    "coa:SubdivisionCode": metadata.get(
                        "subdivision_code_mdp_field_subdivsn_sdat_field_37", ""
                    ),
                    "coa:Grid": metadata.get(
                        "grid_mdp_field_grid_sdat_field_43", ""
                    ),
                    "coa:ZoningCode": metadata.get(
                        "zoning_code_mdp_field_zoning_sdat_field_45", ""
                    ),
                    "coa:LandUseCode": metadata.get(
                        "land_use_code_mdp_field_lu_desclu_sdat_field_50", ""
                    ),
                    "coa:UtilitiesWater": metadata.get(
                        "property_factors_utilities_water_mdp_field_pfuw_sdat_field_63",
                        "",
                    ),
                    "coa:UtilitiesSewer": metadata.get(
                        "property_factors_utilities_sewer_mdp_field_pfus_sdat_field_64",
                        "",
                    ),
                    "coa:Waterfront": metadata.get(
                        "property_factors_location_waterfront_mdp_field_pflw_sdat_field_65",
                        "",
                    ),
                    "coa:StreetPaved": metadata.get(
                        "property_factors_street_paved_mdp_field_pfsp_sdat_field_67",
                        "",
                    ),
                    "coa:StreetUnpaved": metadata.get(
                        "property_factors_street_unpaved_mdp_field_pfsu_sdat_field_68",
                        "",
                    ),
                    "coa:OriginalFileName": document.get("source_name", ""),
                    "coa:FiledAt": datetime.now().isoformat(  # noqa: DTZ005
                        timespec="seconds"
                    ),
                    "coa:Application": "COA Barrett File Identifier and Sorter",
                }
                for key, value in custom_fields.items():
                    if value:
                        meta[key] = str(value)
            pdf.save(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not write XMP metadata to {pdf_path}: {exc}")


def write_pdf_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    """The ``write_pdf_metadata()`` function updates the windows metadata and
    adds custom XML metadata to the file located at the pdf_path parameter using
    the document parameter as the source of the metadata."""

    try:
        write_standard_pdf_metadata(pdf_path, document)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not write standard PDF metadata to {pdf_path}: {exc}")

    write_xmp_metadata(pdf_path, document)
