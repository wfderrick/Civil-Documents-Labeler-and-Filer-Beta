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
    """Metadata keyword text.
    
    Args:
        document: Input used by this operation.
    
    Returns:
        The computed result for the caller. See the function body and type hints for the exact shape.
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
    return "; ".join(f"{key}={value}" for key, value in custom_text.items() if value)


def _ocr_item_pdf_rect(
    item: dict[str, Any],
    x_scale: float,
    y_scale: float,
) -> fitz.Rect | None:
    """Convert one OCR item's pixel geometry to a PDF-point rectangle."""
    raw = item.get("bbox") or item.get("polygon")
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()

    try:
        # Preferred normalized bbox format: [x0, y0, x1, y1].
        if len(raw) == 4 and all(isinstance(value, (int, float)) for value in raw):
            x0, y0, x1, y1 = [float(value) for value in raw]
        else:
            # Compatibility with four-point PaddleOCR polygons.
            points = [
                point
                for point in raw
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if not points:
                return None
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError):
        return None

    rect = fitz.Rect(x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale)
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def add_paddle_searchable_text_layer(pdf_path: Path, document: dict[str, Any]) -> None:
    """Add selectable, invisible text using the stored PaddleOCR geometry.

    PaddleOCR coordinates are measured in rendered-image pixels. They are
    mapped back to PDF points using the image dimensions saved during OCR.
    The text uses PDF render mode 3, so it remains invisible while still being
    searchable and selectable in browsers and PDF readers.
    """
    ocr_pages = document.get("ocr_pages") or []
    if not ocr_pages:
        return

    font = fitz.Font("helv")
    inserted = 0

    with fitz.open(pdf_path) as pdf:
        for page_data in ocr_pages:
            try:
                page_index = int(page_data.get("page_index", 0))
                page = pdf[page_index]
                image_width = float(page_data.get("image_width") or 0)
                image_height = float(page_data.get("image_height") or 0)
            except (IndexError, TypeError, ValueError):
                continue

            if image_width <= 0 or image_height <= 0:
                continue

            x_scale = float(page.rect.width) / image_width
            y_scale = float(page.rect.height) / image_height

            for item in page_data.get("items", []):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                rect = _ocr_item_pdf_rect(item, x_scale, y_scale)
                if rect is None:
                    continue

                # Fit invisible text to the OCR rectangle.  The baseline is
                # derived from Helvetica's ascender / descender rather than
                # being placed at the bottom of the box, which keeps selection
                # geometry aligned with the detected line.
                natural_width = max(font.text_length(text, fontsize=1), 0.01)
                height_size = rect.height / max(font.ascender - font.descender, 0.01)
                width_size = (rect.width * 0.98) / natural_width
                font_size = min(max(min(height_size, width_size), 1.0), 72.0)
                baseline_y = rect.y0 + (font.ascender * font_size)
                baseline = fitz.Point(rect.x0, baseline_y)

                try:
                    page.insert_text(
                        baseline,
                        text,
                        fontsize=font_size,
                        fontname="helv",
                        render_mode=3,
                        overlay=True,
                    )
                    inserted += 1
                except Exception:  # noqa: BLE001, S112
                    continue

        if inserted:
            # Incremental save is fast and preserves the scanned page content.
            pdf.saveIncr()


def write_standard_pdf_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    """Write standard pdf metadata.
    
    Args:
        pdf_path: Input used by this operation.
        document: Input used by this operation.
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
                    "coa:JurisdictionCode": metadata.get("jurisdiction_code_mdp_field_jurscode", ""),
                    "coa:FinderOnlineLink": metadata.get("finder_online_link", ""),
                    "coa:LongitudeWGS84": metadata.get("mdp_longitude_mdp_field_digxcord_converted_to_wgs84", ""),
                    "coa:LatitudeWGS84": metadata.get("mdp_latitude_mdp_field_digycord_converted_to_wgs84", ""),
                    "coa:MappableLatitudeAndLongitude": metadata.get("mappable_latitude_and_longitude", ""),
                    "coa:LegalDescriptionLine1": metadata.get("legal_description_line_1_mdp_field_legal1_sdat_field_17", ""),
                    "coa:LegalDescriptionLine2": metadata.get("legal_description_line_2_mdp_field_legal2_sdat_field_18", ""),
                    "coa:DeedReference1Liber": metadata.get("deed_reference_1_liber_mdp_field_dr1liber_sdat_field_30", ""),
                    "coa:DeedReference1Folio": metadata.get("deed_reference_1_folio_mdp_field_dr1folio_sdat_field_31", ""),
                    "coa:SubdivisionCode": metadata.get("subdivision_code_mdp_field_subdivsn_sdat_field_37", ""),
                    "coa:Grid": metadata.get("grid_mdp_field_grid_sdat_field_43", ""),
                    "coa:ZoningCode": metadata.get("zoning_code_mdp_field_zoning_sdat_field_45", ""),
                    "coa:LandUseCode": metadata.get("land_use_code_mdp_field_lu_desclu_sdat_field_50", ""),
                    "coa:UtilitiesWater": metadata.get("property_factors_utilities_water_mdp_field_pfuw_sdat_field_63", ""),
                    "coa:UtilitiesSewer": metadata.get("property_factors_utilities_sewer_mdp_field_pfus_sdat_field_64", ""),
                    "coa:Waterfront": metadata.get("property_factors_location_waterfront_mdp_field_pflw_sdat_field_65", ""),
                    "coa:StreetPaved": metadata.get("property_factors_street_paved_mdp_field_pfsp_sdat_field_67", ""),
                    "coa:StreetUnpaved": metadata.get("property_factors_street_unpaved_mdp_field_pfsu_sdat_field_68", ""),
                    "coa:OriginalFileName": document.get("source_name", ""),
                    "coa:FiledAt": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
                    "coa:Application": "COA Barrett File Identifier and Sorter",
                }
                for key, value in custom_fields.items():
                    if value:
                        meta[key] = str(value)
            pdf.save(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not write XMP metadata to {pdf_path}: {exc}")


def write_pdf_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    """Write standard metadata, structured XMP, and a PaddleOCR text layer."""
    try:
        add_paddle_searchable_text_layer(pdf_path, document)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not add PaddleOCR searchable text layer to {pdf_path}: {exc}")

    try:
        write_standard_pdf_metadata(pdf_path, document)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not write standard PDF metadata to {pdf_path}: {exc}")

    write_xmp_metadata(pdf_path, document)
