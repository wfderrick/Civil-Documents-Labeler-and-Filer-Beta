# OCR Batch Filing Pipeline

This version is optimized for one related batch of PDFs, such as Site Plan, House Location, Wall Check, and Field Notes.

## Key behavior

- OCRs every PDF in the input folder.
- Each document votes for shared lot and address.
- The winning lot/address is applied to every document in the batch.
- No combined batch text is used for metadata extraction.
- Project code can be entered manually; if left blank, it is extracted from the selected output folder name.
- Metadata now includes lot, address, project code, document type, tax map, parcel, and tax ID.
- SDAT lookup uses county + tax ID, tax map, parcel, and lot with flexible leading-zero matching.
- Manual changes to lot or address on one document update every document in the batch.
- Filing convention remains: folder `Lot # - Address`, file `Document Type - Lot #.pdf`.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m app run
```

Open: http://127.0.0.1:5055

## Config

Edit `config.json` for your county, ignored survey-company addresses, project-code patterns, and OCR extraction patterns.

## Visual Field Notes Classifier

This version uses a binary visual classifier:

```text
field_notes
not_field_notes
```

It runs only as a safety net after OCR metadata extraction, when a batch contains duplicate plan document types. It renders the PDF visually and does not rely on OCR text, which is important because handwritten field notes often OCR poorly.

Training workflow:

```powershell
python train_visual_classifier.py training_data --output visual_field_notes_classifier.joblib
```

Use this folder structure:

```text
training_data/
  field_notes/
  not_field_notes/
```

Put handwritten field-note PDFs in `field_notes`. Put everything else in `not_field_notes`, including Site Plans, House Locations, Wall Checks, Plat/Replats, plats, Construction Permits, letters, and random PDFs.

Recommended minimum:

```text
field_notes:      20–50 PDFs
not_field_notes:  20–50 PDFs
```

After training, keep `visual_field_notes_classifier.joblib` beside `app.py`, then restart the Flask app.

## SDAT lookup-only records

The pipeline recognizes printed Maryland SDAT Real Property Data Search pages by their header and account-identification anchors. These pages are treated as lookup-only documents:

- Only the district and account number are extracted.
- The resulting Tax ID is given priority over OCR-voted map/parcel/address values.
- SDAT fills the shared lot, address, tax map, parcel, Tax ID, and section.
- Lookup-only records are hidden from the normal review queue and are not filed.
- After all permanent documents file successfully, lookup-only source PDFs are moved to the Windows Recycle Bin (or deleted if Recycle Bin support is unavailable).

Property synchronization priority is Tax ID first, then address. Editing Tax ID or address refreshes all shared property fields except project code and document type.


## Document type defaults and lookup records

- Documents whose type cannot be identified default to **Field Notes**.
- SDAT printouts are labeled **Lookup Only** and remain visible in the review queue.
- Lookup Only documents are not filed; they are removed after all permanent documents file successfully.


## Module architecture (Version 1.9 refactor)

- `app.py`: Flask routes and batch workflow orchestration.
- `pipeline.py`: batch metadata voting, batch SDAT synchronization, and compatibility exports.
- `metadata_extraction.py`: OCR-text interpretation, regex/fuzzy matching, addresses, identifiers, and metadata models.
- `ocr_service.py`: PaddleOCR model setup, GPU/CPU selection, PDF rendering, OCR workers, and layout extraction.
- `sdat.py`: Maryland SDAT requests, record filtering, address/Tax ID lookup, and metadata enrichment.
- `pdf_processing.py`: searchable text layer and PDF/XMP metadata writing.
- `document_service.py`: document naming, status, updates, and filing.
- `state_store.py`: persistent application state and configuration loading.
- `tracker.py`: batch filing CSV tracker.
- `scan_status.py`: thread-safe scan progress state.
- `tax_id_utils.py`: Tax ID normalization, validation, parsing, formatting, and comparison.
- `visual_classifier.py`: visual Field Notes classification and duplicate-type correction.

The batch workflow, shared metadata voting, lookup-only behavior, SDAT synchronization, and File All behavior remain in place.

## Mass Scan Tax ID isolation

Mass Scan processes every PDF independently. SDAT Tax IDs are accepted from an
explicit Tax ID lookup, or from an address lookup only when that address maps to
one unambiguous SDAT record. Ambiguous address results are left blank for review
instead of copying the first returned Tax ID into multiple documents. Map/parcel
fallback remains available in Batch mode, but is intentionally disabled for Mass
Scan because weak identifiers can match unrelated parcels.

## Developer Architecture Guide

### Request and data flow

1. The browser loads `templates/index.html`, `static/styles.css`, and `static/app.js`.
2. `static/app.js` requests the current review state and renders the document queue, metadata form, PDF viewer, settings, and scan progress.
3. Flask routes in `app.py` validate each request and delegate work to the service modules rather than directly implementing OCR or file-system logic.
4. `pipeline.py` coordinates PDF discovery, page rendering, OCR, metadata extraction, SDAT enrichment, and suggested filename creation.
5. `state_store.py` serializes all state mutations through a lock and writes the resulting JSON state to `.review_state/documents.json`.
6. User edits are applied through `document_service.py`, which also controls whether shared property fields are synchronized in Batch mode or remain isolated in Mass Scan mode.
7. Filing operations create, rename, or move output files and update the review state so completed records leave the active queue.

### Module responsibilities

| Module | Primary responsibility |
| --- | --- |
| `app.py` | Flask application, API endpoints, background scan startup, and file responses |
| `pipeline.py` | End-to-end Batch and Mass Scan orchestration |
| `ocr_service.py` | PaddleOCR engine caching, OCR execution, and result normalization |
| `pdf_processing.py` | PDF page rendering and review-PDF/text-layer operations |
| `metadata_extraction.py` | OCR-text parsing, document classification, title-block and property extraction |
| `sdat.py` | Maryland SDAT requests, candidate ranking, and property-record normalization |
| `document_service.py` | Metadata merging, edit application, synchronization, and suggested filenames |
| `state_store.py` | Thread-safe JSON persistence and centralized state mutation |
| `scan_status.py` | Shared scan-progress snapshot used by workers and the UI |
| `tax_id_utils.py` | Tax ID normalization, validation, formatting, and comparison |
| `visual_classifier.py` | Optional image-based Field Notes classification |
| `tracker.py` | Tracking of output file changes |
| `static/app.js` | Browser state, rendering, validation, polling, PDF viewing, and API calls |
| `static/styles.css` | Layout, visual hierarchy, validation states, and responsive behavior |

### Batch mode versus Mass Scan mode

Batch mode assumes the selected PDFs may belong to one project or property. It can vote on and synchronize shared metadata across the group. This improves consistency when related plan sheets provide complementary information.

Mass Scan mode treats every PDF as independent. Each file is scanned, enriched, and published separately. Property metadata such as Tax ID must not propagate from one Mass Scan document to another. When modifying this workflow, preserve the one-document scope of metadata voting and SDAT enrichment, and run `test_mass_scan_tax_id_isolation.py`.

### Document classification

Document classification is implemented in `metadata_extraction.py`. Rules should be ordered from the most specific phrase to the most general phrase. A broad term such as `plat` must not override a more specific title such as `site plan and easement plat`.

When adding or changing a type:

1. Add or update the corresponding normalized phrase or regular expression.
2. Confirm precedence against overlapping document names.
3. Update any display-name versus Windows-safe filename mapping.
4. Add a regression test using representative OCR text.
5. Run the full test suite before packaging a release.

### SDAT and Tax ID safety

SDAT results are external records and may include ambiguous or partial matches. The application should only accept a Tax ID when the available property evidence identifies a sufficiently strong record. Mass Scan must not reuse a previous document's lookup result when the current document has no unique match.

Before changing SDAT behavior, review the normalization and ranking logic in `sdat.py`, the Tax ID helpers in `tax_id_utils.py`, and the edit/enrichment flow in `document_service.py`. Preserve blank values for manual review when confidence is insufficient.

### State-management rules

All persistent document changes should pass through `state_store.py`. Do not independently read the state, mutate an old copy, and write it later; that pattern can overwrite edits made while a scan is running. Use `mutate_state()` or one of the narrower helper functions so the latest state is loaded and changed while the lock is held.

Keep state records JSON-serializable. When introducing a field, provide a safe default for existing state files and update both Python and JavaScript consumers.

### Front-end maintenance

Element IDs in `templates/index.html` are referenced directly by `static/app.js` and therefore act as an interface contract. Renaming an ID requires updating every matching JavaScript lookup.

Validation should remain centralized in the JavaScript validation helpers. Queue colors, missing-field highlighting, and action availability should all derive from the same validation result so the interface does not give conflicting signals.

### Testing

Run the project tests from the application directory:

```bash
pytest -q
```

The current regression tests cover Mass Scan Tax ID isolation, Site Plan/Easement Plat precedence, and visual classifier caching behavior. Add focused tests whenever a production document exposes a new OCR, classification, state, or lookup edge case.

A syntax-only check can be run with:

```bash
python -m compileall -q .
```

### Commenting conventions

Comments in this project explain intent, data ownership, safety constraints, and non-obvious edge cases. They should not merely repeat the next line of code. Function docstrings describe inputs, outputs, side effects, and workflow context. When behavior changes, update the associated comments and tests in the same change so the documentation remains trustworthy.
