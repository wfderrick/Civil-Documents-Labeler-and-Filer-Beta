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

Put handwritten field-note PDFs in `field_notes`. Put everything else in `not_field_notes`, including Site Plans, House Locations, Wall Checks, Replats, plats, permits, letters, and random PDFs.

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

