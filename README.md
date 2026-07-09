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

## Run

```bash
python -m pip install -r requirements.txt
python app.py
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
