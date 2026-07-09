# OCR Batch Filing Pipeline

This version is optimized for one related batch of PDFs, such as Site Plan, House Location, Wall Check, and Survey Notes.

## Key behavior

- OCRs every PDF in the input folder.
- Each document votes for shared lot and address.
- The winning lot/address is applied to every document in the batch.
- No combined batch text is used for metadata extraction.
- Project code is extracted from the selected output folder name.
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
