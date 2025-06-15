# Sample Documents Folder

Place your PDF and image files here for testing BetterOCR.

## Supported Formats
- PDF files (.pdf)
- Images (.png, .jpg, .jpeg, .tiff, .bmp)

## Organization
You can organize documents in subfolders:
- `samples/real_estate/` - Real estate documents
- `samples/tax_docs/` - Tax documents
- `samples/handwritten/` - Handwritten samples
- `samples/tables/` - Documents with complex tables

## Testing
To process a document:
```bash
# Single file
python ../betterocr.py document.pdf

# With debug output
python ../betterocr.py document.pdf --debug

# Save to specific output
python ../betterocr.py document.pdf --output results/document_ocr.json
```

## Batch Processing
To process all documents in a folder:
```bash
# Process all PDFs (coming soon)
python ../batch_process.py *.pdf
```