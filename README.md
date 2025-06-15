# BetterOCR - Multi-Engine OCR with Computer Vision

A hybrid OCR system that combines multiple OCR engines with computer vision for maximum accuracy.

## Installation

### Automatic Installation (Recommended)
```bash
# Install all dependencies system-wide
./install.sh
```

### Manual Installation
```bash
# Install system dependencies (macOS)
brew install tesseract tesseract-lang poppler libmagic

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-all poppler-utils libmagic1

# Install Python packages
pip install -r requirements.txt
```

## Quick Start

```bash
# Process a single document
python betterocr.py sample.pdf

# Process with debug info
python betterocr.py sample.pdf --debug
```

## Features

- **Multiple OCR Engines**: Runs Tesseract, EasyOCR, and PaddleOCR in parallel
- **Computer Vision Validation**: Uses CV to verify and enhance OCR results
- **Structured Output**: Generates AI-friendly JSON with confidence scores
- **Flexible Design**: Easy to add new engines or modify behavior

## Sample Usage

Drop your PDFs in the `samples/` folder and run:
```bash
python betterocr.py samples/your_document.pdf
```

The system will:
1. Run all available OCR engines
2. Apply CV validation
3. Generate a comprehensive JSON output with all variations and confidence scores