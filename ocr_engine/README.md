# Adaptive Multi-Engine OCR Library

A comprehensive OCR library that prioritizes accuracy through multiple engines and learns from user feedback to improve extraction for AI agent workflows.

## Features

- **Multi-Engine OCR**: Tesseract, EasyOCR, TrOCR, and Surya engines (4/5 working)
- **Visual Validation**: Computer vision-based quality assessment and cross-validation  
- **Pattern Detection**: Automatic discovery of tables, forms, key-value pairs, invoices, receipts
- **Adaptive Learning**: Learns document types and extraction patterns from user feedback
- **No Hard-Coded Types**: All document types are discovered and learned
- **AI-Optimized Output**: Structured JSON with confidence scores and metadata
- **Continuous Improvement**: Gets smarter with each correction

## Status: ✅ FULLY OPERATIONAL
All critical bugs resolved as of 2025-06-14. System ready for production use.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a document
from src.main_ocr import AdaptiveMultiEngineOCR

ocr = AdaptiveMultiEngineOCR()
result = ocr.process_document("document.pdf")

# Apply user feedback
ocr.apply_user_feedback(result['document_id'], {
    'field': 'total_amount',
    'correct_value': '$1,234.56',
    'extracted_value': '$1.234.56'
})
```

## Architecture

- **Engines**: Multiple OCR engines with hybrid consensus building
- **Visual Validation**: Computer vision quality assessment, layout analysis, bbox validation
- **Preprocessing**: Image enhancement, layout detection, deskewing
- **Pattern Detection**: Automatic discovery of tables, forms, key-value pairs, invoices, receipts
- **Learning System**: SQLite-based pattern storage and improvement
- **AI Integration**: Rich metadata and feedback mechanisms

## Engine Status

| Engine | Status | Notes |
|--------|---------|-------|
| Tesseract | ✅ Working | Standard OCR engine |
| EasyOCR | ✅ Working | Multi-language support |
| TrOCR | ✅ Working | Transformer-based |
| Surya | ✅ Working | Modern architecture (slow but accurate) |
| PaddleOCR | ❌ Disabled | Python 3.13+ incompatible |

## Recent Fixes (2025-06-14)
- ✅ EasyOCR parsing errors resolved
- ✅ Visual validation runtime errors fixed
- ✅ Surya OCR detection issues resolved
- ✅ Pattern detection fully operational
- ✅ Consensus building working correctly
- ✅ LayoutParser graceful fallback implemented

## Learning Capabilities

The system learns from:
- User corrections to field extractions
- Manual document type labels
- Successful extraction patterns
- Engine performance metrics

All learning is stored persistently and improves future extractions.