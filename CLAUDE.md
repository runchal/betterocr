# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: BetterOCR - Multi-Engine OCR with Computer Vision Validation

### Overview
BetterOCR is a sophisticated hybrid document processing system that combines multiple OCR engines with computer vision validation and adaptive learning to achieve maximum accuracy in text extraction and document understanding. The system prioritizes accuracy over speed and includes enterprise-grade features like real-time performance monitoring, document classification, and user feedback integration.

**Status**: ✅ **FULLY FUNCTIONAL** - All critical bugs resolved (as of June 14, 2025)

### Architecture

This project contains **two implementations**:
1. **Basic Implementation** (root directory) - Simple interface demonstrating core concepts
2. **Advanced Implementation** (`/ocr_engine/` directory) - Production-ready system with full features

#### Core OCR Engines (5 Total)
**Advanced Implementation (4/5 Working)**:
- ✅ **Tesseract Engine** - Traditional OCR with high reliability
- ✅ **EasyOCR Engine** - Multi-language support  
- ✅ **TrOCR Engine** - Microsoft's Transformer-based OCR
- ✅ **Surya Engine** - Modern layout-aware OCR (slow but accurate)
- ⚠️ **PaddleOCR Engine** - Gracefully disabled (Python 3.13+ incompatible)

#### Computer Vision Validation System
- **Visual Validator** - Image quality assessment, bounding box validation, cross-engine consistency
- **Layout Analyzer** - LayoutParser integration with Detectron2 fallback, table/column detection
- **Document Structure** - Reading order estimation, document type classification

#### Adaptive Learning & AI Features
- **Pattern Detector** - Automatically identifies 40+ pattern types (currency, dates, addresses, etc.)
- **Document Classifier** - Auto-discovers document types without pre-configuration
- **Learning Database** - SQLite-based persistent learning with user feedback integration
- **Performance Monitoring** - Real-time engine weight adjustment based on accuracy

#### Consensus Building
- **Multi-Strategy Merging** - Word-level voting, character-level consensus, hybrid approaches
- **Confidence Weighting** - Visual validation enhanced confidence scoring
- **Agreement Analysis** - Inter-engine consistency metrics

### Key Design Principles
- **Accuracy First**: Multiple engines + CV validation + adaptive learning
- **Local Processing**: All engines run locally, no cloud dependencies
- **AI-Optimized Output**: Structured JSON designed for downstream AI processing
- **Enterprise-Grade**: Real-time monitoring, user feedback, graceful degradation

### Development Commands

#### Installation
```bash
# Automated system-wide installation (recommended)
./install.sh

# Manual verification
python test_basic.py
```

#### Basic Usage
```bash
# Process single document (basic implementation)
python betterocr.py document.pdf

# Process with debug logging
python betterocr.py document.pdf --debug

# Batch processing
python batch_process.py
```

#### Advanced Usage (Production System)
```bash
# Navigate to advanced implementation
cd ocr_engine/

# Comprehensive processing with all features
python example_usage.py samples/document.pdf

# Check latest results
ls -t output/*.json | head -1 | xargs cat | python -m json.tool

# Run specific tests
python test_surya.py
python test_visual_validation.py
```

### Architecture Details

#### Key Files & Components
- **Basic Implementation**:
  - `betterocr.py` - Main orchestrator
  - `engines/` - OCR engine wrappers (tesseract, easyocr, paddle)
  - `consensus/text_merger.py` - Basic consensus building
  - `vision/layout_analyzer.py` - Basic computer vision
  - `utils/pdf_handler.py` - PDF conversion utilities

- **Advanced Implementation** (`ocr_engine/`):
  - `src/main.py` - Advanced processing pipeline
  - `src/engines/` - All 5 OCR engines with sophisticated wrappers
  - `src/fusion/consensus_merger.py` - Multi-strategy consensus algorithms
  - `src/vision/` - Visual validator + layout analyzer with Detectron2 support
  - `src/adaptive/` - Pattern detection, document classification, learning database
  - `src/processing/` - Document processor with batch capabilities

### Processing Flow
1. **Document Loading** - PDF converted to high-DPI images (300 DPI)
2. **Multi-Engine OCR** - All available engines process independently
3. **Visual Validation** - Computer vision quality assessment and cross-validation
4. **Pattern Detection** - Automatic identification of 40+ pattern types
5. **Consensus Building** - Multi-strategy merging with confidence weighting
6. **Document Classification** - Auto-discovery and similarity matching
7. **Adaptive Learning** - Performance tracking and rule updates
8. **Structured Output** - Comprehensive JSON with metadata and confidence scores

### Current Capabilities ✅
- 4/5 OCR engines operational with graceful degradation
- Visual validation system with cross-engine consistency checks
- Pattern detection identifying 40+ patterns per document (currency, dates, addresses, etc.)
- Consensus building with hybrid algorithms and proper confidence scoring
- Adaptive learning with document classification and user feedback integration
- Layout analysis with OpenCV + LayoutParser (Detectron2 fallback)
- Batch processing with comprehensive error handling
- Real-time performance monitoring and engine weight adjustment

### Output Format
Comprehensive JSON containing:
- Raw OCR results from each engine with confidence scores
- Consensus text with agreement metrics and processing metadata
- Visual validation results and quality indicators
- Detected patterns (40+ types) with confidence levels
- Document classification and similarity scores
- Structured data extraction (tables, key-value pairs)
- Learning opportunities and feedback context
- Processing times and performance metrics

### Testing & Sample Data
- **Test Files**: `test_basic.py`, `test_surya.py`, `test_visual_validation.py`, `example_usage.py`
- **Sample Documents**: Tax returns (1975-2023), invoices, technical manuals, medical results
- **Comprehensive Testing**: Both basic and advanced implementations with extensive document samples

### Known Limitations
- **Detectron2 Layout Analysis**: Currently disabled (not installed), using OpenCV fallback
- **Performance**: Surya engine takes ~79 seconds per document (accuracy over speed)
- **Handwriting**: Limited handwriting recognition optimization
- **Real-time**: System designed for batch/accuracy over real-time processing

### Monitoring Notes
- System includes detailed performance monitoring and is production-ready
- Main enhancement opportunity: Detectron2 installation for advanced layout analysis
- Watch for: Layout analysis quality issues with complex documents
- Consider: Performance optimizations for time-sensitive applications