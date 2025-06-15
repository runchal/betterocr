#!/usr/bin/env python3
"""
Basic test to check if BetterOCR is working
"""

from pathlib import Path
import json

# Simple test without importing the main module yet
print("BetterOCR Test Script")
print("=" * 50)

# Check what's available
print("\nChecking available OCR engines...")

# Test Tesseract
try:
    import pytesseract
    print("✓ Tesseract is available")
except ImportError:
    print("✗ Tesseract not found - install with: pip install pytesseract")
    print("  Also need tesseract binary: brew install tesseract")

# Test EasyOCR  
try:
    import easyocr
    print("✓ EasyOCR is available")
except ImportError:
    print("✗ EasyOCR not found - install with: pip install easyocr")

# Test PaddleOCR
try:
    import paddleocr
    print("✓ PaddleOCR is available")
except ImportError:
    print("✗ PaddleOCR not found - install with: pip install paddlepaddle paddleocr")

# Test OpenCV for computer vision
try:
    import cv2
    print("✓ OpenCV is available") 
except ImportError:
    print("✗ OpenCV not found - install with: pip install opencv-python")

print("\nTo test with a document, place PDFs in the samples/ folder")
print("Then run: python betterocr.py samples/your_document.pdf")