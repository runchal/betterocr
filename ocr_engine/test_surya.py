#!/usr/bin/env python3
"""Test Surya OCR availability and functionality"""

import sys
print(f"Python version: {sys.version}")

try:
    print("Importing Surya...")
    import surya
    print("✓ Surya imported successfully")
    
    print("Checking available modules...")
    from surya.models import DetectionPredictor, RecognitionPredictor, LayoutPredictor
    print("✓ Predictor classes available")
    
    print("Attempting to load detection model...")
    det_model = DetectionPredictor(device="cpu")
    print("✓ Detection model loaded")
    
    print("\nSurya OCR is working!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()