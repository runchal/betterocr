#!/usr/bin/env python3
"""Test Visual Validation System"""

import sys
import json
from pathlib import Path
from src.vision.visual_validator import VisualValidator
from src.engines.tesseract_engine import TesseractEngine

def main():
    print("Testing Visual Validation System")
    print("=" * 40)
    
    # Initialize components
    visual_validator = VisualValidator()
    tesseract_engine = TesseractEngine()
    
    # Test with sample document
    sample_doc = Path("samples/sample_document.pdf")
    if not sample_doc.exists():
        print(f"Sample document not found: {sample_doc}")
        return
    
    print(f"Processing: {sample_doc}")
    
    # Run basic OCR to get some results to validate
    print("1. Running Tesseract OCR...")
    ocr_result = tesseract_engine.extract_text(sample_doc)
    
    # Mock OCR results for testing
    mock_ocr_results = {
        'tesseract': ocr_result
    }
    
    print("2. Running visual validation...")
    try:
        validation_results = visual_validator.validate_ocr_results(sample_doc, mock_ocr_results)
    except Exception as e:
        print(f"Error in visual validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("3. Visual Validation Results:")
    print("-" * 30)
    
    # Display key results
    image_quality = validation_results.get('image_quality', {})
    print(f"Overall Image Quality: {image_quality.get('overall_quality', 0.0):.2f}")
    
    if 'recommendations' in image_quality:
        print("Recommendations:")
        for rec in image_quality['recommendations']:
            print(f"  - {rec}")
    
    layout_analysis = validation_results.get('layout_analysis', {})
    print(f"Layout Confidence: {layout_analysis.get('confidence', 0.0):.2f}")
    print(f"Document Type: {layout_analysis.get('document_type', 'unknown')}")
    
    bbox_validation = validation_results.get('bbox_validation', {})
    engine_scores = bbox_validation.get('engine_scores', {})
    for engine, scores in engine_scores.items():
        print(f"{engine} Bbox Quality: {scores.get('f1_score', 0.0):.2f}")
    
    visual_confidence = validation_results.get('visual_confidence_scores', {})
    print(f"Visual Confidence: {visual_confidence.get('visual_confidence', 0.0):.2f}")
    
    # Save detailed results
    output_file = "visual_validation_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n✓ Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()