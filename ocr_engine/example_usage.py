#!/usr/bin/env python3
"""
Example usage of the Adaptive Multi-Engine OCR Library
Demonstrates the complete workflow including learning from feedback
"""

import json
import logging
from pathlib import Path
from src.main_ocr import AdaptiveMultiEngineOCR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate the OCR library usage"""
    
    print("Adaptive Multi-Engine OCR Library Demo")
    print("=" * 50)
    
    # Initialize the OCR system with learning enabled
    print("\n1. Initializing OCR system...")
    ocr = AdaptiveMultiEngineOCR(learning_enabled=True)
    
    # Check available engines
    print("\n2. Available OCR engines:")
    for engine_name, engine in ocr.engines.items():
        print(f"   ✓ {engine_name}")
    
    if not ocr.engines:
        print("   ❌ No OCR engines available. Please install dependencies.")
        return
    
    # Example document processing
    sample_document = Path("samples/sample_document.pdf")
    
    if not sample_document.exists():
        print(f"\n❌ Sample document not found: {sample_document}")
        print("Please place a PDF file at 'samples/sample_document.pdf' to test")
        return
    
    print(f"\n3. Processing document: {sample_document}")
    
    # Process the document
    try:
        result = ocr.process_document(str(sample_document))
        
        print(f"\n4. Processing Results:")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Engines used: {result['processing_metadata']['engines_used']}")
        print(f"   Successful engines: {result['processing_metadata']['successful_engines']}")
        
        # Display consensus results
        consensus = result['consensus']
        print(f"\n5. Consensus Results:")
        print(f"   Method: {consensus.get('consensus_method', 'unknown')}")
        print(f"   Confidence: {consensus.get('overall_confidence', 0):.2f}")
        print(f"   Agreement: {consensus.get('agreement_score', 0):.2f}")
        
        # Display detected patterns
        patterns = result['detected_patterns']
        print(f"\n6. Detected Patterns:")
        print(f"   Key-value pairs: {len(patterns.get('key_value_pairs', []))}")
        print(f"   Tables: {len(patterns.get('tables', []))}")
        print(f"   Forms: {len(patterns.get('forms', []))}")
        print(f"   Lists: {len(patterns.get('lists', []))}")
        
        # Display document classification
        classification = result['document_classification']
        print(f"\n7. Document Classification:")
        print(f"   Best guess: {classification.get('best_overall_guess', 'unknown')}")
        
        discovered_types = classification.get('discovered_types', {})
        if discovered_types:
            print("   Discovered types:")
            for doc_type, info in discovered_types.items():
                print(f"     - {doc_type}: {info.get('confidence', 0):.2f}")
        
        # Display structured data
        structured_data = result['structured_data']
        fields = structured_data.get('fields', {})
        print(f"\n8. Extracted Fields ({len(fields)} found):")
        for field_name, field_info in list(fields.items())[:5]:  # Show first 5
            print(f"   {field_name}: {field_info['value']} (confidence: {field_info['confidence']:.2f})")
        
        if len(fields) > 5:
            print(f"   ... and {len(fields) - 5} more fields")
        
        # Display learning opportunities
        learning_opps = result.get('learning_opportunities', [])
        if learning_opps:
            print(f"\n9. Learning Opportunities:")
            for opp in learning_opps:
                print(f"   - {opp}")
        
        # Save results
        output_file = Path("output") / f"{result['document_id']}_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Demonstrate feedback workflow
        demonstrate_feedback_workflow(ocr, result)
        
        # Show learning summary
        show_learning_summary(ocr)
        
    except Exception as e:
        print(f"\n❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        ocr.cleanup_resources()
        print("\n✓ OCR system cleaned up")

def demonstrate_feedback_workflow(ocr: AdaptiveMultiEngineOCR, result: dict):
    """Demonstrate the feedback and learning workflow"""
    print(f"\n10. Feedback Workflow Demonstration")
    print("-" * 40)
    
    document_id = result['document_id']
    
    # Example 1: Field correction
    fields = result['structured_data'].get('fields', {})
    if fields:
        first_field = list(fields.keys())[0]
        extracted_value = fields[first_field]['value']
        
        print(f"Example field correction:")
        print(f"   Field: {first_field}")
        print(f"   Extracted: '{extracted_value}'")
        
        # Simulate user correction
        corrected_value = f"CORRECTED_{extracted_value}"
        
        feedback_data = {
            'field_name': first_field,
            'extracted_value': extracted_value,
            'correct_value': corrected_value,
            'correction_type': 'field_value',
            'engine_name': 'tesseract'  # Example
        }
        
        print(f"   User correction: '{corrected_value}'")
        
        # Apply feedback
        feedback_result = ocr.apply_user_feedback(document_id, feedback_data)
        print(f"   ✓ Feedback processed: {feedback_result.get('correction_id', 'unknown')}")
    
    # Example 2: Document type labeling
    classification = result['document_classification']
    auto_guess = classification.get('best_overall_guess', 'unknown')
    
    if auto_guess != 'unknown':
        print(f"\nExample document type correction:")
        print(f"   Auto-detected: {auto_guess}")
        
        # Simulate user providing better label
        user_label = "tax_form_1040_2023"
        print(f"   User label: {user_label}")
        
        label_result = ocr.add_document_type_label(document_id, user_label)
        print(f"   ✓ Document type learned: {label_result.get('learned_type', 'unknown')}")

def show_learning_summary(ocr: AdaptiveMultiEngineOCR):
    """Show learning system summary"""
    print(f"\n11. Learning System Summary")
    print("-" * 30)
    
    summary = ocr.get_learning_summary()
    
    if summary.get('learning_enabled'):
        progress = summary.get('learning_progress', {})
        print(f"   Learning active: {progress.get('learning_active', False)}")
        print(f"   Total corrections: {progress.get('total_corrections', 0)}")
        print(f"   New patterns learned: {progress.get('new_patterns_learned', 0)}")
        
        engine_performance = summary.get('engine_performance', {})
        if engine_performance:
            print(f"   Engine performance:")
            for engine, stats in engine_performance.items():
                avg_perf = stats.get('avg_performance', 0)
                print(f"     {engine}: {avg_perf:.2f} (weight: {stats.get('current_weight', 1.0):.2f})")
    else:
        print("   Learning is disabled")

def create_sample_document():
    """Create a sample document for testing if none exists"""
    sample_path = Path("samples/sample_document.pdf")
    sample_path.parent.mkdir(exist_ok=True)
    
    if sample_path.exists():
        return
    
    print(f"Creating sample document at {sample_path}...")
    
    # This would create a sample PDF document
    # For now, just create a placeholder text file
    sample_text = """
    SAMPLE INVOICE
    
    Invoice Number: INV-2024-001
    Date: January 15, 2024
    
    Bill To:
    John Doe
    123 Main Street
    Anytown, NY 12345
    
    Item                    Quantity    Price       Total
    Widget A                5           $10.00      $50.00
    Widget B                3           $15.00      $45.00
    
    Subtotal:                                       $95.00
    Tax (8.5%):                                     $8.08
    Total:                                          $103.08
    
    Please remit payment within 30 days.
    """
    
    # Create a simple text file instead of PDF for demonstration
    with open(sample_path.with_suffix('.txt'), 'w') as f:
        f.write(sample_text)
    
    print(f"Created sample text file: {sample_path.with_suffix('.txt')}")
    print("Note: For full PDF testing, please provide a real PDF file")

if __name__ == "__main__":
    # Create sample document if needed
    create_sample_document()
    
    # Run the main demo
    main()