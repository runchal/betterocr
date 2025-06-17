#!/usr/bin/env python3
"""
BetterOCR - Multi-engine OCR with Computer Vision validation
Main entry point for document processing
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BetterOCR:
    """Main OCR processing class"""

    def __init__(self, debug=False):
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

        # Initialize components
        self.engines = self._init_engines()

        # Initialize vision validator
        try:
            from vision.layout_analyzer import LayoutAnalyzer
            self.vision_validator = LayoutAnalyzer()
            logger.info("Vision validator initialized")
        except Exception as e:
            logger.warning(f"Vision validation not available: {e}")
            self.vision_validator = None

        # Initialize consensus builder
        try:
            from consensus.text_merger import TextMerger
            self.consensus_builder = TextMerger()
            logger.info("Consensus builder initialized")
        except Exception as e:
            logger.warning(f"Consensus builder not available: {e}")
            self.consensus_builder = None

    def _init_engines(self):
        """Initialize available OCR engines"""
        engines = {}

        # Try to import each engine
        try:
            from engines.tesseract_engine import TesseractEngine
            engines['tesseract'] = TesseractEngine()
            logger.info("Tesseract engine loaded")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")

        try:
            from engines.easyocr_engine import EasyOCREngine
            engines['easyocr'] = EasyOCREngine()
            logger.info("EasyOCR engine loaded")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")

        try:
            from engines.paddle_engine import PaddleEngine
            engines['paddle'] = PaddleEngine()
            logger.info("PaddleOCR engine loaded")
        except Exception as e:
            logger.warning(f"PaddleOCR not available: {e}")

        return engines

    def process_document(self, file_path):
        """Process a single document through all engines"""
        file_path = Path(file_path)
        logger.info(f"Processing: {file_path}")

        results = {
            'file': str(file_path),
            'engines': {},
            'consensus': None,
            'vision_validation': None,
            'final_output': None
        }

        # Run each OCR engine
        for engine_name, engine in self.engines.items():
            try:
                logger.debug(f"Running {engine_name}...")
                engine_result = engine.process(file_path)
                results['engines'][engine_name] = engine_result
            except Exception as e:
                logger.error(f"Error in {engine_name}: {e}")
                results['engines'][engine_name] = {
                    'error': str(e),
                    'status': 'failed'
                }

        # Build consensus
        if self.consensus_builder:
            results['consensus'] = self.consensus_builder.merge_results(results['engines'])
        else:
            results['consensus'] = self._build_consensus(results['engines'])

        # Apply vision validation
        if self.vision_validator and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            try:
                vision_results = self.vision_validator.analyze_image(file_path)
                results['vision_validation'] = vision_results
            except Exception as e:
                logger.error(f"Vision validation failed: {e}")
                results['vision_validation'] = {'status': 'error', 'error': str(e)}
        else:
            results['vision_validation'] = {'status': 'not_applicable'}

        # Generate final output
        results['final_output'] = self._generate_final_output(results)

        return results

    def _build_consensus(self, engine_results):
        """Build consensus from multiple engine results"""
        # For now, just organize the results
        # Will add sophisticated merging later
        return {
            'method': 'all_results',
            'note': 'Currently showing all engine results with confidence scores'
        }

    def _generate_final_output(self, results):
        """Generate the final structured output"""
        output = {
            'document_info': {
                'file': results['file'],
                'processing_complete': True,
                'engines_used': list(results['engines'].keys()),
                'successful_engines': [name for name, r in results['engines'].items()
                                     if r.get('status') == 'success']
            },
            'consensus': {
                'text': '',
                'confidence': 0.0,
                'method': 'none',
                'agreement_score': 0.0
            },
            'text_variations': {},
            'confidence_scores': {},
            'structured_data': {
                'tables': [],
                'key_value_pairs': {},
                'layout_blocks': []
            },
            'vision_analysis': {}
        }

        # Add consensus information if available
        if results.get('consensus') and results['consensus'].get('status') == 'success':
            consensus = results['consensus']
            output['consensus'] = {
                'text': consensus.get('consensus_text', ''),
                'confidence': consensus.get('overall_confidence', 0.0),
                'method': consensus.get('consensus_method', 'unknown'),
                'agreement_score': consensus.get('agreement_score', 0.0)
            }

            # Add variations from consensus
            if 'variations' in consensus:
                output['text_variations'] = {
                    engine: var['text']
                    for engine, var in consensus['variations'].items()
                }
        else:
            # Fallback: collect all text variations
            for engine_name, engine_result in results['engines'].items():
                if 'error' not in engine_result:
                    output['text_variations'][engine_name] = engine_result.get('text', '')
                    output['confidence_scores'][engine_name] = engine_result.get('confidence', 0.0)

        # Add vision analysis results
        if results.get('vision_validation') and 'error' not in results['vision_validation']:
            vision = results['vision_validation']
            output['vision_analysis'] = {
                'layout': vision.get('structure', {}),
                'tables_detected': len(vision.get('tables', [])),
                'text_regions': len(vision.get('text_blocks', [])),
                'tables': vision.get('tables', []),
                'text_blocks': vision.get('text_blocks', [])
            }

        return output


def main():
    parser = argparse.ArgumentParser(description='BetterOCR - Multi-engine OCR with CV validation')
    parser.add_argument('input', help='Input PDF or image file')
    parser.add_argument('--output', '-o', help='Output JSON file (default: input_name.json)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--engines', help='Comma-separated list of engines to use')

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Initialize BetterOCR
    ocr = BetterOCR(debug=args.debug)

    # Process document
    try:
        results = ocr.process_document(input_path)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix('.json')

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        # Print summary
        if results['final_output']:
            print("\n=== OCR Results Summary ===")
            print(f"Engines used: {list(results['engines'].keys())}")
            print(f"Confidence scores: {results['final_output']['confidence_scores']}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
