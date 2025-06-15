#!/usr/bin/env python3
"""
Batch processor for BetterOCR
Process multiple documents at once
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from betterocr import BetterOCR

logger = logging.getLogger(__name__)


def batch_process(input_dir, output_dir=None, pattern="*", debug=False):
    """Process all files matching pattern in input directory"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = input_path / "ocr_results"
    
    output_path.mkdir(exist_ok=True)
    
    # Find files to process
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    files_to_process = []
    
    for ext in supported_extensions:
        files_to_process.extend(input_path.glob(f"**/*{ext}"))
    
    if not files_to_process:
        logger.warning(f"No supported files found in {input_path}")
        return
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Initialize BetterOCR
    ocr = BetterOCR(debug=debug)
    
    # Process each file
    results_summary = []
    
    for i, file_path in enumerate(files_to_process, 1):
        logger.info(f"Processing {i}/{len(files_to_process)}: {file_path.name}")
        
        try:
            # Process document
            results = ocr.process_document(file_path)
            
            # Save individual result
            output_file = output_path / f"{file_path.stem}_ocr.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Add to summary
            summary_item = {
                'file': str(file_path),
                'output': str(output_file),
                'status': 'success',
                'engines_used': results['final_output']['document_info']['engines_used'],
                'consensus_confidence': results['final_output']['consensus']['confidence']
            }
            
            results_summary.append(summary_item)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            results_summary.append({
                'file': str(file_path),
                'status': 'error',
                'error': str(e)
            })
    
    # Save batch summary
    summary_file = output_path / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_files': len(files_to_process),
            'successful': len([r for r in results_summary if r['status'] == 'success']),
            'failed': len([r for r in results_summary if r['status'] == 'error']),
            'results': results_summary
        }, f, indent=2)
    
    logger.info(f"Batch processing complete. Results saved to {output_path}")
    logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='BetterOCR Batch Processor')
    parser.add_argument('input_dir', help='Directory containing documents to process')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: input_dir/ocr_results)')
    parser.add_argument('--pattern', '-p', default='*', help='File pattern to match')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    batch_process(args.input_dir, args.output_dir, args.pattern, args.debug)


if __name__ == '__main__':
    main()