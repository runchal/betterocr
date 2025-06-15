"""
PDF handling utilities for BetterOCR
Converts PDF pages to images for OCR processing
"""

import logging
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)


class PDFHandler:
    """Handle PDF to image conversion"""
    
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300):
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion (higher = better quality but slower)
            
        Returns:
            List of image paths
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            logger.error("pdf2image not installed. Run: pip install pdf2image")
            raise
        
        pdf_path = Path(pdf_path)
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp(prefix='betterocr_')
        logger.debug(f"Created temp directory: {temp_dir}")
        
        try:
            # Convert PDF to images
            logger.info(f"Converting PDF to images at {dpi} DPI...")
            images = convert_from_path(pdf_path, dpi=dpi)
            
            # Save images
            image_paths = []
            for i, image in enumerate(images):
                image_path = Path(temp_dir) / f"page_{i+1:03d}.png"
                image.save(image_path, 'PNG')
                image_paths.append(image_path)
                logger.debug(f"Saved page {i+1} to {image_path}")
            
            logger.info(f"Converted {len(image_paths)} pages")
            return image_paths, temp_dir
            
        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    @staticmethod
    def cleanup_temp_dir(temp_dir):
        """Clean up temporary directory"""
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")