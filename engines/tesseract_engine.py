"""
Tesseract OCR engine wrapper
"""

import logging
from pathlib import Path
from .base_engine import BaseOCREngine

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine implementation"""
    
    def __init__(self):
        super().__init__('tesseract')
        
    def _check_availability(self):
        """Check if Tesseract is installed"""
        try:
            import pytesseract
            # Try to get tesseract version
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    def _process_image(self, image_path):
        """Process image with Tesseract"""
        import pytesseract
        from PIL import Image
        
        # Open image
        image = Image.open(image_path)
        
        # Run OCR with additional data
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Could also get additional data like:
        # - Bounding boxes: pytesseract.image_to_boxes(image)
        # - Detailed data: pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        return text.strip()
    
    def _calculate_confidence(self, text):
        """Tesseract-specific confidence calculation"""
        # Tesseract provides confidence scores we could use
        # For now, use base implementation
        return super()._calculate_confidence(text)