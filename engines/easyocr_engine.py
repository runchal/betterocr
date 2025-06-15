"""
EasyOCR engine wrapper
"""

import logging
from pathlib import Path
from .base_engine import BaseOCREngine

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine implementation"""
    
    def __init__(self):
        super().__init__('easyocr')
        self.reader = None
        
    def _check_availability(self):
        """Check if EasyOCR is installed"""
        try:
            import easyocr
            # Initialize reader with English - will download models on first run
            self.reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
            return True
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")
            return False
    
    def _process_image(self, image_path):
        """Process image with EasyOCR"""
        if not self.reader:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Read text from image
        # Returns list of (bbox, text, confidence) tuples
        results = self.reader.readtext(str(image_path))
        
        # Extract just the text for now
        # We're keeping all the data for future use
        text_parts = []
        total_confidence = 0
        
        for (bbox, text, confidence) in results:
            text_parts.append(text)
            total_confidence += confidence
        
        # Store detailed results for later use
        self.last_detailed_results = results
        
        # Calculate average confidence
        if results:
            self.last_confidence = total_confidence / len(results)
        else:
            self.last_confidence = 0.0
        
        return ' '.join(text_parts)
    
    def _calculate_confidence(self, text):
        """Use EasyOCR's confidence scores"""
        if hasattr(self, 'last_confidence'):
            return self.last_confidence
        return super()._calculate_confidence(text)
    
    def get_detailed_results(self):
        """Get the detailed results with bounding boxes"""
        if hasattr(self, 'last_detailed_results'):
            return self.last_detailed_results
        return None