"""
PaddleOCR engine wrapper
"""

import logging
from pathlib import Path
from .base_engine import BaseOCREngine

logger = logging.getLogger(__name__)


class PaddleEngine(BaseOCREngine):
    """PaddleOCR engine implementation"""
    
    def __init__(self):
        super().__init__('paddleocr')
        self.ocr = None
        
    def _check_availability(self):
        """Check if PaddleOCR is installed"""
        try:
            from paddleocr import PaddleOCR
            # Initialize with English
            # use_angle_cls=True enables text angle classification
            # use_gpu=False for CPU (set True if you have CUDA)
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False
            )
            return True
        except Exception as e:
            logger.warning(f"PaddleOCR not available: {e}")
            return False
    
    def _process_image(self, image_path):
        """Process image with PaddleOCR"""
        if not self.ocr:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en', 
                use_gpu=False,
                show_log=False
            )
        
        # Run OCR
        result = self.ocr.ocr(str(image_path), cls=True)
        
        # PaddleOCR returns nested list structure
        # [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence)], ...]
        text_parts = []
        total_confidence = 0
        count = 0
        
        if result and result[0]:  # Check if results exist
            for line in result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    text_parts.append(text)
                    total_confidence += confidence
                    count += 1
        
        # Store detailed results
        self.last_detailed_results = result
        
        # Calculate average confidence
        if count > 0:
            self.last_confidence = total_confidence / count
        else:
            self.last_confidence = 0.0
        
        return ' '.join(text_parts)
    
    def _calculate_confidence(self, text):
        """Use PaddleOCR's confidence scores"""
        if hasattr(self, 'last_confidence'):
            return self.last_confidence
        return super()._calculate_confidence(text)
    
    def get_detailed_results(self):
        """Get the detailed results with bounding boxes"""
        if hasattr(self, 'last_detailed_results'):
            return self.last_detailed_results
        return None