"""
Base class for all OCR engines
Defines the common interface that all engines must implement
"""

from abc import ABC, abstractmethod
from pathlib import Path
import time


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self, name):
        self.name = name
        self.is_available = self._check_availability()
    
    @abstractmethod
    def _check_availability(self):
        """Check if this engine is available on the system"""
        pass
    
    @abstractmethod
    def _process_image(self, image_path):
        """Process a single image and return OCR results"""
        pass
    
    def process(self, file_path):
        """Process a file (PDF or image) and return results"""
        file_path = Path(file_path)
        start_time = time.time()
        
        result = {
            'engine': self.name,
            'file': str(file_path),
            'timestamp': time.time(),
            'processing_time': 0,
            'status': 'processing'
        }
        
        try:
            if file_path.suffix.lower() == '.pdf':
                # Handle PDF - will need to convert to images
                result['text'] = self._process_pdf(file_path)
            else:
                # Handle image directly
                result['text'] = self._process_image(file_path)
            
            result['status'] = 'success'
            result['confidence'] = self._calculate_confidence(result['text'])
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['text'] = ''
            result['confidence'] = 0.0
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _process_pdf(self, pdf_path):
        """Convert PDF to images and process each page"""
        import sys
        import os
        
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from utils.pdf_handler import PDFHandler
        
        # Convert PDF to images
        image_paths, temp_dir = PDFHandler.pdf_to_images(pdf_path)
        
        try:
            # Process each page
            all_text = []
            for i, image_path in enumerate(image_paths):
                page_text = self._process_image(image_path)
                if page_text:
                    all_text.append(f"[Page {i+1}]\n{page_text}")
            
            # Combine all pages
            combined_text = "\n\n".join(all_text)
            
        finally:
            # Always clean up temp directory
            PDFHandler.cleanup_temp_dir(temp_dir)
        
        return combined_text
    
    def _calculate_confidence(self, text):
        """Calculate a confidence score for the OCR result"""
        # Basic confidence based on text length and quality
        # Each engine can override this with its own method
        if not text:
            return 0.0
        
        # Simple heuristic: longer text with proper words = higher confidence
        words = text.split()
        if len(words) < 5:
            return 0.3
        elif len(words) < 50:
            return 0.6
        else:
            return 0.8