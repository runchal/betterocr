"""
Base OCR Engine Interface
All OCR engines must inherit from this class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import time
import uuid
import tempfile
import os
from pathlib import Path
from pdf2image import convert_from_path


class OCRResult:
    """Standardized OCR result format"""
    
    def __init__(self):
        self.text = ""
        self.confidence = 0.0
        self.words = []  # List of (word, confidence, bbox)
        self.lines = []  # List of (line_text, confidence, bbox)
        self.paragraphs = []  # List of (paragraph_text, confidence, bbox)
        self.processing_time = 0.0
        self.metadata = {}
        self.engine_name = ""
        self.errors = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'words': self.words,
            'lines': self.lines,
            'paragraphs': self.paragraphs,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'engine_name': self.engine_name,
            'errors': self.errors
        }


class BaseOCREngine(ABC):
    """Abstract base class for all OCR engines"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.is_available = self._check_availability()
        
    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if this engine is available on the system"""
        pass
    
    @abstractmethod
    def _extract_text_from_image(self, image_path: Path, **kwargs) -> OCRResult:
        """
        Extract text from a single image
        Must be implemented by each engine
        """
        pass
    
    def extract_text(self, image_path: Path, config: Optional[Dict] = None) -> OCRResult:
        """
        Main method to extract text from an image or PDF
        Handles timing, error catching, and standardization
        """
        if not self.is_available:
            result = OCRResult()
            result.engine_name = self.name
            result.errors.append(f"{self.name} engine is not available")
            return result
            
        start_time = time.time()
        
        # Merge config
        extraction_config = {**self.config}
        if config:
            extraction_config.update(config)
            
        try:
            # Convert PDF to images if needed
            processed_paths = self._prepare_images(image_path)
            
            # Process all pages/images
            all_results = []
            for img_path in processed_paths:
                page_result = self._extract_text_from_image(img_path, **extraction_config)
                all_results.append(page_result)
            
            # Merge results from all pages
            result = self._merge_page_results(all_results)
            result.engine_name = self.name
            result.processing_time = time.time() - start_time
            
            # Ensure all required fields are present
            if not hasattr(result, 'confidence'):
                result.confidence = self._calculate_confidence(result)
                
            return result
            
        except Exception as e:
            result = OCRResult()
            result.engine_name = self.name
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
            return result
    
    def _prepare_images(self, file_path: Path) -> List[Path]:
        """Convert PDF to images if needed, return list of image paths"""
        file_str = str(file_path)
        
        # If it's already an image, return as-is
        if file_str.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return [file_path]
        
        # If it's a PDF, convert to images
        if file_str.lower().endswith('.pdf'):
            try:
                # Convert PDF to images
                images = convert_from_path(file_str, dpi=300)
                
                # Save images to temporary files
                temp_paths = []
                for i, image in enumerate(images):
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=f'_page_{i}.png', 
                        delete=False
                    )
                    temp_file.close()
                    
                    # Save image
                    image.save(temp_file.name, 'PNG')
                    temp_paths.append(Path(temp_file.name))
                
                return temp_paths
                
            except Exception as e:
                raise Exception(f"Failed to convert PDF to images: {e}")
        
        # For other file types, try to treat as image
        return [file_path]
    
    def _merge_page_results(self, page_results: List[OCRResult]) -> OCRResult:
        """Merge results from multiple pages into a single result"""
        if not page_results:
            return OCRResult()
        
        if len(page_results) == 1:
            return page_results[0]
        
        # Merge multiple pages
        merged = OCRResult()
        merged.text = '\n\n--- PAGE BREAK ---\n\n'.join(
            [r.text for r in page_results if r.text]
        )
        
        # Merge words, lines, paragraphs with page offset
        for page_idx, page_result in enumerate(page_results):
            # Add page marker to metadata
            page_marker = f"[PAGE {page_idx + 1}]"
            
            # Merge words
            for word, conf, bbox in page_result.words:
                merged.words.append((f"{page_marker} {word}", conf, bbox))
            
            # Merge lines
            for line, conf, bbox in page_result.lines:
                merged.lines.append((f"{page_marker} {line}", conf, bbox))
            
            # Merge paragraphs
            for para, conf, bbox in page_result.paragraphs:
                merged.paragraphs.append((f"{page_marker} {para}", conf, bbox))
        
        # Calculate overall confidence
        all_confidences = [r.confidence for r in page_results if r.confidence > 0]
        merged.confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Merge errors
        for page_result in page_results:
            merged.errors.extend(page_result.errors)
        
        # Merge metadata
        merged.metadata = {
            'total_pages': len(page_results),
            'individual_page_metadata': [r.metadata for r in page_results]
        }
        
        return merged
    
    def get_confidence_scores(self, result: OCRResult) -> Dict[str, float]:
        """Get detailed confidence scores"""
        scores = {
            'overall': result.confidence,
            'word_average': 0.0,
            'line_average': 0.0
        }
        
        if result.words:
            word_confidences = [w[1] for w in result.words if len(w) > 1]
            if word_confidences:
                scores['word_average'] = sum(word_confidences) / len(word_confidences)
                
        if result.lines:
            line_confidences = [l[1] for l in result.lines if len(l) > 1]
            if line_confidences:
                scores['line_average'] = sum(line_confidences) / len(line_confidences)
                
        return scores
    
    def get_bounding_boxes(self, result: OCRResult) -> Dict[str, List]:
        """Get all bounding boxes"""
        return {
            'words': [w[2] for w in result.words if len(w) > 2],
            'lines': [l[2] for l in result.lines if len(l) > 2],
            'paragraphs': [p[2] for p in result.paragraphs if len(p) > 2]
        }
    
    def get_processing_metadata(self, result: OCRResult) -> Dict[str, Any]:
        """Get processing metadata"""
        metadata = {
            'engine': self.name,
            'processing_time': result.processing_time,
            'success': len(result.errors) == 0,
            'config': self.config
        }
        
        if result.metadata:
            metadata.update(result.metadata)
            
        return metadata
    
    def _calculate_confidence(self, result: OCRResult) -> float:
        """Calculate overall confidence if not provided by engine"""
        if result.words:
            confidences = [w[1] for w in result.words if len(w) > 1 and w[1] > 0]
            if confidences:
                return sum(confidences) / len(confidences)
        
        # Fallback heuristic
        if result.text:
            # Basic confidence based on text quality
            words = result.text.split()
            if len(words) > 50:
                return 0.8
            elif len(words) > 10:
                return 0.6
            else:
                return 0.4
        
        return 0.0
    
    def supports_language(self, language_code: str) -> bool:
        """Check if engine supports a specific language"""
        return True  # Override in specific engines
    
    def supports_batch_processing(self) -> bool:
        """Check if engine supports batch processing"""
        return False  # Override in specific engines