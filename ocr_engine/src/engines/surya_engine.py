"""
Surya OCR Engine Implementation
Modern layout-aware text recognition
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
from PIL import Image

from .base_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class SuryaEngine(BaseOCREngine):
    """Surya OCR engine with advanced layout understanding"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'model_type': 'default',
            'device': 'auto',  # 'auto', 'cuda', 'cpu'
            'batch_size': 1,
            'enable_layout': True,
            'enable_order': True,
            'enable_table': True,
            'confidence_threshold': 0.5,
            'merge_boxes': True,
            'box_expansion': 5,  # Pixels to expand boxes
            'min_text_length': 1,
            'max_text_length': 1000
        }
        
        if config:
            default_config.update(config)
            
        super().__init__('surya', default_config)
        
        self.recognition_model = None
        self.detection_model = None
        self.layout_model = None
        
    def _check_availability(self) -> bool:
        """Check if Surya OCR is available"""
        try:
            import surya
            from surya.models import load_predictors
            return True
        except ImportError:
            logger.warning("Surya OCR not available")
            return False
    
    def _load_model(self):
        """Load Surya OCR model"""
        if self.recognition_model is None:
            try:
                from surya.models import DetectionPredictor, RecognitionPredictor, LayoutPredictor
                
                logger.info("Loading Surya OCR models...")
                
                # Determine device
                import torch
                if self.config['device'] == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                else:
                    device = self.config['device']
                
                # Load models - Surya will download them if needed
                self.detection_model = DetectionPredictor(device=device)
                self.recognition_model = RecognitionPredictor(device=device)
                
                # Load layout model if enabled
                if self.config['enable_layout']:
                    self.layout_model = LayoutPredictor(device=device)
                else:
                    self.layout_model = None
                
                logger.info(f"Surya OCR loaded successfully on {device}")
                
            except Exception as e:
                logger.error(f"Failed to load Surya OCR: {e}")
                raise
    
    def _extract_text_from_image(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text using Surya OCR"""
        result = OCRResult()
        
        # Load model if needed
        self._load_model()
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # First, detect text regions
            det_predictions = self.detection_model([image])
            det_result = det_predictions[0] if det_predictions else None
            
            if not det_result:
                logger.warning("No text detected in image")
                return result
            
            # Check if we have detected text regions
            if not hasattr(det_result, 'bboxes') or not det_result.bboxes:
                logger.warning("No text bboxes found in detection")
                return result
            
            # Run text recognition on detected regions
            # Convert bboxes to the format expected by recognition model
            bboxes_list = []
            for bbox in det_result.bboxes:
                # Convert from polygon format to simple bbox
                if hasattr(bbox, 'polygon'):
                    poly = bbox.polygon
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    bbox_coords = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                else:
                    # Assume it's already in bbox format
                    bbox_coords = list(bbox)
                bboxes_list.append(bbox_coords)
            
            rec_predictions = self.recognition_model([image], bboxes=[bboxes_list])
            rec_result = rec_predictions[0] if rec_predictions else None
            
            # Run layout analysis if enabled
            layout_result = None
            if self.config['enable_layout'] and self.layout_model:
                layout_predictions = self.layout_model([image])
                layout_result = layout_predictions[0] if layout_predictions else None
            
            # Parse and combine results
            result = self._parse_surya_results(rec_result, det_result, layout_result)
            
        except Exception as e:
            logger.error(f"Surya extraction failed: {e}")
            result.errors.append(str(e))
            
        return result
    
    def _parse_surya_results(self, rec_result, det_result, layout_result) -> OCRResult:
        """Parse Surya OCR results"""
        result = OCRResult()
        
        if not rec_result:
            return result
        
        all_text = []
        total_confidence = 0
        line_count = 0
        
        # Process recognition results
        # Surya returns text_lines with bboxes from detection
        if hasattr(rec_result, 'text_lines') and rec_result.text_lines:
            for i, text_line in enumerate(rec_result.text_lines):
                text = text_line.text if hasattr(text_line, 'text') else str(text_line)
                
                # Get confidence (default to high if not provided)
                confidence = getattr(text_line, 'confidence', 0.95)
                
                # Get bbox from text line (already in correct format)
                bbox = None
                if hasattr(text_line, 'bbox') and text_line.bbox:
                    # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = text_line.bbox
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                else:
                    bbox = (0, i * 20, 100, 20)  # Default bbox
                
                # Filter by confidence
                if confidence < self.config['confidence_threshold']:
                    continue
                
                # Store line
                result.lines.append((text, confidence, bbox))
                all_text.append(text)
                total_confidence += confidence
                line_count += 1
                
                # Simple word splitting
                words = text.split()
                if words:
                    word_width = bbox[2] / len(words) if bbox[2] > 0 else 50
                    for j, word in enumerate(words):
                        word_bbox = (
                            bbox[0] + j * word_width,
                            bbox[1],
                            word_width,
                            bbox[3]
                        )
                        result.words.append((word, confidence, word_bbox))
        
        # Combine text
        result.text = '\n'.join(all_text)
        result.confidence = total_confidence / line_count if line_count > 0 else 0.0
        
        # Add layout information if available
        if layout_result:
            result.metadata['layout'] = self._extract_layout_info(layout_result)
        
        # Add metadata
        result.metadata.update({
            'engine': 'surya',
            'total_lines': line_count,
            'has_layout': layout_result is not None
        })
        
        return result
    
    def _convert_surya_bbox(self, bbox) -> Tuple[float, float, float, float]:
        """Convert Surya bbox format to standard format"""
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        elif hasattr(bbox, 'bbox'):
            # Handle object with bbox attribute
            return self._convert_surya_bbox(bbox.bbox)
        else:
            return (0.0, 0.0, 100.0, 20.0)
    
    def _extract_layout_info(self, layout_result) -> Dict:
        """Extract layout information from Surya layout result"""
        layout_info = {
            'blocks': [],
            'columns': 0,
            'has_tables': False,
            'has_images': False
        }
        
        if hasattr(layout_result, 'blocks'):
            for block in layout_result.blocks:
                block_info = {
                    'type': getattr(block, 'type', 'text'),
                    'bbox': self._convert_surya_bbox(getattr(block, 'bbox', [0, 0, 0, 0])),
                    'confidence': getattr(block, 'confidence', 0.0)
                }
                layout_info['blocks'].append(block_info)
                
                if block_info['type'] == 'table':
                    layout_info['has_tables'] = True
                elif block_info['type'] == 'image':
                    layout_info['has_images'] = True
        
        return layout_info
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        # Surya supports many languages
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'bn', 'pa', 'te', 'mr', 'ta', 'ur', 'gu', 'kn'
        ]
    
    def optimize_for_document_type(self, document_type: str) -> Dict:
        """Optimize settings for specific document type"""
        optimizations = {
            'invoice': {
                'enable_table': True,
                'enable_layout': True,
                'confidence_threshold': 0.6
            },
            'form': {
                'enable_layout': True,
                'enable_order': True,
                'merge_boxes': False
            },
            'book': {
                'enable_layout': True,
                'enable_order': True,
                'merge_boxes': True
            },
            'handwritten': {
                'confidence_threshold': 0.4,
                'box_expansion': 10
            }
        }
        
        return optimizations.get(document_type, {})