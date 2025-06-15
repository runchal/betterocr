"""
EasyOCR Engine Implementation
Multi-language OCR with GPU acceleration
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import easyocr
import numpy as np
from PIL import Image

from .base_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine with multi-language support"""
    
    # Common language codes
    LANGUAGE_MAP = {
        'english': 'en',
        'spanish': 'es',
        'french': 'fr',
        'german': 'de',
        'chinese_simplified': 'ch_sim',
        'chinese_traditional': 'ch_tra',
        'japanese': 'ja',
        'korean': 'ko',
        'arabic': 'ar',
        'hindi': 'hi',
        'russian': 'ru'
    }
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'languages': ['en'],
            'gpu': True,
            'detect_network': 'craft',
            'recog_network': 'standard',
            'detail': 1,  # 0=simple, 1=detailed with bbox
            'paragraph': True,
            'width_ths': 0.8,
            'height_ths': 0.8,
            'ycenter_ths': 0.5,
            'x_ths': 1.0,
            'y_ths': 0.5,
            'rotation_info': None,
            'decoder': 'greedy',  # 'greedy', 'beamsearch', 'wordbeamsearch'
            'beamWidth': 5,
            'batch_size': 1,
            'workers': 0,
            'allowlist': None,
            'blocklist': None,
            'text_threshold': 0.7,
            'low_text': 0.4,
            'link_threshold': 0.4,
            'canvas_size': 2560,
            'mag_ratio': 1.0
        }
        
        if config:
            default_config.update(config)
            
        super().__init__('easyocr', default_config)
        
        self.reader = None
        self.current_languages = None
        
    def _check_availability(self) -> bool:
        """Check if EasyOCR is available"""
        try:
            import easyocr
            import cv2
            return True
        except ImportError:
            logger.warning("EasyOCR dependencies not available")
            return False
    
    def _load_reader(self, languages: Optional[List[str]] = None):
        """Load or reload EasyOCR reader with specified languages"""
        if languages is None:
            languages = self.config['languages']
            
        # Check if we need to reload reader (different languages)
        if self.reader is None or self.current_languages != languages:
            logger.info(f"Loading EasyOCR reader for languages: {languages}")
            
            try:
                self.reader = easyocr.Reader(
                    languages,
                    gpu=self.config['gpu'],
                    model_storage_directory=None,
                    download_enabled=True,
                    detector=self.config['detect_network'],
                    recognizer=self.config['recog_network']
                )
                self.current_languages = languages
                logger.info("EasyOCR reader loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR reader: {e}")
                raise
    
    def _extract_text_from_image(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text using EasyOCR"""
        result = OCRResult()
        
        # Determine languages
        languages = kwargs.get('languages', self.config['languages'])
        self._load_reader(languages)
        
        try:
            # Read text with detailed results
            detections = self.reader.readtext(
                str(image_path),
                detail=1,  # Always get detailed results
                paragraph=kwargs.get('paragraph', self.config['paragraph']),
                width_ths=kwargs.get('width_ths', self.config['width_ths']),
                height_ths=kwargs.get('height_ths', self.config['height_ths']),
                ycenter_ths=kwargs.get('ycenter_ths', self.config['ycenter_ths']),
                x_ths=kwargs.get('x_ths', self.config['x_ths']),
                y_ths=kwargs.get('y_ths', self.config['y_ths']),
                rotation_info=kwargs.get('rotation_info', self.config['rotation_info']),
                decoder=kwargs.get('decoder', self.config['decoder']),
                beamWidth=kwargs.get('beamWidth', self.config['beamWidth']),
                batch_size=kwargs.get('batch_size', self.config['batch_size']),
                workers=kwargs.get('workers', self.config['workers']),
                allowlist=kwargs.get('allowlist', self.config['allowlist']),
                blocklist=kwargs.get('blocklist', self.config['blocklist']),
                text_threshold=kwargs.get('text_threshold', self.config['text_threshold']),
                low_text=kwargs.get('low_text', self.config['low_text']),
                link_threshold=kwargs.get('link_threshold', self.config['link_threshold']),
                canvas_size=kwargs.get('canvas_size', self.config['canvas_size']),
                mag_ratio=kwargs.get('mag_ratio', self.config['mag_ratio'])
            )
            
            # Parse results
            result = self._parse_easyocr_results(detections)
            
            # Add language detection info
            result.metadata['detected_languages'] = self._detect_languages(detections)
            result.metadata['languages_used'] = languages
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            result.errors.append(str(e))
            
        return result
    
    def _parse_easyocr_results(self, detections: List) -> OCRResult:
        """Parse EasyOCR detection results"""
        result = OCRResult()
        
        if not detections:
            return result
        
        all_text = []
        total_confidence = 0
        
        for detection in detections:
            # EasyOCR returns: (bbox, text, confidence) or sometimes just (bbox, text)
            if len(detection) == 3:
                bbox_points, text, confidence = detection
            elif len(detection) == 2:
                bbox_points, text = detection
                confidence = 0.8  # Default confidence when not provided
            else:
                logger.warning(f"Unexpected EasyOCR detection format: {detection}")
                continue
            
            # Convert bbox points to (x, y, w, h) format
            bbox = self._convert_bbox_format(bbox_points)
            
            # Store as word (EasyOCR typically returns word/phrase level)
            result.words.append((text, confidence, bbox))
            
            all_text.append(text)
            total_confidence += confidence
        
        # Combine into full text
        result.text = ' '.join(all_text)
        
        # Calculate overall confidence
        result.confidence = total_confidence / len(detections) if detections else 0.0
        
        # Group words into lines based on Y-coordinate proximity
        result.lines = self._group_words_into_lines(result.words)
        
        # Add metadata
        result.metadata = {
            'total_detections': len(detections),
            'decoder_used': self.config['decoder'],
            'gpu_enabled': self.config['gpu']
        }
        
        return result
    
    def _convert_bbox_format(self, bbox_points: List) -> Tuple[int, int, int, int]:
        """Convert EasyOCR bbox format to (x, y, w, h)"""
        if len(bbox_points) == 4:
            # Four corner points
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        return (0, 0, 0, 0)
    
    def _group_words_into_lines(self, words: List[Tuple]) -> List[Tuple]:
        """Group words into lines based on vertical proximity"""
        if not words:
            return []
        
        # Sort words by Y coordinate
        sorted_words = sorted(words, key=lambda w: w[2][1] if len(w) > 2 else 0)
        
        lines = []
        current_line = []
        current_y = None
        y_threshold = 10  # Pixels threshold for same line
        
        for word in sorted_words:
            if len(word) < 3:
                continue
                
            text, conf, bbox = word
            y = bbox[1]
            
            if current_y is None or abs(y - current_y) <= y_threshold:
                current_line.append(word)
                if current_y is None:
                    current_y = y
            else:
                # Start new line
                if current_line:
                    line_text = ' '.join([w[0] for w in current_line])
                    line_conf = sum([w[1] for w in current_line]) / len(current_line)
                    line_bbox = self._merge_bboxes([w[2] for w in current_line])
                    lines.append((line_text, line_conf, line_bbox))
                
                current_line = [word]
                current_y = y
        
        # Add last line
        if current_line:
            line_text = ' '.join([w[0] for w in current_line])
            line_conf = sum([w[1] for w in current_line]) / len(current_line)
            line_bbox = self._merge_bboxes([w[2] for w in current_line])
            lines.append((line_text, line_conf, line_bbox))
        
        return lines
    
    def _merge_bboxes(self, bboxes: List[Tuple]) -> Tuple[int, int, int, int]:
        """Merge multiple bounding boxes into one"""
        if not bboxes:
            return (0, 0, 0, 0)
            
        x_mins = [b[0] for b in bboxes]
        y_mins = [b[1] for b in bboxes]
        x_maxs = [b[0] + b[2] for b in bboxes]
        y_maxs = [b[1] + b[3] for b in bboxes]
        
        x_min = min(x_mins)
        y_min = min(y_mins)
        x_max = max(x_maxs)
        y_max = max(y_maxs)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _detect_languages(self, detections: List) -> List[str]:
        """Detect languages present in the text"""
        # This is a placeholder - EasyOCR doesn't provide language detection
        # In a real implementation, you could use langdetect or fasttext
        return self.current_languages
    
    def detect_and_read(self, image_path: Path) -> OCRResult:
        """Detect language first, then read with appropriate model"""
        # First, try to detect language with a multi-language model
        self._load_reader(['en', 'es', 'fr', 'de'])  # Common languages
        
        # Do initial detection
        initial_result = self._extract_text_from_image(image_path)
        
        if initial_result.text:
            # Use langdetect to identify language
            try:
                from langdetect import detect
                detected_lang = detect(initial_result.text)
                
                # Map to EasyOCR language code
                easyocr_lang = self.LANGUAGE_MAP.get(detected_lang, detected_lang)
                
                # Reload with detected language
                if easyocr_lang not in self.current_languages:
                    self._load_reader([easyocr_lang])
                    return self._extract_text_from_image(image_path)
                    
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        return initial_result
    
    def supports_language(self, language_code: str) -> bool:
        """Check if EasyOCR supports a language"""
        # EasyOCR supports 80+ languages
        # This is a simplified check
        common_supported = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 
                          'ja', 'ko', 'zh', 'ar', 'hi', 'th', 'vi', 'id', 'ms']
        return language_code in common_supported or language_code in self.LANGUAGE_MAP