"""
Tesseract OCR Engine Implementation
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import pytesseract
from PIL import Image
import cv2
import numpy as np

from .base_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine with multiple configuration options"""
    
    # Page Segmentation Modes
    PSM_MODES = {
        'single_block': 6,      # Uniform block of text
        'single_line': 7,       # Single text line
        'single_word': 8,       # Single word
        'single_char': 10,      # Single character
        'auto': 3,              # Fully automatic
        'auto_osd': 1,          # Auto with orientation
        'column': 4,            # Single column
        'vertical_text': 5,     # Vertically aligned text
        'uniform_block': 6,     # Uniform block
        'sparse_text': 11,      # Sparse text
        'sparse_osd': 12,       # Sparse with OSD
        'raw_line': 13          # Raw line
    }
    
    # OCR Engine Modes
    OEM_MODES = {
        'legacy': 0,
        'lstm': 1,
        'legacy_lstm': 2,
        'default': 3
    }
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'lang': 'eng',
            'psm': 3,  # Fully automatic page segmentation
            'oem': 3,  # Default OCR Engine mode
            'whitelist': None,
            'blacklist': None,
            'config_str': ''
        }
        
        if config:
            default_config.update(config)
            
        super().__init__('tesseract', default_config)
        
        # Store successful configurations for different patterns
        self.successful_configs = {}
        
    def _check_availability(self) -> bool:
        """Check if Tesseract is installed"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    def _extract_text_from_image(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text using Tesseract with specified configuration"""
        result = OCRResult()
        
        # Load image
        image = Image.open(image_path)
        
        # Build Tesseract config string
        config_str = self._build_config_string(kwargs)
        
        try:
            # Get detailed data including confidence scores
            data = pytesseract.image_to_data(
                image, 
                lang=kwargs.get('lang', self.config['lang']),
                config=config_str,
                output_type=pytesseract.Output.DICT
            )
            
            # Parse results
            result = self._parse_tesseract_data(data)
            
            # Also get plain text for convenience
            result.text = pytesseract.image_to_string(
                image,
                lang=kwargs.get('lang', self.config['lang']),
                config=config_str
            ).strip()
            
            # Store successful config for pattern learning
            if result.confidence > 0.8:
                self._store_successful_config(image_path, config_str, result.confidence)
                
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            result.errors.append(str(e))
            
        return result
    
    def _build_config_string(self, kwargs: Dict) -> str:
        """Build Tesseract configuration string"""
        config_parts = []
        
        # Page segmentation mode
        psm = kwargs.get('psm', self.config['psm'])
        if isinstance(psm, str) and psm in self.PSM_MODES:
            psm = self.PSM_MODES[psm]
        config_parts.append(f'--psm {psm}')
        
        # OCR Engine mode
        oem = kwargs.get('oem', self.config['oem'])
        if isinstance(oem, str) and oem in self.OEM_MODES:
            oem = self.OEM_MODES[oem]
        config_parts.append(f'--oem {oem}')
        
        # Character whitelist/blacklist
        if kwargs.get('whitelist') or self.config.get('whitelist'):
            whitelist = kwargs.get('whitelist', self.config['whitelist'])
            config_parts.append(f'-c tessedit_char_whitelist={whitelist}')
            
        if kwargs.get('blacklist') or self.config.get('blacklist'):
            blacklist = kwargs.get('blacklist', self.config['blacklist'])
            config_parts.append(f'-c tessedit_char_blacklist={blacklist}')
            
        # Additional config
        if kwargs.get('config_str'):
            config_parts.append(kwargs['config_str'])
        elif self.config.get('config_str'):
            config_parts.append(self.config['config_str'])
            
        return ' '.join(config_parts)
    
    def _parse_tesseract_data(self, data: Dict) -> OCRResult:
        """Parse Tesseract's detailed output data"""
        result = OCRResult()
        
        n_boxes = len(data['level'])
        
        current_line = []
        current_line_conf = []
        current_paragraph = []
        
        for i in range(n_boxes):
            level = data['level'][i]
            text = data['text'][i]
            conf = float(data['conf'][i])
            
            # Skip empty text
            if not text or text.isspace() or conf == -1:
                continue
                
            # Bounding box
            bbox = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            )
            
            # Word level (level 5)
            if level == 5:
                result.words.append((text, conf / 100.0, bbox))
                current_line.append(text)
                current_line_conf.append(conf / 100.0)
                
            # Line level (level 4)
            elif level == 4 and current_line:
                line_text = ' '.join(current_line)
                line_conf = sum(current_line_conf) / len(current_line_conf) if current_line_conf else 0
                result.lines.append((line_text, line_conf, bbox))
                current_paragraph.append(line_text)
                current_line = []
                current_line_conf = []
                
            # Paragraph level (level 3)
            elif level == 3 and current_paragraph:
                para_text = '\n'.join(current_paragraph)
                para_conf = sum(current_line_conf) / len(current_line_conf) if current_line_conf else 0
                result.paragraphs.append((para_text, para_conf, bbox))
                current_paragraph = []
        
        # Calculate overall confidence
        if result.words:
            word_confidences = [w[1] for w in result.words]
            result.confidence = sum(word_confidences) / len(word_confidences)
        
        # Add metadata
        result.metadata = {
            'total_words': len(result.words),
            'total_lines': len(result.lines),
            'total_paragraphs': len(result.paragraphs),
            'language': data.get('lang', self.config['lang'])
        }
        
        return result
    
    def _store_successful_config(self, image_path: Path, config: str, confidence: float):
        """Store successful configurations for pattern learning"""
        # This could be enhanced to analyze image characteristics
        # and store which configs work best for different types
        pattern_key = f"confidence_{int(confidence * 10)}"
        
        if pattern_key not in self.successful_configs:
            self.successful_configs[pattern_key] = []
            
        self.successful_configs[pattern_key].append({
            'config': config,
            'confidence': confidence,
            'image_characteristics': self._analyze_image_characteristics(image_path)
        })
    
    def _analyze_image_characteristics(self, image_path: Path) -> Dict:
        """Analyze image characteristics for pattern learning"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            return {
                'resolution': image.shape,
                'mean_brightness': np.mean(image),
                'std_brightness': np.std(image),
                'has_tables': self._detect_tables_simple(image),
                'text_density': self._estimate_text_density(image)
            }
        except:
            return {}
    
    def _detect_tables_simple(self, image: np.ndarray) -> bool:
        """Simple table detection using line detection"""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        return lines is not None and len(lines) > 10
    
    def _estimate_text_density(self, image: np.ndarray) -> float:
        """Estimate text density in image"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_pixels = np.sum(binary == 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        return text_pixels / total_pixels if total_pixels > 0 else 0
    
    def get_optimal_config(self, image_characteristics: Dict) -> Dict:
        """Get optimal configuration based on image characteristics"""
        # Start with default config
        optimal_config = self.config.copy()
        
        # Adjust based on characteristics
        if image_characteristics.get('has_tables'):
            optimal_config['psm'] = self.PSM_MODES['single_block']
            
        if image_characteristics.get('text_density', 0) < 0.1:
            optimal_config['psm'] = self.PSM_MODES['sparse_text']
            
        return optimal_config
    
    def supports_language(self, language_code: str) -> bool:
        """Check if Tesseract supports a language"""
        try:
            languages = pytesseract.get_languages()
            return language_code in languages
        except:
            return language_code == 'eng'  # Default to English support