"""
PaddleOCR Engine Implementation
Excellent for structured documents and tables
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
from PIL import Image

from .base_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine with table recognition capabilities"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'use_angle_cls': True,
            'lang': 'en',
            'use_gpu': False,
            'det_algorithm': 'DB',  # Text detection algorithm
            'rec_algorithm': 'CRNN',  # Text recognition algorithm
            'det_db_thresh': 0.3,
            'det_db_box_thresh': 0.6,
            'det_db_unclip_ratio': 1.5,
            'max_text_length': 25,
            'rec_batch_num': 6,
            'use_space_char': True,
            'drop_score': 0.5,
            'enable_mkldnn': True,
            'cpu_threads': 10,
            'use_pdserving': False,
            'warmup': False,
            'draw_img_save_dir': None,
            'save_crop_res': False,
            'crop_res_save_dir': None,
            'use_mp': False,
            'total_process_num': 1,
            'process_id': 0,
            'show_log': False,
            'type': 'ocr',  # 'ocr', 'structure', 'table'
            'table': False,
            'layout': False,
            'ocr_version': 'PP-OCRv4'
        }
        
        if config:
            default_config.update(config)
            
        super().__init__('paddleocr', default_config)
        
        self.ocr = None
        self.table_engine = None
        self.layout_engine = None
        
    def _check_availability(self) -> bool:
        """Check if PaddleOCR is available"""
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError as e:
            logger.warning(f"PaddleOCR not available: {e}")
            # For now, disable PaddleOCR on Python 3.13 due to numpy compatibility
            import sys
            if sys.version_info >= (3, 13):
                logger.warning("PaddleOCR is not compatible with Python 3.13+ due to numpy version conflicts")
            return False
    
    def _load_engine(self):
        """Load PaddleOCR engine"""
        if self.ocr is None:
            try:
                from paddleocr import PaddleOCR
                
                logger.info("Loading PaddleOCR engine...")
                
                self.ocr = PaddleOCR(
                    use_angle_cls=self.config['use_angle_cls'],
                    lang=self.config['lang'],
                    use_gpu=self.config['use_gpu'],
                    det_algorithm=self.config['det_algorithm'],
                    rec_algorithm=self.config['rec_algorithm'],
                    det_db_thresh=self.config['det_db_thresh'],
                    det_db_box_thresh=self.config['det_db_box_thresh'],
                    det_db_unclip_ratio=self.config['det_db_unclip_ratio'],
                    max_text_length=self.config['max_text_length'],
                    rec_batch_num=self.config['rec_batch_num'],
                    use_space_char=self.config['use_space_char'],
                    drop_score=self.config['drop_score'],
                    enable_mkldnn=self.config['enable_mkldnn'],
                    cpu_threads=self.config['cpu_threads'],
                    show_log=self.config['show_log'],
                    ocr_version=self.config['ocr_version']
                )
                
                logger.info("PaddleOCR engine loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR: {e}")
                raise
    
    def _load_table_engine(self):
        """Load table recognition engine"""
        if self.table_engine is None and self.config.get('table'):
            try:
                from paddleocr import PPStructure
                
                self.table_engine = PPStructure(
                    table=True,
                    ocr=True,
                    show_log=self.config['show_log']
                )
                
                logger.info("PaddleOCR table engine loaded")
                
            except Exception as e:
                logger.warning(f"Failed to load table engine: {e}")
    
    def _extract_text_from_image(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text using PaddleOCR"""
        result = OCRResult()
        
        # Load engine
        self._load_engine()
        
        try:
            # Determine if we should use table/structure recognition
            use_table = kwargs.get('table', self.config.get('table', False))
            
            if use_table:
                result = self._extract_with_structure(image_path, **kwargs)
            else:
                result = self._extract_standard(image_path, **kwargs)
                
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            result.errors.append(str(e))
            
        return result
    
    def _extract_standard(self, image_path: Path, **kwargs) -> OCRResult:
        """Standard text extraction"""
        result = OCRResult()
        
        # Run OCR
        ocr_result = self.ocr.ocr(str(image_path), cls=True)
        
        if not ocr_result or not ocr_result[0]:
            return result
        
        # Parse results
        all_text = []
        all_words = []
        all_lines = []
        total_confidence = 0
        
        # PaddleOCR returns nested list: [[[box], (text, confidence)], ...]
        for page_result in ocr_result:
            if not page_result:
                continue
                
            for line in page_result:
                if len(line) < 2:
                    continue
                    
                bbox_points = line[0]
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]
                
                # Convert bbox format
                bbox = self._convert_bbox_format(bbox_points)
                
                # Store line result
                all_lines.append((text, confidence, bbox))
                all_text.append(text)
                total_confidence += confidence
                
                # Split into words
                words = text.split()
                if words:
                    word_conf = confidence  # Same confidence for all words in line
                    word_width = bbox[2] / len(words) if len(words) > 0 else bbox[2]
                    
                    for i, word in enumerate(words):
                        word_bbox = (
                            bbox[0] + i * word_width,
                            bbox[1],
                            word_width,
                            bbox[3]
                        )
                        all_words.append((word, word_conf, word_bbox))
        
        # Combine results
        result.text = '\n'.join(all_text)
        result.words = all_words
        result.lines = all_lines
        result.confidence = total_confidence / len(all_lines) if all_lines else 0.0
        
        # Add metadata
        result.metadata = {
            'algorithm': f"{self.config['det_algorithm']}+{self.config['rec_algorithm']}",
            'angle_classification': self.config['use_angle_cls'],
            'language': self.config['lang'],
            'version': self.config['ocr_version']
        }
        
        return result
    
    def _extract_with_structure(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract with structure/table recognition"""
        self._load_table_engine()
        
        if not self.table_engine:
            # Fallback to standard extraction
            return self._extract_standard(image_path, **kwargs)
        
        result = OCRResult()
        
        # Run structure analysis
        structure_result = self.table_engine(str(image_path))
        
        # Parse structure results
        all_text = []
        tables = []
        
        for item in structure_result:
            item_type = item.get('type', '')
            
            if item_type == 'table':
                # Extract table data
                table_data = self._parse_table(item)
                tables.append(table_data)
                
                # Add table text to overall text
                table_text = self._table_to_text(table_data)
                all_text.append(table_text)
                
            elif item_type in ['text', 'title', 'list']:
                # Regular text block
                text = item.get('res', {}).get('text', '')
                confidence = item.get('res', {}).get('confidence', 0.0)
                bbox = item.get('bbox', [0, 0, 0, 0])
                
                result.lines.append((text, confidence, bbox))
                all_text.append(text)
        
        result.text = '\n\n'.join(all_text)
        result.metadata['tables'] = tables
        result.metadata['structure_types'] = [item.get('type') for item in structure_result]
        
        return result
    
    def _convert_bbox_format(self, bbox_points: List) -> Tuple[int, int, int, int]:
        """Convert PaddleOCR bbox format to (x, y, w, h)"""
        if len(bbox_points) == 4:
            # Four corner points
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        return (0, 0, 0, 0)
    
    def _parse_table(self, table_item: Dict) -> Dict:
        """Parse table structure from PaddleOCR result"""
        table_data = {
            'bbox': table_item.get('bbox', []),
            'cells': [],
            'html': table_item.get('res', {}).get('html', ''),
            'confidence': table_item.get('res', {}).get('confidence', 0.0)
        }
        
        # Extract cell data if available
        # This would parse the HTML or structured data
        # For now, return the raw structure
        
        return table_data
    
    def _table_to_text(self, table_data: Dict) -> str:
        """Convert table data to readable text"""
        # Simple conversion - could be enhanced
        if table_data.get('html'):
            # Strip HTML tags for plain text
            import re
            text = re.sub('<.*?>', ' ', table_data['html'])
            return ' '.join(text.split())
        return ""
    
    def analyze_layout(self, image_path: Path) -> Dict:
        """Analyze document layout"""
        self._load_engine()
        
        # Use PaddleOCR's layout analysis
        layout_result = {
            'regions': [],
            'reading_order': [],
            'structure_type': 'unknown'
        }
        
        # Run detection without recognition for speed
        det_result = self.ocr.ocr(str(image_path), rec=False)
        
        if det_result and det_result[0]:
            for i, bbox in enumerate(det_result[0]):
                region = {
                    'id': i,
                    'bbox': self._convert_bbox_format(bbox),
                    'type': 'text',  # Could be enhanced with classification
                    'confidence': 0.9
                }
                layout_result['regions'].append(region)
            
            # Determine reading order based on position
            layout_result['reading_order'] = self._determine_reading_order(
                layout_result['regions']
            )
        
        return layout_result
    
    def _determine_reading_order(self, regions: List[Dict]) -> List[int]:
        """Determine reading order of regions"""
        # Simple top-to-bottom, left-to-right ordering
        sorted_regions = sorted(
            enumerate(regions),
            key=lambda x: (x[1]['bbox'][1], x[1]['bbox'][0])
        )
        
        return [idx for idx, _ in sorted_regions]
    
    def supports_language(self, language_code: str) -> bool:
        """Check if PaddleOCR supports a language"""
        supported_langs = [
            'ch', 'en', 'fr', 'german', 'korean', 'japan', 'chinese_cht',
            'ta', 'te', 'ka', 'ar', 'hi', 'mr', 'ne', 'ur', 'ru', 'rs_latin',
            'rs_cyrillic', 'bg', 'uk', 'be', 'kk', 'ab', 'ug', 'fa', 'pu',
            'ckb', 'ga', 'it', 'es', 'pt', 'oc', 'roa', 'ca', 'eu', 'sq',
            'kaa', 'az', 'bs', 'hr', 'cs', 'sk', 'sl', 'pl', 'hu', 'lt',
            'lv', 'et', 'ro', 'mt', 'nl', 'no', 'da', 'sv', 'is', 'fi',
            'tr', 'el', 'sr'
        ]
        
        return language_code in supported_langs