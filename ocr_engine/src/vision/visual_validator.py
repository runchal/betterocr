"""
Visual Validator
Main orchestrator for computer vision-based OCR validation
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image

from .layout_analyzer import LayoutAnalyzer
from .bbox_comparator import BoundingBoxComparator
from .image_quality_assessor import ImageQualityAssessor

logger = logging.getLogger(__name__)


class VisualValidator:
    """Main visual validation system for OCR results"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'enable_layout_analysis': True,
            'enable_bbox_validation': True,
            'enable_quality_assessment': True,
            'enable_text_line_detection': True,
            'confidence_boost_threshold': 0.8,
            'visual_confidence_weight': 0.3,
            'bbox_overlap_threshold': 0.5,
            'layout_consistency_threshold': 0.7
        }
        
        if config:
            self.config = {**default_config, **config}
        else:
            self.config = default_config
        
        # Initialize sub-components
        self.layout_analyzer = LayoutAnalyzer(config)
        self.bbox_comparator = BoundingBoxComparator(config)
        self.quality_assessor = ImageQualityAssessor(config)
        
    def validate_ocr_results(self, image_path: Path, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive visual validation of OCR results
        
        Args:
            image_path: Path to the source image
            ocr_results: Dictionary of OCR engine results
            
        Returns:
            Enhanced results with visual validation scores
        """
        logger.debug(f"Starting visual validation for {image_path}")
        
        try:
            # Load image (handle both images and PDFs)
            image = self._load_image_or_pdf(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return self._create_empty_validation()
            
            validation_results = {
                'image_quality': {},
                'layout_analysis': {},
                'bbox_validation': {},
                'text_line_validation': {},
                'consensus_enhancement': {},
                'visual_confidence_scores': {},
                'cross_engine_validation': {}
            }
            
            # 1. Image Quality Assessment
            if self.config['enable_quality_assessment']:
                try:
                    validation_results['image_quality'] = self.quality_assessor.assess_image_quality(image)
                except Exception as e:
                    logger.error(f"Image quality assessment failed: {e}")
                    validation_results['image_quality'] = {}
            
            # 2. Layout Analysis
            if self.config['enable_layout_analysis']:
                try:
                    validation_results['layout_analysis'] = self.layout_analyzer.analyze_layout(image)
                except Exception as e:
                    logger.error(f"Layout analysis failed: {e}")
                    validation_results['layout_analysis'] = {}
            
            # 3. Bounding Box Validation
            if self.config['enable_bbox_validation']:
                try:
                    validation_results['bbox_validation'] = self.bbox_comparator.validate_bboxes(
                        image, ocr_results
                    )
                except Exception as e:
                    logger.error(f"Bbox validation failed: {e}")
                    validation_results['bbox_validation'] = {}
            
            # 4. Text Line Detection and Validation
            if self.config['enable_text_line_detection']:
                try:
                    validation_results['text_line_validation'] = self._validate_text_lines(
                        image, ocr_results
                    )
                except Exception as e:
                    logger.error(f"Text line validation failed: {e}")
                    validation_results['text_line_validation'] = {}
            
            # 5. Cross-Engine Visual Consistency
            try:
                validation_results['cross_engine_validation'] = self._validate_cross_engine_consistency(
                    image, ocr_results
                )
            except Exception as e:
                logger.error(f"Cross-engine validation failed: {e}")
                validation_results['cross_engine_validation'] = {}
            
            # 6. Enhanced Consensus Building
            try:
                validation_results['consensus_enhancement'] = self._enhance_consensus_with_vision(
                    ocr_results, validation_results
                )
            except Exception as e:
                logger.error(f"Consensus enhancement failed: {e}")
                validation_results['consensus_enhancement'] = {}
            
            # 7. Calculate Visual Confidence Scores
            try:
                validation_results['visual_confidence_scores'] = self._calculate_visual_confidence(
                    validation_results
                )
            except Exception as e:
                logger.error(f"Visual confidence calculation failed: {e}")
                validation_results['visual_confidence_scores'] = {}
            
            logger.debug("Visual validation completed successfully")
            return validation_results
            
        except Exception as e:
            logger.error(f"Visual validation failed: {e}")
            return self._create_empty_validation()
    
    def _validate_text_lines(self, image: np.ndarray, ocr_results: Dict) -> Dict:
        """Validate detected text lines using computer vision"""
        # Detect text lines using OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (potential text lines)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        text_line_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = w * h
            
            # Filter for text-like regions
            if aspect_ratio > 2 and area > 100:
                text_line_contours.append((x, y, w, h))
        
        # Compare with OCR detected lines
        validation_scores = {}
        for engine_name, result in ocr_results.items():
            if hasattr(result, 'lines') and result.lines:
                score = self._compare_lines_to_cv_detection(result.lines, text_line_contours)
                validation_scores[engine_name] = score
        
        return {
            'cv_detected_lines': len(text_line_contours),
            'line_contours': text_line_contours,
            'engine_line_scores': validation_scores
        }
    
    def _compare_lines_to_cv_detection(self, ocr_lines: List, cv_contours: List) -> float:
        """Compare OCR detected lines with CV detected text regions"""
        if not ocr_lines or not cv_contours:
            return 0.0
        
        matches = 0
        for line_text, line_conf, line_bbox in ocr_lines:
            best_overlap = 0
            for cv_bbox in cv_contours:
                overlap = self._calculate_bbox_overlap(line_bbox, cv_bbox)
                best_overlap = max(best_overlap, overlap)
            
            if best_overlap > self.config['bbox_overlap_threshold']:
                matches += 1
        
        return matches / len(ocr_lines)
    
    def _validate_cross_engine_consistency(self, image: np.ndarray, ocr_results: Dict) -> Dict:
        """Validate consistency between engines using visual cues"""
        consistency_scores = {}
        
        # Compare bounding boxes between engines
        engine_names = list(ocr_results.keys())
        for i, engine1 in enumerate(engine_names):
            for j, engine2 in enumerate(engine_names[i+1:], i+1):
                if (hasattr(ocr_results[engine1], 'words') and 
                    hasattr(ocr_results[engine2], 'words')):
                    
                    score = self._compare_engine_bboxes(
                        ocr_results[engine1].words,
                        ocr_results[engine2].words
                    )
                    consistency_scores[f"{engine1}_vs_{engine2}"] = score
        
        return {
            'bbox_consistency_scores': consistency_scores,
            'average_consistency': np.mean(list(consistency_scores.values())) if consistency_scores else 0.0
        }
    
    def _compare_engine_bboxes(self, words1: List, words2: List) -> float:
        """Compare bounding boxes between two engines"""
        if not words1 or not words2:
            return 0.0
        
        matches = 0
        total_comparisons = 0
        
        for word1_text, word1_conf, word1_bbox in words1:
            best_match = 0
            for word2_text, word2_conf, word2_bbox in words2:
                # Check text similarity
                text_sim = self._text_similarity(word1_text, word2_text)
                bbox_overlap = self._calculate_bbox_overlap(word1_bbox, word2_bbox)
                
                combined_score = (text_sim * 0.6) + (bbox_overlap * 0.4)
                best_match = max(best_match, combined_score)
            
            matches += best_match
            total_comparisons += 1
        
        return matches / total_comparisons if total_comparisons > 0 else 0.0
    
    def _enhance_consensus_with_vision(self, ocr_results: Dict, validation_results: Dict) -> Dict:
        """Enhance consensus building using visual validation results"""
        enhanced_weights = {}
        
        # Adjust engine weights based on visual validation
        for engine_name in ocr_results.keys():
            base_weight = 1.0
            
            # Adjust based on bbox validation
            if 'bbox_validation' in validation_results:
                bbox_scores = validation_results['bbox_validation'].get('engine_scores', {})
                if engine_name in bbox_scores:
                    bbox_score = bbox_scores[engine_name]
                    # Ensure bbox_score is numeric
                    if isinstance(bbox_score, (int, float)):
                        base_weight *= (1.0 + bbox_score * 0.2)  # Up to 20% boost
            
            # Adjust based on line detection accuracy
            if 'text_line_validation' in validation_results:
                line_scores = validation_results['text_line_validation'].get('engine_line_scores', {})
                if engine_name in line_scores:
                    line_score = line_scores[engine_name]
                    # Ensure line_score is numeric
                    if isinstance(line_score, (int, float)):
                        base_weight *= (1.0 + line_score * 0.15)  # Up to 15% boost
            
            enhanced_weights[engine_name] = base_weight
        
        return {
            'enhanced_weights': enhanced_weights,
            'weight_adjustments': {
                name: weight - 1.0 for name, weight in enhanced_weights.items()
            }
        }
    
    def _calculate_visual_confidence(self, validation_results: Dict) -> Dict:
        """Calculate overall visual confidence scores"""
        scores = {}
        
        # Image quality contribution
        quality_score = validation_results.get('image_quality', {}).get('overall_quality', 0.0)
        
        # Layout consistency contribution
        layout_score = validation_results.get('layout_analysis', {}).get('confidence', 0.0)
        
        # Cross-engine consistency contribution
        consistency_score = validation_results.get('cross_engine_validation', {}).get('average_consistency', 0.0)
        
        # Combined visual confidence
        visual_confidence = (quality_score * 0.3 + layout_score * 0.4 + consistency_score * 0.3)
        
        scores['visual_confidence'] = visual_confidence
        scores['quality_contribution'] = quality_score
        scores['layout_contribution'] = layout_score
        scores['consistency_contribution'] = consistency_score
        
        return scores
    
    def _calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap between two bounding boxes"""
        try:
            x1, y1, w1, h1 = bbox1[:4]
            x2, y2, w2, h2 = bbox2[:4]
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _load_image_or_pdf(self, file_path: Path) -> Optional[np.ndarray]:
        """Load image from file path, handling both images and PDFs"""
        file_str = str(file_path)
        
        # Try loading as image first
        if file_str.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return cv2.imread(file_str)
        
        # Handle PDF files
        if file_str.lower().endswith('.pdf'):
            try:
                from pdf2image import convert_from_path
                # Convert first page of PDF to image
                images = convert_from_path(file_str, first_page=1, last_page=1, dpi=300)
                if images:
                    # Convert PIL image to OpenCV format
                    pil_image = images[0]
                    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    return opencv_image
            except Exception as e:
                logger.error(f"Failed to convert PDF to image: {e}")
                return None
        
        # Try loading as generic image
        return cv2.imread(file_str)
    
    def _create_empty_validation(self) -> Dict:
        """Create empty validation result"""
        return {
            'image_quality': {},
            'layout_analysis': {},
            'bbox_validation': {},
            'text_line_validation': {},
            'consensus_enhancement': {},
            'visual_confidence_scores': {},
            'cross_engine_validation': {}
        }