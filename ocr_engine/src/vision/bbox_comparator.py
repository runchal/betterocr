"""
Bounding Box Comparator
Visual validation of OCR bounding boxes using computer vision
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class BoundingBoxComparator:
    """Validates OCR bounding boxes using computer vision techniques"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'overlap_threshold': 0.5,
            'size_variance_threshold': 0.3,
            'aspect_ratio_tolerance': 0.2,
            'enable_text_detection': True,
            'enable_contour_analysis': True,
            'confidence_boost_factor': 0.2,
            'penalty_factor': 0.1
        }
        
        if config:
            self.config = {**default_config, **config}
        else:
            self.config = default_config
    
    def validate_bboxes(self, image: np.ndarray, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate OCR bounding boxes against visual evidence
        
        Args:
            image: Source document image
            ocr_results: Dictionary of OCR engine results
            
        Returns:
            Validation results with scores and recommendations
        """
        validation_results = {
            'engine_scores': {},
            'bbox_quality_scores': {},
            'visual_consistency': {},
            'text_detection_validation': {},
            'recommendations': {}
        }
        
        try:
            # Detect text regions using computer vision
            cv_text_regions = self._detect_text_regions_cv(image)
            
            # Validate each engine's bounding boxes
            for engine_name, ocr_result in ocr_results.items():
                if hasattr(ocr_result, 'words') and ocr_result.words:
                    engine_validation = self._validate_engine_bboxes(
                        ocr_result.words, cv_text_regions, image
                    )
                    validation_results['engine_scores'][engine_name] = engine_validation
            
            # Cross-engine bbox comparison
            validation_results['visual_consistency'] = self._compare_cross_engine_bboxes(ocr_results)
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_bbox_recommendations(
                validation_results
            )
            
        except Exception as e:
            logger.error(f"Bbox validation failed: {e}")
        
        return validation_results
    
    def _detect_text_regions_cv(self, image: np.ndarray) -> List[Tuple]:
        """Detect text regions using computer vision"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Convert regions to bounding boxes
        text_regions = []
        for region in regions:
            if len(region) > 10:  # Filter small regions
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter by size and aspect ratio
                aspect_ratio = w / h
                if (w > 10 and h > 5 and 
                    0.1 < aspect_ratio < 20):  # Reasonable text aspect ratios
                    text_regions.append((x, y, w, h))
        
        # Alternative: Use morphological operations
        morph_regions = self._detect_text_morph(gray)
        text_regions.extend(morph_regions)
        
        # Remove duplicates and merge overlapping regions
        text_regions = self._merge_overlapping_regions(text_regions)
        
        return text_regions
    
    def _detect_text_morph(self, gray_image: np.ndarray) -> List[Tuple]:
        """Detect text using morphological operations"""
        # Apply morphological closing to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if w > 15 and h > 8:
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 15:  # Text-like aspect ratios
                    text_regions.append((x, y, w, h))
        
        return text_regions
    
    def _merge_overlapping_regions(self, regions: List[Tuple], overlap_threshold: float = 0.3) -> List[Tuple]:
        """Merge overlapping text regions"""
        if not regions:
            return []
        
        merged = []
        regions = sorted(regions, key=lambda x: x[0])  # Sort by x-coordinate
        
        for region in regions:
            merged_with_existing = False
            
            for i, existing in enumerate(merged):
                if self._calculate_bbox_overlap(region, existing) > overlap_threshold:
                    # Merge regions
                    x1, y1, w1, h1 = existing
                    x2, y2, w2, h2 = region
                    
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    
                    merged[i] = (new_x, new_y, new_w, new_h)
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append(region)
        
        return merged
    
    def _validate_engine_bboxes(self, ocr_words: List, cv_regions: List[Tuple], image: np.ndarray) -> Dict:
        """Validate an engine's bounding boxes against CV detected regions"""
        validation_scores = {
            'overlap_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'bbox_quality': 0.0,
            'detailed_scores': []
        }
        
        if not ocr_words:
            return validation_scores
        
        # Calculate overlap scores for each OCR word
        word_scores = []
        matched_cv_regions = set()
        
        for word_data in ocr_words:
            word_text, word_conf, word_bbox = word_data[:3]
            
            # Find best matching CV region
            best_overlap = 0.0
            best_cv_region = None
            
            for i, cv_region in enumerate(cv_regions):
                if i not in matched_cv_regions:
                    overlap = self._calculate_bbox_overlap(word_bbox, cv_region)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_cv_region = i
            
            # Calculate quality scores
            quality_score = self._calculate_bbox_quality(word_bbox, image)
            
            word_score = {
                'text': word_text,
                'confidence': word_conf,
                'overlap': best_overlap,
                'quality': quality_score,
                'combined_score': (best_overlap * 0.7) + (quality_score * 0.3)
            }
            
            word_scores.append(word_score)
            
            if best_overlap > self.config['overlap_threshold'] and best_cv_region is not None:
                matched_cv_regions.add(best_cv_region)
        
        # Calculate aggregate scores
        if word_scores:
            validation_scores['overlap_score'] = np.mean([s['overlap'] for s in word_scores])
            validation_scores['bbox_quality'] = np.mean([s['quality'] for s in word_scores])
            
            # Precision: How many OCR boxes match CV regions
            good_matches = sum(1 for s in word_scores if s['overlap'] > self.config['overlap_threshold'])
            validation_scores['precision'] = good_matches / len(word_scores)
            
            # Recall: How many CV regions are matched by OCR boxes
            validation_scores['recall'] = len(matched_cv_regions) / len(cv_regions) if cv_regions else 0.0
            
            # F1 Score
            precision = validation_scores['precision']
            recall = validation_scores['recall']
            if precision + recall > 0:
                validation_scores['f1_score'] = 2 * (precision * recall) / (precision + recall)
        
        validation_scores['detailed_scores'] = word_scores
        return validation_scores
    
    def _calculate_bbox_quality(self, bbox: Tuple, image: np.ndarray) -> float:
        """Calculate quality score for a bounding box based on visual content"""
        try:
            x, y, w, h = bbox[:4]
            
            # Ensure bbox is within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return 0.0
            
            # Extract region
            region = image[y:y+h, x:x+w]
            
            if region.size == 0:
                return 0.0
            
            # Convert to grayscale if needed
            if len(region.shape) == 3:
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = region
            
            # Calculate various quality metrics
            scores = []
            
            # 1. Edge density (text regions should have good edge content)
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            scores.append(min(1.0, edge_density * 5))  # Scale factor
            
            # 2. Contrast (text should have good contrast)
            contrast = np.std(gray_region)
            normalized_contrast = min(1.0, contrast / 50)  # Normalize to 0-1
            scores.append(normalized_contrast)
            
            # 3. Size appropriateness (not too small, not too large)
            area = w * h
            if 100 <= area <= 10000:  # Reasonable text area
                size_score = 1.0
            elif area < 100:
                size_score = area / 100
            else:
                size_score = max(0.1, 10000 / area)
            scores.append(size_score)
            
            # 4. Aspect ratio (text should have reasonable aspect ratio)
            aspect_ratio = w / h
            if 0.5 <= aspect_ratio <= 10:  # Reasonable for text
                aspect_score = 1.0
            else:
                aspect_score = max(0.1, min(1.0, 1.0 / abs(np.log(aspect_ratio))))
            scores.append(aspect_score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"Error calculating bbox quality: {e}")
            return 0.5
    
    def _compare_cross_engine_bboxes(self, ocr_results: Dict[str, Any]) -> Dict:
        """Compare bounding boxes across different OCR engines"""
        consistency_scores = {}
        engine_names = list(ocr_results.keys())
        
        for i, engine1 in enumerate(engine_names):
            for j, engine2 in enumerate(engine_names[i+1:], i+1):
                if (hasattr(ocr_results[engine1], 'words') and 
                    hasattr(ocr_results[engine2], 'words')):
                    
                    consistency = self._calculate_bbox_consistency(
                        ocr_results[engine1].words,
                        ocr_results[engine2].words
                    )
                    consistency_scores[f"{engine1}_vs_{engine2}"] = consistency
        
        return {
            'pairwise_consistency': consistency_scores,
            'average_consistency': np.mean(list(consistency_scores.values())) if consistency_scores else 0.0
        }
    
    def _calculate_bbox_consistency(self, words1: List, words2: List) -> float:
        """Calculate consistency between two sets of word bounding boxes"""
        if not words1 or not words2:
            return 0.0
        
        matches = 0
        total_words = len(words1)
        
        for word1_data in words1:
            word1_text, _, word1_bbox = word1_data[:3]
            best_match_score = 0.0
            
            for word2_data in words2:
                word2_text, _, word2_bbox = word2_data[:3]
                
                # Text similarity
                text_sim = self._calculate_text_similarity(word1_text, word2_text)
                
                # Spatial overlap
                spatial_overlap = self._calculate_bbox_overlap(word1_bbox, word2_bbox)
                
                # Combined score
                combined_score = (text_sim * 0.6) + (spatial_overlap * 0.4)
                best_match_score = max(best_match_score, combined_score)
            
            if best_match_score > 0.7:  # Threshold for considering a match
                matches += 1
        
        return matches / total_words
    
    def _generate_bbox_recommendations(self, validation_results: Dict) -> Dict:
        """Generate recommendations based on bbox validation results"""
        recommendations = {
            'engine_rankings': {},
            'confidence_adjustments': {},
            'quality_warnings': [],
            'improvement_suggestions': []
        }
        
        # Rank engines by bbox quality
        engine_scores = validation_results.get('engine_scores', {})
        if engine_scores:
            ranked_engines = sorted(
                engine_scores.items(),
                key=lambda x: x[1].get('f1_score', 0.0),
                reverse=True
            )
            recommendations['engine_rankings'] = {
                engine: rank for rank, (engine, _) in enumerate(ranked_engines, 1)
            }
            
            # Suggest confidence adjustments
            for engine, scores in engine_scores.items():
                f1_score = scores.get('f1_score', 0.0)
                if f1_score > 0.8:
                    recommendations['confidence_adjustments'][engine] = 1.2  # Boost confidence
                elif f1_score < 0.4:
                    recommendations['confidence_adjustments'][engine] = 0.8  # Reduce confidence
                else:
                    recommendations['confidence_adjustments'][engine] = 1.0  # No change
        
        # Quality warnings
        for engine, scores in engine_scores.items():
            bbox_quality = scores.get('bbox_quality', 0.0)
            if bbox_quality < 0.5:
                recommendations['quality_warnings'].append(
                    f"{engine}: Low bbox quality (score: {bbox_quality:.2f})"
                )
        
        # Improvement suggestions
        consistency = validation_results.get('visual_consistency', {}).get('average_consistency', 0.0)
        if consistency < 0.6:
            recommendations['improvement_suggestions'].append(
                "Low inter-engine consistency detected. Consider adjusting OCR parameters."
            )
        
        return recommendations
    
    def _calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
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
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()