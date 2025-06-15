"""
Layout analysis using computer vision
Detects document structure, tables, and text regions
"""

import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """Analyze document layout using computer vision techniques"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_image(self, image_path):
        """
        Analyze document layout
        
        Returns:
            Dictionary with layout information
        """
        image_path = Path(image_path)
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return {}
        
        results = {
            'image_path': str(image_path),
            'dimensions': {
                'height': image.shape[0],
                'width': image.shape[1]
            },
            'regions': [],
            'tables': [],
            'text_blocks': []
        }
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect text regions
        text_regions = self._detect_text_regions(gray)
        results['text_blocks'] = text_regions
        
        # Detect tables
        tables = self._detect_tables(gray)
        results['tables'] = tables
        
        # Detect document structure
        structure = self._analyze_structure(gray)
        results['structure'] = structure
        
        return results
    
    def _detect_text_regions(self, gray_image):
        """Detect text regions in the image"""
        regions = []
        
        # Apply threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'type': 'text_region',
                    'bbox': [x, y, w, h],
                    'area': area,
                    'confidence': 0.8  # Placeholder confidence
                })
        
        # Sort regions by position (top to bottom, left to right)
        regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        return regions
    
    def _detect_tables(self, gray_image):
        """Detect tables using line detection"""
        tables = []
        
        # Detect horizontal and vertical lines
        horizontal = self._detect_lines(gray_image, horizontal=True)
        vertical = self._detect_lines(gray_image, horizontal=False)
        
        # Find intersections that might indicate tables
        if len(horizontal) > 2 and len(vertical) > 2:
            # Simple heuristic: if we have multiple intersecting lines, it might be a table
            tables.append({
                'type': 'table',
                'horizontal_lines': len(horizontal),
                'vertical_lines': len(vertical),
                'confidence': 0.7
            })
        
        return tables
    
    def _detect_lines(self, gray_image, horizontal=True):
        """Detect horizontal or vertical lines"""
        # Create structure element for morphology
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Apply morphology
        morphed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if horizontal and w > gray_image.shape[1] * 0.3:  # Line is at least 30% of image width
                lines.append([x, y, w, h])
            elif not horizontal and h > gray_image.shape[0] * 0.3:  # Line is at least 30% of image height
                lines.append([x, y, w, h])
        
        return lines
    
    def _analyze_structure(self, gray_image):
        """Analyze overall document structure"""
        height, width = gray_image.shape
        
        # Simple structure analysis
        structure = {
            'orientation': 'portrait' if height > width else 'landscape',
            'estimated_columns': 1,  # Placeholder
            'has_header': False,  # Placeholder
            'has_footer': False   # Placeholder
        }
        
        # Could add more sophisticated analysis here
        
        return structure