"""
Layout Analyzer
Advanced document layout analysis using computer vision
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
from sklearn.cluster import DBSCAN
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    lp = None
    LAYOUTPARSER_AVAILABLE = False

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """Advanced layout analysis for document structure understanding"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'min_text_height': 10,
            'min_text_width': 20,
            'line_clustering_eps': 15,
            'column_detection_threshold': 0.7,
            'table_detection_enabled': True,
            'figure_detection_enabled': True,
            'use_layoutparser': True
        }
        
        if config:
            self.config = {**default_config, **config}
        else:
            self.config = default_config
        self.layout_model = None
        
        # Initialize LayoutParser if available
        if self.config['use_layoutparser'] and LAYOUTPARSER_AVAILABLE:
            try:
                # Check if detectron2 is available
                if lp.is_detectron2_available():
                    # Use AutoLayoutModel instead of deprecated Detectron2LayoutModel
                    self.layout_model = lp.AutoLayoutModel(
                        model_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                    )
                    logger.info("LayoutParser AutoLayoutModel loaded successfully")
                else:
                    logger.warning("LayoutParser available but Detectron2 not installed - using OpenCV-only analysis")
                    self.layout_model = None
            except Exception as e:
                logger.warning(f"LayoutParser model loading failed: {e} - using OpenCV-only analysis")
                self.layout_model = None
        else:
            self.layout_model = None
            if not LAYOUTPARSER_AVAILABLE:
                logger.info("LayoutParser not installed - using OpenCV-only layout analysis")
    
    def analyze_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive layout analysis of document image
        
        Args:
            image: Input document image
            
        Returns:
            Dictionary containing layout analysis results
        """
        layout_results = {
            'text_regions': [],
            'columns': [],
            'tables': [],
            'figures': [],
            'reading_order': [],
            'layout_confidence': 0.0,
            'document_type': 'unknown'
        }
        
        try:
            # Basic OpenCV-based analysis
            cv_analysis = self._opencv_layout_analysis(image)
            layout_results.update(cv_analysis)
            
            # LayoutParser analysis if available
            if self.layout_model is not None:
                lp_analysis = self._layoutparser_analysis(image)
                layout_results = self._merge_layout_results(layout_results, lp_analysis)
            
            # Document type classification
            layout_results['document_type'] = self._classify_document_type(layout_results)
            
            # Overall confidence calculation
            layout_results['confidence'] = self._calculate_layout_confidence(layout_results)
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
        
        return layout_results
    
    def _opencv_layout_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic layout analysis using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Text region detection
        text_regions = self._detect_text_regions(gray)
        
        # Column detection
        columns = self._detect_columns(text_regions)
        
        # Table detection
        tables = self._detect_tables(gray)
        
        # Figure detection (images, charts)
        figures = self._detect_figures(gray)
        
        # Reading order estimation
        reading_order = self._estimate_reading_order(text_regions, columns)
        
        return {
            'text_regions': text_regions,
            'columns': columns,
            'tables': tables,
            'figures': figures,
            'reading_order': reading_order
        }
    
    def _detect_text_regions(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect text regions using morphological operations"""
        # Create kernel for text detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Apply morphological operations
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (w >= self.config['min_text_width'] and 
                h >= self.config['min_text_height']):
                
                text_regions.append({
                    'bbox': (x, y, w, h),
                    'area': w * h,
                    'aspect_ratio': w / h,
                    'type': 'text'
                })
        
        return text_regions
    
    def _detect_columns(self, text_regions: List[Dict]) -> List[Dict]:
        """Detect column structure in the document"""
        if not text_regions:
            return []
        
        # Extract x-coordinates of text regions
        x_coords = [region['bbox'][0] for region in text_regions]
        x_widths = [region['bbox'][2] for region in text_regions]
        
        # Use DBSCAN to cluster x-coordinates
        X = np.array(x_coords).reshape(-1, 1)
        clustering = DBSCAN(eps=50, min_samples=2).fit(X)
        
        columns = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise
                continue
            
            # Get regions in this cluster
            cluster_regions = [region for i, region in enumerate(text_regions) 
                             if clustering.labels_[i] == cluster_id]
            
            if cluster_regions:
                # Calculate column bounds
                min_x = min(region['bbox'][0] for region in cluster_regions)
                max_x = max(region['bbox'][0] + region['bbox'][2] for region in cluster_regions)
                min_y = min(region['bbox'][1] for region in cluster_regions)
                max_y = max(region['bbox'][1] + region['bbox'][3] for region in cluster_regions)
                
                columns.append({
                    'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                    'region_count': len(cluster_regions),
                    'type': 'column'
                })
        
        return sorted(columns, key=lambda x: x['bbox'][0])  # Sort by x-coordinate
    
    def _detect_tables(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect table structures using line detection"""
        if not self.config['table_detection_enabled']:
            return []
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours of potential tables
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if w > 100 and h > 50:
                tables.append({
                    'bbox': (x, y, w, h),
                    'type': 'table',
                    'confidence': self._calculate_table_confidence(gray_image[y:y+h, x:x+w])
                })
        
        return tables
    
    def _detect_figures(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect figures, images, and charts"""
        if not self.config['figure_detection_enabled']:
            return []
        
        # Use edge detection to find potential figures
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Dilate edges to connect nearby features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        figures = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and compactness
            area = cv2.contourArea(contour)
            rect_area = w * h
            compactness = area / rect_area if rect_area > 0 else 0
            
            if w > 50 and h > 50 and compactness > 0.3:
                figures.append({
                    'bbox': (x, y, w, h),
                    'type': 'figure',
                    'compactness': compactness
                })
        
        return figures
    
    def _estimate_reading_order(self, text_regions: List[Dict], columns: List[Dict]) -> List[int]:
        """Estimate reading order of text regions"""
        if not text_regions:
            return []
        
        # Sort by column first, then by y-coordinate within column
        if columns:
            # Multi-column layout
            ordered_regions = []
            for column in columns:
                col_x, col_y, col_w, col_h = column['bbox']
                
                # Find regions in this column
                col_regions = []
                for i, region in enumerate(text_regions):
                    reg_x, reg_y, reg_w, reg_h = region['bbox']
                    if (reg_x >= col_x and reg_x + reg_w <= col_x + col_w):
                        col_regions.append((i, region))
                
                # Sort by y-coordinate within column
                col_regions.sort(key=lambda x: x[1]['bbox'][1])
                ordered_regions.extend([i for i, _ in col_regions])
            
            return ordered_regions
        else:
            # Single column layout - sort by y-coordinate
            indexed_regions = [(i, region) for i, region in enumerate(text_regions)]
            indexed_regions.sort(key=lambda x: x[1]['bbox'][1])
            return [i for i, _ in indexed_regions]
    
    def _layoutparser_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced layout analysis using LayoutParser"""
        try:
            # Convert to PIL Image
            from PIL import Image as PILImage
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            pil_image = PILImage.fromarray(image_rgb)
            
            # Detect layout
            layout = self.layout_model.detect(pil_image)
            
            # Convert to our format
            lp_results = {
                'lp_text_regions': [],
                'lp_tables': [],
                'lp_figures': [],
                'lp_titles': [],
                'lp_lists': []
            }
            
            for block in layout:
                bbox = (int(block.block.x_1), int(block.block.y_1), 
                       int(block.block.width), int(block.block.height))
                
                element = {
                    'bbox': bbox,
                    'type': block.type,
                    'confidence': block.score,
                    'source': 'layoutparser'
                }
                
                if block.type == 'Text':
                    lp_results['lp_text_regions'].append(element)
                elif block.type == 'Table':
                    lp_results['lp_tables'].append(element)
                elif block.type == 'Figure':
                    lp_results['lp_figures'].append(element)
                elif block.type == 'Title':
                    lp_results['lp_titles'].append(element)
                elif block.type == 'List':
                    lp_results['lp_lists'].append(element)
            
            return lp_results
            
        except Exception as e:
            logger.error(f"LayoutParser analysis failed: {e}")
            return {}
    
    def _merge_layout_results(self, cv_results: Dict, lp_results: Dict) -> Dict:
        """Merge OpenCV and LayoutParser results"""
        merged = cv_results.copy()
        
        # Add LayoutParser results
        merged.update(lp_results)
        
        # Enhance confidence based on agreement
        if 'lp_tables' in lp_results and lp_results['lp_tables']:
            for table in merged['tables']:
                # Check if OpenCV table matches LayoutParser table
                best_overlap = 0
                for lp_table in lp_results['lp_tables']:
                    overlap = self._calculate_bbox_overlap(table['bbox'], lp_table['bbox'])
                    best_overlap = max(best_overlap, overlap)
                
                if best_overlap > 0.5:
                    table['confidence'] = min(1.0, table.get('confidence', 0.5) + 0.3)
        
        return merged
    
    def _classify_document_type(self, layout_results: Dict) -> str:
        """Classify document type based on layout analysis"""
        text_regions = len(layout_results.get('text_regions', []))
        tables = len(layout_results.get('tables', []))
        figures = len(layout_results.get('figures', []))
        columns = len(layout_results.get('columns', []))
        
        # Simple heuristics for document classification
        if tables > 2:
            return 'form_or_invoice'
        elif columns > 1:
            return 'multi_column_document'
        elif figures > 1:
            return 'report_or_presentation'
        elif text_regions > 10:
            return 'text_document'
        else:
            return 'simple_document'
    
    def _calculate_layout_confidence(self, layout_results: Dict) -> float:
        """Calculate overall confidence in layout analysis"""
        scores = []
        
        # Text region detection confidence
        text_regions = layout_results.get('text_regions', [])
        if text_regions:
            scores.append(0.8)  # Base confidence for text detection
        
        # Table detection confidence
        tables = layout_results.get('tables', [])
        if tables:
            table_confidences = [t.get('confidence', 0.5) for t in tables]
            scores.append(np.mean(table_confidences))
        
        # LayoutParser confidence
        lp_elements = []
        for key in ['lp_text_regions', 'lp_tables', 'lp_figures', 'lp_titles', 'lp_lists']:
            lp_elements.extend(layout_results.get(key, []))
        
        if lp_elements:
            lp_confidences = [elem.get('confidence', 0.5) for elem in lp_elements]
            scores.append(np.mean(lp_confidences))
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_table_confidence(self, table_region: np.ndarray) -> float:
        """Calculate confidence that a region contains a table"""
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        horizontal = cv2.morphologyEx(table_region, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(table_region, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count line pixels
        h_pixels = np.sum(horizontal > 0)
        v_pixels = np.sum(vertical > 0)
        total_pixels = table_region.shape[0] * table_region.shape[1]
        
        # Calculate confidence based on line density
        line_density = (h_pixels + v_pixels) / total_pixels
        return min(1.0, line_density * 10)  # Scale factor
    
    def _calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
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