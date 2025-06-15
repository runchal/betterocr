"""
Image Quality Assessor
Comprehensive assessment of document image quality for OCR optimization
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
from skimage import filters, measure, morphology
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ImageQualityAssessor:
    """Comprehensive image quality assessment for OCR optimization"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'enable_blur_detection': True,
            'enable_noise_assessment': True,
            'enable_contrast_analysis': True,
            'enable_skew_detection': True,
            'enable_resolution_check': True,
            'min_resolution_dpi': 150,
            'blur_threshold': 100,
            'noise_threshold': 0.1,
            'contrast_threshold': 0.3,
            'skew_threshold': 2.0
        }
        
        if config:
            self.config = {**default_config, **config}
        else:
            self.config = default_config
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive image quality assessment
        
        Args:
            image: Input document image
            
        Returns:
            Dictionary containing quality metrics and recommendations
        """
        quality_metrics = {
            'overall_quality': 0.0,
            'blur_metrics': {},
            'noise_metrics': {},
            'contrast_metrics': {},
            'resolution_metrics': {},
            'skew_metrics': {},
            'text_quality_metrics': {},
            'recommendations': [],
            'optimization_suggestions': {}
        }
        
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 1. Blur Detection
            if self.config['enable_blur_detection']:
                quality_metrics['blur_metrics'] = self._assess_blur(gray)
            
            # 2. Noise Assessment
            if self.config['enable_noise_assessment']:
                quality_metrics['noise_metrics'] = self._assess_noise(gray)
            
            # 3. Contrast Analysis
            if self.config['enable_contrast_analysis']:
                quality_metrics['contrast_metrics'] = self._assess_contrast(gray)
            
            # 4. Resolution Check
            if self.config['enable_resolution_check']:
                quality_metrics['resolution_metrics'] = self._assess_resolution(image)
            
            # 5. Skew Detection
            if self.config['enable_skew_detection']:
                quality_metrics['skew_metrics'] = self._assess_skew(gray)
            
            # 6. Text-specific Quality Metrics
            quality_metrics['text_quality_metrics'] = self._assess_text_quality(gray)
            
            # 7. Calculate Overall Quality Score
            quality_metrics['overall_quality'] = self._calculate_overall_quality(quality_metrics)
            
            # 8. Generate Recommendations
            quality_metrics['recommendations'] = self._generate_quality_recommendations(quality_metrics)
            
            # 9. Optimization Suggestions
            quality_metrics['optimization_suggestions'] = self._generate_optimization_suggestions(quality_metrics)
            
        except Exception as e:
            logger.error(f"Image quality assessment failed: {e}")
        
        return quality_metrics
    
    def _assess_blur(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Assess image blur using multiple methods"""
        blur_metrics = {}
        
        # 1. Laplacian Variance (most common)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        blur_metrics['laplacian_variance'] = laplacian_var
        blur_metrics['is_blurry_laplacian'] = laplacian_var < self.config['blur_threshold']
        
        # 2. Sobel Gradient Magnitude
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        blur_metrics['sobel_variance'] = np.var(sobel_magnitude)
        
        # 3. FFT-based blur detection
        fft = np.fft.fft2(gray_image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        
        # High frequencies indicate sharpness
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        high_freq_region = magnitude_spectrum[center_h-20:center_h+20, center_w-20:center_w+20]
        blur_metrics['fft_high_freq_energy'] = np.mean(high_freq_region)
        
        # 4. Tenengrad Focus Measure
        gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.sqrt(gx**2 + gy**2)
        blur_metrics['tenengrad_focus'] = np.mean(tenengrad)
        
        # Normalize scores (0-1 scale)
        blur_metrics['blur_score'] = min(1.0, laplacian_var / 500)  # Normalize
        
        return blur_metrics
    
    def _assess_noise(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Assess image noise levels"""
        noise_metrics = {}
        
        # 1. Estimate noise using median filter
        median_filtered = cv2.medianBlur(gray_image, 5)
        noise_estimation = np.std(gray_image.astype(float) - median_filtered.astype(float))
        noise_metrics['median_filter_noise'] = noise_estimation
        
        # 2. Local standard deviation method
        kernel = np.ones((3, 3)) / 9
        local_mean = cv2.filter2D(gray_image.astype(float), -1, kernel)
        local_variance = cv2.filter2D((gray_image.astype(float) - local_mean)**2, -1, kernel)
        noise_metrics['local_std_noise'] = np.mean(np.sqrt(local_variance))
        
        # 3. Wavelet-based denoising assessment
        try:
            from skimage.restoration import denoise_wavelet
            denoised = denoise_wavelet(gray_image, sigma=None, mode='soft', method='VisuShrink')
            noise_metrics['wavelet_noise'] = np.std(gray_image.astype(float) - (denoised * 255).astype(float))
        except ImportError:
            noise_metrics['wavelet_noise'] = noise_estimation
        
        # 4. Signal-to-Noise Ratio estimation
        signal_power = np.mean(gray_image.astype(float)**2)
        noise_power = noise_estimation**2
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 50
        noise_metrics['estimated_snr'] = snr
        
        # Normalize noise score (0-1, where 1 is low noise)
        noise_metrics['noise_score'] = max(0.0, min(1.0, 1.0 - (noise_estimation / 50)))
        noise_metrics['is_noisy'] = noise_estimation > (self.config['noise_threshold'] * 255)
        
        return noise_metrics
    
    def _assess_contrast(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Assess image contrast and dynamic range"""
        contrast_metrics = {}
        
        # 1. Global contrast metrics
        contrast_metrics['std_deviation'] = np.std(gray_image)
        contrast_metrics['dynamic_range'] = np.max(gray_image) - np.min(gray_image)
        contrast_metrics['rms_contrast'] = np.sqrt(np.mean((gray_image - np.mean(gray_image))**2))
        
        # 2. Michelson contrast (for periodic patterns)
        max_val = np.max(gray_image)
        min_val = np.min(gray_image)
        if max_val + min_val > 0:
            contrast_metrics['michelson_contrast'] = (max_val - min_val) / (max_val + min_val)
        else:
            contrast_metrics['michelson_contrast'] = 0.0
        
        # 3. Weber contrast (for localized objects)
        background = np.median(gray_image)
        if background > 0:
            contrast_metrics['weber_contrast'] = (np.mean(gray_image) - background) / background
        else:
            contrast_metrics['weber_contrast'] = 0.0
        
        # 4. Local contrast analysis
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(gray_image.astype(float), -1, kernel)
        local_contrast = np.abs(gray_image.astype(float) - local_mean)
        contrast_metrics['average_local_contrast'] = np.mean(local_contrast)
        
        # 5. Histogram-based contrast
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / np.sum(hist)
        contrast_metrics['histogram_entropy'] = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
        
        # Normalize contrast score (0-1)
        contrast_metrics['contrast_score'] = min(1.0, contrast_metrics['std_deviation'] / 64)
        contrast_metrics['is_low_contrast'] = contrast_metrics['std_deviation'] < (self.config['contrast_threshold'] * 255)
        
        return contrast_metrics
    
    def _assess_resolution(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image resolution and DPI"""
        resolution_metrics = {}
        
        h, w = image.shape[:2]
        resolution_metrics['width_pixels'] = w
        resolution_metrics['height_pixels'] = h
        resolution_metrics['total_pixels'] = w * h
        resolution_metrics['aspect_ratio'] = w / h
        
        # Estimate DPI based on document size assumptions
        # Assuming typical document is 8.5x11 inches
        estimated_dpi_width = w / 8.5
        estimated_dpi_height = h / 11
        resolution_metrics['estimated_dpi'] = np.mean([estimated_dpi_width, estimated_dpi_height])
        
        # Resolution adequacy for OCR
        resolution_metrics['adequate_for_ocr'] = resolution_metrics['estimated_dpi'] >= self.config['min_resolution_dpi']
        
        # Pixel density categories
        if resolution_metrics['estimated_dpi'] < 150:
            resolution_metrics['quality_category'] = 'low'
        elif resolution_metrics['estimated_dpi'] < 300:
            resolution_metrics['quality_category'] = 'medium'
        else:
            resolution_metrics['quality_category'] = 'high'
        
        # Resolution score (0-1)
        resolution_metrics['resolution_score'] = min(1.0, resolution_metrics['estimated_dpi'] / 300)
        
        return resolution_metrics
    
    def _assess_skew(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Detect and measure document skew"""
        skew_metrics = {}
        
        try:
            # 1. Hough Line Transform method
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    if -30 < angle < 30:  # Filter reasonable angles
                        angles.append(angle)
                
                if angles:
                    skew_angle = np.median(angles)
                    skew_metrics['hough_skew_angle'] = skew_angle
                else:
                    skew_metrics['hough_skew_angle'] = 0.0
            else:
                skew_metrics['hough_skew_angle'] = 0.0
            
            # 2. Projection profile method
            # Horizontal projection
            horizontal_proj = np.sum(gray_image < 128, axis=1)  # Count dark pixels
            
            # Try different rotation angles and find the one with maximum variance
            angles_to_try = np.linspace(-5, 5, 21)  # -5 to 5 degrees
            variances = []
            
            for angle in angles_to_try:
                # Rotate image
                center = (gray_image.shape[1]//2, gray_image.shape[0]//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray_image, rotation_matrix, (gray_image.shape[1], gray_image.shape[0]))
                
                # Calculate horizontal projection variance
                h_proj = np.sum(rotated < 128, axis=1)
                variances.append(np.var(h_proj))
            
            best_angle_idx = np.argmax(variances)
            skew_metrics['projection_skew_angle'] = angles_to_try[best_angle_idx]
            
            # 3. Final skew estimate (average of methods)
            skew_estimate = np.mean([
                skew_metrics['hough_skew_angle'],
                skew_metrics['projection_skew_angle']
            ])
            skew_metrics['estimated_skew'] = skew_estimate
            skew_metrics['is_skewed'] = abs(skew_estimate) > self.config['skew_threshold']
            
            # Skew score (0-1, where 1 is no skew)
            skew_metrics['skew_score'] = max(0.0, 1.0 - abs(skew_estimate) / 10)
            
        except Exception as e:
            logger.warning(f"Skew detection failed: {e}")
            skew_metrics = {
                'hough_skew_angle': 0.0,
                'projection_skew_angle': 0.0,
                'estimated_skew': 0.0,
                'is_skewed': False,
                'skew_score': 1.0
            }
        
        return skew_metrics
    
    def _assess_text_quality(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Assess text-specific quality metrics"""
        text_metrics = {}
        
        # 1. Text line detection quality
        # Apply morphological operations to detect text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        text_lines = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Count detected text lines
        contours, _ = cv2.findContours(text_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_line_contours = [c for c in contours if cv2.contourArea(c) > 100]
        text_metrics['detected_text_lines'] = len(text_line_contours)
        
        # 2. Character size estimation
        # Use morphological operations to detect individual characters
        char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        chars = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, char_kernel)
        char_contours, _ = cv2.findContours(chars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if char_contours:
            char_sizes = [cv2.boundingRect(c)[3] for c in char_contours]  # Heights
            text_metrics['average_char_height'] = np.mean(char_sizes)
            text_metrics['char_height_std'] = np.std(char_sizes)
        else:
            text_metrics['average_char_height'] = 0
            text_metrics['char_height_std'] = 0
        
        # 3. Text density
        binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text_pixels = np.sum(binary == 0)  # Dark pixels (text)
        total_pixels = binary.size
        text_metrics['text_density'] = text_pixels / total_pixels
        
        # 4. Text clarity (edge sharpness around text)
        edges = cv2.Canny(gray_image, 50, 150)
        text_edges = cv2.bitwise_and(edges, cv2.bitwise_not(binary))
        text_metrics['text_edge_density'] = np.sum(text_edges > 0) / np.sum(binary == 0) if np.sum(binary == 0) > 0 else 0
        
        # Overall text quality score
        scores = []
        if text_metrics['detected_text_lines'] > 0:
            scores.append(min(1.0, text_metrics['detected_text_lines'] / 20))
        if text_metrics['average_char_height'] > 0:
            scores.append(min(1.0, text_metrics['average_char_height'] / 20))
        scores.append(min(1.0, text_metrics['text_density'] * 10))
        scores.append(min(1.0, text_metrics['text_edge_density'] * 5))
        
        text_metrics['text_quality_score'] = np.mean(scores) if scores else 0.0
        
        return text_metrics
    
    def _calculate_overall_quality(self, quality_metrics: Dict) -> float:
        """Calculate overall image quality score"""
        scores = []
        weights = []
        
        # Blur score
        blur_score = quality_metrics.get('blur_metrics', {}).get('blur_score', 0.5)
        scores.append(blur_score)
        weights.append(0.25)
        
        # Noise score
        noise_score = quality_metrics.get('noise_metrics', {}).get('noise_score', 0.5)
        scores.append(noise_score)
        weights.append(0.2)
        
        # Contrast score
        contrast_score = quality_metrics.get('contrast_metrics', {}).get('contrast_score', 0.5)
        scores.append(contrast_score)
        weights.append(0.2)
        
        # Resolution score
        resolution_score = quality_metrics.get('resolution_metrics', {}).get('resolution_score', 0.5)
        scores.append(resolution_score)
        weights.append(0.15)
        
        # Skew score
        skew_score = quality_metrics.get('skew_metrics', {}).get('skew_score', 1.0)
        scores.append(skew_score)
        weights.append(0.1)
        
        # Text quality score
        text_score = quality_metrics.get('text_quality_metrics', {}).get('text_quality_score', 0.5)
        scores.append(text_score)
        weights.append(0.1)
        
        # Weighted average
        return np.average(scores, weights=weights)
    
    def _generate_quality_recommendations(self, quality_metrics: Dict) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Blur recommendations
        blur_metrics = quality_metrics.get('blur_metrics', {})
        if blur_metrics.get('is_blurry_laplacian', False):
            recommendations.append("Image appears blurry. Consider rescanning at higher quality.")
        
        # Noise recommendations
        noise_metrics = quality_metrics.get('noise_metrics', {})
        if noise_metrics.get('is_noisy', False):
            recommendations.append("High noise detected. Apply noise reduction preprocessing.")
        
        # Contrast recommendations
        contrast_metrics = quality_metrics.get('contrast_metrics', {})
        if contrast_metrics.get('is_low_contrast', False):
            recommendations.append("Low contrast detected. Consider histogram equalization or gamma correction.")
        
        # Resolution recommendations
        resolution_metrics = quality_metrics.get('resolution_metrics', {})
        if not resolution_metrics.get('adequate_for_ocr', True):
            recommendations.append(f"Resolution too low for optimal OCR. Current: ~{resolution_metrics.get('estimated_dpi', 0):.0f} DPI, Recommended: ≥{self.config['min_resolution_dpi']} DPI")
        
        # Skew recommendations
        skew_metrics = quality_metrics.get('skew_metrics', {})
        if skew_metrics.get('is_skewed', False):
            skew_angle = skew_metrics.get('estimated_skew', 0)
            recommendations.append(f"Document skew detected ({skew_angle:.1f}°). Consider deskewing preprocessing.")
        
        # Text quality recommendations
        text_metrics = quality_metrics.get('text_quality_metrics', {})
        if text_metrics.get('text_quality_score', 1.0) < 0.5:
            recommendations.append("Poor text quality detected. Check for proper binarization and text clarity.")
        
        return recommendations
    
    def _generate_optimization_suggestions(self, quality_metrics: Dict) -> Dict[str, Any]:
        """Generate optimization suggestions for OCR engines"""
        suggestions = {
            'preprocessing_steps': [],
            'ocr_parameters': {},
            'engine_preferences': []
        }
        
        # Preprocessing suggestions
        if quality_metrics.get('noise_metrics', {}).get('is_noisy', False):
            suggestions['preprocessing_steps'].append('noise_reduction')
        
        if quality_metrics.get('contrast_metrics', {}).get('is_low_contrast', False):
            suggestions['preprocessing_steps'].append('contrast_enhancement')
        
        if quality_metrics.get('skew_metrics', {}).get('is_skewed', False):
            suggestions['preprocessing_steps'].append('deskewing')
        
        # OCR parameter suggestions
        if quality_metrics.get('blur_metrics', {}).get('is_blurry_laplacian', False):
            suggestions['ocr_parameters']['enable_image_sharpening'] = True
        
        resolution_category = quality_metrics.get('resolution_metrics', {}).get('quality_category', 'medium')
        if resolution_category == 'low':
            suggestions['ocr_parameters']['use_conservative_settings'] = True
        elif resolution_category == 'high':
            suggestions['ocr_parameters']['use_aggressive_settings'] = True
        
        # Engine preferences based on quality
        overall_quality = quality_metrics.get('overall_quality', 0.5)
        if overall_quality > 0.8:
            suggestions['engine_preferences'] = ['tesseract', 'surya', 'trocr', 'easyocr']
        elif overall_quality > 0.6:
            suggestions['engine_preferences'] = ['trocr', 'surya', 'tesseract', 'easyocr']
        else:
            suggestions['engine_preferences'] = ['trocr', 'easyocr', 'surya', 'tesseract']
        
        return suggestions