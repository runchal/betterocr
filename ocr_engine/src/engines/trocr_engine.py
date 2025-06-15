"""
TrOCR (Transformer OCR) Engine Implementation
Uses Microsoft's TrOCR models for high-accuracy text recognition
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np

from .base_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class TrOCREngine(BaseOCREngine):
    """TrOCR engine using transformer models"""
    
    AVAILABLE_MODELS = {
        'base_printed': 'microsoft/trocr-base-printed',
        'large_printed': 'microsoft/trocr-large-printed',
        'base_handwritten': 'microsoft/trocr-base-handwritten',
        'large_handwritten': 'microsoft/trocr-large-handwritten',
        'small_printed': 'microsoft/trocr-small-printed',
        'small_handwritten': 'microsoft/trocr-small-handwritten'
    }
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'model_type': 'base_printed',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 1,
            'max_length': 256,
            'beam_search': True,
            'num_beams': 5,
            'confidence_threshold': 0.5
        }
        
        if config:
            default_config.update(config)
            
        super().__init__('trocr', default_config)
        
        self.processor = None
        self.model = None
        self.device = self.config['device']
        
    def _check_availability(self) -> bool:
        """Check if TrOCR dependencies are available"""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            logger.warning("TrOCR dependencies not available")
            return False
    
    def _load_model(self):
        """Load TrOCR model and processor"""
        if self.model is None:
            model_name = self.AVAILABLE_MODELS.get(
                self.config['model_type'], 
                self.AVAILABLE_MODELS['base_printed']
            )
            
            logger.info(f"Loading TrOCR model: {model_name}")
            
            try:
                self.processor = TrOCRProcessor.from_pretrained(model_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"TrOCR model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load TrOCR model: {e}")
                raise
    
    def _extract_text_from_image(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text using TrOCR"""
        result = OCRResult()
        
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # For TrOCR, we need to process the image in patches/regions
            # since it works best on single lines of text
            text_regions = self._extract_text_regions(image)
            
            all_texts = []
            all_confidences = []
            all_bboxes = []
            
            # Process each text region
            for region_image, bbox in text_regions:
                text, confidence = self._process_single_region(region_image, **kwargs)
                if text and confidence > self.config['confidence_threshold']:
                    all_texts.append(text)
                    all_confidences.append(confidence)
                    all_bboxes.append(bbox)
                    
                    # Store as line result
                    result.lines.append((text, confidence, bbox))
            
            # Combine all text
            result.text = '\n'.join(all_texts)
            
            # Calculate overall confidence
            if all_confidences:
                result.confidence = sum(all_confidences) / len(all_confidences)
            
            # Add metadata
            result.metadata = {
                'model': self.config['model_type'],
                'device': self.device,
                'num_regions': len(text_regions),
                'num_successful': len(all_texts)
            }
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            result.errors.append(str(e))
            
        return result
    
    def _extract_text_regions(self, image: Image) -> List[tuple]:
        """Extract text regions from image for processing"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple approach: use the full image as one region
        # In a more sophisticated implementation, this would:
        # 1. Detect text regions using CV
        # 2. Segment into lines
        # 3. Return individual line images
        
        regions = [(image, (0, 0, image.width, image.height))]
        
        # TODO: Implement proper text region detection
        # This could use LayoutParser or custom CV methods
        
        return regions
    
    def _process_single_region(self, region_image: Image, **kwargs) -> tuple:
        """Process a single text region with TrOCR"""
        # Prepare image for model
        pixel_values = self.processor(
            region_image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            if kwargs.get('beam_search', self.config['beam_search']):
                # Use beam search for better quality
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=kwargs.get('max_length', self.config['max_length']),
                    num_beams=kwargs.get('num_beams', self.config['num_beams']),
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Calculate confidence from scores
                confidence = self._calculate_beam_confidence(generated_ids)
            else:
                # Greedy decoding
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=kwargs.get('max_length', self.config['max_length'])
                )
                confidence = 0.8  # Default confidence for greedy
        
        # Decode text
        generated_text = self.processor.batch_decode(
            generated_ids.sequences if hasattr(generated_ids, 'sequences') else generated_ids,
            skip_special_tokens=True
        )[0]
        
        return generated_text, confidence
    
    def _calculate_beam_confidence(self, output) -> float:
        """Calculate confidence score from beam search output"""
        if hasattr(output, 'sequences_scores'):
            # Use the score of the best sequence
            best_score = torch.exp(output.sequences_scores[0]).item()
            return min(best_score, 1.0)
        return 0.8  # Default confidence
    
    def process_batch(self, image_paths: List[Path], **kwargs) -> List[OCRResult]:
        """Process multiple images in batch for efficiency"""
        self._load_model()
        
        results = []
        batch_size = kwargs.get('batch_size', self.config['batch_size'])
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [Image.open(p).convert('RGB') for p in batch_paths]
            
            # Process batch
            pixel_values = self.processor(
                batch_images,
                return_tensors="pt",
                padding=True
            ).pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.config['max_length'],
                    num_beams=self.config['num_beams'] if self.config['beam_search'] else 1
                )
            
            # Decode texts
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Create results
            for path, text in zip(batch_paths, generated_texts):
                result = OCRResult()
                result.text = text
                result.confidence = 0.8  # Default confidence
                result.engine_name = self.name
                results.append(result)
        
        return results
    
    def supports_batch_processing(self) -> bool:
        """TrOCR supports efficient batch processing"""
        return True
    
    def adapt_to_document_type(self, document_type: str):
        """Adapt model selection based on document type"""
        if 'handwritten' in document_type.lower():
            self.config['model_type'] = 'large_handwritten'
            # Clear loaded model to force reload
            self.model = None
            self.processor = None
        elif 'form' in document_type.lower() or 'table' in document_type.lower():
            self.config['model_type'] = 'large_printed'
            self.model = None
            self.processor = None