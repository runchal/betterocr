"""
OCR Engines Module
Provides multiple OCR engine implementations
"""

from .base_engine import BaseOCREngine, OCRResult
from .tesseract_engine import TesseractEngine
from .easyocr_engine import EasyOCREngine
from .paddleocr_engine import PaddleOCREngine
from .trocr_engine import TrOCREngine
from .surya_engine import SuryaEngine

__all__ = [
    'BaseOCREngine',
    'OCRResult',
    'TesseractEngine',
    'EasyOCREngine', 
    'PaddleOCREngine',
    'TrOCREngine',
    'SuryaEngine'
]

# Engine registry for easy access
ENGINE_REGISTRY = {
    'tesseract': TesseractEngine,
    'easyocr': EasyOCREngine,
    'paddleocr': PaddleOCREngine,
    'trocr': TrOCREngine,
    'surya': SuryaEngine
}

def get_engine(engine_name: str, config: dict = None):
    """
    Factory function to get an OCR engine instance
    
    Args:
        engine_name: Name of the engine ('tesseract', 'easyocr', etc.)
        config: Configuration dictionary for the engine
        
    Returns:
        OCR engine instance
        
    Raises:
        ValueError: If engine name is not recognized
    """
    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine: {engine_name}. Available: {list(ENGINE_REGISTRY.keys())}")
    
    engine_class = ENGINE_REGISTRY[engine_name]
    return engine_class(config)

def list_available_engines():
    """
    List all available OCR engines and their availability status
    
    Returns:
        Dictionary of engine names to availability status
    """
    available = {}
    
    for name, engine_class in ENGINE_REGISTRY.items():
        try:
            engine = engine_class()
            available[name] = engine.is_available
        except Exception:
            available[name] = False
    
    return available