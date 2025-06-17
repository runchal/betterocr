import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the ocr_engine directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_engine')))

from src.fusion.consensus_merger import ConsensusMerger
from src.engines.base_engine import OCRResult

class TestConsensusMerger(unittest.TestCase):

    def test_merge_with_mock_ocr_results(self):
        """Test that the merger can combine realistic OCRResult objects."""
        # Arrange
        merger = ConsensusMerger()
        
        # Create mock OCRResult objects
        result1 = OCRResult()
        result1.text = 'hello world'
        result1.confidence = 0.9
        result1.words = [('hello', 0.9, (10, 10, 50, 20)), ('world', 0.9, (70, 10, 120, 20))]

        result2 = OCRResult()
        result2.text = 'hello world'
        result2.confidence = 0.95
        result2.words = [('hello', 0.95, (10, 10, 50, 20)), ('world', 0.95, (70, 10, 120, 20))]

        raw_results = {
            'tesseract': result1,
            'easyocr': result2
        }

        # Act
        result = merger.merge_results(raw_results)

        # Assert
        self.assertEqual(result['consensus_text'], 'hello world')
        self.assertGreater(result['overall_confidence'], 0.9)
        self.assertEqual(len(result['words']), 2)

if __name__ == '__main__':
    unittest.main()