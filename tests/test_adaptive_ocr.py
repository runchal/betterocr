import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the ocr_engine directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_engine')))

from src.main_ocr import AdaptiveMultiEngineOCR

class TestAdaptiveOCR(unittest.TestCase):

    @patch('src.main_ocr.list_available_engines')
    @patch('src.main_ocr.get_engine')
    @patch('src.main_ocr.PatternDetector')
    @patch('src.main_ocr.DocumentClassifier')
    @patch('src.main_ocr.ConsensusMerger')
    @patch('src.main_ocr.ConfidenceScorer')
    @patch('src.main_ocr.VisualValidator')
    @patch('src.main_ocr.LearningDatabase')
    def test_initialization(self, MockLearningDB, MockVisualValidator, MockConfidenceScorer, 
                            MockConsensusMerger, MockDocumentClassifier, MockPatternDetector, 
                            MockGetEngine, MockListEngines):
        """Test that the AdaptiveMultiEngineOCR class initializes correctly."""
        # Arrange
        MockListEngines.return_value = {'tesseract': True}
        MockGetEngine.return_value = MagicMock()

        # Act
        ocr = AdaptiveMultiEngineOCR(learning_enabled=True)

        # Assert
        self.assertIsNotNone(ocr)
        self.assertTrue(ocr.learning_enabled)
        self.assertIn('tesseract', ocr.engines)
        MockLearningDB.assert_called_once()

if __name__ == '__main__':
    unittest.main()