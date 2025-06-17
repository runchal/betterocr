import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the ocr_engine directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_engine')))

from src import main_ocr

class TestMainOCR(unittest.TestCase):

    @patch('src.processing.document_processor.DocumentProcessor')
    def test_main_processing_logic(self, MockDocumentProcessor):
        """Test the main function's argument parsing and processing flow."""
        # Arrange
        mock_processor_instance = MockDocumentProcessor.return_value
        mock_processor_instance.process_document.return_value = {'status': 'success'}
        
        test_args = ['main_ocr.py', 'samples/dummy.pdf', '--output', 'output/dummy.json']

        # Act
        with patch.object(sys, 'argv', test_args):
            main_ocr.main()

        # Assert
        MockDocumentProcessor.assert_called_once()
        mock_processor_instance.process_document.assert_called_with('samples/dummy.pdf')

if __name__ == '__main__':
    unittest.main()
