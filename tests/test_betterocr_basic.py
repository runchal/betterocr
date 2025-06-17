import unittest
import os
import json
from unittest.mock import patch, MagicMock

# Add project root to the Python path to allow importing betterocr
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from betterocr import BetterOCR

class TestBetterOCRBasic(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment for testing"""
        self.sample_file = 'samples/sample_document.pdf'
        # Ensure the sample file exists
        if not os.path.exists(self.sample_file):
            # Create a dummy file for testing purposes
            os.makedirs(os.path.dirname(self.sample_file), exist_ok=True)
            with open(self.sample_file, 'w') as f:
                f.write('dummy content')

    @patch('betterocr.BetterOCR._init_engines')
    def test_initialization(self, mock_init_engines):
        """Test that the BetterOCR class initializes correctly"""
        mock_init_engines.return_value = {'mock_engine': MagicMock()}
        ocr = BetterOCR(debug=True)
        self.assertTrue(ocr.debug)
        self.assertIn('mock_engine', ocr.engines)

    @patch('betterocr.BetterOCR._init_engines')
    def test_document_processing_flow(self, mock_init_engines):
        """Test the main document processing workflow with mock engines"""
        # Mock the engine behavior
        mock_engine = MagicMock()
        mock_engine.process.return_value = {
            'text': 'mocked text',
            'confidence': 0.95,
            'status': 'success'
        }
        mock_init_engines.return_value = {'tesseract': mock_engine}

        # Initialize BetterOCR and process a document
        ocr = BetterOCR()
        results = ocr.process_document(self.sample_file)

        # Verify the results
        self.assertEqual(results['file'], self.sample_file)
        self.assertIn('tesseract', results['engines'])
        self.assertEqual(results['engines']['tesseract']['text'], 'mocked text')
        self.assertIsNotNone(results['final_output'])

if __name__ == '__main__':
    unittest.main()
