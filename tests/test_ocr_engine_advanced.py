import unittest
import os
import sys

# Add the ocr_engine directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_engine')))

class TestOCREngineAdvanced(unittest.TestCase):

    def test_imports(self):
        """Test that the main components of the advanced engine can be imported."""
        try:
            from src import main_ocr as main
            from src.engines import tesseract_engine
            from src.fusion import consensus_merger
            from src.vision import visual_validator
        except ImportError as e:
            self.fail(f"Failed to import advanced components: {e}")

if __name__ == '__main__':
    unittest.main()
