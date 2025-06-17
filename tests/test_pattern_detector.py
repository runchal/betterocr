import unittest
import os
import sys

# Add the ocr_engine directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_engine')))

from src.adaptive.pattern_detector import PatternDetector

class TestPatternDetector(unittest.TestCase):

    def test_detect_email(self):
        """Test that the pattern detector can correctly identify an email address."""
        # Arrange
        detector = PatternDetector()
        text = 'You can contact me at test@example.com.'

        # Act
        patterns = detector.detect_patterns(text)

        # Assert
        self.assertIn('emails', patterns)
        self.assertEqual(len(patterns['emails']), 1)
        self.assertEqual(patterns['emails'][0]['value'], 'test@example.com')

if __name__ == '__main__':
    unittest.main()