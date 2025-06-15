"""
Confidence Scorer
Advanced confidence calculation and validation for OCR results
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import re
from collections import Counter
import string

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Advanced confidence scoring for OCR results"""
    
    def __init__(self):
        # Language models for validation (simplified)
        self.common_words = self._load_common_words()
        self.letter_frequencies = self._get_english_letter_frequencies()
        
        # Pattern validators
        self.validators = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'),
            'date': re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'),
            'currency': re.compile(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$'),
            'ssn': re.compile(r'^\d{3}-\d{2}-\d{4}$'),
            'zip': re.compile(r'^\d{5}(?:-\d{4})?$')
        }
    
    def calculate_comprehensive_confidence(self, result: Any, context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate comprehensive confidence scores for OCR result
        
        Args:
            result: OCR result object
            context: Additional context for scoring
            
        Returns:
            Dictionary of confidence scores
        """
        scores = {}
        
        # Base confidence from engine
        scores['engine_confidence'] = getattr(result, 'confidence', 0.0)
        
        # Text quality metrics
        if hasattr(result, 'text') and result.text:
            scores.update(self._score_text_quality(result.text))
        
        # Word-level confidence
        if hasattr(result, 'words') and result.words:
            scores.update(self._score_word_level(result.words))
        
        # Line-level confidence
        if hasattr(result, 'lines') and result.lines:
            scores.update(self._score_line_level(result.lines))
        
        # Pattern validation
        if hasattr(result, 'text'):
            scores.update(self._score_pattern_validation(result.text))
        
        # Structural consistency
        scores.update(self._score_structural_consistency(result))
        
        # Context-based scoring
        if context:
            scores.update(self._score_context_consistency(result, context))
        
        # Calculate overall confidence
        scores['overall_confidence'] = self._calculate_overall_confidence(scores)
        
        return scores
    
    def _score_text_quality(self, text: str) -> Dict[str, float]:
        """Score overall text quality"""
        scores = {}
        
        if not text:
            return {'text_quality': 0.0, 'readability': 0.0, 'language_consistency': 0.0}
        
        # Basic quality metrics
        scores['text_length_score'] = self._score_text_length(text)
        scores['character_distribution'] = self._score_character_distribution(text)
        scores['readability'] = self._score_readability(text)
        scores['language_consistency'] = self._score_language_consistency(text)
        scores['formatting_consistency'] = self._score_formatting(text)
        
        # Overall text quality
        scores['text_quality'] = np.mean([
            scores['character_distribution'],
            scores['readability'],
            scores['language_consistency'],
            scores['formatting_consistency']
        ])
        
        return scores
    
    def _score_text_length(self, text: str) -> float:
        """Score based on text length (longer usually more reliable)"""
        length = len(text.strip())
        
        if length == 0:
            return 0.0
        elif length < 10:
            return 0.3
        elif length < 50:
            return 0.6
        elif length < 200:
            return 0.8
        else:
            return 0.9
    
    def _score_character_distribution(self, text: str) -> float:
        """Score based on character distribution similarity to English"""
        if not text:
            return 0.0
        
        # Count letter frequencies
        text_clean = ''.join(c.lower() for c in text if c.isalpha())
        
        if not text_clean:
            return 0.5  # No letters, neutral score
        
        char_counts = Counter(text_clean)
        total_chars = len(text_clean)
        
        # Calculate frequency distribution
        text_frequencies = {char: count / total_chars for char, count in char_counts.items()}
        
        # Compare to English frequencies using chi-square like metric
        score = 0.0
        for char in string.ascii_lowercase:
            expected_freq = self.letter_frequencies.get(char, 0.001)
            actual_freq = text_frequencies.get(char, 0.0)
            
            # Penalize large deviations
            deviation = abs(expected_freq - actual_freq)
            score += max(0, 1.0 - deviation * 10)  # Scale factor
        
        return min(1.0, score / 26)  # Normalize by alphabet size
    
    def _score_readability(self, text: str) -> float:
        """Score text readability"""
        words = text.split()
        
        if not words:
            return 0.0
        
        # Check for dictionary words
        valid_words = 0
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
            if clean_word in self.common_words:
                valid_words += 1
        
        dictionary_ratio = valid_words / len(words)
        
        # Check for reasonable word lengths
        avg_word_length = np.mean([len(word) for word in words])
        length_score = 1.0 if 2 <= avg_word_length <= 8 else 0.5
        
        # Check for mixed case (indicates proper capitalization)
        has_uppercase = any(c.isupper() for c in text)
        has_lowercase = any(c.islower() for c in text)
        case_score = 1.0 if has_uppercase and has_lowercase else 0.7
        
        return np.mean([dictionary_ratio, length_score, case_score])
    
    def _score_language_consistency(self, text: str) -> float:
        """Score language consistency"""
        # Simple checks for language consistency
        
        # Check for reasonable punctuation
        punct_count = sum(1 for c in text if c in '.,!?;:')
        word_count = len(text.split())
        punct_ratio = punct_count / max(word_count, 1)
        
        # Reasonable punctuation ratio
        punct_score = 1.0 if 0.0 <= punct_ratio <= 0.3 else 0.5
        
        # Check for excessive special characters
        special_count = sum(1 for c in text if not c.isalnum() and c not in ' .,!?;:-"\'()')
        special_ratio = special_count / max(len(text), 1)
        
        special_score = 1.0 if special_ratio < 0.1 else max(0.0, 1.0 - special_ratio * 5)
        
        return np.mean([punct_score, special_score])
    
    def _score_formatting(self, text: str) -> float:
        """Score formatting consistency"""
        lines = text.split('\n')
        
        # Check for consistent line breaks
        line_lengths = [len(line.strip()) for line in lines if line.strip()]
        
        if not line_lengths:
            return 0.5
        
        # Reasonable variation in line lengths
        if len(line_lengths) > 1:
            cv = np.std(line_lengths) / np.mean(line_lengths)  # Coefficient of variation
            length_consistency = max(0.0, 1.0 - cv)  # Lower CV = more consistent
        else:
            length_consistency = 1.0
        
        # Check for excessive whitespace
        whitespace_ratio = text.count(' ') / max(len(text), 1)
        whitespace_score = 1.0 if 0.1 <= whitespace_ratio <= 0.3 else 0.7
        
        return np.mean([length_consistency, whitespace_score])
    
    def _score_word_level(self, words: List[Tuple]) -> Dict[str, float]:
        """Score word-level confidence"""
        if not words:
            return {'word_level_confidence': 0.0, 'word_consistency': 0.0}
        
        confidences = []
        valid_words = 0
        
        for word_data in words:
            if len(word_data) >= 2:
                word_text = word_data[0]
                word_conf = word_data[1]
                
                confidences.append(word_conf)
                
                # Check if word looks valid
                if self._is_valid_word(word_text):
                    valid_words += 1
        
        scores = {}
        
        # Average word confidence
        scores['word_level_confidence'] = np.mean(confidences) if confidences else 0.0
        
        # Word validity ratio
        scores['word_validity_ratio'] = valid_words / len(words)
        
        # Word confidence consistency
        if len(confidences) > 1:
            scores['word_consistency'] = 1.0 - np.std(confidences)
        else:
            scores['word_consistency'] = 1.0
        
        return scores
    
    def _is_valid_word(self, word: str) -> bool:
        """Check if a word looks valid"""
        if not word:
            return False
        
        clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
        
        # Check dictionary
        if clean_word in self.common_words:
            return True
        
        # Check for reasonable patterns
        if len(clean_word) >= 2 and clean_word.isalpha():
            return True
        
        # Check for numbers (also valid)
        if word.replace(',', '').replace('.', '').replace('$', '').isdigit():
            return True
        
        return False
    
    def _score_line_level(self, lines: List[Tuple]) -> Dict[str, float]:
        """Score line-level confidence"""
        if not lines:
            return {'line_level_confidence': 0.0, 'line_consistency': 0.0}
        
        confidences = []
        for line_data in lines:
            if len(line_data) >= 2:
                confidences.append(line_data[1])
        
        scores = {}
        scores['line_level_confidence'] = np.mean(confidences) if confidences else 0.0
        
        if len(confidences) > 1:
            scores['line_consistency'] = 1.0 - np.std(confidences)
        else:
            scores['line_consistency'] = 1.0
        
        return scores
    
    def _score_pattern_validation(self, text: str) -> Dict[str, float]:
        """Score based on pattern validation"""
        scores = {}
        
        # Extract potential patterns
        patterns_found = {}
        
        for pattern_name, pattern_regex in self.validators.items():
            matches = pattern_regex.findall(text)
            if matches:
                patterns_found[pattern_name] = len(matches)
        
        # Score based on validated patterns
        if patterns_found:
            scores['pattern_validation_score'] = 0.9  # High confidence for validated patterns
            scores['validated_patterns'] = patterns_found
        else:
            # Check for malformed patterns that might indicate OCR errors
            malformed_score = self._check_malformed_patterns(text)
            scores['pattern_validation_score'] = malformed_score
            scores['validated_patterns'] = {}
        
        return scores
    
    def _check_malformed_patterns(self, text: str) -> float:
        """Check for malformed patterns that suggest OCR errors"""
        # Look for patterns that are almost valid but have OCR-like errors
        
        # Almost-emails (missing @ or .)
        almost_emails = len(re.findall(r'[a-zA-Z0-9._%+-]+[@.][a-zA-Z0-9.-]*', text))
        
        # Almost-phones (wrong separators)
        almost_phones = len(re.findall(r'\d{3}[^\d\s]\d{3}[^\d\s]\d{4}', text))
        
        # Almost-dates (wrong separators)
        almost_dates = len(re.findall(r'\d{1,2}[^\d\s]\d{1,2}[^\d\s]\d{2,4}', text))
        
        malformed_count = almost_emails + almost_phones + almost_dates
        
        if malformed_count > 0:
            return 0.4  # Reduced confidence due to malformed patterns
        
        return 0.7  # Neutral confidence
    
    def _score_structural_consistency(self, result: Any) -> Dict[str, float]:
        """Score structural consistency of the result"""
        scores = {}
        
        # Check consistency between different levels (words vs lines vs text)
        if hasattr(result, 'words') and hasattr(result, 'lines') and hasattr(result, 'text'):
            # Count words in different representations
            word_count_from_words = len(result.words) if result.words else 0
            word_count_from_text = len(result.text.split()) if result.text else 0
            
            if word_count_from_text > 0:
                word_consistency = 1.0 - abs(word_count_from_words - word_count_from_text) / word_count_from_text
                scores['word_count_consistency'] = max(0.0, word_consistency)
            else:
                scores['word_count_consistency'] = 0.0
        
        # Check bounding box consistency
        if hasattr(result, 'words') and result.words:
            bbox_scores = []
            for word_data in result.words:
                if len(word_data) >= 3:
                    bbox = word_data[2]
                    if len(bbox) >= 4 and all(isinstance(x, (int, float)) for x in bbox):
                        # Valid bbox
                        bbox_scores.append(1.0)
                    else:
                        bbox_scores.append(0.0)
            
            scores['bbox_consistency'] = np.mean(bbox_scores) if bbox_scores else 0.0
        
        return scores
    
    def _score_context_consistency(self, result: Any, context: Dict) -> Dict[str, float]:
        """Score consistency with provided context"""
        scores = {}
        
        # Document type consistency
        if 'expected_type' in context:
            expected_type = context['expected_type']
            scores['document_type_consistency'] = self._check_document_type_consistency(result, expected_type)
        
        # Language consistency
        if 'expected_language' in context:
            expected_lang = context['expected_language']
            scores['language_consistency'] = self._check_language_consistency(result, expected_lang)
        
        # Field expectations
        if 'expected_fields' in context:
            expected_fields = context['expected_fields']
            scores['field_consistency'] = self._check_field_consistency(result, expected_fields)
        
        return scores
    
    def _check_document_type_consistency(self, result: Any, expected_type: str) -> float:
        """Check if result is consistent with expected document type"""
        if not hasattr(result, 'text') or not result.text:
            return 0.5
        
        text_lower = result.text.lower()
        
        # Define keywords for different document types
        type_keywords = {
            'invoice': ['invoice', 'bill', 'amount', 'total', 'due'],
            'form': ['form', 'name', 'date', 'signature', 'address'],
            'receipt': ['receipt', 'paid', 'change', 'transaction'],
            'letter': ['dear', 'sincerely', 'regards', 'letter'],
            'report': ['report', 'analysis', 'conclusion', 'findings']
        }
        
        if expected_type.lower() in type_keywords:
            keywords = type_keywords[expected_type.lower()]
            found_keywords = sum(1 for keyword in keywords if keyword in text_lower)
            return found_keywords / len(keywords)
        
        return 0.7  # Neutral score for unknown types
    
    def _check_language_consistency(self, result: Any, expected_language: str) -> float:
        """Check language consistency"""
        # Simplified language checking
        if expected_language.lower() == 'english':
            if hasattr(result, 'text') and result.text:
                english_score = self._score_readability(result.text)
                return english_score
        
        return 0.8  # Default for other languages
    
    def _check_field_consistency(self, result: Any, expected_fields: List[str]) -> float:
        """Check if expected fields are present"""
        if not hasattr(result, 'text') or not result.text:
            return 0.0
        
        text_lower = result.text.lower()
        found_fields = 0
        
        for field in expected_fields:
            if field.lower() in text_lower:
                found_fields += 1
        
        return found_fields / len(expected_fields) if expected_fields else 1.0
    
    def _calculate_overall_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall confidence"""
        weights = {
            'engine_confidence': 0.3,
            'text_quality': 0.25,
            'word_level_confidence': 0.2,
            'pattern_validation_score': 0.15,
            'structural_consistency': 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score_name, weight in weights.items():
            if score_name in scores:
                weighted_sum += scores[score_name] * weight
                total_weight += weight
        
        # Add any context-based scores with lower weight
        context_scores = [k for k in scores.keys() if 'consistency' in k and k not in weights]
        if context_scores:
            context_weight = 0.1 / len(context_scores)
            for score_name in context_scores:
                weighted_sum += scores[score_name] * context_weight
                total_weight += context_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _load_common_words(self) -> set:
        """Load common English words for validation"""
        # This is a simplified set - in practice you'd load from a proper dictionary
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'out', 'off', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'should', 'now', 'name', 'date', 'time', 'year',
            'form', 'total', 'amount', 'address', 'phone', 'email', 'number'
        }
        return common_words
    
    def _get_english_letter_frequencies(self) -> Dict[str, float]:
        """Get English letter frequency distribution"""
        # Based on typical English text frequency analysis
        return {
            'e': 0.1202, 't': 0.0910, 'a': 0.0812, 'o': 0.0768, 'i': 0.0731,
            'n': 0.0695, 's': 0.0628, 'h': 0.0592, 'r': 0.0592, 'd': 0.0432,
            'l': 0.0398, 'c': 0.0271, 'u': 0.0288, 'm': 0.0261, 'w': 0.0209,
            'f': 0.0230, 'g': 0.0203, 'y': 0.0211, 'p': 0.0182, 'b': 0.0149,
            'v': 0.0111, 'k': 0.0069, 'j': 0.0010, 'x': 0.0017, 'q': 0.0011,
            'z': 0.0007
        }
    
    def validate_against_patterns(self, text: str, pattern_type: str) -> Tuple[bool, float]:
        """
        Validate text against a specific pattern type
        
        Returns:
            (is_valid, confidence)
        """
        if pattern_type not in self.validators:
            return False, 0.0
        
        pattern = self.validators[pattern_type]
        
        if pattern.match(text.strip()):
            return True, 0.95
        
        # Check for partial matches or OCR errors
        # This could be enhanced with fuzzy matching
        
        return False, 0.1