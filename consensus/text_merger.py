"""
Text merging and consensus building from multiple OCR engines
"""

import logging
from difflib import SequenceMatcher
from collections import Counter

logger = logging.getLogger(__name__)


class TextMerger:
    """Merge and build consensus from multiple OCR results"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
    
    def merge_results(self, engine_results):
        """
        Merge results from multiple engines
        
        Args:
            engine_results: Dict of {engine_name: result_dict}
            
        Returns:
            Merged result with consensus and variations
        """
        # Extract successful results
        valid_results = {}
        for engine_name, result in engine_results.items():
            if result.get('status') == 'success' and result.get('text'):
                valid_results[engine_name] = result
        
        if not valid_results:
            return {
                'status': 'no_valid_results',
                'consensus_text': '',
                'variations': {},
                'confidence': 0.0
            }
        
        # Extract texts and confidences
        texts = {name: r['text'] for name, r in valid_results.items()}
        confidences = {name: r.get('confidence', 0.5) for name, r in valid_results.items()}
        
        # Find consensus
        consensus = self._find_consensus(texts, confidences)
        
        # Calculate variations
        variations = self._calculate_variations(texts, consensus['consensus_text'])
        
        # Merge all information
        merged = {
            'status': 'success',
            'consensus_text': consensus['consensus_text'],
            'consensus_method': consensus['method'],
            'overall_confidence': consensus['confidence'],
            'engine_results': valid_results,
            'variations': variations,
            'agreement_score': self._calculate_agreement(texts)
        }
        
        return merged
    
    def _find_consensus(self, texts, confidences):
        """Find consensus text from multiple sources"""
        if len(texts) == 1:
            # Only one engine succeeded
            engine_name = list(texts.keys())[0]
            return {
                'consensus_text': texts[engine_name],
                'method': 'single_engine',
                'confidence': confidences[engine_name]
            }
        
        # Method 1: If texts are very similar, use highest confidence
        similarity_matrix = self._calculate_similarity_matrix(texts)
        if self._all_similar(similarity_matrix):
            # All texts are similar, pick the one with highest confidence
            best_engine = max(confidences.items(), key=lambda x: x[1])[0]
            return {
                'consensus_text': texts[best_engine],
                'method': 'high_similarity_highest_confidence',
                'confidence': max(confidences.values())
            }
        
        # Method 2: Word-level voting
        consensus_text = self._word_level_consensus(texts, confidences)
        
        return {
            'consensus_text': consensus_text,
            'method': 'word_voting',
            'confidence': sum(confidences.values()) / len(confidences)
        }
    
    def _calculate_similarity_matrix(self, texts):
        """Calculate pairwise similarity between texts"""
        engines = list(texts.keys())
        matrix = {}
        
        for i, engine1 in enumerate(engines):
            for j, engine2 in enumerate(engines):
                if i < j:
                    similarity = SequenceMatcher(None, texts[engine1], texts[engine2]).ratio()
                    matrix[(engine1, engine2)] = similarity
        
        return matrix
    
    def _all_similar(self, similarity_matrix):
        """Check if all texts are similar enough"""
        return all(sim >= self.similarity_threshold for sim in similarity_matrix.values())
    
    def _word_level_consensus(self, texts, confidences):
        """Build consensus at word level using voting"""
        # Tokenize all texts
        tokenized = {}
        for engine, text in texts.items():
            tokenized[engine] = text.split()
        
        # Find the longest text as reference
        reference_engine = max(tokenized.items(), key=lambda x: len(x[1]))[0]
        reference_words = tokenized[reference_engine]
        
        # Build consensus word by word
        consensus_words = []
        
        for i, ref_word in enumerate(reference_words):
            # Collect votes for this position
            word_votes = Counter()
            
            for engine, words in tokenized.items():
                if i < len(words):
                    # Weight vote by confidence
                    word_votes[words[i]] += confidences.get(engine, 0.5)
            
            # Pick the word with highest weighted votes
            if word_votes:
                consensus_word = word_votes.most_common(1)[0][0]
                consensus_words.append(consensus_word)
        
        return ' '.join(consensus_words)
    
    def _calculate_variations(self, texts, consensus_text):
        """Calculate how each engine's output varies from consensus"""
        variations = {}
        
        for engine, text in texts.items():
            similarity = SequenceMatcher(None, text, consensus_text).ratio()
            variations[engine] = {
                'similarity_to_consensus': similarity,
                'length_difference': len(text) - len(consensus_text),
                'text': text
            }
        
        return variations
    
    def _calculate_agreement(self, texts):
        """Calculate overall agreement score between engines"""
        if len(texts) <= 1:
            return 1.0
        
        similarities = []
        engines = list(texts.keys())
        
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                sim = SequenceMatcher(None, texts[engines[i]], texts[engines[j]]).ratio()
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0