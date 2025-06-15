"""
Consensus Merger
Merges results from multiple OCR engines using sophisticated voting and confidence weighting
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher
from collections import Counter, defaultdict
import re

logger = logging.getLogger(__name__)


class ConsensusMerger:
    """Merges OCR results from multiple engines with adaptive weighting"""
    
    def __init__(self):
        self.engine_weights = {
            'tesseract': 1.0,
            'easyocr': 1.0,
            'paddleocr': 1.0,
            'trocr': 1.2,     # Transformer models often more accurate
            'surya': 1.1      # Modern architecture gets slight boost
        }
        
        # Engine performance history for adaptive weighting
        self.engine_performance = defaultdict(list)
        
        # Similarity thresholds
        self.text_similarity_threshold = 0.8
        self.word_similarity_threshold = 0.7
        self.bbox_overlap_threshold = 0.5
        
    def merge_results(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge results from multiple OCR engines
        
        Args:
            engine_results: Dict mapping engine names to OCRResult objects
            
        Returns:
            Merged consensus result
        """
        # Filter successful results
        valid_results = self._filter_valid_results(engine_results)
        
        if not valid_results:
            return self._create_empty_result()
        
        if len(valid_results) == 1:
            return self._single_engine_result(valid_results)
        
        # Merge different aspects
        consensus = {
            'text_consensus': self._merge_text(valid_results),
            'word_consensus': self._merge_words(valid_results),
            'line_consensus': self._merge_lines(valid_results),
            'bbox_consensus': self._merge_bboxes(valid_results),
            'confidence_analysis': self._analyze_confidence(valid_results),
            'engine_agreement': self._calculate_agreement(valid_results),
            'quality_metrics': self._calculate_quality_metrics(valid_results),
            'metadata': self._merge_metadata(valid_results)
        }
        
        # Generate final merged result
        final_result = self._create_final_result(consensus, valid_results)
        
        return final_result
    
    def _filter_valid_results(self, engine_results: Dict) -> Dict:
        """Filter out failed or invalid results"""
        valid = {}
        
        for engine_name, result in engine_results.items():
            if (hasattr(result, 'errors') and not result.errors and 
                hasattr(result, 'text') and result.text and 
                hasattr(result, 'confidence') and result.confidence > 0.1):
                valid[engine_name] = result
        
        return valid
    
    def _create_empty_result(self) -> Dict:
        """Create empty result when no engines succeeded"""
        return {
            'status': 'no_valid_results',
            'consensus_text': '',
            'confidence': 0.0,
            'method': 'none',
            'engine_results': {},
            'variations': {},
            'agreement_score': 0.0
        }
    
    def _single_engine_result(self, valid_results: Dict) -> Dict:
        """Handle case with only one successful engine"""
        engine_name, result = list(valid_results.items())[0]
        
        return {
            'status': 'success',
            'consensus_text': result.text,
            'confidence': result.confidence,
            'method': 'single_engine',
            'primary_engine': engine_name,
            'engine_results': valid_results,
            'variations': {engine_name: result.text},
            'agreement_score': 1.0,
            'words': getattr(result, 'words', []),
            'lines': getattr(result, 'lines', []),
            'metadata': getattr(result, 'metadata', {})
        }
    
    def _merge_text(self, valid_results: Dict) -> Dict:
        """Merge text content using various strategies"""
        texts = {name: result.text for name, result in valid_results.items()}
        confidences = {name: result.confidence for name, result in valid_results.items()}
        
        # Strategy 1: Check for high similarity
        similarity_matrix = self._calculate_text_similarity_matrix(texts)
        
        if self._texts_are_similar(similarity_matrix):
            # Use highest confidence if all similar
            best_engine = max(confidences.items(), key=lambda x: x[1] * self.engine_weights.get(x[0], 1.0))[0]
            
            return {
                'text': texts[best_engine],
                'method': 'high_similarity_best_confidence',
                'confidence': confidences[best_engine],
                'similarity_score': np.mean(list(similarity_matrix.values())),
                'primary_engine': best_engine
            }
        
        # Strategy 2: Word-level voting with confidence weighting
        word_consensus = self._word_level_voting(texts, confidences)
        
        # Strategy 3: Character-level consensus for high-confidence regions
        char_consensus = self._character_level_consensus(texts, confidences)
        
        # Strategy 4: Hybrid approach
        hybrid_text = self._hybrid_text_merge(word_consensus, char_consensus, texts, confidences)
        
        return {
            'text': hybrid_text['text'],
            'method': 'hybrid_consensus',
            'confidence': hybrid_text['confidence'],
            'word_consensus': word_consensus,
            'char_consensus': char_consensus,
            'alternatives': texts
        }
    
    def _calculate_text_similarity_matrix(self, texts: Dict) -> Dict:
        """Calculate pairwise text similarity"""
        engines = list(texts.keys())
        similarities = {}
        
        for i, engine1 in enumerate(engines):
            for j, engine2 in enumerate(engines):
                if i < j:
                    sim = SequenceMatcher(None, texts[engine1], texts[engine2]).ratio()
                    similarities[(engine1, engine2)] = sim
        
        return similarities
    
    def _texts_are_similar(self, similarity_matrix: Dict) -> bool:
        """Check if all texts are similar enough"""
        if not similarity_matrix:
            return True
        
        return all(sim >= self.text_similarity_threshold for sim in similarity_matrix.values())
    
    def _word_level_voting(self, texts: Dict, confidences: Dict) -> Dict:
        """Perform word-level voting across engines"""
        # Tokenize all texts
        tokenized = {}
        for engine, text in texts.items():
            tokenized[engine] = text.split()
        
        # Find maximum length
        max_length = max(len(words) for words in tokenized.values()) if tokenized else 0
        
        consensus_words = []
        word_confidences = []
        
        for pos in range(max_length):
            # Collect votes for this position
            votes = Counter()
            total_weight = 0
            
            for engine, words in tokenized.items():
                if pos < len(words):
                    word = words[pos]
                    weight = confidences[engine] * self.engine_weights.get(engine, 1.0)
                    votes[word] += weight
                    total_weight += weight
            
            if votes:
                # Get most voted word
                best_word = votes.most_common(1)[0][0]
                word_confidence = votes[best_word] / total_weight if total_weight > 0 else 0
                
                consensus_words.append(best_word)
                word_confidences.append(word_confidence)
        
        return {
            'text': ' '.join(consensus_words),
            'confidence': np.mean(word_confidences) if word_confidences else 0,
            'word_confidences': word_confidences
        }
    
    def _character_level_consensus(self, texts: Dict, confidences: Dict) -> Dict:
        """Perform character-level consensus for precise alignment"""
        # This is computationally intensive, so we'll use a simplified approach
        # In practice, you might use sequence alignment algorithms
        
        # Find the text with highest confidence as base
        base_engine = max(confidences.items(), key=lambda x: x[1])[0]
        base_text = texts[base_engine]
        
        # For each character, check agreement across engines
        consensus_chars = []
        char_confidences = []
        
        for i, base_char in enumerate(base_text):
            char_votes = Counter()
            total_weight = 0
            
            for engine, text in texts.items():
                if i < len(text):
                    char = text[i]
                    weight = confidences[engine] * self.engine_weights.get(engine, 1.0)
                    char_votes[char] += weight
                    total_weight += weight
            
            if char_votes:
                best_char = char_votes.most_common(1)[0][0]
                char_confidence = char_votes[best_char] / total_weight if total_weight > 0 else 0
                
                consensus_chars.append(best_char)
                char_confidences.append(char_confidence)
            else:
                consensus_chars.append(base_char)
                char_confidences.append(confidences[base_engine])
        
        return {
            'text': ''.join(consensus_chars),
            'confidence': np.mean(char_confidences) if char_confidences else 0,
            'char_confidences': char_confidences
        }
    
    def _hybrid_text_merge(self, word_consensus: Dict, char_consensus: Dict, 
                          texts: Dict, confidences: Dict) -> Dict:
        """Combine word and character level consensus intelligently"""
        
        # Use word consensus as base
        hybrid_text = word_consensus['text']
        hybrid_confidence = word_consensus['confidence']
        
        # For regions with low word consensus confidence, consider character consensus
        word_confs = word_consensus.get('word_confidences', [])
        words = hybrid_text.split()
        
        if len(words) == len(word_confs):
            # Find low confidence regions
            low_conf_indices = [i for i, conf in enumerate(word_confs) if conf < 0.6]
            
            if low_conf_indices and char_consensus['confidence'] > word_consensus['confidence']:
                # In practice, you'd do more sophisticated region replacement
                # For now, just boost confidence if character consensus is better
                hybrid_confidence = max(hybrid_confidence, char_consensus['confidence'] * 0.8)
        
        return {
            'text': hybrid_text,
            'confidence': hybrid_confidence,
            'method': 'word_char_hybrid'
        }
    
    def _merge_words(self, valid_results: Dict) -> List[Tuple]:
        """Merge word-level results across engines"""
        all_words = {}
        
        # Collect all words from all engines
        for engine_name, result in valid_results.items():
            if hasattr(result, 'words') and result.words:
                for i, word_data in enumerate(result.words):
                    if len(word_data) >= 3:
                        text, conf, bbox = word_data[:3]
                        
                        # Create unique key based on position
                        key = self._create_position_key(bbox)
                        
                        if key not in all_words:
                            all_words[key] = []
                        
                        all_words[key].append({
                            'engine': engine_name,
                            'text': text,
                            'confidence': conf,
                            'bbox': bbox,
                            'weight': self.engine_weights.get(engine_name, 1.0)
                        })
        
        # Merge overlapping words
        merged_words = []
        
        for position_key, word_candidates in all_words.items():
            if len(word_candidates) == 1:
                # Single word, use as is
                word = word_candidates[0]
                merged_words.append((word['text'], word['confidence'], word['bbox']))
            else:
                # Multiple candidates, merge
                merged_word = self._merge_word_candidates(word_candidates)
                merged_words.append(merged_word)
        
        # Sort by position
        merged_words.sort(key=lambda x: (x[2][1], x[2][0]) if len(x) > 2 else (0, 0))
        
        return merged_words
    
    def _create_position_key(self, bbox: Tuple) -> str:
        """Create a position-based key for grouping words"""
        if len(bbox) >= 4:
            # Group words within 10 pixel regions
            x_bucket = bbox[0] // 10
            y_bucket = bbox[1] // 10
            return f"{x_bucket}_{y_bucket}"
        return "0_0"
    
    def _merge_word_candidates(self, candidates: List[Dict]) -> Tuple:
        """Merge multiple word candidates at similar positions"""
        # Count text occurrences weighted by confidence and engine weight
        text_votes = Counter()
        total_weight = 0
        all_bboxes = []
        
        for candidate in candidates:
            weight = candidate['confidence'] * candidate['weight']
            text_votes[candidate['text']] += weight
            total_weight += weight
            all_bboxes.append(candidate['bbox'])
        
        # Get most voted text
        best_text = text_votes.most_common(1)[0][0] if text_votes else ""
        
        # Calculate merged confidence
        merged_confidence = text_votes[best_text] / total_weight if total_weight > 0 else 0
        
        # Merge bounding boxes
        merged_bbox = self._merge_bounding_boxes(all_bboxes)
        
        return (best_text, merged_confidence, merged_bbox)
    
    def _merge_bounding_boxes(self, bboxes: List[Tuple]) -> Tuple:
        """Merge multiple bounding boxes"""
        if not bboxes:
            return (0, 0, 0, 0)
        
        # Extract coordinates
        x_coords = []
        y_coords = []
        x2_coords = []
        y2_coords = []
        
        for bbox in bboxes:
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                x_coords.append(x)
                y_coords.append(y)
                x2_coords.append(x + w)
                y2_coords.append(y + h)
        
        if not x_coords:
            return (0, 0, 0, 0)
        
        # Calculate merged bbox
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x2_coords)
        y_max = max(y2_coords)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _merge_lines(self, valid_results: Dict) -> List[Tuple]:
        """Merge line-level results"""
        # Similar to word merging but at line level
        all_lines = []
        
        for engine_name, result in valid_results.items():
            if hasattr(result, 'lines') and result.lines:
                for line_data in result.lines:
                    if len(line_data) >= 3:
                        text, conf, bbox = line_data[:3]
                        all_lines.append({
                            'engine': engine_name,
                            'text': text,
                            'confidence': conf,
                            'bbox': bbox,
                            'weight': self.engine_weights.get(engine_name, 1.0)
                        })
        
        # Group lines by vertical position
        line_groups = self._group_lines_by_position(all_lines)
        
        # Merge each group
        merged_lines = []
        for group in line_groups:
            if len(group) == 1:
                line = group[0]
                merged_lines.append((line['text'], line['confidence'], line['bbox']))
            else:
                merged_line = self._merge_line_group(group)
                merged_lines.append(merged_line)
        
        return merged_lines
    
    def _group_lines_by_position(self, lines: List[Dict]) -> List[List[Dict]]:
        """Group lines by vertical position"""
        if not lines:
            return []
        
        # Sort by Y position
        sorted_lines = sorted(lines, key=lambda x: x['bbox'][1])
        
        groups = []
        current_group = [sorted_lines[0]]
        y_threshold = 15  # pixels
        
        for line in sorted_lines[1:]:
            last_y = current_group[-1]['bbox'][1]
            curr_y = line['bbox'][1]
            
            if abs(curr_y - last_y) <= y_threshold:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]
        
        groups.append(current_group)
        return groups
    
    def _merge_line_group(self, group: List[Dict]) -> Tuple:
        """Merge a group of lines at similar positions"""
        # Use same voting logic as words
        text_votes = Counter()
        total_weight = 0
        all_bboxes = []
        
        for line in group:
            weight = line['confidence'] * line['weight']
            text_votes[line['text']] += weight
            total_weight += weight
            all_bboxes.append(line['bbox'])
        
        best_text = text_votes.most_common(1)[0][0] if text_votes else ""
        merged_confidence = text_votes[best_text] / total_weight if total_weight > 0 else 0
        merged_bbox = self._merge_bounding_boxes(all_bboxes)
        
        return (best_text, merged_confidence, merged_bbox)
    
    def _merge_bboxes(self, valid_results: Dict) -> Dict:
        """Merge bounding box information"""
        bbox_data = {
            'word_boxes': [],
            'line_boxes': [],
            'confidence_map': {},
            'engine_coverage': {}
        }
        
        for engine_name, result in valid_results.items():
            # Collect word boxes
            if hasattr(result, 'words'):
                for word_data in result.words:
                    if len(word_data) >= 3:
                        bbox_data['word_boxes'].append(word_data[2])
            
            # Collect line boxes  
            if hasattr(result, 'lines'):
                for line_data in result.lines:
                    if len(line_data) >= 3:
                        bbox_data['line_boxes'].append(line_data[2])
        
        return bbox_data
    
    def _analyze_confidence(self, valid_results: Dict) -> Dict:
        """Analyze confidence scores across engines"""
        analysis = {
            'individual_confidences': {},
            'weighted_average': 0.0,
            'confidence_variance': 0.0,
            'engine_reliability': {}
        }
        
        confidences = []
        weights = []
        
        for engine_name, result in valid_results.items():
            conf = result.confidence
            weight = self.engine_weights.get(engine_name, 1.0)
            
            analysis['individual_confidences'][engine_name] = conf
            confidences.append(conf)
            weights.append(weight)
        
        if confidences:
            # Weighted average
            analysis['weighted_average'] = np.average(confidences, weights=weights)
            
            # Variance
            analysis['confidence_variance'] = np.var(confidences)
            
            # Engine reliability (how close each engine is to the average)
            avg_conf = analysis['weighted_average']
            for engine_name, conf in analysis['individual_confidences'].items():
                reliability = 1.0 - abs(conf - avg_conf)
                analysis['engine_reliability'][engine_name] = max(0, reliability)
        
        return analysis
    
    def _calculate_agreement(self, valid_results: Dict) -> Dict:
        """Calculate agreement metrics between engines"""
        if len(valid_results) < 2:
            return {'overall_agreement': 1.0, 'pairwise_agreements': {}}
        
        engines = list(valid_results.keys())
        agreements = {}
        
        # Calculate pairwise text agreement
        for i, engine1 in enumerate(engines):
            for j, engine2 in enumerate(engines):
                if i < j:
                    text1 = valid_results[engine1].text
                    text2 = valid_results[engine2].text
                    
                    agreement = SequenceMatcher(None, text1, text2).ratio()
                    agreements[(engine1, engine2)] = agreement
        
        overall_agreement = np.mean(list(agreements.values())) if agreements else 1.0
        
        return {
            'overall_agreement': overall_agreement,
            'pairwise_agreements': agreements,
            'agreement_level': self._categorize_agreement(overall_agreement)
        }
    
    def _categorize_agreement(self, agreement_score: float) -> str:
        """Categorize agreement level"""
        if agreement_score >= 0.9:
            return 'very_high'
        elif agreement_score >= 0.7:
            return 'high'
        elif agreement_score >= 0.5:
            return 'moderate'
        elif agreement_score >= 0.3:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_quality_metrics(self, valid_results: Dict) -> Dict:
        """Calculate quality metrics for the merged result"""
        metrics = {
            'engine_count': len(valid_results),
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0,
            'engine_success_rate': 1.0,  # All in valid_results succeeded
            'text_length_consistency': 0.0
        }
        
        # Processing time stats
        times = []
        for result in valid_results.values():
            if hasattr(result, 'processing_time'):
                times.append(result.processing_time)
        
        if times:
            metrics['avg_processing_time'] = np.mean(times)
            metrics['total_processing_time'] = sum(times)
        
        # Text length consistency
        lengths = [len(result.text) for result in valid_results.values()]
        if lengths:
            metrics['text_length_consistency'] = 1.0 - (np.std(lengths) / np.mean(lengths))
        
        return metrics
    
    def _merge_metadata(self, valid_results: Dict) -> Dict:
        """Merge metadata from all engines"""
        merged_metadata = {
            'engines_used': list(valid_results.keys()),
            'primary_engine': None,
            'engine_metadata': {},
            'consensus_metadata': {}
        }
        
        # Collect metadata from each engine
        for engine_name, result in valid_results.items():
            if hasattr(result, 'metadata'):
                merged_metadata['engine_metadata'][engine_name] = result.metadata
        
        # Determine primary engine (highest weighted confidence)
        best_engine = max(
            valid_results.items(),
            key=lambda x: x[1].confidence * self.engine_weights.get(x[0], 1.0)
        )
        merged_metadata['primary_engine'] = best_engine[0]
        
        return merged_metadata
    
    def _create_final_result(self, consensus: Dict, valid_results: Dict) -> Dict:
        """Create the final merged result"""
        return {
            'status': 'success',
            'consensus_text': consensus['text_consensus']['text'],
            'consensus_method': consensus['text_consensus']['method'],
            'overall_confidence': consensus['confidence_analysis']['weighted_average'],
            'agreement_score': consensus['engine_agreement']['overall_agreement'],
            'agreement_level': consensus['engine_agreement']['agreement_level'],
            
            # Detailed results
            'words': consensus['word_consensus'],
            'lines': consensus['line_consensus'],
            'bboxes': consensus['bbox_consensus'],
            
            # Analysis
            'confidence_analysis': consensus['confidence_analysis'],
            'quality_metrics': consensus['quality_metrics'],
            'metadata': consensus['metadata'],
            
            # Raw engine results for reference
            'engine_results': {name: result.to_dict() for name, result in valid_results.items()},
            
            # Variations for comparison
            'variations': {
                name: {
                    'text': result.text,
                    'confidence': result.confidence,
                    'similarity_to_consensus': SequenceMatcher(
                        None, result.text, consensus['text_consensus']['text']
                    ).ratio()
                }
                for name, result in valid_results.items()
            }
        }
    
    def update_engine_weights(self, engine_name: str, performance_score: float):
        """Update engine weights based on performance feedback"""
        self.engine_performance[engine_name].append(performance_score)
        
        # Keep only recent performance data
        if len(self.engine_performance[engine_name]) > 100:
            self.engine_performance[engine_name] = self.engine_performance[engine_name][-100:]
        
        # Update weight based on recent performance
        recent_scores = self.engine_performance[engine_name][-20:]  # Last 20 documents
        avg_performance = np.mean(recent_scores)
        
        # Adjust weight (bounded between 0.5 and 2.0)
        self.engine_weights[engine_name] = max(0.5, min(2.0, avg_performance))
        
        logger.info(f"Updated {engine_name} weight to {self.engine_weights[engine_name]:.2f}")
    
    def get_engine_performance_summary(self) -> Dict:
        """Get performance summary for all engines"""
        summary = {}
        
        for engine_name, scores in self.engine_performance.items():
            if scores:
                summary[engine_name] = {
                    'current_weight': self.engine_weights.get(engine_name, 1.0),
                    'avg_performance': np.mean(scores),
                    'recent_performance': np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
                    'sample_count': len(scores),
                    'performance_trend': self._calculate_trend(scores) if len(scores) > 5 else 'insufficient_data'
                }
        
        return summary
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate performance trend"""
        if len(scores) < 6:
            return 'insufficient_data'
        
        # Simple trend calculation
        recent = np.mean(scores[-5:])
        older = np.mean(scores[-10:-5]) if len(scores) >= 10 else np.mean(scores[:-5])
        
        diff = recent - older
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'