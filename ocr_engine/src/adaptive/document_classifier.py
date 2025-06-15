"""
Document Classifier
Auto-discovers document types without pre-definition
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classifies documents by discovering patterns and learning from examples"""
    
    def __init__(self):
        self.keyword_clusters = {}  # Discovered keyword patterns
        self.layout_signatures = {}  # Discovered layout patterns
        self.learned_types = {}  # User-labeled document types
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.document_vectors = {}  # Stored document vectors for similarity
        
    def classify_document(self, patterns: Dict, text_content: str) -> Dict:
        """
        Classify document by discovering types and matching learned patterns
        
        Args:
            patterns: Detected patterns from pattern detector
            text_content: Full text content of document
            
        Returns:
            Classification results with discovered and learned types
        """
        # Auto-discover potential document types
        discovered_types = self.discover_document_types(patterns, text_content)
        
        # Match against learned types
        learned_matches = self.match_learned_types(patterns, text_content)
        
        # Generate descriptive labels
        auto_labels = self.generate_descriptive_labels(patterns, text_content)
        
        # Determine best guess
        best_guess = self._determine_best_guess(discovered_types, learned_matches)
        
        # Explain classification reasoning
        reasoning = self.explain_classification_logic(patterns, text_content)
        
        return {
            'discovered_types': discovered_types,
            'learned_types': learned_matches,
            'auto_generated_labels': auto_labels,
            'best_overall_guess': best_guess,
            'confidence_reasoning': reasoning
        }
    
    def discover_document_types(self, patterns: Dict, text_content: str) -> Dict[str, Dict]:
        """Auto-discover document types based on content analysis"""
        discoveries = {}
        
        # Extract signatures
        keyword_sig = self.extract_keyword_signature(text_content)
        layout_sig = self.extract_layout_signature(patterns)
        form_sig = self.extract_form_signature(patterns)
        
        # Government/Tax forms
        if keyword_sig.get('government_forms', 0) > 0.7:
            discoveries['government_form'] = {
                'confidence': keyword_sig['government_forms'],
                'reasoning': self._get_gov_form_reasoning(keyword_sig, patterns)
            }
        
        # Financial documents
        if keyword_sig.get('financial_documents', 0) > 0.6:
            discoveries['financial_document'] = {
                'confidence': keyword_sig['financial_documents'],
                'reasoning': self._get_financial_reasoning(keyword_sig, patterns)
            }
        
        # Tabular documents
        if layout_sig.get('table_density', 0) > 0.8:
            discoveries['tabular_document'] = {
                'confidence': layout_sig['table_density'],
                'reasoning': ['high table density', 'structured data layout']
            }
        
        # Form documents
        if form_sig.get('field_count', 0) > 10:
            discoveries['form_document'] = {
                'confidence': min(form_sig['field_count'] / 20, 1.0),
                'reasoning': ['multiple form fields', 'structured input layout']
            }
        
        # Legal documents
        if keyword_sig.get('legal_documents', 0) > 0.5:
            discoveries['legal_document'] = {
                'confidence': keyword_sig['legal_documents'],
                'reasoning': self._get_legal_reasoning(keyword_sig, patterns)
            }
        
        # Medical documents
        if keyword_sig.get('medical_documents', 0) > 0.5:
            discoveries['medical_document'] = {
                'confidence': keyword_sig['medical_documents'],
                'reasoning': ['medical terminology', 'health-related content']
            }
        
        # Invoice/Receipt
        if self._is_invoice_like(patterns, keyword_sig):
            discoveries['invoice_receipt'] = {
                'confidence': 0.8,
                'reasoning': ['line items', 'totals', 'transaction structure']
            }
        
        # Report/Letter
        if self._is_report_like(patterns, layout_sig):
            discoveries['report_letter'] = {
                'confidence': 0.7,
                'reasoning': ['narrative structure', 'sections', 'formal layout']
            }
        
        return discoveries
    
    def extract_keyword_signature(self, text_content: str) -> Dict[str, float]:
        """Analyze text for keyword patterns that suggest document types"""
        text_lower = text_content.lower()
        words = text_lower.split()
        
        signatures = {}
        
        # Government/Tax indicators
        gov_keywords = [
            'form', 'irs', 'tax', 'federal', 'state', 'department', 'bureau',
            'agency', 'government', 'treasury', 'revenue', 'return', 'schedule'
        ]
        signatures['government_forms'] = self._calculate_keyword_score(words, gov_keywords)
        
        # Financial indicators
        financial_keywords = [
            'amount', 'total', 'balance', 'payment', 'invoice', 'bill', 'credit',
            'debit', 'transaction', 'account', 'bank', 'interest', 'fee', 'charge'
        ]
        signatures['financial_documents'] = self._calculate_keyword_score(words, financial_keywords)
        
        # Legal/Real Estate indicators
        legal_keywords = [
            'deed', 'property', 'legal', 'agreement', 'contract', 'party', 'whereas',
            'covenant', 'easement', 'title', 'estate', 'plaintiff', 'defendant'
        ]
        signatures['legal_documents'] = self._calculate_keyword_score(words, legal_keywords)
        
        # Medical indicators
        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'doctor', 'medical', 'health',
            'prescription', 'medication', 'hospital', 'clinic', 'symptoms'
        ]
        signatures['medical_documents'] = self._calculate_keyword_score(words, medical_keywords)
        
        # Business correspondence
        business_keywords = [
            'dear', 'sincerely', 'regards', 'subject', 're:', 'memo', 'to:', 'from:',
            'date:', 'cc:', 'attachment', 'enclosed'
        ]
        signatures['business_correspondence'] = self._calculate_keyword_score(words, business_keywords)
        
        return signatures
    
    def _calculate_keyword_score(self, words: List[str], keywords: List[str]) -> float:
        """Calculate keyword match score"""
        if not words:
            return 0.0
        
        # Count keyword occurrences
        keyword_count = sum(1 for word in words if word in keywords)
        
        # Also check for partial matches
        partial_count = sum(
            1 for word in words 
            for keyword in keywords 
            if keyword in word and keyword != word
        ) * 0.5
        
        total_matches = keyword_count + partial_count
        
        # Normalize by document length and keyword list size
        score = total_matches / (np.sqrt(len(words)) * np.sqrt(len(keywords)))
        
        return min(score, 1.0)
    
    def extract_layout_signature(self, patterns: Dict) -> Dict[str, float]:
        """Analyze layout patterns to infer document type"""
        signatures = {}
        
        # Table density
        table_count = len(patterns.get('tables', []))
        total_elements = sum(
            len(patterns.get(key, [])) 
            for key in ['sections', 'lists', 'key_value_pairs']
        )
        
        signatures['table_density'] = table_count / max(total_elements, 1)
        
        # Form field ratio
        form_count = len(patterns.get('forms', []))
        kv_count = len(patterns.get('key_value_pairs', []))
        
        signatures['form_field_ratio'] = form_count / max(kv_count, 1)
        
        # List dominance
        list_count = len(patterns.get('lists', []))
        section_count = len(patterns.get('sections', []))
        
        signatures['list_dominance'] = list_count / max(section_count, 1)
        
        # Numeric density
        numeric_patterns = patterns.get('numbers', {})
        numeric_count = sum(
            len(items) if isinstance(items, list) else 0 
            for items in numeric_patterns.values()
        )
        
        signatures['numeric_density'] = numeric_count / max(kv_count, 1)
        
        # Structure complexity
        signatures['structure_complexity'] = len([
            k for k, v in patterns.items() 
            if isinstance(v, list) and len(v) > 0
        ]) / 10.0
        
        return signatures
    
    def extract_form_signature(self, patterns: Dict) -> Dict[str, Any]:
        """Analyze form-specific characteristics"""
        form_fields = patterns.get('forms', [])
        
        signature = {
            'field_count': len(form_fields),
            'has_checkboxes': False,
            'has_signatures': False,
            'structured_layout': False,
            'empty_fields_ratio': 0.0
        }
        
        if form_fields:
            # Check for checkboxes
            signature['has_checkboxes'] = any(
                field.get('field_type') == 'checkbox' 
                for field in form_fields
            )
            
            # Check for signature fields
            signature['has_signatures'] = any(
                field.get('field_type') == 'signature' 
                for field in form_fields
            )
            
            # Calculate empty fields ratio
            empty_count = sum(
                1 for field in form_fields 
                if not field.get('has_value', True)
            )
            signature['empty_fields_ratio'] = empty_count / len(form_fields)
        
        # Check structured layout
        signature['structured_layout'] = (
            len(patterns.get('sections', [])) > 3 or
            len(patterns.get('tables', [])) > 0
        )
        
        return signature
    
    def generate_descriptive_labels(self, patterns: Dict, text_content: str) -> List[str]:
        """Auto-generate human-readable document type labels"""
        labels = []
        text_words = text_content.lower().split()
        
        # Look for form numbers/identifiers
        form_patterns = [
            r'\bform\s+(\w+-?\w*)\b',
            r'\b(W-\d+|1099|1040|I-\d+)\b',
            r'\bschedule\s+(\w+)\b'
        ]
        
        for pattern in form_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                form_id = match if isinstance(match, str) else match[0]
                labels.append(f"form_document_{form_id.lower()}")
        
        # Look for organization names
        org_patterns = [
            r'(?:from|by)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:inc|llc|corp|company|department|bureau)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:invoice|statement|report|letter)'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text_content)
            for match in matches[:2]:  # Limit to first 2 matches
                org_name = match.lower().replace(' ', '_')
                labels.append(f"document_from_{org_name}")
        
        # Look for document purpose
        purpose_keywords = {
            'statement': 'financial_statement',
            'invoice': 'billing_document',
            'receipt': 'payment_record',
            'agreement': 'legal_contract',
            'deed': 'property_document',
            'report': 'analytical_report',
            'return': 'tax_return',
            'application': 'application_form',
            'claim': 'claim_form',
            'notice': 'official_notice'
        }
        
        for keyword, label in purpose_keywords.items():
            if keyword in text_words:
                labels.append(label)
        
        # Add year if found
        year_match = re.search(r'\b(19|20)\d{2}\b', text_content)
        if year_match and len(labels) > 0:
            labels[0] = f"{labels[0]}_{year_match.group()}"
        
        # Ensure uniqueness
        return list(dict.fromkeys(labels))[:5]  # Return top 5 unique labels
    
    def match_learned_types(self, patterns: Dict, text_content: str) -> Dict[str, Dict]:
        """Match against previously learned document types"""
        if not self.learned_types:
            return {}
        
        # Extract current document features
        current_features = self._extract_document_features(patterns, text_content)
        
        matches = {}
        
        for type_name, type_data in self.learned_types.items():
            similarity = self._calculate_similarity(current_features, type_data['features'])
            
            if similarity > 0.6:  # Threshold for considering a match
                matches[type_name] = {
                    'confidence': similarity,
                    'reasoning': self._explain_similarity(current_features, type_data['features'])
                }
        
        return matches
    
    def _extract_document_features(self, patterns: Dict, text_content: str) -> Dict:
        """Extract feature vector for document comparison"""
        features = {
            'keyword_signature': self.extract_keyword_signature(text_content),
            'layout_signature': self.extract_layout_signature(patterns),
            'form_signature': self.extract_form_signature(patterns),
            'pattern_counts': {
                'tables': len(patterns.get('tables', [])),
                'lists': len(patterns.get('lists', [])),
                'forms': len(patterns.get('forms', [])),
                'sections': len(patterns.get('sections', []))
            },
            'text_features': {
                'length': len(text_content),
                'word_count': len(text_content.split()),
                'line_count': len(text_content.split('\n'))
            }
        }
        
        return features
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two document feature sets"""
        similarities = []
        
        # Compare keyword signatures
        if 'keyword_signature' in features1 and 'keyword_signature' in features2:
            keyword_sim = self._compare_signatures(
                features1['keyword_signature'],
                features2['keyword_signature']
            )
            similarities.append(keyword_sim)
        
        # Compare layout signatures
        if 'layout_signature' in features1 and 'layout_signature' in features2:
            layout_sim = self._compare_signatures(
                features1['layout_signature'],
                features2['layout_signature']
            )
            similarities.append(layout_sim)
        
        # Compare pattern counts
        if 'pattern_counts' in features1 and 'pattern_counts' in features2:
            pattern_sim = self._compare_pattern_counts(
                features1['pattern_counts'],
                features2['pattern_counts']
            )
            similarities.append(pattern_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compare_signatures(self, sig1: Dict, sig2: Dict) -> float:
        """Compare two signature dictionaries"""
        if not sig1 or not sig2:
            return 0.0
        
        common_keys = set(sig1.keys()) & set(sig2.keys())
        if not common_keys:
            return 0.0
        
        differences = []
        for key in common_keys:
            val1, val2 = sig1[key], sig2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(val1, val2, 0.1)
                similarity = 1.0 - abs(val1 - val2) / max_val
                differences.append(similarity)
        
        return np.mean(differences) if differences else 0.0
    
    def _compare_pattern_counts(self, counts1: Dict, counts2: Dict) -> float:
        """Compare pattern count dictionaries"""
        # Convert to vectors
        all_keys = set(counts1.keys()) | set(counts2.keys())
        vec1 = [counts1.get(k, 0) for k in all_keys]
        vec2 = [counts2.get(k, 0) for k in all_keys]
        
        # Cosine similarity
        if sum(vec1) == 0 or sum(vec2) == 0:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = np.sqrt(sum(a * a for a in vec1))
        magnitude2 = np.sqrt(sum(b * b for b in vec2))
        
        return dot_product / (magnitude1 * magnitude2)
    
    def learn_document_type(self, document_id: str, patterns: Dict, text_content: str, user_label: str):
        """Learn a new document type from user labeling"""
        features = self._extract_document_features(patterns, text_content)
        
        if user_label not in self.learned_types:
            self.learned_types[user_label] = {
                'features': features,
                'examples': [],
                'created_at': None  # Would add timestamp
            }
        
        # Update with new example
        self.learned_types[user_label]['examples'].append({
            'document_id': document_id,
            'features': features
        })
        
        # Update average features
        self._update_type_features(user_label)
    
    def _update_type_features(self, type_name: str):
        """Update average features for a document type"""
        if type_name not in self.learned_types:
            return
        
        examples = self.learned_types[type_name]['examples']
        if not examples:
            return
        
        # Average all feature vectors
        # This is simplified - in practice would use more sophisticated averaging
        avg_features = examples[0]['features'].copy()
        
        # Store updated features
        self.learned_types[type_name]['features'] = avg_features
    
    def _determine_best_guess(self, discovered: Dict, learned: Dict) -> str:
        """Determine the best overall document type guess"""
        all_candidates = []
        
        # Add discovered types
        for dtype, info in discovered.items():
            all_candidates.append((dtype, info['confidence'], 'discovered'))
        
        # Add learned types
        for dtype, info in learned.items():
            all_candidates.append((dtype, info['confidence'], 'learned'))
        
        if not all_candidates:
            return 'unknown_document'
        
        # Sort by confidence
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Prefer learned types if confidence is close
        best = all_candidates[0]
        for candidate in all_candidates[1:]:
            if candidate[2] == 'learned' and candidate[1] > best[1] * 0.9:
                best = candidate
                break
        
        return best[0]
    
    def explain_classification_logic(self, patterns: Dict, text_content: str) -> List[str]:
        """Explain the reasoning behind classification"""
        reasoning = []
        
        # Analyze dominant patterns
        pattern_counts = {
            k: len(v) if isinstance(v, list) else 0 
            for k, v in patterns.items()
        }
        
        dominant = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for pattern_type, count in dominant:
            if count > 0:
                reasoning.append(f"Contains {count} {pattern_type}")
        
        # Analyze keywords
        keyword_sig = self.extract_keyword_signature(text_content)
        top_categories = sorted(keyword_sig.items(), key=lambda x: x[1], reverse=True)[:2]
        
        for category, score in top_categories:
            if score > 0.3:
                reasoning.append(f"High {category.replace('_', ' ')} keyword density")
        
        # Analyze structure
        if len(patterns.get('tables', [])) > 2:
            reasoning.append("Multiple tables suggest structured data document")
        
        if len(patterns.get('forms', [])) > 5:
            reasoning.append("Multiple form fields suggest fillable form")
        
        return reasoning
    
    def _get_gov_form_reasoning(self, keyword_sig: Dict, patterns: Dict) -> List[str]:
        """Get reasoning for government form classification"""
        reasons = ['government terminology', 'official format']
        
        if 'form_number' in patterns.get('numbers', {}):
            reasons.append('contains form number')
        
        if keyword_sig.get('government_forms', 0) > 0.8:
            reasons.append('high density of government keywords')
        
        return reasons
    
    def _get_financial_reasoning(self, keyword_sig: Dict, patterns: Dict) -> List[str]:
        """Get reasoning for financial document classification"""
        reasons = ['financial terminology']
        
        if patterns.get('numbers', {}).get('currencies'):
            reasons.append('contains currency amounts')
        
        if patterns.get('tables'):
            reasons.append('tabular financial data')
        
        return reasons
    
    def _get_legal_reasoning(self, keyword_sig: Dict, patterns: Dict) -> List[str]:
        """Get reasoning for legal document classification"""
        reasons = ['legal terminology', 'formal structure']
        
        if patterns.get('sections'):
            reasons.append('structured sections')
        
        return reasons
    
    def _is_invoice_like(self, patterns: Dict, keyword_sig: Dict) -> bool:
        """Check if document looks like an invoice"""
        has_line_items = len(patterns.get('tables', [])) > 0
        has_totals = any(
            'total' in kv.get('key', '').lower() 
            for kv in patterns.get('key_value_pairs', [])
        )
        has_amounts = len(patterns.get('numbers', {}).get('currencies', [])) > 2
        
        return has_line_items and (has_totals or has_amounts)
    
    def _is_report_like(self, patterns: Dict, layout_sig: Dict) -> bool:
        """Check if document looks like a report or letter"""
        has_sections = len(patterns.get('sections', [])) > 2
        has_paragraphs = layout_sig.get('structure_complexity', 0) > 0.3
        low_form_density = layout_sig.get('form_field_ratio', 1) < 0.3
        
        return has_sections and has_paragraphs and low_form_density
    
    def _explain_similarity(self, features1: Dict, features2: Dict) -> List[str]:
        """Explain why two documents are similar"""
        reasons = []
        
        keyword_sim = self._compare_signatures(
            features1.get('keyword_signature', {}),
            features2.get('keyword_signature', {})
        )
        if keyword_sim > 0.7:
            reasons.append(f'keyword_similarity: {keyword_sim:.2f}')
        
        layout_sim = self._compare_signatures(
            features1.get('layout_signature', {}),
            features2.get('layout_signature', {})
        )
        if layout_sim > 0.7:
            reasons.append(f'layout_similarity: {layout_sim:.2f}')
        
        return reasons
    
    def save_state(self, filepath: str):
        """Save classifier state to file"""
        state = {
            'learned_types': self.learned_types,
            'keyword_clusters': self.keyword_clusters,
            'layout_signatures': self.layout_signatures
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load classifier state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.learned_types = state.get('learned_types', {})
            self.keyword_clusters = state.get('keyword_clusters', {})
            self.layout_signatures = state.get('layout_signatures', {})
            
        except Exception as e:
            logger.warning(f"Could not load classifier state: {e}")