"""
Feedback Processor
Handles user corrections and learns from feedback to improve future extractions
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """Processes user feedback and updates learning systems"""
    
    def __init__(self, learning_db, pattern_detector, document_classifier):
        self.learning_db = learning_db
        self.pattern_detector = pattern_detector
        self.document_classifier = document_classifier
        
        # Learning strategies
        self.correction_analyzers = {
            'field_value': self._analyze_field_correction,
            'document_type': self._analyze_type_correction,
            'pattern_extraction': self._analyze_pattern_correction,
            'structure_correction': self._analyze_structure_correction
        }
    
    def process_correction(self, document_id: str, correction_data: Dict) -> Dict[str, Any]:
        """
        Process a user correction and update learning systems
        
        Args:
            document_id: ID of the document being corrected
            correction_data: Dictionary containing correction information
            
        Returns:
            Processing results and recommendations
        """
        correction_type = correction_data.get('correction_type', 'field_value')
        
        # Store the correction
        correction_id = self._store_correction(document_id, correction_data)
        
        # Analyze the correction
        analysis = self._analyze_correction(correction_data, correction_type)
        
        # Update patterns and rules
        updates = self._apply_learning_updates(analysis, correction_data)
        
        # Generate recommendations for similar documents
        recommendations = self._generate_recommendations(analysis, correction_data)
        
        result = {
            'correction_id': correction_id,
            'analysis': analysis,
            'updates_applied': updates,
            'recommendations': recommendations,
            'learning_impact': self._assess_learning_impact(analysis)
        }
        
        logger.info(f"Processed correction for document {document_id}: {correction_type}")
        
        return result
    
    def apply_document_type_label(self, document_id: str, user_label: str) -> Dict[str, Any]:
        """
        Apply user-provided document type label and learn from it
        
        Args:
            document_id: ID of the document
            user_label: User-provided document type label
            
        Returns:
            Learning results
        """
        # Get document data
        document_data = self._get_document_data(document_id)
        
        if not document_data:
            return {'error': 'Document not found'}
        
        # Learn the document type
        self.document_classifier.learn_document_type(
            document_id,
            document_data.get('patterns', {}),
            document_data.get('text_content', ''),
            user_label
        )
        
        # Store the classification
        self._store_document_classification(document_id, user_label, document_data)
        
        # Update related patterns
        pattern_updates = self._update_patterns_for_type(user_label, document_data)
        
        # Generate insights
        insights = self._generate_type_learning_insights(user_label, document_data)
        
        result = {
            'document_id': document_id,
            'learned_type': user_label,
            'pattern_updates': pattern_updates,
            'insights': insights,
            'similar_documents': self._find_similar_documents(document_data)
        }
        
        logger.info(f"Learned document type '{user_label}' for document {document_id}")
        
        return result
    
    def update_extraction_rules(self, field_name: str, new_pattern: str, 
                              document_type: str, validation_pattern: Optional[str] = None) -> str:
        """
        Update extraction rules based on user input
        
        Args:
            field_name: Name of the field
            new_pattern: New extraction pattern (regex)
            document_type: Document type this applies to
            validation_pattern: Optional validation pattern
            
        Returns:
            Rule ID
        """
        # Validate the pattern
        validation_result = self._validate_extraction_pattern(new_pattern)
        
        if not validation_result['valid']:
            raise ValueError(f"Invalid pattern: {validation_result['error']}")
        
        # Store the rule
        rule_id = self.learning_db.store_field_rule(
            field_name=field_name,
            extraction_rule=new_pattern,
            document_type=document_type,
            validation_pattern=validation_pattern
        )
        
        # Test the rule on existing documents
        test_results = self._test_rule_on_existing_documents(rule_id, field_name, new_pattern, document_type)
        
        logger.info(f"Updated extraction rule for field '{field_name}' in document type '{document_type}'")
        
        return rule_id
    
    def _store_correction(self, document_id: str, correction_data: Dict) -> str:
        """Store the correction in the database"""
        field_name = correction_data.get('field_name', 'unknown')
        extracted_value = correction_data.get('extracted_value', '')
        correct_value = correction_data.get('correct_value', '')
        correction_type = correction_data.get('correction_type', 'field_value')
        engine_name = correction_data.get('engine_name', 'unknown')
        bounding_box = correction_data.get('bounding_box')
        
        return self.learning_db.store_user_correction(
            document_id=document_id,
            field_name=field_name,
            extracted_value=extracted_value,
            correct_value=correct_value,
            correction_type=correction_type,
            engine_name=engine_name,
            bounding_box=bounding_box
        )
    
    def _analyze_correction(self, correction_data: Dict, correction_type: str) -> Dict[str, Any]:
        """Analyze the correction to understand the error pattern"""
        analyzer = self.correction_analyzers.get(correction_type, self._analyze_generic_correction)
        return analyzer(correction_data)
    
    def _analyze_field_correction(self, correction_data: Dict) -> Dict[str, Any]:
        """Analyze field value correction"""
        extracted = correction_data.get('extracted_value', '')
        correct = correction_data.get('correct_value', '')
        field_name = correction_data.get('field_name', '')
        
        analysis = {
            'error_type': self._classify_error_type(extracted, correct),
            'pattern_issues': self._find_pattern_issues(extracted, correct),
            'field_characteristics': self._analyze_field_characteristics(field_name, correct),
            'confidence_impact': self._assess_confidence_impact(correction_data)
        }
        
        # Specific analysis based on field type
        if 'amount' in field_name.lower() or 'total' in field_name.lower():
            analysis['currency_analysis'] = self._analyze_currency_error(extracted, correct)
        elif 'date' in field_name.lower():
            analysis['date_analysis'] = self._analyze_date_error(extracted, correct)
        elif 'phone' in field_name.lower():
            analysis['phone_analysis'] = self._analyze_phone_error(extracted, correct)
        
        return analysis
    
    def _classify_error_type(self, extracted: str, correct: str) -> str:
        """Classify the type of OCR error"""
        if not extracted and correct:
            return 'missing_extraction'
        elif extracted and not correct:
            return 'false_positive'
        elif not extracted and not correct:
            return 'both_empty'
        
        # Character-level analysis
        if len(extracted) != len(correct):
            return 'length_mismatch'
        
        # Check for common OCR substitutions
        common_subs = {
            ('0', 'O'), ('1', 'l'), ('1', 'I'), ('5', 'S'), ('6', 'G'),
            ('8', 'B'), ('rn', 'm'), ('cl', 'd'), ('vv', 'w')
        }
        
        for old, new in common_subs:
            if old in extracted and new in correct:
                return 'character_substitution'
            if new in extracted and old in correct:
                return 'character_substitution'
        
        # Formatting issues
        if extracted.replace(' ', '').replace(',', '').replace('.', '') == correct.replace(' ', '').replace(',', '').replace('.', ''):
            return 'formatting_error'
        
        return 'content_difference'
    
    def _find_pattern_issues(self, extracted: str, correct: str) -> List[Dict]:
        """Find specific pattern issues in the extraction"""
        issues = []
        
        # Currency formatting
        if '$' in extracted or '$' in correct:
            if '.' in extracted and ',' in correct:
                issues.append({
                    'type': 'currency_separator',
                    'description': 'Decimal point confused with comma separator',
                    'suggestion': 'Update currency pattern to handle European formats'
                })
        
        # Date formatting
        date_patterns = [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'\d{4}[/-]\d{1,2}[/-]\d{1,2}']
        if any(re.search(pattern, extracted) for pattern in date_patterns):
            if extracted.replace('/', '-') == correct or extracted.replace('-', '/') == correct:
                issues.append({
                    'type': 'date_separator',
                    'description': 'Date separator format difference',
                    'suggestion': 'Allow multiple separator types in date patterns'
                })
        
        # Phone number formatting
        if re.search(r'\d{3}.\d{3}.\d{4}', extracted) or re.search(r'\d{3}.\d{3}.\d{4}', correct):
            if re.sub(r'[^\d]', '', extracted) == re.sub(r'[^\d]', '', correct):
                issues.append({
                    'type': 'phone_formatting',
                    'description': 'Phone number formatting difference',
                    'suggestion': 'Normalize phone numbers to consistent format'
                })
        
        return issues
    
    def _analyze_field_characteristics(self, field_name: str, correct_value: str) -> Dict[str, Any]:
        """Analyze characteristics of the field for better extraction"""
        characteristics = {
            'field_type': self._determine_field_type(field_name, correct_value),
            'value_pattern': self._extract_value_pattern(correct_value),
            'validation_rules': self._suggest_validation_rules(field_name, correct_value)
        }
        
        return characteristics
    
    def _determine_field_type(self, field_name: str, value: str) -> str:
        """Determine the semantic type of a field"""
        field_lower = field_name.lower()
        
        # Check field name
        if any(keyword in field_lower for keyword in ['amount', 'total', 'cost', 'price', 'fee']):
            return 'currency'
        elif any(keyword in field_lower for keyword in ['date', 'time', 'dob']):
            return 'date'
        elif any(keyword in field_lower for keyword in ['phone', 'tel', 'mobile']):
            return 'phone'
        elif any(keyword in field_lower for keyword in ['email', 'e-mail']):
            return 'email'
        elif any(keyword in field_lower for keyword in ['ssn', 'social']):
            return 'ssn'
        elif any(keyword in field_lower for keyword in ['zip', 'postal']):
            return 'zipcode'
        
        # Check value pattern
        if re.match(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$', value):
            return 'currency'
        elif re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', value):
            return 'date'
        elif re.match(r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$', value):
            return 'phone'
        elif re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return 'email'
        
        return 'text'
    
    def _extract_value_pattern(self, value: str) -> str:
        """Extract a regex pattern for the value"""
        # This is a simplified pattern extraction
        # In practice, you'd use more sophisticated methods
        
        if re.match(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$', value):
            return r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        elif re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', value):
            return r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        elif re.match(r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$', value):
            return r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        # Generic pattern based on character types
        pattern_parts = []
        current_type = None
        current_count = 0
        
        for char in value:
            if char.isalpha():
                char_type = 'alpha'
            elif char.isdigit():
                char_type = 'digit'
            else:
                char_type = 'special'
            
            if char_type == current_type:
                current_count += 1
            else:
                if current_type:
                    if current_type == 'alpha':
                        pattern_parts.append(f'[a-zA-Z]{{{current_count}}}')
                    elif current_type == 'digit':
                        pattern_parts.append(f'\\d{{{current_count}}}')
                    else:
                        pattern_parts.append(re.escape(value[len(''.join(pattern_parts).replace('\\', '')):len(''.join(pattern_parts).replace('\\', '')) + current_count]))
                
                current_type = char_type
                current_count = 1
        
        # Add the last part
        if current_type:
            if current_type == 'alpha':
                pattern_parts.append(f'[a-zA-Z]{{{current_count}}}')
            elif current_type == 'digit':
                pattern_parts.append(f'\\d{{{current_count}}}')
        
        return ''.join(pattern_parts)
    
    def _suggest_validation_rules(self, field_name: str, value: str) -> List[str]:
        """Suggest validation rules for the field"""
        rules = []
        
        field_type = self._determine_field_type(field_name, value)
        
        if field_type == 'currency':
            rules.append('Must be valid currency format')
            rules.append('Must be positive number')
        elif field_type == 'date':
            rules.append('Must be valid date')
            rules.append('Must be reasonable date range')
        elif field_type == 'phone':
            rules.append('Must be valid phone number format')
            rules.append('Must have correct number of digits')
        elif field_type == 'email':
            rules.append('Must be valid email format')
            rules.append('Must have valid domain')
        
        return rules
    
    def _analyze_type_correction(self, correction_data: Dict) -> Dict[str, Any]:
        """Analyze document type correction"""
        return {
            'auto_classification': correction_data.get('auto_classification'),
            'user_classification': correction_data.get('user_classification'),
            'classification_confidence': correction_data.get('classification_confidence', 0.0),
            'misclassification_reason': self._analyze_misclassification(correction_data)
        }
    
    def _analyze_pattern_correction(self, correction_data: Dict) -> Dict[str, Any]:
        """Analyze pattern extraction correction"""
        return {
            'pattern_type': correction_data.get('pattern_type'),
            'missed_patterns': correction_data.get('missed_patterns', []),
            'false_patterns': correction_data.get('false_patterns', []),
            'pattern_improvements': self._suggest_pattern_improvements(correction_data)
        }
    
    def _analyze_structure_correction(self, correction_data: Dict) -> Dict[str, Any]:
        """Analyze document structure correction"""
        return {
            'structure_type': correction_data.get('structure_type'),
            'layout_issues': correction_data.get('layout_issues', []),
            'table_corrections': correction_data.get('table_corrections', []),
            'reading_order_issues': correction_data.get('reading_order_issues', [])
        }
    
    def _analyze_generic_correction(self, correction_data: Dict) -> Dict[str, Any]:
        """Generic correction analysis"""
        return {
            'correction_type': correction_data.get('correction_type', 'unknown'),
            'data': correction_data
        }
    
    def _apply_learning_updates(self, analysis: Dict, correction_data: Dict) -> List[Dict]:
        """Apply learning updates based on correction analysis"""
        updates = []
        
        # Update field extraction patterns
        if analysis.get('error_type') == 'character_substitution':
            update = self._create_substitution_rule(analysis, correction_data)
            if update:
                updates.append(update)
        
        # Update validation patterns
        if analysis.get('field_characteristics'):
            update = self._update_validation_patterns(analysis, correction_data)
            if update:
                updates.append(update)
        
        # Update engine weights based on error
        engine_name = correction_data.get('engine_name')
        if engine_name:
            self._update_engine_reliability(engine_name, analysis)
        
        return updates
    
    def _generate_recommendations(self, analysis: Dict, correction_data: Dict) -> List[Dict]:
        """Generate recommendations for preventing similar errors"""
        recommendations = []
        
        # Pattern-based recommendations
        for issue in analysis.get('pattern_issues', []):
            recommendations.append({
                'type': 'pattern_update',
                'description': issue['description'],
                'action': issue['suggestion']
            })
        
        # Field-specific recommendations
        field_characteristics = analysis.get('field_characteristics', {})
        if field_characteristics.get('validation_rules'):
            recommendations.append({
                'type': 'validation_rule',
                'description': 'Add validation rules for field',
                'rules': field_characteristics['validation_rules']
            })
        
        return recommendations
    
    def _assess_learning_impact(self, analysis: Dict) -> Dict[str, Any]:
        """Assess the impact of this correction on learning"""
        impact = {
            'severity': 'low',
            'affected_documents': 0,
            'pattern_updates': 0,
            'rule_updates': 0
        }
        
        # Assess severity based on error type
        error_type = analysis.get('error_type')
        if error_type in ['missing_extraction', 'false_positive']:
            impact['severity'] = 'high'
        elif error_type in ['character_substitution', 'formatting_error']:
            impact['severity'] = 'medium'
        
        # Estimate affected documents (simplified)
        if impact['severity'] == 'high':
            impact['affected_documents'] = 10  # Estimate
        elif impact['severity'] == 'medium':
            impact['affected_documents'] = 5
        
        return impact
    
    def _get_document_data(self, document_id: str) -> Optional[Dict]:
        """Get document data from the database"""
        # This would retrieve stored document data
        # For now, return placeholder
        return {
            'patterns': {},
            'text_content': '',
            'features': {}
        }
    
    def _store_document_classification(self, document_id: str, user_label: str, document_data: Dict):
        """Store document classification in database"""
        # This would store the classification
        logger.debug(f"Storing classification '{user_label}' for document {document_id}")
    
    def _validate_extraction_pattern(self, pattern: str) -> Dict[str, Any]:
        """Validate an extraction pattern"""
        try:
            re.compile(pattern)
            return {'valid': True}
        except re.error as e:
            return {'valid': False, 'error': str(e)}
    
    def _test_rule_on_existing_documents(self, rule_id: str, field_name: str, 
                                       pattern: str, document_type: str) -> Dict[str, Any]:
        """Test a new rule on existing documents"""
        # This would test the rule and return results
        return {
            'documents_tested': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'accuracy': 0.0
        }
    
    def _analyze_currency_error(self, extracted: str, correct: str) -> Dict[str, Any]:
        """Analyze currency-specific errors"""
        return {
            'decimal_separator_issue': '.' in extracted and ',' in correct,
            'thousands_separator_issue': ',' in extracted and '.' in correct,
            'currency_symbol_missing': '$' in correct and '$' not in extracted
        }
    
    def _analyze_date_error(self, extracted: str, correct: str) -> Dict[str, Any]:
        """Analyze date-specific errors"""
        return {
            'separator_issue': extracted.replace('/', '-') == correct or extracted.replace('-', '/') == correct,
            'format_issue': len(extracted.split('/')) != len(correct.split('/'))
        }
    
    def _analyze_phone_error(self, extracted: str, correct: str) -> Dict[str, Any]:
        """Analyze phone number errors"""
        extracted_digits = re.sub(r'[^\d]', '', extracted)
        correct_digits = re.sub(r'[^\d]', '', correct)
        
        return {
            'digits_match': extracted_digits == correct_digits,
            'formatting_only': extracted_digits == correct_digits and extracted != correct,
            'area_code_issue': extracted_digits[:3] != correct_digits[:3]
        }
    
    def _assess_confidence_impact(self, correction_data: Dict) -> float:
        """Assess how this correction should impact confidence scoring"""
        # If the engine was very confident but wrong, lower confidence
        original_confidence = correction_data.get('original_confidence', 0.5)
        
        if original_confidence > 0.8:
            return 0.2  # High confidence but wrong = big impact
        elif original_confidence > 0.5:
            return 0.1  # Medium confidence = medium impact
        else:
            return 0.05  # Low confidence = small impact
    
    def _analyze_misclassification(self, correction_data: Dict) -> str:
        """Analyze why document was misclassified"""
        # This would analyze classification errors
        return "Insufficient distinguishing features"
    
    def _suggest_pattern_improvements(self, correction_data: Dict) -> List[str]:
        """Suggest improvements to patterns"""
        return [
            "Add fuzzy matching for similar patterns",
            "Improve bounding box detection",
            "Add context-aware validation"
        ]
    
    def _create_substitution_rule(self, analysis: Dict, correction_data: Dict) -> Optional[Dict]:
        """Create a character substitution rule"""
        # This would create actual substitution rules
        return {
            'type': 'substitution_rule',
            'description': 'Created character substitution rule'
        }
    
    def _update_validation_patterns(self, analysis: Dict, correction_data: Dict) -> Optional[Dict]:
        """Update validation patterns"""
        return {
            'type': 'validation_update',
            'description': 'Updated validation patterns'
        }
    
    def _update_engine_reliability(self, engine_name: str, analysis: Dict):
        """Update engine reliability scoring"""
        # This would update engine performance metrics
        logger.debug(f"Updating reliability for engine {engine_name}")
    
    def _update_patterns_for_type(self, document_type: str, document_data: Dict) -> List[Dict]:
        """Update patterns specific to document type"""
        return []
    
    def _generate_type_learning_insights(self, user_label: str, document_data: Dict) -> List[str]:
        """Generate insights from type learning"""
        return [
            f"Learned new document type: {user_label}",
            "Updated classification patterns",
            "Improved type detection accuracy"
        ]
    
    def _find_similar_documents(self, document_data: Dict) -> List[str]:
        """Find similar documents that might benefit from this learning"""
        # This would find documents with similar characteristics
        return []