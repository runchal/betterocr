"""
Adaptive Multi-Engine OCR
Main orchestrator that coordinates all components for intelligent OCR processing
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
from datetime import datetime

# Import our modules
from .engines import get_engine, list_available_engines
from .adaptive.pattern_detector import PatternDetector
from .adaptive.document_classifier import DocumentClassifier
from .adaptive.feedback_processor import FeedbackProcessor
from .adaptive.learning_database import LearningDatabase
from .fusion.consensus_merger import ConsensusMerger
from .fusion.confidence_scorer import ConfidenceScorer
from .vision.visual_validator import VisualValidator

logger = logging.getLogger(__name__)


class AdaptiveMultiEngineOCR:
    """
    Main OCR orchestrator with adaptive learning capabilities
    Optimized for AI agent consumption with rich metadata and feedback mechanisms
    """
    
    def __init__(self, learning_enabled: bool = True, db_path: Optional[str] = None):
        self.learning_enabled = learning_enabled
        
        # Initialize core components
        self.engines = self._initialize_engines()
        self.pattern_detector = PatternDetector()
        self.document_classifier = DocumentClassifier()
        self.consensus_merger = ConsensusMerger()
        self.confidence_scorer = ConfidenceScorer()
        self.visual_validator = VisualValidator()
        
        # Initialize learning components
        if learning_enabled:
            self.learning_db = LearningDatabase(db_path or "data/ocr_learning.db")
            self.feedback_processor = FeedbackProcessor(
                self.learning_db, self.pattern_detector, self.document_classifier
            )
            
            # Load existing learned patterns
            self._load_learned_patterns()
        else:
            self.learning_db = None
            self.feedback_processor = None
        
        logger.info("AdaptiveMultiEngineOCR initialized")
    
    def _initialize_engines(self) -> Dict[str, Any]:
        """Initialize available OCR engines"""
        engines = {}
        available_engines = list_available_engines()
        
        for engine_name, is_available in available_engines.items():
            if is_available:
                try:
                    engine = get_engine(engine_name)
                    engines[engine_name] = engine
                    logger.info(f"Initialized {engine_name} engine")
                except Exception as e:
                    logger.warning(f"Failed to initialize {engine_name}: {e}")
        
        if not engines:
            logger.warning("No OCR engines available!")
        
        return engines
    
    def process_document(self, image_path: str, user_hints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main processing method optimized for AI agent consumption
        
        Args:
            image_path: Path to the image or PDF file
            user_hints: Optional hints about document type, expected fields, etc.
            
        Returns:
            Comprehensive OCR results with metadata and learning opportunities
        """
        document_id = self.generate_document_id()
        image_path = Path(image_path)
        
        logger.info(f"Processing document {document_id}: {image_path}")
        
        try:
            # Multi-engine OCR extraction
            raw_ocr_results = self.extract_with_all_engines(image_path, user_hints)
            
            # Build consensus from multiple engines
            consensus_result = self.consensus_merger.merge_results(raw_ocr_results)
            
            # Extract patterns from consensus text
            detected_patterns = self.pattern_detector.detect_patterns(consensus_result)
            
            # Classify document type
            classification = self.document_classifier.classify_document(
                detected_patterns, 
                consensus_result.get('consensus_text', '')
            )
            
            # Apply learned extraction rules
            structured_data = self.extract_structured_data(
                consensus_result, 
                detected_patterns, 
                classification,
                user_hints
            )
            
            # Visual validation using computer vision
            visual_validation = self.visual_validator.validate_ocr_results(
                image_path, raw_ocr_results
            )
            
            # Apply visual enhancements to consensus
            enhanced_consensus = self._apply_visual_enhancements(
                consensus_result, visual_validation
            )
            
            # Calculate comprehensive confidence scores (including visual validation)
            confidence_metrics = self.confidence_scorer.calculate_comprehensive_confidence(
                enhanced_consensus, 
                context={
                    'classification': classification, 
                    'hints': user_hints,
                    'visual_validation': visual_validation
                }
            )
            
            # Prepare comprehensive output for AI agents
            result = {
                'document_id': document_id,
                'status': 'success',
                'processing_metadata': {
                    'engines_used': list(raw_ocr_results.keys()),
                    'successful_engines': [name for name, r in raw_ocr_results.items() 
                                         if not getattr(r, 'errors', [])],
                    'processing_time': sum(getattr(r, 'processing_time', 0) 
                                         for r in raw_ocr_results.values()),
                    'timestamp': datetime.now().isoformat()
                },
                'raw_ocr': {name: result.to_dict() for name, result in raw_ocr_results.items()},
                'consensus': enhanced_consensus,
                'visual_validation': visual_validation,
                'detected_patterns': detected_patterns,
                'document_classification': classification,
                'structured_data': structured_data,
                'confidence_metrics': confidence_metrics,
                'extraction_metadata': self.get_extraction_metadata(
                    consensus_result, detected_patterns, classification
                ),
                'feedback_context': self.prepare_feedback_context(
                    document_id, consensus_result, detected_patterns, classification
                ),
                'learning_opportunities': self.identify_learning_opportunities(
                    consensus_result, detected_patterns, classification
                )
            }
            
            # Store for learning if enabled
            if self.learning_enabled:
                self._store_processing_result(document_id, result, str(image_path))
            
            logger.info(f"Successfully processed document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            return self._create_error_result(document_id, str(e))
    
    def _apply_visual_enhancements(self, consensus_result: Dict, visual_validation: Dict) -> Dict:
        """Apply visual validation enhancements to consensus results"""
        enhanced_consensus = consensus_result.copy()
        
        # Apply visual confidence adjustments
        visual_confidence = visual_validation.get('visual_confidence_scores', {})
        if visual_confidence:
            # Boost or reduce confidence based on visual validation
            visual_boost = visual_confidence.get('visual_confidence', 0.5)
            current_confidence = enhanced_consensus.get('confidence', 0.0)
            
            # Apply weighted adjustment
            weight = 0.3  # 30% influence from visual validation
            enhanced_confidence = (current_confidence * (1 - weight)) + (visual_boost * weight)
            enhanced_consensus['confidence'] = enhanced_confidence
            enhanced_consensus['visual_confidence_applied'] = True
        
        # Apply engine weight adjustments from visual validation
        consensus_enhancement = visual_validation.get('consensus_enhancement', {})
        if 'enhanced_weights' in consensus_enhancement:
            enhanced_consensus['visual_engine_weights'] = consensus_enhancement['enhanced_weights']
            enhanced_consensus['weight_adjustments'] = consensus_enhancement['weight_adjustments']
        
        # Add visual quality recommendations
        image_quality = visual_validation.get('image_quality', {})
        if 'recommendations' in image_quality:
            enhanced_consensus['visual_recommendations'] = image_quality['recommendations']
            enhanced_consensus['optimization_suggestions'] = image_quality.get('optimization_suggestions', {})
        
        return enhanced_consensus
    
    def extract_with_all_engines(self, image_path: Path, user_hints: Optional[Dict] = None) -> Dict:
        """Extract text using all available engines"""
        results = {}
        
        for engine_name, engine in self.engines.items():
            try:
                logger.debug(f"Running {engine_name} on {image_path}")
                
                # Get engine-specific config from hints
                engine_config = {}
                if user_hints and 'engine_configs' in user_hints:
                    engine_config = user_hints['engine_configs'].get(engine_name, {})
                
                # Run OCR
                result = engine.extract_text(image_path, engine_config)
                results[engine_name] = result
                
                # Store performance metrics
                if self.learning_enabled and hasattr(result, 'processing_time'):
                    document_type = user_hints.get('expected_type', 'unknown') if user_hints else 'unknown'
                    self.learning_db.store_engine_performance(
                        engine_name=engine_name,
                        document_type=document_type,
                        accuracy_score=0.8,  # Would be calculated based on validation
                        confidence_score=result.confidence,
                        processing_time=result.processing_time,
                        document_id=str(uuid.uuid4())
                    )
                
            except Exception as e:
                logger.error(f"Engine {engine_name} failed: {e}")
                # Create error result
                from .engines.base_engine import OCRResult
                error_result = OCRResult()
                error_result.engine_name = engine_name
                error_result.errors = [str(e)]
                results[engine_name] = error_result
        
        return results
    
    def extract_structured_data(self, consensus_result: Dict, detected_patterns: Dict, 
                              classification: Dict, user_hints: Optional[Dict] = None) -> Dict:
        """Extract structured data using patterns and learned rules"""
        structured_data = {
            'fields': {},
            'tables': [],
            'sections': [],
            'key_value_pairs': [],
            'forms': [],
            'confidence_scores': {}
        }
        
        # Extract key-value pairs
        kv_pairs = detected_patterns.get('key_value_pairs', [])
        for kv_pair in kv_pairs:
            field_name = kv_pair.get('key', '')
            field_value = kv_pair.get('value', '')
            
            # Apply learned extraction rules
            if self.learning_enabled:
                enhanced_value = self._apply_learned_rules(field_name, field_value, classification)
                if enhanced_value != field_value:
                    field_value = enhanced_value
            
            # Validate and score confidence
            field_confidence = self._validate_field_value(field_name, field_value)
            
            structured_data['fields'][field_name] = {
                'value': field_value,
                'confidence': field_confidence,
                'extraction_method': 'pattern_detection',
                'bounding_box': kv_pair.get('bbox'),
                'raw_extraction': kv_pair
            }
        
        # Extract tables with enhanced structure
        tables = detected_patterns.get('tables', [])
        for table in tables:
            enhanced_table = self._enhance_table_structure(table, classification)
            structured_data['tables'].append(enhanced_table)
        
        # Extract form fields
        forms = detected_patterns.get('forms', [])
        for form_field in forms:
            structured_data['forms'].append({
                'field_name': form_field.get('field_name'),
                'field_type': form_field.get('field_type'),
                'has_value': form_field.get('has_value'),
                'confidence': form_field.get('confidence'),
                'position': form_field.get('position')
            })
        
        # Apply document-type specific extraction
        best_type = classification.get('best_overall_guess', 'unknown')
        if best_type != 'unknown':
            type_specific_data = self._extract_type_specific_data(
                consensus_result, best_type, user_hints
            )
            structured_data.update(type_specific_data)
        
        return structured_data
    
    def _apply_learned_rules(self, field_name: str, field_value: str, classification: Dict) -> str:
        """Apply learned extraction rules to improve field values"""
        if not self.learning_enabled:
            return field_value
        
        # Get applicable rules
        document_type = classification.get('best_overall_guess', 'unknown')
        rules = self.learning_db.get_field_rules(field_name, document_type)
        
        for rule in rules:
            try:
                # Apply rule (simplified - would use actual regex patterns)
                enhanced_value = self._apply_extraction_rule(field_value, rule)
                if enhanced_value != field_value:
                    logger.debug(f"Applied rule {rule['id']} to field {field_name}")
                    return enhanced_value
            except Exception as e:
                logger.warning(f"Failed to apply rule {rule['id']}: {e}")
        
        return field_value
    
    def _apply_extraction_rule(self, value: str, rule: Dict) -> str:
        """Apply a specific extraction rule"""
        # This would implement the actual rule application
        # For now, return the original value
        return value
    
    def _validate_field_value(self, field_name: str, field_value: str) -> float:
        """Validate field value and return confidence score"""
        # Determine field type
        field_type = self._determine_field_type(field_name)
        
        # Validate using confidence scorer
        is_valid, confidence = self.confidence_scorer.validate_against_patterns(field_value, field_type)
        
        return confidence if is_valid else 0.3
    
    def _determine_field_type(self, field_name: str) -> str:
        """Determine field type from field name"""
        field_lower = field_name.lower()
        
        type_map = {
            'email': ['email', 'e-mail'],
            'phone': ['phone', 'tel', 'mobile'],
            'currency': ['amount', 'total', 'cost', 'price'],
            'date': ['date', 'time', 'dob'],
            'ssn': ['ssn', 'social']
        }
        
        for field_type, keywords in type_map.items():
            if any(keyword in field_lower for keyword in keywords):
                return field_type
        
        return 'text'
    
    def _enhance_table_structure(self, table: Dict, classification: Dict) -> Dict:
        """Enhance table structure with learned patterns"""
        enhanced_table = table.copy()
        
        # Add table type classification
        enhanced_table['table_type'] = self._classify_table_type(table, classification)
        
        # Enhanced cell parsing
        if 'rows' in table:
            enhanced_table['structured_rows'] = []
            for row in table['rows']:
                structured_row = self._structure_table_row(row, enhanced_table['table_type'])
                enhanced_table['structured_rows'].append(structured_row)
        
        return enhanced_table
    
    def _classify_table_type(self, table: Dict, classification: Dict) -> str:
        """Classify the type of table"""
        # Simple classification based on content
        if 'financial' in classification.get('best_overall_guess', '').lower():
            return 'financial_table'
        elif any('amount' in str(cell).lower() for row in table.get('rows', []) 
                for cell in row.get('cells', [])):
            return 'amount_table'
        else:
            return 'general_table'
    
    def _structure_table_row(self, row: Dict, table_type: str) -> Dict:
        """Structure a table row based on table type"""
        structured_row = row.copy()
        
        # Add semantic labels to cells based on table type
        if table_type == 'financial_table':
            structured_row['semantic_cells'] = self._label_financial_cells(row.get('cells', []))
        
        return structured_row
    
    def _label_financial_cells(self, cells: List[str]) -> List[Dict]:
        """Label cells in financial tables"""
        labeled_cells = []
        
        for i, cell in enumerate(cells):
            label = 'text'
            confidence = 0.5
            
            # Simple heuristics
            if i == 0:
                label = 'description'
                confidence = 0.8
            elif '$' in cell or any(c.isdigit() for c in cell):
                label = 'amount'
                confidence = 0.9
            
            labeled_cells.append({
                'value': cell,
                'semantic_label': label,
                'confidence': confidence
            })
        
        return labeled_cells
    
    def _extract_type_specific_data(self, consensus_result: Dict, document_type: str, 
                                  user_hints: Optional[Dict] = None) -> Dict:
        """Extract data specific to document type"""
        type_specific = {}
        
        if 'tax' in document_type.lower() or 'form' in document_type.lower():
            type_specific['tax_fields'] = self._extract_tax_fields(consensus_result)
        elif 'invoice' in document_type.lower():
            type_specific['invoice_fields'] = self._extract_invoice_fields(consensus_result)
        elif 'financial' in document_type.lower():
            type_specific['financial_fields'] = self._extract_financial_fields(consensus_result)
        
        return type_specific
    
    def _extract_tax_fields(self, consensus_result: Dict) -> Dict:
        """Extract tax-specific fields"""
        # This would implement tax-specific extraction logic
        return {
            'form_type': 'unknown',
            'tax_year': 'unknown',
            'taxpayer_info': {},
            'income_fields': {},
            'deduction_fields': {}
        }
    
    def _extract_invoice_fields(self, consensus_result: Dict) -> Dict:
        """Extract invoice-specific fields"""
        return {
            'invoice_number': 'unknown',
            'invoice_date': 'unknown',
            'vendor_info': {},
            'line_items': [],
            'totals': {}
        }
    
    def _extract_financial_fields(self, consensus_result: Dict) -> Dict:
        """Extract financial document fields"""
        return {
            'account_numbers': [],
            'balances': [],
            'transactions': [],
            'dates': []
        }
    
    def get_extraction_metadata(self, consensus_result: Dict, detected_patterns: Dict, 
                              classification: Dict) -> Dict:
        """Get metadata about the extraction process"""
        return {
            'extraction_strategy': consensus_result.get('consensus_method', 'unknown'),
            'pattern_confidence': self._calculate_pattern_confidence(detected_patterns),
            'classification_confidence': classification.get('best_overall_guess', 'unknown'),
            'data_completeness': self._assess_data_completeness(detected_patterns),
            'quality_indicators': self._get_quality_indicators(consensus_result, detected_patterns)
        }
    
    def _calculate_pattern_confidence(self, patterns: Dict) -> Dict:
        """Calculate confidence scores for detected patterns"""
        confidences = {}
        
        for pattern_type, pattern_list in patterns.items():
            if isinstance(pattern_list, list) and pattern_list:
                avg_confidence = sum(p.get('confidence', 0.5) for p in pattern_list) / len(pattern_list)
                confidences[pattern_type] = avg_confidence
        
        return confidences
    
    def _assess_data_completeness(self, patterns: Dict) -> Dict:
        """Assess how complete the extracted data is"""
        return {
            'has_key_values': len(patterns.get('key_value_pairs', [])) > 0,
            'has_tables': len(patterns.get('tables', [])) > 0,
            'has_forms': len(patterns.get('forms', [])) > 0,
            'has_structured_data': any(len(patterns.get(k, [])) > 0 
                                     for k in ['key_value_pairs', 'tables', 'forms']),
            'completeness_score': self._calculate_completeness_score(patterns)
        }
    
    def _calculate_completeness_score(self, patterns: Dict) -> float:
        """Calculate overall completeness score"""
        pattern_counts = {
            'key_value_pairs': len(patterns.get('key_value_pairs', [])),
            'tables': len(patterns.get('tables', [])),
            'forms': len(patterns.get('forms', [])),
            'sections': len(patterns.get('sections', []))
        }
        
        # Simple scoring based on presence of different pattern types
        score = 0.0
        max_score = 4.0
        
        for pattern_type, count in pattern_counts.items():
            if count > 0:
                score += 1.0
        
        return score / max_score
    
    def _get_quality_indicators(self, consensus_result: Dict, patterns: Dict) -> List[str]:
        """Get quality indicators for the extraction"""
        indicators = []
        
        # Agreement indicators
        agreement_score = consensus_result.get('agreement_score', 0.0)
        if agreement_score > 0.8:
            indicators.append('high_engine_agreement')
        elif agreement_score < 0.5:
            indicators.append('low_engine_agreement')
        
        # Pattern indicators
        pattern_count = sum(len(p) if isinstance(p, list) else 0 for p in patterns.values())
        if pattern_count > 20:
            indicators.append('rich_structure_detected')
        elif pattern_count < 5:
            indicators.append('sparse_structure')
        
        # Confidence indicators
        confidence = consensus_result.get('overall_confidence', 0.0)
        if confidence > 0.8:
            indicators.append('high_confidence')
        elif confidence < 0.5:
            indicators.append('low_confidence')
        
        return indicators
    
    def prepare_feedback_context(self, document_id: str, consensus_result: Dict, 
                                detected_patterns: Dict, classification: Dict) -> Dict:
        """Prepare context for user feedback"""
        return {
            'document_id': document_id,
            'feedback_ready': True,
            'correction_suggestions': self._generate_correction_suggestions(
                consensus_result, detected_patterns, classification
            ),
            'validation_prompts': self._generate_validation_prompts(detected_patterns),
            'classification_alternatives': classification.get('discovered_types', {}),
            'field_validation_needed': self._identify_fields_needing_validation(detected_patterns)
        }
    
    def _generate_correction_suggestions(self, consensus_result: Dict, patterns: Dict, 
                                       classification: Dict) -> List[Dict]:
        """Generate suggestions for potential corrections"""
        suggestions = []
        
        # Low confidence fields
        for kv_pair in patterns.get('key_value_pairs', []):
            if kv_pair.get('confidence', 1.0) < 0.6:
                suggestions.append({
                    'type': 'low_confidence_field',
                    'field': kv_pair.get('key'),
                    'extracted_value': kv_pair.get('value'),
                    'suggestion': 'Please verify this field value'
                })
        
        # Classification uncertainty
        discovered_types = classification.get('discovered_types', {})
        if len(discovered_types) > 1:
            suggestions.append({
                'type': 'document_classification',
                'suggestion': 'Multiple document types detected - please confirm',
                'alternatives': list(discovered_types.keys())
            })
        
        return suggestions
    
    def _generate_validation_prompts(self, patterns: Dict) -> List[Dict]:
        """Generate prompts for user validation"""
        prompts = []
        
        # Currency validation
        currency_fields = []
        for kv_pair in patterns.get('key_value_pairs', []):
            if '$' in kv_pair.get('value', '') or 'amount' in kv_pair.get('key', '').lower():
                currency_fields.append(kv_pair)
        
        if currency_fields:
            prompts.append({
                'type': 'currency_validation',
                'message': 'Please verify currency amounts are correct',
                'fields': currency_fields
            })
        
        return prompts
    
    def _identify_fields_needing_validation(self, patterns: Dict) -> List[str]:
        """Identify fields that need validation"""
        fields_needing_validation = []
        
        for kv_pair in patterns.get('key_value_pairs', []):
            field_name = kv_pair.get('key', '')
            confidence = kv_pair.get('confidence', 1.0)
            
            # High-importance fields with low confidence
            if confidence < 0.7 and any(keyword in field_name.lower() 
                                      for keyword in ['total', 'amount', 'ssn', 'account']):
                fields_needing_validation.append(field_name)
        
        return fields_needing_validation
    
    def identify_learning_opportunities(self, consensus_result: Dict, patterns: Dict, 
                                      classification: Dict) -> List[str]:
        """Identify opportunities for learning and improvement"""
        opportunities = []
        
        # New document type discovery
        discovered_types = classification.get('discovered_types', {})
        if discovered_types and not classification.get('learned_types'):
            opportunities.append(f"new_document_type_discovered: {list(discovered_types.keys())[0]}")
        
        # Low confidence extractions
        low_conf_count = sum(1 for kv in patterns.get('key_value_pairs', []) 
                           if kv.get('confidence', 1.0) < 0.6)
        if low_conf_count > 0:
            opportunities.append(f"low_confidence_fields: {low_conf_count}")
        
        # Pattern detection improvements
        if patterns.get('metadata', {}).get('total_patterns', 0) < 5:
            opportunities.append("sparse_pattern_detection")
        
        # Engine disagreement
        agreement_score = consensus_result.get('agreement_score', 1.0)
        if agreement_score < 0.5:
            opportunities.append("high_engine_disagreement")
        
        return opportunities
    
    def apply_user_feedback(self, document_id: str, feedback: Dict) -> Dict[str, Any]:
        """
        Learn from user corrections
        
        Args:
            document_id: ID of the document
            feedback: User feedback data
            
        Returns:
            Processing results and learning impact
        """
        if not self.learning_enabled:
            return {'error': 'Learning is disabled'}
        
        try:
            result = self.feedback_processor.process_correction(document_id, feedback)
            
            # Update engine weights if applicable
            if 'engine_name' in feedback:
                engine_name = feedback['engine_name']
                # Calculate performance impact
                performance_score = 0.3 if feedback.get('correction_type') == 'major_error' else 0.7
                self.consensus_merger.update_engine_weights(engine_name, performance_score)
            
            logger.info(f"Applied user feedback for document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply user feedback: {e}")
            return {'error': str(e)}
    
    def add_document_type_label(self, document_id: str, document_type_label: str) -> Dict[str, Any]:
        """
        User manually labels document type
        
        Args:
            document_id: ID of the document
            document_type_label: User-provided label
            
        Returns:
            Learning results
        """
        if not self.learning_enabled:
            return {'error': 'Learning is disabled'}
        
        try:
            result = self.feedback_processor.apply_document_type_label(document_id, document_type_label)
            logger.info(f"Applied document type label '{document_type_label}' for document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply document type label: {e}")
            return {'error': str(e)}
    
    def update_extraction_rules(self, field_name: str, new_pattern: str, document_type: str) -> str:
        """
        User modifies how specific fields are extracted
        
        Args:
            field_name: Name of the field
            new_pattern: New extraction pattern
            document_type: Document type this applies to
            
        Returns:
            Rule ID
        """
        if not self.learning_enabled:
            raise ValueError('Learning is disabled')
        
        try:
            rule_id = self.feedback_processor.update_extraction_rules(
                field_name, new_pattern, document_type
            )
            logger.info(f"Updated extraction rules for field '{field_name}'")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to update extraction rules: {e}")
            raise
    
    def generate_document_id(self) -> str:
        """Generate unique document ID"""
        return str(uuid.uuid4())
    
    def _load_learned_patterns(self):
        """Load previously learned patterns"""
        if not self.learning_enabled:
            return
        
        try:
            # Load document types
            # This would load saved classifier state
            logger.debug("Loading learned patterns from database")
            
            # Load engine performance history
            performance_stats = self.learning_db.get_engine_performance_stats()
            
            # Update engine weights based on historical performance
            for engine_name, stats in performance_stats.items():
                if stats:
                    avg_accuracy = sum(s.get('avg_accuracy', 0.8) for s in stats.values()) / len(stats)
                    self.consensus_merger.update_engine_weights(engine_name, avg_accuracy)
            
        except Exception as e:
            logger.warning(f"Failed to load learned patterns: {e}")
    
    def _store_processing_result(self, document_id: str, result: Dict, file_path: str):
        """Store processing result for learning"""
        try:
            self.learning_db.store_document_example(
                document_id=document_id,
                file_path=file_path,
                patterns=result.get('detected_patterns', {}),
                text_content=result.get('consensus', {}).get('consensus_text', ''),
                features=result.get('extraction_metadata', {}),
                document_type=result.get('document_classification', {}).get('best_overall_guess')
            )
        except Exception as e:
            logger.warning(f"Failed to store processing result: {e}")
    
    def _create_error_result(self, document_id: str, error_message: str) -> Dict:
        """Create error result"""
        return {
            'document_id': document_id,
            'status': 'error',
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'raw_ocr': {},
            'consensus': {},
            'detected_patterns': {},
            'document_classification': {},
            'structured_data': {},
            'confidence_metrics': {},
            'extraction_metadata': {},
            'feedback_context': {'feedback_ready': False},
            'learning_opportunities': []
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        if not self.learning_enabled:
            return {'learning_enabled': False}
        
        try:
            progress = self.learning_db.get_learning_progress()
            engine_performance = self.consensus_merger.get_engine_performance_summary()
            
            return {
                'learning_enabled': True,
                'learning_progress': progress,
                'engine_performance': engine_performance,
                'total_corrections': progress.get('total_corrections', 0),
                'learning_active': progress.get('learning_active', False)
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning summary: {e}")
            return {'error': str(e)}
    
    def cleanup_resources(self):
        """Cleanup resources"""
        if self.learning_db:
            self.learning_db.close()
    
    def __del__(self):
        """Ensure resources are cleaned up"""
        try:
            self.cleanup_resources()
        except:
            pass