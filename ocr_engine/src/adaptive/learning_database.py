"""
Learning Database
Persistent storage for learned patterns, user feedback, and performance metrics
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningDatabase:
    """SQLite-based learning database for OCR improvements"""
    
    def __init__(self, db_path: str = "data/ocr_learning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            self._create_tables()
            logger.info(f"Learning database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create all required tables"""
        cursor = self.connection.cursor()
        
        # Document examples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_examples (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                document_type TEXT,
                user_label TEXT,
                patterns_json TEXT,
                text_content TEXT,
                features_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Extraction patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_patterns (
                id TEXT PRIMARY KEY,
                pattern_name TEXT,
                pattern_regex TEXT,
                pattern_type TEXT,
                document_type TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Field extraction rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS field_rules (
                id TEXT PRIMARY KEY,
                field_name TEXT,
                extraction_rule TEXT,
                document_type TEXT,
                validation_pattern TEXT,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Engine performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engine_performance (
                id TEXT PRIMARY KEY,
                engine_name TEXT,
                document_type TEXT,
                accuracy_score REAL,
                confidence_score REAL,
                processing_time REAL,
                document_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User corrections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_corrections (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                field_name TEXT,
                extracted_value TEXT,
                correct_value TEXT,
                correction_type TEXT,
                engine_name TEXT,
                bounding_box TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Document classifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_classifications (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                auto_classification TEXT,
                user_classification TEXT,
                confidence_score REAL,
                classification_features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Learning metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id TEXT PRIMARY KEY,
                metric_type TEXT,
                metric_value REAL,
                metadata_json TEXT,
                date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        self._create_indexes()
        
        self.connection.commit()
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        cursor = self.connection.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_document_type ON document_examples(document_type)",
            "CREATE INDEX IF NOT EXISTS idx_user_label ON document_examples(user_label)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_type ON extraction_patterns(pattern_type)",
            "CREATE INDEX IF NOT EXISTS idx_field_name ON field_rules(field_name)",
            "CREATE INDEX IF NOT EXISTS idx_engine_name ON engine_performance(engine_name)",
            "CREATE INDEX IF NOT EXISTS idx_document_id ON user_corrections(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_created_at ON user_corrections(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_metric_type ON learning_metrics(metric_type)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def store_document_example(self, document_id: str, file_path: str, 
                             patterns: Dict, text_content: str, 
                             features: Dict, user_label: Optional[str] = None,
                             document_type: Optional[str] = None) -> str:
        """Store a document example for learning"""
        try:
            cursor = self.connection.cursor()
            
            example_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO document_examples 
                (id, file_path, document_type, user_label, patterns_json, 
                 text_content, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                example_id,
                file_path,
                document_type,
                user_label,
                json.dumps(patterns),
                text_content,
                json.dumps(features)
            ))
            
            self.connection.commit()
            logger.debug(f"Stored document example: {example_id}")
            
            return example_id
            
        except Exception as e:
            logger.error(f"Failed to store document example: {e}")
            self.connection.rollback()
            raise
    
    def store_extraction_pattern(self, pattern_name: str, pattern_regex: str,
                               pattern_type: str, document_type: str,
                               confidence_score: float = 0.0) -> str:
        """Store an extraction pattern"""
        try:
            cursor = self.connection.cursor()
            
            pattern_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO extraction_patterns
                (id, pattern_name, pattern_regex, pattern_type, document_type, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                pattern_name,
                pattern_regex,
                pattern_type,
                document_type,
                confidence_score
            ))
            
            self.connection.commit()
            logger.debug(f"Stored extraction pattern: {pattern_name}")
            
            return pattern_id
            
        except Exception as e:
            logger.error(f"Failed to store extraction pattern: {e}")
            self.connection.rollback()
            raise
    
    def store_field_rule(self, field_name: str, extraction_rule: str,
                        document_type: str, validation_pattern: Optional[str] = None) -> str:
        """Store a field extraction rule"""
        try:
            cursor = self.connection.cursor()
            
            rule_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO field_rules
                (id, field_name, extraction_rule, document_type, validation_pattern)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                rule_id,
                field_name,
                extraction_rule,
                document_type,
                validation_pattern
            ))
            
            self.connection.commit()
            logger.debug(f"Stored field rule: {field_name}")
            
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to store field rule: {e}")
            self.connection.rollback()
            raise
    
    def store_user_correction(self, document_id: str, field_name: str,
                            extracted_value: str, correct_value: str,
                            correction_type: str, engine_name: str,
                            bounding_box: Optional[Tuple] = None) -> str:
        """Store user correction for learning"""
        try:
            cursor = self.connection.cursor()
            
            correction_id = str(uuid.uuid4())
            bbox_json = json.dumps(bounding_box) if bounding_box else None
            
            cursor.execute('''
                INSERT INTO user_corrections
                (id, document_id, field_name, extracted_value, correct_value,
                 correction_type, engine_name, bounding_box)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                correction_id,
                document_id,
                field_name,
                extracted_value,
                correct_value,
                correction_type,
                engine_name,
                bbox_json
            ))
            
            self.connection.commit()
            logger.info(f"Stored user correction for field: {field_name}")
            
            # Update related patterns and rules
            self._update_patterns_from_correction(field_name, correct_value, document_id)
            
            return correction_id
            
        except Exception as e:
            logger.error(f"Failed to store user correction: {e}")
            self.connection.rollback()
            raise
    
    def store_engine_performance(self, engine_name: str, document_type: str,
                               accuracy_score: float, confidence_score: float,
                               processing_time: float, document_id: str) -> str:
        """Store engine performance metrics"""
        try:
            cursor = self.connection.cursor()
            
            perf_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO engine_performance
                (id, engine_name, document_type, accuracy_score, confidence_score,
                 processing_time, document_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                perf_id,
                engine_name,
                document_type,
                accuracy_score,
                confidence_score,
                processing_time,
                document_id
            ))
            
            self.connection.commit()
            
            return perf_id
            
        except Exception as e:
            logger.error(f"Failed to store engine performance: {e}")
            self.connection.rollback()
            raise
    
    def get_patterns_for_type(self, document_type: str) -> List[Dict]:
        """Get extraction patterns for a document type"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT * FROM extraction_patterns 
                WHERE document_type = ? 
                ORDER BY confidence_score DESC, success_count DESC
            ''', (document_type,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append(dict(row))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get patterns for type {document_type}: {e}")
            return []
    
    def get_field_rules(self, field_name: Optional[str] = None, 
                       document_type: Optional[str] = None) -> List[Dict]:
        """Get field extraction rules"""
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM field_rules WHERE 1=1"
            params = []
            
            if field_name:
                query += " AND field_name = ?"
                params.append(field_name)
            
            if document_type:
                query += " AND document_type = ?"
                params.append(document_type)
            
            query += " ORDER BY success_rate DESC, usage_count DESC"
            
            cursor.execute(query, params)
            
            rules = []
            for row in cursor.fetchall():
                rules.append(dict(row))
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to get field rules: {e}")
            return []
    
    def get_engine_performance_stats(self, engine_name: Optional[str] = None,
                                   days_back: int = 30) -> Dict[str, Any]:
        """Get engine performance statistics"""
        try:
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = '''
                SELECT engine_name, document_type,
                       AVG(accuracy_score) as avg_accuracy,
                       AVG(confidence_score) as avg_confidence,
                       AVG(processing_time) as avg_processing_time,
                       COUNT(*) as sample_count
                FROM engine_performance 
                WHERE created_at > ?
            '''
            params = [cutoff_date]
            
            if engine_name:
                query += " AND engine_name = ?"
                params.append(engine_name)
            
            query += " GROUP BY engine_name, document_type"
            
            cursor.execute(query, params)
            
            stats = {}
            for row in cursor.fetchall():
                engine = row['engine_name']
                if engine not in stats:
                    stats[engine] = {}
                
                stats[engine][row['document_type']] = {
                    'avg_accuracy': row['avg_accuracy'],
                    'avg_confidence': row['avg_confidence'],
                    'avg_processing_time': row['avg_processing_time'],
                    'sample_count': row['sample_count']
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get engine performance stats: {e}")
            return {}
    
    def get_user_corrections(self, document_id: Optional[str] = None,
                           field_name: Optional[str] = None,
                           days_back: int = 30) -> List[Dict]:
        """Get user corrections"""
        try:
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = "SELECT * FROM user_corrections WHERE created_at > ?"
            params = [cutoff_date]
            
            if document_id:
                query += " AND document_id = ?"
                params.append(document_id)
            
            if field_name:
                query += " AND field_name = ?"
                params.append(field_name)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            
            corrections = []
            for row in cursor.fetchall():
                correction = dict(row)
                # Parse bounding box if present
                if correction['bounding_box']:
                    correction['bounding_box'] = json.loads(correction['bounding_box'])
                corrections.append(correction)
            
            return corrections
            
        except Exception as e:
            logger.error(f"Failed to get user corrections: {e}")
            return []
    
    def update_pattern_success_rate(self, pattern_id: str, success: bool):
        """Update pattern success rate based on usage"""
        try:
            cursor = self.connection.cursor()
            
            if success:
                cursor.execute('''
                    UPDATE extraction_patterns 
                    SET success_count = success_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (pattern_id,))
            else:
                cursor.execute('''
                    UPDATE extraction_patterns 
                    SET failure_count = failure_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (pattern_id,))
            
            # Recalculate confidence score
            cursor.execute('''
                UPDATE extraction_patterns 
                SET confidence_score = CAST(success_count AS REAL) / (success_count + failure_count)
                WHERE id = ? AND (success_count + failure_count) > 0
            ''', (pattern_id,))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to update pattern success rate: {e}")
            self.connection.rollback()
    
    def update_field_rule_performance(self, rule_id: str, success: bool):
        """Update field rule performance"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                UPDATE field_rules 
                SET usage_count = usage_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (rule_id,))
            
            # Update success rate (simplified)
            if success:
                cursor.execute('''
                    UPDATE field_rules 
                    SET success_rate = (success_rate * (usage_count - 1) + 1.0) / usage_count
                    WHERE id = ?
                ''', (rule_id,))
            else:
                cursor.execute('''
                    UPDATE field_rules 
                    SET success_rate = (success_rate * (usage_count - 1)) / usage_count
                    WHERE id = ?
                ''', (rule_id,))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to update field rule performance: {e}")
            self.connection.rollback()
    
    def _update_patterns_from_correction(self, field_name: str, correct_value: str, document_id: str):
        """Update patterns based on user correction"""
        # This would analyze the correction and update relevant patterns
        # For now, just log the correction for manual analysis
        logger.info(f"Processing correction for field {field_name}: '{correct_value}'")
        
        # Could implement automatic pattern learning here
        # For example, if user corrects "$1.234.56" to "$1,234.56",
        # learn that periods in currency should be commas
    
    def get_document_examples_by_type(self, document_type: str) -> List[Dict]:
        """Get document examples for a specific type"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT * FROM document_examples 
                WHERE document_type = ? OR user_label = ?
                ORDER BY created_at DESC
            ''', (document_type, document_type))
            
            examples = []
            for row in cursor.fetchall():
                example = dict(row)
                # Parse JSON fields
                if example['patterns_json']:
                    example['patterns'] = json.loads(example['patterns_json'])
                if example['features_json']:
                    example['features'] = json.loads(example['features_json'])
                examples.append(example)
            
            return examples
            
        except Exception as e:
            logger.error(f"Failed to get document examples: {e}")
            return []
    
    def store_learning_metric(self, metric_type: str, metric_value: float, 
                            metadata: Optional[Dict] = None):
        """Store a learning metric"""
        try:
            cursor = self.connection.cursor()
            
            metric_id = str(uuid.uuid4())
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO learning_metrics (id, metric_type, metric_value, metadata_json)
                VALUES (?, ?, ?, ?)
            ''', (metric_id, metric_type, metric_value, metadata_json))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to store learning metric: {e}")
            self.connection.rollback()
    
    def get_learning_progress(self, days_back: int = 30) -> Dict[str, Any]:
        """Get learning progress metrics"""
        try:
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Count corrections over time
            cursor.execute('''
                SELECT DATE(created_at) as date, COUNT(*) as correction_count
                FROM user_corrections 
                WHERE created_at > ?
                GROUP BY DATE(created_at)
                ORDER BY date
            ''', (cutoff_date,))
            
            corrections_by_date = {row['date']: row['correction_count'] for row in cursor.fetchall()}
            
            # Count new patterns learned
            cursor.execute('''
                SELECT COUNT(*) as pattern_count
                FROM extraction_patterns 
                WHERE created_at > ?
            ''', (cutoff_date,))
            
            new_patterns = cursor.fetchone()['pattern_count']
            
            # Average engine performance trend
            cursor.execute('''
                SELECT engine_name, AVG(accuracy_score) as avg_accuracy
                FROM engine_performance 
                WHERE created_at > ?
                GROUP BY engine_name
            ''', (cutoff_date,))
            
            engine_performance = {row['engine_name']: row['avg_accuracy'] for row in cursor.fetchall()}
            
            return {
                'corrections_by_date': corrections_by_date,
                'new_patterns_learned': new_patterns,
                'engine_performance': engine_performance,
                'total_corrections': sum(corrections_by_date.values()),
                'learning_active': sum(corrections_by_date.values()) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning progress: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to prevent database bloat"""
        try:
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old engine performance records
            cursor.execute('''
                DELETE FROM engine_performance 
                WHERE created_at < ?
            ''', (cutoff_date,))
            
            # Clean up old learning metrics
            cursor.execute('''
                DELETE FROM learning_metrics 
                WHERE date_recorded < ?
            ''', (cutoff_date,))
            
            self.connection.commit()
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            self.connection.rollback()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __del__(self):
        """Ensure connection is closed"""
        self.close()