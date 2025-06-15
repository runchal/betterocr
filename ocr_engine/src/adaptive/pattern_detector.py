"""
Pattern Detector
Detects patterns in OCR results without hard-coded document types
"""

import logging
from typing import Dict, List, Any, Tuple
import re
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detects various patterns in OCR results"""
    
    def __init__(self):
        # Import generic patterns
        from ..patterns.generic_patterns import GenericPatternDetector
        self.generic_detector = GenericPatternDetector()
        
    def detect_patterns(self, ocr_results: Dict) -> Dict[str, Any]:
        """
        Detect all patterns in OCR results
        
        Args:
            ocr_results: Dictionary containing OCR output with text, words, lines, etc.
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'key_value_pairs': [],
            'tables': [],
            'lists': [],
            'sections': [],
            'forms': [],
            'numbers': {},
            'dates': [],
            'addresses': [],
            'metadata': {}
        }
        
        # Get text from results
        text = self._extract_text(ocr_results)
        
        # Use generic pattern detector
        generic_patterns = self.generic_detector.detect_all_patterns(text)
        
        # Process and categorize patterns
        patterns['key_value_pairs'] = generic_patterns.get('key_value_pairs', [])
        patterns['addresses'] = generic_patterns.get('addresses', [])
        patterns['lists'] = generic_patterns.get('lists', [])
        patterns['sections'] = generic_patterns.get('sections', [])
        
        # Extract numeric patterns
        patterns['numbers'] = self._extract_numeric_patterns(generic_patterns)
        
        # Extract date patterns
        patterns['dates'] = self._extract_date_patterns(generic_patterns)
        
        # Detect tables from layout
        if 'lines' in ocr_results:
            patterns['tables'] = self._detect_table_structures(ocr_results['lines'])
        
        # Detect form fields
        patterns['forms'] = self._detect_form_fields(ocr_results, patterns['key_value_pairs'])
        
        # Calculate pattern metadata
        patterns['metadata'] = self._calculate_pattern_metadata(patterns, text)
        
        return patterns
    
    def _extract_text(self, ocr_results: Dict) -> str:
        """Extract text from OCR results"""
        if isinstance(ocr_results, dict):
            if 'text' in ocr_results:
                return ocr_results['text']
            elif 'full_text' in ocr_results:
                return ocr_results['full_text']
            elif 'consensus_text' in ocr_results:
                return ocr_results['consensus_text']
        return str(ocr_results)
    
    def _extract_numeric_patterns(self, generic_patterns: Dict) -> Dict[str, List]:
        """Extract and categorize numeric patterns"""
        numbers = {
            'currencies': [],
            'percentages': [],
            'phone_numbers': [],
            'ssn': [],
            'account_numbers': [],
            'reference_numbers': [],
            'form_numbers': [],
            'other': []
        }
        
        # Map generic patterns to specific categories
        pattern_mapping = {
            'currency': 'currencies',
            'percentage': 'percentages',
            'phone_us': 'phone_numbers',
            'phone_intl': 'phone_numbers',
            'ssn': 'ssn',
            'account': 'account_numbers',
            'reference': 'reference_numbers',
            'form_number': 'form_numbers'
        }
        
        for pattern_type, matches in generic_patterns.items():
            if pattern_type in pattern_mapping:
                category = pattern_mapping[pattern_type]
                numbers[category].extend(matches)
            elif pattern_type in ['integer', 'decimal']:
                # Categorize other numbers based on context
                for match in matches:
                    categorized = self._categorize_number(match)
                    numbers[categorized].append(match)
        
        return numbers
    
    def _categorize_number(self, number_match: Dict) -> str:
        """Categorize a number based on context"""
        # This would look at surrounding text to determine number type
        # For now, return 'other'
        return 'other'
    
    def _extract_date_patterns(self, generic_patterns: Dict) -> List[Dict]:
        """Extract and normalize date patterns"""
        dates = []
        
        date_patterns = ['date_mdy', 'date_dmy', 'date_ymd', 'date_text']
        
        for pattern_type in date_patterns:
            if pattern_type in generic_patterns:
                for match in generic_patterns[pattern_type]:
                    date_info = match.copy()
                    date_info['format'] = pattern_type
                    # Could add date parsing here
                    dates.append(date_info)
        
        return dates
    
    def _detect_table_structures(self, lines: List) -> List[Dict]:
        """Detect table structures from line information"""
        tables = []
        
        if not lines:
            return tables
        
        # Group lines by vertical position
        line_groups = self._group_lines_vertically(lines)
        
        # Look for patterns that suggest tables
        for group in line_groups:
            if self._is_table_like(group):
                table = self._extract_table_structure(group)
                if table:
                    tables.append(table)
        
        return tables
    
    def _group_lines_vertically(self, lines: List) -> List[List]:
        """Group lines that are vertically close together"""
        if not lines:
            return []
        
        groups = []
        current_group = []
        y_threshold = 20  # Pixels
        
        # Sort lines by Y position
        sorted_lines = sorted(lines, key=lambda x: x[2][1] if len(x) > 2 else 0)
        
        for line in sorted_lines:
            if not current_group:
                current_group.append(line)
            else:
                # Check vertical distance
                last_y = current_group[-1][2][1] + current_group[-1][2][3]
                curr_y = line[2][1]
                
                if curr_y - last_y <= y_threshold:
                    current_group.append(line)
                else:
                    if len(current_group) > 2:  # Potential table
                        groups.append(current_group)
                    current_group = [line]
        
        if len(current_group) > 2:
            groups.append(current_group)
        
        return groups
    
    def _is_table_like(self, line_group: List) -> bool:
        """Check if a group of lines looks like a table"""
        if len(line_group) < 3:
            return False
        
        # Check for alignment patterns
        alignments = []
        for line in line_group:
            if len(line) > 2:
                x_pos = line[2][0]
                alignments.append(x_pos)
        
        # If many lines start at similar X positions, might be a table
        alignment_counts = Counter(x // 10 * 10 for x in alignments)  # Group by 10px
        max_aligned = max(alignment_counts.values()) if alignment_counts else 0
        
        return max_aligned >= len(line_group) * 0.5
    
    def _extract_table_structure(self, line_group: List) -> Dict:
        """Extract table structure from grouped lines"""
        table = {
            'type': 'detected_table',
            'rows': [],
            'columns': [],
            'cells': [],
            'confidence': 0.6,
            'bbox': self._calculate_group_bbox(line_group)
        }
        
        # Simple approach: each line is a row
        for i, line in enumerate(line_group):
            row = {
                'index': i,
                'text': line[0] if line else '',
                'cells': self._split_into_cells(line)
            }
            table['rows'].append(row)
        
        # Detect columns based on cell positions
        table['columns'] = self._detect_columns(table['rows'])
        
        return table
    
    def _split_into_cells(self, line: Tuple) -> List[str]:
        """Split a line into table cells"""
        if not line or not line[0]:
            return []
        
        text = line[0]
        
        # Try different delimiters
        delimiters = ['\t', '  ', '|', ',']
        
        for delimiter in delimiters:
            if delimiter in text:
                cells = [cell.strip() for cell in text.split(delimiter)]
                if len(cells) > 1:
                    return cells
        
        # If no delimiter found, treat as single cell
        return [text]
    
    def _detect_columns(self, rows: List[Dict]) -> List[Dict]:
        """Detect column structure from rows"""
        if not rows:
            return []
        
        # Find the row with most cells (likely header)
        max_cells = max(len(row['cells']) for row in rows)
        
        columns = []
        for i in range(max_cells):
            column = {
                'index': i,
                'values': [],
                'type': 'unknown'
            }
            
            # Collect values from each row
            for row in rows:
                if i < len(row['cells']):
                    column['values'].append(row['cells'][i])
            
            # Detect column type
            column['type'] = self._detect_column_type(column['values'])
            columns.append(column)
        
        return columns
    
    def _detect_column_type(self, values: List[str]) -> str:
        """Detect the type of data in a column"""
        if not values:
            return 'unknown'
        
        # Count different types
        numeric_count = 0
        currency_count = 0
        date_count = 0
        
        for value in values:
            if re.match(r'^\$?\d+(?:,\d{3})*(?:\.\d{2})?$', value):
                currency_count += 1
            elif re.match(r'^\d+(?:\.\d+)?$', value):
                numeric_count += 1
            elif re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value):
                date_count += 1
        
        # Determine predominant type
        total = len(values)
        if currency_count > total * 0.5:
            return 'currency'
        elif numeric_count > total * 0.5:
            return 'numeric'
        elif date_count > total * 0.3:
            return 'date'
        else:
            return 'text'
    
    def _calculate_group_bbox(self, line_group: List) -> Tuple[int, int, int, int]:
        """Calculate bounding box for a group of lines"""
        if not line_group:
            return (0, 0, 0, 0)
        
        bboxes = [line[2] for line in line_group if len(line) > 2]
        if not bboxes:
            return (0, 0, 0, 0)
        
        x_min = min(bbox[0] for bbox in bboxes)
        y_min = min(bbox[1] for bbox in bboxes)
        x_max = max(bbox[0] + bbox[2] for bbox in bboxes)
        y_max = max(bbox[1] + bbox[3] for bbox in bboxes)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _detect_form_fields(self, ocr_results: Dict, key_value_pairs: List) -> List[Dict]:
        """Detect form field structures"""
        form_fields = []
        
        # Look for common form field patterns
        field_indicators = [
            'name', 'address', 'phone', 'email', 'date', 'signature',
            'amount', 'account', 'ssn', 'id', 'number', 'code'
        ]
        
        for kv_pair in key_value_pairs:
            key_lower = kv_pair['key'].lower()
            
            # Check if this looks like a form field
            is_form_field = any(indicator in key_lower for indicator in field_indicators)
            
            # Check for empty or placeholder values
            value = kv_pair['value']
            has_placeholder = bool(re.match(r'^[_\-\.]+$', value.strip()))
            is_empty = len(value.strip()) < 3
            
            if is_form_field or has_placeholder or is_empty:
                field = {
                    'field_name': kv_pair['key'],
                    'field_value': value if not (has_placeholder or is_empty) else '',
                    'field_type': self._determine_field_type(kv_pair['key']),
                    'has_value': not (has_placeholder or is_empty),
                    'position': (kv_pair['start'], kv_pair['end']),
                    'confidence': 0.8 if is_form_field else 0.6
                }
                form_fields.append(field)
        
        # Look for checkbox patterns
        checkbox_fields = self._detect_checkboxes(ocr_results)
        form_fields.extend(checkbox_fields)
        
        return form_fields
    
    def _determine_field_type(self, field_name: str) -> str:
        """Determine the type of a form field based on its name"""
        field_lower = field_name.lower()
        
        type_mapping = {
            'name': ['name', 'fname', 'lname', 'first', 'last'],
            'address': ['address', 'addr', 'street', 'city', 'state', 'zip'],
            'phone': ['phone', 'tel', 'mobile', 'cell'],
            'email': ['email', 'e-mail'],
            'date': ['date', 'dob', 'birth', 'expire'],
            'currency': ['amount', 'total', 'price', 'cost', 'fee'],
            'ssn': ['ssn', 'social'],
            'signature': ['signature', 'sign'],
            'checkbox': ['check', 'select', 'choose']
        }
        
        for field_type, keywords in type_mapping.items():
            if any(keyword in field_lower for keyword in keywords):
                return field_type
        
        return 'text'
    
    def _detect_checkboxes(self, ocr_results: Dict) -> List[Dict]:
        """Detect checkbox patterns in OCR results"""
        checkboxes = []
        
        # Look for checkbox indicators
        checkbox_patterns = [
            r'\[\s*[xX✓]\s*\]',  # [x] or [✓]
            r'\(\s*[xX✓]\s*\)',  # (x) or (✓)
            r'☐|☑|□|■|▢|▣',     # Unicode checkboxes
            r'YES\s*/\s*NO',      # Yes/No options
        ]
        
        text = self._extract_text(ocr_results)
        
        for pattern in checkbox_patterns:
            for match in re.finditer(pattern, text):
                checkbox = {
                    'field_type': 'checkbox',
                    'field_name': 'checkbox',
                    'field_value': 'checked' if any(c in match.group() for c in 'xX✓☑■▣') else 'unchecked',
                    'position': (match.start(), match.end()),
                    'confidence': 0.7
                }
                checkboxes.append(checkbox)
        
        return checkboxes
    
    def _calculate_pattern_metadata(self, patterns: Dict, text: str) -> Dict:
        """Calculate metadata about detected patterns"""
        metadata = {
            'total_patterns': 0,
            'pattern_density': {},
            'dominant_patterns': [],
            'text_characteristics': {}
        }
        
        # Count total patterns
        for pattern_type, pattern_list in patterns.items():
            if isinstance(pattern_list, list):
                metadata['total_patterns'] += len(pattern_list)
                metadata['pattern_density'][pattern_type] = len(pattern_list)
            elif isinstance(pattern_list, dict):
                count = sum(len(v) if isinstance(v, list) else 0 for v in pattern_list.values())
                metadata['total_patterns'] += count
                metadata['pattern_density'][pattern_type] = count
        
        # Find dominant patterns
        if metadata['pattern_density']:
            sorted_patterns = sorted(
                metadata['pattern_density'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            metadata['dominant_patterns'] = [p[0] for p in sorted_patterns[:3]]
        
        # Text characteristics
        words = text.split()
        metadata['text_characteristics'] = {
            'total_words': len(words),
            'total_lines': len(text.split('\n')),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'numeric_ratio': sum(1 for w in words if any(c.isdigit() for c in w)) / len(words) if words else 0
        }
        
        return metadata