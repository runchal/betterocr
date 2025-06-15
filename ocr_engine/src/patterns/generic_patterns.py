"""
Generic Pattern Detection
Universal patterns that work across document types
"""

import re
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GenericPatternDetector:
    """Detects common patterns in OCR text"""
    
    # Common regex patterns
    PATTERNS = {
        # Currency
        'currency': r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\.\d{2}',
        
        # Dates
        'date_mdy': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        'date_dmy': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        'date_ymd': r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        'date_text': r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
        
        # Phone numbers
        'phone_us': r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'phone_intl': r'\+\d{1,3}[-.\s]?\d{1,14}',
        
        # Email
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        
        # SSN
        'ssn': r'\d{3}-\d{2}-\d{4}|\d{9}',
        
        # ZIP codes
        'zip_us': r'\d{5}(?:-\d{4})?',
        
        # Percentages
        'percentage': r'\d+(?:\.\d+)?%',
        
        # Time
        'time': r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?',
        
        # URLs
        'url': r'https?://[^\s]+|www\.[^\s]+',
        
        # Numbers
        'integer': r'\b\d+\b',
        'decimal': r'\b\d+\.\d+\b',
        
        # Form numbers
        'form_number': r'Form\s+\w+-?\w*|F\d{4}\w*',
        
        # Account numbers
        'account': r'(?:Account|Acct|A/C)[\s#:]*[\w-]+',
        
        # Reference numbers
        'reference': r'(?:Ref|Reference|REF)[\s#:]*[\w-]+',
    }
    
    def detect_all_patterns(self, text: str) -> Dict[str, List[Dict]]:
        """
        Detect all patterns in text
        
        Returns:
            Dictionary mapping pattern types to list of matches
        """
        results = {}
        
        for pattern_name, pattern_regex in self.PATTERNS.items():
            matches = self._find_pattern_matches(text, pattern_regex, pattern_name)
            if matches:
                results[pattern_name] = matches
        
        # Detect additional complex patterns
        results.update({
            'key_value_pairs': self.detect_key_value_pairs(text),
            'addresses': self.detect_addresses(text),
            'lists': self.detect_lists(text),
            'sections': self.detect_sections(text)
        })
        
        return results
    
    def _find_pattern_matches(self, text: str, pattern: str, pattern_type: str) -> List[Dict]:
        """Find all matches of a pattern in text"""
        matches = []
        
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'type': pattern_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9  # High confidence for exact regex matches
                })
        except Exception as e:
            logger.warning(f"Pattern matching error for {pattern_type}: {e}")
        
        return matches
    
    def detect_key_value_pairs(self, text: str) -> List[Dict]:
        """Detect key-value pairs in various formats"""
        pairs = []
        
        # Patterns for key-value detection
        kv_patterns = [
            # "Key: Value" format
            (r'([A-Za-z][A-Za-z\s]{2,30}):\s*([^\n:]{1,100})', ':'),
            # "Key = Value" format
            (r'([A-Za-z][A-Za-z\s]{2,30})\s*=\s*([^\n=]{1,100})', '='),
            # "Key - Value" format with longer dash
            (r'([A-Za-z][A-Za-z\s]{2,30})\s*[-–—]\s*([^\n\-–—]{1,100})', '-'),
            # Tabular format (key and value separated by multiple spaces)
            (r'([A-Za-z][A-Za-z\s]{2,30})\s{3,}([^\n]{1,100})', 'tab'),
        ]
        
        for pattern, separator in kv_patterns:
            for match in re.finditer(pattern, text):
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Filter out false positives
                if self._is_valid_key_value(key, value):
                    pairs.append({
                        'key': key,
                        'value': value,
                        'separator': separator,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    })
        
        # Deduplicate overlapping matches
        pairs = self._deduplicate_matches(pairs)
        
        return pairs
    
    def _is_valid_key_value(self, key: str, value: str) -> bool:
        """Check if a key-value pair is valid"""
        # Key should not be too short or too long
        if len(key) < 3 or len(key) > 50:
            return False
        
        # Value should not be empty
        if not value or len(value.strip()) < 1:
            return False
        
        # Key should contain at least one letter
        if not any(c.isalpha() for c in key):
            return False
        
        # Avoid matching sentences
        if key.count(' ') > 5:
            return False
        
        return True
    
    def detect_addresses(self, text: str) -> List[Dict]:
        """Detect address patterns"""
        addresses = []
        
        # US address pattern
        us_address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Circle|Cir|Way)\b[,\s]*(?:[A-Za-z\s]+[,\s]+)?(?:[A-Z]{2}\s+\d{5}(?:-\d{4})?)?'
        
        for match in re.finditer(us_address_pattern, text, re.IGNORECASE):
            addresses.append({
                'type': 'us_address',
                'value': match.group().strip(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7
            })
        
        # PO Box pattern
        po_box_pattern = r'(?:P\.?O\.?\s*Box|Post\s*Office\s*Box)\s*\d+'
        
        for match in re.finditer(po_box_pattern, text, re.IGNORECASE):
            addresses.append({
                'type': 'po_box',
                'value': match.group().strip(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        return addresses
    
    def detect_lists(self, text: str) -> List[Dict]:
        """Detect list structures"""
        lists = []
        lines = text.split('\n')
        
        # Numbered lists
        numbered_pattern = r'^\s*\d+[\.\)]\s+(.+)'
        current_list = []
        
        for i, line in enumerate(lines):
            match = re.match(numbered_pattern, line)
            if match:
                current_list.append({
                    'item': match.group(1).strip(),
                    'number': int(re.findall(r'\d+', line)[0]),
                    'line': i
                })
            elif current_list:
                # End of list
                if len(current_list) > 1:
                    lists.append({
                        'type': 'numbered',
                        'items': current_list,
                        'start_line': current_list[0]['line'],
                        'end_line': current_list[-1]['line'],
                        'confidence': 0.8
                    })
                current_list = []
        
        # Bulleted lists
        bullet_pattern = r'^\s*[•·▪▫◦‣⁃]\s+(.+)'
        current_list = []
        
        for i, line in enumerate(lines):
            match = re.match(bullet_pattern, line)
            if match:
                current_list.append({
                    'item': match.group(1).strip(),
                    'line': i
                })
            elif current_list:
                if len(current_list) > 1:
                    lists.append({
                        'type': 'bulleted',
                        'items': current_list,
                        'start_line': current_list[0]['line'],
                        'end_line': current_list[-1]['line'],
                        'confidence': 0.8
                    })
                current_list = []
        
        return lists
    
    def detect_sections(self, text: str) -> List[Dict]:
        """Detect section headers and document structure"""
        sections = []
        lines = text.split('\n')
        
        # Common section patterns
        section_patterns = [
            # All caps headers
            (r'^[A-Z][A-Z\s]{2,50}$', 'all_caps'),
            # Numbered sections
            (r'^\s*\d+\.?\s+[A-Z][A-Za-z\s]{2,50}$', 'numbered'),
            # Section with colon
            (r'^[A-Z][A-Za-z\s]{2,50}:$', 'colon_end'),
            # Underlined sections (detected by next line)
            (r'^[A-Za-z\s]{3,50}$', 'potential_underlined'),
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern, section_type in section_patterns:
                if re.match(pattern, line):
                    # Check if it's underlined
                    is_underlined = False
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if re.match(r'^[-=_]{3,}$', next_line):
                            is_underlined = True
                    
                    sections.append({
                        'text': line,
                        'type': section_type,
                        'line_number': i,
                        'is_underlined': is_underlined,
                        'confidence': 0.7 if not is_underlined else 0.9
                    })
                    break
        
        return sections
    
    def _deduplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """Remove overlapping matches, keeping the most confident ones"""
        if not matches:
            return matches
        
        # Sort by start position and confidence
        sorted_matches = sorted(matches, key=lambda x: (x['start'], -x.get('confidence', 0)))
        
        deduped = []
        last_end = -1
        
        for match in sorted_matches:
            if match['start'] >= last_end:
                deduped.append(match)
                last_end = match['end']
        
        return deduped
    
    def extract_numeric_patterns(self, text: str) -> Dict[str, List]:
        """Extract and categorize numeric patterns"""
        numeric_data = {
            'currencies': [],
            'percentages': [],
            'integers': [],
            'decimals': [],
            'formatted_numbers': []
        }
        
        # Extract currencies
        for match in re.finditer(self.PATTERNS['currency'], text):
            value = match.group()
            numeric_data['currencies'].append({
                'value': value,
                'amount': self._parse_currency(value),
                'position': (match.start(), match.end())
            })
        
        # Extract percentages
        for match in re.finditer(self.PATTERNS['percentage'], text):
            value = match.group()
            numeric_data['percentages'].append({
                'value': value,
                'amount': float(value.rstrip('%')),
                'position': (match.start(), match.end())
            })
        
        return numeric_data
    
    def _parse_currency(self, currency_str: str) -> float:
        """Parse currency string to float"""
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$,]', '', currency_str)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0