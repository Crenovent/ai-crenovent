"""
Dynamic Field Mapper
===================
Universal field mapping system that can automatically detect and map CSV field names
to standardized Crenovent field names. Handles different naming conventions across
various HRMS systems, CSV formats, and data sources.

Key Features:
- Fuzzy string matching for field name detection
- Confidence scoring for mapping quality
- Support for multiple field name variations
- Extensible pattern matching system
- Fallback mechanisms for unknown fields
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher
import pandas as pd

logger = logging.getLogger(__name__)

class DynamicFieldMapper:
    """
    Universal field mapper that can dynamically detect and map field names
    from any CSV format to standardized Crenovent field names.
    """
    
    def __init__(self):
        # Standard Crenovent field definitions with multiple possible source patterns
        self.field_patterns = {
            # Core Identity Fields
            'name': {
                'patterns': [
                    'name', 'full name', 'fullname', 'employee name', 'display name',
                    'first name', 'last name', 'fname', 'lname', 'given name',
                    'complete name', 'person name', 'user name', 'username'
                ],
                'required': True,
                'description': 'Full name of the employee'
            },
            'email': {
                'patterns': [
                    'email', 'email address', 'emailaddress', 'work email', 'business email',
                    'corporate email', 'official email', 'email id', 'e-mail',
                    'primary email', 'contact email', 'user email'
                ],
                'required': True,
                'description': 'Work email address'
            },
            
            # Role & Position Fields
            'title': {
                'patterns': [
                    'title', 'job title', 'jobtitle', 'position', 'role title',
                    'designation', 'business title', 'position title', 'role',
                    'job position', 'work title', 'professional title'
                ],
                'required': True,
                'description': 'Job title or position'
            },
            'department': {
                'patterns': [
                    'department', 'dept', 'team', 'division', 'function',
                    'business unit', 'org', 'organization', 'role function',
                    'business function', 'unit', 'group', 'section'
                ],
                'required': False,
                'description': 'Department or functional area'
            },
            'level': {
                'patterns': [
                    'level', 'grade', 'seniority', 'employee level', 'job level',
                    'position level', 'career level', 'hierarchy level', 'rank',
                    'tier', 'band', 'classification'
                ],
                'required': False,
                'description': 'Seniority or organizational level'
            },
            
            # Manager/Reporting Fields
            'manager_name': {
                'patterns': [
                    'manager', 'manager name', 'reporting manager', 'supervisor',
                    'reports to', 'line manager', 'direct manager', 'boss',
                    'reporting manager name', 'supervisor name', 'manager full name'
                ],
                'required': False,
                'description': 'Manager or supervisor name'
            },
            'manager_email': {
                'patterns': [
                    'manager email', 'supervisor email', 'reporting manager email',
                    'reports to email', 'line manager email', 'boss email',
                    'manager email address', 'supervisor email address',
                    'reporting email'
                ],
                'required': False,
                'description': 'Manager or supervisor email'
            },
            
            # Location Fields
            'location': {
                'patterns': [
                    'location', 'office', 'site', 'workplace', 'work location',
                    'office location', 'primary address', 'address', 'city',
                    'country', 'region', 'geography', 'base location'
                ],
                'required': False,
                'description': 'Work location or address'
            },
            'region': {
                'patterns': [
                    'region', 'area', 'territory', 'geography', 'zone',
                    'market', 'district', 'sector', 'locale'
                ],
                'required': False,
                'description': 'Geographic region or territory'
            },
            
            # Additional Fields
            'employee_id': {
                'patterns': [
                    'employee id', 'emp id', 'id', 'employee number', 'staff id',
                    'worker id', 'person id', 'user id', 'badge number'
                ],
                'required': False,
                'description': 'Employee identifier'
            },
            'hire_date': {
                'patterns': [
                    'hire date', 'start date', 'join date', 'employment date',
                    'onboard date', 'joining date', 'date of joining'
                ],
                'required': False,
                'description': 'Employee hire or start date'
            }
        }
        
        # Crenovent standard field names (what we map TO)
        self.standard_fields = {
            'name': 'Name',
            'email': 'Email', 
            'title': 'Role Title',
            'department': 'Role Function',
            'level': 'Level',
            'manager_name': 'Reporting Manager Name',
            'manager_email': 'Reporting Email',
            'location': 'location',  # Keep lowercase for location processing
            'region': 'Region',
            'employee_id': 'Employee ID',
            'hire_date': 'Hire Date'
        }
    
    def detect_field_mappings(self, csv_columns: List[str], confidence_threshold: float = 0.6) -> Dict[str, Dict[str, Any]]:
        """
        Automatically detect field mappings from CSV columns to Crenovent standard fields.
        
        Args:
            csv_columns: List of column names from the CSV
            confidence_threshold: Minimum confidence score for accepting a mapping
            
        Returns:
            Dictionary mapping standard field names to detected CSV columns with confidence scores
        """
        mappings = {}
        used_columns = set()
        
        logger.info(f"🔍 Detecting field mappings for {len(csv_columns)} CSV columns")
        logger.debug(f"CSV columns: {csv_columns}")
        
        for standard_field, field_config in self.field_patterns.items():
            best_match = None
            best_confidence = 0.0
            best_column = None
            
            for csv_column in csv_columns:
                if csv_column in used_columns:
                    continue
                    
                confidence = self._calculate_field_confidence(csv_column, field_config['patterns'])
                
                if confidence > best_confidence and confidence >= confidence_threshold:
                    best_confidence = confidence
                    best_column = csv_column
                    best_match = {
                        'csv_column': csv_column,
                        'standard_field': self.standard_fields[standard_field],
                        'confidence': confidence,
                        'required': field_config['required'],
                        'description': field_config['description']
                    }
            
            if best_match:
                mappings[standard_field] = best_match
                used_columns.add(best_column)
                logger.info(f"✅ Mapped '{best_column}' -> '{self.standard_fields[standard_field]}' (confidence: {best_confidence:.2f})")
            elif field_config['required']:
                logger.warning(f"⚠️ Required field '{standard_field}' not found in CSV columns")
        
        # Log unmapped columns
        unmapped_columns = [col for col in csv_columns if col not in used_columns]
        if unmapped_columns:
            logger.info(f"📋 Unmapped CSV columns: {unmapped_columns}")
        
        return mappings
    
    def _calculate_field_confidence(self, csv_column: str, patterns: List[str]) -> float:
        """
        Calculate confidence score for mapping a CSV column to a field pattern.
        Uses fuzzy string matching and exact pattern matching.
        """
        csv_column_clean = self._normalize_field_name(csv_column)
        max_confidence = 0.0
        
        for pattern in patterns:
            pattern_clean = self._normalize_field_name(pattern)
            
            # Exact match gets highest confidence
            if csv_column_clean == pattern_clean:
                return 1.0
            
            # Substring match gets high confidence
            if pattern_clean in csv_column_clean or csv_column_clean in pattern_clean:
                confidence = 0.9
                max_confidence = max(max_confidence, confidence)
                continue
            
            # Fuzzy string matching for partial matches
            similarity = SequenceMatcher(None, csv_column_clean, pattern_clean).ratio()
            max_confidence = max(max_confidence, similarity)
        
        return max_confidence
    
    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize field name for comparison (lowercase, no spaces/special chars)"""
        return re.sub(r'[^a-z0-9]', '', field_name.lower().strip())
    
    def map_user_data(self, user_data: Dict[str, Any], field_mappings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map user data from CSV format to Crenovent standard format using detected mappings.
        
        Args:
            user_data: Raw user data from CSV
            field_mappings: Field mappings detected by detect_field_mappings()
            
        Returns:
            User data mapped to Crenovent standard field names
        """
        mapped_data = {}
        
        for standard_field, mapping_info in field_mappings.items():
            csv_column = mapping_info['csv_column']
            standard_field_name = mapping_info['standard_field']
            
            # Get value from original data
            value = user_data.get(csv_column, '')
            
            # Apply any field-specific transformations
            transformed_value = self._transform_field_value(standard_field, value)
            
            mapped_data[standard_field_name] = transformed_value
        
        # Copy any unmapped fields as-is (preserve original data)
        for key, value in user_data.items():
            if key not in [m['csv_column'] for m in field_mappings.values()]:
                mapped_data[key] = value
        
        return mapped_data
    
    def _transform_field_value(self, field_type: str, value: Any) -> Any:
        """Apply field-specific transformations to values"""
        if not value or pd.isna(value):
            return ''
        
        value_str = str(value).strip()
        
        if field_type == 'email':
            # Ensure email is lowercase and valid format
            return value_str.lower()
        elif field_type == 'name':
            # Proper case for names
            return value_str.title()
        elif field_type in ['title', 'department']:
            # Title case for titles and departments
            return value_str.title()
        else:
            return value_str
    
    def get_mapping_summary(self, field_mappings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the field mappings for logging/debugging"""
        total_fields = len(self.field_patterns)
        mapped_fields = len(field_mappings)
        required_fields = sum(1 for config in self.field_patterns.values() if config['required'])
        mapped_required = sum(1 for mapping in field_mappings.values() if mapping['required'])
        
        avg_confidence = sum(mapping['confidence'] for mapping in field_mappings.values()) / max(mapped_fields, 1)
        
        return {
            'total_standard_fields': total_fields,
            'mapped_fields': mapped_fields,
            'mapping_coverage': mapped_fields / total_fields,
            'required_fields_total': required_fields,
            'required_fields_mapped': mapped_required,
            'required_coverage': mapped_required / max(required_fields, 1),
            'average_confidence': avg_confidence,
            'mappings': {
                mapping['csv_column']: {
                    'standard_field': mapping['standard_field'],
                    'confidence': mapping['confidence']
                }
                for mapping in field_mappings.values()
            }
        }

# Global instance for use across the codebase
dynamic_field_mapper = DynamicFieldMapper()
