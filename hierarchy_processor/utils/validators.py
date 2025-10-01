"""
Data Validation Utilities
Validates CSV data and field mappings
"""

import re
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates CSV data and processing results"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.phone_pattern = re.compile(r'^[\+]?[1-9][\d\s\-\(\)]{7,15}$')
    
    def validate_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic CSV structure and data quality"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'stats': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'empty_rows': 0,
                'duplicate_rows': 0,
                'missing_data_percentage': 0
            }
        }
        
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("CSV file is empty")
            return validation_result
        
        # Check for empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        validation_result['stats']['empty_rows'] = int(empty_rows)
        
        if empty_rows > 0:
            validation_result['warnings'].append(f"Found {empty_rows} completely empty rows")
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        validation_result['stats']['duplicate_rows'] = int(duplicate_rows)
        
        if duplicate_rows > 0:
            validation_result['warnings'].append(f"Found {duplicate_rows} duplicate rows")
        
        # Calculate missing data percentage
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        validation_result['stats']['missing_data_percentage'] = round(missing_percentage, 2)
        
        if missing_percentage > 50:
            validation_result['warnings'].append(f"High missing data: {missing_percentage:.1f}%")
        
        # Check column names
        empty_columns = [col for col in df.columns if not col or str(col).strip() == '']
        if empty_columns:
            validation_result['errors'].append(f"Found {len(empty_columns)} columns with empty names")
            validation_result['is_valid'] = False
        
        # Basic structure validation
        if len(df.columns) < 2:
            validation_result['errors'].append("CSV must have at least 2 columns")
            validation_result['is_valid'] = False
        
        if len(df) < 1:
            validation_result['errors'].append("CSV must have at least 1 data row")
            validation_result['is_valid'] = False
        
        validation_result['info'].append(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
        
        return validation_result
    
    def validate_field_mappings(self, mappings: Dict, required_fields: List[str]) -> Dict[str, Any]:
        """Validate field mapping results"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check required fields
        missing_required = []
        for field in required_fields:
            if field not in mappings or not mappings[field]:
                missing_required.append(field)
        
        if missing_required:
            validation_result['errors'].extend([
                f"Required field '{field}' is not mapped" for field in missing_required
            ])
            validation_result['is_valid'] = False
        
        # Check mapping confidence
        low_confidence_mappings = []
        for field, mapping in mappings.items():
            if mapping and mapping.get('confidence', 0) < 0.7:
                low_confidence_mappings.append(f"{field} ({mapping.get('confidence', 0):.2f})")
        
        if low_confidence_mappings:
            validation_result['warnings'].append(
                f"Low confidence mappings: {', '.join(low_confidence_mappings)}"
            )
        
        mapped_count = len([m for m in mappings.values() if m])
        validation_result['info'].append(f"Successfully mapped {mapped_count} fields")
        
        return validation_result
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        if not email or not isinstance(email, str):
            return False
        return bool(self.email_pattern.match(email.strip()))
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        if not phone or not isinstance(phone, str):
            return False
        return bool(self.phone_pattern.match(phone.strip()))
    
    def validate_normalized_data(self, normalized_data: List[Dict]) -> Dict[str, Any]:
        """Validate normalized data records"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'stats': {
                'total_records': len(normalized_data),
                'valid_records': 0,
                'invalid_emails': 0,
                'missing_names': 0,
                'missing_emails': 0
            }
        }
        
        if not normalized_data:
            validation_result['errors'].append("No normalized data to validate")
            validation_result['is_valid'] = False
            return validation_result
        
        valid_records = 0
        invalid_emails = 0
        missing_names = 0
        missing_emails = 0
        
        for i, record in enumerate(normalized_data):
            record_valid = True
            
            # Check required fields
            name = record.get('Name', '').strip()
            email = record.get('Email', '').strip()
            
            if not name:
                missing_names += 1
                record_valid = False
            
            if not email:
                missing_emails += 1
                record_valid = False
            elif not self.validate_email(email):
                invalid_emails += 1
                record_valid = False
            
            if record_valid:
                valid_records += 1
        
        # Update stats
        validation_result['stats']['valid_records'] = valid_records
        validation_result['stats']['invalid_emails'] = invalid_emails
        validation_result['stats']['missing_names'] = missing_names
        validation_result['stats']['missing_emails'] = missing_emails
        
        # Generate messages
        if missing_names > 0:
            validation_result['errors'].append(f"{missing_names} records missing names")
        
        if missing_emails > 0:
            validation_result['errors'].append(f"{missing_emails} records missing emails")
        
        if invalid_emails > 0:
            validation_result['errors'].append(f"{invalid_emails} records have invalid email formats")
        
        # Determine overall validity
        if valid_records == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("No valid records found")
        elif valid_records < len(normalized_data) * 0.8:  # Less than 80% valid
            validation_result['warnings'].append(
                f"Only {valid_records}/{len(normalized_data)} records are valid ({valid_records/len(normalized_data)*100:.1f}%)"
            )
        
        validation_result['info'].append(
            f"Validated {valid_records}/{len(normalized_data)} records successfully"
        )
        
        return validation_result
    
    def validate_hierarchy_data(self, normalized_data: List[Dict]) -> Dict[str, Any]:
        """Validate hierarchy-related data in normalized records"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'hierarchy_stats': {
                'records_with_manager_email': 0,
                'records_with_manager_name': 0,
                'records_with_no_hierarchy': 0,
                'potential_circular_references': []
            }
        }
        
        manager_emails = 0
        manager_names = 0
        no_hierarchy = 0
        
        # Track potential circular references
        email_to_manager = {}
        
        for record in normalized_data:
            email = record.get('Email', '').strip()
            manager_email = record.get('Reporting Email', '').strip()
            manager_name = record.get('Reporting Manager Name', '').strip()
            
            has_manager_email = bool(manager_email)
            has_manager_name = bool(manager_name)
            
            if has_manager_email:
                manager_emails += 1
                if email:
                    email_to_manager[email] = manager_email
            
            if has_manager_name:
                manager_names += 1
            
            if not has_manager_email and not has_manager_name:
                no_hierarchy += 1
        
        # Check for potential circular references
        circular_refs = []
        for email, manager_email in email_to_manager.items():
            if manager_email in email_to_manager:
                # Check if manager's manager is the original person
                managers_manager = email_to_manager[manager_email]
                if managers_manager == email:
                    circular_refs.append(f"{email} <-> {manager_email}")
        
        # Update stats
        validation_result['hierarchy_stats']['records_with_manager_email'] = manager_emails
        validation_result['hierarchy_stats']['records_with_manager_name'] = manager_names
        validation_result['hierarchy_stats']['records_with_no_hierarchy'] = no_hierarchy
        validation_result['hierarchy_stats']['potential_circular_references'] = circular_refs
        
        # Generate messages
        if circular_refs:
            validation_result['warnings'].append(
                f"Potential circular references detected: {', '.join(circular_refs)}"
            )
        
        if no_hierarchy > len(normalized_data) * 0.5:  # More than 50% have no hierarchy
            validation_result['warnings'].append(
                f"High number of records without hierarchy information: {no_hierarchy}/{len(normalized_data)}"
            )
        
        validation_result['info'].append(
            f"Hierarchy validation: {manager_emails} with manager emails, "
            f"{manager_names} with manager names, {no_hierarchy} without hierarchy"
        )
        
        return validation_result
