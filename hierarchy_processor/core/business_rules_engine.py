"""
Dynamic Business Rules Engine for Crenovent Hierarchy Processing

This module loads business rules from YAML configuration files and applies them
dynamically to map CSV data to Crenovent format without hardcoded logic.
"""

import yaml
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BusinessRulesEngine:
    """
    Dynamic engine that loads and applies business rules from YAML configurations.
    """
    
    def __init__(self, config_base_path: str = None):
        """
        Initialize the Business Rules Engine.
        
        Args:
            config_base_path: Base path to the business rules configuration directory
        """
        if config_base_path is None:
            # Default to the business_rules directory
            current_dir = Path(__file__).parent.parent
            config_base_path = current_dir / "config" / "business_rules"
        
        self.config_path = Path(config_base_path)
        
        # Load all rule configurations
        self.role_function_rules = self._load_config("role_function_mapping.yaml")
        self.level_rules = self._load_config("level_mapping.yaml")
        self.business_function_rules = self._load_config("business_function_mapping.yaml")
        self.region_rules = self._load_config("region_mapping.yaml")
        self.segment_rules = self._load_config("segment_mapping.yaml")
        self.default_values = self._load_config("default_values.yaml")
        
        logger.info("✅ Business Rules Engine initialized with dynamic YAML configurations")
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        try:
            config_file = self.config_path / filename
            if not config_file.exists():
                logger.warning(f"⚠️ Configuration file not found: {config_file}")
                return {}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Loaded business rules from {filename}")
                return config
        except Exception as e:
            logger.error(f"❌ Failed to load {filename}: {e}")
            return {}
    
    def map_role_function(self, job_title: str, department: str = "") -> str:
        """
        Map job title to Role Function using dynamic rules.
        
        Args:
            job_title: The job title from CSV
            department: Optional department info
            
        Returns:
            Mapped Role Function
        """
        if not job_title:
            return self.role_function_rules.get('default_role_function', 'Sales')
        
        text_to_analyze = f"{job_title} {department}".lower()
        
        # Find the best matching rule
        best_match = None
        highest_priority = 0
        
        for rule in self.role_function_rules.get('role_function_rules', []):
            keywords = rule.get('keywords', [])
            priority = rule.get('priority', 0)
            
            # Check if any keyword matches
            for keyword in keywords:
                if keyword.lower() in text_to_analyze:
                    if priority > highest_priority:
                        highest_priority = priority
                        best_match = rule
                    break
        
        if best_match:
            logger.info(f"Mapped '{job_title}' -> Role Function: {best_match['role_function']}")
            return best_match['role_function']
        
        default_role = self.role_function_rules.get('default_role_function', 'Sales')
        logger.info(f"No match found for '{job_title}', using default: {default_role}")
        return default_role
    
    def map_level(self, job_title: str, role_function: str = "") -> str:
        """
        Map job title to Level using dynamic rules.
        
        Args:
            job_title: The job title from CSV
            role_function: Optional role function for context
            
        Returns:
            Mapped Level
        """
        if not job_title:
            return self.level_rules.get('default_level', 'IC')
        
        text_to_analyze = f"{job_title} {role_function}".lower()
        
        # Check combined rules first (higher specificity)
        for rule in self.level_rules.get('combined_rules', []):
            keywords = rule.get('keywords', [])
            if all(keyword.lower() in text_to_analyze for keyword in keywords):
                logger.info(f"Combined rule matched '{job_title}' -> Level: {rule['level']}")
                return rule['level']
        
        # Find the best matching single rule
        best_match = None
        highest_priority = 0
        
        for rule in self.level_rules.get('level_rules', []):
            keywords = rule.get('keywords', [])
            priority = rule.get('priority', 0)
            
            # Check if any keyword matches
            for keyword in keywords:
                if keyword.lower() in text_to_analyze:
                    if priority > highest_priority:
                        highest_priority = priority
                        best_match = rule
                    break
        
        if best_match:
            logger.info(f"Mapped '{job_title}' -> Level: {best_match['level']}")
            return best_match['level']
        
        default_level = self.level_rules.get('default_level', 'IC')
        logger.info(f"No match found for '{job_title}', using default: {default_level}")
        return default_level
    
    def map_business_function(self, department: str, role_function: str = "") -> str:
        """
        Map department to Business Function using dynamic rules.
        
        Args:
            department: The department from CSV
            role_function: Role function for secondary mapping
            
        Returns:
            Mapped Business Function
        """
        # Try department-based mapping first
        if department:
            text_to_analyze = department.lower()
            
            best_match = None
            highest_priority = 0
            
            for rule in self.business_function_rules.get('business_function_rules', []):
                keywords = rule.get('keywords', [])
                priority = rule.get('priority', 0)
                
                for keyword in keywords:
                    if keyword.lower() in text_to_analyze:
                        if priority > highest_priority:
                            highest_priority = priority
                            best_match = rule
                        break
            
            if best_match:
                logger.info(f"Department '{department}' -> Business Function: {best_match['business_function']}")
                return best_match['business_function']
        
        # Try role function mapping as fallback
        if role_function:
            role_mapping = self.business_function_rules.get('role_function_mapping', {})
            if role_function in role_mapping:
                logger.info(f"Role Function '{role_function}' -> Business Function: {role_mapping[role_function]}")
                return role_mapping[role_function]
        
        default_bf = self.business_function_rules.get('default_business_function', 'BF2 America')
        logger.info(f"No match found for dept='{department}', role='{role_function}', using default: {default_bf}")
        return default_bf
    
    def map_region(self, location: str) -> str:
        """
        Map location to Region using dynamic rules.
        
        Args:
            location: The location from CSV
            
        Returns:
            Mapped Region
        """
        if not location:
            return self.region_rules.get('default_region', 'America')
        
        text_to_analyze = location.lower()
        
        best_match = None
        highest_priority = 0
        
        for rule in self.region_rules.get('region_rules', []):
            keywords = rule.get('keywords', [])
            priority = rule.get('priority', 0)
            
            for keyword in keywords:
                if keyword.lower() in text_to_analyze:
                    if priority > highest_priority:
                        highest_priority = priority
                        best_match = rule
                    break
        
        if best_match:
            logger.info(f"Location '{location}' -> Region: {best_match['region']}")
            return best_match['region']
        
        default_region = self.region_rules.get('default_region', 'America')
        logger.info(f"No match found for '{location}', using default: {default_region}")
        return default_region
    
    def get_default_values(self) -> Dict[str, str]:
        """Get default values for all fields."""
        return self.default_values.get('defaults', {})
    
    def get_optional_defaults(self) -> Dict[str, str]:
        """Get default values for optional fields."""
        return self.default_values.get('optional_defaults', {})
    
    def get_field_mappings(self) -> Dict[str, str]:
        """Get field name mappings from snake_case to proper case."""
        return self.default_values.get('field_mappings', {})
    
    def map_segment(self, level: str, role_function: str = "") -> str:
        """
        Map level and role function to Segment using dynamic rules.
        
        Args:
            level: The level from CSV
            role_function: Role function for overrides
            
        Returns:
            Mapped Segment
        """
        # Check role function overrides first
        if role_function:
            role_overrides = self.segment_rules.get('role_function_overrides', {})
            if role_function in role_overrides:
                logger.info(f"Role Function override '{role_function}' -> Segment: {role_overrides[role_function]}")
                return role_overrides[role_function]
        
        # Check level-based mapping
        if level:
            best_match = None
            highest_priority = 0
            
            for rule in self.segment_rules.get('segment_rules', []):
                levels = rule.get('levels', [])
                priority = rule.get('priority', 0)
                
                if level in levels and priority > highest_priority:
                    highest_priority = priority
                    best_match = rule
            
            if best_match:
                logger.info(f"Level '{level}' -> Segment: {best_match['segment']}")
                return best_match['segment']
        
        default_segment = self.segment_rules.get('default_segment', 'SMB')
        logger.info(f"No match found for level='{level}', role='{role_function}', using default: {default_segment}")
        return default_segment
    
    def apply_conditional_defaults(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply conditional defaults based on other field values.
        
        Args:
            record: The current record
            
        Returns:
            Updated record with conditional defaults applied
        """
        conditional_rules = self.default_values.get('conditional_defaults', [])
        
        for rule in conditional_rules:
            condition = rule.get('condition', {})
            rule_defaults = rule.get('defaults', {})
            
            # Check if condition matches
            field = condition.get('field')
            expected_value = condition.get('value')
            expected_values = condition.get('values', [])
            
            if field and field in record:
                record_value = record[field]
                
                # Check single value condition
                if expected_value and record_value == expected_value:
                    logger.info(f"Conditional rule matched: {field}={expected_value}")
                    for key, value in rule_defaults.items():
                        if not record.get(key):  # Only set if not already set
                            record[key] = value
                
                # Check multiple values condition
                elif expected_values and record_value in expected_values:
                    logger.info(f"Conditional rule matched: {field} in {expected_values}")
                    for key, value in rule_defaults.items():
                        if not record.get(key):  # Only set if not already set
                            record[key] = value
        
        return record
    
    def reload_configurations(self):
        """Reload all configuration files (useful for runtime updates)."""
        logger.info("🔄 Reloading business rules configurations...")
        self.__init__(str(self.config_path))
