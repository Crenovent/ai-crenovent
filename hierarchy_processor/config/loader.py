"""
Configuration Loader
Manages YAML configuration files for different HRMS systems
"""

import yaml
import os
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Dynamic configuration loader for HRMS system mappings"""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Default to mappings directory relative to this file
            self.config_dir = Path(__file__).parent / "mappings"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_cache = {}
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created config directory: {self.config_dir}")
    
    def load_config(self, system: str) -> Dict:
        """
        Load configuration for a specific HRMS system
        
        Args:
            system: System name (e.g., 'salesforce', 'hubspot', 'zoho')
            
        Returns:
            Configuration dictionary
        """
        # Check cache first
        if system in self.config_cache:
            return self.config_cache[system]
        
        config_file = self.config_dir / f"{system}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            # Return generic config if specific system config doesn't exist
            if system != 'generic':
                return self.load_config('generic')
            else:
                return self._get_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate config structure
            validated_config = self._validate_config(config, system)
            
            # Cache the config
            self.config_cache[system] = validated_config
            
            logger.info(f"Loaded config for system: {system}")
            return validated_config
            
        except Exception as e:
            logger.error(f"Error loading config for {system}: {e}")
            if system != 'generic':
                return self.load_config('generic')
            else:
                return self._get_default_config()
    
    def _validate_config(self, config: Dict, system: str) -> Dict:
        """Validate configuration structure"""
        if not isinstance(config, dict):
            raise ValueError(f"Config for {system} must be a dictionary")
        
        # Ensure required sections exist
        if 'fields' not in config:
            config['fields'] = {}
        
        if 'metadata' not in config:
            config['metadata'] = {
                'system': system,
                'version': '1.0'
            }
        
        # Validate field configurations
        for field_name, field_config in config['fields'].items():
            if isinstance(field_config, dict):
                if 'patterns' not in field_config:
                    field_config['patterns'] = []
                elif not isinstance(field_config['patterns'], list):
                    field_config['patterns'] = [field_config['patterns']]
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default configuration when no specific config is available"""
        return {
            'metadata': {
                'system': 'generic',
                'version': '1.0',
                'description': 'Generic fallback configuration'
            },
            'fields': {
                'Name': {
                    'patterns': ['name', 'full name', 'employee name', 'user name'],
                    'required': True,
                    'type': 'string'
                },
                'Email': {
                    'patterns': ['email', 'email address', 'work email'],
                    'required': True,
                    'type': 'email'
                },
                'Role Title': {
                    'patterns': ['title', 'job title', 'role', 'position'],
                    'required': False,
                    'type': 'string'
                },
                'Reporting Email': {
                    'patterns': ['manager email', 'supervisor email', 'reporting email'],
                    'required': False,
                    'type': 'email'
                },
                'Reporting Manager Name': {
                    'patterns': ['manager', 'supervisor', 'reporting manager'],
                    'required': False,
                    'type': 'string'
                }
            }
        }
    
    def get_available_systems(self) -> List[str]:
        """Get list of available system configurations"""
        if not self.config_dir.exists():
            return ['generic']
        
        systems = []
        for config_file in self.config_dir.glob("*.yaml"):
            system_name = config_file.stem
            systems.append(system_name)
        
        # Always include generic if not present
        if 'generic' not in systems:
            systems.append('generic')
        
        return sorted(systems)
    
    def reload_config(self, system: str) -> Dict:
        """Reload configuration for a system (clears cache)"""
        if system in self.config_cache:
            del self.config_cache[system]
        return self.load_config(system)
    
    def clear_cache(self):
        """Clear all cached configurations"""
        self.config_cache.clear()
        logger.info("Configuration cache cleared")
    
    def create_config_template(self, system: str) -> str:
        """Create a configuration template for a new system"""
        template = {
            'metadata': {
                'system': system,
                'version': '1.0',
                'description': f'{system.title()} HRMS field mappings'
            },
            'detection': {
                'required_fields': {
                    'name': ['full name', 'name'],
                    'email': ['email', 'email address'],
                    'role': ['title', 'job title']
                },
                'unique_indicators': [f'{system} specific field']
            },
            'fields': {
                'Name': {
                    'patterns': ['Full Name', 'Name', 'Employee Name'],
                    'required': True,
                    'type': 'string'
                },
                'Email': {
                    'patterns': ['Email', 'Email Address', 'Work Email'],
                    'required': True,
                    'type': 'email'
                },
                'Role Title': {
                    'patterns': ['Title', 'Job Title', 'Role'],
                    'required': False,
                    'type': 'string'
                },
                'Reporting Email': {
                    'patterns': ['Manager Email', 'Supervisor Email'],
                    'required': False,
                    'type': 'email'
                },
                'Reporting Manager Name': {
                    'patterns': ['Manager', 'Supervisor', 'Reports To'],
                    'required': False,
                    'type': 'string'
                }
            },
            'processing_rules': {
                'name_combination': {
                    'enabled': False,
                    'fields': ['First Name', 'Last Name'],
                    'separator': ' '
                },
                'role_mapping': {
                    f'{system.title()} Admin': 'admin',
                    f'{system.title()} Manager': 'manager',
                    f'{system.title()} User': 'sales'
                }
            }
        }
        
        config_file = self.config_dir / f"{system}.yaml"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created config template: {config_file}")
            return str(config_file)
            
        except Exception as e:
            logger.error(f"Failed to create config template for {system}: {e}")
            raise
