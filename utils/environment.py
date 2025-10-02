"""
Environment Detection and Configuration Utilities
Provides smart environment detection and URL resolution for multi-environment deployment
"""
import os
from typing import Literal, Dict, Any
from enum import Enum

class Environment(str, Enum):
    """Environment types"""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    AZURE = "azure"

class EnvironmentConfig:
    """Environment-aware configuration manager"""
    
    def __init__(self):
        self._environment = self._detect_environment()
        self._config = self._load_environment_config()
    
    def _detect_environment(self) -> Environment:
        """Auto-detect current environment based on various indicators"""
        
        # Check explicit environment variable first
        env_var = os.getenv('ENVIRONMENT', '').lower()
        if env_var in [e.value for e in Environment]:
            return Environment(env_var)
        
        # Azure App Service detection
        if os.getenv('WEBSITE_SITE_NAME') or os.getenv('APPSETTING_WEBSITE_SITE_NAME'):
            return Environment.AZURE
        
        # Kubernetes detection
        if os.getenv('KUBERNETES_SERVICE_HOST'):
            return Environment.PRODUCTION
        
        # Docker detection
        if os.path.exists('/.dockerenv'):
            return Environment.DEVELOPMENT
        
        # Default to local
        return Environment.LOCAL
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        
        configs = {
            Environment.LOCAL: {
                'backend_url': 'http://localhost:3001',
                'ai_service_url': 'http://localhost:8000',
                'service_host': '0.0.0.0',
                'service_port': 8000,
                'debug': True,
                'reload': True
            },
            Environment.DEVELOPMENT: {
                'backend_url': 'http://localhost:3001',
                'ai_service_url': 'http://localhost:8000',
                'service_host': '0.0.0.0',
                'service_port': 8000,
                'debug': True,
                'reload': True
            },
            Environment.STAGING: {
                'backend_url': 'https://revai-api-staging.azurewebsites.net',
                'ai_service_url': 'https://revai-ai-staging.azurewebsites.net',
                'service_host': '0.0.0.0',
                'service_port': 8000,
                'debug': False,
                'reload': False
            },
            Environment.PRODUCTION: {
                'backend_url': 'https://revai-api-mainv2.azurewebsites.net',
                'ai_service_url': 'https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net',
                'service_host': '0.0.0.0',
                'service_port': 8000,
                'debug': False,
                'reload': False
            },
            Environment.AZURE: {
                'backend_url': 'https://revai-api-mainv2.azurewebsites.net',
                'ai_service_url': 'https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net',
                'service_host': '0.0.0.0',
                'service_port': 8000,
                'debug': False,
                'reload': False
            }
        }
        
        return configs.get(self._environment, configs[Environment.LOCAL])
    
    @property
    def environment(self) -> Environment:
        """Get current environment"""
        return self._environment
    
    @property
    def is_local(self) -> bool:
        """Check if running in local environment"""
        return self._environment in [Environment.LOCAL, Environment.DEVELOPMENT]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self._environment in [Environment.PRODUCTION, Environment.AZURE]
    
    def get_backend_url(self) -> str:
        """Get backend URL based on environment"""
        return os.getenv('BACKEND_BASE_URL') or os.getenv('NODEJS_BACKEND_URL') or self._config['backend_url']
    
    def get_ai_service_url(self) -> str:
        """Get AI service URL based on environment"""
        return os.getenv('AI_SERVICE_URL') or self._config['ai_service_url']
    
    def get_service_host(self) -> str:
        """Get service host based on environment"""
        return os.getenv('SERVICE_HOST') or self._config['service_host']
    
    def get_service_port(self) -> int:
        """Get service port based on environment"""
        return int(os.getenv('SERVICE_PORT') or os.getenv('PORT') or self._config['service_port'])
    
    def should_reload(self) -> bool:
        """Check if auto-reload should be enabled"""
        return self._config['reload'] and self.is_local
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode should be enabled"""
        return self._config['debug'] and self.is_local
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            'environment': self._environment.value,
            'backend_url': self.get_backend_url(),
            'ai_service_url': self.get_ai_service_url(),
            'service_host': self.get_service_host(),
            'service_port': self.get_service_port(),
            'debug_mode': self.is_debug_mode(),
            'auto_reload': self.should_reload(),
            'is_local': self.is_local,
            'is_production': self.is_production
        }

# Global environment configuration instance
env_config = EnvironmentConfig()

# Convenience functions for backward compatibility
def get_environment() -> Environment:
    """Get current environment"""
    return env_config.environment

def get_backend_url() -> str:
    """Get backend URL based on environment"""
    return env_config.get_backend_url()

def get_ai_service_url() -> str:
    """Get AI service URL based on environment"""
    return env_config.get_ai_service_url()

def is_local_environment() -> bool:
    """Check if running in local environment"""
    return env_config.is_local

def is_production_environment() -> bool:
    """Check if running in production environment"""
    return env_config.is_production