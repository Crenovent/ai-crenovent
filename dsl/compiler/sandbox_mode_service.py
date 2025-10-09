"""
Sandbox Mode - Task 6.2.75
===========================

Mark plans as non-prod for safer trials
- Manifest flag for sandbox mode
- Runtime honors sandbox restrictions
- Isolated execution environment
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class SandboxLevel(Enum):
    NONE = "none"           # Production mode
    BASIC = "basic"         # Basic sandbox restrictions
    STRICT = "strict"       # Strict sandbox with full isolation
    DEVELOPMENT = "development"  # Development sandbox

class SandboxRestriction(Enum):
    NO_EXTERNAL_CALLS = "no_external_calls"
    READ_ONLY_DATA = "read_only_data"
    LIMITED_RESOURCES = "limited_resources"
    NO_NOTIFICATIONS = "no_notifications"
    MOCK_ML_MODELS = "mock_ml_models"
    TIME_LIMITED = "time_limited"

@dataclass
class SandboxConfiguration:
    sandbox_level: SandboxLevel
    restrictions: List[SandboxRestriction]
    resource_limits: Dict[str, Any]
    allowed_external_domains: List[str]
    max_execution_time_minutes: int
    data_access_scope: str  # "tenant_sandbox", "user_sandbox", "isolated"

class SandboxModeService:
    """Service for managing sandbox mode execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sandbox_configs = self._initialize_sandbox_configs()
    
    def _initialize_sandbox_configs(self) -> Dict[SandboxLevel, SandboxConfiguration]:
        """Initialize sandbox configurations"""
        return {
            SandboxLevel.NONE: SandboxConfiguration(
                sandbox_level=SandboxLevel.NONE,
                restrictions=[],
                resource_limits={},
                allowed_external_domains=[],
                max_execution_time_minutes=0,  # No limit
                data_access_scope="full"
            ),
            SandboxLevel.BASIC: SandboxConfiguration(
                sandbox_level=SandboxLevel.BASIC,
                restrictions=[
                    SandboxRestriction.NO_NOTIFICATIONS,
                    SandboxRestriction.TIME_LIMITED
                ],
                resource_limits={
                    "max_memory_gb": 2,
                    "max_cpu_cores": 2,
                    "max_storage_gb": 10
                },
                allowed_external_domains=["api.sandbox.example.com"],
                max_execution_time_minutes=30,
                data_access_scope="tenant_sandbox"
            ),
            SandboxLevel.STRICT: SandboxConfiguration(
                sandbox_level=SandboxLevel.STRICT,
                restrictions=[
                    SandboxRestriction.NO_EXTERNAL_CALLS,
                    SandboxRestriction.READ_ONLY_DATA,
                    SandboxRestriction.LIMITED_RESOURCES,
                    SandboxRestriction.NO_NOTIFICATIONS,
                    SandboxRestriction.MOCK_ML_MODELS,
                    SandboxRestriction.TIME_LIMITED
                ],
                resource_limits={
                    "max_memory_gb": 1,
                    "max_cpu_cores": 1,
                    "max_storage_gb": 5
                },
                allowed_external_domains=[],
                max_execution_time_minutes=15,
                data_access_scope="isolated"
            ),
            SandboxLevel.DEVELOPMENT: SandboxConfiguration(
                sandbox_level=SandboxLevel.DEVELOPMENT,
                restrictions=[
                    SandboxRestriction.NO_NOTIFICATIONS,
                    SandboxRestriction.MOCK_ML_MODELS,
                    SandboxRestriction.TIME_LIMITED
                ],
                resource_limits={
                    "max_memory_gb": 4,
                    "max_cpu_cores": 4,
                    "max_storage_gb": 20
                },
                allowed_external_domains=["dev-api.example.com", "staging-api.example.com"],
                max_execution_time_minutes=60,
                data_access_scope="user_sandbox"
            )
        }
    
    def enable_sandbox_mode(self, ir_data: Dict[str, Any], 
                           sandbox_level: SandboxLevel = SandboxLevel.BASIC,
                           custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enable sandbox mode for a plan"""
        
        sandbox_config = self.sandbox_configs[sandbox_level]
        
        # Apply custom configuration if provided
        if custom_config:
            sandbox_config = self._merge_sandbox_config(sandbox_config, custom_config)
        
        # Add sandbox metadata to plan
        if 'metadata' not in ir_data:
            ir_data['metadata'] = {}
        
        ir_data['metadata']['sandbox_mode'] = {
            'enabled': True,
            'sandbox_level': sandbox_level.value,
            'restrictions': [r.value for r in sandbox_config.restrictions],
            'resource_limits': sandbox_config.resource_limits,
            'allowed_external_domains': sandbox_config.allowed_external_domains,
            'max_execution_time_minutes': sandbox_config.max_execution_time_minutes,
            'data_access_scope': sandbox_config.data_access_scope,
            'enabled_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Apply sandbox transformations to nodes
        ir_data = self._apply_sandbox_transformations(ir_data, sandbox_config)
        
        return ir_data
    
    def _apply_sandbox_transformations(self, ir_data: Dict[str, Any], 
                                     config: SandboxConfiguration) -> Dict[str, Any]:
        """Apply sandbox transformations to plan nodes"""
        nodes = ir_data.get('nodes', [])
        
        for node in nodes:
            # Apply restrictions to external nodes
            if 'external' in node.get('type', ''):
                if SandboxRestriction.NO_EXTERNAL_CALLS in config.restrictions:
                    node['sandbox_disabled'] = True
                    node['sandbox_reason'] = "External calls disabled in sandbox"
                else:
                    # Restrict to allowed domains
                    self._restrict_external_domains(node, config.allowed_external_domains)
            
            # Apply restrictions to ML nodes
            if 'ml_' in node.get('type', '') or node.get('type') == 'ml_node':
                if SandboxRestriction.MOCK_ML_MODELS in config.restrictions:
                    node['sandbox_mock_model'] = True
                    node['mock_model_config'] = {
                        'mock_confidence': 0.85,
                        'mock_prediction': 'sandbox_prediction',
                        'response_time_ms': 100
                    }
            
            # Apply restrictions to notification nodes
            if 'notification' in node.get('type', '') or 'notify' in node.get('type', ''):
                if SandboxRestriction.NO_NOTIFICATIONS in config.restrictions:
                    node['sandbox_disabled'] = True
                    node['sandbox_reason'] = "Notifications disabled in sandbox"
            
            # Apply data access restrictions
            if 'data' in node.get('type', ''):
                if SandboxRestriction.READ_ONLY_DATA in config.restrictions:
                    node['sandbox_read_only'] = True
                
                # Restrict data scope
                if config.data_access_scope != "full":
                    node['sandbox_data_scope'] = config.data_access_scope
            
            # Add resource limits to all nodes
            if SandboxRestriction.LIMITED_RESOURCES in config.restrictions:
                node['sandbox_resource_limits'] = config.resource_limits
        
        return ir_data
    
    def _restrict_external_domains(self, node: Dict[str, Any], allowed_domains: List[str]):
        """Restrict external node to allowed domains"""
        node_config = node.get('config', {})
        endpoint_url = node_config.get('endpoint_url', '')
        
        if endpoint_url and allowed_domains:
            domain_allowed = any(domain in endpoint_url for domain in allowed_domains)
            if not domain_allowed:
                node['sandbox_disabled'] = True
                node['sandbox_reason'] = f"Domain not in allowed list: {allowed_domains}"
    
    def _merge_sandbox_config(self, base_config: SandboxConfiguration, 
                            custom_config: Dict[str, Any]) -> SandboxConfiguration:
        """Merge custom configuration with base sandbox config"""
        # Create a copy of base config
        merged_config = SandboxConfiguration(
            sandbox_level=base_config.sandbox_level,
            restrictions=base_config.restrictions.copy(),
            resource_limits=base_config.resource_limits.copy(),
            allowed_external_domains=base_config.allowed_external_domains.copy(),
            max_execution_time_minutes=base_config.max_execution_time_minutes,
            data_access_scope=base_config.data_access_scope
        )
        
        # Apply custom overrides
        if 'max_execution_time_minutes' in custom_config:
            merged_config.max_execution_time_minutes = custom_config['max_execution_time_minutes']
        
        if 'resource_limits' in custom_config:
            merged_config.resource_limits.update(custom_config['resource_limits'])
        
        if 'allowed_external_domains' in custom_config:
            merged_config.allowed_external_domains.extend(custom_config['allowed_external_domains'])
        
        return merged_config
    
    def validate_sandbox_compliance(self, ir_data: Dict[str, Any]) -> List[str]:
        """Validate that plan complies with sandbox restrictions"""
        issues = []
        
        sandbox_metadata = ir_data.get('metadata', {}).get('sandbox_mode', {})
        if not sandbox_metadata.get('enabled', False):
            return issues  # Not in sandbox mode
        
        sandbox_level = SandboxLevel(sandbox_metadata.get('sandbox_level', 'basic'))
        config = self.sandbox_configs[sandbox_level]
        restrictions = sandbox_metadata.get('restrictions', [])
        
        nodes = ir_data.get('nodes', [])
        
        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', 'unknown')
            
            # Check external call restrictions
            if ('external' in node_type and 
                SandboxRestriction.NO_EXTERNAL_CALLS.value in restrictions and
                not node.get('sandbox_disabled', False)):
                issues.append(f"Node {node_id}: External calls should be disabled in sandbox")
            
            # Check notification restrictions
            if ('notification' in node_type and 
                SandboxRestriction.NO_NOTIFICATIONS.value in restrictions and
                not node.get('sandbox_disabled', False)):
                issues.append(f"Node {node_id}: Notifications should be disabled in sandbox")
            
            # Check ML model mocking
            if (('ml_' in node_type or node_type == 'ml_node') and 
                SandboxRestriction.MOCK_ML_MODELS.value in restrictions and
                not node.get('sandbox_mock_model', False)):
                issues.append(f"Node {node_id}: ML models should be mocked in sandbox")
        
        return issues
    
    def get_sandbox_status(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get sandbox status for a plan"""
        sandbox_metadata = ir_data.get('metadata', {}).get('sandbox_mode', {})
        
        if not sandbox_metadata.get('enabled', False):
            return {
                'sandbox_enabled': False,
                'sandbox_level': 'none',
                'production_mode': True
            }
        
        return {
            'sandbox_enabled': True,
            'sandbox_level': sandbox_metadata.get('sandbox_level'),
            'restrictions': sandbox_metadata.get('restrictions', []),
            'resource_limits': sandbox_metadata.get('resource_limits', {}),
            'max_execution_time_minutes': sandbox_metadata.get('max_execution_time_minutes'),
            'data_access_scope': sandbox_metadata.get('data_access_scope'),
            'enabled_at': sandbox_metadata.get('enabled_at'),
            'production_mode': False
        }
    
    def disable_sandbox_mode(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Disable sandbox mode and remove restrictions"""
        
        # Remove sandbox metadata
        if 'metadata' in ir_data and 'sandbox_mode' in ir_data['metadata']:
            del ir_data['metadata']['sandbox_mode']
        
        # Remove sandbox attributes from nodes
        nodes = ir_data.get('nodes', [])
        for node in nodes:
            sandbox_keys = [
                'sandbox_disabled', 'sandbox_reason', 'sandbox_mock_model',
                'mock_model_config', 'sandbox_read_only', 'sandbox_data_scope',
                'sandbox_resource_limits'
            ]
            
            for key in sandbox_keys:
                if key in node:
                    del node[key]
        
        return ir_data
    
    def create_sandbox_runtime_config(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create runtime configuration for sandbox execution"""
        sandbox_status = self.get_sandbox_status(ir_data)
        
        if not sandbox_status['sandbox_enabled']:
            return {'sandbox_mode': False}
        
        return {
            'sandbox_mode': True,
            'sandbox_level': sandbox_status['sandbox_level'],
            'resource_limits': sandbox_status.get('resource_limits', {}),
            'execution_timeout_minutes': sandbox_status.get('max_execution_time_minutes', 30),
            'data_access_scope': sandbox_status.get('data_access_scope', 'isolated'),
            'external_calls_allowed': 'no_external_calls' not in sandbox_status.get('restrictions', []),
            'notifications_allowed': 'no_notifications' not in sandbox_status.get('restrictions', []),
            'ml_models_mocked': 'mock_ml_models' in sandbox_status.get('restrictions', [])
        }
    
    def get_available_sandbox_levels(self) -> Dict[str, Dict[str, Any]]:
        """Get available sandbox levels and their descriptions"""
        return {
            level.value: {
                'name': level.value.title(),
                'description': self._get_sandbox_level_description(level),
                'restrictions': [r.value for r in config.restrictions],
                'resource_limits': config.resource_limits,
                'max_execution_time_minutes': config.max_execution_time_minutes
            }
            for level, config in self.sandbox_configs.items()
        }
    
    def _get_sandbox_level_description(self, level: SandboxLevel) -> str:
        """Get description for sandbox level"""
        descriptions = {
            SandboxLevel.NONE: "Production mode with no restrictions",
            SandboxLevel.BASIC: "Basic sandbox with limited external access",
            SandboxLevel.STRICT: "Strict sandbox with full isolation",
            SandboxLevel.DEVELOPMENT: "Development sandbox for testing"
        }
        return descriptions.get(level, "Unknown sandbox level")
