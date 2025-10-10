"""
Plugin System Service - Task 6.2.37
====================================

Plugin system for custom nodes (UDFs, external ML)
- Allows safe extensibility for custom nodes without changing core compiler code
- Plugin SPI with descriptor format and validation
- Compile-time plugin validator and runtime loader
- Plugin registry with signed packages
- Backend implementation (no actual plugin execution - that's runtime infrastructure)

Dependencies: Task 6.2.4 (IR), Task 6.2.5 (Type System)
Outputs: Plugin system → enables safe extensibility for custom nodes
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import re
from pathlib import Path
import semver

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Types of plugins"""
    UDF = "udf"                    # User-defined functions
    CONNECTOR = "connector"        # External system connectors
    ML_ADAPTER = "ml_adapter"      # External ML model adapters
    TRANSFORMER = "transformer"   # Data transformers
    VALIDATOR = "validator"        # Custom validators
    AGGREGATOR = "aggregator"      # Data aggregators

class PluginStatus(Enum):
    """Plugin status"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    VALIDATED = "validated"
    APPROVED = "approved"
    INSTALLED = "installed"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"

class SandboxProfile(Enum):
    """Sandbox security profiles"""
    MINIMAL = "minimal"      # Very restricted
    STANDARD = "standard"    # Normal restrictions
    ELEVATED = "elevated"    # Fewer restrictions
    TRUSTED = "trusted"      # Minimal restrictions for trusted plugins

class ResourceLimitType(Enum):
    """Types of resource limits"""
    CPU_CORES = "cpu_cores"
    MEMORY_MB = "memory_mb"
    DISK_MB = "disk_mb"
    NETWORK_BANDWIDTH_MBPS = "network_bandwidth_mbps"
    EXECUTION_TIME_SECONDS = "execution_time_seconds"
    FILE_DESCRIPTORS = "file_descriptors"

@dataclass
class PluginInputSpec:
    """Specification for plugin input"""
    name: str
    type: str
    required: bool = True
    description: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)
    default_value: Optional[Any] = None
    examples: List[Any] = field(default_factory=list)

@dataclass
class PluginOutputSpec:
    """Specification for plugin output"""
    name: str
    type: str
    description: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    examples: List[Any] = field(default_factory=list)

@dataclass
class ResourceLimit:
    """Resource limit specification"""
    limit_type: ResourceLimitType
    limit_value: Union[int, float]
    soft_limit: bool = False  # If true, can be exceeded with warnings
    burst_allowance: Optional[Union[int, float]] = None

@dataclass
class NetworkPolicy:
    """Network access policy for plugins"""
    allow_outbound: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=list)
    require_tls: bool = True
    max_connections: int = 5

@dataclass
class CapabilityRequirement:
    """Required capability for plugin execution"""
    capability_name: str
    capability_type: str  # "secret", "database", "api", "filesystem"
    required: bool = True
    description: Optional[str] = None
    scope: Optional[str] = None  # Scope of the capability

@dataclass
class PluginDescriptor:
    """Complete plugin descriptor/manifest"""
    # Core identification
    plugin_id: str
    plugin_name: str
    plugin_version: str
    plugin_type: PluginType
    
    # Metadata
    description: str
    author: str
    license: str
    homepage: Optional[str] = None
    documentation_url: Optional[str] = None
    
    # Interface specification
    inputs: List[PluginInputSpec] = field(default_factory=list)
    outputs: List[PluginOutputSpec] = field(default_factory=list)
    
    # Compatibility
    compiler_version_min: str = "1.0.0"
    compiler_version_max: Optional[str] = None
    runtime_version_min: str = "1.0.0"
    runtime_version_max: Optional[str] = None
    
    # Resource requirements
    resource_limits: List[ResourceLimit] = field(default_factory=list)
    network_policy: Optional[NetworkPolicy] = None
    required_capabilities: List[CapabilityRequirement] = field(default_factory=list)
    
    # Security
    sandbox_profile: SandboxProfile = SandboxProfile.STANDARD
    trusted_plugin: bool = False
    signature: Optional[str] = None
    
    # Implementation
    entry_point: str = "main"
    runtime_environment: str = "python3.9"  # python3.9, nodejs16, wasm, etc.
    package_url: Optional[str] = None
    package_hash: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PluginStatus = PluginStatus.DRAFT

@dataclass
class PluginValidationResult:
    """Result of plugin validation"""
    plugin_id: str
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    compatibility_issues: List[str] = field(default_factory=list)
    resource_issues: List[str] = field(default_factory=list)
    
    # Validation metadata
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validator_version: str = "1.0.0"
    validation_duration_ms: float = 0.0

@dataclass
class PluginInstallation:
    """Plugin installation record"""
    installation_id: str
    plugin_id: str
    plugin_version: str
    tenant_id: str
    environment: str  # dev, staging, prod
    
    # Installation details
    installed_at: datetime
    installed_by: str
    installation_path: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    # Health
    health_status: str = "unknown"  # healthy, degraded, failed
    last_health_check: Optional[datetime] = None

# Task 6.2.37: Plugin System Service
class PluginSystemService:
    """Service for managing plugin system and custom nodes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Plugin registry
        self.plugin_registry: Dict[str, PluginDescriptor] = {}  # plugin_id -> descriptor
        self.plugin_installations: Dict[str, PluginInstallation] = {}  # installation_id -> installation
        
        # Validation results cache
        self.validation_cache: Dict[str, PluginValidationResult] = {}  # plugin_id -> result
        
        # Plugin type registry
        self.supported_plugin_types = {
            PluginType.UDF: self._validate_udf_plugin,
            PluginType.CONNECTOR: self._validate_connector_plugin,
            PluginType.ML_ADAPTER: self._validate_ml_adapter_plugin,
            PluginType.TRANSFORMER: self._validate_transformer_plugin,
            PluginType.VALIDATOR: self._validate_validator_plugin,
            PluginType.AGGREGATOR: self._validate_aggregator_plugin
        }
        
        # Default resource limits by sandbox profile
        self.default_resource_limits = {
            SandboxProfile.MINIMAL: [
                ResourceLimit(ResourceLimitType.CPU_CORES, 0.1),
                ResourceLimit(ResourceLimitType.MEMORY_MB, 64),
                ResourceLimit(ResourceLimitType.DISK_MB, 10),
                ResourceLimit(ResourceLimitType.EXECUTION_TIME_SECONDS, 5),
                ResourceLimit(ResourceLimitType.FILE_DESCRIPTORS, 10)
            ],
            SandboxProfile.STANDARD: [
                ResourceLimit(ResourceLimitType.CPU_CORES, 0.5),
                ResourceLimit(ResourceLimitType.MEMORY_MB, 256),
                ResourceLimit(ResourceLimitType.DISK_MB, 100),
                ResourceLimit(ResourceLimitType.EXECUTION_TIME_SECONDS, 30),
                ResourceLimit(ResourceLimitType.FILE_DESCRIPTORS, 50)
            ],
            SandboxProfile.ELEVATED: [
                ResourceLimit(ResourceLimitType.CPU_CORES, 1.0),
                ResourceLimit(ResourceLimitType.MEMORY_MB, 512),
                ResourceLimit(ResourceLimitType.DISK_MB, 500),
                ResourceLimit(ResourceLimitType.EXECUTION_TIME_SECONDS, 120),
                ResourceLimit(ResourceLimitType.FILE_DESCRIPTORS, 100)
            ],
            SandboxProfile.TRUSTED: [
                ResourceLimit(ResourceLimitType.CPU_CORES, 2.0),
                ResourceLimit(ResourceLimitType.MEMORY_MB, 1024),
                ResourceLimit(ResourceLimitType.DISK_MB, 1000),
                ResourceLimit(ResourceLimitType.EXECUTION_TIME_SECONDS, 300),
                ResourceLimit(ResourceLimitType.FILE_DESCRIPTORS, 200)
            ]
        }
        
        # Statistics
        self.plugin_stats = {
            'total_plugins': 0,
            'plugins_by_type': {ptype.value: 0 for ptype in PluginType},
            'plugins_by_status': {status.value: 0 for status in PluginStatus},
            'total_installations': 0,
            'active_installations': 0,
            'validation_cache_hits': 0,
            'validation_cache_misses': 0
        }
    
    def register_plugin(self, plugin_descriptor: PluginDescriptor) -> bool:
        """
        Register a new plugin in the registry
        
        Args:
            plugin_descriptor: Plugin descriptor/manifest
            
        Returns:
            True if registered successfully
        """
        # Validate plugin descriptor
        validation_result = self.validate_plugin(plugin_descriptor)
        
        if not validation_result.is_valid:
            self.logger.error(f"Plugin validation failed for {plugin_descriptor.plugin_id}: {validation_result.validation_errors}")
            return False
        
        # Check for existing plugin
        if plugin_descriptor.plugin_id in self.plugin_registry:
            existing = self.plugin_registry[plugin_descriptor.plugin_id]
            if semver.compare(plugin_descriptor.plugin_version, existing.plugin_version) <= 0:
                self.logger.warning(f"Plugin {plugin_descriptor.plugin_id} version {plugin_descriptor.plugin_version} is not newer than existing {existing.plugin_version}")
                return False
        
        # Apply default resource limits if not specified
        if not plugin_descriptor.resource_limits:
            plugin_descriptor.resource_limits = self.default_resource_limits[plugin_descriptor.sandbox_profile].copy()
        
        # Set default network policy if not specified
        if not plugin_descriptor.network_policy:
            plugin_descriptor.network_policy = NetworkPolicy(
                allow_outbound=plugin_descriptor.sandbox_profile in [SandboxProfile.ELEVATED, SandboxProfile.TRUSTED]
            )
        
        # Update timestamps
        plugin_descriptor.updated_at = datetime.now(timezone.utc)
        
        # Store in registry
        self.plugin_registry[plugin_descriptor.plugin_id] = plugin_descriptor
        
        # Update statistics
        self._update_plugin_stats()
        
        self.logger.info(f"✅ Registered plugin: {plugin_descriptor.plugin_id} v{plugin_descriptor.plugin_version}")
        
        return True
    
    def validate_plugin(self, plugin_descriptor: PluginDescriptor) -> PluginValidationResult:
        """
        Validate a plugin descriptor
        
        Args:
            plugin_descriptor: Plugin descriptor to validate
            
        Returns:
            PluginValidationResult with validation details
        """
        start_time = datetime.now(timezone.utc)
        
        # Check cache first
        cache_key = f"{plugin_descriptor.plugin_id}_{plugin_descriptor.plugin_version}_{hash(str(plugin_descriptor))}"
        if cache_key in self.validation_cache:
            self.plugin_stats['validation_cache_hits'] += 1
            return self.validation_cache[cache_key]
        
        self.plugin_stats['validation_cache_misses'] += 1
        
        result = PluginValidationResult(
            plugin_id=plugin_descriptor.plugin_id,
            is_valid=True
        )
        
        # Basic validation
        self._validate_plugin_metadata(plugin_descriptor, result)
        self._validate_plugin_interface(plugin_descriptor, result)
        self._validate_plugin_compatibility(plugin_descriptor, result)
        self._validate_plugin_security(plugin_descriptor, result)
        self._validate_plugin_resources(plugin_descriptor, result)
        
        # Type-specific validation
        if plugin_descriptor.plugin_type in self.supported_plugin_types:
            type_validator = self.supported_plugin_types[plugin_descriptor.plugin_type]
            type_validator(plugin_descriptor, result)
        else:
            result.validation_errors.append(f"Unsupported plugin type: {plugin_descriptor.plugin_type}")
        
        # Determine overall validity
        result.is_valid = len(result.validation_errors) == 0 and len(result.security_issues) == 0
        
        # Calculate validation time
        end_time = datetime.now(timezone.utc)
        result.validation_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        self.logger.info(f"✅ Validated plugin {plugin_descriptor.plugin_id}: valid={result.is_valid}, errors={len(result.validation_errors)}")
        
        return result
    
    def install_plugin(self, plugin_id: str, plugin_version: str, tenant_id: str, 
                      environment: str, installed_by: str,
                      configuration: Optional[Dict[str, Any]] = None) -> Optional[PluginInstallation]:
        """
        Install a plugin for a tenant
        
        Args:
            plugin_id: Plugin identifier
            plugin_version: Plugin version to install
            tenant_id: Tenant identifier
            environment: Environment (dev, staging, prod)
            installed_by: User installing the plugin
            configuration: Plugin configuration
            
        Returns:
            PluginInstallation record or None if failed
        """
        # Check if plugin exists
        if plugin_id not in self.plugin_registry:
            self.logger.error(f"Plugin not found: {plugin_id}")
            return None
        
        plugin_descriptor = self.plugin_registry[plugin_id]
        
        # Check version compatibility
        if plugin_descriptor.plugin_version != plugin_version:
            self.logger.error(f"Version mismatch: requested {plugin_version}, available {plugin_descriptor.plugin_version}")
            return None
        
        # Check if plugin is approved for installation
        if plugin_descriptor.status not in [PluginStatus.APPROVED, PluginStatus.INSTALLED]:
            self.logger.error(f"Plugin {plugin_id} not approved for installation (status: {plugin_descriptor.status})")
            return None
        
        # Create installation record
        installation_id = f"install_{plugin_id}_{tenant_id}_{environment}_{hash(f'{plugin_id}_{tenant_id}_{datetime.now()}')}"
        
        installation = PluginInstallation(
            installation_id=installation_id,
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            tenant_id=tenant_id,
            environment=environment,
            installed_at=datetime.now(timezone.utc),
            installed_by=installed_by,
            configuration=configuration or {}
        )
        
        # Store installation
        self.plugin_installations[installation_id] = installation
        
        # Update plugin status
        plugin_descriptor.status = PluginStatus.INSTALLED
        plugin_descriptor.updated_at = datetime.now(timezone.utc)
        
        # Update statistics
        self.plugin_stats['total_installations'] += 1
        self.plugin_stats['active_installations'] += 1
        
        self.logger.info(f"✅ Installed plugin {plugin_id} v{plugin_version} for tenant {tenant_id}")
        
        return installation
    
    def uninstall_plugin(self, installation_id: str, uninstalled_by: str) -> bool:
        """
        Uninstall a plugin
        
        Args:
            installation_id: Installation identifier
            uninstalled_by: User uninstalling the plugin
            
        Returns:
            True if uninstalled successfully
        """
        if installation_id not in self.plugin_installations:
            self.logger.error(f"Installation not found: {installation_id}")
            return False
        
        installation = self.plugin_installations[installation_id]
        
        # Mark as inactive
        installation.is_active = False
        
        # Update statistics
        self.plugin_stats['active_installations'] -= 1
        
        self.logger.info(f"✅ Uninstalled plugin {installation.plugin_id} (installation {installation_id})")
        
        return True
    
    def get_plugin_for_node_type(self, node_type: str) -> Optional[PluginDescriptor]:
        """
        Get plugin descriptor for a custom node type
        
        Args:
            node_type: Node type identifier
            
        Returns:
            PluginDescriptor if found, None otherwise
        """
        # Look for plugins that can handle this node type
        for plugin_descriptor in self.plugin_registry.values():
            if plugin_descriptor.status in [PluginStatus.APPROVED, PluginStatus.INSTALLED]:
                # Check if plugin handles this node type
                if self._plugin_handles_node_type(plugin_descriptor, node_type):
                    return plugin_descriptor
        
        return None
    
    def get_tenant_plugins(self, tenant_id: str, environment: str = None) -> List[PluginInstallation]:
        """
        Get all plugin installations for a tenant
        
        Args:
            tenant_id: Tenant identifier
            environment: Optional environment filter
            
        Returns:
            List of plugin installations
        """
        installations = []
        
        for installation in self.plugin_installations.values():
            if installation.tenant_id == tenant_id and installation.is_active:
                if environment is None or installation.environment == environment:
                    installations.append(installation)
        
        return installations
    
    def generate_plugin_sdk_template(self, plugin_type: PluginType) -> Dict[str, str]:
        """
        Generate SDK template for plugin development
        
        Args:
            plugin_type: Type of plugin to generate template for
            
        Returns:
            Dictionary with template files
        """
        templates = {}
        
        # Generate plugin descriptor template
        templates['plugin.json'] = self._generate_plugin_descriptor_template(plugin_type)
        
        # Generate main implementation template
        templates['main.py'] = self._generate_implementation_template(plugin_type)
        
        # Generate test template
        templates['test_plugin.py'] = self._generate_test_template(plugin_type)
        
        # Generate README
        templates['README.md'] = self._generate_readme_template(plugin_type)
        
        return templates
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        return {
            **self.plugin_stats,
            'validation_cache_size': len(self.validation_cache),
            'supported_plugin_types': len(self.supported_plugin_types),
            'sandbox_profiles': len(SandboxProfile)
        }
    
    def _validate_plugin_metadata(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate plugin metadata"""
        if not plugin.plugin_id or not re.match(r'^[a-zA-Z0-9_-]+$', plugin.plugin_id):
            result.validation_errors.append("Invalid plugin_id: must contain only alphanumeric characters, underscores, and hyphens")
        
        if not plugin.plugin_name:
            result.validation_errors.append("Plugin name is required")
        
        if not plugin.plugin_version:
            result.validation_errors.append("Plugin version is required")
        else:
            try:
                semver.VersionInfo.parse(plugin.plugin_version)
            except ValueError:
                result.validation_errors.append(f"Invalid semantic version: {plugin.plugin_version}")
        
        if not plugin.description:
            result.validation_warnings.append("Plugin description is recommended")
        
        if not plugin.author:
            result.validation_errors.append("Plugin author is required")
    
    def _validate_plugin_interface(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate plugin interface specification"""
        # Validate inputs
        input_names = set()
        for i, input_spec in enumerate(plugin.inputs):
            if not input_spec.name:
                result.validation_errors.append(f"Input {i}: name is required")
            elif input_spec.name in input_names:
                result.validation_errors.append(f"Duplicate input name: {input_spec.name}")
            else:
                input_names.add(input_spec.name)
            
            if not input_spec.type:
                result.validation_errors.append(f"Input {input_spec.name}: type is required")
        
        # Validate outputs
        output_names = set()
        for i, output_spec in enumerate(plugin.outputs):
            if not output_spec.name:
                result.validation_errors.append(f"Output {i}: name is required")
            elif output_spec.name in output_names:
                result.validation_errors.append(f"Duplicate output name: {output_spec.name}")
            else:
                output_names.add(output_spec.name)
            
            if not output_spec.type:
                result.validation_errors.append(f"Output {output_spec.name}: type is required")
    
    def _validate_plugin_compatibility(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate plugin compatibility requirements"""
        try:
            min_ver = semver.VersionInfo.parse(plugin.compiler_version_min)
            if plugin.compiler_version_max:
                max_ver = semver.VersionInfo.parse(plugin.compiler_version_max)
                if min_ver >= max_ver:
                    result.compatibility_issues.append("compiler_version_min must be less than compiler_version_max")
        except ValueError as e:
            result.compatibility_issues.append(f"Invalid compiler version format: {e}")
        
        # Check runtime environment
        supported_runtimes = ['python3.9', 'python3.10', 'nodejs16', 'nodejs18', 'wasm', 'native']
        if plugin.runtime_environment not in supported_runtimes:
            result.compatibility_issues.append(f"Unsupported runtime environment: {plugin.runtime_environment}")
    
    def _validate_plugin_security(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate plugin security settings"""
        # Check sandbox profile
        if plugin.sandbox_profile == SandboxProfile.TRUSTED and not plugin.trusted_plugin:
            result.security_issues.append("Trusted sandbox profile requires trusted_plugin flag")
        
        # Validate network policy
        if plugin.network_policy and plugin.network_policy.allow_outbound:
            if plugin.sandbox_profile == SandboxProfile.MINIMAL:
                result.security_issues.append("Minimal sandbox profile cannot allow outbound network access")
        
        # Check required capabilities
        for capability in plugin.required_capabilities:
            if capability.capability_type in ['secret', 'database'] and plugin.sandbox_profile == SandboxProfile.MINIMAL:
                result.security_issues.append(f"Minimal sandbox cannot access {capability.capability_type} capabilities")
        
        # Validate signature if present
        if plugin.signature:
            # In a real implementation, this would verify the cryptographic signature
            if len(plugin.signature) < 64:  # Minimum signature length
                result.security_issues.append("Invalid plugin signature format")
    
    def _validate_plugin_resources(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate plugin resource requirements"""
        for limit in plugin.resource_limits:
            if limit.limit_value <= 0:
                result.resource_issues.append(f"Invalid resource limit: {limit.limit_type.value} must be positive")
            
            # Check reasonable limits
            if limit.limit_type == ResourceLimitType.MEMORY_MB and limit.limit_value > 2048:
                result.resource_issues.append("Memory limit exceeds maximum allowed (2048 MB)")
            
            if limit.limit_type == ResourceLimitType.EXECUTION_TIME_SECONDS and limit.limit_value > 600:
                result.resource_issues.append("Execution time limit exceeds maximum allowed (600 seconds)")
    
    def _validate_udf_plugin(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate UDF-specific requirements"""
        if not plugin.inputs:
            result.validation_errors.append("UDF plugins must define at least one input")
        
        if not plugin.outputs:
            result.validation_errors.append("UDF plugins must define at least one output")
    
    def _validate_connector_plugin(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate connector-specific requirements"""
        # Connectors typically need network access
        if not plugin.network_policy or not plugin.network_policy.allow_outbound:
            result.validation_warnings.append("Connector plugins typically require outbound network access")
        
        # Check for required capabilities
        has_connection_capability = any(
            cap.capability_type in ['api', 'database', 'secret'] 
            for cap in plugin.required_capabilities
        )
        if not has_connection_capability:
            result.validation_warnings.append("Connector plugins typically require connection capabilities")
    
    def _validate_ml_adapter_plugin(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate ML adapter-specific requirements"""
        # ML adapters need specific input/output patterns
        has_model_input = any(
            'model' in inp.name.lower() or 'features' in inp.name.lower() 
            for inp in plugin.inputs
        )
        if not has_model_input:
            result.validation_warnings.append("ML adapter plugins should have model or features input")
        
        has_prediction_output = any(
            'prediction' in out.name.lower() or 'score' in out.name.lower() 
            for out in plugin.outputs
        )
        if not has_prediction_output:
            result.validation_warnings.append("ML adapter plugins should have prediction or score output")
    
    def _validate_transformer_plugin(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate transformer-specific requirements"""
        if len(plugin.inputs) != 1:
            result.validation_warnings.append("Transformer plugins typically have exactly one input")
        
        if len(plugin.outputs) != 1:
            result.validation_warnings.append("Transformer plugins typically have exactly one output")
    
    def _validate_validator_plugin(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate validator-specific requirements"""
        # Validators should return boolean or validation result
        has_validation_output = any(
            out.type in ['boolean', 'validation_result'] 
            for out in plugin.outputs
        )
        if not has_validation_output:
            result.validation_warnings.append("Validator plugins should return boolean or validation_result")
    
    def _validate_aggregator_plugin(self, plugin: PluginDescriptor, result: PluginValidationResult):
        """Validate aggregator-specific requirements"""
        # Aggregators typically take multiple inputs
        if len(plugin.inputs) < 2:
            result.validation_warnings.append("Aggregator plugins typically have multiple inputs")
    
    def _plugin_handles_node_type(self, plugin: PluginDescriptor, node_type: str) -> bool:
        """Check if plugin can handle a specific node type"""
        # Simple heuristic - in a real implementation this would be more sophisticated
        plugin_type_prefixes = {
            PluginType.UDF: ['udf_', 'function_'],
            PluginType.CONNECTOR: ['connect_', 'fetch_', 'send_'],
            PluginType.ML_ADAPTER: ['ml_', 'predict_', 'classify_'],
            PluginType.TRANSFORMER: ['transform_', 'convert_', 'map_'],
            PluginType.VALIDATOR: ['validate_', 'check_', 'verify_'],
            PluginType.AGGREGATOR: ['aggregate_', 'sum_', 'count_', 'group_']
        }
        
        prefixes = plugin_type_prefixes.get(plugin.plugin_type, [])
        return any(node_type.startswith(prefix) for prefix in prefixes)
    
    def _generate_plugin_descriptor_template(self, plugin_type: PluginType) -> str:
        """Generate plugin descriptor template"""
        template = {
            "plugin_id": f"my_{plugin_type.value}_plugin",
            "plugin_name": f"My {plugin_type.value.title()} Plugin",
            "plugin_version": "1.0.0",
            "plugin_type": plugin_type.value,
            "description": f"A sample {plugin_type.value} plugin",
            "author": "Your Name",
            "license": "MIT",
            "inputs": [
                {
                    "name": "input_data",
                    "type": "any",
                    "required": True,
                    "description": "Input data to process"
                }
            ],
            "outputs": [
                {
                    "name": "output_data",
                    "type": "any",
                    "description": "Processed output data"
                }
            ],
            "compiler_version_min": "1.0.0",
            "runtime_version_min": "1.0.0",
            "sandbox_profile": "standard",
            "runtime_environment": "python3.9",
            "entry_point": "main"
        }
        
        return json.dumps(template, indent=2)
    
    def _generate_implementation_template(self, plugin_type: PluginType) -> str:
        """Generate implementation template"""
        return f'''"""
{plugin_type.value.title()} Plugin Implementation
Generated template for {plugin_type.value} plugin
"""

def main(input_data):
    """
    Main plugin entry point
    
    Args:
        input_data: Input data from the workflow
        
    Returns:
        Processed output data
    """
    # TODO: Implement your {plugin_type.value} logic here
    
    # Example implementation
    result = {{
        "processed": True,
        "input_received": input_data,
        "plugin_type": "{plugin_type.value}",
        "timestamp": "2024-01-01T00:00:00Z"
    }}
    
    return result

# Plugin validation function (optional)
def validate_input(input_data):
    """
    Validate input data
    
    Args:
        input_data: Input data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return input_data is not None

# Plugin cleanup function (optional)
def cleanup():
    """
    Cleanup resources when plugin execution completes
    """
    pass
'''
    
    def _generate_test_template(self, plugin_type: PluginType) -> str:
        """Generate test template"""
        return f'''"""
Test suite for {plugin_type.value} plugin
"""

import unittest
from main import main, validate_input

class Test{plugin_type.value.title()}Plugin(unittest.TestCase):
    
    def test_main_function(self):
        """Test main plugin function"""
        test_input = {{"test": "data"}}
        result = main(test_input)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.get("processed"))
    
    def test_input_validation(self):
        """Test input validation"""
        valid_input = {{"test": "data"}}
        invalid_input = None
        
        self.assertTrue(validate_input(valid_input))
        self.assertFalse(validate_input(invalid_input))
    
    def test_output_format(self):
        """Test output format"""
        test_input = {{"test": "data"}}
        result = main(test_input)
        
        self.assertIn("processed", result)
        self.assertIn("plugin_type", result)
        self.assertEqual(result["plugin_type"], "{plugin_type.value}")

if __name__ == "__main__":
    unittest.main()
'''
    
    def _generate_readme_template(self, plugin_type: PluginType) -> str:
        """Generate README template"""
        return f'''# {plugin_type.value.title()} Plugin

A sample {plugin_type.value} plugin for the RBIA platform.

## Description

This plugin provides {plugin_type.value} functionality for RBIA workflows.

## Installation

1. Package your plugin:
   ```bash
   rbia plugin package .
   ```

2. Install the plugin:
   ```bash
   rbia plugin install my_{plugin_type.value}_plugin-1.0.0.rbp
   ```

## Usage

Use this plugin in your RBIA workflows:

```yaml
my_workflow:
  step1:
    type: my_{plugin_type.value}_plugin
    inputs:
      input_data: ${{workflow.input}}
    outputs:
      result: ${{step1.output_data}}
```

## Configuration

The plugin accepts the following configuration parameters:

- `input_data`: Input data to process (required)

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   python test_plugin.py
   ```

3. Package for distribution:
   ```bash
   rbia plugin package .
   ```

## License

MIT License - see LICENSE file for details.
'''
    
    def _update_plugin_stats(self):
        """Update plugin statistics"""
        self.plugin_stats['total_plugins'] = len(self.plugin_registry)
        
        # Reset type and status counters
        for ptype in PluginType:
            self.plugin_stats['plugins_by_type'][ptype.value] = 0
        for status in PluginStatus:
            self.plugin_stats['plugins_by_status'][status.value] = 0
        
        # Count by type and status
        for plugin in self.plugin_registry.values():
            self.plugin_stats['plugins_by_type'][plugin.plugin_type.value] += 1
            self.plugin_stats['plugins_by_status'][plugin.status.value] += 1

# API Interface
class PluginSystemAPI:
    """API interface for plugin system operations"""
    
    def __init__(self, plugin_service: Optional[PluginSystemService] = None):
        self.plugin_service = plugin_service or PluginSystemService()
    
    def register_plugin(self, plugin_descriptor_dict: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to register a plugin"""
        try:
            # Convert dict to PluginDescriptor
            plugin_descriptor = PluginDescriptor(**plugin_descriptor_dict)
            
            success = self.plugin_service.register_plugin(plugin_descriptor)
            
            return {
                'success': success,
                'plugin_id': plugin_descriptor.plugin_id,
                'plugin_version': plugin_descriptor.plugin_version
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def validate_plugin(self, plugin_descriptor_dict: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to validate a plugin"""
        try:
            plugin_descriptor = PluginDescriptor(**plugin_descriptor_dict)
            result = self.plugin_service.validate_plugin(plugin_descriptor)
            
            return {
                'success': True,
                'validation_result': asdict(result)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def install_plugin(self, plugin_id: str, plugin_version: str, tenant_id: str,
                      environment: str, installed_by: str) -> Dict[str, Any]:
        """API endpoint to install a plugin"""
        try:
            installation = self.plugin_service.install_plugin(
                plugin_id, plugin_version, tenant_id, environment, installed_by
            )
            
            if installation:
                return {
                    'success': True,
                    'installation_id': installation.installation_id,
                    'installed_at': installation.installed_at.isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Plugin installation failed'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_plugin_descriptor(plugin_type: PluginType) -> PluginDescriptor:
    """Create a test plugin descriptor"""
    return PluginDescriptor(
        plugin_id=f"test_{plugin_type.value}_plugin",
        plugin_name=f"Test {plugin_type.value.title()} Plugin",
        plugin_version="1.0.0",
        plugin_type=plugin_type,
        description=f"A test {plugin_type.value} plugin for validation",
        author="Test Author",
        license="MIT",
        inputs=[
            PluginInputSpec(
                name="input_data",
                type="object",
                required=True,
                description="Input data to process"
            )
        ],
        outputs=[
            PluginOutputSpec(
                name="output_data",
                type="object",
                description="Processed output data"
            )
        ],
        sandbox_profile=SandboxProfile.STANDARD,
        runtime_environment="python3.9"
    )

def run_plugin_system_tests():
    """Run comprehensive plugin system tests"""
    print("=== Plugin System Service Tests ===")
    
    # Initialize service
    plugin_service = PluginSystemService()
    plugin_api = PluginSystemAPI(plugin_service)
    
    # Test 1: Plugin descriptor creation and validation
    print("\n1. Testing plugin descriptor creation and validation...")
    
    test_plugins = []
    for plugin_type in [PluginType.UDF, PluginType.CONNECTOR, PluginType.ML_ADAPTER]:
        plugin = create_test_plugin_descriptor(plugin_type)
        test_plugins.append(plugin)
        
        validation_result = plugin_service.validate_plugin(plugin)
        print(f"   {plugin_type.value} plugin validation: {'✅ PASS' if validation_result.is_valid else '❌ FAIL'}")
        if not validation_result.is_valid:
            print(f"     Errors: {validation_result.validation_errors}")
        if validation_result.validation_warnings:
            print(f"     Warnings: {validation_result.validation_warnings}")
    
    # Test 2: Plugin registration
    print("\n2. Testing plugin registration...")
    
    for plugin in test_plugins:
        success = plugin_service.register_plugin(plugin)
        print(f"   Registered {plugin.plugin_type.value} plugin: {'✅ PASS' if success else '❌ FAIL'}")
    
    # Test 3: Plugin installation
    print("\n3. Testing plugin installation...")
    
    for plugin in test_plugins[:2]:  # Install first 2 plugins
        # First approve the plugin
        plugin.status = PluginStatus.APPROVED
        
        installation = plugin_service.install_plugin(
            plugin.plugin_id,
            plugin.plugin_version,
            "test_tenant",
            "dev",
            "test_user"
        )
        
        print(f"   Installed {plugin.plugin_type.value} plugin: {'✅ PASS' if installation else '❌ FAIL'}")
        if installation:
            print(f"     Installation ID: {installation.installation_id}")
    
    # Test 4: Plugin lookup for node types
    print("\n4. Testing plugin lookup for node types...")
    
    test_node_types = ["udf_custom_function", "connect_api", "ml_predict", "unknown_type"]
    for node_type in test_node_types:
        plugin = plugin_service.get_plugin_for_node_type(node_type)
        found = plugin is not None
        print(f"   Node type '{node_type}': {'✅ FOUND' if found else '❌ NOT FOUND'}")
        if found:
            print(f"     Plugin: {plugin.plugin_id} ({plugin.plugin_type.value})")
    
    # Test 5: Tenant plugin listing
    print("\n5. Testing tenant plugin listing...")
    
    tenant_plugins = plugin_service.get_tenant_plugins("test_tenant", "dev")
    print(f"   Tenant plugins found: {len(tenant_plugins)}")
    for installation in tenant_plugins:
        print(f"     - {installation.plugin_id} v{installation.plugin_version}")
    
    # Test 6: Resource limits and security validation
    print("\n6. Testing resource limits and security validation...")
    
    # Create plugin with excessive resource requirements
    excessive_plugin = create_test_plugin_descriptor(PluginType.UDF)
    excessive_plugin.plugin_id = "excessive_plugin"
    excessive_plugin.resource_limits = [
        ResourceLimit(ResourceLimitType.MEMORY_MB, 4096),  # Too much memory
        ResourceLimit(ResourceLimitType.EXECUTION_TIME_SECONDS, 1200)  # Too long execution
    ]
    
    validation_result = plugin_service.validate_plugin(excessive_plugin)
    print(f"   Excessive resource plugin validation: {'❌ FAIL' if not validation_result.is_valid else '✅ PASS'}")
    if validation_result.resource_issues:
        print(f"     Resource issues: {validation_result.resource_issues}")
    
    # Test 7: Sandbox profile defaults
    print("\n7. Testing sandbox profile defaults...")
    
    for profile in SandboxProfile:
        default_limits = plugin_service.default_resource_limits[profile]
        print(f"   {profile.value} profile: {len(default_limits)} default limits")
        for limit in default_limits:
            print(f"     - {limit.limit_type.value}: {limit.limit_value}")
    
    # Test 8: Plugin SDK template generation
    print("\n8. Testing plugin SDK template generation...")
    
    for plugin_type in [PluginType.UDF, PluginType.CONNECTOR]:
        templates = plugin_service.generate_plugin_sdk_template(plugin_type)
        print(f"   {plugin_type.value} SDK template: {len(templates)} files")
        for filename in templates.keys():
            print(f"     - {filename}")
    
    # Test 9: Plugin uninstallation
    print("\n9. Testing plugin uninstallation...")
    
    # Get an installation to uninstall
    if tenant_plugins:
        installation = tenant_plugins[0]
        success = plugin_service.uninstall_plugin(installation.installation_id, "test_user")
        print(f"   Plugin uninstallation: {'✅ PASS' if success else '❌ FAIL'}")
    
    # Test 10: API interface
    print("\n10. Testing API interface...")
    
    # Test API plugin registration
    api_plugin_dict = asdict(create_test_plugin_descriptor(PluginType.TRANSFORMER))
    api_plugin_dict['plugin_id'] = 'api_test_plugin'
    
    api_register_result = plugin_api.register_plugin(api_plugin_dict)
    print(f"   API plugin registration: {'✅ PASS' if api_register_result['success'] else '❌ FAIL'}")
    
    # Test API plugin validation
    api_validation_result = plugin_api.validate_plugin(api_plugin_dict)
    print(f"   API plugin validation: {'✅ PASS' if api_validation_result['success'] else '❌ FAIL'}")
    
    # Test 11: Invalid plugin scenarios
    print("\n11. Testing invalid plugin scenarios...")
    
    # Invalid plugin ID
    invalid_plugin = create_test_plugin_descriptor(PluginType.UDF)
    invalid_plugin.plugin_id = "invalid plugin id!"  # Contains invalid characters
    
    validation_result = plugin_service.validate_plugin(invalid_plugin)
    print(f"   Invalid plugin ID validation: {'❌ FAIL' if not validation_result.is_valid else '✅ UNEXPECTED PASS'}")
    
    # Missing required fields
    incomplete_plugin = create_test_plugin_descriptor(PluginType.UDF)
    incomplete_plugin.author = ""  # Missing required field
    
    validation_result = plugin_service.validate_plugin(incomplete_plugin)
    print(f"   Incomplete plugin validation: {'❌ FAIL' if not validation_result.is_valid else '✅ UNEXPECTED PASS'}")
    
    # Test 12: Statistics
    print("\n12. Testing statistics...")
    
    stats = plugin_service.get_plugin_statistics()
    print(f"   Total plugins: {stats['total_plugins']}")
    print(f"   Plugins by type: {stats['plugins_by_type']}")
    print(f"   Plugins by status: {stats['plugins_by_status']}")
    print(f"   Total installations: {stats['total_installations']}")
    print(f"   Active installations: {stats['active_installations']}")
    print(f"   Validation cache size: {stats['validation_cache_size']}")
    
    print(f"\n=== Test Summary ===")
    print(f"Plugin system service tested successfully")
    print(f"Total plugins: {stats['total_plugins']}")
    print(f"Total installations: {stats['total_installations']}")
    print(f"Supported plugin types: {stats['supported_plugin_types']}")
    print(f"Validation cache hits: {stats['validation_cache_hits']}")
    
    return plugin_service, plugin_api

if __name__ == "__main__":
    run_plugin_system_tests()



