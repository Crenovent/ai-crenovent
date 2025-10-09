"""
Version Negotiation Service - Task 6.2.36
==========================================

Version negotiation: runtime ↔ plan compatibility
- Ensures runtime can safely execute manifests from different compiler/runtime versions
- Explicit compatibility metadata and negotiation logic
- Runtime handshake protocol with fallback strategies
- Backend implementation (no actual runtime deployment - that's infrastructure)

Dependencies: Task 6.2.26 (Plan Manifest Generator)
Outputs: Version compatibility system → enables safe cross-version execution
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from packaging import version
import re

logger = logging.getLogger(__name__)

class CompatibilityLevel(Enum):
    """Levels of compatibility between versions"""
    FULLY_COMPATIBLE = "fully_compatible"
    COMPATIBLE_WITH_WARNINGS = "compatible_with_warnings"
    DEGRADED_COMPATIBILITY = "degraded_compatibility"
    INCOMPATIBLE = "incompatible"
    REQUIRES_MIGRATION = "requires_migration"

class NegotiationStrategy(Enum):
    """Strategies for handling version mismatches"""
    STRICT = "strict"                    # Fail on any incompatibility
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Use safe subset of features
    EMULATION = "emulation"              # Emulate older semantics
    FALLBACK_TO_LAST_GOOD = "fallback_to_last_good"  # Use last compatible plan
    FORCE_MIGRATION = "force_migration"  # Require plan migration

class VersionComponent(Enum):
    """Components of version compatibility"""
    COMPILER_VERSION = "compiler_version"
    IR_SCHEMA_VERSION = "ir_schema_version"
    RUNTIME_VERSION = "runtime_version"
    DSL_VERSION = "dsl_version"
    POLICY_VERSION = "policy_version"

@dataclass
class VersionRange:
    """Version range specification"""
    min_version: str
    max_version: Optional[str] = None
    exclude_versions: List[str] = field(default_factory=list)
    include_prereleases: bool = False

@dataclass
class CompatibilityMetadata:
    """Compatibility metadata for manifests"""
    # Core version information
    compiler_version: str
    ir_schema_version: str
    dsl_version: str
    
    # Runtime requirements
    required_runtime_min: str
    required_runtime_max: Optional[str] = None
    
    # Feature requirements
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)
    deprecated_features_used: List[str] = field(default_factory=list)
    
    # Compatibility flags
    backward_compatible: bool = True
    forward_compatible: bool = False
    breaking_changes: List[str] = field(default_factory=list)
    
    # Negotiation preferences
    preferred_negotiation_strategy: NegotiationStrategy = NegotiationStrategy.GRACEFUL_DEGRADATION
    fallback_strategies: List[NegotiationStrategy] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    compatibility_hash: Optional[str] = None

@dataclass
class RuntimeCapabilities:
    """Runtime capabilities and version information"""
    runtime_version: str
    runtime_name: str
    
    # Supported versions
    supported_compiler_versions: VersionRange
    supported_ir_schema_versions: VersionRange
    supported_dsl_versions: VersionRange
    
    # Feature support
    supported_features: List[str] = field(default_factory=list)
    experimental_features: List[str] = field(default_factory=list)
    deprecated_features: List[str] = field(default_factory=list)
    
    # Compatibility settings
    strict_compatibility: bool = False
    allow_degraded_execution: bool = True
    emulation_support: bool = True
    
    # Performance characteristics
    max_plan_size_mb: float = 100.0
    max_execution_time_minutes: int = 60
    
    # Metadata
    capabilities_version: str = "1.0"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class NegotiationResult:
    """Result of version negotiation"""
    negotiation_id: str
    compatibility_level: CompatibilityLevel
    negotiation_strategy: NegotiationStrategy
    
    # Version information
    manifest_compatibility: CompatibilityMetadata
    runtime_capabilities: RuntimeCapabilities
    
    # Negotiation outcome
    can_execute: bool
    execution_mode: str  # "normal", "degraded", "emulated", "fallback"
    
    # Warnings and issues
    compatibility_warnings: List[str] = field(default_factory=list)
    feature_limitations: List[str] = field(default_factory=list)
    performance_impacts: List[str] = field(default_factory=list)
    
    # Execution parameters
    execution_parameters: Dict[str, Any] = field(default_factory=dict)
    fallback_plan_id: Optional[str] = None
    
    # Metadata
    negotiated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    negotiation_time_ms: float = 0.0

@dataclass
class VersionMigrationPlan:
    """Plan for migrating between versions"""
    migration_id: str
    source_version: str
    target_version: str
    
    # Migration steps
    required_changes: List[str] = field(default_factory=list)
    optional_optimizations: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    
    # Migration complexity
    estimated_effort_hours: float = 0.0
    risk_level: str = "medium"
    rollback_available: bool = True
    
    # Validation
    validation_tests: List[str] = field(default_factory=list)
    compatibility_tests: List[str] = field(default_factory=list)

# Task 6.2.36: Version Negotiation Service
class VersionNegotiationService:
    """Service for managing version compatibility and negotiation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Version compatibility matrix
        self.compatibility_matrix: Dict[str, Dict[str, CompatibilityLevel]] = {}
        
        # Runtime registry
        self.registered_runtimes: Dict[str, RuntimeCapabilities] = {}
        
        # Negotiation history
        self.negotiation_history: Dict[str, NegotiationResult] = {}
        
        # Migration plans
        self.migration_plans: Dict[str, VersionMigrationPlan] = {}
        
        # Initialize compatibility matrix and runtime capabilities
        self._initialize_compatibility_matrix()
        self._initialize_default_runtime_capabilities()
        
        # Statistics
        self.negotiation_stats = {
            'total_negotiations': 0,
            'successful_negotiations': 0,
            'failed_negotiations': 0,
            'degraded_executions': 0,
            'fallback_executions': 0,
            'compatibility_levels': {level.value: 0 for level in CompatibilityLevel}
        }
    
    def create_compatibility_metadata(self, compiler_version: str, ir_schema_version: str,
                                    dsl_version: str, required_runtime_min: str,
                                    required_features: Optional[List[str]] = None,
                                    breaking_changes: Optional[List[str]] = None) -> CompatibilityMetadata:
        """
        Create compatibility metadata for a manifest
        
        Args:
            compiler_version: Version of compiler used
            ir_schema_version: IR schema version
            dsl_version: DSL version used
            required_runtime_min: Minimum runtime version required
            required_features: List of required features
            breaking_changes: List of breaking changes
            
        Returns:
            CompatibilityMetadata
        """
        metadata = CompatibilityMetadata(
            compiler_version=compiler_version,
            ir_schema_version=ir_schema_version,
            dsl_version=dsl_version,
            required_runtime_min=required_runtime_min,
            required_features=required_features or [],
            breaking_changes=breaking_changes or []
        )
        
        # Determine compatibility flags
        metadata.backward_compatible = len(metadata.breaking_changes) == 0
        metadata.forward_compatible = self._is_forward_compatible(metadata)
        
        # Set negotiation preferences
        if metadata.breaking_changes:
            metadata.preferred_negotiation_strategy = NegotiationStrategy.FORCE_MIGRATION
            metadata.fallback_strategies = [NegotiationStrategy.FALLBACK_TO_LAST_GOOD]
        else:
            metadata.preferred_negotiation_strategy = NegotiationStrategy.GRACEFUL_DEGRADATION
            metadata.fallback_strategies = [
                NegotiationStrategy.EMULATION,
                NegotiationStrategy.FALLBACK_TO_LAST_GOOD
            ]
        
        # Generate compatibility hash
        metadata.compatibility_hash = self._generate_compatibility_hash(metadata)
        
        self.logger.info(f"✅ Created compatibility metadata: compiler={compiler_version}, runtime_min={required_runtime_min}")
        
        return metadata
    
    def register_runtime_capabilities(self, runtime_name: str, runtime_version: str,
                                    supported_compiler_versions: VersionRange,
                                    supported_features: List[str]) -> RuntimeCapabilities:
        """
        Register runtime capabilities
        
        Args:
            runtime_name: Name of the runtime
            runtime_version: Version of the runtime
            supported_compiler_versions: Range of supported compiler versions
            supported_features: List of supported features
            
        Returns:
            RuntimeCapabilities
        """
        capabilities = RuntimeCapabilities(
            runtime_version=runtime_version,
            runtime_name=runtime_name,
            supported_compiler_versions=supported_compiler_versions,
            supported_ir_schema_versions=VersionRange(min_version="1.0.0", max_version="2.0.0"),
            supported_dsl_versions=VersionRange(min_version="1.0.0", max_version="2.0.0"),
            supported_features=supported_features
        )
        
        # Store capabilities
        runtime_key = f"{runtime_name}_{runtime_version}"
        self.registered_runtimes[runtime_key] = capabilities
        
        self.logger.info(f"✅ Registered runtime capabilities: {runtime_name} v{runtime_version}")
        
        return capabilities
    
    def negotiate_compatibility(self, compatibility_metadata: CompatibilityMetadata,
                              runtime_capabilities: RuntimeCapabilities,
                              negotiation_strategy: Optional[NegotiationStrategy] = None) -> NegotiationResult:
        """
        Negotiate compatibility between manifest and runtime
        
        Args:
            compatibility_metadata: Manifest compatibility metadata
            runtime_capabilities: Runtime capabilities
            negotiation_strategy: Preferred negotiation strategy
            
        Returns:
            NegotiationResult with negotiation outcome
        """
        start_time = datetime.now(timezone.utc)
        negotiation_id = f"neg_{hash(f'{compatibility_metadata.compiler_version}_{runtime_capabilities.runtime_version}_{start_time}')}"
        
        # Use provided strategy or metadata preference
        strategy = negotiation_strategy or compatibility_metadata.preferred_negotiation_strategy
        
        # Perform compatibility analysis
        compatibility_level = self._analyze_compatibility(compatibility_metadata, runtime_capabilities)
        
        # Create negotiation result
        result = NegotiationResult(
            negotiation_id=negotiation_id,
            compatibility_level=compatibility_level,
            negotiation_strategy=strategy,
            manifest_compatibility=compatibility_metadata,
            runtime_capabilities=runtime_capabilities
        )
        
        # Determine if execution is possible
        result.can_execute, result.execution_mode = self._determine_execution_capability(
            compatibility_level, strategy, compatibility_metadata, runtime_capabilities
        )
        
        # Generate warnings and limitations
        result.compatibility_warnings = self._generate_compatibility_warnings(
            compatibility_metadata, runtime_capabilities, compatibility_level
        )
        
        result.feature_limitations = self._identify_feature_limitations(
            compatibility_metadata, runtime_capabilities
        )
        
        result.performance_impacts = self._assess_performance_impacts(
            compatibility_level, strategy
        )
        
        # Set execution parameters
        result.execution_parameters = self._generate_execution_parameters(
            compatibility_level, strategy, compatibility_metadata, runtime_capabilities
        )
        
        # Handle fallback scenarios
        if not result.can_execute or result.execution_mode == "fallback":
            result.fallback_plan_id = self._identify_fallback_plan(
                compatibility_metadata, runtime_capabilities
            )
        
        # Calculate negotiation time
        end_time = datetime.now(timezone.utc)
        result.negotiation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Store result
        self.negotiation_history[negotiation_id] = result
        
        # Update statistics
        self._update_negotiation_stats(result)
        
        self.logger.info(f"✅ Negotiated compatibility: {compatibility_level.value}, can_execute={result.can_execute}")
        
        return result
    
    def check_version_compatibility(self, manifest_version: str, runtime_version: str,
                                  component: VersionComponent = VersionComponent.COMPILER_VERSION) -> CompatibilityLevel:
        """
        Check compatibility between two versions for a specific component
        
        Args:
            manifest_version: Version from manifest
            runtime_version: Runtime version
            component: Version component to check
            
        Returns:
            CompatibilityLevel
        """
        # Use compatibility matrix if available
        if manifest_version in self.compatibility_matrix:
            if runtime_version in self.compatibility_matrix[manifest_version]:
                return self.compatibility_matrix[manifest_version][runtime_version]
        
        # Fallback to semantic version analysis
        try:
            manifest_ver = version.parse(manifest_version)
            runtime_ver = version.parse(runtime_version)
            
            # Same version - fully compatible
            if manifest_ver == runtime_ver:
                return CompatibilityLevel.FULLY_COMPATIBLE
            
            # Major version differences - likely incompatible
            if manifest_ver.major != runtime_ver.major:
                return CompatibilityLevel.INCOMPATIBLE
            
            # Minor version differences - check direction
            if manifest_ver.minor != runtime_ver.minor:
                if manifest_ver.minor < runtime_ver.minor:
                    # Older manifest, newer runtime - usually compatible with warnings
                    return CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
                else:
                    # Newer manifest, older runtime - degraded compatibility
                    return CompatibilityLevel.DEGRADED_COMPATIBILITY
            
            # Patch version differences - usually compatible
            return CompatibilityLevel.FULLY_COMPATIBLE
            
        except Exception as e:
            self.logger.warning(f"Error parsing versions {manifest_version}/{runtime_version}: {e}")
            return CompatibilityLevel.INCOMPATIBLE
    
    def create_migration_plan(self, source_version: str, target_version: str,
                            component: VersionComponent = VersionComponent.COMPILER_VERSION) -> VersionMigrationPlan:
        """
        Create a migration plan between versions
        
        Args:
            source_version: Current version
            target_version: Target version
            component: Component being migrated
            
        Returns:
            VersionMigrationPlan
        """
        migration_id = f"mig_{hash(f'{source_version}_{target_version}_{component.value}')}"
        
        plan = VersionMigrationPlan(
            migration_id=migration_id,
            source_version=source_version,
            target_version=target_version
        )
        
        # Analyze version differences
        try:
            source_ver = version.parse(source_version)
            target_ver = version.parse(target_version)
            
            # Determine migration complexity
            if target_ver.major > source_ver.major:
                plan.risk_level = "high"
                plan.estimated_effort_hours = 40.0  # 1 week
                plan.required_changes.extend([
                    "Review breaking changes documentation",
                    "Update manifest schema",
                    "Test with new runtime version"
                ])
                plan.breaking_changes.append("Major version upgrade with breaking changes")
                plan.rollback_available = False
                
            elif target_ver.minor > source_ver.minor:
                plan.risk_level = "medium"
                plan.estimated_effort_hours = 8.0  # 1 day
                plan.required_changes.extend([
                    "Update version references",
                    "Review new features",
                    "Validate compatibility"
                ])
                plan.optional_optimizations.append("Adopt new features for improved performance")
                
            else:
                plan.risk_level = "low"
                plan.estimated_effort_hours = 2.0  # 2 hours
                plan.required_changes.append("Update version references")
            
            # Add validation tests
            plan.validation_tests = [
                "Compile with new version",
                "Run unit tests",
                "Validate runtime execution"
            ]
            
            plan.compatibility_tests = [
                "Cross-version compatibility test",
                "Feature compatibility validation",
                "Performance regression test"
            ]
            
        except Exception as e:
            self.logger.error(f"Error creating migration plan: {e}")
            plan.risk_level = "high"
            plan.estimated_effort_hours = 16.0
        
        # Store migration plan
        self.migration_plans[migration_id] = plan
        
        self.logger.info(f"✅ Created migration plan: {source_version} → {target_version} ({plan.risk_level} risk)")
        
        return plan
    
    def simulate_runtime_handshake(self, manifest_compatibility: CompatibilityMetadata,
                                 runtime_name: str, runtime_version: str) -> Dict[str, Any]:
        """
        Simulate runtime handshake protocol
        
        Args:
            manifest_compatibility: Manifest compatibility metadata
            runtime_name: Runtime name
            runtime_version: Runtime version
            
        Returns:
            Dictionary with handshake result
        """
        runtime_key = f"{runtime_name}_{runtime_version}"
        
        if runtime_key not in self.registered_runtimes:
            return {
                'success': False,
                'error': f'Unknown runtime: {runtime_name} v{runtime_version}',
                'action': 'reject'
            }
        
        runtime_capabilities = self.registered_runtimes[runtime_key]
        
        # Perform negotiation
        negotiation_result = self.negotiate_compatibility(
            manifest_compatibility, runtime_capabilities
        )
        
        # Determine handshake outcome
        if negotiation_result.can_execute:
            if negotiation_result.execution_mode == "normal":
                action = "accept"
            elif negotiation_result.execution_mode == "degraded":
                action = "accept_with_degradation"
            elif negotiation_result.execution_mode == "emulated":
                action = "accept_with_emulation"
            else:
                action = "fallback_to_last_good"
        else:
            action = "reject"
        
        return {
            'success': True,
            'negotiation_id': negotiation_result.negotiation_id,
            'compatibility_level': negotiation_result.compatibility_level.value,
            'action': action,
            'execution_mode': negotiation_result.execution_mode,
            'warnings': negotiation_result.compatibility_warnings,
            'limitations': negotiation_result.feature_limitations,
            'fallback_plan_id': negotiation_result.fallback_plan_id,
            'execution_parameters': negotiation_result.execution_parameters
        }
    
    def get_negotiation_statistics(self) -> Dict[str, Any]:
        """Get version negotiation statistics"""
        return {
            **self.negotiation_stats,
            'registered_runtimes': len(self.registered_runtimes),
            'compatibility_matrix_entries': sum(len(versions) for versions in self.compatibility_matrix.values()),
            'migration_plans_created': len(self.migration_plans)
        }
    
    def _initialize_compatibility_matrix(self):
        """Initialize version compatibility matrix"""
        # Define known compatibility relationships
        compatibility_data = [
            # Format: (manifest_version, runtime_version, compatibility_level)
            ("1.0.0", "1.0.0", CompatibilityLevel.FULLY_COMPATIBLE),
            ("1.0.0", "1.1.0", CompatibilityLevel.COMPATIBLE_WITH_WARNINGS),
            ("1.1.0", "1.0.0", CompatibilityLevel.DEGRADED_COMPATIBILITY),
            ("1.1.0", "1.1.0", CompatibilityLevel.FULLY_COMPATIBLE),
            ("1.0.0", "2.0.0", CompatibilityLevel.INCOMPATIBLE),
            ("2.0.0", "1.0.0", CompatibilityLevel.INCOMPATIBLE),
        ]
        
        for manifest_ver, runtime_ver, compat_level in compatibility_data:
            if manifest_ver not in self.compatibility_matrix:
                self.compatibility_matrix[manifest_ver] = {}
            self.compatibility_matrix[manifest_ver][runtime_ver] = compat_level
        
        self.logger.info(f"✅ Initialized compatibility matrix with {len(compatibility_data)} entries")
    
    def _initialize_default_runtime_capabilities(self):
        """Initialize default runtime capabilities"""
        default_runtimes = [
            {
                'name': 'rbia-runtime',
                'version': '1.0.0',
                'compiler_versions': VersionRange(min_version="1.0.0", max_version="1.2.0"),
                'features': ['basic_execution', 'ml_inference', 'policy_enforcement']
            },
            {
                'name': 'rbia-runtime',
                'version': '1.1.0',
                'compiler_versions': VersionRange(min_version="1.0.0", max_version="1.5.0"),
                'features': ['basic_execution', 'ml_inference', 'policy_enforcement', 'advanced_fallbacks', 'explainability']
            },
            {
                'name': 'rbia-runtime-lite',
                'version': '1.0.0',
                'compiler_versions': VersionRange(min_version="1.0.0", max_version="1.1.0"),
                'features': ['basic_execution', 'simple_ml']
            }
        ]
        
        for runtime_info in default_runtimes:
            self.register_runtime_capabilities(
                runtime_info['name'],
                runtime_info['version'],
                runtime_info['compiler_versions'],
                runtime_info['features']
            )
    
    def _is_forward_compatible(self, metadata: CompatibilityMetadata) -> bool:
        """Check if metadata indicates forward compatibility"""
        # Simple heuristic: no breaking changes and uses stable features
        return len(metadata.breaking_changes) == 0 and len(metadata.deprecated_features_used) == 0
    
    def _generate_compatibility_hash(self, metadata: CompatibilityMetadata) -> str:
        """Generate hash for compatibility metadata"""
        import hashlib
        
        hash_data = {
            'compiler_version': metadata.compiler_version,
            'ir_schema_version': metadata.ir_schema_version,
            'dsl_version': metadata.dsl_version,
            'required_features': sorted(metadata.required_features),
            'breaking_changes': sorted(metadata.breaking_changes)
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    def _analyze_compatibility(self, metadata: CompatibilityMetadata, 
                             capabilities: RuntimeCapabilities) -> CompatibilityLevel:
        """Analyze compatibility between metadata and capabilities"""
        # Check compiler version compatibility
        compiler_compat = self._check_version_in_range(
            metadata.compiler_version, capabilities.supported_compiler_versions
        )
        
        # Check IR schema compatibility
        ir_compat = self._check_version_in_range(
            metadata.ir_schema_version, capabilities.supported_ir_schema_versions
        )
        
        # Check DSL version compatibility
        dsl_compat = self._check_version_in_range(
            metadata.dsl_version, capabilities.supported_dsl_versions
        )
        
        # Check feature compatibility
        missing_features = set(metadata.required_features) - set(capabilities.supported_features)
        
        # Determine overall compatibility
        if not compiler_compat or not ir_compat or not dsl_compat:
            return CompatibilityLevel.INCOMPATIBLE
        
        if missing_features:
            if capabilities.allow_degraded_execution:
                return CompatibilityLevel.DEGRADED_COMPATIBILITY
            else:
                return CompatibilityLevel.INCOMPATIBLE
        
        if metadata.breaking_changes:
            return CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        
        return CompatibilityLevel.FULLY_COMPATIBLE
    
    def _check_version_in_range(self, version_str: str, version_range: VersionRange) -> bool:
        """Check if version is within range"""
        try:
            ver = version.parse(version_str)
            min_ver = version.parse(version_range.min_version)
            
            if ver < min_ver:
                return False
            
            if version_range.max_version:
                max_ver = version.parse(version_range.max_version)
                if ver > max_ver:
                    return False
            
            # Check excluded versions
            for excluded in version_range.exclude_versions:
                if ver == version.parse(excluded):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _determine_execution_capability(self, compatibility_level: CompatibilityLevel,
                                      strategy: NegotiationStrategy,
                                      metadata: CompatibilityMetadata,
                                      capabilities: RuntimeCapabilities) -> Tuple[bool, str]:
        """Determine if execution is possible and in what mode"""
        if compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE:
            return True, "normal"
        
        elif compatibility_level == CompatibilityLevel.COMPATIBLE_WITH_WARNINGS:
            return True, "normal"
        
        elif compatibility_level == CompatibilityLevel.DEGRADED_COMPATIBILITY:
            if strategy == NegotiationStrategy.GRACEFUL_DEGRADATION and capabilities.allow_degraded_execution:
                return True, "degraded"
            elif strategy == NegotiationStrategy.EMULATION and capabilities.emulation_support:
                return True, "emulated"
            elif strategy == NegotiationStrategy.FALLBACK_TO_LAST_GOOD:
                return True, "fallback"
            else:
                return False, "incompatible"
        
        elif compatibility_level == CompatibilityLevel.INCOMPATIBLE:
            if strategy == NegotiationStrategy.FALLBACK_TO_LAST_GOOD:
                return True, "fallback"
            else:
                return False, "incompatible"
        
        else:  # REQUIRES_MIGRATION
            return False, "requires_migration"
    
    def _generate_compatibility_warnings(self, metadata: CompatibilityMetadata,
                                       capabilities: RuntimeCapabilities,
                                       compatibility_level: CompatibilityLevel) -> List[str]:
        """Generate compatibility warnings"""
        warnings = []
        
        if compatibility_level == CompatibilityLevel.COMPATIBLE_WITH_WARNINGS:
            warnings.append("Minor version differences detected")
        
        if compatibility_level == CompatibilityLevel.DEGRADED_COMPATIBILITY:
            warnings.append("Some features may not be available")
        
        # Check for deprecated features
        deprecated_used = set(metadata.deprecated_features_used) & set(capabilities.deprecated_features)
        if deprecated_used:
            warnings.append(f"Using deprecated features: {', '.join(deprecated_used)}")
        
        # Check for missing features
        missing_features = set(metadata.required_features) - set(capabilities.supported_features)
        if missing_features:
            warnings.append(f"Missing features: {', '.join(missing_features)}")
        
        return warnings
    
    def _identify_feature_limitations(self, metadata: CompatibilityMetadata,
                                    capabilities: RuntimeCapabilities) -> List[str]:
        """Identify feature limitations"""
        limitations = []
        
        # Check required features not supported
        missing_required = set(metadata.required_features) - set(capabilities.supported_features)
        for feature in missing_required:
            limitations.append(f"Feature '{feature}' not available - functionality may be limited")
        
        # Check optional features not supported
        missing_optional = set(metadata.optional_features) - set(capabilities.supported_features)
        for feature in missing_optional:
            limitations.append(f"Optional feature '{feature}' not available")
        
        return limitations
    
    def _assess_performance_impacts(self, compatibility_level: CompatibilityLevel,
                                  strategy: NegotiationStrategy) -> List[str]:
        """Assess performance impacts of compatibility mode"""
        impacts = []
        
        if strategy == NegotiationStrategy.EMULATION:
            impacts.append("Emulation mode may reduce performance by 10-20%")
        
        if compatibility_level == CompatibilityLevel.DEGRADED_COMPATIBILITY:
            impacts.append("Degraded mode may have reduced throughput")
        
        if strategy == NegotiationStrategy.FALLBACK_TO_LAST_GOOD:
            impacts.append("Using fallback plan may not include latest optimizations")
        
        return impacts
    
    def _generate_execution_parameters(self, compatibility_level: CompatibilityLevel,
                                     strategy: NegotiationStrategy,
                                     metadata: CompatibilityMetadata,
                                     capabilities: RuntimeCapabilities) -> Dict[str, Any]:
        """Generate execution parameters for the negotiated compatibility"""
        parameters = {
            'compatibility_mode': compatibility_level.value,
            'negotiation_strategy': strategy.value,
            'strict_validation': capabilities.strict_compatibility
        }
        
        if strategy == NegotiationStrategy.EMULATION:
            parameters['emulation_target_version'] = metadata.compiler_version
        
        if compatibility_level == CompatibilityLevel.DEGRADED_COMPATIBILITY:
            parameters['disable_features'] = list(
                set(metadata.required_features) - set(capabilities.supported_features)
            )
        
        return parameters
    
    def _identify_fallback_plan(self, metadata: CompatibilityMetadata,
                              capabilities: RuntimeCapabilities) -> Optional[str]:
        """Identify fallback plan for incompatible scenarios"""
        # In a real implementation, this would look up the last compatible plan
        # For now, return a placeholder
        return f"fallback_plan_for_{metadata.compiler_version}"
    
    def _update_negotiation_stats(self, result: NegotiationResult):
        """Update negotiation statistics"""
        self.negotiation_stats['total_negotiations'] += 1
        
        if result.can_execute:
            self.negotiation_stats['successful_negotiations'] += 1
            
            if result.execution_mode == "degraded":
                self.negotiation_stats['degraded_executions'] += 1
            elif result.execution_mode == "fallback":
                self.negotiation_stats['fallback_executions'] += 1
        else:
            self.negotiation_stats['failed_negotiations'] += 1
        
        # Update compatibility level stats
        level = result.compatibility_level.value
        self.negotiation_stats['compatibility_levels'][level] += 1

# API Interface
class VersionNegotiationAPI:
    """API interface for version negotiation operations"""
    
    def __init__(self, negotiation_service: Optional[VersionNegotiationService] = None):
        self.negotiation_service = negotiation_service or VersionNegotiationService()
    
    def check_compatibility(self, manifest_version: str, runtime_version: str,
                          component: str = "compiler_version") -> Dict[str, Any]:
        """API endpoint to check version compatibility"""
        try:
            component_enum = VersionComponent(component)
            compatibility_level = self.negotiation_service.check_version_compatibility(
                manifest_version, runtime_version, component_enum
            )
            
            return {
                'success': True,
                'manifest_version': manifest_version,
                'runtime_version': runtime_version,
                'component': component,
                'compatibility_level': compatibility_level.value,
                'compatible': compatibility_level in [
                    CompatibilityLevel.FULLY_COMPATIBLE,
                    CompatibilityLevel.COMPATIBLE_WITH_WARNINGS,
                    CompatibilityLevel.DEGRADED_COMPATIBILITY
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def negotiate_compatibility(self, manifest_metadata: Dict[str, Any],
                              runtime_name: str, runtime_version: str) -> Dict[str, Any]:
        """API endpoint to negotiate compatibility"""
        try:
            # Convert dict to CompatibilityMetadata
            metadata = CompatibilityMetadata(**manifest_metadata)
            
            # Get runtime capabilities
            runtime_key = f"{runtime_name}_{runtime_version}"
            if runtime_key not in self.negotiation_service.registered_runtimes:
                return {
                    'success': False,
                    'error': f'Unknown runtime: {runtime_name} v{runtime_version}'
                }
            
            capabilities = self.negotiation_service.registered_runtimes[runtime_key]
            
            # Perform negotiation
            result = self.negotiation_service.negotiate_compatibility(metadata, capabilities)
            
            return {
                'success': True,
                'negotiation_id': result.negotiation_id,
                'can_execute': result.can_execute,
                'execution_mode': result.execution_mode,
                'compatibility_level': result.compatibility_level.value,
                'warnings': result.compatibility_warnings,
                'limitations': result.feature_limitations,
                'performance_impacts': result.performance_impacts
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def simulate_handshake(self, manifest_metadata: Dict[str, Any],
                         runtime_name: str, runtime_version: str) -> Dict[str, Any]:
        """API endpoint to simulate runtime handshake"""
        try:
            metadata = CompatibilityMetadata(**manifest_metadata)
            result = self.negotiation_service.simulate_runtime_handshake(
                metadata, runtime_name, runtime_version
            )
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def run_version_negotiation_tests():
    """Run comprehensive version negotiation tests"""
    print("=== Version Negotiation Service Tests ===")
    
    # Initialize service
    negotiation_service = VersionNegotiationService()
    negotiation_api = VersionNegotiationAPI(negotiation_service)
    
    # Test 1: Compatibility metadata creation
    print("\n1. Testing compatibility metadata creation...")
    
    metadata = negotiation_service.create_compatibility_metadata(
        compiler_version="1.1.0",
        ir_schema_version="1.0.0",
        dsl_version="1.1.0",
        required_runtime_min="1.0.0",
        required_features=["ml_inference", "policy_enforcement"],
        breaking_changes=[]
    )
    
    print(f"   Compatibility metadata created:")
    print(f"     Compiler version: {metadata.compiler_version}")
    print(f"     Required runtime min: {metadata.required_runtime_min}")
    print(f"     Required features: {metadata.required_features}")
    print(f"     Backward compatible: {metadata.backward_compatible}")
    print(f"     Preferred strategy: {metadata.preferred_negotiation_strategy.value}")
    
    # Test 2: Runtime capabilities registration
    print("\n2. Testing runtime capabilities registration...")
    
    test_runtime = negotiation_service.register_runtime_capabilities(
        runtime_name="test-runtime",
        runtime_version="1.1.0",
        supported_compiler_versions=VersionRange(min_version="1.0.0", max_version="1.2.0"),
        supported_features=["ml_inference", "policy_enforcement", "basic_execution"]
    )
    
    print(f"   Runtime registered: {test_runtime.runtime_name} v{test_runtime.runtime_version}")
    print(f"   Supported compiler versions: {test_runtime.supported_compiler_versions.min_version} - {test_runtime.supported_compiler_versions.max_version}")
    print(f"   Supported features: {test_runtime.supported_features}")
    
    # Test 3: Version compatibility checking
    print("\n3. Testing version compatibility checking...")
    
    version_pairs = [
        ("1.0.0", "1.0.0"),
        ("1.0.0", "1.1.0"),
        ("1.1.0", "1.0.0"),
        ("1.0.0", "2.0.0"),
        ("2.0.0", "1.0.0")
    ]
    
    for manifest_ver, runtime_ver in version_pairs:
        compat_level = negotiation_service.check_version_compatibility(manifest_ver, runtime_ver)
        print(f"   {manifest_ver} → {runtime_ver}: {compat_level.value}")
    
    # Test 4: Compatibility negotiation
    print("\n4. Testing compatibility negotiation...")
    
    # Get a registered runtime for testing
    runtime_key = "rbia-runtime_1.1.0"
    if runtime_key in negotiation_service.registered_runtimes:
        runtime_capabilities = negotiation_service.registered_runtimes[runtime_key]
        
        negotiation_result = negotiation_service.negotiate_compatibility(metadata, runtime_capabilities)
        
        print(f"   Negotiation result:")
        print(f"     Can execute: {negotiation_result.can_execute}")
        print(f"     Execution mode: {negotiation_result.execution_mode}")
        print(f"     Compatibility level: {negotiation_result.compatibility_level.value}")
        print(f"     Warnings: {len(negotiation_result.compatibility_warnings)}")
        print(f"     Limitations: {len(negotiation_result.feature_limitations)}")
        print(f"     Negotiation time: {negotiation_result.negotiation_time_ms:.2f}ms")
    
    # Test 5: Runtime handshake simulation
    print("\n5. Testing runtime handshake simulation...")
    
    handshake_result = negotiation_service.simulate_runtime_handshake(
        metadata, "rbia-runtime", "1.1.0"
    )
    
    print(f"   Handshake result:")
    print(f"     Success: {handshake_result['success']}")
    print(f"     Action: {handshake_result.get('action', 'N/A')}")
    print(f"     Execution mode: {handshake_result.get('execution_mode', 'N/A')}")
    print(f"     Warnings: {len(handshake_result.get('warnings', []))}")
    
    # Test 6: Migration plan creation
    print("\n6. Testing migration plan creation...")
    
    migration_scenarios = [
        ("1.0.0", "1.1.0"),  # Minor version upgrade
        ("1.0.0", "2.0.0"),  # Major version upgrade
        ("1.2.0", "1.2.1")   # Patch version upgrade
    ]
    
    for source_ver, target_ver in migration_scenarios:
        migration_plan = negotiation_service.create_migration_plan(source_ver, target_ver)
        
        print(f"   Migration {source_ver} → {target_ver}:")
        print(f"     Risk level: {migration_plan.risk_level}")
        print(f"     Estimated effort: {migration_plan.estimated_effort_hours} hours")
        print(f"     Required changes: {len(migration_plan.required_changes)}")
        print(f"     Breaking changes: {len(migration_plan.breaking_changes)}")
        print(f"     Rollback available: {migration_plan.rollback_available}")
    
    # Test 7: Incompatible scenario
    print("\n7. Testing incompatible scenario...")
    
    # Create metadata with features not supported by runtime
    incompatible_metadata = negotiation_service.create_compatibility_metadata(
        compiler_version="2.0.0",
        ir_schema_version="2.0.0",
        dsl_version="2.0.0",
        required_runtime_min="2.0.0",
        required_features=["advanced_feature_not_supported"],
        breaking_changes=["major_api_change"]
    )
    
    # Try to negotiate with older runtime
    if runtime_key in negotiation_service.registered_runtimes:
        incompatible_result = negotiation_service.negotiate_compatibility(
            incompatible_metadata, runtime_capabilities
        )
        
        print(f"   Incompatible negotiation:")
        print(f"     Can execute: {incompatible_result.can_execute}")
        print(f"     Compatibility level: {incompatible_result.compatibility_level.value}")
        print(f"     Fallback plan: {incompatible_result.fallback_plan_id}")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Test API compatibility check
    api_compat_result = negotiation_api.check_compatibility("1.1.0", "1.0.0")
    print(f"   API compatibility check: {'✅ PASS' if api_compat_result['success'] else '❌ FAIL'}")
    if api_compat_result['success']:
        print(f"     Compatibility level: {api_compat_result['compatibility_level']}")
        print(f"     Compatible: {api_compat_result['compatible']}")
    
    # Test API negotiation
    api_negotiation_result = negotiation_api.negotiate_compatibility(
        asdict(metadata), "rbia-runtime", "1.1.0"
    )
    print(f"   API negotiation: {'✅ PASS' if api_negotiation_result['success'] else '❌ FAIL'}")
    
    # Test API handshake
    api_handshake_result = negotiation_api.simulate_handshake(
        asdict(metadata), "rbia-runtime", "1.1.0"
    )
    print(f"   API handshake: {'✅ PASS' if api_handshake_result['success'] else '❌ FAIL'}")
    
    # Test 9: Statistics
    print("\n9. Testing statistics...")
    
    stats = negotiation_service.get_negotiation_statistics()
    print(f"   Total negotiations: {stats['total_negotiations']}")
    print(f"   Successful negotiations: {stats['successful_negotiations']}")
    print(f"   Failed negotiations: {stats['failed_negotiations']}")
    print(f"   Registered runtimes: {stats['registered_runtimes']}")
    print(f"   Compatibility matrix entries: {stats['compatibility_matrix_entries']}")
    print(f"   Migration plans created: {stats['migration_plans_created']}")
    
    # Test 10: Edge cases
    print("\n10. Testing edge cases...")
    
    # Test with invalid version formats
    try:
        invalid_compat = negotiation_service.check_version_compatibility("invalid", "1.0.0")
        print(f"   Invalid version handling: {invalid_compat.value}")
    except Exception as e:
        print(f"   Invalid version handling: Exception caught ({type(e).__name__})")
    
    # Test with empty features
    empty_metadata = negotiation_service.create_compatibility_metadata(
        compiler_version="1.0.0",
        ir_schema_version="1.0.0", 
        dsl_version="1.0.0",
        required_runtime_min="1.0.0",
        required_features=[],
        breaking_changes=[]
    )
    print(f"   Empty features metadata: {empty_metadata.backward_compatible}")
    
    print(f"\n=== Test Summary ===")
    final_stats = negotiation_service.get_negotiation_statistics()
    print(f"Version negotiation service tested successfully")
    print(f"Total negotiations: {final_stats['total_negotiations']}")
    print(f"Success rate: {final_stats['successful_negotiations']}/{final_stats['total_negotiations']}")
    print(f"Registered runtimes: {final_stats['registered_runtimes']}")
    print(f"Migration plans: {final_stats['migration_plans_created']}")
    
    return negotiation_service, negotiation_api

if __name__ == "__main__":
    run_version_negotiation_tests()



