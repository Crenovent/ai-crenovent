"""
Task 7.3-T15: Add Compatibility Matrix (runtime/compiler min/max)
Safe upgrades with version compatibility checking and validation
"""

import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

from packaging import version as pkg_version
from packaging.specifiers import SpecifierSet


class CompatibilityStatus(Enum):
    """Compatibility status levels"""
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    DEPRECATED = "deprecated"
    UNTESTED = "untested"
    UNKNOWN = "unknown"


class PlatformType(Enum):
    """Supported platform types"""
    LINUX_X64 = "linux-x64"
    LINUX_ARM64 = "linux-arm64"
    WINDOWS_X64 = "windows-x64"
    MACOS_X64 = "macos-x64"
    MACOS_ARM64 = "macos-arm64"
    CONTAINER = "container"
    KUBERNETES = "kubernetes"


class FeatureType(Enum):
    """Feature requirement types"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEPRECATED = "deprecated"


@dataclass
class VersionRange:
    """Version range specification"""
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    exclude_versions: List[str] = field(default_factory=list)
    include_prereleases: bool = False
    
    def __post_init__(self):
        """Validate version range"""
        if self.min_version:
            try:
                pkg_version.parse(self.min_version)
            except Exception as e:
                raise ValueError(f"Invalid min_version: {self.min_version}") from e
        
        if self.max_version:
            try:
                pkg_version.parse(self.max_version)
            except Exception as e:
                raise ValueError(f"Invalid max_version: {self.max_version}") from e
        
        if self.min_version and self.max_version:
            min_ver = pkg_version.parse(self.min_version)
            max_ver = pkg_version.parse(self.max_version)
            if min_ver >= max_ver:
                raise ValueError(f"min_version {self.min_version} must be less than max_version {self.max_version}")
    
    def is_compatible(self, version_str: str) -> bool:
        """Check if a version is compatible with this range"""
        try:
            version = pkg_version.parse(version_str)
            
            # Check excluded versions
            if version_str in self.exclude_versions:
                return False
            
            # Check prerelease policy
            if version.is_prerelease and not self.include_prereleases:
                return False
            
            # Check min version
            if self.min_version:
                min_ver = pkg_version.parse(self.min_version)
                if version < min_ver:
                    return False
            
            # Check max version
            if self.max_version:
                max_ver = pkg_version.parse(self.max_version)
                if version > max_ver:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def to_specifier(self) -> str:
        """Convert to PEP 440 version specifier"""
        parts = []
        
        if self.min_version:
            parts.append(f">={self.min_version}")
        
        if self.max_version:
            parts.append(f"<={self.max_version}")
        
        for excluded in self.exclude_versions:
            parts.append(f"!={excluded}")
        
        return ",".join(parts) if parts else "*"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "min_version": self.min_version,
            "max_version": self.max_version,
            "exclude_versions": self.exclude_versions,
            "include_prereleases": self.include_prereleases,
            "specifier": self.to_specifier()
        }


@dataclass
class Feature:
    """Feature requirement specification"""
    name: str
    requirement_type: FeatureType
    min_version: Optional[str] = None
    description: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "name": self.name,
            "requirement_type": self.requirement_type.value
        }
        
        if self.min_version:
            result["min_version"] = self.min_version
        if self.description:
            result["description"] = self.description
        if self.alternatives:
            result["alternatives"] = self.alternatives
        
        return result


@dataclass
class ConnectorRequirement:
    """Connector version requirement"""
    connector_name: str
    version_range: VersionRange
    requirement_type: FeatureType = FeatureType.REQUIRED
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "connector_name": self.connector_name,
            "version_range": self.version_range.to_dict(),
            "requirement_type": self.requirement_type.value,
            "configuration": self.configuration
        }


@dataclass
class CompatibilityTest:
    """Compatibility test result"""
    test_name: str
    platform: PlatformType
    runtime_version: str
    compiler_version: str
    status: CompatibilityStatus
    test_date: datetime
    test_duration_seconds: float
    error_message: Optional[str] = None
    test_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "test_name": self.test_name,
            "platform": self.platform.value,
            "runtime_version": self.runtime_version,
            "compiler_version": self.compiler_version,
            "status": self.status.value,
            "test_date": self.test_date.isoformat(),
            "test_duration_seconds": self.test_duration_seconds
        }
        
        if self.error_message:
            result["error_message"] = self.error_message
        if self.test_details:
            result["test_details"] = self.test_details
        
        return result


@dataclass
class CompatibilityMatrix:
    """Complete compatibility matrix for a workflow version"""
    version_id: str
    workflow_name: str
    workflow_version: str
    
    # Runtime compatibility
    runtime_requirements: VersionRange = field(default_factory=VersionRange)
    compiler_requirements: VersionRange = field(default_factory=VersionRange)
    
    # Platform compatibility
    supported_platforms: List[PlatformType] = field(default_factory=list)
    required_features: List[Feature] = field(default_factory=list)
    optional_features: List[Feature] = field(default_factory=list)
    deprecated_features: List[Feature] = field(default_factory=list)
    
    # Dependency compatibility
    dsl_version_requirements: VersionRange = field(default_factory=VersionRange)
    required_connectors: List[ConnectorRequirement] = field(default_factory=list)
    optional_connectors: List[ConnectorRequirement] = field(default_factory=list)
    
    # Backward compatibility
    backward_compatible_versions: List[str] = field(default_factory=list)
    breaking_changes_from: List[str] = field(default_factory=list)
    migration_required_from: List[str] = field(default_factory=list)
    deprecation_warnings: List[str] = field(default_factory=list)
    
    # Testing and validation
    compatibility_tested: bool = False
    test_results: List[CompatibilityTest] = field(default_factory=list)
    last_tested_at: Optional[datetime] = None
    test_coverage_percentage: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_compatible_with_runtime(self, runtime_version: str) -> bool:
        """Check if compatible with a specific runtime version"""
        return self.runtime_requirements.is_compatible(runtime_version)
    
    def is_compatible_with_compiler(self, compiler_version: str) -> bool:
        """Check if compatible with a specific compiler version"""
        return self.compiler_requirements.is_compatible(compiler_version)
    
    def is_platform_supported(self, platform: PlatformType) -> bool:
        """Check if platform is supported"""
        return platform in self.supported_platforms
    
    def get_required_features(self) -> List[str]:
        """Get list of required feature names"""
        return [f.name for f in self.required_features]
    
    def get_optional_features(self) -> List[str]:
        """Get list of optional feature names"""
        return [f.name for f in self.optional_features]
    
    def get_deprecated_features(self) -> List[str]:
        """Get list of deprecated feature names"""
        return [f.name for f in self.deprecated_features]
    
    def is_backward_compatible_with(self, version: str) -> bool:
        """Check if backward compatible with a specific version"""
        return version in self.backward_compatible_versions
    
    def has_breaking_changes_from(self, version: str) -> bool:
        """Check if there are breaking changes from a specific version"""
        return version in self.breaking_changes_from
    
    def requires_migration_from(self, version: str) -> bool:
        """Check if migration is required from a specific version"""
        return version in self.migration_required_from
    
    def get_test_coverage(self) -> float:
        """Get test coverage percentage"""
        return self.test_coverage_percentage
    
    def get_compatibility_score(self) -> float:
        """Calculate overall compatibility score (0-1)"""
        if not self.test_results:
            return 0.0
        
        compatible_tests = sum(1 for test in self.test_results if test.status == CompatibilityStatus.COMPATIBLE)
        total_tests = len(self.test_results)
        
        return compatible_tests / total_tests if total_tests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version_id": self.version_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "runtime_requirements": self.runtime_requirements.to_dict(),
            "compiler_requirements": self.compiler_requirements.to_dict(),
            "supported_platforms": [p.value for p in self.supported_platforms],
            "required_features": [f.to_dict() for f in self.required_features],
            "optional_features": [f.to_dict() for f in self.optional_features],
            "deprecated_features": [f.to_dict() for f in self.deprecated_features],
            "dsl_version_requirements": self.dsl_version_requirements.to_dict(),
            "required_connectors": [c.to_dict() for c in self.required_connectors],
            "optional_connectors": [c.to_dict() for c in self.optional_connectors],
            "backward_compatible_versions": self.backward_compatible_versions,
            "breaking_changes_from": self.breaking_changes_from,
            "migration_required_from": self.migration_required_from,
            "deprecation_warnings": self.deprecation_warnings,
            "compatibility_tested": self.compatibility_tested,
            "test_results": [t.to_dict() for t in self.test_results],
            "last_tested_at": self.last_tested_at.isoformat() if self.last_tested_at else None,
            "test_coverage_percentage": self.test_coverage_percentage,
            "compatibility_score": self.get_compatibility_score(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class CompatibilityMatrixBuilder:
    """Builder for creating compatibility matrices"""
    
    def __init__(self):
        self.matrix = None
    
    def create_matrix(
        self,
        version_id: str,
        workflow_name: str,
        workflow_version: str
    ) -> 'CompatibilityMatrixBuilder':
        """Create a new compatibility matrix"""
        self.matrix = CompatibilityMatrix(
            version_id=version_id,
            workflow_name=workflow_name,
            workflow_version=workflow_version
        )
        return self
    
    def set_runtime_requirements(
        self,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
        exclude_versions: Optional[List[str]] = None
    ) -> 'CompatibilityMatrixBuilder':
        """Set runtime version requirements"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        self.matrix.runtime_requirements = VersionRange(
            min_version=min_version,
            max_version=max_version,
            exclude_versions=exclude_versions or []
        )
        return self
    
    def set_compiler_requirements(
        self,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
        exclude_versions: Optional[List[str]] = None
    ) -> 'CompatibilityMatrixBuilder':
        """Set compiler version requirements"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        self.matrix.compiler_requirements = VersionRange(
            min_version=min_version,
            max_version=max_version,
            exclude_versions=exclude_versions or []
        )
        return self
    
    def add_supported_platforms(self, platforms: List[PlatformType]) -> 'CompatibilityMatrixBuilder':
        """Add supported platforms"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        self.matrix.supported_platforms.extend(platforms)
        return self
    
    def add_required_feature(
        self,
        name: str,
        min_version: Optional[str] = None,
        description: Optional[str] = None,
        alternatives: Optional[List[str]] = None
    ) -> 'CompatibilityMatrixBuilder':
        """Add a required feature"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        feature = Feature(
            name=name,
            requirement_type=FeatureType.REQUIRED,
            min_version=min_version,
            description=description,
            alternatives=alternatives or []
        )
        self.matrix.required_features.append(feature)
        return self
    
    def add_optional_feature(
        self,
        name: str,
        min_version: Optional[str] = None,
        description: Optional[str] = None
    ) -> 'CompatibilityMatrixBuilder':
        """Add an optional feature"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        feature = Feature(
            name=name,
            requirement_type=FeatureType.OPTIONAL,
            min_version=min_version,
            description=description
        )
        self.matrix.optional_features.append(feature)
        return self
    
    def add_connector_requirement(
        self,
        connector_name: str,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
        required: bool = True,
        configuration: Optional[Dict[str, Any]] = None
    ) -> 'CompatibilityMatrixBuilder':
        """Add a connector requirement"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        version_range = VersionRange(
            min_version=min_version,
            max_version=max_version
        )
        
        requirement = ConnectorRequirement(
            connector_name=connector_name,
            version_range=version_range,
            requirement_type=FeatureType.REQUIRED if required else FeatureType.OPTIONAL,
            configuration=configuration or {}
        )
        
        if required:
            self.matrix.required_connectors.append(requirement)
        else:
            self.matrix.optional_connectors.append(requirement)
        
        return self
    
    def set_backward_compatibility(
        self,
        compatible_versions: List[str],
        breaking_changes_from: Optional[List[str]] = None,
        migration_required_from: Optional[List[str]] = None
    ) -> 'CompatibilityMatrixBuilder':
        """Set backward compatibility information"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        self.matrix.backward_compatible_versions = compatible_versions
        self.matrix.breaking_changes_from = breaking_changes_from or []
        self.matrix.migration_required_from = migration_required_from or []
        return self
    
    def add_test_result(
        self,
        test_name: str,
        platform: PlatformType,
        runtime_version: str,
        compiler_version: str,
        status: CompatibilityStatus,
        test_duration_seconds: float = 0.0,
        error_message: Optional[str] = None,
        test_details: Optional[Dict[str, Any]] = None
    ) -> 'CompatibilityMatrixBuilder':
        """Add a test result"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        test_result = CompatibilityTest(
            test_name=test_name,
            platform=platform,
            runtime_version=runtime_version,
            compiler_version=compiler_version,
            status=status,
            test_date=datetime.now(timezone.utc),
            test_duration_seconds=test_duration_seconds,
            error_message=error_message,
            test_details=test_details or {}
        )
        
        self.matrix.test_results.append(test_result)
        self.matrix.last_tested_at = datetime.now(timezone.utc)
        return self
    
    def build(self) -> CompatibilityMatrix:
        """Build and return the compatibility matrix"""
        if self.matrix is None:
            raise ValueError("Must create matrix first")
        
        # Calculate test coverage
        if self.matrix.test_results:
            self.matrix.compatibility_tested = True
            # Simple coverage calculation based on platform/version combinations
            total_combinations = len(self.matrix.supported_platforms) * 2  # Assume 2 versions tested per platform
            tested_combinations = len(self.matrix.test_results)
            self.matrix.test_coverage_percentage = min(100.0, (tested_combinations / max(1, total_combinations)) * 100)
        
        return self.matrix


class CompatibilityChecker:
    """Utility for checking compatibility between versions and environments"""
    
    def __init__(self):
        self.matrices: Dict[str, CompatibilityMatrix] = {}
    
    def register_matrix(self, matrix: CompatibilityMatrix) -> None:
        """Register a compatibility matrix"""
        self.matrices[matrix.version_id] = matrix
    
    def check_runtime_compatibility(
        self,
        version_id: str,
        runtime_version: str
    ) -> Dict[str, Any]:
        """Check runtime compatibility for a version"""
        if version_id not in self.matrices:
            return {
                "compatible": False,
                "reason": f"No compatibility matrix found for version {version_id}"
            }
        
        matrix = self.matrices[version_id]
        compatible = matrix.is_compatible_with_runtime(runtime_version)
        
        return {
            "compatible": compatible,
            "version_id": version_id,
            "runtime_version": runtime_version,
            "requirements": matrix.runtime_requirements.to_dict(),
            "reason": "Compatible" if compatible else "Runtime version not supported"
        }
    
    def check_platform_compatibility(
        self,
        version_id: str,
        platform: PlatformType
    ) -> Dict[str, Any]:
        """Check platform compatibility for a version"""
        if version_id not in self.matrices:
            return {
                "compatible": False,
                "reason": f"No compatibility matrix found for version {version_id}"
            }
        
        matrix = self.matrices[version_id]
        compatible = matrix.is_platform_supported(platform)
        
        return {
            "compatible": compatible,
            "version_id": version_id,
            "platform": platform.value,
            "supported_platforms": [p.value for p in matrix.supported_platforms],
            "reason": "Compatible" if compatible else "Platform not supported"
        }
    
    def check_upgrade_compatibility(
        self,
        from_version_id: str,
        to_version_id: str
    ) -> Dict[str, Any]:
        """Check upgrade compatibility between versions"""
        if to_version_id not in self.matrices:
            return {
                "compatible": False,
                "reason": f"No compatibility matrix found for target version {to_version_id}"
            }
        
        to_matrix = self.matrices[to_version_id]
        
        # Get source version info
        from_version = None
        if from_version_id in self.matrices:
            from_version = self.matrices[from_version_id].workflow_version
        
        if not from_version:
            return {
                "compatible": False,
                "reason": f"Source version {from_version_id} not found"
            }
        
        # Check backward compatibility
        backward_compatible = to_matrix.is_backward_compatible_with(from_version)
        has_breaking_changes = to_matrix.has_breaking_changes_from(from_version)
        requires_migration = to_matrix.requires_migration_from(from_version)
        
        return {
            "compatible": backward_compatible,
            "from_version": from_version,
            "to_version": to_matrix.workflow_version,
            "backward_compatible": backward_compatible,
            "has_breaking_changes": has_breaking_changes,
            "requires_migration": requires_migration,
            "migration_required": requires_migration,
            "reason": self._get_upgrade_reason(backward_compatible, has_breaking_changes, requires_migration)
        }
    
    def get_compatibility_report(self, version_id: str) -> Dict[str, Any]:
        """Get comprehensive compatibility report for a version"""
        if version_id not in self.matrices:
            return {
                "error": f"No compatibility matrix found for version {version_id}"
            }
        
        matrix = self.matrices[version_id]
        
        return {
            "version_id": version_id,
            "workflow_name": matrix.workflow_name,
            "workflow_version": matrix.workflow_version,
            "compatibility_score": matrix.get_compatibility_score(),
            "test_coverage": matrix.get_test_coverage(),
            "runtime_requirements": matrix.runtime_requirements.to_dict(),
            "compiler_requirements": matrix.compiler_requirements.to_dict(),
            "supported_platforms": [p.value for p in matrix.supported_platforms],
            "required_features": matrix.get_required_features(),
            "optional_features": matrix.get_optional_features(),
            "deprecated_features": matrix.get_deprecated_features(),
            "backward_compatible_versions": matrix.backward_compatible_versions,
            "breaking_changes_from": matrix.breaking_changes_from,
            "test_summary": {
                "total_tests": len(matrix.test_results),
                "compatible_tests": sum(1 for t in matrix.test_results if t.status == CompatibilityStatus.COMPATIBLE),
                "failed_tests": sum(1 for t in matrix.test_results if t.status == CompatibilityStatus.INCOMPATIBLE),
                "last_tested": matrix.last_tested_at.isoformat() if matrix.last_tested_at else None
            }
        }
    
    def _get_upgrade_reason(
        self,
        backward_compatible: bool,
        has_breaking_changes: bool,
        requires_migration: bool
    ) -> str:
        """Get human-readable upgrade compatibility reason"""
        if backward_compatible and not has_breaking_changes:
            return "Safe upgrade - fully backward compatible"
        elif backward_compatible and has_breaking_changes:
            return "Upgrade possible but has breaking changes - review required"
        elif requires_migration:
            return "Upgrade requires migration - follow migration guide"
        else:
            return "Upgrade not recommended - incompatible versions"


# Example usage
def example_compatibility_matrix():
    """Example of creating and using a compatibility matrix"""
    
    # Create a compatibility matrix
    builder = CompatibilityMatrixBuilder()
    
    matrix = (builder
        .create_matrix("version-123", "Pipeline Hygiene", "2.1.0")
        .set_runtime_requirements(min_version="1.8.0", max_version="2.0.0")
        .set_compiler_requirements(min_version="1.5.0")
        .add_supported_platforms([
            PlatformType.LINUX_X64,
            PlatformType.LINUX_ARM64,
            PlatformType.CONTAINER
        ])
        .add_required_feature("async_execution", min_version="1.0.0", description="Async workflow execution")
        .add_optional_feature("gpu_acceleration", description="GPU-accelerated ML inference")
        .add_connector_requirement("salesforce", min_version="2.0.0", max_version="3.0.0")
        .add_connector_requirement("slack", min_version="1.5.0", required=False)
        .set_backward_compatibility(
            compatible_versions=["2.0.0", "2.0.1", "2.0.2"],
            breaking_changes_from=["1.9.0", "1.8.0"],
            migration_required_from=["1.7.0"]
        )
        .add_test_result(
            "runtime_compatibility_test",
            PlatformType.LINUX_X64,
            "1.9.0",
            "1.6.0",
            CompatibilityStatus.COMPATIBLE,
            test_duration_seconds=45.2
        )
        .build()
    )
    
    # Create compatibility checker
    checker = CompatibilityChecker()
    checker.register_matrix(matrix)
    
    # Check various compatibility scenarios
    runtime_check = checker.check_runtime_compatibility("version-123", "1.9.0")
    print(f"Runtime compatibility: {runtime_check}")
    
    platform_check = checker.check_platform_compatibility("version-123", PlatformType.LINUX_X64)
    print(f"Platform compatibility: {platform_check}")
    
    # Get full compatibility report
    report = checker.get_compatibility_report("version-123")
    print(f"Compatibility report: {json.dumps(report, indent=2)}")
    
    return matrix


if __name__ == "__main__":
    example_compatibility_matrix()
