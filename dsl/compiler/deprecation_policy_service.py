"""
Deprecation Policy Service - Task 6.2.35
=========================================

Deprecation policy for DSL keywords
- Establishes formal deprecation mechanism for DSL keywords and constructs
- Deprecation registry with versioning and migration hints
- Compiler-linter support for deprecation warnings and errors
- Documentation and release notes automation
- Backend implementation (no actual CLI/Builder UI - that's interface layer)

Dependencies: Task 6.2.32 (Diagnostic Catalog Service)
Outputs: Deprecation management → enables safe evolution of DSL syntax
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from packaging import version

logger = logging.getLogger(__name__)

class DeprecationSeverity(Enum):
    """Severity levels for deprecation warnings"""
    INFO = "info"           # Informational notice
    WARNING = "warning"     # Deprecation warning
    ERROR = "error"         # Will be removed, use alternative
    CRITICAL = "critical"   # Removed, compilation fails

class DeprecationCategory(Enum):
    """Categories of deprecated features"""
    KEYWORD = "keyword"
    SYNTAX = "syntax"
    FUNCTION = "function"
    OPERATOR = "operator"
    ATTRIBUTE = "attribute"
    CONFIGURATION = "configuration"
    API = "api"

class MigrationComplexity(Enum):
    """Complexity of migration to new syntax"""
    TRIVIAL = "trivial"         # Simple find/replace
    SIMPLE = "simple"           # Minor syntax changes
    MODERATE = "moderate"       # Some restructuring needed
    COMPLEX = "complex"         # Significant changes required
    BREAKING = "breaking"       # Major breaking changes

@dataclass
class DeprecationEntry:
    """Entry in the deprecation registry"""
    # Core identification
    deprecated_feature: str
    feature_type: DeprecationCategory
    
    # Versioning
    deprecated_in_version: str
    remove_in_version: str
    current_version: str = "1.0.0"
    
    # Migration information
    migration_hint: str
    replacement_syntax: Optional[str] = None
    migration_complexity: MigrationComplexity = MigrationComplexity.SIMPLE
    
    # Documentation
    deprecation_reason: str
    impact_description: str
    examples_old: List[str] = field(default_factory=list)
    examples_new: List[str] = field(default_factory=list)
    
    # Policy settings
    grace_period_months: int = 12
    allow_override: bool = True
    auto_migration_available: bool = False
    
    # Status tracking
    usage_count: int = 0
    last_usage_detected: Optional[datetime] = None
    migration_adoption_rate: float = 0.0  # 0.0 to 1.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"

@dataclass
class DeprecationWarning:
    """A deprecation warning found during compilation"""
    warning_id: str
    deprecated_feature: str
    severity: DeprecationSeverity
    
    # Location information
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    source_context: Optional[str] = None
    
    # Deprecation details
    deprecation_entry: Optional[DeprecationEntry] = None
    migration_hint: str = ""
    suggested_replacement: Optional[str] = None
    
    # Timeline information
    removal_timeline: Optional[str] = None
    grace_period_remaining: Optional[str] = None
    
    # Auto-migration
    auto_fix_available: bool = False
    auto_fix_command: Optional[str] = None
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class CompatibilityReport:
    """Compatibility report for a release"""
    report_id: str
    target_version: str
    generated_at: datetime
    
    # Deprecation summary
    total_deprecations: int
    new_deprecations: int
    removed_features: int
    
    # By category
    deprecations_by_category: Dict[DeprecationCategory, int] = field(default_factory=dict)
    deprecations_by_severity: Dict[DeprecationSeverity, int] = field(default_factory=dict)
    
    # Migration information
    trivial_migrations: int = 0
    complex_migrations: int = 0
    breaking_changes: int = 0
    
    # Timeline analysis
    features_removed_this_version: List[str] = field(default_factory=list)
    features_deprecated_this_version: List[str] = field(default_factory=list)
    features_to_remove_next_version: List[str] = field(default_factory=list)
    
    # Recommendations
    migration_recommendations: List[str] = field(default_factory=list)
    upgrade_blockers: List[str] = field(default_factory=list)

# Task 6.2.35: Deprecation Policy Service
class DeprecationPolicyService:
    """Service for managing DSL deprecation policy and warnings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Deprecation registry
        self.deprecation_registry: Dict[str, DeprecationEntry] = {}
        
        # Detected warnings
        self.deprecation_warnings: Dict[str, DeprecationWarning] = {}  # warning_id -> warning
        
        # Compatibility reports
        self.compatibility_reports: Dict[str, CompatibilityReport] = {}  # version -> report
        
        # Version tracking
        self.current_dsl_version = "1.0.0"
        self.supported_versions = ["1.0.0", "1.1.0", "1.2.0"]
        
        # Initialize standard deprecations
        self._initialize_standard_deprecations()
        
        # Statistics
        self.deprecation_stats = {
            'total_entries': 0,
            'active_deprecations': 0,
            'removed_features': 0,
            'warnings_generated': 0,
            'auto_migrations_available': 0
        }
    
    def register_deprecation(self, deprecated_feature: str, feature_type: DeprecationCategory,
                           deprecated_in_version: str, remove_in_version: str,
                           migration_hint: str, deprecation_reason: str,
                           replacement_syntax: Optional[str] = None,
                           migration_complexity: MigrationComplexity = MigrationComplexity.SIMPLE,
                           grace_period_months: int = 12) -> DeprecationEntry:
        """
        Register a new deprecation entry
        
        Args:
            deprecated_feature: Feature being deprecated
            feature_type: Type of feature
            deprecated_in_version: Version when deprecated
            remove_in_version: Version when it will be removed
            migration_hint: How to migrate
            deprecation_reason: Why it's being deprecated
            replacement_syntax: New syntax to use
            migration_complexity: Complexity of migration
            grace_period_months: Grace period in months
            
        Returns:
            DeprecationEntry
        """
        entry = DeprecationEntry(
            deprecated_feature=deprecated_feature,
            feature_type=feature_type,
            deprecated_in_version=deprecated_in_version,
            remove_in_version=remove_in_version,
            current_version=self.current_dsl_version,
            migration_hint=migration_hint,
            replacement_syntax=replacement_syntax,
            migration_complexity=migration_complexity,
            deprecation_reason=deprecation_reason,
            impact_description=f"{feature_type.value} '{deprecated_feature}' will be removed",
            grace_period_months=grace_period_months
        )
        
        # Store in registry
        self.deprecation_registry[deprecated_feature] = entry
        
        # Update statistics
        self._update_deprecation_stats()
        
        self.logger.info(f"✅ Registered deprecation: {deprecated_feature} (remove in {remove_in_version})")
        
        return entry
    
    def check_for_deprecations(self, source_code: str, file_path: Optional[str] = None) -> List[DeprecationWarning]:
        """
        Check source code for deprecated features
        
        Args:
            source_code: Source code to check
            file_path: Optional file path for context
            
        Returns:
            List of deprecation warnings found
        """
        warnings = []
        lines = source_code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check each registered deprecation
            for deprecated_feature, entry in self.deprecation_registry.items():
                if self._feature_found_in_line(deprecated_feature, line, entry.feature_type):
                    warning = self._create_deprecation_warning(
                        deprecated_feature, entry, file_path, line_num, line
                    )
                    warnings.append(warning)
                    
                    # Update usage tracking
                    entry.usage_count += 1
                    entry.last_usage_detected = datetime.now(timezone.utc)
        
        # Store warnings
        for warning in warnings:
            self.deprecation_warnings[warning.warning_id] = warning
        
        # Update statistics
        self.deprecation_stats['warnings_generated'] += len(warnings)
        
        return warnings
    
    def get_deprecation_severity(self, deprecated_feature: str, current_version: Optional[str] = None) -> DeprecationSeverity:
        """
        Get the current severity level for a deprecated feature
        
        Args:
            deprecated_feature: Feature to check
            current_version: Version to check against (defaults to current)
            
        Returns:
            DeprecationSeverity level
        """
        if deprecated_feature not in self.deprecation_registry:
            return DeprecationSeverity.INFO
        
        entry = self.deprecation_registry[deprecated_feature]
        check_version = current_version or self.current_dsl_version
        
        # Parse version numbers
        try:
            current_ver = version.parse(check_version)
            deprecated_ver = version.parse(entry.deprecated_in_version)
            remove_ver = version.parse(entry.remove_in_version)
        except Exception:
            return DeprecationSeverity.WARNING
        
        # Determine severity based on version timeline
        if current_ver >= remove_ver:
            return DeprecationSeverity.CRITICAL  # Feature should be removed
        elif current_ver >= deprecated_ver:
            # Calculate how close we are to removal
            total_period = remove_ver - deprecated_ver
            current_period = current_ver - deprecated_ver
            
            if total_period.major == 0 and total_period.minor == 0:
                # Same major.minor, use patch level
                progress = current_period.micro / max(1, total_period.micro)
            else:
                # Use minor version progress
                progress = (current_period.major * 10 + current_period.minor) / max(1, total_period.major * 10 + total_period.minor)
            
            if progress >= 0.8:  # 80% through deprecation period
                return DeprecationSeverity.ERROR
            elif progress >= 0.5:  # 50% through deprecation period
                return DeprecationSeverity.WARNING
            else:
                return DeprecationSeverity.INFO
        else:
            return DeprecationSeverity.INFO  # Not yet deprecated
    
    def generate_migration_suggestions(self, deprecated_feature: str) -> Dict[str, Any]:
        """
        Generate migration suggestions for a deprecated feature
        
        Args:
            deprecated_feature: Feature to generate suggestions for
            
        Returns:
            Dictionary with migration suggestions
        """
        if deprecated_feature not in self.deprecation_registry:
            return {'error': f'Unknown deprecated feature: {deprecated_feature}'}
        
        entry = self.deprecation_registry[deprecated_feature]
        
        suggestions = {
            'deprecated_feature': deprecated_feature,
            'migration_hint': entry.migration_hint,
            'replacement_syntax': entry.replacement_syntax,
            'migration_complexity': entry.migration_complexity.value,
            'examples': {
                'old': entry.examples_old,
                'new': entry.examples_new
            },
            'auto_migration_available': entry.auto_migration_available,
            'estimated_effort': self._estimate_migration_effort(entry),
            'timeline': {
                'deprecated_in': entry.deprecated_in_version,
                'remove_in': entry.remove_in_version,
                'grace_period_months': entry.grace_period_months
            }
        }
        
        # Add auto-migration command if available
        if entry.auto_migration_available:
            suggestions['auto_migration_command'] = self._generate_auto_migration_command(deprecated_feature, entry)
        
        return suggestions
    
    def create_compatibility_report(self, target_version: str) -> CompatibilityReport:
        """
        Create a compatibility report for a target version
        
        Args:
            target_version: Version to create report for
            
        Returns:
            CompatibilityReport
        """
        report_id = f"compat_{target_version}_{datetime.now().strftime('%Y%m%d')}"
        
        report = CompatibilityReport(
            report_id=report_id,
            target_version=target_version,
            generated_at=datetime.now(timezone.utc)
        )
        
        # Analyze deprecations for this version
        target_ver = version.parse(target_version)
        
        for deprecated_feature, entry in self.deprecation_registry.items():
            deprecated_ver = version.parse(entry.deprecated_in_version)
            remove_ver = version.parse(entry.remove_in_version)
            
            # Count total deprecations
            if deprecated_ver <= target_ver:
                report.total_deprecations += 1
                
                # Count by category
                category = entry.feature_type
                report.deprecations_by_category[category] = report.deprecations_by_category.get(category, 0) + 1
                
                # Count by severity
                severity = self.get_deprecation_severity(deprecated_feature, target_version)
                report.deprecations_by_severity[severity] = report.deprecations_by_severity.get(severity, 0) + 1
                
                # Count by migration complexity
                if entry.migration_complexity == MigrationComplexity.TRIVIAL:
                    report.trivial_migrations += 1
                elif entry.migration_complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.BREAKING]:
                    report.complex_migrations += 1
                
                if entry.migration_complexity == MigrationComplexity.BREAKING:
                    report.breaking_changes += 1
            
            # Check if deprecated in this version
            if deprecated_ver == target_ver:
                report.new_deprecations += 1
                report.features_deprecated_this_version.append(deprecated_feature)
            
            # Check if removed in this version
            if remove_ver == target_ver:
                report.removed_features += 1
                report.features_removed_this_version.append(deprecated_feature)
            
            # Check if will be removed in next version
            next_ver = version.parse(f"{target_ver.major}.{target_ver.minor + 1}.0")
            if remove_ver == next_ver:
                report.features_to_remove_next_version.append(deprecated_feature)
        
        # Generate recommendations
        report.migration_recommendations = self._generate_compatibility_recommendations(report)
        report.upgrade_blockers = self._identify_upgrade_blockers(report)
        
        # Store report
        self.compatibility_reports[target_version] = report
        
        self.logger.info(f"✅ Generated compatibility report for version {target_version}")
        
        return report
    
    def get_deprecation_timeline(self, months_ahead: int = 12) -> Dict[str, Any]:
        """
        Get deprecation timeline for the next N months
        
        Args:
            months_ahead: Number of months to look ahead
            
        Returns:
            Dictionary with timeline information
        """
        timeline = {
            'timeline_months': months_ahead,
            'current_version': self.current_dsl_version,
            'upcoming_removals': [],
            'upcoming_deprecations': [],
            'grace_period_expiring': []
        }
        
        current_date = datetime.now(timezone.utc)
        future_date = current_date + timedelta(days=months_ahead * 30)
        
        for deprecated_feature, entry in self.deprecation_registry.items():
            # Check if removal is upcoming
            try:
                remove_ver = version.parse(entry.remove_in_version)
                current_ver = version.parse(self.current_dsl_version)
                
                # Simple heuristic: assume versions are released every 3 months
                versions_ahead = (remove_ver.major - current_ver.major) * 4 + (remove_ver.minor - current_ver.minor)
                months_to_removal = versions_ahead * 3
                
                if 0 < months_to_removal <= months_ahead:
                    timeline['upcoming_removals'].append({
                        'feature': deprecated_feature,
                        'remove_version': entry.remove_in_version,
                        'estimated_months': months_to_removal,
                        'migration_complexity': entry.migration_complexity.value
                    })
                
                # Check grace period expiration
                deprecated_date = entry.created_at
                grace_expiry = deprecated_date + timedelta(days=entry.grace_period_months * 30)
                
                if current_date <= grace_expiry <= future_date:
                    timeline['grace_period_expiring'].append({
                        'feature': deprecated_feature,
                        'expiry_date': grace_expiry.isoformat(),
                        'days_remaining': (grace_expiry - current_date).days
                    })
                
            except Exception as e:
                self.logger.warning(f"Error processing timeline for {deprecated_feature}: {e}")
        
        # Sort by urgency
        timeline['upcoming_removals'].sort(key=lambda x: x['estimated_months'])
        timeline['grace_period_expiring'].sort(key=lambda x: x['days_remaining'])
        
        return timeline
    
    def export_deprecation_registry(self, format: str = "json") -> str:
        """
        Export the deprecation registry
        
        Args:
            format: Export format (json, yaml, markdown)
            
        Returns:
            Exported registry as string
        """
        if format == "json":
            registry_data = {
                'version': self.current_dsl_version,
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'total_entries': len(self.deprecation_registry),
                'entries': {
                    feature: asdict(entry)
                    for feature, entry in self.deprecation_registry.items()
                }
            }
            return json.dumps(registry_data, indent=2, default=str)
        
        elif format == "markdown":
            return self._export_as_markdown()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_deprecation_statistics(self) -> Dict[str, Any]:
        """Get deprecation service statistics"""
        # Update active deprecations count
        active_count = 0
        removed_count = 0
        
        for entry in self.deprecation_registry.values():
            severity = self.get_deprecation_severity(entry.deprecated_feature)
            if severity == DeprecationSeverity.CRITICAL:
                removed_count += 1
            else:
                active_count += 1
        
        self.deprecation_stats.update({
            'active_deprecations': active_count,
            'removed_features': removed_count,
            'auto_migrations_available': len([e for e in self.deprecation_registry.values() if e.auto_migration_available])
        })
        
        return {
            **self.deprecation_stats,
            'registry_entries': len(self.deprecation_registry),
            'compatibility_reports': len(self.compatibility_reports),
            'current_dsl_version': self.current_dsl_version
        }
    
    def _initialize_standard_deprecations(self):
        """Initialize standard deprecation entries"""
        
        # Example deprecations for common DSL evolution patterns
        standard_deprecations = [
            {
                'feature': 'old_syntax_rule',
                'type': DeprecationCategory.SYNTAX,
                'deprecated_in': '1.0.0',
                'remove_in': '2.0.0',
                'hint': 'Use new_syntax_rule instead',
                'reason': 'Simplified syntax for better readability',
                'replacement': 'new_syntax_rule',
                'complexity': MigrationComplexity.TRIVIAL,
                'examples_old': ['old_syntax_rule: condition'],
                'examples_new': ['new_syntax_rule: condition']
            },
            {
                'feature': 'legacy_ml_config',
                'type': DeprecationCategory.CONFIGURATION,
                'deprecated_in': '1.1.0',
                'remove_in': '2.0.0',
                'hint': 'Use ml_node configuration instead',
                'reason': 'Unified ML node configuration',
                'replacement': 'ml_node',
                'complexity': MigrationComplexity.MODERATE,
                'examples_old': ['legacy_ml_config: { model: "xyz" }'],
                'examples_new': ['ml_node: { model_id: "xyz", type: "predict" }']
            },
            {
                'feature': 'deprecated_operator',
                'type': DeprecationCategory.OPERATOR,
                'deprecated_in': '1.0.0',
                'remove_in': '1.5.0',
                'hint': 'Use standard_operator instead',
                'reason': 'Operator naming consistency',
                'replacement': 'standard_operator',
                'complexity': MigrationComplexity.SIMPLE,
                'auto_migration': True
            }
        ]
        
        for dep in standard_deprecations:
            entry = DeprecationEntry(
                deprecated_feature=dep['feature'],
                feature_type=dep['type'],
                deprecated_in_version=dep['deprecated_in'],
                remove_in_version=dep['remove_in'],
                migration_hint=dep['hint'],
                deprecation_reason=dep['reason'],
                replacement_syntax=dep.get('replacement'),
                migration_complexity=dep['complexity'],
                impact_description=f"Feature {dep['feature']} will be removed",
                examples_old=dep.get('examples_old', []),
                examples_new=dep.get('examples_new', []),
                auto_migration_available=dep.get('auto_migration', False)
            )
            
            self.deprecation_registry[dep['feature']] = entry
        
        self._update_deprecation_stats()
        
        self.logger.info(f"✅ Initialized {len(standard_deprecations)} standard deprecation entries")
    
    def _feature_found_in_line(self, feature: str, line: str, feature_type: DeprecationCategory) -> bool:
        """Check if deprecated feature is found in a line of code"""
        # Simple pattern matching - in a real implementation this would be more sophisticated
        if feature_type == DeprecationCategory.KEYWORD:
            # Look for keyword usage
            pattern = r'\b' + re.escape(feature) + r'\b'
            return bool(re.search(pattern, line))
        elif feature_type == DeprecationCategory.SYNTAX:
            # Look for syntax pattern
            return feature in line
        elif feature_type == DeprecationCategory.OPERATOR:
            # Look for operator usage
            return feature in line
        else:
            # General text search
            return feature in line
    
    def _create_deprecation_warning(self, deprecated_feature: str, entry: DeprecationEntry,
                                  file_path: Optional[str], line_number: int, line_content: str) -> DeprecationWarning:
        """Create a deprecation warning"""
        warning_id = f"dep_{hash(f'{deprecated_feature}_{file_path}_{line_number}')}"
        
        severity = self.get_deprecation_severity(deprecated_feature)
        
        # Calculate timeline information
        removal_timeline = f"Will be removed in version {entry.remove_in_version}"
        
        # Calculate grace period remaining
        grace_period_remaining = None
        if entry.last_usage_detected:
            grace_expiry = entry.created_at + timedelta(days=entry.grace_period_months * 30)
            if datetime.now(timezone.utc) < grace_expiry:
                days_remaining = (grace_expiry - datetime.now(timezone.utc)).days
                grace_period_remaining = f"{days_remaining} days"
        
        warning = DeprecationWarning(
            warning_id=warning_id,
            deprecated_feature=deprecated_feature,
            severity=severity,
            file_path=file_path,
            line_number=line_number,
            source_context=line_content.strip(),
            deprecation_entry=entry,
            migration_hint=entry.migration_hint,
            suggested_replacement=entry.replacement_syntax,
            removal_timeline=removal_timeline,
            grace_period_remaining=grace_period_remaining,
            auto_fix_available=entry.auto_migration_available
        )
        
        if entry.auto_migration_available:
            warning.auto_fix_command = self._generate_auto_migration_command(deprecated_feature, entry)
        
        return warning
    
    def _estimate_migration_effort(self, entry: DeprecationEntry) -> str:
        """Estimate migration effort for a deprecation"""
        if entry.migration_complexity == MigrationComplexity.TRIVIAL:
            return "< 1 hour"
        elif entry.migration_complexity == MigrationComplexity.SIMPLE:
            return "1-4 hours"
        elif entry.migration_complexity == MigrationComplexity.MODERATE:
            return "1-2 days"
        elif entry.migration_complexity == MigrationComplexity.COMPLEX:
            return "1-2 weeks"
        else:  # BREAKING
            return "2+ weeks"
    
    def _generate_auto_migration_command(self, deprecated_feature: str, entry: DeprecationEntry) -> str:
        """Generate auto-migration command"""
        if entry.replacement_syntax:
            return f"rbia migrate --replace '{deprecated_feature}' --with '{entry.replacement_syntax}'"
        else:
            return f"rbia migrate --deprecation '{deprecated_feature}'"
    
    def _generate_compatibility_recommendations(self, report: CompatibilityReport) -> List[str]:
        """Generate compatibility recommendations"""
        recommendations = []
        
        if report.breaking_changes > 0:
            recommendations.append(f"Review {report.breaking_changes} breaking changes before upgrading")
        
        if report.complex_migrations > 0:
            recommendations.append(f"Plan for {report.complex_migrations} complex migrations")
        
        if report.features_to_remove_next_version:
            recommendations.append(f"Prepare for removal of {len(report.features_to_remove_next_version)} features in next version")
        
        if report.trivial_migrations > report.complex_migrations:
            recommendations.append("Consider applying trivial migrations first to reduce technical debt")
        
        return recommendations
    
    def _identify_upgrade_blockers(self, report: CompatibilityReport) -> List[str]:
        """Identify upgrade blockers"""
        blockers = []
        
        # Critical severity items block upgrades
        critical_count = report.deprecations_by_severity.get(DeprecationSeverity.CRITICAL, 0)
        if critical_count > 0:
            blockers.append(f"{critical_count} features have been removed and must be migrated")
        
        # Breaking changes are blockers
        if report.breaking_changes > 0:
            blockers.append(f"{report.breaking_changes} breaking changes require manual intervention")
        
        return blockers
    
    def _export_as_markdown(self) -> str:
        """Export deprecation registry as markdown"""
        lines = []
        lines.append("# DSL Deprecation Registry")
        lines.append("")
        lines.append(f"**Current Version:** {self.current_dsl_version}")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"**Total Entries:** {len(self.deprecation_registry)}")
        lines.append("")
        
        # Group by category
        by_category = {}
        for entry in self.deprecation_registry.values():
            category = entry.feature_type.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(entry)
        
        for category, entries in by_category.items():
            lines.append(f"## {category.title()}")
            lines.append("")
            
            for entry in sorted(entries, key=lambda x: x.deprecated_feature):
                severity = self.get_deprecation_severity(entry.deprecated_feature)
                lines.append(f"### {entry.deprecated_feature}")
                lines.append("")
                lines.append(f"- **Status:** {severity.value.title()}")
                lines.append(f"- **Deprecated in:** {entry.deprecated_in_version}")
                lines.append(f"- **Remove in:** {entry.remove_in_version}")
                lines.append(f"- **Migration:** {entry.migration_hint}")
                if entry.replacement_syntax:
                    lines.append(f"- **Replacement:** `{entry.replacement_syntax}`")
                lines.append(f"- **Complexity:** {entry.migration_complexity.value}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _update_deprecation_stats(self):
        """Update deprecation statistics"""
        self.deprecation_stats['total_entries'] = len(self.deprecation_registry)

# API Interface
class DeprecationPolicyAPI:
    """API interface for deprecation policy operations"""
    
    def __init__(self, deprecation_service: Optional[DeprecationPolicyService] = None):
        self.deprecation_service = deprecation_service or DeprecationPolicyService()
    
    def check_deprecations(self, source_code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """API endpoint to check for deprecations"""
        try:
            warnings = self.deprecation_service.check_for_deprecations(source_code, file_path)
            
            return {
                'success': True,
                'file_path': file_path,
                'warnings_found': len(warnings),
                'warnings': [
                    {
                        'deprecated_feature': w.deprecated_feature,
                        'severity': w.severity.value,
                        'line_number': w.line_number,
                        'migration_hint': w.migration_hint,
                        'auto_fix_available': w.auto_fix_available,
                        'removal_timeline': w.removal_timeline
                    }
                    for w in warnings
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_migration_suggestions(self, deprecated_feature: str) -> Dict[str, Any]:
        """API endpoint to get migration suggestions"""
        suggestions = self.deprecation_service.generate_migration_suggestions(deprecated_feature)
        
        return {
            'success': 'error' not in suggestions,
            **suggestions
        }
    
    def create_compatibility_report(self, target_version: str) -> Dict[str, Any]:
        """API endpoint to create compatibility report"""
        try:
            report = self.deprecation_service.create_compatibility_report(target_version)
            
            return {
                'success': True,
                'report': asdict(report)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_deprecation_timeline(self, months_ahead: int = 12) -> Dict[str, Any]:
        """API endpoint to get deprecation timeline"""
        timeline = self.deprecation_service.get_deprecation_timeline(months_ahead)
        
        return {
            'success': True,
            'timeline': timeline
        }

# Test Functions
def run_deprecation_policy_tests():
    """Run comprehensive deprecation policy tests"""
    print("=== Deprecation Policy Service Tests ===")
    
    # Initialize service
    deprecation_service = DeprecationPolicyService()
    deprecation_api = DeprecationPolicyAPI(deprecation_service)
    
    # Test 1: Standard deprecations initialization
    print("\n1. Testing standard deprecations initialization...")
    stats = deprecation_service.get_deprecation_statistics()
    print(f"   Registry entries loaded: {stats['registry_entries']}")
    print(f"   Active deprecations: {stats['active_deprecations']}")
    print(f"   Auto-migrations available: {stats['auto_migrations_available']}")
    
    # Test 2: Custom deprecation registration
    print("\n2. Testing custom deprecation registration...")
    
    custom_entry = deprecation_service.register_deprecation(
        deprecated_feature="test_deprecated_keyword",
        feature_type=DeprecationCategory.KEYWORD,
        deprecated_in_version="1.2.0",
        remove_in_version="2.0.0",
        migration_hint="Use new_keyword instead",
        deprecation_reason="Improved functionality and consistency",
        replacement_syntax="new_keyword",
        migration_complexity=MigrationComplexity.SIMPLE
    )
    
    print(f"   Custom deprecation registered: {custom_entry.deprecated_feature}")
    print(f"   Migration complexity: {custom_entry.migration_complexity.value}")
    print(f"   Grace period: {custom_entry.grace_period_months} months")
    
    # Test 3: Deprecation detection in source code
    print("\n3. Testing deprecation detection...")
    
    test_source_code = """
# Test DSL code with deprecated features
workflow test_workflow:
    step1:
        old_syntax_rule: customer_age > 25
        action: approve
    
    step2:
        legacy_ml_config:
            model: "churn_predictor"
            threshold: 0.8
        
    step3:
        test_deprecated_keyword: some_value
        deprecated_operator: calculate_score()
"""
    
    warnings = deprecation_service.check_for_deprecations(test_source_code, "test_workflow.yaml")
    print(f"   Deprecation warnings found: {len(warnings)}")
    
    for warning in warnings:
        print(f"     - {warning.deprecated_feature} (line {warning.line_number}): {warning.severity.value}")
        print(f"       Hint: {warning.migration_hint}")
        if warning.auto_fix_available:
            print(f"       Auto-fix: {warning.auto_fix_command}")
    
    # Test 4: Severity determination
    print("\n4. Testing severity determination...")
    
    test_features = ["old_syntax_rule", "legacy_ml_config", "test_deprecated_keyword"]
    test_versions = ["1.0.0", "1.5.0", "2.0.0"]
    
    for feature in test_features:
        print(f"   {feature}:")
        for version in test_versions:
            severity = deprecation_service.get_deprecation_severity(feature, version)
            print(f"     Version {version}: {severity.value}")
    
    # Test 5: Migration suggestions
    print("\n5. Testing migration suggestions...")
    
    for feature in test_features[:2]:  # Test first 2 features
        suggestions = deprecation_service.generate_migration_suggestions(feature)
        if 'error' not in suggestions:
            print(f"   {feature}:")
            print(f"     Migration hint: {suggestions['migration_hint']}")
            print(f"     Replacement: {suggestions['replacement_syntax']}")
            print(f"     Complexity: {suggestions['migration_complexity']}")
            print(f"     Estimated effort: {suggestions['estimated_effort']}")
    
    # Test 6: Compatibility report generation
    print("\n6. Testing compatibility report generation...")
    
    target_versions = ["1.5.0", "2.0.0"]
    for target_version in target_versions:
        report = deprecation_service.create_compatibility_report(target_version)
        print(f"   Version {target_version}:")
        print(f"     Total deprecations: {report.total_deprecations}")
        print(f"     New deprecations: {report.new_deprecations}")
        print(f"     Features removed: {report.removed_features}")
        print(f"     Breaking changes: {report.breaking_changes}")
        print(f"     Recommendations: {len(report.migration_recommendations)}")
    
    # Test 7: Deprecation timeline
    print("\n7. Testing deprecation timeline...")
    
    timeline = deprecation_service.get_deprecation_timeline(18)  # 18 months ahead
    print(f"   Timeline for next {timeline['timeline_months']} months:")
    print(f"   Upcoming removals: {len(timeline['upcoming_removals'])}")
    print(f"   Grace periods expiring: {len(timeline['grace_period_expiring'])}")
    
    for removal in timeline['upcoming_removals'][:3]:  # Show first 3
        print(f"     - {removal['feature']} in ~{removal['estimated_months']} months")
    
    # Test 8: Registry export
    print("\n8. Testing registry export...")
    
    try:
        json_export = deprecation_service.export_deprecation_registry("json")
        json_data = json.loads(json_export)
        print(f"   JSON export: {json_data['total_entries']} entries")
        
        markdown_export = deprecation_service.export_deprecation_registry("markdown")
        print(f"   Markdown export: {len(markdown_export)} characters")
        
        export_success = True
    except Exception as e:
        print(f"   Export failed: {e}")
        export_success = False
    
    print(f"   Registry export: {'✅ PASS' if export_success else '❌ FAIL'}")
    
    # Test 9: API interface
    print("\n9. Testing API interface...")
    
    # Test API deprecation check
    api_check_result = deprecation_api.check_deprecations(test_source_code, "api_test.yaml")
    print(f"   API deprecation check: {'✅ PASS' if api_check_result['success'] else '❌ FAIL'}")
    if api_check_result['success']:
        print(f"   API warnings found: {api_check_result['warnings_found']}")
    
    # Test API migration suggestions
    api_suggestions_result = deprecation_api.get_migration_suggestions("old_syntax_rule")
    print(f"   API migration suggestions: {'✅ PASS' if api_suggestions_result['success'] else '❌ FAIL'}")
    
    # Test API compatibility report
    api_report_result = deprecation_api.create_compatibility_report("2.0.0")
    print(f"   API compatibility report: {'✅ PASS' if api_report_result['success'] else '❌ FAIL'}")
    
    # Test 10: Usage tracking
    print("\n10. Testing usage tracking...")
    
    # Check usage counts after running deprecation detection
    for feature in test_features:
        if feature in deprecation_service.deprecation_registry:
            entry = deprecation_service.deprecation_registry[feature]
            print(f"   {feature}: {entry.usage_count} usages detected")
            if entry.last_usage_detected:
                print(f"     Last detected: {entry.last_usage_detected.strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n=== Test Summary ===")
    final_stats = deprecation_service.get_deprecation_statistics()
    print(f"Deprecation policy service tested successfully")
    print(f"Registry entries: {final_stats['registry_entries']}")
    print(f"Active deprecations: {final_stats['active_deprecations']}")
    print(f"Warnings generated: {final_stats['warnings_generated']}")
    print(f"Compatibility reports: {final_stats['compatibility_reports']}")
    
    return deprecation_service, deprecation_api

if __name__ == "__main__":
    run_deprecation_policy_tests()



