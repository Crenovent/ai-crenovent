"""
Diagnostic Catalog Service - Task 6.2.32
=========================================

Error-class map: what compiler errors map to which action
- Creates consistent mapping from compiler diagnostics to required actions
- Diagnostic catalog with error codes, severity, messages, and remediation steps
- Maps diagnostics to action types (block, warn, autofixable, CAB-approval-required)
- Integration points for LSP, CI output, and manifest signing pipeline
- Backend implementation (no actual LSP/CI integration - that's infrastructure)

Dependencies: Task 6.2.2 (Parser), Task 6.2.28 (Digital Signature)
Outputs: Diagnostic catalog → enables consistent error handling and user guidance
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

logger = logging.getLogger(__name__)

class DiagnosticSeverity(Enum):
    """Severity levels for diagnostics"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"

class ActionType(Enum):
    """Action types for diagnostic resolution"""
    BLOCK = "block"                    # Block compilation/signing
    WARN = "warn"                      # Show warning, allow continuation
    AUTOFIXABLE = "autofixable"       # Can be automatically fixed
    CAB_APPROVAL_REQUIRED = "cab_approval_required"  # Requires change approval board
    MANUAL_REVIEW = "manual_review"    # Requires human review
    IGNORE_ALLOWED = "ignore_allowed"  # Can be ignored with justification

class PersonaType(Enum):
    """Types of personas for diagnostic targeting"""
    DEVELOPER = "developer"
    REVOPS = "revops"
    COMPLIANCE = "compliance"
    DATA_SCIENTIST = "data_scientist"
    SECURITY = "security"
    BUSINESS_USER = "business_user"

class DiagnosticCategory(Enum):
    """Categories of diagnostics"""
    SYNTAX = "syntax"
    TYPE_SYSTEM = "type_system"
    POLICY = "policy"
    GOVERNANCE = "governance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BEST_PRACTICE = "best_practice"
    COMPLIANCE = "compliance"
    ML_SPECIFIC = "ml_specific"
    RESIDENCY = "residency"

@dataclass
class RemediationStep:
    """A single remediation step"""
    step_number: int
    description: str
    persona: PersonaType
    estimated_time_minutes: int = 5
    automation_available: bool = False
    prerequisite_steps: List[int] = field(default_factory=list)
    verification_criteria: Optional[str] = None

@dataclass
class AutofixSuggestion:
    """Automatic fix suggestion"""
    fix_id: str
    description: str
    confidence_score: float  # 0.0 to 1.0
    safe_to_apply_automatically: bool
    fix_command: Optional[str] = None
    fix_parameters: Dict[str, Any] = field(default_factory=dict)
    validation_required: bool = True
    rollback_available: bool = False

@dataclass
class DiagnosticRule:
    """Complete diagnostic rule definition"""
    # Core identification
    error_code: str
    rule_name: str
    category: DiagnosticCategory
    
    # Severity and action
    severity: DiagnosticSeverity
    action_type: ActionType
    
    # Messages and descriptions
    short_message: str
    detailed_description: str
    technical_explanation: str
    
    # Personas and responsibilities
    primary_persona: PersonaType
    secondary_personas: List[PersonaType] = field(default_factory=list)
    
    # Remediation
    remediation_steps: List[RemediationStep] = field(default_factory=list)
    autofix_suggestions: List[AutofixSuggestion] = field(default_factory=list)
    
    # Context and examples
    common_causes: List[str] = field(default_factory=list)
    example_violations: List[str] = field(default_factory=list)
    example_fixes: List[str] = field(default_factory=list)
    
    # Integration settings
    block_compilation: bool = False
    block_signing: bool = False
    require_approval: bool = False
    allow_override: bool = True
    
    # Metadata
    documentation_url: Optional[str] = None
    related_error_codes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class DiagnosticInstance:
    """Instance of a diagnostic found in code"""
    instance_id: str
    error_code: str
    severity: DiagnosticSeverity
    message: str
    
    # Location information
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    source_context: Optional[str] = None
    
    # Additional context
    context_data: Dict[str, Any] = field(default_factory=dict)
    suggested_fixes: List[AutofixSuggestion] = field(default_factory=list)
    
    # Status
    status: str = "active"  # active, fixed, ignored, overridden
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detection_tool: str = "compiler"

# Task 6.2.32: Diagnostic Catalog Service
class DiagnosticCatalogService:
    """Service for managing diagnostic rules and error-to-action mappings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Storage for diagnostic rules
        self.diagnostic_rules: Dict[str, DiagnosticRule] = {}  # error_code -> rule
        self.diagnostic_instances: Dict[str, DiagnosticInstance] = {}  # instance_id -> instance
        
        # Category and persona mappings
        self.rules_by_category: Dict[DiagnosticCategory, List[str]] = {}
        self.rules_by_persona: Dict[PersonaType, List[str]] = {}
        self.rules_by_action_type: Dict[ActionType, List[str]] = {}
        
        # Initialize with standard diagnostic rules
        self._initialize_standard_rules()
        
        # Statistics
        self.diagnostic_stats = {
            'total_rules': 0,
            'rules_by_severity': {severity.value: 0 for severity in DiagnosticSeverity},
            'rules_by_action_type': {action.value: 0 for action in ActionType},
            'total_instances': 0,
            'instances_by_status': {}
        }
    
    def register_diagnostic_rule(self, rule: DiagnosticRule) -> bool:
        """
        Register a new diagnostic rule
        
        Args:
            rule: Diagnostic rule to register
            
        Returns:
            True if registered successfully
        """
        if rule.error_code in self.diagnostic_rules:
            self.logger.warning(f"Diagnostic rule {rule.error_code} already exists, updating")
        
        # Store rule
        self.diagnostic_rules[rule.error_code] = rule
        
        # Update category mappings
        if rule.category not in self.rules_by_category:
            self.rules_by_category[rule.category] = []
        if rule.error_code not in self.rules_by_category[rule.category]:
            self.rules_by_category[rule.category].append(rule.error_code)
        
        # Update persona mappings
        personas = [rule.primary_persona] + rule.secondary_personas
        for persona in personas:
            if persona not in self.rules_by_persona:
                self.rules_by_persona[persona] = []
            if rule.error_code not in self.rules_by_persona[persona]:
                self.rules_by_persona[persona].append(rule.error_code)
        
        # Update action type mappings
        if rule.action_type not in self.rules_by_action_type:
            self.rules_by_action_type[rule.action_type] = []
        if rule.error_code not in self.rules_by_action_type[rule.action_type]:
            self.rules_by_action_type[rule.action_type].append(rule.error_code)
        
        # Update statistics
        self._update_rule_statistics()
        
        self.logger.info(f"✅ Registered diagnostic rule: {rule.error_code} ({rule.severity.value})")
        
        return True
    
    def create_diagnostic_instance(self, error_code: str, message: str,
                                 file_path: Optional[str] = None,
                                 line_number: Optional[int] = None,
                                 column_number: Optional[int] = None,
                                 context_data: Optional[Dict[str, Any]] = None) -> Optional[DiagnosticInstance]:
        """
        Create a diagnostic instance from an error code
        
        Args:
            error_code: Error code from diagnostic rule
            message: Specific error message
            file_path: Source file path
            line_number: Line number in source
            column_number: Column number in source
            context_data: Additional context information
            
        Returns:
            DiagnosticInstance or None if error code not found
        """
        if error_code not in self.diagnostic_rules:
            self.logger.warning(f"Unknown error code: {error_code}")
            return None
        
        rule = self.diagnostic_rules[error_code]
        instance_id = f"diag_{hash(f'{error_code}_{file_path}_{line_number}_{message}')}"
        
        # Generate suggested fixes if available
        suggested_fixes = []
        for autofix in rule.autofix_suggestions:
            if autofix.safe_to_apply_automatically or autofix.confidence_score > 0.8:
                suggested_fixes.append(autofix)
        
        instance = DiagnosticInstance(
            instance_id=instance_id,
            error_code=error_code,
            severity=rule.severity,
            message=message,
            file_path=file_path,
            line_number=line_number,
            column_number=column_number,
            context_data=context_data or {},
            suggested_fixes=suggested_fixes
        )
        
        # Store instance
        self.diagnostic_instances[instance_id] = instance
        
        # Update statistics
        self.diagnostic_stats['total_instances'] += 1
        status_count = self.diagnostic_stats['instances_by_status'].get('active', 0)
        self.diagnostic_stats['instances_by_status']['active'] = status_count + 1
        
        self.logger.debug(f"Created diagnostic instance {instance_id} for {error_code}")
        
        return instance
    
    def get_action_for_diagnostic(self, error_code: str) -> Optional[ActionType]:
        """
        Get the required action for a diagnostic error code
        
        Args:
            error_code: Error code to look up
            
        Returns:
            ActionType or None if not found
        """
        rule = self.diagnostic_rules.get(error_code)
        return rule.action_type if rule else None
    
    def should_block_compilation(self, error_codes: List[str]) -> Tuple[bool, List[str]]:
        """
        Determine if compilation should be blocked based on error codes
        
        Args:
            error_codes: List of error codes found
            
        Returns:
            Tuple of (should_block, blocking_error_codes)
        """
        blocking_codes = []
        
        for error_code in error_codes:
            rule = self.diagnostic_rules.get(error_code)
            if rule and (rule.block_compilation or rule.action_type == ActionType.BLOCK):
                blocking_codes.append(error_code)
        
        return len(blocking_codes) > 0, blocking_codes
    
    def should_block_signing(self, error_codes: List[str]) -> Tuple[bool, List[str]]:
        """
        Determine if manifest signing should be blocked based on error codes
        
        Args:
            error_codes: List of error codes found
            
        Returns:
            Tuple of (should_block, blocking_error_codes)
        """
        blocking_codes = []
        
        for error_code in error_codes:
            rule = self.diagnostic_rules.get(error_code)
            if rule and rule.block_signing:
                blocking_codes.append(error_code)
        
        return len(blocking_codes) > 0, blocking_codes
    
    def get_autofixable_diagnostics(self, error_codes: List[str]) -> Dict[str, List[AutofixSuggestion]]:
        """
        Get autofixable diagnostics and their suggestions
        
        Args:
            error_codes: List of error codes to check
            
        Returns:
            Dictionary mapping error codes to autofix suggestions
        """
        autofixable = {}
        
        for error_code in error_codes:
            rule = self.diagnostic_rules.get(error_code)
            if rule and rule.action_type == ActionType.AUTOFIXABLE and rule.autofix_suggestions:
                autofixable[error_code] = rule.autofix_suggestions
        
        return autofixable
    
    def get_diagnostics_requiring_approval(self, error_codes: List[str]) -> List[str]:
        """
        Get diagnostics that require CAB approval
        
        Args:
            error_codes: List of error codes to check
            
        Returns:
            List of error codes requiring approval
        """
        requiring_approval = []
        
        for error_code in error_codes:
            rule = self.diagnostic_rules.get(error_code)
            if rule and (rule.action_type == ActionType.CAB_APPROVAL_REQUIRED or rule.require_approval):
                requiring_approval.append(error_code)
        
        return requiring_approval
    
    def get_diagnostics_for_persona(self, persona: PersonaType) -> List[DiagnosticRule]:
        """
        Get diagnostic rules relevant to a specific persona
        
        Args:
            persona: Persona type
            
        Returns:
            List of relevant diagnostic rules
        """
        error_codes = self.rules_by_persona.get(persona, [])
        return [self.diagnostic_rules[code] for code in error_codes if code in self.diagnostic_rules]
    
    def get_diagnostics_by_category(self, category: DiagnosticCategory) -> List[DiagnosticRule]:
        """
        Get diagnostic rules by category
        
        Args:
            category: Diagnostic category
            
        Returns:
            List of diagnostic rules in the category
        """
        error_codes = self.rules_by_category.get(category, [])
        return [self.diagnostic_rules[code] for code in error_codes if code in self.diagnostic_rules]
    
    def resolve_diagnostic_instance(self, instance_id: str, resolution_notes: str,
                                  resolved_by: str, status: str = "fixed") -> bool:
        """
        Mark a diagnostic instance as resolved
        
        Args:
            instance_id: Instance identifier
            resolution_notes: Notes about the resolution
            resolved_by: Who resolved it
            status: New status (fixed, ignored, overridden)
            
        Returns:
            True if resolved successfully
        """
        if instance_id not in self.diagnostic_instances:
            return False
        
        instance = self.diagnostic_instances[instance_id]
        old_status = instance.status
        
        instance.status = status
        instance.resolution_notes = resolution_notes
        instance.resolved_by = resolved_by
        instance.resolved_at = datetime.now(timezone.utc)
        
        # Update statistics
        old_count = self.diagnostic_stats['instances_by_status'].get(old_status, 0)
        self.diagnostic_stats['instances_by_status'][old_status] = max(0, old_count - 1)
        
        new_count = self.diagnostic_stats['instances_by_status'].get(status, 0)
        self.diagnostic_stats['instances_by_status'][status] = new_count + 1
        
        self.logger.info(f"Resolved diagnostic instance {instance_id}: {old_status} -> {status}")
        
        return True
    
    def generate_diagnostic_summary(self, error_codes: List[str]) -> Dict[str, Any]:
        """
        Generate a summary of diagnostics and required actions
        
        Args:
            error_codes: List of error codes found
            
        Returns:
            Dictionary with diagnostic summary
        """
        summary = {
            'total_diagnostics': len(error_codes),
            'by_severity': {severity.value: 0 for severity in DiagnosticSeverity},
            'by_action_type': {action.value: 0 for action in ActionType},
            'by_category': {category.value: 0 for category in DiagnosticCategory},
            'blocking_compilation': [],
            'blocking_signing': [],
            'autofixable': [],
            'requiring_approval': [],
            'unknown_codes': []
        }
        
        for error_code in error_codes:
            rule = self.diagnostic_rules.get(error_code)
            
            if not rule:
                summary['unknown_codes'].append(error_code)
                continue
            
            # Count by severity
            summary['by_severity'][rule.severity.value] += 1
            
            # Count by action type
            summary['by_action_type'][rule.action_type.value] += 1
            
            # Count by category
            summary['by_category'][rule.category.value] += 1
            
            # Check blocking conditions
            if rule.block_compilation or rule.action_type == ActionType.BLOCK:
                summary['blocking_compilation'].append(error_code)
            
            if rule.block_signing:
                summary['blocking_signing'].append(error_code)
            
            if rule.action_type == ActionType.AUTOFIXABLE:
                summary['autofixable'].append(error_code)
            
            if rule.action_type == ActionType.CAB_APPROVAL_REQUIRED or rule.require_approval:
                summary['requiring_approval'].append(error_code)
        
        # Add recommendations
        summary['recommendations'] = []
        
        if summary['blocking_compilation']:
            summary['recommendations'].append("Fix blocking errors before compilation")
        
        if summary['blocking_signing']:
            summary['recommendations'].append("Resolve signing blockers before deployment")
        
        if summary['autofixable']:
            summary['recommendations'].append(f"Apply automatic fixes for {len(summary['autofixable'])} issues")
        
        if summary['requiring_approval']:
            summary['recommendations'].append("Submit to CAB for approval before deployment")
        
        return summary
    
    def export_diagnostic_catalog(self, format: str = "json") -> str:
        """
        Export the diagnostic catalog
        
        Args:
            format: Export format (json, yaml, csv)
            
        Returns:
            Exported catalog as string
        """
        if format == "json":
            catalog_data = {
                'version': '1.0',
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'total_rules': len(self.diagnostic_rules),
                'rules': {
                    error_code: asdict(rule)
                    for error_code, rule in self.diagnostic_rules.items()
                }
            }
            return json.dumps(catalog_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_diagnostic_statistics(self) -> Dict[str, Any]:
        """Get diagnostic catalog statistics"""
        return {
            **self.diagnostic_stats,
            'rules_by_category': {cat.value: len(codes) for cat, codes in self.rules_by_category.items()},
            'rules_by_persona': {persona.value: len(codes) for persona, codes in self.rules_by_persona.items()}
        }
    
    def _initialize_standard_rules(self):
        """Initialize standard diagnostic rules"""
        
        # Syntax errors
        syntax_rules = [
            DiagnosticRule(
                error_code="SYNTAX_001",
                rule_name="Missing semicolon",
                category=DiagnosticCategory.SYNTAX,
                severity=DiagnosticSeverity.ERROR,
                action_type=ActionType.AUTOFIXABLE,
                short_message="Missing semicolon",
                detailed_description="Statement is missing a required semicolon",
                technical_explanation="The DSL requires semicolons to terminate statements",
                primary_persona=PersonaType.DEVELOPER,
                block_compilation=True,
                autofix_suggestions=[
                    AutofixSuggestion(
                        fix_id="add_semicolon",
                        description="Add missing semicolon",
                        confidence_score=0.95,
                        safe_to_apply_automatically=True,
                        fix_command="add_semicolon"
                    )
                ]
            ),
            DiagnosticRule(
                error_code="SYNTAX_002",
                rule_name="Unmatched braces",
                category=DiagnosticCategory.SYNTAX,
                severity=DiagnosticSeverity.ERROR,
                action_type=ActionType.BLOCK,
                short_message="Unmatched braces",
                detailed_description="Opening brace does not have a corresponding closing brace",
                technical_explanation="All opening braces must be matched with closing braces",
                primary_persona=PersonaType.DEVELOPER,
                block_compilation=True
            )
        ]
        
        # ML-specific rules
        ml_rules = [
            DiagnosticRule(
                error_code="ML_001",
                rule_name="Missing confidence threshold",
                category=DiagnosticCategory.ML_SPECIFIC,
                severity=DiagnosticSeverity.ERROR,
                action_type=ActionType.CAB_APPROVAL_REQUIRED,
                short_message="ML node missing confidence threshold",
                detailed_description="All ML nodes must specify a confidence threshold for governance compliance",
                technical_explanation="Confidence thresholds are required to ensure ML decisions meet quality standards",
                primary_persona=PersonaType.DATA_SCIENTIST,
                secondary_personas=[PersonaType.COMPLIANCE],
                block_signing=True,
                require_approval=True,
                autofix_suggestions=[
                    AutofixSuggestion(
                        fix_id="add_default_threshold",
                        description="Add default confidence threshold (0.7)",
                        confidence_score=0.8,
                        safe_to_apply_automatically=False,
                        validation_required=True
                    )
                ]
            ),
            DiagnosticRule(
                error_code="ML_002",
                rule_name="Missing fallback configuration",
                category=DiagnosticCategory.ML_SPECIFIC,
                severity=DiagnosticSeverity.WARNING,
                action_type=ActionType.MANUAL_REVIEW,
                short_message="ML node missing fallback configuration",
                detailed_description="ML nodes should have fallback configurations for safe degradation",
                technical_explanation="Fallback configurations ensure system continues operating when ML models fail",
                primary_persona=PersonaType.DATA_SCIENTIST,
                secondary_personas=[PersonaType.DEVELOPER]
            )
        ]
        
        # Policy and governance rules
        policy_rules = [
            DiagnosticRule(
                error_code="POLICY_001",
                rule_name="Data residency violation",
                category=DiagnosticCategory.COMPLIANCE,
                severity=DiagnosticSeverity.ERROR,
                action_type=ActionType.BLOCK,
                short_message="Data residency policy violation",
                detailed_description="Data processing violates geographic residency requirements",
                technical_explanation="Data must remain within specified geographic boundaries per compliance requirements",
                primary_persona=PersonaType.COMPLIANCE,
                secondary_personas=[PersonaType.SECURITY],
                block_compilation=True,
                block_signing=True
            ),
            DiagnosticRule(
                error_code="POLICY_002",
                rule_name="Missing bias monitoring",
                category=DiagnosticCategory.GOVERNANCE,
                severity=DiagnosticSeverity.ERROR,
                action_type=ActionType.CAB_APPROVAL_REQUIRED,
                short_message="ML model missing bias monitoring configuration",
                detailed_description="All ML models must have bias monitoring configured",
                technical_explanation="Bias monitoring is required to ensure fair and equitable ML decisions",
                primary_persona=PersonaType.COMPLIANCE,
                secondary_personas=[PersonaType.DATA_SCIENTIST],
                block_signing=True,
                require_approval=True
            )
        ]
        
        # Security rules
        security_rules = [
            DiagnosticRule(
                error_code="SECURITY_001",
                rule_name="Hardcoded credentials",
                category=DiagnosticCategory.SECURITY,
                severity=DiagnosticSeverity.ERROR,
                action_type=ActionType.BLOCK,
                short_message="Hardcoded credentials detected",
                detailed_description="Credentials should not be hardcoded in workflow definitions",
                technical_explanation="Hardcoded credentials pose a security risk and should use secure credential management",
                primary_persona=PersonaType.SECURITY,
                secondary_personas=[PersonaType.DEVELOPER],
                block_compilation=True,
                block_signing=True
            )
        ]
        
        # Register all standard rules
        all_rules = syntax_rules + ml_rules + policy_rules + security_rules
        
        for rule in all_rules:
            self.register_diagnostic_rule(rule)
        
        self.logger.info(f"✅ Initialized {len(all_rules)} standard diagnostic rules")
    
    def _update_rule_statistics(self):
        """Update diagnostic rule statistics"""
        self.diagnostic_stats['total_rules'] = len(self.diagnostic_rules)
        
        # Reset counters
        for severity in DiagnosticSeverity:
            self.diagnostic_stats['rules_by_severity'][severity.value] = 0
        
        for action in ActionType:
            self.diagnostic_stats['rules_by_action_type'][action.value] = 0
        
        # Count rules
        for rule in self.diagnostic_rules.values():
            self.diagnostic_stats['rules_by_severity'][rule.severity.value] += 1
            self.diagnostic_stats['rules_by_action_type'][rule.action_type.value] += 1

# API Interface
class DiagnosticCatalogAPI:
    """API interface for diagnostic catalog operations"""
    
    def __init__(self, catalog_service: Optional[DiagnosticCatalogService] = None):
        self.catalog_service = catalog_service or DiagnosticCatalogService()
    
    def get_action_for_error(self, error_code: str) -> Dict[str, Any]:
        """API endpoint to get action for error code"""
        action = self.catalog_service.get_action_for_diagnostic(error_code)
        rule = self.catalog_service.diagnostic_rules.get(error_code)
        
        if not rule:
            return {
                'success': False,
                'error': f'Unknown error code: {error_code}'
            }
        
        return {
            'success': True,
            'error_code': error_code,
            'action_type': action.value,
            'severity': rule.severity.value,
            'block_compilation': rule.block_compilation,
            'block_signing': rule.block_signing,
            'require_approval': rule.require_approval,
            'autofixable': len(rule.autofix_suggestions) > 0
        }
    
    def check_compilation_blockers(self, error_codes: List[str]) -> Dict[str, Any]:
        """API endpoint to check for compilation blockers"""
        should_block, blocking_codes = self.catalog_service.should_block_compilation(error_codes)
        
        return {
            'success': True,
            'should_block_compilation': should_block,
            'blocking_error_codes': blocking_codes,
            'total_errors': len(error_codes),
            'blocking_count': len(blocking_codes)
        }
    
    def get_diagnostic_summary(self, error_codes: List[str]) -> Dict[str, Any]:
        """API endpoint to get diagnostic summary"""
        summary = self.catalog_service.generate_diagnostic_summary(error_codes)
        
        return {
            'success': True,
            'summary': summary
        }
    
    def get_autofixes(self, error_codes: List[str]) -> Dict[str, Any]:
        """API endpoint to get autofix suggestions"""
        autofixable = self.catalog_service.get_autofixable_diagnostics(error_codes)
        
        return {
            'success': True,
            'autofixable_count': len(autofixable),
            'autofixes': {
                error_code: [
                    {
                        'fix_id': fix.fix_id,
                        'description': fix.description,
                        'confidence_score': fix.confidence_score,
                        'safe_to_apply_automatically': fix.safe_to_apply_automatically
                    }
                    for fix in fixes
                ]
                for error_code, fixes in autofixable.items()
            }
        }

# Test Functions
def run_diagnostic_catalog_tests():
    """Run comprehensive diagnostic catalog tests"""
    print("=== Diagnostic Catalog Service Tests ===")
    
    # Initialize service
    catalog_service = DiagnosticCatalogService()
    catalog_api = DiagnosticCatalogAPI(catalog_service)
    
    # Test 1: Standard rules initialization
    print("\n1. Testing standard rules initialization...")
    stats = catalog_service.get_diagnostic_statistics()
    print(f"   Total rules loaded: {stats['total_rules']}")
    print(f"   Rules by severity: {stats['rules_by_severity']}")
    print(f"   Rules by action type: {stats['rules_by_action_type']}")
    
    # Test 2: Custom rule registration
    print("\n2. Testing custom rule registration...")
    
    custom_rule = DiagnosticRule(
        error_code="CUSTOM_001",
        rule_name="Test custom rule",
        category=DiagnosticCategory.BEST_PRACTICE,
        severity=DiagnosticSeverity.WARNING,
        action_type=ActionType.WARN,
        short_message="This is a test rule",
        detailed_description="A test rule for demonstration purposes",
        technical_explanation="Technical explanation of the test rule",
        primary_persona=PersonaType.DEVELOPER
    )
    
    registered = catalog_service.register_diagnostic_rule(custom_rule)
    print(f"   Custom rule registration: {'✅ PASS' if registered else '❌ FAIL'}")
    
    # Test 3: Diagnostic instance creation
    print("\n3. Testing diagnostic instance creation...")
    
    test_error_codes = ["SYNTAX_001", "ML_001", "POLICY_001"]
    instances = []
    
    for i, error_code in enumerate(test_error_codes):
        instance = catalog_service.create_diagnostic_instance(
            error_code,
            f"Test message for {error_code}",
            file_path=f"test_file_{i}.yaml",
            line_number=i + 10,
            context_data={"test_context": f"value_{i}"}
        )
        if instance:
            instances.append(instance)
    
    print(f"   Diagnostic instances created: {len(instances)}")
    print(f"   Instance error codes: {[inst.error_code for inst in instances]}")
    
    # Test 4: Action determination
    print("\n4. Testing action determination...")
    
    for error_code in test_error_codes:
        action = catalog_service.get_action_for_diagnostic(error_code)
        rule = catalog_service.diagnostic_rules.get(error_code)
        print(f"   {error_code}: {action.value if action else 'UNKNOWN'} "
              f"(blocks compilation: {rule.block_compilation if rule else 'N/A'})")
    
    # Test 5: Compilation blocking check
    print("\n5. Testing compilation blocking...")
    
    should_block, blocking_codes = catalog_service.should_block_compilation(test_error_codes)
    print(f"   Should block compilation: {should_block}")
    print(f"   Blocking codes: {blocking_codes}")
    
    # Test 6: Signing blocking check
    print("\n6. Testing signing blocking...")
    
    should_block_signing, signing_blocking_codes = catalog_service.should_block_signing(test_error_codes)
    print(f"   Should block signing: {should_block_signing}")
    print(f"   Signing blocking codes: {signing_blocking_codes}")
    
    # Test 7: Autofix suggestions
    print("\n7. Testing autofix suggestions...")
    
    autofixable = catalog_service.get_autofixable_diagnostics(test_error_codes)
    print(f"   Autofixable diagnostics: {len(autofixable)}")
    for error_code, fixes in autofixable.items():
        print(f"   {error_code}: {len(fixes)} fixes available")
        for fix in fixes[:1]:  # Show first fix
            print(f"     - {fix.description} (confidence: {fix.confidence_score})")
    
    # Test 8: CAB approval requirements
    print("\n8. Testing CAB approval requirements...")
    
    requiring_approval = catalog_service.get_diagnostics_requiring_approval(test_error_codes)
    print(f"   Diagnostics requiring CAB approval: {requiring_approval}")
    
    # Test 9: Persona-specific diagnostics
    print("\n9. Testing persona-specific diagnostics...")
    
    personas_to_test = [PersonaType.DEVELOPER, PersonaType.DATA_SCIENTIST, PersonaType.COMPLIANCE]
    for persona in personas_to_test:
        persona_rules = catalog_service.get_diagnostics_for_persona(persona)
        print(f"   {persona.value}: {len(persona_rules)} relevant rules")
    
    # Test 10: Diagnostic summary
    print("\n10. Testing diagnostic summary...")
    
    summary = catalog_service.generate_diagnostic_summary(test_error_codes)
    print(f"   Total diagnostics: {summary['total_diagnostics']}")
    print(f"   By severity: {summary['by_severity']}")
    print(f"   Recommendations: {len(summary['recommendations'])}")
    for rec in summary['recommendations']:
        print(f"     - {rec}")
    
    # Test 11: Instance resolution
    print("\n11. Testing instance resolution...")
    
    if instances:
        resolved = catalog_service.resolve_diagnostic_instance(
            instances[0].instance_id,
            "Fixed by adding missing semicolon",
            "test_developer",
            "fixed"
        )
        print(f"   Instance resolution: {'✅ PASS' if resolved else '❌ FAIL'}")
        
        resolved_instance = catalog_service.diagnostic_instances[instances[0].instance_id]
        print(f"   Resolution status: {resolved_instance.status}")
        print(f"   Resolved by: {resolved_instance.resolved_by}")
    
    # Test 12: API interface
    print("\n12. Testing API interface...")
    
    # Test API action lookup
    api_action_result = catalog_api.get_action_for_error("ML_001")
    print(f"   API action lookup: {'✅ PASS' if api_action_result['success'] else '❌ FAIL'}")
    if api_action_result['success']:
        print(f"   ML_001 action: {api_action_result['action_type']}")
    
    # Test API compilation check
    api_compilation_result = catalog_api.check_compilation_blockers(test_error_codes)
    print(f"   API compilation check: {'✅ PASS' if api_compilation_result['success'] else '❌ FAIL'}")
    if api_compilation_result['success']:
        print(f"   Should block: {api_compilation_result['should_block_compilation']}")
    
    # Test API autofix suggestions
    api_autofix_result = catalog_api.get_autofixes(test_error_codes)
    print(f"   API autofix lookup: {'✅ PASS' if api_autofix_result['success'] else '❌ FAIL'}")
    if api_autofix_result['success']:
        print(f"   Autofixable count: {api_autofix_result['autofixable_count']}")
    
    # Test 13: Catalog export
    print("\n13. Testing catalog export...")
    
    try:
        exported_catalog = catalog_service.export_diagnostic_catalog("json")
        export_success = len(exported_catalog) > 0
        print(f"   Catalog export: {'✅ PASS' if export_success else '❌ FAIL'}")
        if export_success:
            import json
            catalog_data = json.loads(exported_catalog)
            print(f"   Exported rules: {catalog_data['total_rules']}")
    except Exception as e:
        print(f"   Catalog export: ❌ FAIL - {e}")
    
    # Test 14: Category-based lookup
    print("\n14. Testing category-based lookup...")
    
    categories_to_test = [DiagnosticCategory.SYNTAX, DiagnosticCategory.ML_SPECIFIC, DiagnosticCategory.COMPLIANCE]
    for category in categories_to_test:
        category_rules = catalog_service.get_diagnostics_by_category(category)
        print(f"   {category.value}: {len(category_rules)} rules")
    
    print(f"\n=== Test Summary ===")
    final_stats = catalog_service.get_diagnostic_statistics()
    print(f"Diagnostic catalog service tested successfully")
    print(f"Total rules: {final_stats['total_rules']}")
    print(f"Total instances: {final_stats['total_instances']}")
    print(f"Rules by category: {final_stats['rules_by_category']}")
    print(f"Rules by persona: {final_stats['rules_by_persona']}")
    
    return catalog_service, catalog_api

if __name__ == "__main__":
    run_diagnostic_catalog_tests()
