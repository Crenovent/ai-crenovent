"""
Task 8.4-T10: Build overlay validation engine
Industry-specific governance overlay validation for workflows and policies
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class IndustryType(Enum):
    """Supported industry types"""
    SAAS = "saas"
    BANKING = "banking"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    ECOMMERCE = "ecommerce"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    RBI = "rbi"
    IRDAI = "irdai"
    PCI_DSS = "pci_dss"
    BASEL_III = "basel_iii"
    DPDP = "dpdp"
    CCPA = "ccpa"


class ValidationSeverity(Enum):
    """Validation result severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class OverlayRule:
    """Industry overlay validation rule"""
    rule_id: str
    rule_name: str
    industry_type: IndustryType
    compliance_framework: ComplianceFramework
    
    # Rule definition
    rule_type: str  # field_required, field_format, workflow_step, approval_chain, etc.
    rule_expression: str  # JSONPath or custom expression
    expected_value: Any = None
    
    # Validation settings
    severity: ValidationSeverity = ValidationSeverity.HIGH
    is_mandatory: bool = True
    error_message: str = ""
    remediation_guidance: str = ""
    
    # Applicability
    applies_to: List[str] = field(default_factory=list)  # workflow_types, policy_types
    
    # Metadata
    description: str = ""
    reference_url: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "industry_type": self.industry_type.value,
            "compliance_framework": self.compliance_framework.value,
            "rule_type": self.rule_type,
            "rule_expression": self.rule_expression,
            "expected_value": self.expected_value,
            "severity": self.severity.value,
            "is_mandatory": self.is_mandatory,
            "error_message": self.error_message,
            "remediation_guidance": self.remediation_guidance,
            "applies_to": self.applies_to,
            "description": self.description,
            "reference_url": self.reference_url,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }


@dataclass
class ValidationViolation:
    """Overlay validation violation"""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    
    # Violation details
    violation_type: str = ""
    severity: ValidationSeverity = ValidationSeverity.HIGH
    message: str = ""
    
    # Context
    object_type: str = ""  # workflow, policy, step
    object_id: str = ""
    field_path: Optional[str] = None
    current_value: Any = None
    expected_value: Any = None
    
    # Compliance
    industry_type: IndustryType = IndustryType.SAAS
    compliance_framework: ComplianceFramework = ComplianceFramework.SOX
    
    # Remediation
    remediation_guidance: str = ""
    auto_fixable: bool = False
    
    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
            "message": self.message,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "field_path": self.field_path,
            "current_value": self.current_value,
            "expected_value": self.expected_value,
            "industry_type": self.industry_type.value,
            "compliance_framework": self.compliance_framework.value,
            "remediation_guidance": self.remediation_guidance,
            "auto_fixable": self.auto_fixable,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class OverlayValidationResult:
    """Result of overlay validation"""
    object_id: str
    object_type: str
    industry_type: IndustryType
    compliance_frameworks: List[ComplianceFramework]
    
    # Validation results
    is_valid: bool = True
    violations: List[ValidationViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Statistics
    rules_evaluated: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    
    # Performance
    validation_duration_ms: float = 0.0
    
    # Timestamps
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "industry_type": self.industry_type.value,
            "compliance_frameworks": [cf.value for cf in self.compliance_frameworks],
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "rules_evaluated": self.rules_evaluated,
            "critical_violations": self.critical_violations,
            "high_violations": self.high_violations,
            "medium_violations": self.medium_violations,
            "low_violations": self.low_violations,
            "validation_duration_ms": self.validation_duration_ms,
            "validated_at": self.validated_at.isoformat()
        }


class OverlayValidationEngine:
    """
    Overlay Validation Engine - Task 8.4-T10
    
    Validates workflows and policies against industry-specific governance overlays
    Ensures compliance with regulatory requirements per industry
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'strict_validation': True,
            'fail_on_critical_violations': True,
            'fail_on_high_violations': False,
            'enable_auto_remediation': True,
            'cache_rules': True,
            'validation_timeout_seconds': 60,
            'parallel_validation': True
        }
        
        # Rules cache
        self.overlay_rules: Dict[str, List[OverlayRule]] = {}  # industry_framework -> rules
        self.rules_last_loaded: Optional[datetime] = None
        self.cache_ttl_minutes = 30
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'critical_violations_detected': 0,
            'auto_remediations_applied': 0,
            'average_validation_time_ms': 0.0
        }
        
        # Initialize default overlay rules
        self._initialize_default_overlay_rules()
    
    async def initialize(self) -> bool:
        """Initialize overlay validation engine"""
        try:
            await self._create_overlay_validation_tables()
            await self._load_overlay_rules()
            self.logger.info("✅ Overlay validation engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize overlay validation engine: {e}")
            return False
    
    async def validate_workflow_overlay(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: int,
        industry_type: IndustryType = IndustryType.SAAS,
        compliance_frameworks: List[ComplianceFramework] = None
    ) -> OverlayValidationResult:
        """
        Validate workflow against industry overlay rules
        
        This is the main entry point for workflow validation
        """
        
        start_time = datetime.now(timezone.utc)
        
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        
        # Initialize result
        result = OverlayValidationResult(
            object_id=workflow_id,
            object_type="workflow",
            industry_type=industry_type,
            compliance_frameworks=compliance_frameworks or [ComplianceFramework.SOX]
        )
        
        try:
            # Refresh rules if needed
            await self._refresh_rules_if_needed()
            
            # Get applicable overlay rules
            applicable_rules = self._get_applicable_overlay_rules(
                industry_type, compliance_frameworks or [ComplianceFramework.SOX], "workflow"
            )
            
            result.rules_evaluated = len(applicable_rules)
            
            # Validate against each rule
            if self.config['parallel_validation']:
                validation_tasks = [
                    self._validate_against_overlay_rule(workflow_definition, rule, "workflow")
                    for rule in applicable_rules
                ]
                violations_lists = await asyncio.gather(*validation_tasks, return_exceptions=True)
                
                for violations in violations_lists:
                    if isinstance(violations, Exception):
                        self.logger.error(f"Validation error: {violations}")
                        continue
                    result.violations.extend(violations)
            else:
                for rule in applicable_rules:
                    violations = await self._validate_against_overlay_rule(
                        workflow_definition, rule, "workflow"
                    )
                    result.violations.extend(violations)
            
            # Count violations by severity
            for violation in result.violations:
                if violation.severity == ValidationSeverity.CRITICAL:
                    result.critical_violations += 1
                elif violation.severity == ValidationSeverity.HIGH:
                    result.high_violations += 1
                elif violation.severity == ValidationSeverity.MEDIUM:
                    result.medium_violations += 1
                elif violation.severity == ValidationSeverity.LOW:
                    result.low_violations += 1
            
            # Determine overall validation result
            if self.config['fail_on_critical_violations'] and result.critical_violations > 0:
                result.is_valid = False
            elif self.config['fail_on_high_violations'] and result.high_violations > 0:
                result.is_valid = False
            
            # Add warnings for non-blocking violations
            if result.high_violations > 0 and result.is_valid:
                result.warnings.append(f"High severity violations detected: {result.high_violations}")
            
            # Apply auto-remediation if enabled
            if self.config['enable_auto_remediation'] and not result.is_valid:
                auto_fixed = await self._apply_auto_remediation(workflow_definition, result.violations)
                if auto_fixed > 0:
                    self.validation_stats['auto_remediations_applied'] += auto_fixed
                    result.warnings.append(f"Auto-remediation applied to {auto_fixed} violations")
            
            # Calculate validation duration
            end_time = datetime.now(timezone.utc)
            result.validation_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_validation_stats(result)
            
            # Store validation result
            if self.db_pool:
                await self._store_validation_result(result, tenant_id)
            
            if result.is_valid:
                self.logger.info(f"✅ Overlay validation passed for workflow {workflow_id} ({result.validation_duration_ms:.2f}ms)")
            else:
                self.logger.warning(f"❌ Overlay validation failed for workflow {workflow_id}")
                for violation in result.violations:
                    if violation.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]:
                        self.logger.warning(f"🚫 {violation.severity.value.upper()}: {violation.message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Overlay validation error for workflow {workflow_id}: {e}")
            
            # Fail closed on validation errors
            result.is_valid = False
            result.violations.append(ValidationViolation(
                rule_id="system_error",
                rule_name="System Error",
                violation_type="validation_error",
                severity=ValidationSeverity.CRITICAL,
                message=f"Overlay validation system error: {str(e)}",
                object_type="workflow",
                object_id=workflow_id,
                industry_type=industry_type,
                compliance_framework=compliance_frameworks[0] if compliance_frameworks else ComplianceFramework.SOX,
                remediation_guidance="Contact system administrator"
            ))
            
            result.validation_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
    
    async def validate_policy_overlay(
        self,
        policy_definition: Dict[str, Any],
        tenant_id: int,
        industry_type: IndustryType = IndustryType.SAAS,
        compliance_frameworks: List[ComplianceFramework] = None
    ) -> OverlayValidationResult:
        """
        Validate policy against industry overlay rules
        """
        
        start_time = datetime.now(timezone.utc)
        
        policy_id = policy_definition.get('policy_id', 'unknown')
        
        # Initialize result
        result = OverlayValidationResult(
            object_id=policy_id,
            object_type="policy",
            industry_type=industry_type,
            compliance_frameworks=compliance_frameworks or [ComplianceFramework.SOX]
        )
        
        try:
            # Get applicable overlay rules
            applicable_rules = self._get_applicable_overlay_rules(
                industry_type, compliance_frameworks or [ComplianceFramework.SOX], "policy"
            )
            
            result.rules_evaluated = len(applicable_rules)
            
            # Validate against each rule
            for rule in applicable_rules:
                violations = await self._validate_against_overlay_rule(
                    policy_definition, rule, "policy"
                )
                result.violations.extend(violations)
            
            # Count violations by severity
            for violation in result.violations:
                if violation.severity == ValidationSeverity.CRITICAL:
                    result.critical_violations += 1
                elif violation.severity == ValidationSeverity.HIGH:
                    result.high_violations += 1
                elif violation.severity == ValidationSeverity.MEDIUM:
                    result.medium_violations += 1
                elif violation.severity == ValidationSeverity.LOW:
                    result.low_violations += 1
            
            # Determine overall validation result
            if self.config['fail_on_critical_violations'] and result.critical_violations > 0:
                result.is_valid = False
            elif self.config['fail_on_high_violations'] and result.high_violations > 0:
                result.is_valid = False
            
            # Calculate validation duration
            end_time = datetime.now(timezone.utc)
            result.validation_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_validation_stats(result)
            
            # Store validation result
            if self.db_pool:
                await self._store_validation_result(result, tenant_id)
            
            if result.is_valid:
                self.logger.info(f"✅ Overlay validation passed for policy {policy_id}")
            else:
                self.logger.warning(f"❌ Overlay validation failed for policy {policy_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Overlay validation error for policy {policy_id}: {e}")
            
            # Fail closed on validation errors
            result.is_valid = False
            result.violations.append(ValidationViolation(
                rule_id="system_error",
                rule_name="System Error",
                violation_type="validation_error",
                severity=ValidationSeverity.CRITICAL,
                message=f"Overlay validation system error: {str(e)}",
                object_type="policy",
                object_id=policy_id,
                industry_type=industry_type,
                compliance_framework=compliance_frameworks[0] if compliance_frameworks else ComplianceFramework.SOX
            ))
            
            result.validation_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
    
    def _initialize_default_overlay_rules(self):
        """Initialize default overlay rules for different industries"""
        
        # SaaS + SOX rules
        saas_sox_rules = [
            OverlayRule(
                rule_id="saas_sox_approval_chain",
                rule_name="SOX Approval Chain Required",
                industry_type=IndustryType.SAAS,
                compliance_framework=ComplianceFramework.SOX,
                rule_type="approval_chain",
                rule_expression="$.steps[?(@.type=='approval')].length",
                expected_value=2,
                severity=ValidationSeverity.CRITICAL,
                is_mandatory=True,
                error_message="SOX requires minimum 2 approval steps in financial workflows",
                remediation_guidance="Add additional approval steps with different roles",
                applies_to=["financial", "revenue", "billing"],
                description="SOX compliance requires segregation of duties in financial processes"
            ),
            
            OverlayRule(
                rule_id="saas_sox_audit_trail",
                rule_name="SOX Audit Trail Required",
                industry_type=IndustryType.SAAS,
                compliance_framework=ComplianceFramework.SOX,
                rule_type="field_required",
                rule_expression="$.governance.evidence_pack_required",
                expected_value=True,
                severity=ValidationSeverity.CRITICAL,
                is_mandatory=True,
                error_message="SOX requires evidence pack generation for audit trail",
                remediation_guidance="Enable evidence_pack_required in governance section",
                applies_to=["financial", "revenue", "compliance"],
                description="SOX requires comprehensive audit trails for all financial processes"
            ),
            
            OverlayRule(
                rule_id="saas_sox_maker_checker",
                rule_name="SOX Maker-Checker Separation",
                industry_type=IndustryType.SAAS,
                compliance_framework=ComplianceFramework.SOX,
                rule_type="role_separation",
                rule_expression="$.steps[*].allowed_roles",
                expected_value="no_overlap_create_approve",
                severity=ValidationSeverity.CRITICAL,
                is_mandatory=True,
                error_message="SOX requires maker-checker separation: creator cannot approve",
                remediation_guidance="Ensure different roles for creation and approval steps",
                applies_to=["financial", "payment", "invoice"],
                description="SOX maker-checker principle prevents self-approval"
            )
        ]
        
        # Banking + RBI rules
        banking_rbi_rules = [
            OverlayRule(
                rule_id="banking_rbi_dual_control",
                rule_name="RBI Dual Control Required",
                industry_type=IndustryType.BANKING,
                compliance_framework=ComplianceFramework.RBI,
                rule_type="dual_control",
                rule_expression="$.steps[?(@.type=='approval')].min_approvers",
                expected_value=2,
                severity=ValidationSeverity.CRITICAL,
                is_mandatory=True,
                error_message="RBI requires dual control for high-value transactions",
                remediation_guidance="Set min_approvers to 2 for approval steps",
                applies_to=["loan", "credit", "payment", "transfer"],
                description="RBI dual control requirements for financial transactions"
            ),
            
            OverlayRule(
                rule_id="banking_rbi_risk_assessment",
                rule_name="RBI Risk Assessment Required",
                industry_type=IndustryType.BANKING,
                compliance_framework=ComplianceFramework.RBI,
                rule_type="field_required",
                rule_expression="$.risk_assessment.enabled",
                expected_value=True,
                severity=ValidationSeverity.HIGH,
                is_mandatory=True,
                error_message="RBI requires risk assessment for loan workflows",
                remediation_guidance="Enable risk assessment in workflow configuration",
                applies_to=["loan", "credit", "mortgage"],
                description="RBI requires comprehensive risk assessment for lending"
            )
        ]
        
        # Insurance + IRDAI rules
        insurance_irdai_rules = [
            OverlayRule(
                rule_id="insurance_irdai_claim_validation",
                rule_name="IRDAI Claim Validation Required",
                industry_type=IndustryType.INSURANCE,
                compliance_framework=ComplianceFramework.IRDAI,
                rule_type="validation_step",
                rule_expression="$.steps[?(@.type=='validation')].length",
                expected_value=1,
                severity=ValidationSeverity.CRITICAL,
                is_mandatory=True,
                error_message="IRDAI requires validation step in claim processing",
                remediation_guidance="Add validation step before claim approval",
                applies_to=["claim", "settlement", "payout"],
                description="IRDAI requires validation of all insurance claims"
            ),
            
            OverlayRule(
                rule_id="insurance_irdai_fraud_check",
                rule_name="IRDAI Fraud Detection Required",
                industry_type=IndustryType.INSURANCE,
                compliance_framework=ComplianceFramework.IRDAI,
                rule_type="fraud_check",
                rule_expression="$.fraud_detection.enabled",
                expected_value=True,
                severity=ValidationSeverity.HIGH,
                is_mandatory=True,
                error_message="IRDAI requires fraud detection in claim workflows",
                remediation_guidance="Enable fraud detection in workflow configuration",
                applies_to=["claim", "settlement"],
                description="IRDAI requires fraud detection for claim processing"
            )
        ]
        
        # Healthcare + HIPAA rules
        healthcare_hipaa_rules = [
            OverlayRule(
                rule_id="healthcare_hipaa_pii_protection",
                rule_name="HIPAA PII Protection Required",
                industry_type=IndustryType.HEALTHCARE,
                compliance_framework=ComplianceFramework.HIPAA,
                rule_type="pii_protection",
                rule_expression="$.data_protection.pii_redaction_enabled",
                expected_value=True,
                severity=ValidationSeverity.CRITICAL,
                is_mandatory=True,
                error_message="HIPAA requires PII redaction for patient data workflows",
                remediation_guidance="Enable PII redaction in data protection settings",
                applies_to=["patient", "medical", "billing"],
                description="HIPAA requires protection of patient health information"
            ),
            
            OverlayRule(
                rule_id="healthcare_hipaa_access_logging",
                rule_name="HIPAA Access Logging Required",
                industry_type=IndustryType.HEALTHCARE,
                compliance_framework=ComplianceFramework.HIPAA,
                rule_type="access_logging",
                rule_expression="$.governance.access_logging_enabled",
                expected_value=True,
                severity=ValidationSeverity.CRITICAL,
                is_mandatory=True,
                error_message="HIPAA requires comprehensive access logging",
                remediation_guidance="Enable access logging in governance settings",
                applies_to=["patient", "medical", "phi"],
                description="HIPAA requires detailed access logs for PHI"
            )
        ]
        
        # Store rules by industry-framework key
        self.overlay_rules = {
            f"{IndustryType.SAAS.value}_{ComplianceFramework.SOX.value}": saas_sox_rules,
            f"{IndustryType.BANKING.value}_{ComplianceFramework.RBI.value}": banking_rbi_rules,
            f"{IndustryType.INSURANCE.value}_{ComplianceFramework.IRDAI.value}": insurance_irdai_rules,
            f"{IndustryType.HEALTHCARE.value}_{ComplianceFramework.HIPAA.value}": healthcare_hipaa_rules
        }
    
    def _get_applicable_overlay_rules(
        self,
        industry_type: IndustryType,
        compliance_frameworks: List[ComplianceFramework],
        object_type: str
    ) -> List[OverlayRule]:
        """Get overlay rules applicable to this validation"""
        
        applicable_rules = []
        
        for framework in compliance_frameworks:
            key = f"{industry_type.value}_{framework.value}"
            rules = self.overlay_rules.get(key, [])
            
            for rule in rules:
                if not rule.is_active:
                    continue
                
                # Check if rule applies to this object type
                if rule.applies_to and object_type not in rule.applies_to:
                    continue
                
                applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _validate_against_overlay_rule(
        self,
        object_definition: Dict[str, Any],
        rule: OverlayRule,
        object_type: str
    ) -> List[ValidationViolation]:
        """Validate object against specific overlay rule"""
        
        violations = []
        object_id = object_definition.get(f'{object_type}_id', 'unknown')
        
        try:
            if rule.rule_type == "field_required":
                violations.extend(await self._validate_field_required(object_definition, rule, object_type, object_id))
            elif rule.rule_type == "approval_chain":
                violations.extend(await self._validate_approval_chain(object_definition, rule, object_type, object_id))
            elif rule.rule_type == "role_separation":
                violations.extend(await self._validate_role_separation(object_definition, rule, object_type, object_id))
            elif rule.rule_type == "dual_control":
                violations.extend(await self._validate_dual_control(object_definition, rule, object_type, object_id))
            elif rule.rule_type == "validation_step":
                violations.extend(await self._validate_validation_step(object_definition, rule, object_type, object_id))
            elif rule.rule_type == "fraud_check":
                violations.extend(await self._validate_fraud_check(object_definition, rule, object_type, object_id))
            elif rule.rule_type == "pii_protection":
                violations.extend(await self._validate_pii_protection(object_definition, rule, object_type, object_id))
            elif rule.rule_type == "access_logging":
                violations.extend(await self._validate_access_logging(object_definition, rule, object_type, object_id))
            
        except Exception as e:
            self.logger.error(f"Error validating rule {rule.rule_id}: {e}")
            violations.append(ValidationViolation(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                violation_type="validation_error",
                severity=ValidationSeverity.HIGH,
                message=f"Error validating rule: {str(e)}",
                object_type=object_type,
                object_id=object_id,
                industry_type=rule.industry_type,
                compliance_framework=rule.compliance_framework
            ))
        
        return violations
    
    # Specific validation methods for different rule types
    
    async def _validate_field_required(
        self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str
    ) -> List[ValidationViolation]:
        """Validate required field rule"""
        
        violations = []
        
        # Simple JSONPath-like evaluation
        field_path = rule.rule_expression.replace('$.', '')
        field_parts = field_path.split('.')
        
        current_value = obj
        for part in field_parts:
            if isinstance(current_value, dict) and part in current_value:
                current_value = current_value[part]
            else:
                current_value = None
                break
        
        if current_value != rule.expected_value:
            violations.append(ValidationViolation(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                violation_type="field_required",
                severity=rule.severity,
                message=rule.error_message,
                object_type=object_type,
                object_id=object_id,
                field_path=field_path,
                current_value=current_value,
                expected_value=rule.expected_value,
                industry_type=rule.industry_type,
                compliance_framework=rule.compliance_framework,
                remediation_guidance=rule.remediation_guidance,
                auto_fixable=True
            ))
        
        return violations
    
    async def _validate_approval_chain(
        self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str
    ) -> List[ValidationViolation]:
        """Validate approval chain rule"""
        
        violations = []
        
        steps = obj.get('steps', [])
        approval_steps = [s for s in steps if s.get('type') == 'approval']
        
        if len(approval_steps) < rule.expected_value:
            violations.append(ValidationViolation(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                violation_type="approval_chain",
                severity=rule.severity,
                message=rule.error_message,
                object_type=object_type,
                object_id=object_id,
                field_path="steps",
                current_value=len(approval_steps),
                expected_value=rule.expected_value,
                industry_type=rule.industry_type,
                compliance_framework=rule.compliance_framework,
                remediation_guidance=rule.remediation_guidance
            ))
        
        return violations
    
    async def _validate_role_separation(
        self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str
    ) -> List[ValidationViolation]:
        """Validate role separation rule"""
        
        violations = []
        
        steps = obj.get('steps', [])
        create_roles = set()
        approve_roles = set()
        
        for step in steps:
            if step.get('action') in ['create', 'submit']:
                create_roles.update(step.get('allowed_roles', []))
            elif step.get('action') in ['approve', 'authorize']:
                approve_roles.update(step.get('allowed_roles', []))
        
        # Check for role overlap
        overlapping_roles = create_roles.intersection(approve_roles)
        
        if overlapping_roles:
            violations.append(ValidationViolation(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                violation_type="role_separation",
                severity=rule.severity,
                message=rule.error_message,
                object_type=object_type,
                object_id=object_id,
                field_path="steps.allowed_roles",
                current_value=list(overlapping_roles),
                expected_value="no_overlap",
                industry_type=rule.industry_type,
                compliance_framework=rule.compliance_framework,
                remediation_guidance=rule.remediation_guidance
            ))
        
        return violations
    
    # Additional validation methods for other rule types...
    async def _validate_dual_control(self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str) -> List[ValidationViolation]:
        return []  # Implementation placeholder
    
    async def _validate_validation_step(self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str) -> List[ValidationViolation]:
        return []  # Implementation placeholder
    
    async def _validate_fraud_check(self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str) -> List[ValidationViolation]:
        return []  # Implementation placeholder
    
    async def _validate_pii_protection(self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str) -> List[ValidationViolation]:
        return []  # Implementation placeholder
    
    async def _validate_access_logging(self, obj: Dict[str, Any], rule: OverlayRule, object_type: str, object_id: str) -> List[ValidationViolation]:
        return []  # Implementation placeholder
    
    async def _apply_auto_remediation(
        self, obj: Dict[str, Any], violations: List[ValidationViolation]
    ) -> int:
        """Apply auto-remediation for fixable violations"""
        
        auto_fixed = 0
        
        for violation in violations:
            if not violation.auto_fixable:
                continue
            
            try:
                if violation.violation_type == "field_required" and violation.field_path:
                    # Auto-fix required field
                    field_parts = violation.field_path.split('.')
                    current = obj
                    
                    # Navigate to parent
                    for part in field_parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Set the required value
                    current[field_parts[-1]] = violation.expected_value
                    auto_fixed += 1
                    
                    self.logger.info(f"🔧 Auto-fixed field {violation.field_path} = {violation.expected_value}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to auto-fix violation {violation.violation_id}: {e}")
        
        return auto_fixed
    
    def _update_validation_stats(self, result: OverlayValidationResult):
        """Update validation statistics"""
        
        self.validation_stats['total_validations'] += 1
        
        if result.is_valid:
            self.validation_stats['validations_passed'] += 1
        else:
            self.validation_stats['validations_failed'] += 1
        
        self.validation_stats['critical_violations_detected'] += result.critical_violations
        
        # Update average validation time
        current_avg = self.validation_stats['average_validation_time_ms']
        total_validations = self.validation_stats['total_validations']
        self.validation_stats['average_validation_time_ms'] = (
            (current_avg * (total_validations - 1) + result.validation_duration_ms) / total_validations
        )
    
    async def _refresh_rules_if_needed(self):
        """Refresh rules from database if cache is stale"""
        
        if not self.config['cache_rules']:
            await self._load_overlay_rules()
            return
        
        if (not self.rules_last_loaded or 
            (datetime.now(timezone.utc) - self.rules_last_loaded).total_seconds() > self.cache_ttl_minutes * 60):
            await self._load_overlay_rules()
    
    async def _load_overlay_rules(self):
        """Load overlay rules from database"""
        # In production, this would load custom rules from database
        # For now, we use the default rules initialized in memory
        self.rules_last_loaded = datetime.now(timezone.utc)
    
    async def _create_overlay_validation_tables(self):
        """Create overlay validation database tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Overlay validation results
        CREATE TABLE IF NOT EXISTS overlay_validations (
            validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            object_id VARCHAR(100) NOT NULL,
            object_type VARCHAR(50) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Industry and compliance
            industry_type VARCHAR(50) NOT NULL,
            compliance_frameworks TEXT[] NOT NULL,
            
            -- Validation results
            is_valid BOOLEAN NOT NULL,
            violations_count INTEGER NOT NULL DEFAULT 0,
            warnings_count INTEGER NOT NULL DEFAULT 0,
            rules_evaluated INTEGER NOT NULL DEFAULT 0,
            
            -- Violation counts by severity
            critical_violations INTEGER NOT NULL DEFAULT 0,
            high_violations INTEGER NOT NULL DEFAULT 0,
            medium_violations INTEGER NOT NULL DEFAULT 0,
            low_violations INTEGER NOT NULL DEFAULT 0,
            
            -- Performance
            validation_duration_ms FLOAT NOT NULL DEFAULT 0,
            
            -- Timestamps
            validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_overlay_object_type CHECK (object_type IN ('workflow', 'policy', 'step')),
            CONSTRAINT chk_overlay_industry_type CHECK (industry_type IN ('saas', 'banking', 'insurance', 'healthcare', 'fintech', 'ecommerce'))
        );
        
        -- Overlay validation violations
        CREATE TABLE IF NOT EXISTS overlay_validation_violations (
            violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            validation_id UUID REFERENCES overlay_validations(validation_id) ON DELETE CASCADE,
            
            -- Rule details
            rule_id VARCHAR(100) NOT NULL,
            rule_name VARCHAR(200) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            
            -- Violation details
            message TEXT NOT NULL,
            object_type VARCHAR(50) NOT NULL,
            object_id VARCHAR(100) NOT NULL,
            field_path VARCHAR(200),
            current_value TEXT,
            expected_value TEXT,
            
            -- Compliance context
            industry_type VARCHAR(50) NOT NULL,
            compliance_framework VARCHAR(50) NOT NULL,
            
            -- Remediation
            remediation_guidance TEXT,
            auto_fixable BOOLEAN NOT NULL DEFAULT FALSE,
            
            -- Timestamps
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_overlay_violation_severity CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_overlay_validations_object ON overlay_validations(object_id, object_type, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_overlay_validations_tenant ON overlay_validations(tenant_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_overlay_validations_industry ON overlay_validations(industry_type, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_overlay_validations_valid ON overlay_validations(is_valid, validated_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_validation ON overlay_validation_violations(validation_id);
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_rule ON overlay_validation_violations(rule_id, severity);
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_object ON overlay_validation_violations(object_id, detected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_compliance ON overlay_validation_violations(compliance_framework, industry_type);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Overlay validation tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create overlay validation tables: {e}")
            raise
    
    async def _store_validation_result(
        self, result: OverlayValidationResult, tenant_id: int
    ):
        """Store validation result in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert validation result
                validation_query = """
                    INSERT INTO overlay_validations (
                        object_id, object_type, tenant_id, industry_type,
                        compliance_frameworks, is_valid, violations_count,
                        warnings_count, rules_evaluated, critical_violations,
                        high_violations, medium_violations, low_violations,
                        validation_duration_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING validation_id
                """
                
                validation_id = await conn.fetchval(
                    validation_query,
                    result.object_id,
                    result.object_type,
                    tenant_id,
                    result.industry_type.value,
                    [cf.value for cf in result.compliance_frameworks],
                    result.is_valid,
                    len(result.violations),
                    len(result.warnings),
                    result.rules_evaluated,
                    result.critical_violations,
                    result.high_violations,
                    result.medium_violations,
                    result.low_violations,
                    result.validation_duration_ms
                )
                
                # Insert violations
                if result.violations:
                    violation_query = """
                        INSERT INTO overlay_validation_violations (
                            validation_id, rule_id, rule_name, violation_type,
                            severity, message, object_type, object_id,
                            field_path, current_value, expected_value,
                            industry_type, compliance_framework, remediation_guidance,
                            auto_fixable, detected_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """
                    
                    for violation in result.violations:
                        await conn.execute(
                            violation_query,
                            validation_id,
                            violation.rule_id,
                            violation.rule_name,
                            violation.violation_type,
                            violation.severity.value,
                            violation.message,
                            violation.object_type,
                            violation.object_id,
                            violation.field_path,
                            str(violation.current_value) if violation.current_value is not None else None,
                            str(violation.expected_value) if violation.expected_value is not None else None,
                            violation.industry_type.value,
                            violation.compliance_framework.value,
                            violation.remediation_guidance,
                            violation.auto_fixable,
                            violation.detected_at
                        )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store overlay validation result: {e}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        success_rate = 0.0
        if self.validation_stats['total_validations'] > 0:
            success_rate = (
                self.validation_stats['validations_passed'] / 
                self.validation_stats['total_validations']
            ) * 100
        
        return {
            'total_validations': self.validation_stats['total_validations'],
            'validations_passed': self.validation_stats['validations_passed'],
            'validations_failed': self.validation_stats['validations_failed'],
            'critical_violations_detected': self.validation_stats['critical_violations_detected'],
            'auto_remediations_applied': self.validation_stats['auto_remediations_applied'],
            'success_rate_percentage': round(success_rate, 2),
            'average_validation_time_ms': round(self.validation_stats['average_validation_time_ms'], 2),
            'cached_rule_sets': len(self.overlay_rules),
            'strict_validation_enabled': self.config['strict_validation'],
            'auto_remediation_enabled': self.config['enable_auto_remediation']
        }


# Global overlay validation engine instance
overlay_validation_engine = OverlayValidationEngine()
