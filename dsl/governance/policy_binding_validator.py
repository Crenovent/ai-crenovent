"""
Task 8.1-T20: Train compiler/runtime to reject workflows missing policy bindings
Fail-closed validation ensuring all workflows have required policy bindings
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class PolicyBindingType(Enum):
    """Types of policy bindings"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"


class ValidationSeverity(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PolicyBindingRequirement:
    """Policy binding requirement definition"""
    requirement_id: str
    policy_type: str
    binding_type: PolicyBindingType
    
    # Conditions
    workflow_types: List[str] = field(default_factory=list)
    industry_overlays: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    
    # Validation
    severity: ValidationSeverity = ValidationSeverity.CRITICAL
    error_message: str = ""
    
    # Metadata
    description: str = ""
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "policy_type": self.policy_type,
            "binding_type": self.binding_type.value,
            "workflow_types": self.workflow_types,
            "industry_overlays": self.industry_overlays,
            "compliance_frameworks": self.compliance_frameworks,
            "severity": self.severity.value,
            "error_message": self.error_message,
            "description": self.description,
            "is_active": self.is_active
        }


@dataclass
class PolicyBindingViolation:
    """Policy binding validation violation"""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requirement_id: str = ""
    
    # Violation details
    violation_type: str = "missing_binding"
    severity: ValidationSeverity = ValidationSeverity.CRITICAL
    message: str = ""
    
    # Context
    workflow_id: str = ""
    workflow_type: str = ""
    missing_policy_type: str = ""
    
    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "requirement_id": self.requirement_id,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
            "message": self.message,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "missing_policy_type": self.missing_policy_type,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class PolicyBindingValidationResult:
    """Result of policy binding validation"""
    workflow_id: str
    is_valid: bool = True
    violations: List[PolicyBindingViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Statistics
    requirements_checked: int = 0
    bindings_found: int = 0
    critical_violations: int = 0
    
    # Performance
    validation_duration_ms: float = 0.0
    
    # Timestamps
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "requirements_checked": self.requirements_checked,
            "bindings_found": self.bindings_found,
            "critical_violations": self.critical_violations,
            "validation_duration_ms": self.validation_duration_ms,
            "validated_at": self.validated_at.isoformat()
        }


class PolicyBindingValidator:
    """
    Policy Binding Validator - Task 8.1-T20
    
    Ensures all workflows have required policy bindings before execution
    Implements fail-closed validation for governance compliance
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'fail_closed': True,
            'strict_validation': True,
            'enable_audit_trail': True,
            'cache_requirements': True,
            'validation_timeout_seconds': 30
        }
        
        # Policy binding requirements
        self.binding_requirements: List[PolicyBindingRequirement] = []
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'workflows_rejected': 0,
            'critical_violations_detected': 0,
            'average_validation_time_ms': 0.0
        }
        
        # Initialize default requirements
        self._initialize_default_requirements()
    
    async def initialize(self) -> bool:
        """Initialize policy binding validator"""
        try:
            await self._create_validation_tables()
            await self._load_binding_requirements()
            self.logger.info("✅ Policy binding validator initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize policy binding validator: {e}")
            return False
    
    async def validate_workflow_policy_bindings(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: int,
        industry_overlay: str = "SaaS",
        compliance_frameworks: List[str] = None
    ) -> PolicyBindingValidationResult:
        """
        Validate workflow policy bindings
        
        This is the main entry point called by compiler/runtime
        """
        
        start_time = datetime.now(timezone.utc)
        
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_type = workflow_definition.get('type', 'general')
        
        # Initialize result
        result = PolicyBindingValidationResult(workflow_id=workflow_id)
        
        try:
            # Get applicable requirements
            applicable_requirements = self._get_applicable_requirements(
                workflow_type, industry_overlay, compliance_frameworks or []
            )
            
            result.requirements_checked = len(applicable_requirements)
            
            # Get workflow policy bindings
            workflow_bindings = self._extract_policy_bindings(workflow_definition)
            result.bindings_found = len(workflow_bindings)
            
            # Validate each requirement
            for requirement in applicable_requirements:
                violation = await self._validate_requirement(
                    requirement, workflow_definition, workflow_bindings
                )
                
                if violation:
                    result.violations.append(violation)
                    
                    if violation.severity == ValidationSeverity.CRITICAL:
                        result.critical_violations += 1
            
            # Determine overall validation result
            if result.critical_violations > 0 and self.config['fail_closed']:
                result.is_valid = False
                self.logger.warning(f"🚫 Workflow {workflow_id} REJECTED: {result.critical_violations} critical policy binding violations")
            
            # Add warnings for non-critical violations
            non_critical_violations = [v for v in result.violations if v.severity != ValidationSeverity.CRITICAL]
            if non_critical_violations:
                result.warnings.append(f"Non-critical policy binding issues detected: {len(non_critical_violations)}")
            
            # Calculate validation duration
            end_time = datetime.now(timezone.utc)
            result.validation_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_validation_stats(result)
            
            # Store validation result
            if self.db_pool and self.config['enable_audit_trail']:
                await self._store_validation_result(result, tenant_id)
            
            if result.is_valid:
                self.logger.info(f"✅ Policy binding validation passed for workflow {workflow_id}")
            else:
                self.logger.error(f"❌ Policy binding validation FAILED for workflow {workflow_id}")
                for violation in result.violations:
                    if violation.severity == ValidationSeverity.CRITICAL:
                        self.logger.error(f"🚫 CRITICAL: {violation.message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Policy binding validation error: {e}")
            
            # Fail closed on validation errors
            if self.config['fail_closed']:
                result.is_valid = False
                result.violations.append(PolicyBindingViolation(
                    requirement_id="system_error",
                    violation_type="validation_error",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Policy binding validation system error: {str(e)}",
                    workflow_id=workflow_id,
                    workflow_type=workflow_type
                ))
            
            result.validation_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
    
    def _initialize_default_requirements(self):
        """Initialize default policy binding requirements"""
        
        # Financial workflows must have SOX compliance policy
        sox_requirement = PolicyBindingRequirement(
            requirement_id="sox_financial_binding",
            policy_type="sox_compliance",
            binding_type=PolicyBindingType.REQUIRED,
            workflow_types=["financial", "revenue", "billing", "payment"],
            industry_overlays=["SaaS", "Banking", "Insurance"],
            compliance_frameworks=["SOX"],
            severity=ValidationSeverity.CRITICAL,
            error_message="Financial workflows must have SOX compliance policy binding",
            description="SOX compliance requires policy binding for financial processes"
        )
        
        # Data processing workflows must have GDPR policy
        gdpr_requirement = PolicyBindingRequirement(
            requirement_id="gdpr_data_binding",
            policy_type="gdpr_compliance",
            binding_type=PolicyBindingType.REQUIRED,
            workflow_types=["data_processing", "customer_data", "analytics"],
            industry_overlays=["SaaS", "Healthcare"],
            compliance_frameworks=["GDPR"],
            severity=ValidationSeverity.CRITICAL,
            error_message="Data processing workflows must have GDPR compliance policy binding",
            description="GDPR compliance requires policy binding for data processing"
        )
        
        # Healthcare workflows must have HIPAA policy
        hipaa_requirement = PolicyBindingRequirement(
            requirement_id="hipaa_healthcare_binding",
            policy_type="hipaa_compliance",
            binding_type=PolicyBindingType.REQUIRED,
            workflow_types=["patient_data", "medical", "phi_processing"],
            industry_overlays=["Healthcare"],
            compliance_frameworks=["HIPAA"],
            severity=ValidationSeverity.CRITICAL,
            error_message="Healthcare workflows must have HIPAA compliance policy binding",
            description="HIPAA compliance requires policy binding for PHI processing"
        )
        
        # Banking workflows must have RBI policy
        rbi_requirement = PolicyBindingRequirement(
            requirement_id="rbi_banking_binding",
            policy_type="rbi_compliance",
            binding_type=PolicyBindingType.REQUIRED,
            workflow_types=["loan", "credit", "banking", "kyc"],
            industry_overlays=["Banking"],
            compliance_frameworks=["RBI"],
            severity=ValidationSeverity.CRITICAL,
            error_message="Banking workflows must have RBI compliance policy binding",
            description="RBI compliance requires policy binding for banking operations"
        )
        
        # All workflows must have audit policy
        audit_requirement = PolicyBindingRequirement(
            requirement_id="universal_audit_binding",
            policy_type="audit_trail",
            binding_type=PolicyBindingType.REQUIRED,
            workflow_types=[],  # Applies to all workflow types
            industry_overlays=["SaaS", "Banking", "Insurance", "Healthcare"],
            compliance_frameworks=["SOX", "GDPR", "HIPAA", "RBI"],
            severity=ValidationSeverity.HIGH,
            error_message="All workflows must have audit trail policy binding",
            description="Universal audit trail requirement for compliance"
        )
        
        # Store requirements
        self.binding_requirements = [
            sox_requirement, gdpr_requirement, hipaa_requirement, 
            rbi_requirement, audit_requirement
        ]
    
    def _get_applicable_requirements(
        self,
        workflow_type: str,
        industry_overlay: str,
        compliance_frameworks: List[str]
    ) -> List[PolicyBindingRequirement]:
        """Get applicable policy binding requirements"""
        
        applicable = []
        
        for requirement in self.binding_requirements:
            if not requirement.is_active:
                continue
            
            # Check workflow type (empty list means applies to all)
            if requirement.workflow_types and workflow_type not in requirement.workflow_types:
                continue
            
            # Check industry overlay
            if requirement.industry_overlays and industry_overlay not in requirement.industry_overlays:
                continue
            
            # Check compliance frameworks
            if requirement.compliance_frameworks:
                if not any(cf in compliance_frameworks for cf in requirement.compliance_frameworks):
                    continue
            
            applicable.append(requirement)
        
        return applicable
    
    def _extract_policy_bindings(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Extract policy bindings from workflow definition"""
        
        bindings = {}
        
        # Check governance section
        governance = workflow_definition.get('governance', {})
        policy_bindings = governance.get('policy_bindings', {})
        
        for policy_type, binding_config in policy_bindings.items():
            bindings[policy_type] = binding_config
        
        # Check for legacy policy references
        policies = workflow_definition.get('policies', [])
        for policy in policies:
            policy_type = policy.get('type', 'unknown')
            bindings[policy_type] = policy
        
        # Check steps for policy references
        steps = workflow_definition.get('steps', [])
        for step in steps:
            step_policies = step.get('policies', [])
            for policy in step_policies:
                policy_type = policy.get('type', 'unknown')
                bindings[f"step_{policy_type}"] = policy
        
        return bindings
    
    async def _validate_requirement(
        self,
        requirement: PolicyBindingRequirement,
        workflow_definition: Dict[str, Any],
        workflow_bindings: Dict[str, Any]
    ) -> Optional[PolicyBindingViolation]:
        """Validate individual policy binding requirement"""
        
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_type = workflow_definition.get('type', 'general')
        
        # Check if required policy binding exists
        if requirement.binding_type == PolicyBindingType.REQUIRED:
            if requirement.policy_type not in workflow_bindings:
                return PolicyBindingViolation(
                    requirement_id=requirement.requirement_id,
                    violation_type="missing_required_binding",
                    severity=requirement.severity,
                    message=requirement.error_message,
                    workflow_id=workflow_id,
                    workflow_type=workflow_type,
                    missing_policy_type=requirement.policy_type
                )
        
        # Check conditional bindings
        elif requirement.binding_type == PolicyBindingType.CONDITIONAL:
            # Implement conditional logic based on workflow characteristics
            if self._should_have_conditional_binding(workflow_definition, requirement):
                if requirement.policy_type not in workflow_bindings:
                    return PolicyBindingViolation(
                        requirement_id=requirement.requirement_id,
                        violation_type="missing_conditional_binding",
                        severity=requirement.severity,
                        message=f"Conditional policy binding required: {requirement.error_message}",
                        workflow_id=workflow_id,
                        workflow_type=workflow_type,
                        missing_policy_type=requirement.policy_type
                    )
        
        return None
    
    def _should_have_conditional_binding(
        self, workflow_definition: Dict[str, Any], requirement: PolicyBindingRequirement
    ) -> bool:
        """Determine if conditional policy binding should be required"""
        
        # Example conditional logic
        if requirement.policy_type == "pii_protection":
            # Check if workflow processes PII
            steps = workflow_definition.get('steps', [])
            for step in steps:
                if step.get('processes_pii', False):
                    return True
                if 'customer_data' in step.get('data_sources', []):
                    return True
        
        return False
    
    def _update_validation_stats(self, result: PolicyBindingValidationResult):
        """Update validation statistics"""
        
        self.validation_stats['total_validations'] += 1
        
        if result.is_valid:
            self.validation_stats['validations_passed'] += 1
        else:
            self.validation_stats['validations_failed'] += 1
            self.validation_stats['workflows_rejected'] += 1
        
        self.validation_stats['critical_violations_detected'] += result.critical_violations
        
        # Update average validation time
        current_avg = self.validation_stats['average_validation_time_ms']
        total_validations = self.validation_stats['total_validations']
        self.validation_stats['average_validation_time_ms'] = (
            (current_avg * (total_validations - 1) + result.validation_duration_ms) / total_validations
        )
    
    async def _create_validation_tables(self):
        """Create policy binding validation tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Policy binding validation results
        CREATE TABLE IF NOT EXISTS policy_binding_validations (
            validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Validation results
            is_valid BOOLEAN NOT NULL,
            violations_count INTEGER NOT NULL DEFAULT 0,
            warnings_count INTEGER NOT NULL DEFAULT 0,
            requirements_checked INTEGER NOT NULL DEFAULT 0,
            bindings_found INTEGER NOT NULL DEFAULT 0,
            critical_violations INTEGER NOT NULL DEFAULT 0,
            
            -- Performance
            validation_duration_ms FLOAT NOT NULL DEFAULT 0,
            
            -- Context
            workflow_type VARCHAR(50),
            industry_overlay VARCHAR(50),
            compliance_frameworks TEXT[] DEFAULT ARRAY[],
            
            -- Timestamps
            validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_policy_binding_counts CHECK (violations_count >= 0 AND warnings_count >= 0)
        );
        
        -- Policy binding violations
        CREATE TABLE IF NOT EXISTS policy_binding_violations (
            violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            validation_id UUID REFERENCES policy_binding_validations(validation_id) ON DELETE CASCADE,
            
            -- Violation details
            requirement_id VARCHAR(100) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            
            -- Context
            workflow_id VARCHAR(100) NOT NULL,
            workflow_type VARCHAR(50),
            missing_policy_type VARCHAR(100),
            
            -- Timestamps
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_policy_binding_severity CHECK (severity IN ('critical', 'high', 'medium', 'low'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_policy_binding_validations_workflow ON policy_binding_validations(workflow_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_policy_binding_validations_tenant ON policy_binding_validations(tenant_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_policy_binding_validations_valid ON policy_binding_validations(is_valid, validated_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_policy_binding_violations_validation ON policy_binding_violations(validation_id);
        CREATE INDEX IF NOT EXISTS idx_policy_binding_violations_requirement ON policy_binding_violations(requirement_id, severity);
        CREATE INDEX IF NOT EXISTS idx_policy_binding_violations_workflow ON policy_binding_violations(workflow_id, detected_at DESC);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Policy binding validation tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create policy binding validation tables: {e}")
            raise
    
    async def _load_binding_requirements(self):
        """Load custom binding requirements from database"""
        # In production, this would load custom requirements from database
        # For now, we use the default requirements initialized in memory
        pass
    
    async def _store_validation_result(
        self, result: PolicyBindingValidationResult, tenant_id: int
    ):
        """Store validation result in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert validation result
                validation_query = """
                    INSERT INTO policy_binding_validations (
                        workflow_id, tenant_id, is_valid, violations_count,
                        warnings_count, requirements_checked, bindings_found,
                        critical_violations, validation_duration_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING validation_id
                """
                
                validation_id = await conn.fetchval(
                    validation_query,
                    result.workflow_id,
                    tenant_id,
                    result.is_valid,
                    len(result.violations),
                    len(result.warnings),
                    result.requirements_checked,
                    result.bindings_found,
                    result.critical_violations,
                    result.validation_duration_ms
                )
                
                # Insert violations
                if result.violations:
                    violation_query = """
                        INSERT INTO policy_binding_violations (
                            validation_id, requirement_id, violation_type,
                            severity, message, workflow_id, workflow_type,
                            missing_policy_type, detected_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """
                    
                    for violation in result.violations:
                        await conn.execute(
                            violation_query,
                            validation_id,
                            violation.requirement_id,
                            violation.violation_type,
                            violation.severity.value,
                            violation.message,
                            violation.workflow_id,
                            violation.workflow_type,
                            violation.missing_policy_type,
                            violation.detected_at
                        )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store policy binding validation result: {e}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        success_rate = 0.0
        if self.validation_stats['total_validations'] > 0:
            success_rate = (
                self.validation_stats['validations_passed'] / 
                self.validation_stats['total_validations']
            ) * 100
        
        rejection_rate = 0.0
        if self.validation_stats['total_validations'] > 0:
            rejection_rate = (
                self.validation_stats['workflows_rejected'] / 
                self.validation_stats['total_validations']
            ) * 100
        
        return {
            'total_validations': self.validation_stats['total_validations'],
            'validations_passed': self.validation_stats['validations_passed'],
            'validations_failed': self.validation_stats['validations_failed'],
            'workflows_rejected': self.validation_stats['workflows_rejected'],
            'critical_violations_detected': self.validation_stats['critical_violations_detected'],
            'success_rate_percentage': round(success_rate, 2),
            'rejection_rate_percentage': round(rejection_rate, 2),
            'average_validation_time_ms': round(self.validation_stats['average_validation_time_ms'], 2),
            'active_requirements_count': len([r for r in self.binding_requirements if r.is_active]),
            'fail_closed_enabled': self.config['fail_closed'],
            'strict_validation_enabled': self.config['strict_validation']
        }


# Global policy binding validator instance
policy_binding_validator = PolicyBindingValidator()