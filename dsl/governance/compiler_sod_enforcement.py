"""
Task 8.3-T10: Implement SoD enforcement in Compiler
Segregation of Duties validation during workflow compilation
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class SoDRuleType(Enum):
    """Types of SoD rules"""
    MAKER_CHECKER = "maker_checker"
    DUAL_CONTROL = "dual_control"
    ROLE_SEPARATION = "role_separation"
    TEMPORAL_SEPARATION = "temporal_separation"
    APPROVAL_CHAIN = "approval_chain"


class SoDViolationSeverity(Enum):
    """SoD violation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SoDRule:
    """Segregation of Duties rule definition"""
    rule_id: str
    rule_name: str
    rule_type: SoDRuleType
    
    # Rule configuration
    conflicting_roles: List[str] = field(default_factory=list)
    conflicting_actions: List[str] = field(default_factory=list)
    required_separation_hours: int = 0
    min_approvers: int = 2
    
    # Applicability
    workflow_types: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    industry_overlay: Optional[str] = None
    
    # Metadata
    severity: SoDViolationSeverity = SoDViolationSeverity.HIGH
    description: str = ""
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "conflicting_roles": self.conflicting_roles,
            "conflicting_actions": self.conflicting_actions,
            "required_separation_hours": self.required_separation_hours,
            "min_approvers": self.min_approvers,
            "workflow_types": self.workflow_types,
            "compliance_frameworks": self.compliance_frameworks,
            "industry_overlay": self.industry_overlay,
            "severity": self.severity.value,
            "description": self.description,
            "is_active": self.is_active
        }


@dataclass
class CompilerSoDViolation:
    """SoD violation detected during compilation"""
    violation_id: str
    rule_id: str
    rule_name: str
    violation_type: str
    severity: SoDViolationSeverity
    
    # Workflow context
    workflow_id: str
    workflow_name: str
    step_id: Optional[str] = None
    
    # Violation details
    violation_message: str = ""
    conflicting_elements: List[str] = field(default_factory=list)
    
    # Recommendations
    remediation_suggestions: List[str] = field(default_factory=list)
    
    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "step_id": self.step_id,
            "violation_message": self.violation_message,
            "conflicting_elements": self.conflicting_elements,
            "remediation_suggestions": self.remediation_suggestions,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class CompilerSoDValidationResult:
    """Result of SoD validation during compilation"""
    workflow_id: str
    is_valid: bool
    violations: List[CompilerSoDViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Validation metadata
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_duration_ms: float = 0.0
    rules_evaluated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "validation_duration_ms": self.validation_duration_ms,
            "rules_evaluated": self.rules_evaluated
        }


class CompilerSoDEnforcer:
    """
    SoD Enforcement in Compiler - Task 8.3-T10
    
    Validates Segregation of Duties rules during workflow compilation
    Prevents deployment of workflows that violate SoD requirements
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'strict_enforcement': True,
            'fail_on_critical_violations': True,
            'fail_on_high_violations': False,
            'enable_remediation_suggestions': True,
            'cache_rules': True,
            'validation_timeout_seconds': 30
        }
        
        # SoD rules cache
        self.sod_rules: Dict[str, SoDRule] = {}
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'critical_violations_detected': 0,
            'workflows_rejected': 0,
            'average_validation_time_ms': 0.0
        }
        
        # Initialize default SoD rules
        self._initialize_default_sod_rules()
    
    async def initialize(self) -> bool:
        """Initialize compiler SoD enforcer"""
        try:
            await self._create_sod_tables()
            await self._load_sod_rules()
            self.logger.info("✅ Compiler SoD enforcer initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize compiler SoD enforcer: {e}")
            return False
    
    async def validate_workflow_sod(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: int,
        industry_overlay: str = "SaaS",
        compliance_frameworks: List[str] = None
    ) -> CompilerSoDValidationResult:
        """
        Validate workflow against SoD rules during compilation
        
        This is the main entry point called by the DSL compiler
        """
        
        start_time = datetime.now(timezone.utc)
        
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_name = workflow_definition.get('name', 'Unknown Workflow')
        
        # Initialize result
        result = CompilerSoDValidationResult(
            workflow_id=workflow_id,
            is_valid=True
        )
        
        try:
            # Get applicable SoD rules
            applicable_rules = self._get_applicable_rules(
                workflow_definition, tenant_id, industry_overlay, compliance_frameworks or []
            )
            
            result.rules_evaluated = len(applicable_rules)
            
            # Validate against each rule
            for rule in applicable_rules:
                violations = await self._validate_against_rule(
                    workflow_definition, rule, tenant_id
                )
                result.violations.extend(violations)
            
            # Determine overall validation result
            critical_violations = [v for v in result.violations if v.severity == SoDViolationSeverity.CRITICAL]
            high_violations = [v for v in result.violations if v.severity == SoDViolationSeverity.HIGH]
            
            if self.config['fail_on_critical_violations'] and critical_violations:
                result.is_valid = False
                self.logger.warning(f"🚫 Workflow {workflow_id} rejected due to {len(critical_violations)} critical SoD violations")
            elif self.config['fail_on_high_violations'] and high_violations:
                result.is_valid = False
                self.logger.warning(f"🚫 Workflow {workflow_id} rejected due to {len(high_violations)} high SoD violations")
            
            # Add warnings for non-blocking violations
            if not result.is_valid:
                pass  # Already handled above
            elif high_violations:
                result.warnings.append(f"High severity SoD violations detected: {len(high_violations)}")
            
            # Calculate validation duration
            end_time = datetime.now(timezone.utc)
            result.validation_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_validation_stats(result)
            
            # Store validation result
            if self.db_pool:
                await self._store_validation_result(result, tenant_id)
            
            if result.is_valid:
                self.logger.info(f"✅ SoD validation passed for workflow {workflow_id} ({result.validation_duration_ms:.2f}ms)")
            else:
                self.logger.warning(f"❌ SoD validation failed for workflow {workflow_id}")
                for violation in result.violations:
                    if violation.severity in [SoDViolationSeverity.CRITICAL, SoDViolationSeverity.HIGH]:
                        self.logger.warning(f"🚫 {violation.severity.value.upper()}: {violation.violation_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ SoD validation error for workflow {workflow_id}: {e}")
            
            # Fail closed on validation errors
            result.is_valid = False
            result.violations.append(CompilerSoDViolation(
                violation_id=f"validation_error_{workflow_id}",
                rule_id="system_error",
                rule_name="System Error",
                violation_type="validation_error",
                severity=SoDViolationSeverity.CRITICAL,
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                violation_message=f"SoD validation system error: {str(e)}",
                remediation_suggestions=["Contact system administrator", "Check workflow definition syntax"]
            ))
            
            result.validation_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
    
    def _initialize_default_sod_rules(self):
        """Initialize default SoD rules"""
        
        # Maker-Checker rule for financial workflows
        maker_checker_rule = SoDRule(
            rule_id="maker_checker_financial",
            rule_name="Maker-Checker for Financial Workflows",
            rule_type=SoDRuleType.MAKER_CHECKER,
            conflicting_actions=["create", "approve"],
            min_approvers=2,
            workflow_types=["financial", "payment", "invoice"],
            compliance_frameworks=["SOX", "RBI"],
            severity=SoDViolationSeverity.CRITICAL,
            description="Financial workflows require maker-checker separation"
        )
        
        # Role separation rule
        role_separation_rule = SoDRule(
            rule_id="role_separation_approval",
            rule_name="Role Separation for Approvals",
            rule_type=SoDRuleType.ROLE_SEPARATION,
            conflicting_roles=["creator", "approver"],
            workflow_types=["approval", "authorization"],
            compliance_frameworks=["SOX", "HIPAA"],
            severity=SoDViolationSeverity.HIGH,
            description="Creators cannot approve their own workflows"
        )
        
        # Dual control for high-risk operations
        dual_control_rule = SoDRule(
            rule_id="dual_control_high_risk",
            rule_name="Dual Control for High-Risk Operations",
            rule_type=SoDRuleType.DUAL_CONTROL,
            min_approvers=2,
            workflow_types=["high_risk", "critical", "security"],
            compliance_frameworks=["SOX", "RBI", "HIPAA"],
            severity=SoDViolationSeverity.CRITICAL,
            description="High-risk operations require dual control"
        )
        
        # Temporal separation rule
        temporal_separation_rule = SoDRule(
            rule_id="temporal_separation_24h",
            rule_name="24-Hour Temporal Separation",
            rule_type=SoDRuleType.TEMPORAL_SEPARATION,
            conflicting_actions=["create", "execute"],
            required_separation_hours=24,
            workflow_types=["scheduled", "batch"],
            compliance_frameworks=["SOX"],
            severity=SoDViolationSeverity.MEDIUM,
            description="24-hour separation between creation and execution"
        )
        
        # Store rules
        self.sod_rules = {
            rule.rule_id: rule for rule in [
                maker_checker_rule, role_separation_rule, 
                dual_control_rule, temporal_separation_rule
            ]
        }
    
    def _get_applicable_rules(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: int,
        industry_overlay: str,
        compliance_frameworks: List[str]
    ) -> List[SoDRule]:
        """Get SoD rules applicable to this workflow"""
        
        applicable_rules = []
        workflow_type = workflow_definition.get('type', 'general')
        workflow_tags = workflow_definition.get('tags', [])
        
        for rule in self.sod_rules.values():
            if not rule.is_active:
                continue
            
            # Check industry overlay
            if rule.industry_overlay and rule.industry_overlay != industry_overlay:
                continue
            
            # Check workflow types
            if rule.workflow_types:
                if not any(wt in [workflow_type] + workflow_tags for wt in rule.workflow_types):
                    continue
            
            # Check compliance frameworks
            if rule.compliance_frameworks:
                if not any(cf in compliance_frameworks for cf in rule.compliance_frameworks):
                    continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _validate_against_rule(
        self,
        workflow_definition: Dict[str, Any],
        rule: SoDRule,
        tenant_id: int
    ) -> List[CompilerSoDViolation]:
        """Validate workflow against specific SoD rule"""
        
        violations = []
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_name = workflow_definition.get('name', 'Unknown Workflow')
        
        if rule.rule_type == SoDRuleType.MAKER_CHECKER:
            violations.extend(await self._validate_maker_checker(
                workflow_definition, rule, tenant_id
            ))
        
        elif rule.rule_type == SoDRuleType.ROLE_SEPARATION:
            violations.extend(await self._validate_role_separation(
                workflow_definition, rule, tenant_id
            ))
        
        elif rule.rule_type == SoDRuleType.DUAL_CONTROL:
            violations.extend(await self._validate_dual_control(
                workflow_definition, rule, tenant_id
            ))
        
        elif rule.rule_type == SoDRuleType.TEMPORAL_SEPARATION:
            violations.extend(await self._validate_temporal_separation(
                workflow_definition, rule, tenant_id
            ))
        
        elif rule.rule_type == SoDRuleType.APPROVAL_CHAIN:
            violations.extend(await self._validate_approval_chain(
                workflow_definition, rule, tenant_id
            ))
        
        return violations
    
    async def _validate_maker_checker(
        self,
        workflow_definition: Dict[str, Any],
        rule: SoDRule,
        tenant_id: int
    ) -> List[CompilerSoDViolation]:
        """Validate maker-checker separation"""
        
        violations = []
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_name = workflow_definition.get('name', 'Unknown Workflow')
        
        # Check if workflow has both create and approve steps
        steps = workflow_definition.get('steps', [])
        create_steps = [s for s in steps if s.get('action') in ['create', 'submit', 'request']]
        approve_steps = [s for s in steps if s.get('action') in ['approve', 'authorize', 'sign_off']]
        
        if create_steps and approve_steps:
            # Check if same role can perform both actions
            create_roles = set()
            approve_roles = set()
            
            for step in create_steps:
                step_roles = step.get('allowed_roles', [])
                create_roles.update(step_roles)
            
            for step in approve_steps:
                step_roles = step.get('allowed_roles', [])
                approve_roles.update(step_roles)
            
            # Check for role overlap
            overlapping_roles = create_roles.intersection(approve_roles)
            
            if overlapping_roles:
                violation = CompilerSoDViolation(
                    violation_id=f"maker_checker_{workflow_id}_{rule.rule_id}",
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    violation_type="maker_checker_violation",
                    severity=rule.severity,
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    violation_message=f"Maker-checker violation: Roles {list(overlapping_roles)} can both create and approve",
                    conflicting_elements=list(overlapping_roles),
                    remediation_suggestions=[
                        "Separate create and approve roles",
                        "Add approval step with different role",
                        "Implement dual control mechanism"
                    ]
                )
                violations.append(violation)
        
        return violations
    
    async def _validate_role_separation(
        self,
        workflow_definition: Dict[str, Any],
        rule: SoDRule,
        tenant_id: int
    ) -> List[CompilerSoDViolation]:
        """Validate role separation requirements"""
        
        violations = []
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_name = workflow_definition.get('name', 'Unknown Workflow')
        
        # Check for conflicting roles in workflow steps
        steps = workflow_definition.get('steps', [])
        
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps[i+1:], i+1):
                roles1 = set(step1.get('allowed_roles', []))
                roles2 = set(step2.get('allowed_roles', []))
                
                # Check for conflicting role combinations
                for conflicting_role_pair in [rule.conflicting_roles[i:i+2] for i in range(0, len(rule.conflicting_roles), 2)]:
                    if len(conflicting_role_pair) == 2:
                        role1, role2 = conflicting_role_pair
                        
                        if role1 in roles1 and role2 in roles2:
                            violation = CompilerSoDViolation(
                                violation_id=f"role_separation_{workflow_id}_{rule.rule_id}_{i}_{j}",
                                rule_id=rule.rule_id,
                                rule_name=rule.rule_name,
                                violation_type="role_separation_violation",
                                severity=rule.severity,
                                workflow_id=workflow_id,
                                workflow_name=workflow_name,
                                step_id=f"{step1.get('id', i)}_{step2.get('id', j)}",
                                violation_message=f"Role separation violation: {role1} and {role2} cannot be in same workflow",
                                conflicting_elements=[role1, role2],
                                remediation_suggestions=[
                                    f"Remove {role1} or {role2} from workflow steps",
                                    "Use different roles for conflicting actions",
                                    "Add intermediate approval step"
                                ]
                            )
                            violations.append(violation)
        
        return violations
    
    async def _validate_dual_control(
        self,
        workflow_definition: Dict[str, Any],
        rule: SoDRule,
        tenant_id: int
    ) -> List[CompilerSoDViolation]:
        """Validate dual control requirements"""
        
        violations = []
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_name = workflow_definition.get('name', 'Unknown Workflow')
        
        # Check if workflow has sufficient approvers
        steps = workflow_definition.get('steps', [])
        approval_steps = [s for s in steps if s.get('type') == 'approval' or 'approve' in s.get('action', '')]
        
        for step in approval_steps:
            min_approvers = step.get('min_approvers', 1)
            
            if min_approvers < rule.min_approvers:
                violation = CompilerSoDViolation(
                    violation_id=f"dual_control_{workflow_id}_{rule.rule_id}_{step.get('id', 'unknown')}",
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    violation_type="dual_control_violation",
                    severity=rule.severity,
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    step_id=step.get('id', 'unknown'),
                    violation_message=f"Dual control violation: Step requires {rule.min_approvers} approvers, only {min_approvers} configured",
                    conflicting_elements=[f"min_approvers: {min_approvers}"],
                    remediation_suggestions=[
                        f"Increase min_approvers to {rule.min_approvers}",
                        "Add additional approval steps",
                        "Configure approval chain with multiple roles"
                    ]
                )
                violations.append(violation)
        
        return violations
    
    async def _validate_temporal_separation(
        self,
        workflow_definition: Dict[str, Any],
        rule: SoDRule,
        tenant_id: int
    ) -> List[CompilerSoDViolation]:
        """Validate temporal separation requirements"""
        
        violations = []
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_name = workflow_definition.get('name', 'Unknown Workflow')
        
        # Check if workflow has temporal separation configured
        steps = workflow_definition.get('steps', [])
        
        # Look for conflicting actions that should be temporally separated
        conflicting_steps = []
        for step in steps:
            step_action = step.get('action', '')
            if step_action in rule.conflicting_actions:
                conflicting_steps.append((step, step_action))
        
        if len(conflicting_steps) >= 2:
            # Check if temporal separation is configured
            has_temporal_separation = False
            
            for step, action in conflicting_steps:
                if step.get('delay_hours', 0) >= rule.required_separation_hours:
                    has_temporal_separation = True
                    break
            
            if not has_temporal_separation:
                violation = CompilerSoDViolation(
                    violation_id=f"temporal_separation_{workflow_id}_{rule.rule_id}",
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    violation_type="temporal_separation_violation",
                    severity=rule.severity,
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    violation_message=f"Temporal separation violation: {rule.required_separation_hours}h separation required between {rule.conflicting_actions}",
                    conflicting_elements=rule.conflicting_actions,
                    remediation_suggestions=[
                        f"Add {rule.required_separation_hours}h delay between conflicting actions",
                        "Use scheduled execution for temporal separation",
                        "Split workflow into separate time-delayed workflows"
                    ]
                )
                violations.append(violation)
        
        return violations
    
    async def _validate_approval_chain(
        self,
        workflow_definition: Dict[str, Any],
        rule: SoDRule,
        tenant_id: int
    ) -> List[CompilerSoDViolation]:
        """Validate approval chain requirements"""
        
        violations = []
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        workflow_name = workflow_definition.get('name', 'Unknown Workflow')
        
        # Check if workflow has proper approval chain
        steps = workflow_definition.get('steps', [])
        approval_steps = [s for s in steps if s.get('type') == 'approval']
        
        if len(approval_steps) < rule.min_approvers:
            violation = CompilerSoDViolation(
                violation_id=f"approval_chain_{workflow_id}_{rule.rule_id}",
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                violation_type="approval_chain_violation",
                severity=rule.severity,
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                violation_message=f"Approval chain violation: {rule.min_approvers} approval steps required, only {len(approval_steps)} configured",
                conflicting_elements=[f"approval_steps: {len(approval_steps)}"],
                remediation_suggestions=[
                    f"Add {rule.min_approvers - len(approval_steps)} more approval steps",
                    "Configure hierarchical approval chain",
                    "Add role-based approval sequence"
                ]
            )
            violations.append(violation)
        
        return violations
    
    def _update_validation_stats(self, result: CompilerSoDValidationResult):
        """Update validation statistics"""
        
        self.validation_stats['total_validations'] += 1
        
        if result.is_valid:
            self.validation_stats['validations_passed'] += 1
        else:
            self.validation_stats['validations_failed'] += 1
            self.validation_stats['workflows_rejected'] += 1
        
        # Count critical violations
        critical_violations = [v for v in result.violations if v.severity == SoDViolationSeverity.CRITICAL]
        self.validation_stats['critical_violations_detected'] += len(critical_violations)
        
        # Update average validation time
        current_avg = self.validation_stats['average_validation_time_ms']
        total_validations = self.validation_stats['total_validations']
        self.validation_stats['average_validation_time_ms'] = (
            (current_avg * (total_validations - 1) + result.validation_duration_ms) / total_validations
        )
    
    async def _create_sod_tables(self):
        """Create SoD validation tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Compiler SoD validation results
        CREATE TABLE IF NOT EXISTS compiler_sod_validations (
            validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Validation results
            is_valid BOOLEAN NOT NULL,
            violations_count INTEGER NOT NULL DEFAULT 0,
            warnings_count INTEGER NOT NULL DEFAULT 0,
            rules_evaluated INTEGER NOT NULL DEFAULT 0,
            
            -- Performance
            validation_duration_ms FLOAT NOT NULL DEFAULT 0,
            
            -- Context
            industry_overlay VARCHAR(50),
            compliance_frameworks TEXT[] DEFAULT ARRAY[],
            
            -- Timestamps
            validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_compiler_sod_counts CHECK (violations_count >= 0 AND warnings_count >= 0)
        );
        
        -- Compiler SoD violations
        CREATE TABLE IF NOT EXISTS compiler_sod_violations (
            violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            validation_id UUID REFERENCES compiler_sod_validations(validation_id) ON DELETE CASCADE,
            
            -- Violation details
            rule_id VARCHAR(100) NOT NULL,
            rule_name VARCHAR(200) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            
            -- Context
            workflow_id VARCHAR(100) NOT NULL,
            workflow_name VARCHAR(200),
            step_id VARCHAR(100),
            
            -- Details
            violation_message TEXT NOT NULL,
            conflicting_elements TEXT[] DEFAULT ARRAY[],
            remediation_suggestions TEXT[] DEFAULT ARRAY[],
            
            -- Timestamps
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_compiler_sod_severity CHECK (severity IN ('critical', 'high', 'medium', 'low'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_compiler_sod_validations_workflow ON compiler_sod_validations(workflow_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compiler_sod_validations_tenant ON compiler_sod_validations(tenant_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compiler_sod_validations_valid ON compiler_sod_validations(is_valid, validated_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_compiler_sod_violations_validation ON compiler_sod_violations(validation_id);
        CREATE INDEX IF NOT EXISTS idx_compiler_sod_violations_rule ON compiler_sod_violations(rule_id, severity);
        CREATE INDEX IF NOT EXISTS idx_compiler_sod_violations_workflow ON compiler_sod_violations(workflow_id, detected_at DESC);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Compiler SoD validation tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create compiler SoD tables: {e}")
            raise
    
    async def _load_sod_rules(self):
        """Load custom SoD rules from database"""
        # In production, this would load custom rules from database
        # For now, we use the default rules initialized in memory
        pass
    
    async def _store_validation_result(
        self,
        result: CompilerSoDValidationResult,
        tenant_id: int
    ):
        """Store validation result in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert validation result
                validation_query = """
                    INSERT INTO compiler_sod_validations (
                        workflow_id, tenant_id, is_valid, violations_count,
                        warnings_count, rules_evaluated, validation_duration_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING validation_id
                """
                
                validation_id = await conn.fetchval(
                    validation_query,
                    result.workflow_id,
                    tenant_id,
                    result.is_valid,
                    len(result.violations),
                    len(result.warnings),
                    result.rules_evaluated,
                    result.validation_duration_ms
                )
                
                # Insert violations
                if result.violations:
                    violation_query = """
                        INSERT INTO compiler_sod_violations (
                            validation_id, rule_id, rule_name, violation_type,
                            severity, workflow_id, workflow_name, step_id,
                            violation_message, conflicting_elements, remediation_suggestions,
                            detected_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """
                    
                    for violation in result.violations:
                        await conn.execute(
                            violation_query,
                            validation_id,
                            violation.rule_id,
                            violation.rule_name,
                            violation.violation_type,
                            violation.severity.value,
                            violation.workflow_id,
                            violation.workflow_name,
                            violation.step_id,
                            violation.violation_message,
                            violation.conflicting_elements,
                            violation.remediation_suggestions,
                            violation.detected_at
                        )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store compiler SoD validation result: {e}")
    
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
            'workflows_rejected': self.validation_stats['workflows_rejected'],
            'critical_violations_detected': self.validation_stats['critical_violations_detected'],
            'success_rate_percentage': round(success_rate, 2),
            'average_validation_time_ms': round(self.validation_stats['average_validation_time_ms'], 2),
            'active_rules_count': len([r for r in self.sod_rules.values() if r.is_active]),
            'strict_enforcement_enabled': self.config['strict_enforcement']
        }


# Global compiler SoD enforcer instance
compiler_sod_enforcer = CompilerSoDEnforcer()
