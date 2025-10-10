"""
Task 8.3-T11: Add SoD enforcement in Runtime
Real-time Segregation of Duties validation during workflow execution
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class RuntimeSoDCheckType(Enum):
    """Types of runtime SoD checks"""
    PRE_EXECUTION = "pre_execution"
    STEP_EXECUTION = "step_execution"
    POST_EXECUTION = "post_execution"
    APPROVAL_CHECK = "approval_check"


class RuntimeSoDAction(Enum):
    """Actions to take on SoD violations"""
    BLOCK = "block"
    WARN = "warn"
    ESCALATE = "escalate"
    AUDIT_ONLY = "audit_only"


@dataclass
class RuntimeSoDContext:
    """Runtime context for SoD validation"""
    execution_id: str
    workflow_id: str
    step_id: Optional[str]
    tenant_id: int
    
    # Actor information
    current_actor_id: str
    current_actor_role: str
    current_actor_department: Optional[str] = None
    
    # Execution context
    execution_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_data: Dict[str, Any] = field(default_factory=dict)
    
    # Previous execution history
    previous_actors: List[Dict[str, Any]] = field(default_factory=list)
    approval_chain: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "tenant_id": self.tenant_id,
            "current_actor_id": self.current_actor_id,
            "current_actor_role": self.current_actor_role,
            "current_actor_department": self.current_actor_department,
            "execution_timestamp": self.execution_timestamp.isoformat(),
            "execution_data": self.execution_data,
            "previous_actors": self.previous_actors,
            "approval_chain": self.approval_chain
        }


@dataclass
class RuntimeSoDViolation:
    """Runtime SoD violation"""
    violation_id: str
    execution_id: str
    check_type: RuntimeSoDCheckType
    
    # Violation details
    violation_type: str
    severity: str
    violation_message: str
    
    # Context
    workflow_id: str
    step_id: Optional[str]
    current_actor_id: str
    conflicting_actor_id: Optional[str] = None
    
    # Action taken
    action_taken: RuntimeSoDAction = RuntimeSoDAction.BLOCK
    escalation_required: bool = False
    
    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "execution_id": self.execution_id,
            "check_type": self.check_type.value,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "violation_message": self.violation_message,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "current_actor_id": self.current_actor_id,
            "conflicting_actor_id": self.conflicting_actor_id,
            "action_taken": self.action_taken.value,
            "escalation_required": self.escalation_required,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class RuntimeSoDValidationResult:
    """Result of runtime SoD validation"""
    execution_id: str
    is_allowed: bool
    violations: List[RuntimeSoDViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Action to take
    recommended_action: RuntimeSoDAction = RuntimeSoDAction.BLOCK
    escalation_required: bool = False
    
    # Validation metadata
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_duration_ms: float = 0.0
    checks_performed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "is_allowed": self.is_allowed,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "recommended_action": self.recommended_action.value,
            "escalation_required": self.escalation_required,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "validation_duration_ms": self.validation_duration_ms,
            "checks_performed": self.checks_performed
        }


class RuntimeSoDEnforcer:
    """
    Runtime SoD Enforcement - Task 8.3-T11
    
    Enforces Segregation of Duties rules during workflow execution
    Provides real-time validation and blocking of SoD violations
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'strict_enforcement': True,
            'fail_closed_on_error': True,
            'enable_real_time_checks': True,
            'escalation_enabled': True,
            'audit_all_checks': True,
            'performance_threshold_ms': 50,
            'cache_actor_history': True,
            'cache_ttl_minutes': 15
        }
        
        # Actor history cache for performance
        self.actor_history_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Statistics
        self.enforcement_stats = {
            'total_checks': 0,
            'checks_passed': 0,
            'checks_failed': 0,
            'executions_blocked': 0,
            'escalations_triggered': 0,
            'average_check_time_ms': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize runtime SoD enforcer"""
        try:
            await self._create_runtime_sod_tables()
            self.logger.info("✅ Runtime SoD enforcer initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize runtime SoD enforcer: {e}")
            return False
    
    async def validate_execution_sod(
        self,
        context: RuntimeSoDContext,
        check_type: RuntimeSoDCheckType = RuntimeSoDCheckType.STEP_EXECUTION
    ) -> RuntimeSoDValidationResult:
        """
        Validate SoD compliance during workflow execution
        
        This is the main entry point called by the workflow runtime
        """
        
        start_time = datetime.now(timezone.utc)
        
        # Initialize result
        result = RuntimeSoDValidationResult(
            execution_id=context.execution_id,
            is_allowed=True
        )
        
        try:
            # Perform different types of SoD checks
            if check_type == RuntimeSoDCheckType.PRE_EXECUTION:
                await self._perform_pre_execution_checks(context, result)
            elif check_type == RuntimeSoDCheckType.STEP_EXECUTION:
                await self._perform_step_execution_checks(context, result)
            elif check_type == RuntimeSoDCheckType.POST_EXECUTION:
                await self._perform_post_execution_checks(context, result)
            elif check_type == RuntimeSoDCheckType.APPROVAL_CHECK:
                await self._perform_approval_checks(context, result)
            
            # Determine overall result
            critical_violations = [v for v in result.violations if v.severity == 'critical']
            high_violations = [v for v in result.violations if v.severity == 'high']
            
            if critical_violations:
                result.is_allowed = False
                result.recommended_action = RuntimeSoDAction.BLOCK
                result.escalation_required = True
            elif high_violations and self.config['strict_enforcement']:
                result.is_allowed = False
                result.recommended_action = RuntimeSoDAction.ESCALATE
                result.escalation_required = True
            elif high_violations:
                result.is_allowed = True
                result.recommended_action = RuntimeSoDAction.WARN
                result.warnings.append(f"High severity SoD violations detected: {len(high_violations)}")
            
            # Calculate validation duration
            end_time = datetime.now(timezone.utc)
            result.validation_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_enforcement_stats(result)
            
            # Store validation result
            if self.db_pool and self.config['audit_all_checks']:
                await self._store_validation_result(result, context)
            
            # Log result
            if result.is_allowed:
                self.logger.debug(f"✅ Runtime SoD check passed for execution {context.execution_id}")
            else:
                self.logger.warning(f"🚫 Runtime SoD check failed for execution {context.execution_id}: {result.recommended_action.value}")
                for violation in result.violations:
                    if violation.severity in ['critical', 'high']:
                        self.logger.warning(f"🚫 {violation.severity.upper()}: {violation.violation_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Runtime SoD validation error: {e}")
            
            # Fail closed on validation errors
            if self.config['fail_closed_on_error']:
                result.is_allowed = False
                result.recommended_action = RuntimeSoDAction.BLOCK
                result.violations.append(RuntimeSoDViolation(
                    violation_id=f"runtime_error_{context.execution_id}",
                    execution_id=context.execution_id,
                    check_type=check_type,
                    violation_type="system_error",
                    severity="critical",
                    violation_message=f"Runtime SoD validation system error: {str(e)}",
                    workflow_id=context.workflow_id,
                    step_id=context.step_id,
                    current_actor_id=context.current_actor_id,
                    action_taken=RuntimeSoDAction.BLOCK
                ))
            
            result.validation_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
    
    async def _perform_pre_execution_checks(
        self,
        context: RuntimeSoDContext,
        result: RuntimeSoDValidationResult
    ):
        """Perform SoD checks before workflow execution starts"""
        
        result.checks_performed += 1
        
        # Check if actor has permission to start this workflow
        if not await self._check_workflow_start_permission(context):
            violation = RuntimeSoDViolation(
                violation_id=f"pre_exec_{context.execution_id}_permission",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.PRE_EXECUTION,
                violation_type="insufficient_permission",
                severity="high",
                violation_message=f"Actor {context.current_actor_id} lacks permission to start workflow {context.workflow_id}",
                workflow_id=context.workflow_id,
                current_actor_id=context.current_actor_id,
                action_taken=RuntimeSoDAction.BLOCK
            )
            result.violations.append(violation)
        
        # Check for recent conflicting executions by same actor
        recent_executions = await self._get_recent_actor_executions(
            context.current_actor_id, context.tenant_id, hours=24
        )
        
        conflicting_executions = [
            exec for exec in recent_executions 
            if exec['workflow_id'] == context.workflow_id and 
               exec['execution_id'] != context.execution_id
        ]
        
        if len(conflicting_executions) >= 3:  # Threshold for suspicious activity
            violation = RuntimeSoDViolation(
                violation_id=f"pre_exec_{context.execution_id}_frequency",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.PRE_EXECUTION,
                violation_type="excessive_frequency",
                severity="medium",
                violation_message=f"Actor {context.current_actor_id} has executed this workflow {len(conflicting_executions)} times in 24h",
                workflow_id=context.workflow_id,
                current_actor_id=context.current_actor_id,
                action_taken=RuntimeSoDAction.WARN
            )
            result.violations.append(violation)
    
    async def _perform_step_execution_checks(
        self,
        context: RuntimeSoDContext,
        result: RuntimeSoDValidationResult
    ):
        """Perform SoD checks during step execution"""
        
        result.checks_performed += 1
        
        # Check for self-approval violations
        if await self._is_self_approval(context):
            violation = RuntimeSoDViolation(
                violation_id=f"step_exec_{context.execution_id}_self_approval",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.STEP_EXECUTION,
                violation_type="self_approval",
                severity="critical",
                violation_message=f"Self-approval detected: {context.current_actor_id} cannot approve their own request",
                workflow_id=context.workflow_id,
                step_id=context.step_id,
                current_actor_id=context.current_actor_id,
                action_taken=RuntimeSoDAction.BLOCK
            )
            result.violations.append(violation)
        
        # Check for same-role conflicts
        same_role_conflicts = await self._check_same_role_conflicts(context)
        for conflict in same_role_conflicts:
            violation = RuntimeSoDViolation(
                violation_id=f"step_exec_{context.execution_id}_same_role_{conflict['actor_id']}",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.STEP_EXECUTION,
                violation_type="same_role_conflict",
                severity="high",
                violation_message=f"Same role conflict: {context.current_actor_role} already involved by {conflict['actor_id']}",
                workflow_id=context.workflow_id,
                step_id=context.step_id,
                current_actor_id=context.current_actor_id,
                conflicting_actor_id=conflict['actor_id'],
                action_taken=RuntimeSoDAction.ESCALATE
            )
            result.violations.append(violation)
        
        # Check for department conflicts
        if await self._check_department_conflict(context):
            violation = RuntimeSoDViolation(
                violation_id=f"step_exec_{context.execution_id}_dept_conflict",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.STEP_EXECUTION,
                violation_type="department_conflict",
                severity="medium",
                violation_message=f"Department conflict: Multiple actors from {context.current_actor_department}",
                workflow_id=context.workflow_id,
                step_id=context.step_id,
                current_actor_id=context.current_actor_id,
                action_taken=RuntimeSoDAction.WARN
            )
            result.violations.append(violation)
    
    async def _perform_post_execution_checks(
        self,
        context: RuntimeSoDContext,
        result: RuntimeSoDValidationResult
    ):
        """Perform SoD checks after workflow execution completes"""
        
        result.checks_performed += 1
        
        # Validate complete approval chain
        approval_chain_valid = await self._validate_approval_chain(context)
        if not approval_chain_valid:
            violation = RuntimeSoDViolation(
                violation_id=f"post_exec_{context.execution_id}_approval_chain",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.POST_EXECUTION,
                violation_type="invalid_approval_chain",
                severity="high",
                violation_message="Approval chain validation failed - insufficient separation",
                workflow_id=context.workflow_id,
                current_actor_id=context.current_actor_id,
                action_taken=RuntimeSoDAction.AUDIT_ONLY  # Post-execution, can only audit
            )
            result.violations.append(violation)
        
        # Check for pattern anomalies
        if await self._detect_execution_patterns(context):
            violation = RuntimeSoDViolation(
                violation_id=f"post_exec_{context.execution_id}_pattern_anomaly",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.POST_EXECUTION,
                violation_type="pattern_anomaly",
                severity="medium",
                violation_message="Unusual execution pattern detected - potential SoD circumvention",
                workflow_id=context.workflow_id,
                current_actor_id=context.current_actor_id,
                action_taken=RuntimeSoDAction.AUDIT_ONLY
            )
            result.violations.append(violation)
    
    async def _perform_approval_checks(
        self,
        context: RuntimeSoDContext,
        result: RuntimeSoDValidationResult
    ):
        """Perform SoD checks specific to approval steps"""
        
        result.checks_performed += 1
        
        # Check approval authority
        if not await self._check_approval_authority(context):
            violation = RuntimeSoDViolation(
                violation_id=f"approval_{context.execution_id}_authority",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.APPROVAL_CHECK,
                violation_type="insufficient_authority",
                severity="critical",
                violation_message=f"Actor {context.current_actor_id} lacks authority for this approval",
                workflow_id=context.workflow_id,
                step_id=context.step_id,
                current_actor_id=context.current_actor_id,
                action_taken=RuntimeSoDAction.BLOCK
            )
            result.violations.append(violation)
        
        # Check for approval conflicts
        approval_conflicts = await self._check_approval_conflicts(context)
        for conflict in approval_conflicts:
            violation = RuntimeSoDViolation(
                violation_id=f"approval_{context.execution_id}_conflict_{conflict['type']}",
                execution_id=context.execution_id,
                check_type=RuntimeSoDCheckType.APPROVAL_CHECK,
                violation_type=f"approval_conflict_{conflict['type']}",
                severity=conflict['severity'],
                violation_message=conflict['message'],
                workflow_id=context.workflow_id,
                step_id=context.step_id,
                current_actor_id=context.current_actor_id,
                conflicting_actor_id=conflict.get('conflicting_actor'),
                action_taken=RuntimeSoDAction.BLOCK if conflict['severity'] == 'critical' else RuntimeSoDAction.ESCALATE
            )
            result.violations.append(violation)
    
    # Helper methods for SoD checks
    
    async def _check_workflow_start_permission(self, context: RuntimeSoDContext) -> bool:
        """Check if actor has permission to start workflow"""
        # In production, this would check RBAC permissions
        # For now, return True (assume permission granted)
        return True
    
    async def _get_recent_actor_executions(
        self, actor_id: str, tenant_id: int, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent executions by actor"""
        
        if not self.db_pool:
            return []
        
        try:
            query = """
                SELECT execution_id, workflow_id, created_at
                FROM workflow_executions
                WHERE actor_id = $1 AND tenant_id = $2
                AND created_at >= NOW() - INTERVAL '%s hours'
                ORDER BY created_at DESC
            """ % hours
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, actor_id, tenant_id)
                return [dict(row) for row in rows]
        except:
            return []
    
    async def _is_self_approval(self, context: RuntimeSoDContext) -> bool:
        """Check if this is a self-approval scenario"""
        
        # Check if current actor created the workflow
        for actor_info in context.previous_actors:
            if (actor_info.get('actor_id') == context.current_actor_id and 
                actor_info.get('action') in ['create', 'submit', 'request']):
                return True
        
        return False
    
    async def _check_same_role_conflicts(self, context: RuntimeSoDContext) -> List[Dict[str, Any]]:
        """Check for same-role conflicts"""
        
        conflicts = []
        
        for actor_info in context.previous_actors:
            if (actor_info.get('role') == context.current_actor_role and
                actor_info.get('actor_id') != context.current_actor_id):
                conflicts.append({
                    'actor_id': actor_info.get('actor_id'),
                    'role': actor_info.get('role'),
                    'action': actor_info.get('action')
                })
        
        return conflicts
    
    async def _check_department_conflict(self, context: RuntimeSoDContext) -> bool:
        """Check for department conflicts"""
        
        if not context.current_actor_department:
            return False
        
        same_dept_count = 0
        for actor_info in context.previous_actors:
            if actor_info.get('department') == context.current_actor_department:
                same_dept_count += 1
        
        # Flag if more than 2 people from same department
        return same_dept_count >= 2
    
    async def _validate_approval_chain(self, context: RuntimeSoDContext) -> bool:
        """Validate the complete approval chain"""
        
        # Check if approval chain has sufficient separation
        approvers = [actor for actor in context.approval_chain if actor.get('action') == 'approve']
        
        if len(approvers) < 2:
            return False  # Need at least 2 approvers
        
        # Check for role diversity
        approver_roles = set(approver.get('role') for approver in approvers)
        if len(approver_roles) < 2:
            return False  # Need different roles
        
        return True
    
    async def _detect_execution_patterns(self, context: RuntimeSoDContext) -> bool:
        """Detect unusual execution patterns"""
        
        # Simple pattern detection - in production would use ML
        execution_hour = context.execution_timestamp.hour
        
        # Flag executions outside business hours as potentially suspicious
        if execution_hour < 6 or execution_hour > 22:
            return True
        
        return False
    
    async def _check_approval_authority(self, context: RuntimeSoDContext) -> bool:
        """Check if actor has approval authority"""
        
        # Check role-based approval authority
        approval_roles = ['manager', 'director', 'vp', 'cro', 'cfo']
        return context.current_actor_role.lower() in approval_roles
    
    async def _check_approval_conflicts(self, context: RuntimeSoDContext) -> List[Dict[str, Any]]:
        """Check for approval conflicts"""
        
        conflicts = []
        
        # Check for conflicting interests
        for actor_info in context.previous_actors:
            if actor_info.get('actor_id') == context.current_actor_id:
                conflicts.append({
                    'type': 'self_involvement',
                    'severity': 'critical',
                    'message': f"Actor {context.current_actor_id} was previously involved in this workflow",
                    'conflicting_actor': actor_info.get('actor_id')
                })
        
        return conflicts
    
    def _update_enforcement_stats(self, result: RuntimeSoDValidationResult):
        """Update enforcement statistics"""
        
        self.enforcement_stats['total_checks'] += 1
        
        if result.is_allowed:
            self.enforcement_stats['checks_passed'] += 1
        else:
            self.enforcement_stats['checks_failed'] += 1
            if result.recommended_action == RuntimeSoDAction.BLOCK:
                self.enforcement_stats['executions_blocked'] += 1
        
        if result.escalation_required:
            self.enforcement_stats['escalations_triggered'] += 1
        
        # Update average check time
        current_avg = self.enforcement_stats['average_check_time_ms']
        total_checks = self.enforcement_stats['total_checks']
        self.enforcement_stats['average_check_time_ms'] = (
            (current_avg * (total_checks - 1) + result.validation_duration_ms) / total_checks
        )
    
    async def _create_runtime_sod_tables(self):
        """Create runtime SoD validation tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Runtime SoD validation results
        CREATE TABLE IF NOT EXISTS runtime_sod_validations (
            validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            execution_id VARCHAR(100) NOT NULL,
            workflow_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Validation results
            is_allowed BOOLEAN NOT NULL,
            violations_count INTEGER NOT NULL DEFAULT 0,
            warnings_count INTEGER NOT NULL DEFAULT 0,
            checks_performed INTEGER NOT NULL DEFAULT 0,
            
            -- Action taken
            recommended_action VARCHAR(20) NOT NULL,
            escalation_required BOOLEAN NOT NULL DEFAULT FALSE,
            
            -- Performance
            validation_duration_ms FLOAT NOT NULL DEFAULT 0,
            
            -- Context
            current_actor_id VARCHAR(100) NOT NULL,
            current_actor_role VARCHAR(100),
            check_type VARCHAR(20) NOT NULL,
            
            -- Timestamps
            validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_runtime_sod_action CHECK (recommended_action IN ('block', 'warn', 'escalate', 'audit_only')),
            CONSTRAINT chk_runtime_sod_check_type CHECK (check_type IN ('pre_execution', 'step_execution', 'post_execution', 'approval_check'))
        );
        
        -- Runtime SoD violations
        CREATE TABLE IF NOT EXISTS runtime_sod_violations (
            violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            validation_id UUID REFERENCES runtime_sod_validations(validation_id) ON DELETE CASCADE,
            
            -- Violation details
            execution_id VARCHAR(100) NOT NULL,
            check_type VARCHAR(20) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            violation_message TEXT NOT NULL,
            
            -- Context
            workflow_id VARCHAR(100) NOT NULL,
            step_id VARCHAR(100),
            current_actor_id VARCHAR(100) NOT NULL,
            conflicting_actor_id VARCHAR(100),
            
            -- Action
            action_taken VARCHAR(20) NOT NULL,
            escalation_required BOOLEAN NOT NULL DEFAULT FALSE,
            
            -- Timestamps
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_runtime_sod_violation_severity CHECK (severity IN ('critical', 'high', 'medium', 'low')),
            CONSTRAINT chk_runtime_sod_violation_action CHECK (action_taken IN ('block', 'warn', 'escalate', 'audit_only'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_validations_execution ON runtime_sod_validations(execution_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_validations_workflow ON runtime_sod_validations(workflow_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_validations_actor ON runtime_sod_validations(current_actor_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_validations_allowed ON runtime_sod_validations(is_allowed, validated_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_violations_validation ON runtime_sod_violations(validation_id);
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_violations_execution ON runtime_sod_violations(execution_id, detected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_violations_type ON runtime_sod_violations(violation_type, severity);
        CREATE INDEX IF NOT EXISTS idx_runtime_sod_violations_actor ON runtime_sod_violations(current_actor_id, detected_at DESC);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Runtime SoD validation tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create runtime SoD tables: {e}")
            raise
    
    async def _store_validation_result(
        self,
        result: RuntimeSoDValidationResult,
        context: RuntimeSoDContext
    ):
        """Store validation result in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert validation result
                validation_query = """
                    INSERT INTO runtime_sod_validations (
                        execution_id, workflow_id, tenant_id, is_allowed,
                        violations_count, warnings_count, checks_performed,
                        recommended_action, escalation_required, validation_duration_ms,
                        current_actor_id, current_actor_role, check_type
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING validation_id
                """
                
                validation_id = await conn.fetchval(
                    validation_query,
                    result.execution_id,
                    context.workflow_id,
                    context.tenant_id,
                    result.is_allowed,
                    len(result.violations),
                    len(result.warnings),
                    result.checks_performed,
                    result.recommended_action.value,
                    result.escalation_required,
                    result.validation_duration_ms,
                    context.current_actor_id,
                    context.current_actor_role,
                    "step_execution"  # Default check type
                )
                
                # Insert violations
                if result.violations:
                    violation_query = """
                        INSERT INTO runtime_sod_violations (
                            validation_id, execution_id, check_type, violation_type,
                            severity, violation_message, workflow_id, step_id,
                            current_actor_id, conflicting_actor_id, action_taken,
                            escalation_required, detected_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """
                    
                    for violation in result.violations:
                        await conn.execute(
                            violation_query,
                            validation_id,
                            violation.execution_id,
                            violation.check_type.value,
                            violation.violation_type,
                            violation.severity,
                            violation.violation_message,
                            violation.workflow_id,
                            violation.step_id,
                            violation.current_actor_id,
                            violation.conflicting_actor_id,
                            violation.action_taken.value,
                            violation.escalation_required,
                            violation.detected_at
                        )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store runtime SoD validation result: {e}")
    
    def get_enforcement_statistics(self) -> Dict[str, Any]:
        """Get enforcement statistics"""
        
        success_rate = 0.0
        if self.enforcement_stats['total_checks'] > 0:
            success_rate = (
                self.enforcement_stats['checks_passed'] / 
                self.enforcement_stats['total_checks']
            ) * 100
        
        block_rate = 0.0
        if self.enforcement_stats['total_checks'] > 0:
            block_rate = (
                self.enforcement_stats['executions_blocked'] / 
                self.enforcement_stats['total_checks']
            ) * 100
        
        return {
            'total_checks': self.enforcement_stats['total_checks'],
            'checks_passed': self.enforcement_stats['checks_passed'],
            'checks_failed': self.enforcement_stats['checks_failed'],
            'executions_blocked': self.enforcement_stats['executions_blocked'],
            'escalations_triggered': self.enforcement_stats['escalations_triggered'],
            'success_rate_percentage': round(success_rate, 2),
            'block_rate_percentage': round(block_rate, 2),
            'average_check_time_ms': round(self.enforcement_stats['average_check_time_ms'], 2),
            'performance_threshold_ms': self.config['performance_threshold_ms'],
            'strict_enforcement_enabled': self.config['strict_enforcement'],
            'fail_closed_enabled': self.config['fail_closed_on_error']
        }


# Global runtime SoD enforcer instance
runtime_sod_enforcer = RuntimeSoDEnforcer()
