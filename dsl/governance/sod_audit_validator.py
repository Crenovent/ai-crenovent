"""
Task 8.2-T17: Implement SoD validation at audit write
Governance control with Policy engine - Reject invalid SoD
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class SoDViolationType(Enum):
    """Types of SoD violations"""
    SELF_APPROVAL = "self_approval"
    SAME_ROLE_CONFLICT = "same_role_conflict"
    DEPARTMENT_CONFLICT = "department_conflict"
    REPORTING_LINE_CONFLICT = "reporting_line_conflict"
    TEMPORAL_CONFLICT = "temporal_conflict"
    CUMULATIVE_AUTHORITY = "cumulative_authority"


class ValidationSeverity(Enum):
    """SoD validation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SoDViolation:
    """SoD violation record"""
    violation_id: str
    violation_type: SoDViolationType
    severity: ValidationSeverity
    
    # Actors involved
    primary_actor_id: str
    secondary_actor_id: str
    primary_role: str
    secondary_role: str
    
    # Context
    workflow_id: str
    step_id: str
    action_type: str
    resource_id: str
    
    # Violation details
    violation_message: str
    rule_violated: str
    compliance_framework: str
    
    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "primary_actor_id": self.primary_actor_id,
            "secondary_actor_id": self.secondary_actor_id,
            "primary_role": self.primary_role,
            "secondary_role": self.secondary_role,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "action_type": self.action_type,
            "resource_id": self.resource_id,
            "violation_message": self.violation_message,
            "rule_violated": self.rule_violated,
            "compliance_framework": self.compliance_framework,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class SoDValidationResult:
    """Result of SoD validation"""
    is_valid: bool
    violations: List[SoDViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Context
    audit_event_id: str = ""
    tenant_id: int = 0
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "audit_event_id": self.audit_event_id,
            "tenant_id": self.tenant_id,
            "validation_timestamp": self.validation_timestamp.isoformat()
        }


class SoDValidatorAtAuditWrite:
    """
    SoD Validator at Audit Write - Task 8.2-T17
    
    Validates segregation of duties before writing audit events
    Rejects invalid SoD combinations with policy engine integration
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'strict_validation': True,
            'fail_closed_on_error': True,
            'enable_temporal_validation': True,
            'temporal_window_hours': 24,
            'enable_cumulative_validation': True,
            'cache_validation_results': True,
            'cache_ttl_minutes': 15
        }
        
        # SoD rules cache
        self.sod_rules_cache: Dict[str, Dict[str, Any]] = {}
        self.validation_cache: Dict[str, SoDValidationResult] = {}
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'violations_detected': 0,
            'average_validation_time_ms': 0.0
        }
        
        # Initialize default SoD rules
        self._initialize_default_sod_rules()
    
    async def initialize(self) -> bool:
        """Initialize SoD validator"""
        try:
            await self._create_sod_validation_tables()
            await self._load_sod_rules()
            self.logger.info("✅ SoD audit validator initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SoD validator: {e}")
            return False
    
    async def validate_sod_at_audit_write(
        self,
        audit_event: Dict[str, Any],
        tenant_id: int
    ) -> SoDValidationResult:
        """
        Validate SoD compliance before writing audit event
        
        This is the main validation entry point called before audit write
        """
        
        start_time = datetime.now(timezone.utc)
        
        # Initialize result
        result = SoDValidationResult(
            is_valid=True,
            audit_event_id=audit_event.get('event_id', ''),
            tenant_id=tenant_id
        )
        
        try:
            # Extract actors and roles from audit event
            primary_actor = audit_event.get('actor_id', '')
            action_type = audit_event.get('action', '')
            workflow_id = audit_event.get('workflow_id', '')
            resource_id = audit_event.get('resource_id', '')
            
            # Get actor role information
            primary_role = await self._get_actor_role(primary_actor, tenant_id)
            
            # Check for self-approval violations
            self_approval_violations = await self._check_self_approval(
                audit_event, primary_actor, primary_role, tenant_id
            )
            result.violations.extend(self_approval_violations)
            
            # Check for same-role conflicts
            role_conflict_violations = await self._check_same_role_conflicts(
                audit_event, primary_actor, primary_role, tenant_id
            )
            result.violations.extend(role_conflict_violations)
            
            # Check for department conflicts
            dept_conflict_violations = await self._check_department_conflicts(
                audit_event, primary_actor, primary_role, tenant_id
            )
            result.violations.extend(dept_conflict_violations)
            
            # Check for reporting line conflicts
            reporting_violations = await self._check_reporting_line_conflicts(
                audit_event, primary_actor, primary_role, tenant_id
            )
            result.violations.extend(reporting_violations)
            
            # Check temporal conflicts (if enabled)
            if self.config['enable_temporal_validation']:
                temporal_violations = await self._check_temporal_conflicts(
                    audit_event, primary_actor, primary_role, tenant_id
                )
                result.violations.extend(temporal_violations)
            
            # Check cumulative authority violations (if enabled)
            if self.config['enable_cumulative_validation']:
                cumulative_violations = await self._check_cumulative_authority(
                    audit_event, primary_actor, primary_role, tenant_id
                )
                result.violations.extend(cumulative_violations)
            
            # Determine overall validation result
            critical_violations = [v for v in result.violations if v.severity == ValidationSeverity.CRITICAL]
            high_violations = [v for v in result.violations if v.severity == ValidationSeverity.HIGH]
            
            if self.config['strict_validation']:
                # Strict mode: fail on critical or high severity violations
                if critical_violations or high_violations:
                    result.is_valid = False
                    self.logger.warning(f"🚫 SoD validation failed: {len(critical_violations + high_violations)} violations")
            else:
                # Lenient mode: only fail on critical violations
                if critical_violations:
                    result.is_valid = False
                    self.logger.warning(f"🚫 SoD validation failed: {len(critical_violations)} critical violations")
                elif high_violations:
                    result.warnings.append(f"High severity SoD violations detected: {len(high_violations)}")
            
            # Update statistics
            self.validation_stats['total_validations'] += 1
            if result.is_valid:
                self.validation_stats['validations_passed'] += 1
            else:
                self.validation_stats['validations_failed'] += 1
            
            self.validation_stats['violations_detected'] += len(result.violations)
            
            # Calculate validation time
            end_time = datetime.now(timezone.utc)
            validation_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update average validation time
            current_avg = self.validation_stats['average_validation_time_ms']
            total_validations = self.validation_stats['total_validations']
            self.validation_stats['average_validation_time_ms'] = (
                (current_avg * (total_validations - 1) + validation_time_ms) / total_validations
            )
            
            # Store validation result
            if self.db_pool:
                await self._store_validation_result(result)
            
            # Cache result if enabled
            if self.config['cache_validation_results']:
                cache_key = f"{audit_event.get('event_id')}:{tenant_id}"
                self.validation_cache[cache_key] = result
            
            if result.is_valid:
                self.logger.debug(f"✅ SoD validation passed for audit event {result.audit_event_id}")
            else:
                self.logger.warning(f"❌ SoD validation failed for audit event {result.audit_event_id}")
                for violation in result.violations:
                    self.logger.warning(f"🚫 SoD Violation: {violation.violation_message}")
            
            return result
            
        except Exception as e:
            # Fail closed on validation errors
            if self.config['fail_closed_on_error']:
                self.logger.error(f"❌ SoD validation error (fail-closed): {e}")
                result.is_valid = False
                result.violations.append(SoDViolation(
                    violation_id=f"validation_error_{audit_event.get('event_id')}",
                    violation_type=SoDViolationType.SAME_ROLE_CONFLICT,
                    severity=ValidationSeverity.CRITICAL,
                    primary_actor_id=audit_event.get('actor_id', ''),
                    secondary_actor_id="system",
                    primary_role="unknown",
                    secondary_role="system",
                    workflow_id=audit_event.get('workflow_id', ''),
                    step_id=audit_event.get('step_id', ''),
                    action_type=audit_event.get('action', ''),
                    resource_id=audit_event.get('resource_id', ''),
                    violation_message=f"SoD validation system error: {str(e)}",
                    rule_violated="system_error",
                    compliance_framework="SOX"
                ))
            else:
                self.logger.error(f"❌ SoD validation error (fail-open): {e}")
                result.is_valid = True
                result.warnings.append(f"SoD validation system error: {str(e)}")
            
            return result
    
    def _initialize_default_sod_rules(self):
        """Initialize default SoD rules"""
        
        self.sod_rules_cache = {
            # Self-approval rules
            'no_self_approval': {
                'rule_type': 'self_approval',
                'description': 'Users cannot approve their own requests',
                'severity': 'critical',
                'compliance_frameworks': ['SOX', 'RBI', 'HIPAA'],
                'applicable_actions': ['approve', 'authorize', 'sign_off']
            },
            
            # Same role conflicts
            'no_same_role_approval': {
                'rule_type': 'same_role_conflict',
                'description': 'Users with same role cannot approve each other',
                'severity': 'high',
                'compliance_frameworks': ['SOX', 'RBI'],
                'applicable_actions': ['approve', 'authorize'],
                'conflicting_roles': ['manager', 'supervisor', 'approver']
            },
            
            # Department conflicts
            'cross_department_approval': {
                'rule_type': 'department_conflict',
                'description': 'Approvals must come from different departments',
                'severity': 'medium',
                'compliance_frameworks': ['SOX'],
                'applicable_actions': ['approve', 'authorize'],
                'require_different_departments': True
            },
            
            # Reporting line conflicts
            'no_direct_report_approval': {
                'rule_type': 'reporting_line_conflict',
                'description': 'Direct reports cannot approve their manager requests',
                'severity': 'high',
                'compliance_frameworks': ['SOX', 'RBI'],
                'applicable_actions': ['approve', 'authorize']
            },
            
            # Temporal conflicts
            'temporal_separation': {
                'rule_type': 'temporal_conflict',
                'description': 'Same user cannot perform conflicting actions within time window',
                'severity': 'medium',
                'compliance_frameworks': ['SOX'],
                'time_window_hours': 24,
                'conflicting_action_pairs': [
                    ['create', 'approve'],
                    ['modify', 'authorize'],
                    ['request', 'approve']
                ]
            },
            
            # Cumulative authority
            'cumulative_authority_limit': {
                'rule_type': 'cumulative_authority',
                'description': 'Users cannot exceed cumulative authority limits',
                'severity': 'high',
                'compliance_frameworks': ['SOX', 'RBI'],
                'authority_limits': {
                    'financial_amount': 10000,
                    'approval_count_per_day': 50,
                    'high_risk_approvals_per_week': 10
                }
            }
        }
    
    async def _check_self_approval(
        self,
        audit_event: Dict[str, Any],
        actor_id: str,
        actor_role: str,
        tenant_id: int
    ) -> List[SoDViolation]:
        """Check for self-approval violations"""
        
        violations = []
        action_type = audit_event.get('action', '')
        
        # Check if this is an approval action
        if action_type in ['approve', 'authorize', 'sign_off']:
            # Check if the actor is approving their own request
            workflow_id = audit_event.get('workflow_id', '')
            
            # Get workflow creator
            workflow_creator = await self._get_workflow_creator(workflow_id, tenant_id)
            
            if workflow_creator == actor_id:
                violation = SoDViolation(
                    violation_id=f"self_approval_{audit_event.get('event_id')}",
                    violation_type=SoDViolationType.SELF_APPROVAL,
                    severity=ValidationSeverity.CRITICAL,
                    primary_actor_id=actor_id,
                    secondary_actor_id=actor_id,
                    primary_role=actor_role,
                    secondary_role=actor_role,
                    workflow_id=workflow_id,
                    step_id=audit_event.get('step_id', ''),
                    action_type=action_type,
                    resource_id=audit_event.get('resource_id', ''),
                    violation_message=f"Self-approval detected: {actor_id} cannot approve their own request",
                    rule_violated="no_self_approval",
                    compliance_framework="SOX"
                )
                violations.append(violation)
        
        return violations
    
    async def _check_same_role_conflicts(
        self,
        audit_event: Dict[str, Any],
        actor_id: str,
        actor_role: str,
        tenant_id: int
    ) -> List[SoDViolation]:
        """Check for same-role conflicts"""
        
        violations = []
        action_type = audit_event.get('action', '')
        
        # Check if this is a conflicting action
        if action_type in ['approve', 'authorize']:
            workflow_id = audit_event.get('workflow_id', '')
            
            # Get previous actors in this workflow
            previous_actors = await self._get_workflow_actors(workflow_id, tenant_id)
            
            for prev_actor_id, prev_role in previous_actors:
                if prev_actor_id != actor_id and prev_role == actor_role:
                    # Same role conflict detected
                    if actor_role in ['manager', 'supervisor', 'approver']:
                        violation = SoDViolation(
                            violation_id=f"same_role_{audit_event.get('event_id')}_{prev_actor_id}",
                            violation_type=SoDViolationType.SAME_ROLE_CONFLICT,
                            severity=ValidationSeverity.HIGH,
                            primary_actor_id=actor_id,
                            secondary_actor_id=prev_actor_id,
                            primary_role=actor_role,
                            secondary_role=prev_role,
                            workflow_id=workflow_id,
                            step_id=audit_event.get('step_id', ''),
                            action_type=action_type,
                            resource_id=audit_event.get('resource_id', ''),
                            violation_message=f"Same role conflict: {actor_role} {actor_id} and {prev_role} {prev_actor_id}",
                            rule_violated="no_same_role_approval",
                            compliance_framework="SOX"
                        )
                        violations.append(violation)
        
        return violations
    
    async def _check_department_conflicts(
        self,
        audit_event: Dict[str, Any],
        actor_id: str,
        actor_role: str,
        tenant_id: int
    ) -> List[SoDViolation]:
        """Check for department conflicts"""
        
        violations = []
        action_type = audit_event.get('action', '')
        
        if action_type in ['approve', 'authorize']:
            # Get actor's department
            actor_dept = await self._get_actor_department(actor_id, tenant_id)
            
            # Get workflow previous approvers' departments
            workflow_id = audit_event.get('workflow_id', '')
            previous_approvers = await self._get_workflow_approvers(workflow_id, tenant_id)
            
            same_dept_approvers = [
                approver for approver in previous_approvers
                if approver['department'] == actor_dept and approver['actor_id'] != actor_id
            ]
            
            if same_dept_approvers:
                violation = SoDViolation(
                    violation_id=f"dept_conflict_{audit_event.get('event_id')}",
                    violation_type=SoDViolationType.DEPARTMENT_CONFLICT,
                    severity=ValidationSeverity.MEDIUM,
                    primary_actor_id=actor_id,
                    secondary_actor_id=same_dept_approvers[0]['actor_id'],
                    primary_role=actor_role,
                    secondary_role=same_dept_approvers[0]['role'],
                    workflow_id=workflow_id,
                    step_id=audit_event.get('step_id', ''),
                    action_type=action_type,
                    resource_id=audit_event.get('resource_id', ''),
                    violation_message=f"Department conflict: Multiple approvers from {actor_dept}",
                    rule_violated="cross_department_approval",
                    compliance_framework="SOX"
                )
                violations.append(violation)
        
        return violations
    
    async def _check_reporting_line_conflicts(
        self,
        audit_event: Dict[str, Any],
        actor_id: str,
        actor_role: str,
        tenant_id: int
    ) -> List[SoDViolation]:
        """Check for reporting line conflicts"""
        
        violations = []
        action_type = audit_event.get('action', '')
        
        if action_type in ['approve', 'authorize']:
            workflow_id = audit_event.get('workflow_id', '')
            
            # Get workflow creator
            workflow_creator = await self._get_workflow_creator(workflow_id, tenant_id)
            
            # Check if actor reports to workflow creator or vice versa
            reporting_relationship = await self._get_reporting_relationship(
                actor_id, workflow_creator, tenant_id
            )
            
            if reporting_relationship in ['direct_report', 'manager']:
                violation = SoDViolation(
                    violation_id=f"reporting_conflict_{audit_event.get('event_id')}",
                    violation_type=SoDViolationType.REPORTING_LINE_CONFLICT,
                    severity=ValidationSeverity.HIGH,
                    primary_actor_id=actor_id,
                    secondary_actor_id=workflow_creator,
                    primary_role=actor_role,
                    secondary_role="workflow_creator",
                    workflow_id=workflow_id,
                    step_id=audit_event.get('step_id', ''),
                    action_type=action_type,
                    resource_id=audit_event.get('resource_id', ''),
                    violation_message=f"Reporting line conflict: {reporting_relationship} relationship detected",
                    rule_violated="no_direct_report_approval",
                    compliance_framework="SOX"
                )
                violations.append(violation)
        
        return violations
    
    async def _check_temporal_conflicts(
        self,
        audit_event: Dict[str, Any],
        actor_id: str,
        actor_role: str,
        tenant_id: int
    ) -> List[SoDViolation]:
        """Check for temporal conflicts"""
        
        violations = []
        action_type = audit_event.get('action', '')
        workflow_id = audit_event.get('workflow_id', '')
        
        # Get recent actions by this actor
        time_window = self.config['temporal_window_hours']
        recent_actions = await self._get_recent_actor_actions(
            actor_id, workflow_id, time_window, tenant_id
        )
        
        # Check for conflicting action pairs
        conflicting_pairs = [
            ['create', 'approve'],
            ['modify', 'authorize'],
            ['request', 'approve']
        ]
        
        for pair in conflicting_pairs:
            if action_type == pair[1]:  # Current action is second in pair
                # Check if actor performed first action recently
                if any(action['action_type'] == pair[0] for action in recent_actions):
                    violation = SoDViolation(
                        violation_id=f"temporal_conflict_{audit_event.get('event_id')}",
                        violation_type=SoDViolationType.TEMPORAL_CONFLICT,
                        severity=ValidationSeverity.MEDIUM,
                        primary_actor_id=actor_id,
                        secondary_actor_id=actor_id,
                        primary_role=actor_role,
                        secondary_role=actor_role,
                        workflow_id=workflow_id,
                        step_id=audit_event.get('step_id', ''),
                        action_type=action_type,
                        resource_id=audit_event.get('resource_id', ''),
                        violation_message=f"Temporal conflict: {actor_id} performed {pair[0]} and {pair[1]} within {time_window}h",
                        rule_violated="temporal_separation",
                        compliance_framework="SOX"
                    )
                    violations.append(violation)
        
        return violations
    
    async def _check_cumulative_authority(
        self,
        audit_event: Dict[str, Any],
        actor_id: str,
        actor_role: str,
        tenant_id: int
    ) -> List[SoDViolation]:
        """Check for cumulative authority violations"""
        
        violations = []
        action_type = audit_event.get('action', '')
        
        if action_type in ['approve', 'authorize']:
            # Get actor's recent approvals
            daily_approvals = await self._get_daily_approval_count(actor_id, tenant_id)
            weekly_high_risk_approvals = await self._get_weekly_high_risk_approvals(actor_id, tenant_id)
            
            # Check limits
            authority_limits = self.sod_rules_cache['cumulative_authority_limit']['authority_limits']
            
            if daily_approvals >= authority_limits['approval_count_per_day']:
                violation = SoDViolation(
                    violation_id=f"cumulative_daily_{audit_event.get('event_id')}",
                    violation_type=SoDViolationType.CUMULATIVE_AUTHORITY,
                    severity=ValidationSeverity.HIGH,
                    primary_actor_id=actor_id,
                    secondary_actor_id=actor_id,
                    primary_role=actor_role,
                    secondary_role=actor_role,
                    workflow_id=audit_event.get('workflow_id', ''),
                    step_id=audit_event.get('step_id', ''),
                    action_type=action_type,
                    resource_id=audit_event.get('resource_id', ''),
                    violation_message=f"Daily approval limit exceeded: {daily_approvals}/{authority_limits['approval_count_per_day']}",
                    rule_violated="cumulative_authority_limit",
                    compliance_framework="SOX"
                )
                violations.append(violation)
            
            if weekly_high_risk_approvals >= authority_limits['high_risk_approvals_per_week']:
                violation = SoDViolation(
                    violation_id=f"cumulative_weekly_{audit_event.get('event_id')}",
                    violation_type=SoDViolationType.CUMULATIVE_AUTHORITY,
                    severity=ValidationSeverity.HIGH,
                    primary_actor_id=actor_id,
                    secondary_actor_id=actor_id,
                    primary_role=actor_role,
                    secondary_role=actor_role,
                    workflow_id=audit_event.get('workflow_id', ''),
                    step_id=audit_event.get('step_id', ''),
                    action_type=action_type,
                    resource_id=audit_event.get('resource_id', ''),
                    violation_message=f"Weekly high-risk approval limit exceeded: {weekly_high_risk_approvals}/{authority_limits['high_risk_approvals_per_week']}",
                    rule_violated="cumulative_authority_limit",
                    compliance_framework="SOX"
                )
                violations.append(violation)
        
        return violations
    
    # Helper methods for database queries
    async def _get_actor_role(self, actor_id: str, tenant_id: int) -> str:
        """Get actor's role"""
        if not self.db_pool:
            return "unknown"
        
        try:
            query = "SELECT role FROM users WHERE user_id = $1 AND tenant_id = $2"
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, actor_id, tenant_id)
                return result or "unknown"
        except:
            return "unknown"
    
    async def _get_workflow_creator(self, workflow_id: str, tenant_id: int) -> str:
        """Get workflow creator"""
        if not self.db_pool:
            return "unknown"
        
        try:
            query = "SELECT created_by FROM workflows WHERE workflow_id = $1 AND tenant_id = $2"
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, workflow_id, tenant_id)
                return result or "unknown"
        except:
            return "unknown"
    
    async def _get_workflow_actors(self, workflow_id: str, tenant_id: int) -> List[Tuple[str, str]]:
        """Get all actors who have acted on this workflow"""
        if not self.db_pool:
            return []
        
        try:
            query = """
                SELECT DISTINCT actor_id, actor_role 
                FROM audit_events 
                WHERE workflow_id = $1 AND tenant_id = $2
            """
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, workflow_id, tenant_id)
                return [(row['actor_id'], row['actor_role']) for row in rows]
        except:
            return []
    
    async def _get_actor_department(self, actor_id: str, tenant_id: int) -> str:
        """Get actor's department"""
        if not self.db_pool:
            return "unknown"
        
        try:
            query = "SELECT department FROM users WHERE user_id = $1 AND tenant_id = $2"
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, actor_id, tenant_id)
                return result or "unknown"
        except:
            return "unknown"
    
    async def _get_workflow_approvers(self, workflow_id: str, tenant_id: int) -> List[Dict[str, Any]]:
        """Get workflow approvers with their departments"""
        if not self.db_pool:
            return []
        
        try:
            query = """
                SELECT DISTINCT ae.actor_id, u.role, u.department
                FROM audit_events ae
                JOIN users u ON ae.actor_id = u.user_id
                WHERE ae.workflow_id = $1 AND ae.tenant_id = $2 
                AND ae.action IN ('approve', 'authorize')
            """
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, workflow_id, tenant_id)
                return [dict(row) for row in rows]
        except:
            return []
    
    async def _get_reporting_relationship(self, actor1_id: str, actor2_id: str, tenant_id: int) -> str:
        """Get reporting relationship between two actors"""
        if not self.db_pool:
            return "none"
        
        try:
            query = """
                SELECT 
                    CASE 
                        WHEN u1.manager_id = u2.user_id THEN 'direct_report'
                        WHEN u2.manager_id = u1.user_id THEN 'manager'
                        ELSE 'none'
                    END as relationship
                FROM users u1, users u2
                WHERE u1.user_id = $1 AND u2.user_id = $2 AND u1.tenant_id = $3
            """
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, actor1_id, actor2_id, tenant_id)
                return result or "none"
        except:
            return "none"
    
    async def _get_recent_actor_actions(
        self, actor_id: str, workflow_id: str, hours: int, tenant_id: int
    ) -> List[Dict[str, Any]]:
        """Get recent actions by actor"""
        if not self.db_pool:
            return []
        
        try:
            query = """
                SELECT action_type, created_at
                FROM audit_events
                WHERE actor_id = $1 AND workflow_id = $2 AND tenant_id = $3
                AND created_at >= NOW() - INTERVAL '%s hours'
                ORDER BY created_at DESC
            """ % hours
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, actor_id, workflow_id, tenant_id)
                return [dict(row) for row in rows]
        except:
            return []
    
    async def _get_daily_approval_count(self, actor_id: str, tenant_id: int) -> int:
        """Get daily approval count for actor"""
        if not self.db_pool:
            return 0
        
        try:
            query = """
                SELECT COUNT(*) 
                FROM audit_events
                WHERE actor_id = $1 AND tenant_id = $2
                AND action IN ('approve', 'authorize')
                AND created_at >= CURRENT_DATE
            """
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, actor_id, tenant_id)
                return result or 0
        except:
            return 0
    
    async def _get_weekly_high_risk_approvals(self, actor_id: str, tenant_id: int) -> int:
        """Get weekly high-risk approval count for actor"""
        if not self.db_pool:
            return 0
        
        try:
            query = """
                SELECT COUNT(*) 
                FROM audit_events
                WHERE actor_id = $1 AND tenant_id = $2
                AND action IN ('approve', 'authorize')
                AND risk_level = 'high'
                AND created_at >= CURRENT_DATE - INTERVAL '7 days'
            """
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, actor_id, tenant_id)
                return result or 0
        except:
            return 0
    
    async def _create_sod_validation_tables(self):
        """Create SoD validation tables"""
        if not self.db_pool:
            return
        
        schema_sql = """
        -- SoD validation results
        CREATE TABLE IF NOT EXISTS sod_validation_results (
            validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            audit_event_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Validation results
            is_valid BOOLEAN NOT NULL,
            violations_count INTEGER NOT NULL DEFAULT 0,
            warnings_count INTEGER NOT NULL DEFAULT 0,
            
            -- Validation details
            validation_details JSONB DEFAULT '{}',
            
            -- Timestamps
            validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_validation_counts CHECK (violations_count >= 0 AND warnings_count >= 0)
        );
        
        -- SoD violations
        CREATE TABLE IF NOT EXISTS sod_violations (
            violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            validation_id UUID REFERENCES sod_validation_results(validation_id) ON DELETE CASCADE,
            
            -- Violation details
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            violation_message TEXT NOT NULL,
            rule_violated VARCHAR(100) NOT NULL,
            compliance_framework VARCHAR(50) NOT NULL,
            
            -- Actors
            primary_actor_id VARCHAR(100) NOT NULL,
            secondary_actor_id VARCHAR(100),
            primary_role VARCHAR(100),
            secondary_role VARCHAR(100),
            
            -- Context
            workflow_id VARCHAR(100),
            step_id VARCHAR(100),
            action_type VARCHAR(50),
            resource_id VARCHAR(100),
            
            -- Timestamps
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_violation_severity CHECK (severity IN ('critical', 'high', 'medium', 'low'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_sod_validation_tenant_time ON sod_validation_results(tenant_id, validated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_sod_validation_event ON sod_validation_results(audit_event_id);
        CREATE INDEX IF NOT EXISTS idx_sod_validation_valid ON sod_validation_results(is_valid, validated_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_sod_violations_validation ON sod_violations(validation_id);
        CREATE INDEX IF NOT EXISTS idx_sod_violations_type ON sod_violations(violation_type, severity);
        CREATE INDEX IF NOT EXISTS idx_sod_violations_actor ON sod_violations(primary_actor_id, detected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_sod_violations_workflow ON sod_violations(workflow_id) WHERE workflow_id IS NOT NULL;
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ SoD validation tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create SoD validation tables: {e}")
            raise
    
    async def _load_sod_rules(self):
        """Load SoD rules from database"""
        # In production, this would load custom SoD rules from database
        # For now, we use the default rules initialized in memory
        pass
    
    async def _store_validation_result(self, result: SoDValidationResult):
        """Store validation result in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert validation result
                validation_query = """
                    INSERT INTO sod_validation_results (
                        audit_event_id, tenant_id, is_valid, violations_count,
                        warnings_count, validation_details
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING validation_id
                """
                
                validation_id = await conn.fetchval(
                    validation_query,
                    result.audit_event_id,
                    result.tenant_id,
                    result.is_valid,
                    len(result.violations),
                    len(result.warnings),
                    json.dumps(result.to_dict())
                )
                
                # Insert violations
                if result.violations:
                    violation_query = """
                        INSERT INTO sod_violations (
                            validation_id, violation_type, severity, violation_message,
                            rule_violated, compliance_framework, primary_actor_id,
                            secondary_actor_id, primary_role, secondary_role,
                            workflow_id, step_id, action_type, resource_id, detected_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """
                    
                    for violation in result.violations:
                        await conn.execute(
                            violation_query,
                            validation_id,
                            violation.violation_type.value,
                            violation.severity.value,
                            violation.violation_message,
                            violation.rule_violated,
                            violation.compliance_framework,
                            violation.primary_actor_id,
                            violation.secondary_actor_id,
                            violation.primary_role,
                            violation.secondary_role,
                            violation.workflow_id,
                            violation.step_id,
                            violation.action_type,
                            violation.resource_id,
                            violation.detected_at
                        )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store SoD validation result: {e}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get SoD validation statistics"""
        
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
            'violations_detected': self.validation_stats['violations_detected'],
            'success_rate_percentage': round(success_rate, 2),
            'average_validation_time_ms': round(self.validation_stats['average_validation_time_ms'], 2),
            'sod_rules_count': len(self.sod_rules_cache),
            'cache_size': len(self.validation_cache),
            'strict_validation_enabled': self.config['strict_validation'],
            'fail_closed_enabled': self.config['fail_closed_on_error']
        }


# Global SoD validator instance
sod_audit_validator = SoDValidatorAtAuditWrite()
