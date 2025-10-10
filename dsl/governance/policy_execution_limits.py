# Policy-driven Execution Limits
# Tasks 8.1-T01 to T25: Policy pack definitions, enforcement, and audit trails

import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    """Types of execution policies"""
    RATE_LIMIT = "rate_limit"
    BUDGET_CAP = "budget_cap"
    SCOPE_LIMIT = "scope_limit"
    RESIDENCY = "residency"
    SLA_LIMIT = "sla_limit"
    TIME_WINDOW = "time_window"
    RESOURCE_LIMIT = "resource_limit"

class PolicySeverity(Enum):
    """Policy violation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class IndustryOverlay(Enum):
    """Industry-specific policy overlays"""
    SAAS = "saas"
    BANKING = "banking"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    ECOMMERCE = "ecommerce"

class PolicyScope(Enum):
    """Policy application scope"""
    GLOBAL = "global"
    INDUSTRY = "industry"
    TENANT = "tenant"
    WORKFLOW = "workflow"
    USER = "user"

@dataclass
class PolicyLimit:
    """Individual policy limit definition"""
    limit_type: str
    limit_value: Union[int, float, str]
    unit: str
    time_window: Optional[str] = None
    description: str = ""

@dataclass
class ExecutionPolicy:
    """Complete execution policy definition"""
    # Core identification
    policy_id: str
    policy_name: str
    policy_type: PolicyType
    policy_version: str
    
    # Scope and targeting
    scope: PolicyScope
    industry_overlay: Optional[IndustryOverlay] = None
    tenant_id: Optional[int] = None
    workflow_pattern: Optional[str] = None
    role_restrictions: List[str] = field(default_factory=list)
    
    # Policy limits
    limits: List[PolicyLimit] = field(default_factory=list)
    
    # Enforcement settings
    enforcement_mode: str = "enforce"  # enforce, warn, monitor
    severity: PolicySeverity = PolicySeverity.ERROR
    
    # Metadata
    description: str = ""
    created_by: str = "system"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True
    
    # Audit and compliance
    compliance_frameworks: List[str] = field(default_factory=list)
    audit_required: bool = True
    evidence_retention_days: int = 2555  # 7 years default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def calculate_hash(self) -> str:
        """Calculate policy hash for integrity"""
        policy_data = {
            'policy_id': self.policy_id,
            'policy_type': self.policy_type.value,
            'limits': [asdict(limit) for limit in self.limits],
            'scope': self.scope.value,
            'enforcement_mode': self.enforcement_mode
        }
        
        content_json = json.dumps(policy_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content_json.encode()).hexdigest()

@dataclass
class PolicyViolation:
    """Policy violation record"""
    violation_id: str
    policy_id: str
    workflow_id: str
    execution_id: str
    tenant_id: int
    user_id: str
    
    # Violation details
    violation_type: str
    violation_message: str
    severity: PolicySeverity
    
    # Context
    attempted_value: Any
    policy_limit: Any
    violation_context: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Resolution
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class PolicyEnforcementResult:
    """Result of policy enforcement check"""
    allowed: bool
    policy_id: str
    violations: List[PolicyViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    enforcement_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class PolicyRegistry:
    """
    Policy Registry for versioned policy packs
    Tasks 8.1-T02, T03: Centralized policy control with inheritance
    """
    
    def __init__(self):
        self.policies: Dict[str, ExecutionPolicy] = {}
        self.policy_inheritance_cache: Dict[str, List[ExecutionPolicy]] = {}
        
        # Initialize with default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self) -> None:
        """Initialize default policy set"""
        
        # Global rate limit policy
        global_rate_policy = ExecutionPolicy(
            policy_id="global_rate_limit_v1",
            policy_name="Global Workflow Rate Limit",
            policy_type=PolicyType.RATE_LIMIT,
            policy_version="1.0.0",
            scope=PolicyScope.GLOBAL,
            limits=[
                PolicyLimit("max_executions_per_minute", 100, "executions/minute", "1m", "Maximum workflow executions per minute globally"),
                PolicyLimit("max_executions_per_hour", 1000, "executions/hour", "1h", "Maximum workflow executions per hour globally")
            ],
            description="Global rate limiting to prevent system overload",
            compliance_frameworks=["SOX", "GDPR"]
        )
        
        # SaaS industry overlay
        saas_rate_policy = ExecutionPolicy(
            policy_id="saas_rate_limit_v1",
            policy_name="SaaS Industry Rate Limit",
            policy_type=PolicyType.RATE_LIMIT,
            policy_version="1.0.0",
            scope=PolicyScope.INDUSTRY,
            industry_overlay=IndustryOverlay.SAAS,
            limits=[
                PolicyLimit("max_executions_per_minute", 200, "executions/minute", "1m", "SaaS-specific rate limit"),
                PolicyLimit("max_concurrent_workflows", 50, "concurrent", None, "Maximum concurrent workflows for SaaS")
            ],
            description="SaaS industry-specific rate limiting",
            compliance_frameworks=["SOX_SAAS", "GDPR_SAAS"]
        )
        
        # Banking industry overlay (stricter)
        banking_rate_policy = ExecutionPolicy(
            policy_id="banking_rate_limit_v1",
            policy_name="Banking Industry Rate Limit",
            policy_type=PolicyType.RATE_LIMIT,
            policy_version="1.0.0",
            scope=PolicyScope.INDUSTRY,
            industry_overlay=IndustryOverlay.BANKING,
            limits=[
                PolicyLimit("max_executions_per_minute", 50, "executions/minute", "1m", "Banking-specific rate limit (stricter)"),
                PolicyLimit("max_concurrent_workflows", 20, "concurrent", None, "Maximum concurrent workflows for Banking")
            ],
            description="Banking industry-specific rate limiting (stricter compliance)",
            compliance_frameworks=["RBI", "SOX_BANKING", "BASEL_III"]
        )
        
        # Budget cap policy
        budget_cap_policy = ExecutionPolicy(
            policy_id="global_budget_cap_v1",
            policy_name="Global Budget Cap Policy",
            policy_type=PolicyType.BUDGET_CAP,
            policy_version="1.0.0",
            scope=PolicyScope.GLOBAL,
            limits=[
                PolicyLimit("max_records_per_execution", 10000, "records", None, "Maximum records processed per execution"),
                PolicyLimit("max_notifications_per_execution", 100, "notifications", None, "Maximum notifications sent per execution"),
                PolicyLimit("max_api_calls_per_execution", 1000, "api_calls", None, "Maximum external API calls per execution")
            ],
            description="Global budget caps to prevent resource exhaustion",
            compliance_frameworks=["SOX", "GDPR"]
        )
        
        # Data residency policy
        residency_policy = ExecutionPolicy(
            policy_id="data_residency_v1",
            policy_name="Data Residency Policy",
            policy_type=PolicyType.RESIDENCY,
            policy_version="1.0.0",
            scope=PolicyScope.GLOBAL,
            limits=[
                PolicyLimit("allowed_regions", "US,EU,IN", "regions", None, "Allowed data processing regions"),
                PolicyLimit("cross_region_transfer", False, "boolean", None, "Cross-region data transfer allowed")
            ],
            description="Data residency compliance for GDPR/DPDP",
            compliance_frameworks=["GDPR", "DPDP", "CCPA"],
            severity=PolicySeverity.CRITICAL
        )
        
        # SLA limit policy
        sla_policy = ExecutionPolicy(
            policy_id="global_sla_limits_v1",
            policy_name="Global SLA Limits",
            policy_type=PolicyType.SLA_LIMIT,
            policy_version="1.0.0",
            scope=PolicyScope.GLOBAL,
            limits=[
                PolicyLimit("max_execution_time_seconds", 300, "seconds", None, "Maximum workflow execution time"),
                PolicyLimit("policy_enforcement_latency_ms", 100, "milliseconds", None, "Maximum policy enforcement latency")
            ],
            description="Global SLA limits for performance compliance",
            compliance_frameworks=["SOX", "INTERNAL_SLA"]
        )
        
        # Store policies
        for policy in [global_rate_policy, saas_rate_policy, banking_rate_policy, budget_cap_policy, residency_policy, sla_policy]:
            self.policies[policy.policy_id] = policy
        
        logger.info(f"✅ Initialized {len(self.policies)} default policies")
    
    def register_policy(self, policy: ExecutionPolicy) -> bool:
        """Register a new policy"""
        try:
            # Validate policy
            if not policy.policy_id or not policy.policy_name:
                raise ValueError("Policy must have ID and name")
            
            # Calculate and store hash
            policy_hash = policy.calculate_hash()
            
            # Store policy
            self.policies[policy.policy_id] = policy
            
            # Clear inheritance cache
            self.policy_inheritance_cache.clear()
            
            logger.info(f"✅ Registered policy: {policy.policy_id} (hash: {policy_hash[:8]}...)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register policy {policy.policy_id}: {e}")
            return False
    
    def get_policy(self, policy_id: str) -> Optional[ExecutionPolicy]:
        """Get policy by ID"""
        return self.policies.get(policy_id)
    
    def get_applicable_policies(
        self,
        tenant_id: Optional[int] = None,
        industry_overlay: Optional[IndustryOverlay] = None,
        workflow_id: Optional[str] = None,
        user_role: Optional[str] = None
    ) -> List[ExecutionPolicy]:
        """
        Get applicable policies with inheritance resolution
        Tasks 8.1-T03: Policy inheritance (global → industry → tenant → workflow)
        """
        
        cache_key = f"{tenant_id}_{industry_overlay}_{workflow_id}_{user_role}"
        
        if cache_key in self.policy_inheritance_cache:
            return self.policy_inheritance_cache[cache_key]
        
        applicable_policies = []
        
        # 1. Global policies (always apply)
        global_policies = [p for p in self.policies.values() if p.scope == PolicyScope.GLOBAL and p.is_active]
        applicable_policies.extend(global_policies)
        
        # 2. Industry-specific policies
        if industry_overlay:
            industry_policies = [
                p for p in self.policies.values() 
                if p.scope == PolicyScope.INDUSTRY 
                and p.industry_overlay == industry_overlay 
                and p.is_active
            ]
            applicable_policies.extend(industry_policies)
        
        # 3. Tenant-specific policies
        if tenant_id:
            tenant_policies = [
                p for p in self.policies.values() 
                if p.scope == PolicyScope.TENANT 
                and p.tenant_id == tenant_id 
                and p.is_active
            ]
            applicable_policies.extend(tenant_policies)
        
        # 4. Workflow-specific policies
        if workflow_id:
            workflow_policies = [
                p for p in self.policies.values() 
                if p.scope == PolicyScope.WORKFLOW 
                and p.workflow_pattern 
                and workflow_id.startswith(p.workflow_pattern) 
                and p.is_active
            ]
            applicable_policies.extend(workflow_policies)
        
        # 5. Role-based filtering
        if user_role:
            applicable_policies = [
                p for p in applicable_policies
                if not p.role_restrictions or user_role in p.role_restrictions
            ]
        
        # Cache result
        self.policy_inheritance_cache[cache_key] = applicable_policies
        
        logger.debug(f"🔍 Found {len(applicable_policies)} applicable policies for {cache_key}")
        return applicable_policies
    
    def list_policies(
        self,
        policy_type: Optional[PolicyType] = None,
        scope: Optional[PolicyScope] = None,
        active_only: bool = True
    ) -> List[ExecutionPolicy]:
        """List policies with optional filtering"""
        
        policies = list(self.policies.values())
        
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        
        if scope:
            policies = [p for p in policies if p.scope == scope]
        
        if active_only:
            policies = [p for p in policies if p.is_active]
        
        return policies
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get policy registry statistics"""
        
        total_policies = len(self.policies)
        active_policies = len([p for p in self.policies.values() if p.is_active])
        
        policies_by_type = {}
        policies_by_scope = {}
        policies_by_industry = {}
        
        for policy in self.policies.values():
            # By type
            policy_type = policy.policy_type.value
            policies_by_type[policy_type] = policies_by_type.get(policy_type, 0) + 1
            
            # By scope
            scope = policy.scope.value
            policies_by_scope[scope] = policies_by_scope.get(scope, 0) + 1
            
            # By industry
            if policy.industry_overlay:
                industry = policy.industry_overlay.value
                policies_by_industry[industry] = policies_by_industry.get(industry, 0) + 1
        
        return {
            'total_policies': total_policies,
            'active_policies': active_policies,
            'inactive_policies': total_policies - active_policies,
            'policies_by_type': policies_by_type,
            'policies_by_scope': policies_by_scope,
            'policies_by_industry': policies_by_industry,
            'cache_size': len(self.policy_inheritance_cache)
        }

class PolicyEnforcer:
    """
    Policy Enforcement Engine
    Tasks 8.1-T09 to T18: Runtime enforcement, violation logging, alerts
    """
    
    def __init__(self, policy_registry: PolicyRegistry):
        self.policy_registry = policy_registry
        self.violation_log: List[PolicyViolation] = []
        self.enforcement_metrics = {
            'total_checks': 0,
            'violations_detected': 0,
            'policies_enforced': 0,
            'average_enforcement_time_ms': 0.0
        }
    
    async def enforce_policies(
        self,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        user_id: str,
        user_role: str,
        industry_overlay: IndustryOverlay,
        execution_context: Dict[str, Any]
    ) -> PolicyEnforcementResult:
        """
        Enforce applicable policies for workflow execution
        Tasks 8.1-T10, T11: Runtime enforcement with violation logging
        """
        
        start_time = datetime.now()
        
        # Get applicable policies
        applicable_policies = self.policy_registry.get_applicable_policies(
            tenant_id=tenant_id,
            industry_overlay=industry_overlay,
            workflow_id=workflow_id,
            user_role=user_role
        )
        
        violations = []
        warnings = []
        overall_allowed = True
        
        # Check each applicable policy
        for policy in applicable_policies:
            try:
                policy_result = await self._check_policy(
                    policy, workflow_id, execution_id, tenant_id, user_id, execution_context
                )
                
                if not policy_result.allowed:
                    overall_allowed = False
                    violations.extend(policy_result.violations)
                
                warnings.extend(policy_result.warnings)
                
            except Exception as e:
                logger.error(f"❌ Policy enforcement error for {policy.policy_id}: {e}")
                
                # Create violation for enforcement failure
                violation = PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id=policy.policy_id,
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    violation_type="enforcement_error",
                    violation_message=f"Policy enforcement failed: {str(e)}",
                    severity=PolicySeverity.CRITICAL,
                    attempted_value="unknown",
                    policy_limit="unknown"
                )
                
                violations.append(violation)
                overall_allowed = False
        
        # Calculate enforcement time
        end_time = datetime.now()
        enforcement_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update metrics
        self.enforcement_metrics['total_checks'] += 1
        self.enforcement_metrics['violations_detected'] += len(violations)
        self.enforcement_metrics['policies_enforced'] += len(applicable_policies)
        
        # Update average enforcement time
        current_avg = self.enforcement_metrics['average_enforcement_time_ms']
        total_checks = self.enforcement_metrics['total_checks']
        self.enforcement_metrics['average_enforcement_time_ms'] = (
            (current_avg * (total_checks - 1) + enforcement_time_ms) / total_checks
        )
        
        # Log violations
        for violation in violations:
            self.violation_log.append(violation)
            logger.warning(f"🚨 Policy violation: {violation.violation_message} (Policy: {violation.policy_id})")
        
        # Check SLA compliance (enforcement should be < 100ms)
        if enforcement_time_ms > 100:
            logger.warning(f"⚠️ Policy enforcement SLA breach: {enforcement_time_ms:.2f}ms > 100ms")
        
        result = PolicyEnforcementResult(
            allowed=overall_allowed,
            policy_id=f"enforcement_{len(applicable_policies)}_policies",
            violations=violations,
            warnings=warnings,
            enforcement_time_ms=enforcement_time_ms
        )
        
        logger.info(f"🛡️ Policy enforcement: {len(applicable_policies)} policies checked, {len(violations)} violations, {enforcement_time_ms:.2f}ms")
        
        return result
    
    async def _check_policy(
        self,
        policy: ExecutionPolicy,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        user_id: str,
        execution_context: Dict[str, Any]
    ) -> PolicyEnforcementResult:
        """Check individual policy against execution context"""
        
        violations = []
        warnings = []
        
        for limit in policy.limits:
            violation = await self._check_limit(
                policy, limit, workflow_id, execution_id, tenant_id, user_id, execution_context
            )
            
            if violation:
                if policy.enforcement_mode == "enforce":
                    violations.append(violation)
                elif policy.enforcement_mode == "warn":
                    warnings.append(f"Policy warning: {violation.violation_message}")
                # "monitor" mode just logs, no action
        
        return PolicyEnforcementResult(
            allowed=len(violations) == 0,
            policy_id=policy.policy_id,
            violations=violations,
            warnings=warnings
        )
    
    async def _check_limit(
        self,
        policy: ExecutionPolicy,
        limit: PolicyLimit,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        user_id: str,
        execution_context: Dict[str, Any]
    ) -> Optional[PolicyViolation]:
        """Check individual limit against execution context"""
        
        try:
            # Get current value from execution context
            current_value = execution_context.get(limit.limit_type)
            
            if current_value is None:
                # If we can't find the value, assume it's within limits
                return None
            
            # Check different limit types
            if limit.limit_type == "max_executions_per_minute":
                # Mock implementation - would check actual execution rate
                if isinstance(current_value, (int, float)) and current_value > limit.limit_value:
                    return self._create_violation(
                        policy, limit, workflow_id, execution_id, tenant_id, user_id,
                        f"Execution rate {current_value} exceeds limit {limit.limit_value}",
                        current_value, limit.limit_value
                    )
            
            elif limit.limit_type == "max_records_per_execution":
                if isinstance(current_value, (int, float)) and current_value > limit.limit_value:
                    return self._create_violation(
                        policy, limit, workflow_id, execution_id, tenant_id, user_id,
                        f"Record count {current_value} exceeds limit {limit.limit_value}",
                        current_value, limit.limit_value
                    )
            
            elif limit.limit_type == "allowed_regions":
                current_region = execution_context.get("data_region", "US")
                allowed_regions = str(limit.limit_value).split(",")
                if current_region not in allowed_regions:
                    return self._create_violation(
                        policy, limit, workflow_id, execution_id, tenant_id, user_id,
                        f"Data region {current_region} not in allowed regions {allowed_regions}",
                        current_region, allowed_regions
                    )
            
            elif limit.limit_type == "max_execution_time_seconds":
                if isinstance(current_value, (int, float)) and current_value > limit.limit_value:
                    return self._create_violation(
                        policy, limit, workflow_id, execution_id, tenant_id, user_id,
                        f"Execution time {current_value}s exceeds limit {limit.limit_value}s",
                        current_value, limit.limit_value
                    )
            
            # Add more limit type checks as needed
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error checking limit {limit.limit_type}: {e}")
            return self._create_violation(
                policy, limit, workflow_id, execution_id, tenant_id, user_id,
                f"Limit check error: {str(e)}",
                "error", "unknown"
            )
    
    def _create_violation(
        self,
        policy: ExecutionPolicy,
        limit: PolicyLimit,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        user_id: str,
        message: str,
        attempted_value: Any,
        policy_limit: Any
    ) -> PolicyViolation:
        """Create a policy violation record"""
        
        return PolicyViolation(
            violation_id=str(uuid.uuid4()),
            policy_id=policy.policy_id,
            workflow_id=workflow_id,
            execution_id=execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            violation_type=limit.limit_type,
            violation_message=message,
            severity=policy.severity,
            attempted_value=attempted_value,
            policy_limit=policy_limit,
            violation_context={
                'policy_name': policy.policy_name,
                'limit_description': limit.description,
                'enforcement_mode': policy.enforcement_mode
            }
        )
    
    def get_violations(
        self,
        tenant_id: Optional[int] = None,
        policy_id: Optional[str] = None,
        severity: Optional[PolicySeverity] = None,
        limit: int = 100
    ) -> List[PolicyViolation]:
        """Get policy violations with optional filtering"""
        
        violations = self.violation_log
        
        if tenant_id:
            violations = [v for v in violations if v.tenant_id == tenant_id]
        
        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        # Sort by most recent first
        violations.sort(key=lambda v: v.occurred_at, reverse=True)
        
        return violations[:limit]
    
    def get_enforcement_metrics(self) -> Dict[str, Any]:
        """Get policy enforcement metrics"""
        
        recent_violations = [
            v for v in self.violation_log
            if datetime.fromisoformat(v.occurred_at.replace('Z', '+00:00')) > datetime.now(timezone.utc) - timedelta(hours=24)
        ]
        
        violations_by_policy = {}
        violations_by_severity = {}
        
        for violation in recent_violations:
            # By policy
            policy_id = violation.policy_id
            violations_by_policy[policy_id] = violations_by_policy.get(policy_id, 0) + 1
            
            # By severity
            severity = violation.severity.value
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        
        return {
            **self.enforcement_metrics,
            'recent_violations_24h': len(recent_violations),
            'total_violations': len(self.violation_log),
            'violations_by_policy': violations_by_policy,
            'violations_by_severity': violations_by_severity,
            'violation_rate_percent': (self.enforcement_metrics['violations_detected'] / max(self.enforcement_metrics['total_checks'], 1)) * 100
        }

class PolicyAuditLogger:
    """
    Policy Audit Logger for immutable audit trails
    Tasks 8.1-T12, T13: Hash/sign policy packs and link to evidence
    """
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
    
    async def log_policy_application(
        self,
        policy: ExecutionPolicy,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        user_id: str,
        enforcement_result: PolicyEnforcementResult,
        evidence_pack_id: Optional[str] = None
    ) -> str:
        """Log policy application for audit trail"""
        
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': 'policy_application',
            'policy_id': policy.policy_id,
            'policy_hash': policy.calculate_hash(),
            'workflow_id': workflow_id,
            'execution_id': execution_id,
            'tenant_id': tenant_id,
            'user_id': user_id,
            'enforcement_result': enforcement_result.to_dict(),
            'evidence_pack_id': evidence_pack_id,
            'compliance_frameworks': policy.compliance_frameworks
        }
        
        # Calculate audit entry hash for integrity
        audit_hash = hashlib.sha256(
            json.dumps(audit_entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        audit_entry['audit_hash'] = audit_hash
        
        self.audit_log.append(audit_entry)
        
        logger.info(f"📋 Policy application logged: {audit_entry['audit_id']} (hash: {audit_hash[:8]}...)")
        
        return audit_entry['audit_id']
    
    async def log_policy_violation(
        self,
        violation: PolicyViolation,
        evidence_pack_id: Optional[str] = None
    ) -> str:
        """Log policy violation for audit trail"""
        
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': 'policy_violation',
            'violation': violation.to_dict(),
            'evidence_pack_id': evidence_pack_id
        }
        
        # Calculate audit entry hash
        audit_hash = hashlib.sha256(
            json.dumps(audit_entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        audit_entry['audit_hash'] = audit_hash
        
        self.audit_log.append(audit_entry)
        
        logger.warning(f"🚨 Policy violation logged: {audit_entry['audit_id']} (hash: {audit_hash[:8]}...)")
        
        return audit_entry['audit_id']
    
    def get_audit_trail(
        self,
        tenant_id: Optional[int] = None,
        policy_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit trail with optional filtering"""
        
        entries = self.audit_log
        
        if tenant_id:
            entries = [e for e in entries if e.get('tenant_id') == tenant_id or e.get('violation', {}).get('tenant_id') == tenant_id]
        
        if policy_id:
            entries = [e for e in entries if e.get('policy_id') == policy_id or e.get('violation', {}).get('policy_id') == policy_id]
        
        if event_type:
            entries = [e for e in entries if e.get('event_type') == event_type]
        
        # Sort by most recent first
        entries.sort(key=lambda e: e['timestamp'], reverse=True)
        
        return entries[:limit]
    
    def verify_audit_integrity(self, audit_id: str) -> bool:
        """Verify audit entry integrity using hash"""
        
        entry = next((e for e in self.audit_log if e['audit_id'] == audit_id), None)
        
        if not entry:
            return False
        
        # Recalculate hash
        entry_copy = entry.copy()
        stored_hash = entry_copy.pop('audit_hash')
        
        calculated_hash = hashlib.sha256(
            json.dumps(entry_copy, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        return stored_hash == calculated_hash

# Global instances
policy_registry = PolicyRegistry()
policy_enforcer = PolicyEnforcer(policy_registry)
policy_audit_logger = PolicyAuditLogger()
