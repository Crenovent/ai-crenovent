"""
Chapter 8.5 Governance Hooks Middleware
Tasks 8.5.1-8.5.42: Comprehensive governance enforcement across all planes

SaaS-Focused Implementation:
- Control Plane: Workflow orchestration governance
- Execution Plane: Agent execution governance  
- Data Plane: Data access and transformation governance
- Governance Plane: Evidence and audit governance
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import hashlib
import hmac

logger = logging.getLogger(__name__)

class GovernancePlane(Enum):
    """Governance enforcement planes (Task 8.5.1)"""
    CONTROL = "control_plane"      # Orchestration governance
    EXECUTION = "execution_plane"   # Agent execution governance
    DATA = "data_plane"            # Data access governance
    GOVERNANCE = "governance_plane" # Evidence/audit governance

class GovernanceHookType(Enum):
    """Governance hook taxonomy (Task 8.5.1)"""
    POLICY = "policy_enforcement"
    CONSENT = "consent_validation"
    SLA = "sla_monitoring"
    FINOPS = "finops_tagging"
    LINEAGE = "lineage_tracking"
    TRUST = "trust_scoring"
    SOD = "segregation_of_duties"
    RESIDENCY = "data_residency"
    ANOMALY = "anomaly_detection"

class GovernanceDecision(Enum):
    """Governance enforcement decisions"""
    ALLOW = "allow"
    DENY = "deny"
    OVERRIDE_REQUIRED = "override_required"
    ESCALATE = "escalate"

@dataclass
class GovernanceContext:
    """Context for governance enforcement"""
    tenant_id: int
    user_id: str
    user_role: str
    workflow_id: str
    execution_id: str
    plane: GovernancePlane
    hook_type: GovernanceHookType
    request_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    
class GovernanceHookResult:
    """Result of governance hook execution"""
    def __init__(self, decision: GovernanceDecision, hook_type: GovernanceHookType, 
                 context: GovernanceContext, violations: List[str] = None,
                 evidence_pack_id: str = None, override_required: bool = False):
        self.decision = decision
        self.hook_type = hook_type
        self.context = context
        self.violations = violations or []
        self.evidence_pack_id = evidence_pack_id
        self.override_required = override_required
        self.execution_time_ms = 0
        self.trust_score_impact = 0.0
        self.lineage_tags = []

class GovernanceHooksMiddleware:
    """
    Core governance hooks middleware for all planes
    Tasks 8.5.2-8.5.5: Middleware for Control/Execution/Data/Governance planes
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # SaaS-specific governance configurations
        self.saas_governance_config = {
            "policy_enforcement": {
                "strict_mode": True,
                "fail_closed": True,
                "override_approval_required": True
            },
            "consent_validation": {
                "gdpr_required": True,
                "ccpa_required": True,
                "consent_expiry_check": True
            },
            "sla_monitoring": {
                "error_budget_enforcement": True,
                "burn_rate_alerts": True,
                "escalation_thresholds": {
                    "warning": 0.7,
                    "critical": 0.9
                }
            },
            "finops_tagging": {
                "cost_attribution": True,
                "chargeback_enabled": True,
                "budget_alerts": True
            },
            "trust_scoring": {
                "minimum_trust_threshold": 0.7,
                "block_low_trust": True,
                "trust_decay_enabled": True
            },
            "sod_enforcement": {
                "strict_separation": True,
                "role_conflict_detection": True,
                "approval_chain_validation": True
            }
        }
        
        # Evidence pack generation config (Task 8.5.7)
        self.evidence_config = {
            "auto_generation": True,
            "worm_storage": True,
            "digital_signatures": True,
            "hash_chaining": True,
            "retention_years": 7
        }
        
        # Initialize hook registry
        self.hook_registry = {}
        self._initialize_saas_hooks()
    
    def _initialize_saas_hooks(self):
        """Initialize SaaS-specific governance hooks"""
        # Register SaaS governance hooks for each plane
        self.hook_registry = {
            GovernancePlane.CONTROL: [
                GovernanceHookType.POLICY,
                GovernanceHookType.SLA,
                GovernanceHookType.TRUST,
                GovernanceHookType.SOD
            ],
            GovernancePlane.EXECUTION: [
                GovernanceHookType.POLICY,
                GovernanceHookType.CONSENT,
                GovernanceHookType.TRUST,
                GovernanceHookType.ANOMALY
            ],
            GovernancePlane.DATA: [
                GovernanceHookType.CONSENT,
                GovernanceHookType.RESIDENCY,
                GovernanceHookType.LINEAGE,
                GovernanceHookType.SOD
            ],
            GovernancePlane.GOVERNANCE: [
                GovernanceHookType.POLICY,
                GovernanceHookType.FINOPS,
                GovernanceHookType.LINEAGE,
                GovernanceHookType.ANOMALY
            ]
        }
    
    async def enforce_governance(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Main governance enforcement entry point
        Executes all applicable hooks for the given plane and context
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🔒 Enforcing governance for {context.plane.value} - {context.hook_type.value}")
            
            # Get applicable hooks for this plane
            applicable_hooks = self.hook_registry.get(context.plane, [])
            
            if context.hook_type not in applicable_hooks:
                return GovernanceHookResult(
                    decision=GovernanceDecision.ALLOW,
                    hook_type=context.hook_type,
                    context=context
                )
            
            # Execute governance hook based on type
            result = await self._execute_governance_hook(context)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            # Generate evidence pack (Task 8.5.7)
            if self.evidence_config["auto_generation"]:
                evidence_pack_id = await self._generate_evidence_pack(context, result)
                result.evidence_pack_id = evidence_pack_id
            
            # Update override ledger if needed (Task 8.5.6)
            if result.override_required:
                await self._log_override_requirement(context, result)
            
            # Log governance event
            await self._log_governance_event(context, result)
            
            self.logger.info(f"✅ Governance enforcement completed: {result.decision.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Governance enforcement failed: {e}")
            # Fail closed - deny on error
            return GovernanceHookResult(
                decision=GovernanceDecision.DENY,
                hook_type=context.hook_type,
                context=context,
                violations=[f"Governance enforcement error: {str(e)}"]
            )
    
    async def _execute_governance_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """Execute specific governance hook based on type"""
        
        if context.hook_type == GovernanceHookType.POLICY:
            return await self._enforce_policy_hook(context)
        elif context.hook_type == GovernanceHookType.CONSENT:
            return await self._enforce_consent_hook(context)
        elif context.hook_type == GovernanceHookType.SLA:
            return await self._enforce_sla_hook(context)
        elif context.hook_type == GovernanceHookType.FINOPS:
            return await self._enforce_finops_hook(context)
        elif context.hook_type == GovernanceHookType.LINEAGE:
            return await self._enforce_lineage_hook(context)
        elif context.hook_type == GovernanceHookType.TRUST:
            return await self._enforce_trust_hook(context)
        elif context.hook_type == GovernanceHookType.SOD:
            return await self._enforce_sod_hook(context)
        elif context.hook_type == GovernanceHookType.RESIDENCY:
            return await self._enforce_residency_hook(context)
        elif context.hook_type == GovernanceHookType.ANOMALY:
            return await self._enforce_anomaly_hook(context)
        else:
            return GovernanceHookResult(
                decision=GovernanceDecision.ALLOW,
                hook_type=context.hook_type,
                context=context
            )
    
    async def _enforce_policy_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Policy enforcement hook (Task 8.5.2, 8.5.3)
        SaaS-specific policy validation
        """
        violations = []
        
        # SaaS policy checks
        saas_policies = {
            "data_retention": {
                "max_days": 2555,  # 7 years
                "required": True
            },
            "encryption_at_rest": {
                "required": True,
                "algorithm": "AES-256"
            },
            "api_rate_limiting": {
                "requests_per_minute": 1000,
                "burst_limit": 100
            },
            "multi_tenancy": {
                "isolation_required": True,
                "cross_tenant_access": False
            }
        }
        
        # Check SaaS-specific policies
        request_data = context.request_data
        
        # Data retention policy
        if "data_retention_days" in request_data:
            retention_days = request_data["data_retention_days"]
            if retention_days > saas_policies["data_retention"]["max_days"]:
                violations.append(f"Data retention exceeds maximum: {retention_days} > {saas_policies['data_retention']['max_days']}")
        
        # Multi-tenancy isolation
        if context.tenant_id and "target_tenant_id" in request_data:
            target_tenant = request_data["target_tenant_id"]
            if target_tenant != context.tenant_id:
                violations.append(f"Cross-tenant access violation: {context.tenant_id} -> {target_tenant}")
        
        # API rate limiting for SaaS
        if context.plane == GovernancePlane.EXECUTION:
            # Check rate limits (simplified)
            current_rate = request_data.get("current_api_rate", 0)
            if current_rate > saas_policies["api_rate_limiting"]["requests_per_minute"]:
                violations.append(f"API rate limit exceeded: {current_rate} > {saas_policies['api_rate_limiting']['requests_per_minute']}")
        
        decision = GovernanceDecision.DENY if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def _enforce_consent_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Consent validation hook (Task 8.5.9)
        GDPR/CCPA compliance for SaaS
        """
        violations = []
        request_data = context.request_data
        
        # Check for required consent fields
        required_consent_fields = ["consent_id", "consent_type", "consent_expiry"]
        
        for field in required_consent_fields:
            if field not in request_data:
                violations.append(f"Missing required consent field: {field}")
        
        # Validate consent expiry
        if "consent_expiry" in request_data:
            try:
                expiry_date = datetime.fromisoformat(request_data["consent_expiry"])
                if expiry_date < datetime.now():
                    violations.append(f"Consent expired: {request_data['consent_expiry']}")
            except ValueError:
                violations.append(f"Invalid consent expiry format: {request_data['consent_expiry']}")
        
        # Check consent scope for SaaS operations
        consent_scope = request_data.get("consent_scope", [])
        operation_type = request_data.get("operation_type", "unknown")
        
        saas_operations_requiring_consent = [
            "data_processing", "analytics", "marketing", "personalization"
        ]
        
        if operation_type in saas_operations_requiring_consent and operation_type not in consent_scope:
            violations.append(f"Operation '{operation_type}' not covered by consent scope")
        
        decision = GovernanceDecision.DENY if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def _enforce_sla_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        SLA monitoring hook (Task 8.5.8)
        SaaS SLA enforcement and error budget validation
        """
        violations = []
        request_data = context.request_data
        
        # SaaS SLA tiers
        sla_tiers = {
            "enterprise": {"uptime": 99.99, "response_time_ms": 100},
            "professional": {"uptime": 99.9, "response_time_ms": 200},
            "standard": {"uptime": 99.5, "response_time_ms": 500}
        }
        
        tenant_tier = request_data.get("sla_tier", "standard")
        current_uptime = request_data.get("current_uptime", 100.0)
        response_time = request_data.get("response_time_ms", 0)
        
        if tenant_tier in sla_tiers:
            sla_config = sla_tiers[tenant_tier]
            
            # Check uptime SLA
            if current_uptime < sla_config["uptime"]:
                violations.append(f"Uptime SLA breach: {current_uptime}% < {sla_config['uptime']}%")
            
            # Check response time SLA
            if response_time > sla_config["response_time_ms"]:
                violations.append(f"Response time SLA breach: {response_time}ms > {sla_config['response_time_ms']}ms")
        
        # Check error budget
        error_budget = request_data.get("error_budget_remaining", 1.0)
        if error_budget < self.saas_governance_config["sla_monitoring"]["escalation_thresholds"]["critical"]:
            violations.append(f"Critical error budget depletion: {error_budget}")
        
        decision = GovernanceDecision.ESCALATE if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def _enforce_finops_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        FinOps tagging hook (Task 8.5.10)
        Cost attribution and budget enforcement for SaaS
        """
        violations = []
        request_data = context.request_data
        
        # Required FinOps tags for SaaS
        required_tags = ["tenant_id", "region_id", "cost_center", "environment"]
        
        for tag in required_tags:
            if tag not in request_data:
                violations.append(f"Missing required FinOps tag: {tag}")
        
        # Budget validation
        estimated_cost = request_data.get("estimated_cost_usd", 0)
        tenant_budget = request_data.get("tenant_monthly_budget", float('inf'))
        
        if estimated_cost > tenant_budget * 0.1:  # 10% of monthly budget
            violations.append(f"Operation cost exceeds 10% of monthly budget: ${estimated_cost}")
        
        decision = GovernanceDecision.OVERRIDE_REQUIRED if violations else GovernanceDecision.ALLOW
        
        result = GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
        
        # Add FinOps lineage tags
        result.lineage_tags = [
            f"tenant_id:{context.tenant_id}",
            f"cost_center:{request_data.get('cost_center', 'unknown')}",
            f"estimated_cost:{estimated_cost}"
        ]
        
        return result
    
    async def _enforce_lineage_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Lineage tracking hook (Task 8.5.13)
        Data lineage and traceability for SaaS
        """
        violations = []
        request_data = context.request_data
        
        # Required lineage metadata
        required_lineage_fields = ["source_system", "data_classification", "processing_purpose"]
        
        for field in required_lineage_fields:
            if field not in request_data:
                violations.append(f"Missing lineage field: {field}")
        
        # Validate data classification
        valid_classifications = ["public", "internal", "confidential", "restricted"]
        data_classification = request_data.get("data_classification", "").lower()
        
        if data_classification and data_classification not in valid_classifications:
            violations.append(f"Invalid data classification: {data_classification}")
        
        decision = GovernanceDecision.ALLOW  # Lineage is informational
        
        result = GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
        
        # Generate lineage tags
        result.lineage_tags = [
            f"source:{request_data.get('source_system', 'unknown')}",
            f"classification:{data_classification}",
            f"purpose:{request_data.get('processing_purpose', 'unknown')}",
            f"workflow_id:{context.workflow_id}",
            f"execution_id:{context.execution_id}"
        ]
        
        return result
    
    async def _enforce_trust_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Trust score enforcement hook (Task 8.5.11)
        Block low-trust actions for SaaS
        """
        violations = []
        request_data = context.request_data
        
        # Get current trust score
        current_trust_score = request_data.get("trust_score", 1.0)
        minimum_threshold = self.saas_governance_config["trust_scoring"]["minimum_trust_threshold"]
        
        if current_trust_score < minimum_threshold:
            violations.append(f"Trust score below threshold: {current_trust_score} < {minimum_threshold}")
        
        # Check trust score decay
        last_update = request_data.get("trust_score_last_update")
        if last_update:
            try:
                last_update_date = datetime.fromisoformat(last_update)
                days_since_update = (datetime.now() - last_update_date).days
                
                if days_since_update > 30:  # Trust score stale
                    violations.append(f"Trust score stale: {days_since_update} days since last update")
            except ValueError:
                violations.append(f"Invalid trust score timestamp: {last_update}")
        
        decision = GovernanceDecision.OVERRIDE_REQUIRED if violations else GovernanceDecision.ALLOW
        
        result = GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations,
            override_required=bool(violations)
        )
        
        result.trust_score_impact = -0.1 if violations else 0.05  # Penalty or reward
        
        return result
    
    async def _enforce_sod_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Segregation of Duties enforcement hook (Task 8.5.12)
        Prevent conflicts of duty in SaaS operations
        """
        violations = []
        request_data = context.request_data
        
        # SaaS SoD rules
        sod_conflicts = {
            "financial_approver": ["financial_requester", "payment_processor"],
            "data_admin": ["data_auditor"],
            "security_admin": ["security_auditor"],
            "tenant_admin": ["tenant_auditor"]
        }
        
        user_role = context.user_role
        operation_roles = request_data.get("operation_roles", [])
        
        # Check for SoD conflicts
        if user_role in sod_conflicts:
            conflicting_roles = sod_conflicts[user_role]
            for conflict_role in conflicting_roles:
                if conflict_role in operation_roles:
                    violations.append(f"SoD violation: {user_role} cannot perform {conflict_role} operations")
        
        # Check approval chain validation
        if "approval_chain" in request_data:
            approval_chain = request_data["approval_chain"]
            if len(set(approval_chain)) != len(approval_chain):
                violations.append("SoD violation: Duplicate approvers in approval chain")
        
        decision = GovernanceDecision.DENY if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def _enforce_residency_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Data residency enforcement hook (Task 8.5.17)
        Prevent cross-border data breaches for SaaS
        """
        violations = []
        request_data = context.request_data
        
        # Get tenant's data residency requirements
        tenant_region = request_data.get("tenant_region", "GLOBAL")
        target_region = request_data.get("target_region", tenant_region)
        data_classification = request_data.get("data_classification", "public")
        
        # Residency rules for different regions
        residency_rules = {
            "EU": {
                "allowed_regions": ["EU", "UK"],  # GDPR adequacy decisions
                "restricted_data": ["personal", "sensitive"]
            },
            "IN": {
                "allowed_regions": ["IN"],  # DPDP strict localization
                "restricted_data": ["critical_personal", "financial"]
            },
            "US": {
                "allowed_regions": ["US", "CA"],  # USMCA agreement
                "restricted_data": ["government", "defense"]
            }
        }
        
        if tenant_region in residency_rules:
            rules = residency_rules[tenant_region]
            
            # Check if target region is allowed
            if target_region not in rules["allowed_regions"]:
                violations.append(f"Cross-border transfer violation: {tenant_region} -> {target_region}")
            
            # Check data classification restrictions
            if data_classification in rules["restricted_data"] and target_region != tenant_region:
                violations.append(f"Restricted data transfer: {data_classification} data cannot leave {tenant_region}")
        
        decision = GovernanceDecision.DENY if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def _enforce_anomaly_hook(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Anomaly detection hook (Task 8.5.15)
        Detect rogue behavior in SaaS operations
        """
        violations = []
        request_data = context.request_data
        
        # Anomaly detection rules for SaaS
        anomaly_thresholds = {
            "api_calls_per_minute": 1000,
            "data_volume_mb": 1000,
            "concurrent_sessions": 100,
            "failed_auth_attempts": 5
        }
        
        # Check for anomalies
        for metric, threshold in anomaly_thresholds.items():
            current_value = request_data.get(metric, 0)
            if current_value > threshold:
                violations.append(f"Anomaly detected - {metric}: {current_value} > {threshold}")
        
        # Time-based anomaly detection
        request_time = context.timestamp
        if request_time.hour < 6 or request_time.hour > 22:  # Outside business hours
            operation_type = request_data.get("operation_type", "")
            if operation_type in ["data_export", "admin_access", "bulk_delete"]:
                violations.append(f"Suspicious off-hours operation: {operation_type} at {request_time.hour}:00")
        
        decision = GovernanceDecision.ESCALATE if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def _generate_evidence_pack(self, context: GovernanceContext, result: GovernanceHookResult) -> str:
        """
        Generate evidence pack for governance event (Task 8.5.7)
        Immutable compliance artifacts with WORM storage
        """
        try:
            evidence_pack_id = str(uuid.uuid4())
            
            evidence_pack = {
                "evidence_pack_id": evidence_pack_id,
                "tenant_id": context.tenant_id,
                "governance_event": {
                    "plane": context.plane.value,
                    "hook_type": context.hook_type.value,
                    "decision": result.decision.value,
                    "execution_time_ms": result.execution_time_ms,
                    "violations": result.violations,
                    "trust_score_impact": result.trust_score_impact
                },
                "context": {
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id,
                    "user_id": context.user_id,
                    "user_role": context.user_role,
                    "timestamp": context.timestamp.isoformat(),
                    "request_data_hash": hashlib.sha256(
                        json.dumps(context.request_data, sort_keys=True).encode()
                    ).hexdigest()
                },
                "compliance_metadata": {
                    "framework": "SaaS_Governance",
                    "retention_years": self.evidence_config["retention_years"],
                    "worm_storage": self.evidence_config["worm_storage"],
                    "digital_signature": self.evidence_config["digital_signatures"]
                },
                "lineage_tags": result.lineage_tags,
                "created_at": datetime.now().isoformat()
            }
            
            # Generate digital signature
            if self.evidence_config["digital_signatures"]:
                signature = self._generate_digital_signature(evidence_pack)
                evidence_pack["digital_signature"] = signature
            
            # Store in WORM storage (simulated)
            await self._store_evidence_pack_worm(evidence_pack_id, evidence_pack)
            
            self.logger.info(f"✅ Evidence pack generated: {evidence_pack_id}")
            return evidence_pack_id
            
        except Exception as e:
            self.logger.error(f"❌ Evidence pack generation failed: {e}")
            return None
    
    def _generate_digital_signature(self, evidence_pack: Dict[str, Any]) -> str:
        """Generate HMAC-SHA256 digital signature for evidence pack"""
        # In production, use proper PKI/HSM
        secret_key = "governance_evidence_signing_key"  # Should be from secure key management
        message = json.dumps(evidence_pack, sort_keys=True).encode()
        signature = hmac.new(secret_key.encode(), message, hashlib.sha256).hexdigest()
        return signature
    
    async def _store_evidence_pack_worm(self, evidence_pack_id: str, evidence_pack: Dict[str, Any]):
        """Store evidence pack in WORM storage (simulated)"""
        # In production, integrate with actual WORM storage system
        self.logger.info(f"📦 Storing evidence pack {evidence_pack_id} in WORM storage")
        
        # Store metadata in database for retrieval
        if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
            try:
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO governance_evidence_packs 
                        (evidence_pack_id, tenant_id, plane, hook_type, decision, 
                         violations_count, created_at, evidence_data)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, evidence_pack_id, evidence_pack["tenant_id"], 
                         evidence_pack["governance_event"]["plane"],
                         evidence_pack["governance_event"]["hook_type"],
                         evidence_pack["governance_event"]["decision"],
                         len(evidence_pack["governance_event"]["violations"]),
                         datetime.now(), json.dumps(evidence_pack))
            except Exception as e:
                self.logger.error(f"Failed to store evidence pack metadata: {e}")
    
    async def _log_override_requirement(self, context: GovernanceContext, result: GovernanceHookResult):
        """Log override requirement to override ledger (Task 8.5.6)"""
        try:
            override_entry = {
                "override_id": str(uuid.uuid4()),
                "tenant_id": context.tenant_id,
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id,
                "governance_hook": context.hook_type.value,
                "violations": result.violations,
                "required_approver_role": "governance_admin",
                "status": "pending_approval",
                "created_at": datetime.now().isoformat(),
                "evidence_pack_id": result.evidence_pack_id
            }
            
            # Store in override ledger
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO governance_override_ledger 
                        (override_id, tenant_id, workflow_id, execution_id, 
                         governance_hook, violations, status, created_at, evidence_pack_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """, override_entry["override_id"], override_entry["tenant_id"],
                         override_entry["workflow_id"], override_entry["execution_id"],
                         override_entry["governance_hook"], json.dumps(override_entry["violations"]),
                         override_entry["status"], datetime.now(), override_entry["evidence_pack_id"])
            
            self.logger.info(f"📝 Override requirement logged: {override_entry['override_id']}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log override requirement: {e}")
    
    async def _log_governance_event(self, context: GovernanceContext, result: GovernanceHookResult):
        """Log governance event for audit trail"""
        try:
            governance_log = {
                "event_id": str(uuid.uuid4()),
                "tenant_id": context.tenant_id,
                "plane": context.plane.value,
                "hook_type": context.hook_type.value,
                "decision": result.decision.value,
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id,
                "user_id": context.user_id,
                "user_role": context.user_role,
                "violations_count": len(result.violations),
                "execution_time_ms": result.execution_time_ms,
                "trust_score_impact": result.trust_score_impact,
                "evidence_pack_id": result.evidence_pack_id,
                "timestamp": context.timestamp.isoformat()
            }
            
            # Store governance event log
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO governance_event_logs 
                        (event_id, tenant_id, plane, hook_type, decision, 
                         workflow_id, execution_id, user_id, user_role,
                         violations_count, execution_time_ms, trust_score_impact,
                         evidence_pack_id, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """, governance_log["event_id"], governance_log["tenant_id"],
                         governance_log["plane"], governance_log["hook_type"],
                         governance_log["decision"], governance_log["workflow_id"],
                         governance_log["execution_id"], governance_log["user_id"],
                         governance_log["user_role"], governance_log["violations_count"],
                         governance_log["execution_time_ms"], governance_log["trust_score_impact"],
                         governance_log["evidence_pack_id"], datetime.now())
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log governance event: {e}")

# Convenience functions for different planes

async def enforce_control_plane_governance(middleware: GovernanceHooksMiddleware, 
                                         tenant_id: int, user_id: str, user_role: str,
                                         workflow_id: str, execution_id: str,
                                         request_data: Dict[str, Any]) -> GovernanceHookResult:
    """Enforce governance for Control Plane (Task 8.5.2)"""
    context = GovernanceContext(
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=user_role,
        workflow_id=workflow_id,
        execution_id=execution_id,
        plane=GovernancePlane.CONTROL,
        hook_type=GovernanceHookType.POLICY,
        request_data=request_data,
        metadata={},
        timestamp=datetime.now()
    )
    return await middleware.enforce_governance(context)

async def enforce_execution_plane_governance(middleware: GovernanceHooksMiddleware,
                                           tenant_id: int, user_id: str, user_role: str,
                                           workflow_id: str, execution_id: str,
                                           request_data: Dict[str, Any]) -> GovernanceHookResult:
    """Enforce governance for Execution Plane (Task 8.5.3)"""
    context = GovernanceContext(
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=user_role,
        workflow_id=workflow_id,
        execution_id=execution_id,
        plane=GovernancePlane.EXECUTION,
        hook_type=GovernanceHookType.TRUST,
        request_data=request_data,
        metadata={},
        timestamp=datetime.now()
    )
    return await middleware.enforce_governance(context)

async def enforce_data_plane_governance(middleware: GovernanceHooksMiddleware,
                                      tenant_id: int, user_id: str, user_role: str,
                                      workflow_id: str, execution_id: str,
                                      request_data: Dict[str, Any]) -> GovernanceHookResult:
    """Enforce governance for Data Plane (Task 8.5.4)"""
    context = GovernanceContext(
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=user_role,
        workflow_id=workflow_id,
        execution_id=execution_id,
        plane=GovernancePlane.DATA,
        hook_type=GovernanceHookType.CONSENT,
        request_data=request_data,
        metadata={},
        timestamp=datetime.now()
    )
    return await middleware.enforce_governance(context)

async def enforce_governance_plane_governance(middleware: GovernanceHooksMiddleware,
                                            tenant_id: int, user_id: str, user_role: str,
                                            workflow_id: str, execution_id: str,
                                            request_data: Dict[str, Any]) -> GovernanceHookResult:
    """Enforce governance for Governance Plane (Task 8.5.5)"""
    context = GovernanceContext(
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=user_role,
        workflow_id=workflow_id,
        execution_id=execution_id,
        plane=GovernancePlane.GOVERNANCE,
        hook_type=GovernanceHookType.LINEAGE,
        request_data=request_data,
        metadata={},
        timestamp=datetime.now()
    )
    return await middleware.enforce_governance(context)
