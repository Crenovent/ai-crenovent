"""
Policy Gate - Deterministic Policy Evaluation Layer
===================================================

Implements fail-closed policy enforcement before routing decisions.
Ensures compliance (GDPR, SOX, HIPAA) without LLM costs.

Based on Vision Doc Chapter 18 - Policy-First Enforcement
"""

import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class PolicyResult(Enum):
    """Policy evaluation results"""
    ALLOWED = "allowed"
    BLOCKED = "blocked" 
    REQUIRES_APPROVAL = "requires_approval"
    DEGRADED_MODE = "degraded_mode"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    RBI = "rbi"
    DPDP = "dpdp"
    CCPA = "ccpa"

@dataclass
class PolicyEvaluation:
    """Result of policy evaluation"""
    result: PolicyResult
    allowed_workflows: Set[str]
    blocked_workflows: Set[str]
    required_approvals: List[str]
    compliance_frameworks: Set[ComplianceFramework]
    evaluation_time_ms: float
    policy_violations: List[str]
    recommendations: List[str]
    tenant_restrictions: Dict[str, Any]

@dataclass
class TenantPolicyProfile:
    """Tenant-specific policy configuration"""
    tenant_id: str
    industry: str
    compliance_frameworks: Set[ComplianceFramework]
    sla_tier: str  # T0, T1, T2
    region: str
    allowed_workflows: Set[str]
    blocked_workflows: Set[str]
    budget_limits: Dict[str, float]
    approval_required_workflows: Set[str]
    data_residency_requirements: List[str]
    created_at: datetime
    last_updated: datetime

class PolicyGate:
    """
    Enterprise policy gate for routing decisions
    
    Features:
    - Deterministic policy evaluation (no LLM costs)
    - Multi-tenant policy enforcement
    - Compliance framework support
    - Fail-closed security model
    - Performance tracking
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Tenant policy profiles (in production, load from database)
        self.tenant_profiles: Dict[str, TenantPolicyProfile] = {}
        
        # Default policy rules
        self.default_policies = self._initialize_default_policies()
        
        # Performance tracking
        self.evaluation_stats = {
            'total_evaluations': 0,
            'blocked_requests': 0,
            'approval_required': 0,
            'avg_evaluation_time_ms': 0.0
        }
        
        self.logger.info("🏛️ Policy Gate initialized with fail-closed enforcement")
    
    async def initialize(self):
        """Initialize policy gate with tenant configurations"""
        try:
            # Load tenant policy profiles from database
            await self._load_tenant_profiles()
            self.logger.info(f"✅ Policy Gate initialized with {len(self.tenant_profiles)} tenant profiles")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load tenant profiles: {e}")
            # Initialize with default profiles for testing
            await self._initialize_default_tenant_profiles()
    
    async def evaluate_policy(
        self,
        user_input: str,
        workflow_category: str,
        tenant_id: str,
        user_id: str = None,
        context: Dict[str, Any] = None
    ) -> PolicyEvaluation:
        """
        Evaluate policy for routing request
        
        Returns:
            PolicyEvaluation with allow/block decision and restrictions
        """
        start_time = time.time()
        self.evaluation_stats['total_evaluations'] += 1
        
        try:
            # Get tenant policy profile
            tenant_profile = self.tenant_profiles.get(
                tenant_id, 
                self._get_default_tenant_profile(tenant_id)
            )
            
            # Initialize evaluation result
            evaluation = PolicyEvaluation(
                result=PolicyResult.ALLOWED,
                allowed_workflows=set(),
                blocked_workflows=set(),
                required_approvals=[],
                compliance_frameworks=tenant_profile.compliance_frameworks,
                evaluation_time_ms=0.0,
                policy_violations=[],
                recommendations=[],
                tenant_restrictions={}
            )
            
            # 1. Check workflow allowlist/blocklist
            if workflow_category in tenant_profile.blocked_workflows:
                evaluation.result = PolicyResult.BLOCKED
                evaluation.blocked_workflows.add(workflow_category)
                evaluation.policy_violations.append(f"Workflow '{workflow_category}' is blocked for tenant {tenant_id}")
                self.evaluation_stats['blocked_requests'] += 1
                
                self.logger.warning(f"🚫 Workflow blocked: {workflow_category} for tenant {tenant_id}")
                return self._finalize_evaluation(evaluation, start_time)
            
            # 2. Check if approval required
            if workflow_category in tenant_profile.approval_required_workflows:
                evaluation.result = PolicyResult.REQUIRES_APPROVAL
                evaluation.required_approvals.append(f"Manager approval required for {workflow_category}")
                self.evaluation_stats['approval_required'] += 1
            
            # 3. Apply compliance framework rules
            compliance_result = self._evaluate_compliance_rules(
                workflow_category, tenant_profile, user_input, context
            )
            
            if compliance_result['blocked']:
                evaluation.result = PolicyResult.BLOCKED
                evaluation.policy_violations.extend(compliance_result['violations'])
                self.evaluation_stats['blocked_requests'] += 1
                
                self.logger.warning(f"🚫 Compliance violation: {compliance_result['violations']}")
                return self._finalize_evaluation(evaluation, start_time)
            
            # 4. Check SLA tier restrictions
            sla_restrictions = self._evaluate_sla_restrictions(tenant_profile, workflow_category)
            if sla_restrictions['degraded']:
                evaluation.result = PolicyResult.DEGRADED_MODE
                evaluation.recommendations.extend(sla_restrictions['recommendations'])
            
            # 5. Check budget limits
            budget_check = self._evaluate_budget_limits(tenant_profile, workflow_category)
            if budget_check['over_limit']:
                evaluation.result = PolicyResult.BLOCKED
                evaluation.policy_violations.append(budget_check['violation'])
                self.evaluation_stats['blocked_requests'] += 1
                
                self.logger.warning(f"🚫 Budget limit exceeded for tenant {tenant_id}")
                return self._finalize_evaluation(evaluation, start_time)
            
            # 6. Set allowed workflows based on profile
            evaluation.allowed_workflows = tenant_profile.allowed_workflows.copy()
            
            # 7. Add tenant-specific restrictions
            evaluation.tenant_restrictions = {
                'sla_tier': tenant_profile.sla_tier,
                'region': tenant_profile.region,
                'industry': tenant_profile.industry,
                'data_residency': tenant_profile.data_residency_requirements
            }
            
            self.logger.info(f"✅ Policy evaluation passed for {workflow_category} (tenant: {tenant_id})")
            
            return self._finalize_evaluation(evaluation, start_time)
            
        except Exception as e:
            self.logger.error(f"❌ Policy evaluation error: {e}")
            # Fail closed - block on error
            return PolicyEvaluation(
                result=PolicyResult.BLOCKED,
                allowed_workflows=set(),
                blocked_workflows={workflow_category},
                required_approvals=[],
                compliance_frameworks=set(),
                evaluation_time_ms=(time.time() - start_time) * 1000,
                policy_violations=[f"Policy evaluation error: {str(e)}"],
                recommendations=["Contact system administrator"],
                tenant_restrictions={}
            )
    
    def _evaluate_compliance_rules(
        self,
        workflow_category: str,
        tenant_profile: TenantPolicyProfile,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate compliance framework rules"""
        
        violations = []
        blocked = False
        
        # GDPR Rules
        if ComplianceFramework.GDPR in tenant_profile.compliance_frameworks:
            if tenant_profile.region == "EU":
                # Check for PII data processing restrictions
                if any(term in user_input.lower() for term in ['personal', 'pii', 'customer data', 'email', 'phone']):
                    if workflow_category not in ['data_quality_audit']:  # Only certain workflows allowed for PII
                        violations.append("GDPR: PII processing restricted to approved workflows")
                        blocked = True
        
        # SOX Rules
        if ComplianceFramework.SOX in tenant_profile.compliance_frameworks:
            # Financial data workflows require additional controls
            if workflow_category in ['pipeline_hygiene_stale_deals', 'risk_scoring_analysis']:
                if tenant_profile.sla_tier not in ['T0', 'T1']:  # Only high-tier tenants
                    violations.append("SOX: Financial workflows require T0/T1 SLA tier")
                    blocked = True
        
        # HIPAA Rules
        if ComplianceFramework.HIPAA in tenant_profile.compliance_frameworks:
            # Healthcare industry restrictions
            if tenant_profile.industry == "Healthcare":
                if any(term in user_input.lower() for term in ['patient', 'medical', 'health']):
                    violations.append("HIPAA: Healthcare data processing requires special approval")
                    blocked = True
        
        return {
            'blocked': blocked,
            'violations': violations
        }
    
    def _evaluate_sla_restrictions(
        self,
        tenant_profile: TenantPolicyProfile,
        workflow_category: str
    ) -> Dict[str, Any]:
        """Evaluate SLA tier restrictions"""
        
        recommendations = []
        degraded = False
        
        # T2 tenants get degraded service
        if tenant_profile.sla_tier == "T2":
            if workflow_category in ['risk_scoring_analysis', 'duplicate_detection']:
                degraded = True
                recommendations.append("T2 SLA: Workflow will use cached results where possible")
        
        return {
            'degraded': degraded,
            'recommendations': recommendations
        }
    
    def _evaluate_budget_limits(
        self,
        tenant_profile: TenantPolicyProfile,
        workflow_category: str
    ) -> Dict[str, Any]:
        """Evaluate budget limits"""
        
        # Simplified budget check (in production, check actual usage)
        monthly_limit = tenant_profile.budget_limits.get('monthly_usd', 1000.0)
        
        # For demo, assume all tenants are within budget
        # In production, query actual usage from billing system
        
        return {
            'over_limit': False,
            'violation': None
        }
    
    def _finalize_evaluation(self, evaluation: PolicyEvaluation, start_time: float) -> PolicyEvaluation:
        """Finalize evaluation with timing and stats"""
        evaluation.evaluation_time_ms = (time.time() - start_time) * 1000
        
        # Update average evaluation time
        current_avg = self.evaluation_stats['avg_evaluation_time_ms']
        total_evals = self.evaluation_stats['total_evaluations']
        new_avg = ((current_avg * (total_evals - 1)) + evaluation.evaluation_time_ms) / total_evals
        self.evaluation_stats['avg_evaluation_time_ms'] = new_avg
        
        return evaluation
    
    def _initialize_default_policies(self) -> Dict[str, Any]:
        """Initialize default policy rules"""
        return {
            'allowed_workflows': {
                'pipeline_hygiene_stale_deals',
                'data_quality_audit',
                'duplicate_detection',
                'ownerless_deals_detection',
                'activity_tracking_audit',
                'risk_scoring_analysis'
            },
            'blocked_workflows': set(),
            'approval_required_workflows': {
                'risk_scoring_analysis'  # Requires approval by default
            }
        }
    
    async def _load_tenant_profiles(self):
        """Load tenant policy profiles from database"""
        # TODO: Implement database loading
        pass
    
    async def _initialize_default_tenant_profiles(self):
        """Initialize default tenant profiles for testing"""
        # Default SaaS tenant
        self.tenant_profiles["1300"] = TenantPolicyProfile(
            tenant_id="1300",
            industry="SaaS",
            compliance_frameworks={ComplianceFramework.SOX, ComplianceFramework.GDPR},
            sla_tier="T1",
            region="US",
            allowed_workflows=self.default_policies['allowed_workflows'],
            blocked_workflows=set(),
            budget_limits={'monthly_usd': 1000.0},
            approval_required_workflows={'risk_scoring_analysis'},
            data_residency_requirements=['US'],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.logger.info("✅ Initialized default tenant policy profiles")
    
    def _get_default_tenant_profile(self, tenant_id: str) -> TenantPolicyProfile:
        """Get default policy profile for unknown tenant"""
        return TenantPolicyProfile(
            tenant_id=tenant_id,
            industry="Unknown",
            compliance_frameworks={ComplianceFramework.GDPR},
            sla_tier="T2",
            region="US",
            allowed_workflows=self.default_policies['allowed_workflows'],
            blocked_workflows=set(),
            budget_limits={'monthly_usd': 100.0},  # Lower limit for unknown tenants
            approval_required_workflows=self.default_policies['approval_required_workflows'],
            data_residency_requirements=['US'],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get policy gate performance statistics"""
        total = self.evaluation_stats['total_evaluations']
        blocked = self.evaluation_stats['blocked_requests']
        approval_required = self.evaluation_stats['approval_required']
        
        block_rate = (blocked / total * 100) if total > 0 else 0
        approval_rate = (approval_required / total * 100) if total > 0 else 0
        
        return {
            'total_evaluations': total,
            'blocked_requests': blocked,
            'approval_required_requests': approval_required,
            'block_rate_percent': round(block_rate, 2),
            'approval_rate_percent': round(approval_rate, 2),
            'avg_evaluation_time_ms': round(self.evaluation_stats['avg_evaluation_time_ms'], 3),
            'tenant_profiles_loaded': len(self.tenant_profiles)
        }
    
    async def add_tenant_profile(self, profile: TenantPolicyProfile):
        """Add or update tenant policy profile"""
        self.tenant_profiles[profile.tenant_id] = profile
        self.logger.info(f"✅ Added/updated policy profile for tenant: {profile.tenant_id}")
    
    async def get_tenant_profile(self, tenant_id: str) -> Optional[TenantPolicyProfile]:
        """Get tenant policy profile"""
        return self.tenant_profiles.get(tenant_id)
