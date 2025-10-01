"""
Enhanced Policy Manager - Tasks 16.1.1, 16.1.2, 16.1.3
Implements comprehensive policy metadata management with version lifecycle and state machine.

Features:
- Policy metadata storage with industry/SLA context (Task 16.1.1)
- Policy version tracking and lifecycle management (Task 16.1.2)
- State machine: Draft → Published → Promoted → Retired (Task 16.1.3)
- Full tenant isolation and dynamic configuration
- SaaS-focused with extensibility for other industries
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

class PolicyStatus(Enum):
    """Policy lifecycle states - Task 16.1.3"""
    DRAFT = "draft"
    PUBLISHED = "published"
    PROMOTED = "promoted"
    RETIRED = "retired"

class PolicyType(Enum):
    """Configurable policy types with SaaS focus"""
    SOX_SAAS = "SOX_SAAS"
    GDPR_SAAS = "GDPR_SAAS"
    SAAS_BUSINESS_RULES = "SAAS_BUSINESS_RULES"
    # Future extensibility
    SOX_BANKING = "SOX_BANKING"
    RBI_BANKING = "RBI_BANKING"
    HIPAA_INSURANCE = "HIPAA_INSURANCE"
    NAIC_INSURANCE = "NAIC_INSURANCE"

class EnforcementLevel(Enum):
    """Dynamic enforcement levels"""
    STRICT = "strict"
    ADVISORY = "advisory"
    DISABLED = "disabled"

class IndustryCode(Enum):
    """Configurable industry codes"""
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    FINTECH = "FinTech"
    ECOMMERCE = "E-commerce"
    IT_SERVICES = "IT_Services"

class TenantTier(Enum):
    """Dynamic tenant tier classification"""
    T0 = "T0"  # Regulated/Enterprise
    T1 = "T1"  # Mid-market
    T2 = "T2"  # SMB

@dataclass
class PolicyMetadata:
    """Policy metadata structure - Task 16.1.1"""
    policy_id: str
    tenant_id: int
    policy_name: str
    industry_code: IndustryCode
    region_code: str
    policy_type: PolicyType
    
    # Version and lifecycle - Task 16.1.2
    version: str
    status: PolicyStatus
    
    # Policy content
    rego_rules: str
    enforcement_level: EnforcementLevel
    
    # SaaS-specific metadata
    saas_metrics: List[str] = None
    tenant_tier: TenantTier = TenantTier.T2
    
    # Governance
    created_by_user_id: int = None
    approved_by_user_id: int = None
    approval_timestamp: Optional[datetime] = None
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    published_at: Optional[datetime] = None
    promoted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.saas_metrics is None:
            self.saas_metrics = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PolicyVersionHistory:
    """Policy version history tracking - Task 16.1.2"""
    version_id: str
    policy_id: str
    tenant_id: int
    version: str
    previous_version: Optional[str]
    status: PolicyStatus
    changes_summary: str
    changed_by_user_id: int
    change_timestamp: datetime
    change_reason: str
    rollback_available: bool = True

@dataclass
class PolicyStateTransition:
    """Policy state transition tracking - Task 16.1.3"""
    transition_id: str
    policy_id: str
    tenant_id: int
    from_status: PolicyStatus
    to_status: PolicyStatus
    transition_timestamp: datetime
    triggered_by_user_id: int
    approval_required: bool
    approval_granted: bool = False
    approval_by_user_id: Optional[int] = None
    transition_reason: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EnhancedPolicyManager:
    """
    Enhanced Policy Manager with full lifecycle management and tenant isolation.
    Implements Tasks 16.1.1, 16.1.2, 16.1.3.
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Dynamic configuration (no hardcoding)
        self.policy_templates = self._load_policy_templates()
        self.state_transition_rules = self._load_state_transition_rules()
        self.approval_requirements = self._load_approval_requirements()
        self.industry_policy_mapping = self._load_industry_policy_mapping()
        
        # Cache for performance
        self.policy_cache: Dict[str, PolicyMetadata] = {}
        self.version_cache: Dict[str, List[PolicyVersionHistory]] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Tenant isolation
        self.tenant_policies: Dict[int, List[str]] = {}
        
    def _load_policy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load configurable policy templates"""
        return {
            'SOX_SAAS': {
                'name': 'SOX Financial Controls for SaaS',
                'description': 'Sarbanes-Oxley compliance for SaaS companies',
                'default_rego': '''
package sox_saas
default allow = false

allow {
    saas_financial_controls
    revenue_recognition_controls
    subscription_audit_trail
    segregation_of_duties
}

saas_financial_controls {
    input.data_type in ["arr", "mrr", "revenue", "subscription", "billing"]
    input.has_approval == true
    input.tenant_id != null
    input.approver_role in ["cfo", "controller", "finance_manager", "revenue_ops"]
}

revenue_recognition_controls {
    input.subscription_start_date != null
    input.revenue_recognition_method in ["monthly", "annual", "usage_based"]
    input.contract_terms_validated == true
}

subscription_audit_trail {
    input.audit_log == true
    input.tenant_id != null
    input.action_type in ["create_subscription", "modify_subscription", "cancel_subscription"]
}

segregation_of_duties {
    input.creator_id != input.approver_id
    count(input.approval_chain) >= 2
    not (input.creator_role == "sales" and input.approver_role == "sales")
}
''',
                'saas_metrics': ['ARR', 'MRR', 'Revenue', 'Subscriptions'],
                'enforcement_level': 'strict',
                'approval_required': True
            },
            'GDPR_SAAS': {
                'name': 'GDPR Data Protection for SaaS',
                'description': 'GDPR compliance for SaaS customer data',
                'default_rego': '''
package gdpr_saas
default allow = false

allow {
    saas_consent_validation
    saas_data_minimization
    multi_tenant_isolation
}

saas_consent_validation {
    input.data_type in ["customer_data", "usage_data", "billing_data"]
    input.consent_given == true
    input.tenant_id != null
    input.subscription_active == true
}

saas_data_minimization {
    input.service_delivery_required == true
    count(input.data_fields) <= input.max_allowed_fields
}

multi_tenant_isolation {
    input.tenant_id != null
    input.cross_tenant_access == false
}
''',
                'saas_metrics': ['CustomerData', 'UsageData', 'ConsentStatus'],
                'enforcement_level': 'strict',
                'approval_required': True
            },
            'SAAS_BUSINESS_RULES': {
                'name': 'SaaS Business Rules',
                'description': 'SaaS-specific business logic and operational policies',
                'default_rego': '''
package saas_business_rules
default allow = false

allow {
    subscription_lifecycle_controls
    churn_prevention_controls
    usage_based_billing_controls
}

subscription_lifecycle_controls {
    input.operation_type in ["create_subscription", "modify_subscription", "cancel_subscription"]
    input.tenant_id != null
    input.customer_verified == true
}

churn_prevention_controls {
    input.customer_health_score >= 0.3
    input.usage_trend_monitored == true
}

usage_based_billing_controls {
    input.billing_calculation_verified == true
    input.usage_metering_accurate == true
}
''',
                'saas_metrics': ['SubscriptionHealth', 'CustomerSuccess', 'UsageBilling'],
                'enforcement_level': 'advisory',
                'approval_required': False
            }
        }
    
    def _load_state_transition_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load state transition rules - Task 16.1.3"""
        return {
            'draft_to_published': {
                'from': PolicyStatus.DRAFT,
                'to': PolicyStatus.PUBLISHED,
                'approval_required': True,
                'required_roles': ['policy_admin', 'compliance_officer'],
                'validation_checks': ['syntax_valid', 'policy_complete', 'test_passed']
            },
            'published_to_promoted': {
                'from': PolicyStatus.PUBLISHED,
                'to': PolicyStatus.PROMOTED,
                'approval_required': True,
                'required_roles': ['policy_admin', 'compliance_officer', 'cro'],
                'validation_checks': ['production_tested', 'business_approved', 'compliance_verified']
            },
            'promoted_to_retired': {
                'from': PolicyStatus.PROMOTED,
                'to': PolicyStatus.RETIRED,
                'approval_required': True,
                'required_roles': ['policy_admin', 'compliance_officer'],
                'validation_checks': ['replacement_ready', 'migration_plan']
            },
            'any_to_draft': {
                'from': 'any',
                'to': PolicyStatus.DRAFT,
                'approval_required': False,
                'required_roles': ['policy_admin'],
                'validation_checks': []
            }
        }
    
    def _load_approval_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Load approval requirements by tenant tier and policy type"""
        return {
            'T0': {  # Regulated tenants - strictest approvals
                'SOX_SAAS': {'approvers_required': 3, 'roles': ['cfo', 'compliance_officer', 'policy_admin']},
                'GDPR_SAAS': {'approvers_required': 2, 'roles': ['dpo', 'compliance_officer']},
                'SAAS_BUSINESS_RULES': {'approvers_required': 2, 'roles': ['cro', 'policy_admin']}
            },
            'T1': {  # Enterprise tenants - balanced approvals
                'SOX_SAAS': {'approvers_required': 2, 'roles': ['finance_manager', 'compliance_officer']},
                'GDPR_SAAS': {'approvers_required': 2, 'roles': ['dpo', 'compliance_officer']},
                'SAAS_BUSINESS_RULES': {'approvers_required': 1, 'roles': ['revenue_ops']}
            },
            'T2': {  # Mid-market tenants - streamlined approvals
                'SOX_SAAS': {'approvers_required': 1, 'roles': ['finance_manager']},
                'GDPR_SAAS': {'approvers_required': 1, 'roles': ['compliance_officer']},
                'SAAS_BUSINESS_RULES': {'approvers_required': 1, 'roles': ['revenue_ops']}
            }
        }
    
    def _load_industry_policy_mapping(self) -> Dict[str, List[str]]:
        """Load industry-specific policy requirements"""
        return {
            'SaaS': ['SOX_SAAS', 'GDPR_SAAS', 'SAAS_BUSINESS_RULES'],
            'Banking': ['SOX_BANKING', 'RBI_BANKING', 'GDPR_SAAS'],
            'Insurance': ['HIPAA_INSURANCE', 'NAIC_INSURANCE', 'GDPR_SAAS'],
            'FinTech': ['SOX_SAAS', 'GDPR_SAAS', 'RBI_BANKING'],
            'E-commerce': ['GDPR_SAAS', 'SAAS_BUSINESS_RULES'],
            'IT_Services': ['SOX_SAAS', 'GDPR_SAAS', 'SAAS_BUSINESS_RULES']
        }
    
    async def initialize(self):
        """Initialize the enhanced policy manager"""
        try:
            await self._ensure_database_tables()
            await self._load_existing_policies()
            self.logger.info("Enhanced Policy Manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Policy Manager: {e}")
            raise
    
    async def _ensure_database_tables(self):
        """Ensure all required database tables exist"""
        if not self.pool_manager:
            return
            
        async with self.pool_manager.get_connection() as conn:
            # Enhanced policy metadata table - Task 16.1.1
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_policy_metadata (
                    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    policy_name VARCHAR(255) NOT NULL,
                    industry_code VARCHAR(20) NOT NULL,
                    region_code VARCHAR(10) NOT NULL DEFAULT 'US',
                    policy_type VARCHAR(50) NOT NULL,
                    
                    -- Version and lifecycle
                    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
                    status VARCHAR(20) NOT NULL DEFAULT 'draft',
                    
                    -- Policy content
                    rego_rules TEXT NOT NULL,
                    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'strict',
                    
                    -- SaaS-specific metadata
                    saas_metrics JSONB DEFAULT '[]',
                    tenant_tier VARCHAR(10) DEFAULT 'T2',
                    
                    -- Governance
                    created_by_user_id INTEGER,
                    approved_by_user_id INTEGER,
                    approval_timestamp TIMESTAMPTZ,
                    
                    -- Metadata
                    description TEXT,
                    metadata JSONB DEFAULT '{}',
                    
                    -- Timestamps
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    published_at TIMESTAMPTZ,
                    promoted_at TIMESTAMPTZ,
                    retired_at TIMESTAMPTZ,
                    
                    CONSTRAINT unique_enhanced_policy_per_tenant UNIQUE (tenant_id, policy_name, version),
                    CONSTRAINT valid_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'FinTech', 'E-commerce', 'IT_Services')),
                    CONSTRAINT valid_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2')),
                    CONSTRAINT valid_status CHECK (status IN ('draft', 'published', 'promoted', 'retired')),
                    CONSTRAINT valid_enforcement_level CHECK (enforcement_level IN ('strict', 'advisory', 'disabled'))
                );
            """)
            
            # Policy version history table - Task 16.1.2
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_version_history (
                    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    policy_id UUID NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    previous_version VARCHAR(20),
                    status VARCHAR(20) NOT NULL,
                    changes_summary TEXT NOT NULL,
                    changed_by_user_id INTEGER NOT NULL,
                    change_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    change_reason TEXT,
                    rollback_available BOOLEAN DEFAULT true,
                    
                    FOREIGN KEY (policy_id) REFERENCES enhanced_policy_metadata(policy_id)
                );
            """)
            
            # Policy state transitions table - Task 16.1.3
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_state_transitions (
                    transition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    policy_id UUID NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    from_status VARCHAR(20) NOT NULL,
                    to_status VARCHAR(20) NOT NULL,
                    transition_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    triggered_by_user_id INTEGER NOT NULL,
                    approval_required BOOLEAN NOT NULL DEFAULT false,
                    approval_granted BOOLEAN DEFAULT false,
                    approval_by_user_id INTEGER,
                    transition_reason TEXT,
                    metadata JSONB DEFAULT '{}',
                    
                    FOREIGN KEY (policy_id) REFERENCES enhanced_policy_metadata(policy_id),
                    CONSTRAINT valid_from_status CHECK (from_status IN ('draft', 'published', 'promoted', 'retired')),
                    CONSTRAINT valid_to_status CHECK (to_status IN ('draft', 'published', 'promoted', 'retired'))
                );
            """)
            
            # Enable RLS on all tables
            for table in ['enhanced_policy_metadata', 'policy_version_history', 'policy_state_transitions']:
                await conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
                await conn.execute(f"""
                    DROP POLICY IF EXISTS {table}_rls_policy ON {table};
                    CREATE POLICY {table}_rls_policy ON {table}
                        FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
                """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_policy_tenant_id ON enhanced_policy_metadata(tenant_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_policy_type ON enhanced_policy_metadata(policy_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_policy_status ON enhanced_policy_metadata(status);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_version_policy_id ON policy_version_history(policy_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_transitions_policy_id ON policy_state_transitions(policy_id);")
    
    async def create_policy(self, policy_metadata: PolicyMetadata) -> bool:
        """Create a new policy - Task 16.1.1"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                self.policy_cache[policy_metadata.policy_id] = policy_metadata
                self._update_tenant_mapping(policy_metadata.tenant_id, policy_metadata.policy_id)
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{policy_metadata.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO enhanced_policy_metadata (
                        policy_id, tenant_id, policy_name, industry_code, region_code,
                        policy_type, version, status, rego_rules, enforcement_level,
                        saas_metrics, tenant_tier, created_by_user_id, description, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, 
                policy_metadata.policy_id, policy_metadata.tenant_id, policy_metadata.policy_name,
                policy_metadata.industry_code.value, policy_metadata.region_code,
                policy_metadata.policy_type.value, policy_metadata.version,
                policy_metadata.status.value, policy_metadata.rego_rules,
                policy_metadata.enforcement_level.value, json.dumps(policy_metadata.saas_metrics),
                policy_metadata.tenant_tier.value, policy_metadata.created_by_user_id,
                policy_metadata.description, json.dumps(policy_metadata.metadata))
            
            # Create initial version history record
            await self._create_version_history(
                policy_metadata.policy_id, policy_metadata.tenant_id, policy_metadata.version,
                None, policy_metadata.status, "Initial policy creation",
                policy_metadata.created_by_user_id, "Policy created"
            )
            
            # Update cache and tenant mapping
            self.policy_cache[policy_metadata.policy_id] = policy_metadata
            self._update_tenant_mapping(policy_metadata.tenant_id, policy_metadata.policy_id)
            
            self.logger.info(f"Created policy {policy_metadata.policy_name} for tenant {policy_metadata.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create policy {policy_metadata.policy_name}: {e}")
            return False
    
    async def update_policy_version(self, policy_id: str, tenant_id: int, new_version: str,
                                  changes_summary: str, changed_by_user_id: int,
                                  change_reason: str = "") -> bool:
        """Update policy version - Task 16.1.2"""
        try:
            if not self.pool_manager:
                # Update cache for testing
                if policy_id in self.policy_cache:
                    policy = self.policy_cache[policy_id]
                    old_version = policy.version
                    policy.version = new_version
                    policy.updated_at = datetime.utcnow()
                    # Create mock version history
                    return True
                return False
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Get current version
                current_policy = await conn.fetchrow(
                    "SELECT version FROM enhanced_policy_metadata WHERE policy_id = $1 AND tenant_id = $2",
                    policy_id, tenant_id
                )
                
                if not current_policy:
                    self.logger.error(f"Policy {policy_id} not found for tenant {tenant_id}")
                    return False
                
                old_version = current_policy['version']
                
                # Update policy version
                await conn.execute("""
                    UPDATE enhanced_policy_metadata 
                    SET version = $1, updated_at = NOW()
                    WHERE policy_id = $2 AND tenant_id = $3
                """, new_version, policy_id, tenant_id)
                
                # Create version history record
                await self._create_version_history(
                    policy_id, tenant_id, new_version, old_version,
                    PolicyStatus.DRAFT, changes_summary, changed_by_user_id, change_reason
                )
                
                # Update cache
                if policy_id in self.policy_cache:
                    self.policy_cache[policy_id].version = new_version
                    self.policy_cache[policy_id].updated_at = datetime.utcnow()
            
            self.logger.info(f"Updated policy {policy_id} to version {new_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update policy version {policy_id}: {e}")
            return False
    
    async def transition_policy_status(self, policy_id: str, tenant_id: int, 
                                     to_status: PolicyStatus, triggered_by_user_id: int,
                                     transition_reason: str = "") -> bool:
        """Transition policy status through state machine - Task 16.1.3"""
        try:
            if not self.pool_manager:
                # Update cache for testing
                if policy_id in self.policy_cache:
                    policy = self.policy_cache[policy_id]
                    from_status = policy.status
                    policy.status = to_status
                    policy.updated_at = datetime.utcnow()
                    
                    # Set status-specific timestamps
                    if to_status == PolicyStatus.PUBLISHED:
                        policy.published_at = datetime.utcnow()
                    elif to_status == PolicyStatus.PROMOTED:
                        policy.promoted_at = datetime.utcnow()
                    elif to_status == PolicyStatus.RETIRED:
                        policy.retired_at = datetime.utcnow()
                    
                    return True
                return False
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Get current policy status
                current_policy = await conn.fetchrow("""
                    SELECT status, tenant_tier, policy_type FROM enhanced_policy_metadata 
                    WHERE policy_id = $1 AND tenant_id = $2
                """, policy_id, tenant_id)
                
                if not current_policy:
                    self.logger.error(f"Policy {policy_id} not found for tenant {tenant_id}")
                    return False
                
                from_status = PolicyStatus(current_policy['status'])
                tenant_tier = current_policy['tenant_tier']
                policy_type = current_policy['policy_type']
                
                # Validate state transition
                if not self._is_valid_transition(from_status, to_status):
                    self.logger.error(f"Invalid transition from {from_status.value} to {to_status.value}")
                    return False
                
                # Check approval requirements
                approval_required = self._requires_approval(from_status, to_status, tenant_tier, policy_type)
                
                # Update policy status and timestamps
                update_fields = ["status = $1", "updated_at = NOW()"]
                params = [to_status.value]
                param_count = 2
                
                if to_status == PolicyStatus.PUBLISHED:
                    update_fields.append(f"published_at = NOW()")
                elif to_status == PolicyStatus.PROMOTED:
                    update_fields.append(f"promoted_at = NOW()")
                elif to_status == PolicyStatus.RETIRED:
                    update_fields.append(f"retired_at = NOW()")
                
                update_query = f"""
                    UPDATE enhanced_policy_metadata 
                    SET {', '.join(update_fields)}
                    WHERE policy_id = ${param_count} AND tenant_id = ${param_count + 1}
                """
                params.extend([policy_id, tenant_id])
                
                await conn.execute(update_query, *params)
                
                # Record state transition
                await self._record_state_transition(
                    policy_id, tenant_id, from_status, to_status,
                    triggered_by_user_id, approval_required, transition_reason
                )
                
                # Update cache
                if policy_id in self.policy_cache:
                    policy = self.policy_cache[policy_id]
                    policy.status = to_status
                    policy.updated_at = datetime.utcnow()
                    
                    if to_status == PolicyStatus.PUBLISHED:
                        policy.published_at = datetime.utcnow()
                    elif to_status == PolicyStatus.PROMOTED:
                        policy.promoted_at = datetime.utcnow()
                    elif to_status == PolicyStatus.RETIRED:
                        policy.retired_at = datetime.utcnow()
            
            self.logger.info(f"Transitioned policy {policy_id} from {from_status.value} to {to_status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to transition policy status {policy_id}: {e}")
            return False
    
    def _is_valid_transition(self, from_status: PolicyStatus, to_status: PolicyStatus) -> bool:
        """Validate state transition according to state machine rules"""
        valid_transitions = {
            PolicyStatus.DRAFT: [PolicyStatus.PUBLISHED, PolicyStatus.RETIRED],
            PolicyStatus.PUBLISHED: [PolicyStatus.PROMOTED, PolicyStatus.DRAFT, PolicyStatus.RETIRED],
            PolicyStatus.PROMOTED: [PolicyStatus.RETIRED, PolicyStatus.DRAFT],
            PolicyStatus.RETIRED: [PolicyStatus.DRAFT]
        }
        
        return to_status in valid_transitions.get(from_status, [])
    
    def _requires_approval(self, from_status: PolicyStatus, to_status: PolicyStatus,
                          tenant_tier: str, policy_type: str) -> bool:
        """Check if state transition requires approval"""
        # High-impact transitions always require approval
        high_impact_transitions = [
            (PolicyStatus.DRAFT, PolicyStatus.PUBLISHED),
            (PolicyStatus.PUBLISHED, PolicyStatus.PROMOTED),
            (PolicyStatus.PROMOTED, PolicyStatus.RETIRED)
        ]
        
        if (from_status, to_status) in high_impact_transitions:
            return True
        
        # Check tenant tier and policy type specific requirements
        requirements = self.approval_requirements.get(tenant_tier, {}).get(policy_type, {})
        return requirements.get('approvers_required', 0) > 0
    
    async def _create_version_history(self, policy_id: str, tenant_id: int, version: str,
                                    previous_version: Optional[str], status: PolicyStatus,
                                    changes_summary: str, changed_by_user_id: int,
                                    change_reason: str):
        """Create version history record - Task 16.1.2"""
        if not self.pool_manager:
            return
            
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO policy_version_history (
                        policy_id, tenant_id, version, previous_version, status,
                        changes_summary, changed_by_user_id, change_reason
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, policy_id, tenant_id, version, previous_version, status.value,
                changes_summary, changed_by_user_id, change_reason)
                
        except Exception as e:
            self.logger.error(f"Failed to create version history for {policy_id}: {e}")
    
    async def _record_state_transition(self, policy_id: str, tenant_id: int,
                                     from_status: PolicyStatus, to_status: PolicyStatus,
                                     triggered_by_user_id: int, approval_required: bool,
                                     transition_reason: str):
        """Record state transition - Task 16.1.3"""
        if not self.pool_manager:
            return
            
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO policy_state_transitions (
                        policy_id, tenant_id, from_status, to_status,
                        triggered_by_user_id, approval_required, transition_reason
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, policy_id, tenant_id, from_status.value, to_status.value,
                triggered_by_user_id, approval_required, transition_reason)
                
        except Exception as e:
            self.logger.error(f"Failed to record state transition for {policy_id}: {e}")
    
    async def get_policy_by_id(self, policy_id: str, tenant_id: int) -> Optional[PolicyMetadata]:
        """Get policy by ID with tenant isolation"""
        try:
            if not self.pool_manager:
                return self.policy_cache.get(policy_id)
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                row = await conn.fetchrow("""
                    SELECT * FROM enhanced_policy_metadata 
                    WHERE policy_id = $1 AND tenant_id = $2
                """, policy_id, tenant_id)
                
                if row:
                    policy = self._row_to_policy_metadata(row)
                    self.policy_cache[policy_id] = policy
                    return policy
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get policy {policy_id}: {e}")
            return None
    
    async def get_tenant_policies(self, tenant_id: int, status_filter: Optional[PolicyStatus] = None,
                                policy_type_filter: Optional[PolicyType] = None) -> List[PolicyMetadata]:
        """Get all policies for a tenant with optional filters"""
        try:
            if not self.pool_manager:
                # Return from cache for testing
                tenant_policies = []
                for policy_id in self.tenant_policies.get(tenant_id, []):
                    if policy_id in self.policy_cache:
                        policy = self.policy_cache[policy_id]
                        if status_filter and policy.status != status_filter:
                            continue
                        if policy_type_filter and policy.policy_type != policy_type_filter:
                            continue
                        tenant_policies.append(policy)
                return tenant_policies
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                query = "SELECT * FROM enhanced_policy_metadata WHERE tenant_id = $1"
                params = [tenant_id]
                
                if status_filter:
                    query += " AND status = $2"
                    params.append(status_filter.value)
                
                if policy_type_filter:
                    query += f" AND policy_type = ${len(params) + 1}"
                    params.append(policy_type_filter.value)
                
                rows = await conn.fetch(query, *params)
                
                policies = []
                for row in rows:
                    policy = self._row_to_policy_metadata(row)
                    policies.append(policy)
                    # Update cache
                    self.policy_cache[policy.policy_id] = policy
                
                return policies
                
        except Exception as e:
            self.logger.error(f"Failed to get tenant policies for {tenant_id}: {e}")
            return []
    
    async def create_saas_policy_from_template(self, tenant_id: int, tenant_tier: TenantTier,
                                             policy_type: PolicyType, created_by_user_id: int,
                                             customizations: Optional[Dict[str, Any]] = None) -> Optional[PolicyMetadata]:
        """Create SaaS policy from template"""
        try:
            if policy_type.value not in self.policy_templates:
                self.logger.error(f"Unknown policy template: {policy_type.value}")
                return None
            
            template = self.policy_templates[policy_type.value]
            
            # Generate unique policy ID
            policy_id = str(uuid.uuid4())
            
            # Apply customizations to template
            rego_rules = template['default_rego']
            saas_metrics = template['saas_metrics'].copy()
            enforcement_level = EnforcementLevel(template['enforcement_level'])
            
            if customizations:
                if 'rego_rules' in customizations:
                    rego_rules = customizations['rego_rules']
                if 'saas_metrics' in customizations:
                    saas_metrics.extend(customizations['saas_metrics'])
                if 'enforcement_level' in customizations:
                    enforcement_level = EnforcementLevel(customizations['enforcement_level'])
            
            # Create policy metadata
            policy_metadata = PolicyMetadata(
                policy_id=policy_id,
                tenant_id=tenant_id,
                policy_name=template['name'],
                industry_code=IndustryCode.SAAS,
                region_code="US",  # Default, can be customized
                policy_type=policy_type,
                version="1.0.0",
                status=PolicyStatus.DRAFT,
                rego_rules=rego_rules,
                enforcement_level=enforcement_level,
                saas_metrics=saas_metrics,
                tenant_tier=tenant_tier,
                created_by_user_id=created_by_user_id,
                description=template['description']
            )
            
            # Create the policy
            success = await self.create_policy(policy_metadata)
            if success:
                self.logger.info(f"Created SaaS policy from template: {policy_type.value} for tenant {tenant_id}")
                return policy_metadata
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create SaaS policy from template {policy_type.value}: {e}")
            return None
    
    def _update_tenant_mapping(self, tenant_id: int, policy_id: str):
        """Update tenant-to-policy mapping"""
        if tenant_id not in self.tenant_policies:
            self.tenant_policies[tenant_id] = []
        
        if policy_id not in self.tenant_policies[tenant_id]:
            self.tenant_policies[tenant_id].append(policy_id)
    
    async def _load_existing_policies(self):
        """Load existing policies from database into cache"""
        if not self.pool_manager:
            return
            
        try:
            async with self.pool_manager.get_connection() as conn:
                rows = await conn.fetch("SELECT * FROM enhanced_policy_metadata LIMIT 1000")
                
                for row in rows:
                    policy = self._row_to_policy_metadata(row)
                    self.policy_cache[policy.policy_id] = policy
                    self._update_tenant_mapping(policy.tenant_id, policy.policy_id)
                
                self.logger.info(f"Loaded {len(rows)} existing policies into cache")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing policies: {e}")
    
    def _row_to_policy_metadata(self, row) -> PolicyMetadata:
        """Convert database row to PolicyMetadata object"""
        return PolicyMetadata(
            policy_id=str(row['policy_id']),
            tenant_id=row['tenant_id'],
            policy_name=row['policy_name'],
            industry_code=IndustryCode(row['industry_code']),
            region_code=row['region_code'],
            policy_type=PolicyType(row['policy_type']),
            version=row['version'],
            status=PolicyStatus(row['status']),
            rego_rules=row['rego_rules'],
            enforcement_level=EnforcementLevel(row['enforcement_level']),
            saas_metrics=json.loads(row['saas_metrics']) if row['saas_metrics'] else [],
            tenant_tier=TenantTier(row['tenant_tier']),
            created_by_user_id=row['created_by_user_id'],
            approved_by_user_id=row['approved_by_user_id'],
            approval_timestamp=row['approval_timestamp'],
            description=row['description'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            published_at=row['published_at'],
            promoted_at=row['promoted_at'],
            retired_at=row['retired_at']
        )

# Singleton instance for global access
_enhanced_policy_manager = None

def get_enhanced_policy_manager(pool_manager=None) -> EnhancedPolicyManager:
    """Get singleton instance of Enhanced Policy Manager"""
    global _enhanced_policy_manager
    if _enhanced_policy_manager is None:
        _enhanced_policy_manager = EnhancedPolicyManager(pool_manager)
    return _enhanced_policy_manager
