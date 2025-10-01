#!/usr/bin/env python3
"""
Multi-Tenant Enforcement Taxonomy - Task 9.4.1
===============================================

Implements Build Plan Epic 1 Chapter 9.4.1 - Define multi-tenant enforcement taxonomy
with shared language across teams for isolation, residency, FinOps, and SoD.

This serves as the foundational framework for all multi-tenant enforcement tasks.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid


# =============================================================================
# CORE TAXONOMY ENUMERATIONS
# =============================================================================

class TenantTier(Enum):
    """SLA Tiers as defined in Vision Doc Chapter 4.2"""
    T0_REGULATED = "T0_REGULATED"      # Banking, Insurance, Healthcare (strict compliance)
    T1_ENTERPRISE = "T1_ENTERPRISE"    # Large enterprise with custom SLAs
    T2_MIDMARKET = "T2_MIDMARKET"      # Standard SaaS offering

class IsolationLevel(Enum):
    """Data isolation enforcement levels"""
    SHARED_SCHEMA_RLS = "SHARED_SCHEMA_RLS"        # Default: Shared schema + Row Level Security
    SCHEMA_PER_TENANT = "SCHEMA_PER_TENANT"        # Escalation: Dedicated schema per tenant
    DATABASE_PER_TENANT = "DATABASE_PER_TENANT"    # Maximum: Dedicated database per tenant

class ResidencyRequirement(Enum):
    """Data residency enforcement levels"""
    GLOBAL = "GLOBAL"                    # No residency restrictions
    REGION_PREFERRED = "REGION_PREFERRED" # Prefer region but allow failover
    REGION_STRICT = "REGION_STRICT"      # Must stay in region, fail-closed
    COUNTRY_STRICT = "COUNTRY_STRICT"    # Must stay in country boundaries

class ComplianceFramework(Enum):
    """Regulatory compliance frameworks"""
    GDPR = "GDPR"           # European data protection
    SOX = "SOX"             # Sarbanes-Oxley (US financial)
    HIPAA = "HIPAA"         # US healthcare
    RBI = "RBI"             # Reserve Bank of India
    DPDP = "DPDP"           # Digital Personal Data Protection (India)
    IRDAI = "IRDAI"         # Insurance Regulatory Authority (India)
    NAIC = "NAIC"           # National Association Insurance Commissioners
    PCI_DSS = "PCI_DSS"     # Payment card industry

class SeparationOfDuties(Enum):
    """Segregation of duties enforcement levels"""
    BASIC = "BASIC"         # Role-based separation
    STRICT = "STRICT"       # Dual authorization required
    REGULATED = "REGULATED" # Full audit trail + approval chains

class FinOpsScope(Enum):
    """Financial operations attribution levels"""
    TENANT_LEVEL = "TENANT_LEVEL"       # Cost attribution per tenant
    WORKLOAD_LEVEL = "WORKLOAD_LEVEL"   # Cost per workflow/automation
    RESOURCE_LEVEL = "RESOURCE_LEVEL"   # Granular compute/storage costs


# =============================================================================
# TAXONOMY DATA MODELS
# =============================================================================

@dataclass
class TenantContext:
    """Complete tenant context for enforcement decisions"""
    tenant_id: int
    tenant_name: str
    tenant_tier: TenantTier
    isolation_level: IsolationLevel
    residency_requirement: ResidencyRequirement
    primary_region: str
    allowed_regions: Set[str]
    compliance_frameworks: Set[ComplianceFramework]
    separation_of_duties: SeparationOfDuties
    finops_scope: FinOpsScope
    
    # Governance metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_validated: Optional[datetime] = None
    validation_status: str = "pending"
    
    # Enforcement flags
    fail_closed_enabled: bool = True
    audit_trail_required: bool = True
    evidence_pack_required: bool = True

@dataclass
class IsolationPolicy:
    """Isolation enforcement policy definition"""
    policy_id: str
    policy_name: str
    tenant_tier: TenantTier
    isolation_level: IsolationLevel
    
    # Database isolation rules
    rls_required: bool = True
    schema_isolation: bool = False
    database_isolation: bool = False
    
    # Query enforcement rules
    tenant_id_mandatory: bool = True
    cross_tenant_joins_blocked: bool = True
    raw_table_access_blocked: bool = True
    
    # Cache isolation rules
    cache_key_prefixing: bool = True
    cache_ttl_per_tier: Dict[TenantTier, int] = field(default_factory=dict)
    
    # Vector search isolation
    vector_index_partitioning: bool = True
    embedding_tenant_filtering: bool = True

@dataclass
class ResidencyPolicy:
    """Data residency enforcement policy"""
    policy_id: str
    policy_name: str
    residency_requirement: ResidencyRequirement
    compliance_frameworks: Set[ComplianceFramework]
    
    # Region enforcement
    allowed_regions: Set[str]
    primary_region: str
    failover_allowed: bool = False
    
    # Data movement restrictions
    cross_region_replication_blocked: bool = True
    backup_region_restricted: bool = True
    
    # Processing restrictions
    compute_region_restricted: bool = True
    model_inference_region_restricted: bool = True

@dataclass
class GovernancePolicy:
    """Governance and audit enforcement policy"""
    policy_id: str
    policy_name: str
    separation_of_duties: SeparationOfDuties
    finops_scope: FinOpsScope
    
    # Audit requirements
    evidence_pack_mandatory: bool = True
    override_ledger_mandatory: bool = True
    execution_trace_mandatory: bool = True
    
    # Approval workflows
    dual_authorization_required: bool = False
    approval_chain_length: int = 1
    
    # Retention policies
    audit_retention_days: int = 2555  # 7 years for regulated industries
    evidence_retention_days: int = 2555
    trace_retention_days: int = 365


# =============================================================================
# ENFORCEMENT ENGINE
# =============================================================================

class MultiTenantEnforcementEngine:
    """
    Core enforcement engine that validates and enforces multi-tenant policies
    
    This is the central component that all other services must use to ensure
    proper tenant isolation, residency compliance, and governance enforcement.
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.tenant_contexts: Dict[int, TenantContext] = {}
        self.isolation_policies: Dict[str, IsolationPolicy] = {}
        self.residency_policies: Dict[str, ResidencyPolicy] = {}
        self.governance_policies: Dict[str, GovernancePolicy] = {}
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default enforcement policies"""
        
        # Default isolation policies per tier
        self.isolation_policies["t0_regulated"] = IsolationPolicy(
            policy_id="t0_regulated",
            policy_name="T0 Regulated Tenant Isolation",
            tenant_tier=TenantTier.T0_REGULATED,
            isolation_level=IsolationLevel.SCHEMA_PER_TENANT,
            rls_required=True,
            schema_isolation=True,
            cross_tenant_joins_blocked=True,
            raw_table_access_blocked=True,
            cache_key_prefixing=True,
            vector_index_partitioning=True,
            embedding_tenant_filtering=True
        )
        
        self.isolation_policies["t1_enterprise"] = IsolationPolicy(
            policy_id="t1_enterprise",
            policy_name="T1 Enterprise Isolation",
            tenant_tier=TenantTier.T1_ENTERPRISE,
            isolation_level=IsolationLevel.SHARED_SCHEMA_RLS,
            rls_required=True,
            schema_isolation=False,
            cross_tenant_joins_blocked=True,
            raw_table_access_blocked=True,
            cache_key_prefixing=True,
            vector_index_partitioning=True,
            embedding_tenant_filtering=True
        )
        
        self.isolation_policies["t2_midmarket"] = IsolationPolicy(
            policy_id="t2_midmarket", 
            policy_name="T2 Mid-Market Isolation",
            tenant_tier=TenantTier.T2_MIDMARKET,
            isolation_level=IsolationLevel.SHARED_SCHEMA_RLS,
            rls_required=True,
            schema_isolation=False,
            cross_tenant_joins_blocked=True,
            raw_table_access_blocked=False,  # More permissive for performance
            cache_key_prefixing=True,
            vector_index_partitioning=False,  # Shared indexes for cost optimization
            embedding_tenant_filtering=True
        )
        
        # Default residency policies
        self.residency_policies["gdpr_strict"] = ResidencyPolicy(
            policy_id="gdpr_strict",
            policy_name="GDPR Strict Residency",
            residency_requirement=ResidencyRequirement.REGION_STRICT,
            compliance_frameworks={ComplianceFramework.GDPR},
            allowed_regions={"eu-west-1", "eu-central-1", "eu-north-1"},
            primary_region="eu-west-1",
            failover_allowed=False,
            cross_region_replication_blocked=True,
            backup_region_restricted=True,
            compute_region_restricted=True,
            model_inference_region_restricted=True
        )
        
        self.residency_policies["rbi_strict"] = ResidencyPolicy(
            policy_id="rbi_strict",
            policy_name="RBI Strict Data Localization",
            residency_requirement=ResidencyRequirement.COUNTRY_STRICT,
            compliance_frameworks={ComplianceFramework.RBI, ComplianceFramework.DPDP},
            allowed_regions={"ap-south-1", "ap-south-2"},
            primary_region="ap-south-1",
            failover_allowed=False,
            cross_region_replication_blocked=True,
            backup_region_restricted=True,
            compute_region_restricted=True,
            model_inference_region_restricted=True
        )
        
        # Default governance policies
        self.governance_policies["regulated_strict"] = GovernancePolicy(
            policy_id="regulated_strict",
            policy_name="Regulated Industry Governance",
            separation_of_duties=SeparationOfDuties.REGULATED,
            finops_scope=FinOpsScope.RESOURCE_LEVEL,
            evidence_pack_mandatory=True,
            override_ledger_mandatory=True,
            execution_trace_mandatory=True,
            dual_authorization_required=True,
            approval_chain_length=2,
            audit_retention_days=2555,  # 7 years
            evidence_retention_days=2555,
            trace_retention_days=2555
        )
    
    async def validate_tenant_context(self, tenant_id: int, operation_context: Dict[str, Any]) -> bool:
        """
        Validate tenant context for any operation
        
        This is the main entry point for all tenant validation operations.
        Must be called before any data access or workflow execution.
        """
        try:
            # Get tenant context
            tenant_context = await self.get_tenant_context(tenant_id)
            if not tenant_context:
                raise ValueError(f"Tenant context not found for tenant_id: {tenant_id}")
            
            # Validate isolation requirements
            isolation_valid = await self._validate_isolation(tenant_context, operation_context)
            if not isolation_valid:
                return False
            
            # Validate residency requirements
            residency_valid = await self._validate_residency(tenant_context, operation_context)
            if not residency_valid:
                return False
            
            # Validate governance requirements
            governance_valid = await self._validate_governance(tenant_context, operation_context)
            if not governance_valid:
                return False
            
            return True
            
        except Exception as e:
            # Fail-closed: any validation error blocks operation
            if tenant_context and tenant_context.fail_closed_enabled:
                raise ValueError(f"Tenant validation failed (fail-closed): {e}")
            return False
    
    async def get_tenant_context(self, tenant_id: int) -> Optional[TenantContext]:
        """Retrieve complete tenant context"""
        if tenant_id in self.tenant_contexts:
            return self.tenant_contexts[tenant_id]
        
        # In full implementation, this would query the database
        # For now, return a default context for testing
        return TenantContext(
            tenant_id=tenant_id,
            tenant_name=f"Tenant-{tenant_id}",
            tenant_tier=TenantTier.T1_ENTERPRISE,
            isolation_level=IsolationLevel.SHARED_SCHEMA_RLS,
            residency_requirement=ResidencyRequirement.REGION_PREFERRED,
            primary_region="us-east-1",
            allowed_regions={"us-east-1", "us-west-2"},
            compliance_frameworks={ComplianceFramework.SOX},
            separation_of_duties=SeparationOfDuties.BASIC,
            finops_scope=FinOpsScope.TENANT_LEVEL
        )
    
    async def _validate_isolation(self, tenant_context: TenantContext, operation_context: Dict[str, Any]) -> bool:
        """Validate isolation requirements"""
        policy_key = tenant_context.tenant_tier.value.lower()
        policy = self.isolation_policies.get(policy_key)
        
        if not policy:
            return False
        
        # Check if tenant_id is present in operation context
        if policy.tenant_id_mandatory and 'tenant_id' not in operation_context:
            return False
        
        # Additional isolation checks would go here
        return True
    
    async def _validate_residency(self, tenant_context: TenantContext, operation_context: Dict[str, Any]) -> bool:
        """Validate residency requirements"""
        current_region = operation_context.get('region', 'unknown')
        
        if current_region not in tenant_context.allowed_regions:
            if tenant_context.residency_requirement in [ResidencyRequirement.REGION_STRICT, ResidencyRequirement.COUNTRY_STRICT]:
                return False
        
        return True
    
    async def _validate_governance(self, tenant_context: TenantContext, operation_context: Dict[str, Any]) -> bool:
        """Validate governance requirements"""
        # Check if audit trail is required and present
        if tenant_context.audit_trail_required and 'audit_context' not in operation_context:
            return False
        
        return True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_tenant_tier_from_compliance(compliance_frameworks: Set[ComplianceFramework]) -> TenantTier:
    """Determine tenant tier based on compliance requirements"""
    regulated_frameworks = {
        ComplianceFramework.RBI, 
        ComplianceFramework.IRDAI, 
        ComplianceFramework.HIPAA,
        ComplianceFramework.PCI_DSS
    }
    
    if compliance_frameworks.intersection(regulated_frameworks):
        return TenantTier.T0_REGULATED
    elif ComplianceFramework.SOX in compliance_frameworks:
        return TenantTier.T1_ENTERPRISE
    else:
        return TenantTier.T2_MIDMARKET

def get_isolation_level_for_tier(tenant_tier: TenantTier) -> IsolationLevel:
    """Get recommended isolation level for tenant tier"""
    isolation_mapping = {
        TenantTier.T0_REGULATED: IsolationLevel.SCHEMA_PER_TENANT,
        TenantTier.T1_ENTERPRISE: IsolationLevel.SHARED_SCHEMA_RLS,
        TenantTier.T2_MIDMARKET: IsolationLevel.SHARED_SCHEMA_RLS
    }
    return isolation_mapping.get(tenant_tier, IsolationLevel.SHARED_SCHEMA_RLS)


# =============================================================================
# EXPORT INTERFACE
# =============================================================================

__all__ = [
    'TenantTier',
    'IsolationLevel', 
    'ResidencyRequirement',
    'ComplianceFramework',
    'SeparationOfDuties',
    'FinOpsScope',
    'TenantContext',
    'IsolationPolicy',
    'ResidencyPolicy', 
    'GovernancePolicy',
    'MultiTenantEnforcementEngine',
    'get_tenant_tier_from_compliance',
    'get_isolation_level_for_tier'
]
