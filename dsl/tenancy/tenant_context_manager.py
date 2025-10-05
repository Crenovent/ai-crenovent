# Multi-Tenant Context Manager
# Tasks 6.5-T01, T02: Tenant context model and propagation across runtime

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from contextvars import ContextVar

logger = logging.getLogger(__name__)

class TenantTier(Enum):
    """Tenant service tiers with different resource allocations"""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"

class DataResidency(Enum):
    """Data residency regions for compliance"""
    US = "us"
    EU = "eu"
    INDIA = "india"
    CANADA = "canada"
    AUSTRALIA = "australia"
    SINGAPORE = "singapore"

class IndustryCode(Enum):
    """Industry codes for compliance overlays"""
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    FINTECH = "FinTech"
    ECOMMERCE = "E-commerce"
    MANUFACTURING = "Manufacturing"
    GOVERNMENT = "Government"

@dataclass
class TenantQuotas:
    """Resource quotas per tenant"""
    max_concurrent_workflows: int
    max_workflows_per_hour: int
    max_storage_gb: int
    max_compute_units: int
    max_api_calls_per_minute: int
    max_evidence_retention_days: int
    priority_level: int  # 1-10, higher = more priority

@dataclass
class TenantContext:
    """Comprehensive tenant context model"""
    tenant_id: int
    tenant_name: str
    tenant_tier: TenantTier
    industry_code: IndustryCode
    data_residency: DataResidency
    compliance_frameworks: List[str]
    created_at: str
    updated_at: str
    
    # Resource management
    quotas: TenantQuotas
    current_usage: Dict[str, Any]
    
    # Security and governance
    rbac_roles: List[str]
    abac_attributes: Dict[str, Any]
    policy_pack_version: str
    trust_score: float
    
    # Operational metadata
    contact_email: str
    support_tier: str
    sla_tier: str
    timezone: str
    
    # Feature flags and configurations
    feature_flags: Dict[str, bool]
    custom_configs: Dict[str, Any]
    
    # Compliance and audit
    audit_retention_days: int
    encryption_required: bool
    pii_handling_rules: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantContext':
        """Create from dictionary"""
        # Handle enum conversions
        data['tenant_tier'] = TenantTier(data['tenant_tier'])
        data['industry_code'] = IndustryCode(data['industry_code'])
        data['data_residency'] = DataResidency(data['data_residency'])
        
        # Handle nested objects
        if 'quotas' in data and isinstance(data['quotas'], dict):
            data['quotas'] = TenantQuotas(**data['quotas'])
        
        return cls(**data)

# Context variable for tenant context propagation
tenant_context_var: ContextVar[Optional[TenantContext]] = ContextVar('tenant_context', default=None)

class TenantContextManager:
    """
    Multi-Tenant Context Manager
    Tasks 6.5-T01, T02: Tenant context model and runtime propagation
    """
    
    def __init__(self, db_pool=None, cache_client=None):
        self.db_pool = db_pool
        self.cache_client = cache_client
        self._tenant_cache: Dict[int, TenantContext] = {}
        self._cache_lock = threading.RLock()
        
        # Default quotas by tier
        self.default_quotas = {
            TenantTier.STARTER: TenantQuotas(
                max_concurrent_workflows=5,
                max_workflows_per_hour=100,
                max_storage_gb=10,
                max_compute_units=50,
                max_api_calls_per_minute=100,
                max_evidence_retention_days=90,
                priority_level=3
            ),
            TenantTier.PROFESSIONAL: TenantQuotas(
                max_concurrent_workflows=25,
                max_workflows_per_hour=1000,
                max_storage_gb=100,
                max_compute_units=500,
                max_api_calls_per_minute=1000,
                max_evidence_retention_days=365,
                priority_level=5
            ),
            TenantTier.ENTERPRISE: TenantQuotas(
                max_concurrent_workflows=100,
                max_workflows_per_hour=10000,
                max_storage_gb=1000,
                max_compute_units=5000,
                max_api_calls_per_minute=10000,
                max_evidence_retention_days=2555,  # 7 years
                priority_level=8
            ),
            TenantTier.ENTERPRISE_PLUS: TenantQuotas(
                max_concurrent_workflows=500,
                max_workflows_per_hour=100000,
                max_storage_gb=10000,
                max_compute_units=50000,
                max_api_calls_per_minute=100000,
                max_evidence_retention_days=3650,  # 10 years
                priority_level=10
            )
        }
        
        # Industry-specific compliance frameworks
        self.industry_compliance_frameworks = {
            IndustryCode.SAAS: ["SOX_SAAS", "GDPR_SAAS", "SOC2_TYPE2"],
            IndustryCode.BANKING: ["RBI_INDIA", "BASEL_III", "KYC_AML", "GDPR_BANKING"],
            IndustryCode.INSURANCE: ["IRDAI_INDIA", "SOLVENCY_II", "GDPR_INSURANCE"],
            IndustryCode.HEALTHCARE: ["HIPAA", "GDPR_HEALTHCARE", "FDA_21CFR11"],
            IndustryCode.FINTECH: ["PCI_DSS", "SOX_FINTECH", "GDPR_FINTECH"],
            IndustryCode.ECOMMERCE: ["PCI_DSS", "GDPR_ECOMMERCE", "CCPA"],
            IndustryCode.MANUFACTURING: ["ISO_27001", "GDPR_MANUFACTURING"],
            IndustryCode.GOVERNMENT: ["FISMA", "GDPR_GOVERNMENT", "SOC2_TYPE2"]
        }
    
    async def get_tenant_context(self, tenant_id: int) -> Optional[TenantContext]:
        """Get tenant context with caching"""
        
        # Check cache first
        with self._cache_lock:
            if tenant_id in self._tenant_cache:
                return self._tenant_cache[tenant_id]
        
        # Load from database
        tenant_context = await self._load_tenant_from_db(tenant_id)
        
        if tenant_context:
            # Cache the result
            with self._cache_lock:
                self._tenant_cache[tenant_id] = tenant_context
        
        return tenant_context
    
    async def set_tenant_context(self, tenant_context: TenantContext) -> None:
        """Set tenant context in current execution context"""
        tenant_context_var.set(tenant_context)
        logger.debug(f"Set tenant context: {tenant_context.tenant_id} ({tenant_context.tenant_name})")
    
    def get_current_tenant_context(self) -> Optional[TenantContext]:
        """Get current tenant context from execution context"""
        return tenant_context_var.get()
    
    async def create_tenant_context(
        self,
        tenant_name: str,
        industry_code: IndustryCode,
        data_residency: DataResidency,
        tenant_tier: TenantTier = TenantTier.PROFESSIONAL,
        contact_email: str = "",
        custom_configs: Dict[str, Any] = None
    ) -> TenantContext:
        """Create new tenant context"""
        
        tenant_id = await self._generate_tenant_id()
        
        # Get default quotas for tier
        quotas = self.default_quotas[tenant_tier]
        
        # Get compliance frameworks for industry
        compliance_frameworks = self.industry_compliance_frameworks.get(
            industry_code, 
            ["GDPR_GENERIC", "SOC2_TYPE2"]
        )
        
        # Create tenant context
        tenant_context = TenantContext(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tenant_tier=tenant_tier,
            industry_code=industry_code,
            data_residency=data_residency,
            compliance_frameworks=compliance_frameworks,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            quotas=quotas,
            current_usage={
                'concurrent_workflows': 0,
                'workflows_this_hour': 0,
                'storage_used_gb': 0,
                'compute_units_used': 0,
                'api_calls_this_minute': 0
            },
            rbac_roles=self._get_default_roles(tenant_tier),
            abac_attributes=self._get_default_attributes(industry_code, data_residency),
            policy_pack_version=f"{industry_code.value.lower()}_v1.0",
            trust_score=1.0,  # Start with perfect trust
            contact_email=contact_email,
            support_tier=self._get_support_tier(tenant_tier),
            sla_tier=self._get_sla_tier(tenant_tier),
            timezone="UTC",
            feature_flags=self._get_default_feature_flags(tenant_tier),
            custom_configs=custom_configs or {},
            audit_retention_days=quotas.max_evidence_retention_days,
            encryption_required=tenant_tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS],
            pii_handling_rules=self._get_pii_rules(industry_code, data_residency)
        )
        
        # Store in database
        await self._store_tenant_in_db(tenant_context)
        
        # Cache the result
        with self._cache_lock:
            self._tenant_cache[tenant_id] = tenant_context
        
        logger.info(f"Created tenant context: {tenant_id} ({tenant_name}) - {industry_code.value}")
        return tenant_context
    
    async def update_tenant_usage(
        self,
        tenant_id: int,
        usage_type: str,
        increment: int = 1
    ) -> bool:
        """Update tenant resource usage"""
        
        tenant_context = await self.get_tenant_context(tenant_id)
        if not tenant_context:
            return False
        
        # Update usage
        current_usage = tenant_context.current_usage.get(usage_type, 0)
        tenant_context.current_usage[usage_type] = current_usage + increment
        
        # Check quotas
        quota_exceeded = self._check_quota_exceeded(tenant_context, usage_type)
        if quota_exceeded:
            logger.warning(f"Quota exceeded for tenant {tenant_id}: {usage_type}")
            return False
        
        # Update in cache and database
        with self._cache_lock:
            self._tenant_cache[tenant_id] = tenant_context
        
        await self._update_tenant_usage_in_db(tenant_id, usage_type, tenant_context.current_usage[usage_type])
        
        return True
    
    async def check_tenant_permissions(
        self,
        tenant_id: int,
        required_permissions: List[str],
        resource_attributes: Dict[str, Any] = None
    ) -> Tuple[bool, List[str]]:
        """Check tenant permissions (RBAC + ABAC)"""
        
        tenant_context = await self.get_tenant_context(tenant_id)
        if not tenant_context:
            return False, ["Tenant not found"]
        
        violations = []
        
        # RBAC check
        tenant_roles = set(tenant_context.rbac_roles)
        for permission in required_permissions:
            if not self._check_rbac_permission(tenant_roles, permission):
                violations.append(f"Missing RBAC permission: {permission}")
        
        # ABAC check
        if resource_attributes:
            abac_violations = self._check_abac_permissions(
                tenant_context.abac_attributes,
                resource_attributes
            )
            violations.extend(abac_violations)
        
        return len(violations) == 0, violations
    
    async def get_tenant_residency_rules(self, tenant_id: int) -> Dict[str, Any]:
        """Get data residency rules for tenant"""
        
        tenant_context = await self.get_tenant_context(tenant_id)
        if not tenant_context:
            return {}
        
        residency_rules = {
            'data_residency': tenant_context.data_residency.value,
            'allowed_regions': self._get_allowed_regions(tenant_context.data_residency),
            'encryption_required': tenant_context.encryption_required,
            'cross_border_restrictions': self._get_cross_border_restrictions(tenant_context.data_residency),
            'compliance_frameworks': tenant_context.compliance_frameworks
        }
        
        return residency_rules
    
    def _get_allowed_regions(self, data_residency: DataResidency) -> List[str]:
        """Get allowed regions for data residency"""
        region_mappings = {
            DataResidency.US: ["us-east-1", "us-west-2", "us-central-1"],
            DataResidency.EU: ["eu-west-1", "eu-central-1", "eu-north-1"],
            DataResidency.INDIA: ["ap-south-1", "ap-south-2"],
            DataResidency.CANADA: ["ca-central-1"],
            DataResidency.AUSTRALIA: ["ap-southeast-2"],
            DataResidency.SINGAPORE: ["ap-southeast-1"]
        }
        return region_mappings.get(data_residency, ["us-east-1"])
    
    def _get_cross_border_restrictions(self, data_residency: DataResidency) -> Dict[str, Any]:
        """Get cross-border data transfer restrictions"""
        restrictions = {
            DataResidency.EU: {
                'gdpr_compliance_required': True,
                'adequacy_decision_required': True,
                'standard_contractual_clauses': True
            },
            DataResidency.INDIA: {
                'dpdp_compliance_required': True,
                'local_storage_required': True,
                'cross_border_approval_required': True
            }
        }
        return restrictions.get(data_residency, {})
    
    def _check_quota_exceeded(self, tenant_context: TenantContext, usage_type: str) -> bool:
        """Check if quota is exceeded for usage type"""
        current_usage = tenant_context.current_usage.get(usage_type, 0)
        
        quota_mappings = {
            'concurrent_workflows': tenant_context.quotas.max_concurrent_workflows,
            'workflows_this_hour': tenant_context.quotas.max_workflows_per_hour,
            'storage_used_gb': tenant_context.quotas.max_storage_gb,
            'compute_units_used': tenant_context.quotas.max_compute_units,
            'api_calls_this_minute': tenant_context.quotas.max_api_calls_per_minute
        }
        
        quota_limit = quota_mappings.get(usage_type, float('inf'))
        return current_usage >= quota_limit
    
    def _check_rbac_permission(self, tenant_roles: set, permission: str) -> bool:
        """Check RBAC permission"""
        # Simplified RBAC - in practice would be more sophisticated
        role_permissions = {
            'admin': ['*'],
            'workflow_manager': ['workflow:create', 'workflow:execute', 'workflow:view'],
            'workflow_viewer': ['workflow:view'],
            'compliance_officer': ['governance:view', 'evidence:view', 'audit:view'],
            'ops_manager': ['monitoring:view', 'alerts:manage']
        }
        
        for role in tenant_roles:
            permissions = role_permissions.get(role, [])
            if '*' in permissions or permission in permissions:
                return True
        
        return False
    
    def _check_abac_permissions(
        self,
        tenant_attributes: Dict[str, Any],
        resource_attributes: Dict[str, Any]
    ) -> List[str]:
        """Check ABAC permissions"""
        violations = []
        
        # Industry-specific checks
        tenant_industry = tenant_attributes.get('industry_code')
        resource_industry = resource_attributes.get('industry_code')
        
        if resource_industry and tenant_industry != resource_industry:
            violations.append(f"Industry mismatch: tenant={tenant_industry}, resource={resource_industry}")
        
        # Data residency checks
        tenant_residency = tenant_attributes.get('data_residency')
        resource_residency = resource_attributes.get('data_residency')
        
        if resource_residency and tenant_residency != resource_residency:
            violations.append(f"Data residency mismatch: tenant={tenant_residency}, resource={resource_residency}")
        
        return violations
    
    def _get_default_roles(self, tenant_tier: TenantTier) -> List[str]:
        """Get default roles for tenant tier"""
        role_mappings = {
            TenantTier.STARTER: ['workflow_viewer'],
            TenantTier.PROFESSIONAL: ['workflow_manager', 'workflow_viewer'],
            TenantTier.ENTERPRISE: ['admin', 'workflow_manager', 'compliance_officer'],
            TenantTier.ENTERPRISE_PLUS: ['admin', 'workflow_manager', 'compliance_officer', 'ops_manager']
        }
        return role_mappings.get(tenant_tier, ['workflow_viewer'])
    
    def _get_default_attributes(
        self,
        industry_code: IndustryCode,
        data_residency: DataResidency
    ) -> Dict[str, Any]:
        """Get default ABAC attributes"""
        return {
            'industry_code': industry_code.value,
            'data_residency': data_residency.value,
            'compliance_level': 'standard',
            'security_clearance': 'internal'
        }
    
    def _get_support_tier(self, tenant_tier: TenantTier) -> str:
        """Get support tier for tenant tier"""
        support_mappings = {
            TenantTier.STARTER: 'community',
            TenantTier.PROFESSIONAL: 'standard',
            TenantTier.ENTERPRISE: 'premium',
            TenantTier.ENTERPRISE_PLUS: 'white_glove'
        }
        return support_mappings.get(tenant_tier, 'standard')
    
    def _get_sla_tier(self, tenant_tier: TenantTier) -> str:
        """Get SLA tier for tenant tier"""
        sla_mappings = {
            TenantTier.STARTER: '99.0',
            TenantTier.PROFESSIONAL: '99.5',
            TenantTier.ENTERPRISE: '99.9',
            TenantTier.ENTERPRISE_PLUS: '99.99'
        }
        return sla_mappings.get(tenant_tier, '99.5')
    
    def _get_default_feature_flags(self, tenant_tier: TenantTier) -> Dict[str, bool]:
        """Get default feature flags for tenant tier"""
        base_features = {
            'workflow_execution': True,
            'basic_monitoring': True,
            'email_notifications': True
        }
        
        if tenant_tier in [TenantTier.PROFESSIONAL, TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
            base_features.update({
                'advanced_analytics': True,
                'custom_integrations': True,
                'slack_notifications': True
            })
        
        if tenant_tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
            base_features.update({
                'compliance_dashboards': True,
                'audit_trails': True,
                'custom_policies': True,
                'white_label_ui': True
            })
        
        if tenant_tier == TenantTier.ENTERPRISE_PLUS:
            base_features.update({
                'dedicated_support': True,
                'custom_sla': True,
                'priority_execution': True,
                'advanced_security': True
            })
        
        return base_features
    
    def _get_pii_rules(
        self,
        industry_code: IndustryCode,
        data_residency: DataResidency
    ) -> Dict[str, Any]:
        """Get PII handling rules for industry and residency"""
        base_rules = {
            'encryption_at_rest': True,
            'encryption_in_transit': True,
            'access_logging': True,
            'retention_limit_days': 365
        }
        
        # Industry-specific rules
        if industry_code in [IndustryCode.HEALTHCARE]:
            base_rules.update({
                'hipaa_compliance': True,
                'phi_encryption': True,
                'access_audit_required': True,
                'retention_limit_days': 2555  # 7 years
            })
        
        if industry_code in [IndustryCode.BANKING, IndustryCode.FINTECH]:
            base_rules.update({
                'financial_data_protection': True,
                'pci_compliance': True,
                'retention_limit_days': 2555  # 7 years
            })
        
        # Residency-specific rules
        if data_residency == DataResidency.EU:
            base_rules.update({
                'gdpr_compliance': True,
                'right_to_erasure': True,
                'data_portability': True,
                'consent_management': True
            })
        
        if data_residency == DataResidency.INDIA:
            base_rules.update({
                'dpdp_compliance': True,
                'local_storage_required': True,
                'cross_border_restrictions': True
            })
        
        return base_rules
    
    async def _generate_tenant_id(self) -> int:
        """Generate unique tenant ID"""
        # In practice, this would use a proper ID generation service
        import random
        return random.randint(1000, 999999)
    
    async def _load_tenant_from_db(self, tenant_id: int) -> Optional[TenantContext]:
        """Load tenant context from database"""
        if not self.db_pool:
            # Mock data for testing
            return self._get_mock_tenant_context(tenant_id)
        
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM tenant_contexts WHERE tenant_id = $1
                """, tenant_id)
                
                if row:
                    return TenantContext.from_dict(dict(row))
                
        except Exception as e:
            logger.error(f"Failed to load tenant {tenant_id} from database: {e}")
        
        return None
    
    async def _store_tenant_in_db(self, tenant_context: TenantContext) -> None:
        """Store tenant context in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO tenant_contexts (
                        tenant_id, tenant_name, tenant_tier, industry_code,
                        data_residency, compliance_frameworks, quotas,
                        rbac_roles, abac_attributes, policy_pack_version,
                        trust_score, contact_email, support_tier, sla_tier,
                        feature_flags, custom_configs, audit_retention_days,
                        encryption_required, pii_handling_rules, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                """,
                    tenant_context.tenant_id,
                    tenant_context.tenant_name,
                    tenant_context.tenant_tier.value,
                    tenant_context.industry_code.value,
                    tenant_context.data_residency.value,
                    json.dumps(tenant_context.compliance_frameworks),
                    json.dumps(asdict(tenant_context.quotas)),
                    json.dumps(tenant_context.rbac_roles),
                    json.dumps(tenant_context.abac_attributes),
                    tenant_context.policy_pack_version,
                    tenant_context.trust_score,
                    tenant_context.contact_email,
                    tenant_context.support_tier,
                    tenant_context.sla_tier,
                    json.dumps(tenant_context.feature_flags),
                    json.dumps(tenant_context.custom_configs),
                    tenant_context.audit_retention_days,
                    tenant_context.encryption_required,
                    json.dumps(tenant_context.pii_handling_rules),
                    tenant_context.created_at,
                    tenant_context.updated_at
                )
                
        except Exception as e:
            logger.error(f"Failed to store tenant {tenant_context.tenant_id} in database: {e}")
    
    async def _update_tenant_usage_in_db(self, tenant_id: int, usage_type: str, new_value: int) -> None:
        """Update tenant usage in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE tenant_contexts 
                    SET current_usage = jsonb_set(current_usage, $2, $3),
                        updated_at = $4
                    WHERE tenant_id = $1
                """, 
                    tenant_id,
                    f'{{{usage_type}}}',
                    json.dumps(new_value),
                    datetime.now(timezone.utc).isoformat()
                )
                
        except Exception as e:
            logger.error(f"Failed to update tenant usage for {tenant_id}: {e}")
    
    def _get_mock_tenant_context(self, tenant_id: int) -> TenantContext:
        """Get mock tenant context for testing"""
        return TenantContext(
            tenant_id=tenant_id,
            tenant_name=f"Tenant_{tenant_id}",
            tenant_tier=TenantTier.PROFESSIONAL,
            industry_code=IndustryCode.SAAS,
            data_residency=DataResidency.US,
            compliance_frameworks=["SOX_SAAS", "GDPR_SAAS"],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            quotas=self.default_quotas[TenantTier.PROFESSIONAL],
            current_usage={
                'concurrent_workflows': 0,
                'workflows_this_hour': 0,
                'storage_used_gb': 0,
                'compute_units_used': 0,
                'api_calls_this_minute': 0
            },
            rbac_roles=['workflow_manager', 'workflow_viewer'],
            abac_attributes={
                'industry_code': 'SaaS',
                'data_residency': 'us',
                'compliance_level': 'standard'
            },
            policy_pack_version="saas_v1.0",
            trust_score=1.0,
            contact_email=f"admin@tenant{tenant_id}.com",
            support_tier="standard",
            sla_tier="99.5",
            timezone="UTC",
            feature_flags=self._get_default_feature_flags(TenantTier.PROFESSIONAL),
            custom_configs={},
            audit_retention_days=365,
            encryption_required=False,
            pii_handling_rules=self._get_pii_rules(IndustryCode.SAAS, DataResidency.US)
        )

# Global tenant context manager
tenant_context_manager = TenantContextManager()
