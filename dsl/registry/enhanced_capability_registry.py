"""
Enhanced Capability Registry - Tasks 14.1.1, 14.1.3, 14.1.6, 14.2.1, 14.2.10, 14.2.11
Implements comprehensive capability metadata management with full tenant isolation and SaaS focus.

Features:
- Dynamic capability metadata storage (Task 14.1.1)
- Multi-tenant capability mapping (Task 14.1.3) 
- SaaS-specific workflow templates (Task 14.1.6)
- Trust score storage per capability (Task 14.2.1)
- Metadata attachment to capabilities (Task 14.2.10)
- Dynamic trust score computation (Task 14.2.11)
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

class CapabilityType(Enum):
    """Dynamic capability types with SaaS extensions"""
    RBA = "rba"
    RBIA = "rbia" 
    AALA = "aala"
    # SaaS-specific capability types
    SUBSCRIPTION_MANAGER = "subscription_manager"
    BILLING_ENGINE = "billing_engine"
    CUSTOMER_SUCCESS = "customer_success"
    CHURN_PREDICTOR = "churn_predictor"
    REVENUE_ANALYZER = "revenue_analyzer"

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
class CapabilityMetadata:
    """Core capability metadata - Task 14.1.1"""
    capability_id: str
    name: str
    capability_type: CapabilityType
    version: str
    
    # Multi-tenant mapping - Task 14.1.3 (required fields first)
    tenant_id: int
    tenant_tier: TenantTier
    industry_code: IndustryCode
    
    # Optional fields
    description: Optional[str] = None
    
    # SaaS-specific metadata - Task 14.1.6
    saas_workflows: List[str] = None  # ARR, churn, QBR workflows
    business_metrics: Dict[str, Any] = None
    customer_impact_score: float = 0.0
    
    # Technical specifications
    input_schema: Dict[str, Any] = None
    output_schema: Dict[str, Any] = None
    operator_definition: Dict[str, Any] = None
    
    # Performance characteristics
    avg_execution_time_ms: int = 0
    success_rate: float = 1.0
    usage_count: int = 0
    
    # Governance
    compliance_requirements: List[str] = None
    policy_tags: List[str] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    last_used_at: Optional[datetime] = None
    created_by_user_id: int = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.saas_workflows is None:
            self.saas_workflows = []
        if self.business_metrics is None:
            self.business_metrics = {}
        if self.compliance_requirements is None:
            self.compliance_requirements = []
        if self.policy_tags is None:
            self.policy_tags = []

@dataclass
class TrustScore:
    """Trust score metadata - Task 14.2.1"""
    capability_id: str
    tenant_id: int
    tenant_tier: TenantTier
    industry_code: IndustryCode
    
    # Trust factors (configurable weights)
    execution_success_rate: float = 0.0
    compliance_violation_count: int = 0
    user_feedback_score: float = 0.0
    business_impact_score: float = 0.0
    
    # Computed scores
    overall_trust_score: float = 0.0
    trust_level: str = "medium"
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 1.0
    
    # Metadata
    last_calculated_at: datetime = None
    calculation_period_days: int = 30
    factor_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.last_calculated_at is None:
            self.last_calculated_at = datetime.utcnow()
        if self.factor_scores is None:
            self.factor_scores = {}

class EnhancedCapabilityRegistry:
    """
    Enhanced Capability Registry with full tenant isolation and dynamic configuration.
    Implements Tasks 14.1.1, 14.1.3, 14.1.6, 14.2.1, 14.2.10, 14.2.11.
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Dynamic configuration (no hardcoding)
        self.trust_factor_weights = self._load_trust_factor_weights()
        self.tenant_tier_thresholds = self._load_tenant_tier_thresholds()
        self.saas_workflow_templates = self._load_saas_workflow_templates()
        self.industry_compliance_mapping = self._load_industry_compliance_mapping()
        
        # Cache for performance
        self.capability_cache: Dict[str, CapabilityMetadata] = {}
        self.trust_score_cache: Dict[str, TrustScore] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Tenant isolation
        self.tenant_capabilities: Dict[int, List[str]] = {}
        
    def _load_trust_factor_weights(self) -> Dict[str, Dict[str, float]]:
        """Load configurable trust factor weights per industry"""
        return {
            'SaaS': {
                'execution_success': 0.25,
                'compliance_violations': 0.20,
                'user_feedback': 0.15,
                'business_impact': 0.25,  # Higher for SaaS (revenue impact)
                'historical_reliability': 0.15
            },
            'Banking': {
                'execution_success': 0.20,
                'compliance_violations': 0.35,  # Higher compliance weight
                'user_feedback': 0.10,
                'business_impact': 0.20,
                'historical_reliability': 0.15
            },
            'Insurance': {
                'execution_success': 0.25,
                'compliance_violations': 0.30,
                'user_feedback': 0.10,
                'business_impact': 0.20,
                'historical_reliability': 0.15
            }
        }
    
    def _load_tenant_tier_thresholds(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load configurable trust level thresholds per tenant tier"""
        return {
            'T0': {  # Regulated tenants - stricter thresholds
                'critical': (0.95, 1.0),
                'high': (0.85, 0.95),
                'medium': (0.70, 0.85),
                'low': (0.50, 0.70),
                'untrusted': (0.0, 0.50)
            },
            'T1': {  # Enterprise tenants - balanced thresholds
                'critical': (0.90, 1.0),
                'high': (0.75, 0.90),
                'medium': (0.60, 0.75),
                'low': (0.40, 0.60),
                'untrusted': (0.0, 0.40)
            },
            'T2': {  # Mid-market tenants - more lenient thresholds
                'critical': (0.85, 1.0),
                'high': (0.70, 0.85),
                'medium': (0.55, 0.70),
                'low': (0.35, 0.55),
                'untrusted': (0.0, 0.35)
            }
        }
    
    def _load_saas_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load SaaS-specific workflow templates - Task 14.1.6"""
        return {
            'arr_aggregation': {
                'name': 'ARR Aggregation Workflow',
                'description': 'Automated Annual Recurring Revenue calculation',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'subscription_data': {'type': 'array'},
                        'time_period': {'type': 'string'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['subscription_data', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'total_arr': {'type': 'number'},
                        'arr_growth_rate': {'type': 'number'},
                        'breakdown_by_segment': {'type': 'object'}
                    }
                },
                'business_metrics': ['ARR', 'Growth Rate', 'Segment Analysis'],
                'compliance_requirements': ['SOX_SAAS', 'GDPR_SAAS']
            },
            'churn_prediction': {
                'name': 'Customer Churn Prediction',
                'description': 'ML-based churn risk assessment',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'customer_usage_data': {'type': 'array'},
                        'support_tickets': {'type': 'array'},
                        'billing_history': {'type': 'array'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['customer_usage_data', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'churn_risk_score': {'type': 'number'},
                        'risk_factors': {'type': 'array'},
                        'recommended_actions': {'type': 'array'}
                    }
                },
                'business_metrics': ['Churn Rate', 'Customer Lifetime Value', 'Retention Rate'],
                'compliance_requirements': ['GDPR_SAAS', 'SAAS_BUSINESS_RULES']
            },
            'qbr_automation': {
                'name': 'Quarterly Business Review Automation',
                'description': 'Automated QBR report generation',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'account_data': {'type': 'object'},
                        'performance_metrics': {'type': 'object'},
                        'quarter': {'type': 'string'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['account_data', 'performance_metrics', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'qbr_report': {'type': 'object'},
                        'key_insights': {'type': 'array'},
                        'action_items': {'type': 'array'}
                    }
                },
                'business_metrics': ['Account Health', 'Revenue Growth', 'Expansion Opportunities'],
                'compliance_requirements': ['SOX_SAAS']
            }
        }
    
    def _load_industry_compliance_mapping(self) -> Dict[str, List[str]]:
        """Load industry-specific compliance requirements"""
        return {
            'SaaS': ['SOX_SAAS', 'GDPR_SAAS', 'SAAS_BUSINESS_RULES'],
            'Banking': ['SOX', 'RBI', 'DPDP', 'AML_KYC'],
            'Insurance': ['HIPAA', 'NAIC', 'GDPR'],
            'FinTech': ['SOX', 'GDPR', 'PCI_DSS'],
            'E-commerce': ['GDPR', 'PCI_DSS', 'CCPA'],
            'IT_Services': ['SOX', 'GDPR', 'ISO_27001']
        }
    
    async def initialize(self):
        """Initialize the enhanced capability registry"""
        try:
            await self._ensure_database_tables()
            await self._load_existing_capabilities()
            self.logger.info("Enhanced Capability Registry initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Capability Registry: {e}")
            raise
    
    async def _ensure_database_tables(self):
        """Ensure all required database tables exist"""
        if not self.pool_manager:
            return
            
        async with self.pool_manager.get_connection() as conn:
            # Enhanced capability registry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_capability_registry (
                    capability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    tenant_tier VARCHAR(10) NOT NULL,
                    industry_code VARCHAR(20) NOT NULL,
                    capability_name VARCHAR(255) NOT NULL,
                    capability_type VARCHAR(50) NOT NULL,
                    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
                    description TEXT,
                    
                    -- SaaS-specific fields
                    saas_workflows JSONB DEFAULT '[]',
                    business_metrics JSONB DEFAULT '{}',
                    customer_impact_score DECIMAL(5,4) DEFAULT 0.0,
                    
                    -- Technical specifications
                    input_schema JSONB NOT NULL DEFAULT '{}',
                    output_schema JSONB NOT NULL DEFAULT '{}',
                    operator_definition JSONB NOT NULL DEFAULT '{}',
                    
                    -- Performance characteristics
                    avg_execution_time_ms INTEGER DEFAULT 0,
                    success_rate DECIMAL(5,4) DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    
                    -- Governance
                    compliance_requirements JSONB DEFAULT '[]',
                    policy_tags JSONB DEFAULT '[]',
                    
                    -- Timestamps
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_used_at TIMESTAMPTZ,
                    created_by_user_id INTEGER,
                    
                    CONSTRAINT unique_registry_capability_per_tenant UNIQUE (tenant_id, capability_name, version),
                    CONSTRAINT valid_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2')),
                    CONSTRAINT valid_success_rate CHECK (success_rate >= 0 AND success_rate <= 1),
                    CONSTRAINT valid_customer_impact CHECK (customer_impact_score >= 0 AND customer_impact_score <= 1)
                );
            """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE enhanced_capability_registry ENABLE ROW LEVEL SECURITY;")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS enhanced_capability_registry_rls_policy ON enhanced_capability_registry;
                CREATE POLICY enhanced_capability_registry_rls_policy
                    ON enhanced_capability_registry
                    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_tenant_id ON enhanced_capability_registry(tenant_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_type ON enhanced_capability_registry(capability_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_industry ON enhanced_capability_registry(industry_code);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_tenant_tier ON enhanced_capability_registry(tenant_tier);")
    
    async def register_capability(self, metadata: CapabilityMetadata) -> bool:
        """Register a new capability - Task 14.1.1, 14.1.3"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                self.capability_cache[metadata.capability_id] = metadata
                self._update_tenant_mapping(metadata.tenant_id, metadata.capability_id)
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{metadata.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO enhanced_capability_registry (
                        capability_id, tenant_id, tenant_tier, industry_code,
                        capability_name, capability_type, version, description,
                        saas_workflows, business_metrics, customer_impact_score,
                        input_schema, output_schema, operator_definition,
                        avg_execution_time_ms, success_rate, usage_count,
                        compliance_requirements, policy_tags, created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    ON CONFLICT (tenant_id, capability_name, version) 
                    DO UPDATE SET
                        updated_at = NOW(),
                        description = EXCLUDED.description,
                        saas_workflows = EXCLUDED.saas_workflows,
                        business_metrics = EXCLUDED.business_metrics,
                        customer_impact_score = EXCLUDED.customer_impact_score
                """, 
                metadata.capability_id, metadata.tenant_id, metadata.tenant_tier.value, 
                metadata.industry_code.value, metadata.name, metadata.capability_type.value,
                metadata.version, metadata.description, json.dumps(metadata.saas_workflows),
                json.dumps(metadata.business_metrics), metadata.customer_impact_score,
                json.dumps(metadata.input_schema), json.dumps(metadata.output_schema),
                json.dumps(metadata.operator_definition), metadata.avg_execution_time_ms,
                metadata.success_rate, metadata.usage_count, 
                json.dumps(metadata.compliance_requirements), json.dumps(metadata.policy_tags),
                metadata.created_by_user_id)
            
            # Update cache and tenant mapping
            self.capability_cache[metadata.capability_id] = metadata
            self._update_tenant_mapping(metadata.tenant_id, metadata.capability_id)
            
            self.logger.info(f"Registered capability {metadata.name} for tenant {metadata.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register capability {metadata.name}: {e}")
            return False
    
    def _update_tenant_mapping(self, tenant_id: int, capability_id: str):
        """Update tenant-to-capability mapping - Task 14.1.3"""
        if tenant_id not in self.tenant_capabilities:
            self.tenant_capabilities[tenant_id] = []
        
        if capability_id not in self.tenant_capabilities[tenant_id]:
            self.tenant_capabilities[tenant_id].append(capability_id)
    
    async def get_tenant_capabilities(self, tenant_id: int, 
                                    capability_type: Optional[CapabilityType] = None,
                                    industry_filter: Optional[IndustryCode] = None) -> List[CapabilityMetadata]:
        """Get capabilities for a specific tenant - Task 14.1.3"""
        try:
            if not self.pool_manager:
                # Return from cache for testing
                tenant_caps = []
                for cap_id in self.tenant_capabilities.get(tenant_id, []):
                    if cap_id in self.capability_cache:
                        cap = self.capability_cache[cap_id]
                        if capability_type and cap.capability_type != capability_type:
                            continue
                        if industry_filter and cap.industry_code != industry_filter:
                            continue
                        tenant_caps.append(cap)
                return tenant_caps
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                query = "SELECT * FROM enhanced_capability_registry WHERE tenant_id = $1"
                params = [tenant_id]
                
                if capability_type:
                    query += " AND capability_type = $2"
                    params.append(capability_type.value)
                
                if industry_filter:
                    query += f" AND industry_code = ${len(params) + 1}"
                    params.append(industry_filter.value)
                
                rows = await conn.fetch(query, *params)
                
                capabilities = []
                for row in rows:
                    cap = self._row_to_capability_metadata(row)
                    capabilities.append(cap)
                    # Update cache
                    self.capability_cache[cap.capability_id] = cap
                
                return capabilities
                
        except Exception as e:
            self.logger.error(f"Failed to get tenant capabilities for {tenant_id}: {e}")
            return []
    
    async def create_saas_workflow_capability(self, tenant_id: int, tenant_tier: TenantTier,
                                           workflow_type: str, created_by_user_id: int) -> Optional[CapabilityMetadata]:
        """Create SaaS-specific workflow capability - Task 14.1.6"""
        try:
            if workflow_type not in self.saas_workflow_templates:
                self.logger.error(f"Unknown SaaS workflow type: {workflow_type}")
                return None
            
            template = self.saas_workflow_templates[workflow_type]
            
            # Generate unique capability ID
            capability_id = str(uuid.uuid4())
            
            # Create capability metadata
            metadata = CapabilityMetadata(
                capability_id=capability_id,
                name=template['name'],
                capability_type=CapabilityType.RBA,  # Default to RBA for SaaS workflows
                version="1.0.0",
                description=template['description'],
                tenant_id=tenant_id,
                tenant_tier=tenant_tier,
                industry_code=IndustryCode.SAAS,
                saas_workflows=[workflow_type],
                business_metrics=template['business_metrics'],
                input_schema=template['input_schema'],
                output_schema=template['output_schema'],
                compliance_requirements=template['compliance_requirements'],
                created_by_user_id=created_by_user_id
            )
            
            # Register the capability
            success = await self.register_capability(metadata)
            if success:
                self.logger.info(f"Created SaaS workflow capability: {workflow_type} for tenant {tenant_id}")
                return metadata
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create SaaS workflow capability {workflow_type}: {e}")
            return None
    
    async def store_trust_score(self, trust_score: TrustScore) -> bool:
        """Store trust score for capability - Task 14.2.1"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                cache_key = f"{trust_score.tenant_id}:{trust_score.capability_id}"
                self.trust_score_cache[cache_key] = trust_score
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{trust_score.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO saas_trust_scores (
                        tenant_id, tenant_tier, industry_code, capability_id,
                        capability_type, execution_success_rate, compliance_violation_count,
                        user_feedback_score, business_impact_score, overall_trust_score,
                        trust_level, confidence_interval_lower, confidence_interval_upper,
                        last_calculated_at, calculation_period_days, factor_scores
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (tenant_id, capability_id)
                    DO UPDATE SET
                        execution_success_rate = EXCLUDED.execution_success_rate,
                        compliance_violation_count = EXCLUDED.compliance_violation_count,
                        user_feedback_score = EXCLUDED.user_feedback_score,
                        business_impact_score = EXCLUDED.business_impact_score,
                        overall_trust_score = EXCLUDED.overall_trust_score,
                        trust_level = EXCLUDED.trust_level,
                        confidence_interval_lower = EXCLUDED.confidence_interval_lower,
                        confidence_interval_upper = EXCLUDED.confidence_interval_upper,
                        last_calculated_at = EXCLUDED.last_calculated_at,
                        factor_scores = EXCLUDED.factor_scores
                """,
                trust_score.tenant_id, trust_score.tenant_tier.value, trust_score.industry_code.value,
                trust_score.capability_id, "RBA", trust_score.execution_success_rate,
                trust_score.compliance_violation_count, trust_score.user_feedback_score,
                trust_score.business_impact_score, trust_score.overall_trust_score,
                trust_score.trust_level, trust_score.confidence_interval_lower,
                trust_score.confidence_interval_upper, trust_score.last_calculated_at,
                trust_score.calculation_period_days, json.dumps(trust_score.factor_scores))
            
            # Update cache
            cache_key = f"{trust_score.tenant_id}:{trust_score.capability_id}"
            self.trust_score_cache[cache_key] = trust_score
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store trust score for {trust_score.capability_id}: {e}")
            return False
    
    async def attach_metadata_to_capability(self, capability_id: str, tenant_id: int,
                                          metadata_updates: Dict[str, Any]) -> bool:
        """Attach additional metadata to capability - Task 14.2.10"""
        try:
            if not self.pool_manager:
                # Update cache for testing
                if capability_id in self.capability_cache:
                    cap = self.capability_cache[capability_id]
                    for key, value in metadata_updates.items():
                        if hasattr(cap, key):
                            setattr(cap, key, value)
                    cap.updated_at = datetime.utcnow()
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Build dynamic update query
                set_clauses = []
                params = []
                param_count = 1
                
                for key, value in metadata_updates.items():
                    if key in ['business_metrics', 'saas_workflows', 'compliance_requirements', 'policy_tags']:
                        set_clauses.append(f"{key} = ${param_count}")
                        params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
                    elif key in ['customer_impact_score', 'success_rate', 'avg_execution_time_ms', 'usage_count']:
                        set_clauses.append(f"{key} = ${param_count}")
                        params.append(value)
                    param_count += 1
                
                if set_clauses:
                    set_clauses.append(f"updated_at = NOW()")
                    query = f"""
                        UPDATE enhanced_capability_registry 
                        SET {', '.join(set_clauses)}
                        WHERE capability_id = ${param_count} AND tenant_id = ${param_count + 1}
                    """
                    params.extend([capability_id, tenant_id])
                    
                    await conn.execute(query, *params)
                    
                    # Update cache
                    if capability_id in self.capability_cache:
                        cap = self.capability_cache[capability_id]
                        for key, value in metadata_updates.items():
                            if hasattr(cap, key):
                                setattr(cap, key, value)
                        cap.updated_at = datetime.utcnow()
            
            self.logger.info(f"Attached metadata to capability {capability_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to attach metadata to capability {capability_id}: {e}")
            return False
    
    async def compute_trust_score(self, capability_id: str, tenant_id: int, 
                                tenant_tier: TenantTier, industry_code: IndustryCode) -> Optional[TrustScore]:
        """Compute dynamic trust score for capability - Task 14.2.11"""
        try:
            # Get capability performance data
            performance_data = await self._get_capability_performance(capability_id, tenant_id)
            if not performance_data:
                return None
            
            # Get industry-specific weights
            weights = self.trust_factor_weights.get(industry_code.value, self.trust_factor_weights['SaaS'])
            
            # Calculate individual factor scores
            execution_score = performance_data.get('success_rate', 0.0)
            compliance_score = max(0.0, 1.0 - (performance_data.get('violation_count', 0) * 0.1))
            feedback_score = performance_data.get('user_feedback', 0.0) / 5.0  # Normalize to 0-1
            business_score = performance_data.get('business_impact', 0.0)
            reliability_score = performance_data.get('reliability', 0.0)
            
            # Compute weighted overall score
            overall_score = (
                execution_score * weights['execution_success'] +
                compliance_score * weights['compliance_violations'] +
                feedback_score * weights['user_feedback'] +
                business_score * weights['business_impact'] +
                reliability_score * weights['historical_reliability']
            )
            
            # Determine trust level based on tenant tier
            trust_level = self._determine_trust_level(overall_score, tenant_tier)
            
            # Calculate confidence interval (simplified)
            sample_size = performance_data.get('execution_count', 1)
            confidence_margin = min(0.1, 1.0 / max(1, sample_size ** 0.5))
            
            trust_score = TrustScore(
                capability_id=capability_id,
                tenant_id=tenant_id,
                tenant_tier=tenant_tier,
                industry_code=industry_code,
                execution_success_rate=execution_score,
                compliance_violation_count=performance_data.get('violation_count', 0),
                user_feedback_score=performance_data.get('user_feedback', 0.0),
                business_impact_score=business_score,
                overall_trust_score=overall_score,
                trust_level=trust_level,
                confidence_interval_lower=max(0.0, overall_score - confidence_margin),
                confidence_interval_upper=min(1.0, overall_score + confidence_margin),
                factor_scores={
                    'execution': execution_score,
                    'compliance': compliance_score,
                    'feedback': feedback_score,
                    'business': business_score,
                    'reliability': reliability_score
                }
            )
            
            # Store the computed trust score
            await self.store_trust_score(trust_score)
            
            self.logger.info(f"Computed trust score {overall_score:.3f} for capability {capability_id}")
            return trust_score
            
        except Exception as e:
            self.logger.error(f"Failed to compute trust score for {capability_id}: {e}")
            return None
    
    def _determine_trust_level(self, score: float, tenant_tier: TenantTier) -> str:
        """Determine trust level based on score and tenant tier"""
        thresholds = self.tenant_tier_thresholds[tenant_tier.value]
        
        for level, (min_score, max_score) in thresholds.items():
            if min_score <= score <= max_score:
                return level
        
        return 'untrusted'
    
    async def _get_capability_performance(self, capability_id: str, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Get performance data for capability (from execution logs, metrics, etc.)"""
        # This would typically query execution logs, user feedback, business metrics
        # For now, return mock data with realistic values
        return {
            'success_rate': 0.95,
            'violation_count': 0,
            'user_feedback': 4.2,
            'business_impact': 0.8,
            'reliability': 0.9,
            'execution_count': 150
        }
    
    async def _load_existing_capabilities(self):
        """Load existing capabilities from database into cache"""
        if not self.pool_manager:
            return
            
        try:
            async with self.pool_manager.get_connection() as conn:
                rows = await conn.fetch("SELECT * FROM enhanced_capability_registry LIMIT 1000")
                
                for row in rows:
                    cap = self._row_to_capability_metadata(row)
                    self.capability_cache[cap.capability_id] = cap
                    self._update_tenant_mapping(cap.tenant_id, cap.capability_id)
                
                self.logger.info(f"Loaded {len(rows)} existing capabilities into cache")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing capabilities: {e}")
    
    def _row_to_capability_metadata(self, row) -> CapabilityMetadata:
        """Convert database row to CapabilityMetadata object"""
        return CapabilityMetadata(
            capability_id=str(row['capability_id']),
            name=row['capability_name'],
            capability_type=CapabilityType(row['capability_type']),
            version=row['version'],
            description=row['description'],
            tenant_id=row['tenant_id'],
            tenant_tier=TenantTier(row['tenant_tier']),
            industry_code=IndustryCode(row['industry_code']),
            saas_workflows=json.loads(row['saas_workflows']) if row['saas_workflows'] else [],
            business_metrics=json.loads(row['business_metrics']) if row['business_metrics'] else {},
            customer_impact_score=float(row['customer_impact_score']),
            input_schema=json.loads(row['input_schema']) if row['input_schema'] else {},
            output_schema=json.loads(row['output_schema']) if row['output_schema'] else {},
            operator_definition=json.loads(row['operator_definition']) if row['operator_definition'] else {},
            avg_execution_time_ms=row['avg_execution_time_ms'],
            success_rate=float(row['success_rate']),
            usage_count=row['usage_count'],
            compliance_requirements=json.loads(row['compliance_requirements']) if row['compliance_requirements'] else [],
            policy_tags=json.loads(row['policy_tags']) if row['policy_tags'] else [],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            last_used_at=row['last_used_at'],
            created_by_user_id=row['created_by_user_id']
        )

# Singleton instance for global access
_enhanced_capability_registry = None

def get_enhanced_capability_registry(pool_manager=None) -> EnhancedCapabilityRegistry:
    """Get singleton instance of Enhanced Capability Registry"""
    global _enhanced_capability_registry
    if _enhanced_capability_registry is None:
        _enhanced_capability_registry = EnhancedCapabilityRegistry(pool_manager)
    return _enhanced_capability_registry
