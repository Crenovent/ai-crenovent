"""
Dynamic Decision Logging System
==============================
Implements immutable decision logging with full tenant separation.
Tasks: 15.2.14, 15.6.15 - Store every decision immutably with tenant context

Features:
- Tenant-isolated decision logging
- Dynamic industry-specific decision types
- Immutable audit trails with hash chaining
- SaaS-focused decision categories
- Real-time decision analytics
- Compliance-ready evidence generation
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Dynamic decision types - extensible per industry"""
    # SaaS-specific decisions
    SUBSCRIPTION_APPROVAL = "subscription_approval"
    BILLING_ADJUSTMENT = "billing_adjustment"
    CHURN_INTERVENTION = "churn_intervention"
    REVENUE_RECOGNITION = "revenue_recognition"
    CUSTOMER_SUCCESS_ACTION = "customer_success_action"
    USAGE_LIMIT_OVERRIDE = "usage_limit_override"
    
    # General business decisions
    POLICY_ENFORCEMENT = "policy_enforcement"
    WORKFLOW_ROUTING = "workflow_routing"
    CAPABILITY_SELECTION = "capability_selection"
    ESCALATION_TRIGGER = "escalation_trigger"
    
    # Governance decisions
    COMPLIANCE_OVERRIDE = "compliance_override"
    AUDIT_EXCEPTION = "audit_exception"
    RISK_ASSESSMENT = "risk_assessment"

class DecisionOutcome(Enum):
    APPROVED = "approved"
    DENIED = "denied"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    AUTOMATED = "automated"

@dataclass
class DecisionContext:
    """Dynamic decision context - no hardcoded fields"""
    tenant_id: int
    tenant_tier: str  # T0, T1, T2
    industry_code: str  # SaaS, Banking, Insurance
    user_id: Optional[int] = None
    capability_id: Optional[str] = None
    workflow_id: Optional[str] = None
    business_context: Dict[str, Any] = None  # SaaS: ARR, MRR, customer_id, etc.
    compliance_context: Dict[str, Any] = None  # Policy violations, overrides, etc.
    
    def __post_init__(self):
        if self.business_context is None:
            self.business_context = {}
        if self.compliance_context is None:
            self.compliance_context = {}

@dataclass
class DecisionRecord:
    """Immutable decision record with tenant isolation"""
    decision_id: str
    tenant_id: int
    decision_type: DecisionType
    decision_outcome: DecisionOutcome
    context: DecisionContext
    reasoning: str
    confidence_score: float
    automated: bool
    decision_maker: str  # user_id or system component
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time_ms: float
    created_at: datetime
    hash_signature: str  # For tamper-evident logging
    previous_hash: Optional[str] = None  # Hash chain

class DecisionLoggingSystem:
    """
    Dynamic tenant-aware decision logging system
    
    Implements:
    - Task 15.2.14: Log all pass/fail decisions immutably
    - Task 15.6.15: Store every decision with tenant context
    - Tamper-evident logging with hash chaining
    - Dynamic decision types per industry
    - Real-time decision analytics
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Dynamic industry-specific decision configurations
        self.industry_decision_config = {
            'SaaS': {
                'critical_decisions': [
                    DecisionType.SUBSCRIPTION_APPROVAL,
                    DecisionType.BILLING_ADJUSTMENT,
                    DecisionType.REVENUE_RECOGNITION
                ],
                'business_metrics': ['arr_impact', 'mrr_impact', 'customer_count', 'churn_rate'],
                'compliance_requirements': ['SOX_SAAS', 'GDPR_SAAS'],
                'escalation_thresholds': {
                    'high_value': 100000,  # $100k+ decisions
                    'customer_impact': 1000,  # 1000+ customers affected
                    'compliance_risk': 0.8   # High compliance risk
                }
            },
            'Banking': {  # Future extensibility
                'critical_decisions': [DecisionType.RISK_ASSESSMENT, DecisionType.COMPLIANCE_OVERRIDE],
                'business_metrics': ['credit_risk', 'regulatory_capital', 'npa_ratio'],
                'compliance_requirements': ['RBI', 'BASEL_III'],
                'escalation_thresholds': {'credit_limit': 1000000, 'regulatory_breach': 0.9}
            },
            'Insurance': {  # Future extensibility
                'critical_decisions': [DecisionType.RISK_ASSESSMENT, DecisionType.POLICY_ENFORCEMENT],
                'business_metrics': ['claims_ratio', 'premium_growth', 'solvency_ratio'],
                'compliance_requirements': ['NAIC', 'SOLVENCY_II'],
                'escalation_thresholds': {'claims_amount': 500000, 'solvency_risk': 0.85}
            }
        }
        
        # Hash chain for tamper-evident logging (per tenant)
        self.tenant_hash_chains: Dict[int, str] = {}
        
        # Decision analytics cache
        self.analytics_cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(minutes=10)
        
    async def initialize(self) -> bool:
        """Initialize decision logging system"""
        try:
            self.logger.info("📝 Initializing Decision Logging System...")
            
            # Initialize database tables
            await self._initialize_decision_tables()
            
            # Load existing hash chains
            await self._load_hash_chains()
            
            self.logger.info("✅ Decision Logging System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Decision Logging System: {e}")
            return False
    
    async def log_decision(self, decision_type: DecisionType, outcome: DecisionOutcome,
                          context: DecisionContext, reasoning: str, confidence_score: float,
                          automated: bool, decision_maker: str, input_data: Dict[str, Any],
                          output_data: Dict[str, Any], execution_time_ms: float) -> str:
        """
        Log a decision with full tenant isolation and tamper-evident logging
        
        Returns:
            decision_id: Unique identifier for the logged decision
        """
        try:
            decision_id = str(uuid.uuid4())
            
            # Get previous hash for chain (tenant-specific)
            previous_hash = self.tenant_hash_chains.get(context.tenant_id, "genesis")
            
            # Create decision record
            decision_record = DecisionRecord(
                decision_id=decision_id,
                tenant_id=context.tenant_id,
                decision_type=decision_type,
                decision_outcome=outcome,
                context=context,
                reasoning=reasoning,
                confidence_score=confidence_score,
                automated=automated,
                decision_maker=decision_maker,
                input_data=input_data,
                output_data=output_data,
                execution_time_ms=execution_time_ms,
                created_at=datetime.utcnow(),
                hash_signature="",  # Will be calculated
                previous_hash=previous_hash
            )
            
            # Calculate hash signature for tamper-evident logging
            decision_record.hash_signature = self._calculate_decision_hash(decision_record)
            
            # Store decision with tenant isolation
            await self._store_decision_record(decision_record)
            
            # Update hash chain for tenant
            self.tenant_hash_chains[context.tenant_id] = decision_record.hash_signature
            
            # Check for escalation requirements
            await self._check_escalation_requirements(decision_record)
            
            # Update real-time analytics
            await self._update_decision_analytics(decision_record)
            
            self.logger.info(f"✅ Decision logged: {decision_id} (tenant {context.tenant_id})")
            return decision_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log decision: {e}")
            raise
    
    async def get_tenant_decision_analytics(self, tenant_id: int, 
                                          time_window_days: int = 30) -> Dict[str, Any]:
        """Get decision analytics for a specific tenant"""
        try:
            cache_key = f"analytics_{tenant_id}_{time_window_days}"
            
            # Check cache
            if cache_key in self.analytics_cache:
                cached_data, cached_time = self.analytics_cache[cache_key]
                if datetime.utcnow() - cached_time < self.cache_ttl:
                    return cached_data
            
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return {}
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, true)", str(tenant_id))
                
                # Query decision analytics (automatically tenant-scoped)
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_decisions,
                        COUNT(CASE WHEN decision_outcome = 'approved' THEN 1 END) as approved_count,
                        COUNT(CASE WHEN decision_outcome = 'denied' THEN 1 END) as denied_count,
                        COUNT(CASE WHEN decision_outcome = 'escalated' THEN 1 END) as escalated_count,
                        COUNT(CASE WHEN automated = true THEN 1 END) as automated_count,
                        AVG(confidence_score) as avg_confidence,
                        AVG(execution_time_ms) as avg_execution_time,
                        COUNT(DISTINCT decision_type) as unique_decision_types
                    FROM decision_records 
                    WHERE tenant_id = $1 
                      AND created_at >= NOW() - INTERVAL '%s days'
                """, tenant_id, time_window_days)
                
                # Query decision type breakdown
                type_breakdown = await conn.fetch("""
                    SELECT 
                        decision_type,
                        COUNT(*) as count,
                        AVG(confidence_score) as avg_confidence
                    FROM decision_records 
                    WHERE tenant_id = $1 
                      AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY decision_type
                    ORDER BY count DESC
                """, tenant_id, time_window_days)
                
                analytics = {
                    'tenant_id': tenant_id,
                    'time_window_days': time_window_days,
                    'summary': {
                        'total_decisions': result['total_decisions'],
                        'approved_count': result['approved_count'],
                        'denied_count': result['denied_count'],
                        'escalated_count': result['escalated_count'],
                        'automated_count': result['automated_count'],
                        'approval_rate': result['approved_count'] / max(result['total_decisions'], 1),
                        'automation_rate': result['automated_count'] / max(result['total_decisions'], 1),
                        'avg_confidence': float(result['avg_confidence'] or 0),
                        'avg_execution_time_ms': float(result['avg_execution_time'] or 0),
                        'unique_decision_types': result['unique_decision_types']
                    },
                    'decision_type_breakdown': [
                        {
                            'decision_type': row['decision_type'],
                            'count': row['count'],
                            'avg_confidence': float(row['avg_confidence'] or 0)
                        }
                        for row in type_breakdown
                    ],
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                # Cache the result
                self.analytics_cache[cache_key] = (analytics, datetime.utcnow())
                
                return analytics
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get decision analytics for tenant {tenant_id}: {e}")
            return {}
    
    def _calculate_decision_hash(self, decision_record: DecisionRecord) -> str:
        """Calculate tamper-evident hash for decision record"""
        # Create deterministic hash input
        hash_input = {
            'decision_id': decision_record.decision_id,
            'tenant_id': decision_record.tenant_id,
            'decision_type': decision_record.decision_type.value,
            'decision_outcome': decision_record.decision_outcome.value,
            'reasoning': decision_record.reasoning,
            'confidence_score': decision_record.confidence_score,
            'created_at': decision_record.created_at.isoformat(),
            'previous_hash': decision_record.previous_hash
        }
        
        # Calculate SHA-256 hash
        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    async def _initialize_decision_tables(self):
        """Initialize decision logging tables"""
        if not self.pool_manager or not self.pool_manager.postgres_pool:
            self.logger.warning("⚠️ PostgreSQL pool not available for decision tables")
            return
        
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Create decision records table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS decision_records (
                        decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id INTEGER NOT NULL,
                        decision_type VARCHAR(100) NOT NULL,
                        decision_outcome VARCHAR(50) NOT NULL,
                        reasoning TEXT NOT NULL,
                        confidence_score DECIMAL(5,4) NOT NULL,
                        automated BOOLEAN NOT NULL DEFAULT false,
                        decision_maker VARCHAR(255) NOT NULL,
                        input_data JSONB NOT NULL DEFAULT '{}',
                        output_data JSONB NOT NULL DEFAULT '{}',
                        execution_time_ms DECIMAL(10,2) NOT NULL,
                        business_context JSONB DEFAULT '{}',
                        compliance_context JSONB DEFAULT '{}',
                        hash_signature VARCHAR(64) NOT NULL,
                        previous_hash VARCHAR(64),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        
                        CONSTRAINT valid_confidence_score CHECK (confidence_score >= 0 AND confidence_score <= 1),
                        CONSTRAINT valid_execution_time CHECK (execution_time_ms >= 0)
                    );
                """)
                
                # Enable RLS for tenant isolation
                await conn.execute("ALTER TABLE decision_records ENABLE ROW LEVEL SECURITY;")
                await conn.execute("""
                    DROP POLICY IF EXISTS decision_records_rls_policy ON decision_records;
                    CREATE POLICY decision_records_rls_policy ON decision_records
                        FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
                """)
                
                # Create indexes for performance
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_records_tenant_created ON decision_records(tenant_id, created_at);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_records_type ON decision_records(decision_type);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_records_outcome ON decision_records(decision_outcome);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_records_hash ON decision_records(hash_signature);")
                
                self.logger.info("✅ Decision logging tables initialized successfully")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize decision tables: {e}")
    
    async def _store_decision_record(self, decision_record: DecisionRecord):
        """Store decision record with tenant isolation"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, true)", str(decision_record.tenant_id))
                
                await conn.execute("""
                    INSERT INTO decision_records 
                    (decision_id, tenant_id, decision_type, decision_outcome, reasoning, 
                     confidence_score, automated, decision_maker, input_data, output_data,
                     execution_time_ms, business_context, compliance_context, 
                     hash_signature, previous_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, 
                decision_record.decision_id, decision_record.tenant_id,
                decision_record.decision_type.value, decision_record.decision_outcome.value,
                decision_record.reasoning, decision_record.confidence_score,
                decision_record.automated, decision_record.decision_maker,
                json.dumps(decision_record.input_data), json.dumps(decision_record.output_data),
                decision_record.execution_time_ms,
                json.dumps(decision_record.context.business_context),
                json.dumps(decision_record.context.compliance_context),
                decision_record.hash_signature, decision_record.previous_hash)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store decision record: {e}")
            raise
    
    async def _load_hash_chains(self):
        """Load existing hash chains for all tenants"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Get latest hash for each tenant
                results = await conn.fetch("""
                    SELECT DISTINCT ON (tenant_id) 
                        tenant_id, hash_signature
                    FROM decision_records 
                    ORDER BY tenant_id, created_at DESC
                """)
                
                for row in results:
                    self.tenant_hash_chains[row['tenant_id']] = row['hash_signature']
                
                self.logger.info(f"✅ Loaded hash chains for {len(self.tenant_hash_chains)} tenants")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load hash chains: {e}")
    
    async def _check_escalation_requirements(self, decision_record: DecisionRecord):
        """Check if decision requires escalation based on dynamic thresholds"""
        try:
            industry_config = self.industry_decision_config.get(
                decision_record.context.industry_code, 
                self.industry_decision_config['SaaS']
            )
            
            escalation_thresholds = industry_config['escalation_thresholds']
            
            # Check various escalation triggers
            should_escalate = False
            escalation_reasons = []
            
            # High-value decision check
            if 'high_value' in escalation_thresholds:
                business_value = decision_record.context.business_context.get('financial_impact', 0)
                if business_value > escalation_thresholds['high_value']:
                    should_escalate = True
                    escalation_reasons.append(f"High financial impact: ${business_value}")
            
            # Customer impact check
            if 'customer_impact' in escalation_thresholds:
                customer_count = decision_record.context.business_context.get('affected_customers', 0)
                if customer_count > escalation_thresholds['customer_impact']:
                    should_escalate = True
                    escalation_reasons.append(f"High customer impact: {customer_count} customers")
            
            # Compliance risk check
            if 'compliance_risk' in escalation_thresholds:
                compliance_risk = decision_record.context.compliance_context.get('risk_score', 0)
                if compliance_risk > escalation_thresholds['compliance_risk']:
                    should_escalate = True
                    escalation_reasons.append(f"High compliance risk: {compliance_risk}")
            
            if should_escalate:
                await self._trigger_escalation(decision_record, escalation_reasons)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to check escalation requirements: {e}")
    
    async def _trigger_escalation(self, decision_record: DecisionRecord, reasons: List[str]):
        """Trigger escalation for high-risk decisions"""
        self.logger.warning(f"🚨 Escalation triggered for decision {decision_record.decision_id}: {', '.join(reasons)}")
        # Implementation would trigger actual escalation workflow
    
    async def _update_decision_analytics(self, decision_record: DecisionRecord):
        """Update real-time decision analytics"""
        # Clear relevant cache entries
        cache_keys_to_clear = [
            key for key in self.analytics_cache.keys() 
            if f"analytics_{decision_record.tenant_id}" in key
        ]
        for key in cache_keys_to_clear:
            del self.analytics_cache[key]

# Global instance
decision_logging_system = None

def get_decision_logging_system(pool_manager=None):
    """Get or create decision logging system instance"""
    global decision_logging_system
    if decision_logging_system is None:
        decision_logging_system = DecisionLoggingSystem(pool_manager)
    return decision_logging_system
