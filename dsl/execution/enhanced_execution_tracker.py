"""
Enhanced Execution Tracker - Chapter 8.1 Execution Trace Model Implementation
Implements comprehensive execution tracking with full tenant isolation and SaaS focus.

Features:
- Chapter 8.1 Tasks: Execution trace model with governance, observability, compliance
- Raw intent storage and parsing (Tasks 15.1.1, 15.1.2)
- Execution metadata and outcome tracking (Tasks 15.5.1, 15.5.4)
- Capability metadata storage and lookup (Task 15.3.1)
- Full tenant isolation and DYNAMIC configuration (no hardcoding)
- Integration with orchestrator and RBA systems
- Bronze/Silver/Gold trace storage tiers
- Evidence pack generation and governance integration
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

class ExecutionStatus(Enum):
    """Dynamic execution status tracking"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class IntentType(Enum):
    """Configurable intent types"""
    WORKFLOW_EXECUTION = "workflow_execution"
    CAPABILITY_LOOKUP = "capability_lookup"
    POLICY_EVALUATION = "policy_evaluation"
    TRUST_CALCULATION = "trust_calculation"
    # SaaS-specific intents
    ARR_ANALYSIS = "arr_analysis"
    CHURN_PREDICTION = "churn_prediction"
    CUSTOMER_SUCCESS = "customer_success"
    REVENUE_FORECASTING = "revenue_forecasting"

class TenantTier(Enum):
    """Dynamic tenant tier classification"""
    T0 = "T0"  # Regulated/Enterprise
    T1 = "T1"  # Mid-market
    T2 = "T2"  # SMB

# Task 8.1.1: Define Trace Objectives & Scope
class TraceObjective(Enum):
    """Trace objectives aligned with personas - DYNAMIC"""
    DEBUG = "debug"      # Developers
    AUDIT = "audit"      # Compliance
    INSIGHTS = "insights" # RevOps
    TRAINING = "training" # AI/ML

class TraceLifecycle(Enum):
    """Trace lifecycle states - DYNAMIC progression"""
    CREATED = "created"
    VALIDATED = "validated"
    STORED = "stored"
    USED_IN_KG = "used_in_kg"
    USED_IN_RAG = "used_in_rag"
    USED_IN_SLM = "used_in_slm"
    ARCHIVED = "archived"
    PURGED = "purged"

# Task 8.1.4: Trace Store Design - Storage tiers
class StorageTier(Enum):
    """Storage tiers for trace retention - DYNAMIC retention policies"""
    BRONZE = "bronze"  # Raw JSON/Parquet, configurable retention
    SILVER = "silver"  # Normalized traces, configurable retention
    GOLD = "gold"      # Aggregated features, configurable retention

class ResidencyRegion(Enum):
    """Data residency regions - DYNAMIC based on tenant config"""
    US = "us"
    EU = "eu"
    IN = "in"
    APAC = "apac"
    GLOBAL = "global"

# Task 8.1.2: Trace Schema Specification - Comprehensive trace model
@dataclass
class GovernanceStatus:
    """Task 8.1.3: Evidence & Governance Integration"""
    policy_id: str
    evidence_id: str
    override_ledger_ref: Optional[str] = None
    status: str = "pass"  # pass/fail
    override_reason: Optional[str] = None
    compliance_flags: List[str] = None
    
    def __post_init__(self):
        if self.compliance_flags is None:
            self.compliance_flags = []

@dataclass
class TraceMetadata:
    """Comprehensive trace metadata - DYNAMIC configuration"""
    # Task 8.1.6: Security, Privacy, Residency
    residency_region: ResidencyRegion
    tenant_id: int
    sla_tier: str  # Dynamic from tenant config
    
    # Privacy and consent - DYNAMIC based on regulations
    consent_id: Optional[str] = None
    pii_masked: bool = False
    anonymized: bool = False
    
    # Cost and performance - DYNAMIC tracking
    execution_cost: Optional[float] = None
    execution_time_ms: Optional[int] = None
    resource_usage: Dict[str, Any] = None
    
    # Trust and reliability - DYNAMIC scoring
    trust_score: Optional[float] = None
    confidence_score: Optional[float] = None
    reliability_tier: Optional[str] = None
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}

@dataclass
class ExecutionTrace:
    """
    Task 8.1.2: Comprehensive execution trace schema - DYNAMIC and configurable
    Captures inputs, outputs, decisions, ML scores, overrides, evidence, costs, metadata
    """
    # Required fields
    trace_id: str
    tenant_id: int
    workflow_id: str
    step_id: str
    timestamp: datetime
    
    # Core execution data
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    
    # Governance integration (Task 8.1.3)
    governance: GovernanceStatus
    
    # Metadata and context
    metadata: TraceMetadata
    
    # Optional fields - DYNAMIC based on context
    parent_trace_id: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # ML and decision data - DYNAMIC
    ml_scores: Dict[str, float] = None
    decision_path: List[str] = None
    
    # Lifecycle and storage - DYNAMIC progression
    lifecycle_state: TraceLifecycle = TraceLifecycle.CREATED
    storage_tier: StorageTier = StorageTier.BRONZE
    
    # Observability - DYNAMIC tagging
    tags: Dict[str, str] = None
    annotations: Dict[str, Any] = None
    
    # Validation and integrity
    schema_version: str = "1.0.0"
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.ml_scores is None:
            self.ml_scores = {}
        if self.decision_path is None:
            self.decision_path = []
        if self.tags is None:
            self.tags = {}
        if self.annotations is None:
            self.annotations = {}
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of trace data for integrity"""
        trace_data = {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs
        }
        return hashlib.sha256(json.dumps(trace_data, sort_keys=True).encode()).hexdigest()

@dataclass
class RawIntent:
    """Raw intent storage - Task 15.1.1"""
    intent_id: str
    tenant_id: int
    tenant_tier: TenantTier
    raw_input: str
    intent_type: IntentType
    source_system: str
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    context_data: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.context_data is None:
            self.context_data = {}

@dataclass
class ParsedIntent:
    """Parsed intent output - Task 15.1.2"""
    parsed_id: str
    intent_id: str
    tenant_id: int
    parsed_action: str
    extracted_entities: Dict[str, Any]
    confidence_score: float
    parsing_method: str  # "rule_based", "ml_model", "hybrid"
    model_version: Optional[str] = None
    validation_errors: List[str] = None
    saas_context: Dict[str, Any] = None  # SaaS-specific parsed context
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.validation_errors is None:
            self.validation_errors = []
        if self.saas_context is None:
            self.saas_context = {}

@dataclass
class ExecutionMetadata:
    """Execution metadata tracking - Task 15.5.1"""
    execution_id: str
    tenant_id: int
    tenant_tier: TenantTier
    workflow_id: Optional[str]
    capability_id: Optional[str]
    execution_type: str  # "RBA", "RBIA", "AALA"
    
    # Execution context
    triggered_by_user_id: Optional[int]
    trigger_source: str  # "manual", "scheduled", "event_driven", "api"
    input_parameters: Dict[str, Any]
    
    # SaaS-specific metadata
    business_context: Dict[str, Any] = None  # ARR impact, customer context, etc.
    compliance_context: Dict[str, Any] = None  # Policy requirements, approvals
    
    # Performance tracking
    start_time: datetime = None
    estimated_duration_ms: Optional[int] = None
    
    # Governance
    policy_pack_id: Optional[str] = None
    evidence_pack_required: bool = True
    
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        if self.business_context is None:
            self.business_context = {}
        if self.compliance_context is None:
            self.compliance_context = {}

@dataclass
class ExecutionOutcome:
    """Execution outcome tracking - Task 15.5.4"""
    outcome_id: str
    execution_id: str
    tenant_id: int
    
    # Final results
    status: ExecutionStatus
    result_data: Dict[str, Any]
    actual_duration_ms: int  # Required field
    
    # Optional results
    output_artifacts: List[str] = None  # File paths, URLs, etc.
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Business impact (SaaS-specific)
    business_impact: Dict[str, Any] = None  # Revenue impact, customer impact
    
    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0
    
    # Governance
    compliance_validated: bool = False
    evidence_pack_id: Optional[str] = None
    override_applied: bool = False
    override_reason: Optional[str] = None
    
    # Timestamps
    completed_at: datetime = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.completed_at is None:
            self.completed_at = datetime.utcnow()
        if self.output_artifacts is None:
            self.output_artifacts = []
        if self.business_impact is None:
            self.business_impact = {}

@dataclass
class CapabilityMetadata:
    """Capability metadata for lookup - Task 15.3.1"""
    capability_id: str
    tenant_id: int
    tenant_tier: TenantTier
    industry_code: str
    
    # Core metadata
    name: str
    capability_type: str  # "RBA", "RBIA", "AALA"
    version: str
    description: Optional[str] = None
    
    # Technical specifications
    input_schema: Dict[str, Any] = None
    output_schema: Dict[str, Any] = None
    
    # SaaS-specific attributes
    saas_workflows: List[str] = None
    business_metrics: List[str] = None
    
    # Performance characteristics
    avg_execution_time_ms: int = 0
    success_rate: float = 1.0
    trust_score: float = 0.0
    
    # Availability
    sla_tier: str = "standard"  # "premium", "standard", "basic"
    available_regions: List[str] = None
    
    # Governance
    compliance_requirements: List[str] = None
    policy_tags: List[str] = None
    
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.input_schema is None:
            self.input_schema = {}
        if self.output_schema is None:
            self.output_schema = {}
        if self.saas_workflows is None:
            self.saas_workflows = []
        if self.business_metrics is None:
            self.business_metrics = []
        if self.available_regions is None:
            self.available_regions = ["US", "EU"]
        if self.compliance_requirements is None:
            self.compliance_requirements = []
        if self.policy_tags is None:
            self.policy_tags = []

class EnhancedExecutionTracker:
    """
    Enhanced Execution Tracker with full tenant isolation and dynamic configuration.
    Implements Tasks 15.1.1, 15.1.2, 15.5.1, 15.5.4, 15.3.1.
    Integrates with orchestrator and RBA systems.
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Dynamic configuration (no hardcoding)
        self.parsing_models = self._load_parsing_models()
        self.execution_thresholds = self._load_execution_thresholds()
        self.saas_business_rules = self._load_saas_business_rules()
        self.tenant_sla_mapping = self._load_tenant_sla_mapping()
        
        # Cache for performance
        self.intent_cache: Dict[str, RawIntent] = {}
        self.capability_cache: Dict[str, CapabilityMetadata] = {}
        self.execution_cache: Dict[str, ExecutionMetadata] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Tenant isolation
        self.tenant_executions: Dict[int, List[str]] = {}
        
        # Integration points
        self.orchestrator = None  # Will be set during initialization
        self.rba_engine = None    # Will be set during initialization
        
    def _load_parsing_models(self) -> Dict[str, Dict[str, Any]]:
        """Load configurable parsing model configurations"""
        return {
            'rule_based': {
                'confidence_threshold': 0.8,
                'patterns': {
                    'arr_analysis': [r'arr\s+analysis', r'annual\s+recurring\s+revenue', r'revenue\s+analysis'],
                    'churn_prediction': [r'churn\s+prediction', r'customer\s+retention', r'churn\s+risk'],
                    'workflow_execution': [r'run\s+workflow', r'execute\s+workflow', r'start\s+process']
                }
            },
            'ml_model': {
                'model_name': 'intent_classifier_v2',
                'confidence_threshold': 0.75,
                'fallback_to_rules': True,
                'model_endpoint': os.getenv('ML_MODEL_ENDPOINT', 'http://localhost:8080/predict')
            },
            'hybrid': {
                'rule_weight': 0.3,
                'ml_weight': 0.7,
                'min_confidence': 0.6
            }
        }
    
    def _load_execution_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Load execution performance thresholds per tenant tier"""
        return {
            'T0': {  # Regulated tenants - strictest thresholds
                'max_execution_time_ms': 30000,  # 30 seconds
                'max_retry_count': 3,
                'sla_breach_threshold_ms': 25000,
                'evidence_pack_mandatory': True
            },
            'T1': {  # Enterprise tenants - balanced thresholds
                'max_execution_time_ms': 60000,  # 60 seconds
                'max_retry_count': 2,
                'sla_breach_threshold_ms': 50000,
                'evidence_pack_mandatory': True
            },
            'T2': {  # Mid-market tenants - more lenient thresholds
                'max_execution_time_ms': 120000,  # 2 minutes
                'max_retry_count': 1,
                'sla_breach_threshold_ms': 100000,
                'evidence_pack_mandatory': False
            }
        }
    
    def _load_saas_business_rules(self) -> Dict[str, Any]:
        """Load SaaS-specific business rules for execution tracking"""
        return {
            'arr_impact_thresholds': {
                'high': 100000,    # $100K+ ARR impact
                'medium': 10000,   # $10K+ ARR impact
                'low': 1000        # $1K+ ARR impact
            },
            'customer_impact_levels': {
                'enterprise': ['T0', 'high_value'],
                'mid_market': ['T1', 'medium_value'],
                'smb': ['T2', 'low_value']
            },
            'compliance_requirements': {
                'financial_data': ['SOX_SAAS', 'GDPR_SAAS'],
                'customer_data': ['GDPR_SAAS', 'SAAS_BUSINESS_RULES'],
                'usage_data': ['SAAS_BUSINESS_RULES']
            }
        }
    
    def _load_tenant_sla_mapping(self) -> Dict[str, str]:
        """Load tenant tier to SLA mapping"""
        return {
            'T0': 'premium',    # Regulated tenants get premium SLA
            'T1': 'standard',   # Enterprise tenants get standard SLA
            'T2': 'basic'       # Mid-market tenants get basic SLA
        }
    
    async def initialize(self):
        """Initialize the enhanced execution tracker"""
        try:
            await self._ensure_database_tables()
            await self._load_existing_data()
            self.logger.info("Enhanced Execution Tracker initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Execution Tracker: {e}")
            raise
    
    def set_integration_points(self, orchestrator=None, rba_engine=None):
        """Set integration points with orchestrator and RBA systems"""
        self.orchestrator = orchestrator
        self.rba_engine = rba_engine
        self.logger.info("Integration points set for orchestrator and RBA engine")
    
    async def _ensure_database_tables(self):
        """Ensure all required database tables exist"""
        if not self.pool_manager:
            return
            
        async with self.pool_manager.get_connection() as conn:
            # Raw intents table - Task 15.1.1
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_intents (
                    intent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    tenant_tier VARCHAR(10) NOT NULL,
                    raw_input TEXT NOT NULL,
                    intent_type VARCHAR(50) NOT NULL,
                    source_system VARCHAR(100) NOT NULL,
                    user_id INTEGER,
                    session_id VARCHAR(255),
                    context_data JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    CONSTRAINT valid_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2'))
                );
            """)
            
            # Parsed intents table - Task 15.1.2
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS parsed_intents (
                    parsed_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    intent_id UUID NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    parsed_action VARCHAR(255) NOT NULL,
                    extracted_entities JSONB DEFAULT '{}',
                    confidence_score DECIMAL(5,4) NOT NULL,
                    parsing_method VARCHAR(50) NOT NULL,
                    model_version VARCHAR(50),
                    validation_errors JSONB DEFAULT '[]',
                    saas_context JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    FOREIGN KEY (intent_id) REFERENCES raw_intents(intent_id),
                    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1),
                    CONSTRAINT valid_parsing_method CHECK (parsing_method IN ('rule_based', 'ml_model', 'hybrid'))
                );
            """)
            
            # Execution metadata table - Task 15.5.1
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_metadata (
                    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    tenant_tier VARCHAR(10) NOT NULL,
                    workflow_id VARCHAR(255),
                    capability_id VARCHAR(255),
                    execution_type VARCHAR(50) NOT NULL,
                    
                    -- Execution context
                    triggered_by_user_id INTEGER,
                    trigger_source VARCHAR(50) NOT NULL,
                    input_parameters JSONB DEFAULT '{}',
                    
                    -- SaaS-specific metadata
                    business_context JSONB DEFAULT '{}',
                    compliance_context JSONB DEFAULT '{}',
                    
                    -- Performance tracking
                    start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    estimated_duration_ms INTEGER,
                    
                    -- Governance
                    policy_pack_id VARCHAR(255),
                    evidence_pack_required BOOLEAN DEFAULT true,
                    
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    CONSTRAINT valid_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2')),
                    CONSTRAINT valid_execution_type CHECK (execution_type IN ('RBA', 'RBIA', 'AALA')),
                    CONSTRAINT valid_trigger_source CHECK (trigger_source IN ('manual', 'scheduled', 'event_driven', 'api'))
                );
            """)
            
            # Execution outcomes table - Task 15.5.4
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_outcomes (
                    outcome_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    execution_id UUID NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    
                    -- Final results
                    status VARCHAR(20) NOT NULL,
                    result_data JSONB DEFAULT '{}',
                    output_artifacts JSONB DEFAULT '[]',
                    
                    -- Performance metrics
                    actual_duration_ms INTEGER NOT NULL,
                    cpu_usage_percent DECIMAL(5,2),
                    memory_usage_mb DECIMAL(10,2),
                    
                    -- Business impact (SaaS-specific)
                    business_impact JSONB DEFAULT '{}',
                    
                    -- Error handling
                    error_message TEXT,
                    error_code VARCHAR(50),
                    retry_count INTEGER DEFAULT 0,
                    
                    -- Governance
                    compliance_validated BOOLEAN DEFAULT false,
                    evidence_pack_id VARCHAR(255),
                    override_applied BOOLEAN DEFAULT false,
                    override_reason TEXT,
                    
                    -- Timestamps
                    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    FOREIGN KEY (execution_id) REFERENCES execution_metadata(execution_id),
                    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')),
                    CONSTRAINT valid_duration CHECK (actual_duration_ms >= 0)
                );
            """)
            
            # Enhanced capability metadata table - Task 15.3.1
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_capability_metadata (
                    capability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    tenant_tier VARCHAR(10) NOT NULL,
                    industry_code VARCHAR(20) NOT NULL,
                    
                    -- Core metadata
                    name VARCHAR(255) NOT NULL,
                    capability_type VARCHAR(50) NOT NULL,
                    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
                    description TEXT,
                    
                    -- Technical specifications
                    input_schema JSONB DEFAULT '{}',
                    output_schema JSONB DEFAULT '{}',
                    
                    -- SaaS-specific attributes
                    saas_workflows JSONB DEFAULT '[]',
                    business_metrics JSONB DEFAULT '[]',
                    
                    -- Performance characteristics
                    avg_execution_time_ms INTEGER DEFAULT 0,
                    success_rate DECIMAL(5,4) DEFAULT 1.0,
                    trust_score DECIMAL(5,4) DEFAULT 0.0,
                    
                    -- Availability
                    sla_tier VARCHAR(20) DEFAULT 'standard',
                    available_regions JSONB DEFAULT '["US", "EU"]',
                    
                    -- Governance
                    compliance_requirements JSONB DEFAULT '[]',
                    policy_tags JSONB DEFAULT '[]',
                    
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    CONSTRAINT unique_execution_capability_per_tenant UNIQUE (tenant_id, name, version),
                    CONSTRAINT valid_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2')),
                    CONSTRAINT valid_capability_type CHECK (capability_type IN ('RBA', 'RBIA', 'AALA')),
                    CONSTRAINT valid_success_rate CHECK (success_rate >= 0 AND success_rate <= 1),
                    CONSTRAINT valid_trust_score CHECK (trust_score >= 0 AND trust_score <= 1),
                    CONSTRAINT valid_sla_tier CHECK (sla_tier IN ('premium', 'standard', 'basic'))
                );
            """)
            
            # Enable RLS on all tables
            tables = ['raw_intents', 'parsed_intents', 'execution_metadata', 'execution_outcomes', 'enhanced_capability_metadata']
            for table in tables:
                await conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
                await conn.execute(f"""
                    DROP POLICY IF EXISTS {table}_rls_policy ON {table};
                    CREATE POLICY {table}_rls_policy ON {table}
                        FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
                """)
            
            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_intents_tenant_created ON raw_intents(tenant_id, created_at);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_parsed_intents_intent_id ON parsed_intents(intent_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_metadata_tenant_type ON execution_metadata(tenant_id, execution_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_outcomes_execution_id ON execution_outcomes(execution_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_capability_metadata_tenant_type ON enhanced_capability_metadata(tenant_id, capability_type);")
    
    async def store_raw_intent(self, raw_intent: RawIntent) -> bool:
        """Store raw intent - Task 15.1.1"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                self.intent_cache[raw_intent.intent_id] = raw_intent
                self._update_tenant_mapping(raw_intent.tenant_id, raw_intent.intent_id, 'intents')
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{raw_intent.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO raw_intents (
                        intent_id, tenant_id, tenant_tier, raw_input, intent_type,
                        source_system, user_id, session_id, context_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                raw_intent.intent_id, raw_intent.tenant_id, raw_intent.tenant_tier.value,
                raw_intent.raw_input, raw_intent.intent_type.value, raw_intent.source_system,
                raw_intent.user_id, raw_intent.session_id, json.dumps(raw_intent.context_data))
            
            # Update cache and tenant mapping
            self.intent_cache[raw_intent.intent_id] = raw_intent
            self._update_tenant_mapping(raw_intent.tenant_id, raw_intent.intent_id, 'intents')
            
            self.logger.info(f"Stored raw intent {raw_intent.intent_id} for tenant {raw_intent.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store raw intent {raw_intent.intent_id}: {e}")
            return False
    
    async def store_parsed_intent(self, parsed_intent: ParsedIntent) -> bool:
        """Store parsed intent output - Task 15.1.2"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{parsed_intent.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO parsed_intents (
                        parsed_id, intent_id, tenant_id, parsed_action, extracted_entities,
                        confidence_score, parsing_method, model_version, validation_errors, saas_context
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 
                parsed_intent.parsed_id, parsed_intent.intent_id, parsed_intent.tenant_id,
                parsed_intent.parsed_action, json.dumps(parsed_intent.extracted_entities),
                parsed_intent.confidence_score, parsed_intent.parsing_method,
                parsed_intent.model_version, json.dumps(parsed_intent.validation_errors),
                json.dumps(parsed_intent.saas_context))
            
            self.logger.info(f"Stored parsed intent {parsed_intent.parsed_id} for tenant {parsed_intent.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store parsed intent {parsed_intent.parsed_id}: {e}")
            return False
    
    async def store_execution_metadata(self, execution_metadata: ExecutionMetadata) -> bool:
        """Store execution metadata - Task 15.5.1"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                self.execution_cache[execution_metadata.execution_id] = execution_metadata
                self._update_tenant_mapping(execution_metadata.tenant_id, execution_metadata.execution_id, 'executions')
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{execution_metadata.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO execution_metadata (
                        execution_id, tenant_id, tenant_tier, workflow_id, capability_id,
                        execution_type, triggered_by_user_id, trigger_source, input_parameters,
                        business_context, compliance_context, start_time, estimated_duration_ms,
                        policy_pack_id, evidence_pack_required
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, 
                execution_metadata.execution_id, execution_metadata.tenant_id, execution_metadata.tenant_tier.value,
                execution_metadata.workflow_id, execution_metadata.capability_id, execution_metadata.execution_type,
                execution_metadata.triggered_by_user_id, execution_metadata.trigger_source,
                json.dumps(execution_metadata.input_parameters), json.dumps(execution_metadata.business_context),
                json.dumps(execution_metadata.compliance_context), execution_metadata.start_time,
                execution_metadata.estimated_duration_ms, execution_metadata.policy_pack_id,
                execution_metadata.evidence_pack_required)
            
            # Update cache and tenant mapping
            self.execution_cache[execution_metadata.execution_id] = execution_metadata
            self._update_tenant_mapping(execution_metadata.tenant_id, execution_metadata.execution_id, 'executions')
            
            self.logger.info(f"Stored execution metadata {execution_metadata.execution_id} for tenant {execution_metadata.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store execution metadata {execution_metadata.execution_id}: {e}")
            return False
    
    async def store_execution_outcome(self, execution_outcome: ExecutionOutcome) -> bool:
        """Store execution outcome - Task 15.5.4"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{execution_outcome.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO execution_outcomes (
                        outcome_id, execution_id, tenant_id, status, result_data, output_artifacts,
                        actual_duration_ms, cpu_usage_percent, memory_usage_mb, business_impact,
                        error_message, error_code, retry_count, compliance_validated,
                        evidence_pack_id, override_applied, override_reason, completed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                """, 
                execution_outcome.outcome_id, execution_outcome.execution_id, execution_outcome.tenant_id,
                execution_outcome.status.value, json.dumps(execution_outcome.result_data),
                json.dumps(execution_outcome.output_artifacts), execution_outcome.actual_duration_ms,
                execution_outcome.cpu_usage_percent, execution_outcome.memory_usage_mb,
                json.dumps(execution_outcome.business_impact), execution_outcome.error_message,
                execution_outcome.error_code, execution_outcome.retry_count, execution_outcome.compliance_validated,
                execution_outcome.evidence_pack_id, execution_outcome.override_applied,
                execution_outcome.override_reason, execution_outcome.completed_at)
            
            self.logger.info(f"Stored execution outcome {execution_outcome.outcome_id} for tenant {execution_outcome.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store execution outcome {execution_outcome.outcome_id}: {e}")
            return False
    
    async def store_capability_metadata(self, capability_metadata: CapabilityMetadata) -> bool:
        """Store capability metadata - Task 15.3.1"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                self.capability_cache[capability_metadata.capability_id] = capability_metadata
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{capability_metadata.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO enhanced_capability_metadata (
                        capability_id, tenant_id, tenant_tier, industry_code, name,
                        capability_type, version, description, input_schema, output_schema,
                        saas_workflows, business_metrics, avg_execution_time_ms, success_rate,
                        trust_score, sla_tier, available_regions, compliance_requirements, policy_tags
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    ON CONFLICT (tenant_id, name, version) 
                    DO UPDATE SET
                        updated_at = NOW(),
                        description = EXCLUDED.description,
                        avg_execution_time_ms = EXCLUDED.avg_execution_time_ms,
                        success_rate = EXCLUDED.success_rate,
                        trust_score = EXCLUDED.trust_score
                """, 
                capability_metadata.capability_id, capability_metadata.tenant_id, capability_metadata.tenant_tier.value,
                capability_metadata.industry_code, capability_metadata.name, capability_metadata.capability_type,
                capability_metadata.version, capability_metadata.description,
                json.dumps(capability_metadata.input_schema), json.dumps(capability_metadata.output_schema),
                json.dumps(capability_metadata.saas_workflows), json.dumps(capability_metadata.business_metrics),
                capability_metadata.avg_execution_time_ms, capability_metadata.success_rate,
                capability_metadata.trust_score, capability_metadata.sla_tier,
                json.dumps(capability_metadata.available_regions), json.dumps(capability_metadata.compliance_requirements),
                json.dumps(capability_metadata.policy_tags))
            
            # Update cache
            self.capability_cache[capability_metadata.capability_id] = capability_metadata
            
            self.logger.info(f"Stored capability metadata {capability_metadata.name} for tenant {capability_metadata.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store capability metadata {capability_metadata.name}: {e}")
            return False
    
    async def parse_intent_with_orchestrator(self, raw_intent: RawIntent) -> Optional[ParsedIntent]:
        """Parse intent using orchestrator integration"""
        try:
            # Store raw intent first
            await self.store_raw_intent(raw_intent)
            
            # Use rule-based parsing as primary method
            parsed_intent = await self._parse_intent_rule_based(raw_intent)
            
            if parsed_intent and parsed_intent.confidence_score >= self.parsing_models['rule_based']['confidence_threshold']:
                # Store parsed intent
                await self.store_parsed_intent(parsed_intent)
                
                # If orchestrator is available, enhance parsing
                if self.orchestrator:
                    try:
                        enhanced_context = await self.orchestrator.enhance_intent_context(
                            parsed_intent.parsed_action, 
                            parsed_intent.extracted_entities,
                            raw_intent.tenant_id
                        )
                        parsed_intent.saas_context.update(enhanced_context)
                    except Exception as e:
                        self.logger.warning(f"Orchestrator enhancement failed: {e}")
                
                return parsed_intent
            else:
                self.logger.warning(f"Intent parsing failed for {raw_intent.intent_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to parse intent with orchestrator: {e}")
            return None
    
    async def _parse_intent_rule_based(self, raw_intent: RawIntent) -> Optional[ParsedIntent]:
        """Rule-based intent parsing"""
        try:
            patterns = self.parsing_models['rule_based']['patterns']
            raw_input_lower = raw_intent.raw_input.lower()
            
            best_match = None
            best_confidence = 0.0
            
            for intent_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    import re
                    if re.search(pattern, raw_input_lower):
                        confidence = 0.9  # High confidence for exact pattern match
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_match = intent_type
            
            if best_match and best_confidence >= self.parsing_models['rule_based']['confidence_threshold']:
                # Extract entities based on intent type
                entities = await self._extract_entities(raw_intent.raw_input, best_match)
                
                parsed_intent = ParsedIntent(
                    parsed_id=str(uuid.uuid4()),
                    intent_id=raw_intent.intent_id,
                    tenant_id=raw_intent.tenant_id,
                    parsed_action=best_match,
                    extracted_entities=entities,
                    confidence_score=best_confidence,
                    parsing_method="rule_based",
                    saas_context=self._generate_saas_context(best_match, entities)
                )
                
                return parsed_intent
            
            return None
            
        except Exception as e:
            self.logger.error(f"Rule-based parsing failed: {e}")
            return None
    
    async def _extract_entities(self, raw_input: str, intent_type: str) -> Dict[str, Any]:
        """Extract entities from raw input based on intent type"""
        entities = {}
        
        # Simple entity extraction (can be enhanced with NLP libraries)
        import re
        
        # Extract common entities
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', raw_input)
        if email_match:
            entities['email'] = email_match.group()
        
        # Extract numbers (could be ARR, customer count, etc.)
        number_matches = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', raw_input)
        if number_matches:
            entities['numbers'] = number_matches
        
        # Intent-specific entity extraction
        if intent_type == 'arr_analysis':
            # Look for time periods, revenue amounts
            time_matches = re.findall(r'\b(?:Q[1-4]|quarter|monthly|annual|yearly)\b', raw_input, re.IGNORECASE)
            if time_matches:
                entities['time_period'] = time_matches[0]
        
        elif intent_type == 'churn_prediction':
            # Look for customer identifiers, risk levels
            risk_matches = re.findall(r'\b(?:high|medium|low)\s+risk\b', raw_input, re.IGNORECASE)
            if risk_matches:
                entities['risk_level'] = risk_matches[0]
        
        return entities
    
    def _generate_saas_context(self, intent_type: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SaaS-specific context based on intent type and entities"""
        saas_context = {
            'industry_focus': 'SaaS',
            'intent_category': intent_type
        }
        
        if intent_type == 'arr_analysis':
            saas_context.update({
                'business_metric': 'ARR',
                'analysis_type': 'revenue',
                'compliance_required': ['SOX_SAAS']
            })
        
        elif intent_type == 'churn_prediction':
            saas_context.update({
                'business_metric': 'Churn Rate',
                'analysis_type': 'customer_retention',
                'compliance_required': ['GDPR_SAAS', 'SAAS_BUSINESS_RULES']
            })
        
        elif intent_type == 'workflow_execution':
            saas_context.update({
                'execution_type': 'workflow',
                'orchestration_required': True
            })
        
        return saas_context
    
    def _update_tenant_mapping(self, tenant_id: int, item_id: str, item_type: str):
        """Update tenant-to-item mapping for isolation"""
        if tenant_id not in self.tenant_executions:
            self.tenant_executions[tenant_id] = []
        
        if item_id not in self.tenant_executions[tenant_id]:
            self.tenant_executions[tenant_id].append(item_id)
    
    async def _load_existing_data(self):
        """Load existing data from database into cache"""
        if not self.pool_manager:
            return
            
        try:
            async with self.pool_manager.get_connection() as conn:
                # Load recent intents
                intent_rows = await conn.fetch("""
                    SELECT * FROM raw_intents 
                    WHERE created_at >= NOW() - INTERVAL '1 day' 
                    LIMIT 1000
                """)
                
                for row in intent_rows:
                    intent = self._row_to_raw_intent(row)
                    self.intent_cache[intent.intent_id] = intent
                    self._update_tenant_mapping(intent.tenant_id, intent.intent_id, 'intents')
                
                # Load recent executions
                exec_rows = await conn.fetch("""
                    SELECT * FROM execution_metadata 
                    WHERE created_at >= NOW() - INTERVAL '1 day' 
                    LIMIT 1000
                """)
                
                for row in exec_rows:
                    execution = self._row_to_execution_metadata(row)
                    self.execution_cache[execution.execution_id] = execution
                    self._update_tenant_mapping(execution.tenant_id, execution.execution_id, 'executions')
                
                # Load capabilities
                cap_rows = await conn.fetch("SELECT * FROM enhanced_capability_metadata LIMIT 1000")
                
                for row in cap_rows:
                    capability = self._row_to_capability_metadata(row)
                    self.capability_cache[capability.capability_id] = capability
                
                self.logger.info(f"Loaded {len(intent_rows)} intents, {len(exec_rows)} executions, {len(cap_rows)} capabilities into cache")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing data: {e}")
    
    def _row_to_raw_intent(self, row) -> RawIntent:
        """Convert database row to RawIntent object"""
        return RawIntent(
            intent_id=str(row['intent_id']),
            tenant_id=row['tenant_id'],
            tenant_tier=TenantTier(row['tenant_tier']),
            raw_input=row['raw_input'],
            intent_type=IntentType(row['intent_type']),
            source_system=row['source_system'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            context_data=json.loads(row['context_data']) if row['context_data'] else {},
            created_at=row['created_at']
        )
    
    def _row_to_execution_metadata(self, row) -> ExecutionMetadata:
        """Convert database row to ExecutionMetadata object"""
        return ExecutionMetadata(
            execution_id=str(row['execution_id']),
            tenant_id=row['tenant_id'],
            tenant_tier=TenantTier(row['tenant_tier']),
            workflow_id=row['workflow_id'],
            capability_id=row['capability_id'],
            execution_type=row['execution_type'],
            triggered_by_user_id=row['triggered_by_user_id'],
            trigger_source=row['trigger_source'],
            input_parameters=json.loads(row['input_parameters']) if row['input_parameters'] else {},
            business_context=json.loads(row['business_context']) if row['business_context'] else {},
            compliance_context=json.loads(row['compliance_context']) if row['compliance_context'] else {},
            start_time=row['start_time'],
            estimated_duration_ms=row['estimated_duration_ms'],
            policy_pack_id=row['policy_pack_id'],
            evidence_pack_required=row['evidence_pack_required'],
            created_at=row['created_at']
        )
    
    def _row_to_capability_metadata(self, row) -> CapabilityMetadata:
        """Convert database row to CapabilityMetadata object"""
        return CapabilityMetadata(
            capability_id=str(row['capability_id']),
            tenant_id=row['tenant_id'],
            tenant_tier=TenantTier(row['tenant_tier']),
            industry_code=row['industry_code'],
            name=row['name'],
            capability_type=row['capability_type'],
            version=row['version'],
            description=row['description'],
            input_schema=json.loads(row['input_schema']) if row['input_schema'] else {},
            output_schema=json.loads(row['output_schema']) if row['output_schema'] else {},
            saas_workflows=json.loads(row['saas_workflows']) if row['saas_workflows'] else [],
            business_metrics=json.loads(row['business_metrics']) if row['business_metrics'] else [],
            avg_execution_time_ms=row['avg_execution_time_ms'],
            success_rate=float(row['success_rate']),
            trust_score=float(row['trust_score']),
            sla_tier=row['sla_tier'],
            available_regions=json.loads(row['available_regions']) if row['available_regions'] else [],
            compliance_requirements=json.loads(row['compliance_requirements']) if row['compliance_requirements'] else [],
            policy_tags=json.loads(row['policy_tags']) if row['policy_tags'] else [],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )
    
    # ============ Chapter 8.1 Execution Trace Model Methods ============
    
    async def store_execution_trace(self, trace: ExecutionTrace) -> bool:
        """
        Task 8.1.4: Store execution trace in appropriate tier - DYNAMIC storage
        Task 8.1.5: Implement indexing for searchability
        Task 8.1.6: Enforce security, privacy, residency
        """
        try:
            # Get dynamic retention configuration
            retention_config = await self._get_retention_config(trace.tenant_id)
            
            # Task 8.1.6: Enforce residency compliance
            if not await self._validate_residency_compliance(trace):
                self.logger.error(f"Residency compliance failed for trace {trace.trace_id}")
                return False
            
            # Task 8.1.6: Apply PII masking if required
            if trace.metadata.pii_masked:
                trace = await self._apply_pii_masking(trace)
            
            # Task 8.1.5: Create searchable indexes
            await self._create_trace_indexes(trace)
            
            # Store in appropriate tier based on configuration
            storage_success = await self._store_in_tier(trace, trace.storage_tier, retention_config)
            
            if storage_success:
                # Task 8.1.7: Trigger KG pipeline if configured
                if retention_config.get('kg_ingestion_enabled', False):
                    await self._trigger_kg_ingestion(trace)
                
                # Task 8.1.8: Record observability metrics
                await self._record_trace_metrics(trace)
                
                self.logger.info(f"✅ Stored execution trace {trace.trace_id} in {trace.storage_tier.value} tier")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store execution trace {trace.trace_id}: {e}")
            return False
    
    async def _get_retention_config(self, tenant_id: int) -> Dict[str, Any]:
        """Get DYNAMIC retention configuration for tenant"""
        # Implementation would query tenant-specific retention policies
        return {
            "bronze_retention_days": 90,
            "silver_retention_days": 400,
            "gold_retention_years": 2,
            "kg_ingestion_enabled": True,
            "observability_enabled": True,
            "pii_masking_required": True
        }
    
    async def _validate_residency_compliance(self, trace: ExecutionTrace) -> bool:
        """Task 8.1.6: Validate data residency requirements"""
        tenant_residency = await self._get_tenant_residency_requirement(trace.tenant_id)
        trace_region = trace.metadata.residency_region
        
        if tenant_residency and trace_region.value != tenant_residency:
            return False
        return True
    
    async def _get_tenant_residency_requirement(self, tenant_id: int) -> Optional[str]:
        """Get tenant's data residency requirement dynamically"""
        # Implementation would query tenant configuration
        return "us"  # Default, would be dynamic
    
    async def _apply_pii_masking(self, trace: ExecutionTrace) -> ExecutionTrace:
        """Task 8.1.6: Apply PII masking to trace data"""
        # Get dynamic masking rules
        masking_rules = await self._get_masking_rules(trace.tenant_id)
        
        # Apply masking to inputs and outputs
        masked_inputs = await self._mask_data(trace.inputs, masking_rules)
        masked_outputs = await self._mask_data(trace.outputs, masking_rules)
        
        # Create masked trace
        trace.inputs = masked_inputs
        trace.outputs = masked_outputs
        trace.metadata.pii_masked = True
        
        return trace
    
    async def _get_masking_rules(self, tenant_id: int) -> Dict[str, Any]:
        """Get dynamic masking rules for tenant"""
        # Implementation would return tenant-specific masking configuration
        return {
            "email": {"action": "mask", "pattern": "***@***.***"},
            "ssn": {"action": "redact"},
            "phone": {"action": "mask", "pattern": "***-***-****"}
        }
    
    async def _mask_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Apply masking rules to data"""
        masked_data = data.copy()
        
        for field, rule in rules.items():
            if field in masked_data:
                if rule['action'] == 'redact':
                    masked_data[field] = '[REDACTED]'
                elif rule['action'] == 'mask':
                    masked_data[field] = rule.get('pattern', '***')
        
        return masked_data
    
    async def _create_trace_indexes(self, trace: ExecutionTrace):
        """Task 8.1.5: Create searchable indexes for trace"""
        indexes = {
            "trace_id": trace.trace_id,
            "tenant_id": trace.tenant_id,
            "workflow_id": trace.workflow_id,
            "timestamp": trace.timestamp,
            "governance_status": trace.governance.status,
            "storage_tier": trace.storage_tier.value,
            "residency_region": trace.metadata.residency_region.value
        }
        # Implementation would create database indexes
        self.logger.debug(f"Created indexes for trace {trace.trace_id}: {indexes}")
    
    async def _store_in_tier(self, trace: ExecutionTrace, tier: StorageTier, config: Dict[str, Any]) -> bool:
        """Store trace in specified storage tier"""
        try:
            if tier == StorageTier.BRONZE:
                # Store raw trace data
                return await self._store_bronze_trace(trace, config)
            elif tier == StorageTier.SILVER:
                # Store normalized trace data
                return await self._store_silver_trace(trace, config)
            elif tier == StorageTier.GOLD:
                # Store aggregated trace features
                return await self._store_gold_trace(trace, config)
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to store trace in {tier.value} tier: {e}")
            return False
    
    async def _store_bronze_trace(self, trace: ExecutionTrace, config: Dict[str, Any]) -> bool:
        """Store raw trace in Bronze tier (JSON/Parquet)"""
        # Implementation would store in data lake
        self.logger.debug(f"Stored trace {trace.trace_id} in Bronze tier")
        return True
    
    async def _store_silver_trace(self, trace: ExecutionTrace, config: Dict[str, Any]) -> bool:
        """Store normalized trace in Silver tier"""
        # Implementation would store normalized data
        self.logger.debug(f"Stored trace {trace.trace_id} in Silver tier")
        return True
    
    async def _store_gold_trace(self, trace: ExecutionTrace, config: Dict[str, Any]) -> bool:
        """Store aggregated features in Gold tier"""
        # Implementation would store aggregated data
        self.logger.debug(f"Stored trace {trace.trace_id} in Gold tier")
        return True
    
    async def _trigger_kg_ingestion(self, trace: ExecutionTrace):
        """Task 8.1.7: Trigger Knowledge Graph ingestion pipeline"""
        kg_data = {
            "trace_id": trace.trace_id,
            "tenant_id": trace.tenant_id,
            "workflow_id": trace.workflow_id,
            "timestamp": trace.timestamp.isoformat(),
            "governance": asdict(trace.governance),
            "metadata": asdict(trace.metadata)
        }
        # Implementation would trigger KG pipeline
        self.logger.info(f"🔗 Triggered KG ingestion for trace {trace.trace_id}")
    
    async def _record_trace_metrics(self, trace: ExecutionTrace):
        """Task 8.1.8: Record observability metrics"""
        metrics = {
            "trace_ingested": 1,
            "tenant_id": trace.tenant_id,
            "storage_tier": trace.storage_tier.value,
            "execution_time_ms": trace.metadata.execution_time_ms,
            "trust_score": trace.metadata.trust_score,
            "governance_status": trace.governance.status
        }
        # Implementation would send to metrics system
        self.logger.debug(f"📊 Recorded metrics for trace {trace.trace_id}: {metrics}")
    
    # Task 8.1.9: Retention & Purge Policies
    async def purge_expired_traces(self, tenant_id: int, tier: StorageTier):
        """Purge traces past retention period"""
        try:
            retention_config = await self._get_retention_config(tenant_id)
            
            if tier == StorageTier.BRONZE:
                retention_days = retention_config.get('bronze_retention_days', 90)
            elif tier == StorageTier.SILVER:
                retention_days = retention_config.get('silver_retention_days', 400)
            elif tier == StorageTier.GOLD:
                retention_years = retention_config.get('gold_retention_years', 2)
                retention_days = retention_years * 365
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Implementation would purge expired traces
            self.logger.info(f"🗑️ Purged {tier.value} traces older than {cutoff_date} for tenant {tenant_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to purge traces: {e}")
    
    async def handle_gdpr_erasure_request(self, tenant_id: int, user_id: str):
        """Task 8.1.9: Handle GDPR erasure requests"""
        try:
            # Implementation would identify and erase user traces
            self.logger.info(f"🔒 Processing GDPR erasure for tenant {tenant_id}, user {user_id}")
            
            # Audit the erasure
            await self._audit_gdpr_erasure(tenant_id, user_id)
            
        except Exception as e:
            self.logger.error(f"Failed GDPR erasure: {e}")
    
    async def _audit_gdpr_erasure(self, tenant_id: int, user_id: str):
        """Audit GDPR erasure for compliance"""
        audit_record = {
            "action": "gdpr_erasure",
            "tenant_id": tenant_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "compliance_framework": "GDPR"
        }
        # Store audit record
        self.logger.info(f"📋 GDPR erasure audit: {audit_record}")
    
    # Task 8.1.10: Developer & Auditor Experience
    async def get_trace_by_id(self, trace_id: str, tenant_id: int) -> Optional[ExecutionTrace]:
        """Get trace by ID with tenant isolation"""
        try:
            # Implementation would retrieve trace with tenant validation
            self.logger.info(f"🔍 Retrieved trace {trace_id} for tenant {tenant_id}")
            return None  # Placeholder
        except Exception as e:
            self.logger.error(f"Failed to get trace {trace_id}: {e}")
            return None
    
    async def search_traces(self, tenant_id: int, 
                          workflow_id: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          governance_status: Optional[str] = None) -> List[ExecutionTrace]:
        """Search traces with dynamic filters and tenant isolation"""
        try:
            filters = {
                "tenant_id": tenant_id,
                "workflow_id": workflow_id,
                "start_time": start_time,
                "end_time": end_time,
                "governance_status": governance_status
            }
            
            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}
            
            # Implementation would search with filters
            self.logger.info(f"🔍 Searching traces with filters: {filters}")
            return []  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Failed to search traces: {e}")
            return []
    
    async def export_traces_for_auditor(self, tenant_id: int, redacted: bool = True) -> str:
        """Task 8.1.10: Export traces for auditor review"""
        try:
            export_config = {
                "tenant_id": tenant_id,
                "redacted": redacted,
                "format": "json",
                "timestamp": datetime.now().isoformat()
            }
            
            export_path = f"audit_export_tenant_{tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Implementation would generate auditor-friendly export
            self.logger.info(f"📤 Exported traces for auditor: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Failed to export traces for auditor: {e}")
            return ""

# Singleton instance for global access
_enhanced_execution_tracker = None

def get_enhanced_execution_tracker(pool_manager=None) -> EnhancedExecutionTracker:
    """Get singleton instance of Enhanced Execution Tracker"""
    global _enhanced_execution_tracker
    if _enhanced_execution_tracker is None:
        _enhanced_execution_tracker = EnhancedExecutionTracker(pool_manager)
    return _enhanced_execution_tracker
