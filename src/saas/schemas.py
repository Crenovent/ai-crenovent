"""
SaaS Industry Schema Extensions
Extended schemas for trace, evidence, and audit for SaaS RBA workflows
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import uuid4
import json

class SaaSWorkflowType(Enum):
    PIPELINE_HYGIENE = "pipeline_hygiene"
    FORECAST_ACCURACY = "forecast_accuracy"
    LEAD_SCORING = "lead_scoring"
    CHURN_PREVENTION = "churn_prevention"
    REVENUE_RECOGNITION = "revenue_recognition"

class SaaSEventType(Enum):
    OPPORTUNITY_CREATED = "opportunity_created"
    OPPORTUNITY_UPDATED = "opportunity_updated"
    STAGE_CHANGED = "stage_changed"
    FORECAST_SUBMITTED = "forecast_submitted"
    LEAD_SCORED = "lead_scored"
    HYGIENE_CHECK_COMPLETED = "hygiene_check_completed"
    NOTIFICATION_SENT = "notification_sent"
    RULE_VIOLATION = "rule_violation"
    OVERRIDE_APPLIED = "override_applied"

class ComplianceFramework(Enum):
    SOX = "sox"
    GDPR = "gdpr"
    DSAR = "dsar"
    CCPA = "ccpa"

# Base Schema Models
class SaaSBaseSchema(BaseModel):
    """Base schema for all SaaS workflow schemas"""
    schema_version: str = "1.0.0"
    tenant_id: str
    workflow_type: SaaSWorkflowType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True

# Trace Schema Extensions
class SaaSOpportunityTrace(BaseModel):
    """Trace schema for opportunity-related events"""
    opportunity_id: str
    opportunity_name: str
    account_id: str
    account_name: str
    owner_id: str
    owner_name: str
    owner_email: str
    current_stage: str
    previous_stage: Optional[str] = None
    stage_entry_date: datetime
    stage_duration_days: int
    amount: float
    currency: str = "USD"
    close_date: datetime
    probability: float
    created_date: datetime
    last_activity_date: Optional[datetime] = None
    activity_count: int = 0
    
    # SaaS-specific fields
    arr_value: Optional[float] = None  # Annual Recurring Revenue
    contract_length_months: Optional[int] = None
    product_line: Optional[str] = None
    deal_source: Optional[str] = None
    competitor: Optional[str] = None

class SaaSPipelineMetricsTrace(BaseModel):
    """Trace schema for pipeline metrics"""
    calculation_date: datetime
    total_pipeline_value: float
    quarterly_quota: float
    pipeline_coverage_ratio: float
    stalled_count: int
    stalled_value: float
    hygiene_score: float
    average_deal_size: float
    sales_velocity: float  # Days
    conversion_rate: float
    
    # Stage-specific metrics
    stage_metrics: Dict[str, Dict[str, Any]] = {}
    
    # Team metrics
    team_metrics: Dict[str, Dict[str, Any]] = {}

class SaaSForecastTrace(BaseModel):
    """Trace schema for forecast-related events"""
    forecast_id: str
    forecast_period: str  # "2024-Q1"
    forecast_type: str  # "commit", "best_case", "pipeline"
    submitted_by: str
    submitted_at: datetime
    forecasted_amount: float
    actual_amount: Optional[float] = None
    variance: Optional[float] = None
    accuracy_score: Optional[float] = None
    
    # Breakdown by categories
    forecast_breakdown: Dict[str, float] = {}
    
    # Historical comparison
    previous_forecast: Optional[float] = None
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"

class SaaSLeadScoringTrace(BaseModel):
    """Trace schema for lead scoring events"""
    lead_id: str
    lead_email: str
    company_name: str
    job_title: str
    lead_source: str
    score: int
    grade: str  # A, B, C, D
    scoring_factors: Dict[str, Any]
    demographic_score: float
    behavioral_score: float
    engagement_score: float
    
    # Scoring history
    previous_score: Optional[int] = None
    score_change: Optional[int] = None
    scoring_reason: str

# Extended Trace Schema
class SaaSWorkflowTrace(SaaSBaseSchema):
    """Extended trace schema for SaaS workflows"""
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_id: str
    workflow_id: str
    workflow_name: str
    step_id: str
    step_name: str
    step_type: str
    event_type: SaaSEventType
    
    # Execution context
    execution_context: Dict[str, Any] = {}
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    
    # Performance metrics
    execution_time_ms: int
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Business context
    business_impact: Optional[str] = None
    risk_level: str = "low"  # low, medium, high, critical
    
    # SaaS-specific traces
    opportunity_trace: Optional[SaaSOpportunityTrace] = None
    pipeline_metrics_trace: Optional[SaaSPipelineMetricsTrace] = None
    forecast_trace: Optional[SaaSForecastTrace] = None
    lead_scoring_trace: Optional[SaaSLeadScoringTrace] = None
    
    # Compliance and governance
    compliance_frameworks: List[ComplianceFramework] = []
    audit_required: bool = False
    retention_days: int = 2555  # 7 years for SOX
    
    # Error handling
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

# Evidence Schema Extensions
class SaaSComplianceEvidence(BaseModel):
    """Evidence schema for SaaS compliance requirements"""
    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_type: str
    compliance_framework: ComplianceFramework
    regulation_reference: str
    evidence_data: Dict[str, Any]
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    hash_value: str  # For immutability verification
    digital_signature: Optional[str] = None

class SaaSSOXEvidence(SaaSComplianceEvidence):
    """SOX-specific evidence for financial controls"""
    control_id: str
    control_description: str
    test_procedure: str
    test_result: str  # "passed", "failed", "warning"
    auditor_notes: Optional[str] = None
    remediation_required: bool = False
    
    # Financial data
    financial_impact: Optional[float] = None
    revenue_impact: Optional[float] = None
    
    def __init__(self, **data):
        data['evidence_type'] = 'sox_control_evidence'
        data['compliance_framework'] = ComplianceFramework.SOX
        super().__init__(**data)

class SaaSGDPREvidence(SaaSComplianceEvidence):
    """GDPR-specific evidence for data privacy"""
    data_subject_id: str
    processing_purpose: str
    legal_basis: str
    consent_status: str
    data_categories: List[str]
    retention_period: int  # days
    
    def __init__(self, **data):
        data['evidence_type'] = 'gdpr_processing_evidence'
        data['compliance_framework'] = ComplianceFramework.GDPR
        super().__init__(**data)

class SaaSEvidencePack(SaaSBaseSchema):
    """Evidence pack for SaaS workflows"""
    evidence_pack_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_execution_id: str
    pack_name: str
    pack_description: str
    
    # Evidence collection
    sox_evidence: List[SaaSSOXEvidence] = []
    gdpr_evidence: List[SaaSGDPREvidence] = []
    general_evidence: List[SaaSComplianceEvidence] = []
    
    # Metadata
    collection_start: datetime = Field(default_factory=datetime.utcnow)
    collection_end: Optional[datetime] = None
    evidence_count: int = 0
    total_size_bytes: int = 0
    
    # Validation
    is_complete: bool = False
    validation_errors: List[str] = []
    
    # Export information
    export_formats: List[str] = ["json", "pdf", "csv"]
    export_location: Optional[str] = None

# Audit Schema Extensions
class SaaSRuleViolation(BaseModel):
    """Schema for SaaS rule violations"""
    violation_id: str = Field(default_factory=lambda: str(uuid4()))
    rule_id: str
    rule_name: str
    rule_description: str
    violation_type: str  # "threshold", "compliance", "data_quality"
    severity: str  # "low", "medium", "high", "critical"
    
    # Violation details
    expected_value: Any
    actual_value: Any
    threshold_exceeded: Optional[float] = None
    
    # Business impact
    business_impact_description: str
    financial_impact: Optional[float] = None
    risk_category: str
    
    # Resolution
    auto_resolved: bool = False
    resolution_required: bool = True
    resolution_deadline: Optional[datetime] = None

class SaaSOverrideRecord(BaseModel):
    """Schema for SaaS override records"""
    override_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_execution_id: str
    rule_id: str
    rule_name: str
    
    # Override details
    override_reason: str
    override_justification: str
    override_type: str  # "manual", "automatic", "emergency"
    
    # Approval workflow
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approval_status: str = "pending"  # "pending", "approved", "rejected"
    
    # Original vs Override values
    original_value: Any
    override_value: Any
    
    # Risk assessment
    risk_assessment: str
    business_justification: str
    
    # Compliance impact
    compliance_impact: Optional[str] = None
    audit_notification_required: bool = False

class SaaSAuditTrail(SaaSBaseSchema):
    """Comprehensive audit trail for SaaS workflows"""
    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_execution_id: str
    audit_type: str  # "execution", "compliance", "security", "performance"
    
    # Execution details
    execution_summary: Dict[str, Any]
    steps_executed: List[Dict[str, Any]]
    
    # Rule violations and overrides
    rule_violations: List[SaaSRuleViolation] = []
    override_records: List[SaaSOverrideRecord] = []
    
    # Compliance status
    compliance_status: Dict[ComplianceFramework, str] = {}
    compliance_score: float = 1.0
    
    # Performance metrics
    total_execution_time_ms: int
    resource_usage: Dict[str, Any] = {}
    
    # Data lineage
    data_sources: List[str] = []
    data_transformations: List[Dict[str, Any]] = []
    data_outputs: List[str] = []
    
    # Security events
    security_events: List[Dict[str, Any]] = []
    access_log: List[Dict[str, Any]] = []
    
    # Immutability and integrity
    hash_chain: List[str] = []  # For blockchain-like verification
    digital_signature: Optional[str] = None
    tamper_evident: bool = True

# Schema Registry
class SaaSSchemaRegistry:
    """Registry for managing SaaS schema versions and validation"""
    
    def __init__(self):
        self.schemas = {
            "trace": {
                "1.0.0": SaaSWorkflowTrace,
                "current": "1.0.0"
            },
            "evidence": {
                "1.0.0": SaaSEvidencePack,
                "current": "1.0.0"
            },
            "audit": {
                "1.0.0": SaaSAuditTrail,
                "current": "1.0.0"
            }
        }
    
    def get_schema(self, schema_type: str, version: str = None) -> BaseModel:
        """Get schema class by type and version"""
        if schema_type not in self.schemas:
            raise ValueError(f"Unknown schema type: {schema_type}")
        
        schema_versions = self.schemas[schema_type]
        
        if version is None:
            version = schema_versions["current"]
        
        if version not in schema_versions:
            raise ValueError(f"Unknown version {version} for schema {schema_type}")
        
        return schema_versions[version]
    
    def validate_data(self, schema_type: str, data: Dict[str, Any], version: str = None) -> Dict[str, Any]:
        """Validate data against schema"""
        schema_class = self.get_schema(schema_type, version)
        
        try:
            validated_instance = schema_class(**data)
            return {
                "is_valid": True,
                "validated_data": validated_instance.dict(),
                "errors": []
            }
        except Exception as e:
            return {
                "is_valid": False,
                "validated_data": None,
                "errors": [str(e)]
            }
    
    def get_schema_json(self, schema_type: str, version: str = None) -> Dict[str, Any]:
        """Get JSON schema definition"""
        schema_class = self.get_schema(schema_type, version)
        return schema_class.schema()
    
    def list_schemas(self) -> Dict[str, List[str]]:
        """List all available schemas and versions"""
        result = {}
        for schema_type, versions in self.schemas.items():
            result[schema_type] = [v for v in versions.keys() if v != "current"]
        return result

# Schema Factory
class SaaSSchemaFactory:
    """Factory for creating SaaS schema instances"""
    
    def __init__(self):
        self.registry = SaaSSchemaRegistry()
    
    def create_trace(
        self,
        tenant_id: str,
        workflow_type: SaaSWorkflowType,
        execution_id: str,
        workflow_id: str,
        step_id: str,
        event_type: SaaSEventType,
        **kwargs
    ) -> SaaSWorkflowTrace:
        """Create a new trace instance"""
        return SaaSWorkflowTrace(
            tenant_id=tenant_id,
            workflow_type=workflow_type,
            execution_id=execution_id,
            workflow_id=workflow_id,
            step_id=step_id,
            event_type=event_type,
            **kwargs
        )
    
    def create_evidence_pack(
        self,
        tenant_id: str,
        workflow_type: SaaSWorkflowType,
        workflow_execution_id: str,
        pack_name: str,
        **kwargs
    ) -> SaaSEvidencePack:
        """Create a new evidence pack"""
        return SaaSEvidencePack(
            tenant_id=tenant_id,
            workflow_type=workflow_type,
            workflow_execution_id=workflow_execution_id,
            pack_name=pack_name,
            **kwargs
        )
    
    def create_audit_trail(
        self,
        tenant_id: str,
        workflow_type: SaaSWorkflowType,
        workflow_execution_id: str,
        **kwargs
    ) -> SaaSAuditTrail:
        """Create a new audit trail"""
        return SaaSAuditTrail(
            tenant_id=tenant_id,
            workflow_type=workflow_type,
            workflow_execution_id=workflow_execution_id,
            **kwargs
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize schema factory
    factory = SaaSSchemaFactory()
    
    # Create a trace instance
    trace = factory.create_trace(
        tenant_id="tenant_123",
        workflow_type=SaaSWorkflowType.PIPELINE_HYGIENE,
        execution_id="exec_456",
        workflow_id="saas_pipeline_hygiene_v1.0",
        step_id="calculate_pipeline_metrics",
        event_type=SaaSEventType.HYGIENE_CHECK_COMPLETED,
        workflow_name="Pipeline Hygiene Monitor",
        step_name="Calculate Pipeline Metrics",
        step_type="calculation",
        execution_time_ms=1500
    )
    
    print(f"Created trace: {trace.trace_id}")
    print(f"Trace schema version: {trace.schema_version}")
    
    # Test schema validation
    registry = SaaSSchemaRegistry()
    validation_result = registry.validate_data("trace", trace.dict())
    print(f"Validation result: {validation_result['is_valid']}")
    
    # List available schemas
    available_schemas = registry.list_schemas()
    print(f"Available schemas: {available_schemas}")
