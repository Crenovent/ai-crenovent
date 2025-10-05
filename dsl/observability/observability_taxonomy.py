"""
Observability Taxonomy
=====================

Task 7.4.1: Define observability taxonomy (metrics, logs, traces, events)
Standardized taxonomy for observability across the RBA system
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TelemetryType(Enum):
    """Types of telemetry data"""
    METRIC = "metric"
    LOG = "log"
    TRACE = "trace"
    EVENT = "event"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"

class TraceSpanType(Enum):
    """Types of trace spans"""
    WORKFLOW = "workflow"
    STEP = "step"
    OPERATOR = "operator"
    CONNECTOR = "connector"
    GOVERNANCE = "governance"
    POLICY = "policy"
    AGENT = "agent"
    DATABASE = "database"
    EXTERNAL_API = "external_api"

class EventType(Enum):
    """Types of events"""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_FAILED = "workflow_failed"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    STEP_FAILED = "step_failed"
    POLICY_VIOLATION = "policy_violation"
    GOVERNANCE_CHECK = "governance_check"
    OVERRIDE_APPLIED = "override_applied"
    TRUST_SCORE_UPDATE = "trust_score_update"
    SLA_BREACH = "sla_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    COMPLIANCE_AUDIT = "compliance_audit"
    EVIDENCE_GENERATED = "evidence_generated"

class SeverityLevel(Enum):
    """Severity levels for events and alerts"""
    SEV_0 = "sev_0"  # Critical - immediate action required
    SEV_1 = "sev_1"  # High - action required within 1 hour
    SEV_2 = "sev_2"  # Medium - action required within 24 hours
    SEV_3 = "sev_3"  # Low - action required within 1 week
    SEV_4 = "sev_4"  # Info - no action required

class PersonaType(Enum):
    """Persona types for scoped observability"""
    EXECUTIVE = "executive"      # CRO, CFO - business impact focus
    OPERATIONS = "operations"    # DevOps, SRE - system health focus
    COMPLIANCE = "compliance"    # Compliance officers - regulatory focus
    REVENUE_OPS = "revenue_ops"  # RevOps - business process focus
    REGULATOR = "regulator"      # External regulators - audit focus

@dataclass
class ObservabilityContext:
    """Context information for all observability data"""
    tenant_id: int
    user_id: Optional[int] = None
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    step_id: Optional[str] = None
    region_id: str = "GLOBAL"
    sla_tier: str = "T2"
    industry_code: str = "SAAS"
    policy_pack_id: Optional[str] = None
    consent_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    labels: List[str] = field(default_factory=list)
    help_text: str = ""
    persona_visibility: List[PersonaType] = field(default_factory=list)
    sla_relevant: bool = False
    compliance_relevant: bool = False
    retention_days: int = 90

@dataclass
class LogDefinition:
    """Definition of a log entry"""
    source: str
    level: LogLevel
    message_template: str
    structured_fields: List[str] = field(default_factory=list)
    persona_visibility: List[PersonaType] = field(default_factory=list)
    pii_fields: List[str] = field(default_factory=list)
    compliance_relevant: bool = False
    retention_days: int = 30

@dataclass
class TraceDefinition:
    """Definition of a trace span"""
    operation_name: str
    span_type: TraceSpanType
    description: str
    expected_duration_ms: int
    tags: List[str] = field(default_factory=list)
    persona_visibility: List[PersonaType] = field(default_factory=list)
    sla_relevant: bool = False
    compliance_relevant: bool = False

@dataclass
class EventDefinition:
    """Definition of an event"""
    event_type: EventType
    description: str
    severity: SeverityLevel
    payload_schema: Dict[str, Any] = field(default_factory=dict)
    persona_visibility: List[PersonaType] = field(default_factory=list)
    alert_rules: List[str] = field(default_factory=list)
    compliance_relevant: bool = False
    evidence_generation: bool = False

class ObservabilityTaxonomy:
    """
    Centralized observability taxonomy for RBA system
    
    Provides standardized definitions for metrics, logs, traces, and events
    across all components of the system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize taxonomy definitions
        self.metrics = self._define_metrics()
        self.logs = self._define_logs()
        self.traces = self._define_traces()
        self.events = self._define_events()
        
        # Persona-specific views
        self.persona_metrics = self._build_persona_metrics()
        self.persona_logs = self._build_persona_logs()
        self.persona_traces = self._build_persona_traces()
        self.persona_events = self._build_persona_events()
    
    def _define_metrics(self) -> Dict[str, MetricDefinition]:
        """Define standard metrics for the RBA system"""
        
        return {
            # Workflow Metrics
            "rba_workflow_executions_total": MetricDefinition(
                name="rba_workflow_executions_total",
                metric_type=MetricType.COUNTER,
                description="Total number of workflow executions",
                unit="count",
                labels=["tenant_id", "workflow_id", "status", "sla_tier"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                sla_relevant=True,
                compliance_relevant=True,
                retention_days=365
            ),
            
            "rba_workflow_duration_seconds": MetricDefinition(
                name="rba_workflow_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Workflow execution duration",
                unit="seconds",
                labels=["tenant_id", "workflow_id", "sla_tier"],
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                sla_relevant=True,
                retention_days=90
            ),
            
            "rba_workflow_success_rate": MetricDefinition(
                name="rba_workflow_success_rate",
                metric_type=MetricType.GAUGE,
                description="Workflow success rate percentage",
                unit="percentage",
                labels=["tenant_id", "workflow_id", "sla_tier"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.OPERATIONS],
                sla_relevant=True,
                retention_days=365
            ),
            
            # Step Metrics
            "rba_step_executions_total": MetricDefinition(
                name="rba_step_executions_total",
                metric_type=MetricType.COUNTER,
                description="Total number of step executions",
                unit="count",
                labels=["tenant_id", "workflow_id", "step_id", "step_type", "status"],
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                sla_relevant=True,
                retention_days=90
            ),
            
            "rba_step_duration_seconds": MetricDefinition(
                name="rba_step_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Step execution duration",
                unit="seconds",
                labels=["tenant_id", "workflow_id", "step_id", "step_type"],
                persona_visibility=[PersonaType.OPERATIONS],
                sla_relevant=True,
                retention_days=90
            ),
            
            # Governance Metrics
            "rba_policy_violations_total": MetricDefinition(
                name="rba_policy_violations_total",
                metric_type=MetricType.COUNTER,
                description="Total number of policy violations",
                unit="count",
                labels=["tenant_id", "policy_pack_id", "violation_type", "severity"],
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.EXECUTIVE],
                compliance_relevant=True,
                retention_days=2555  # 7 years
            ),
            
            "rba_trust_score": MetricDefinition(
                name="rba_trust_score",
                metric_type=MetricType.GAUGE,
                description="Current trust score for workflows/agents",
                unit="score",
                labels=["tenant_id", "entity_id", "entity_type"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.COMPLIANCE],
                sla_relevant=True,
                compliance_relevant=True,
                retention_days=365
            ),
            
            "rba_override_count": MetricDefinition(
                name="rba_override_count",
                metric_type=MetricType.COUNTER,
                description="Number of manual overrides applied",
                unit="count",
                labels=["tenant_id", "workflow_id", "override_type", "approver_role"],
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.EXECUTIVE],
                compliance_relevant=True,
                retention_days=2555  # 7 years
            ),
            
            # SLA Metrics
            "rba_sla_compliance_rate": MetricDefinition(
                name="rba_sla_compliance_rate",
                metric_type=MetricType.GAUGE,
                description="SLA compliance rate percentage",
                unit="percentage",
                labels=["tenant_id", "sla_tier", "metric_type"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.OPERATIONS],
                sla_relevant=True,
                retention_days=365
            ),
            
            "rba_sla_breaches_total": MetricDefinition(
                name="rba_sla_breaches_total",
                metric_type=MetricType.COUNTER,
                description="Total number of SLA breaches",
                unit="count",
                labels=["tenant_id", "sla_tier", "breach_type"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.OPERATIONS],
                sla_relevant=True,
                retention_days=365
            ),
            
            # Resource Metrics
            "rba_resource_usage_cpu": MetricDefinition(
                name="rba_resource_usage_cpu",
                metric_type=MetricType.GAUGE,
                description="CPU usage percentage",
                unit="percentage",
                labels=["tenant_id", "component", "instance"],
                persona_visibility=[PersonaType.OPERATIONS],
                retention_days=30
            ),
            
            "rba_resource_usage_memory": MetricDefinition(
                name="rba_resource_usage_memory",
                metric_type=MetricType.GAUGE,
                description="Memory usage in bytes",
                unit="bytes",
                labels=["tenant_id", "component", "instance"],
                persona_visibility=[PersonaType.OPERATIONS],
                retention_days=30
            ),
            
            # Business Metrics
            "rba_business_impact_value": MetricDefinition(
                name="rba_business_impact_value",
                metric_type=MetricType.GAUGE,
                description="Business impact value in currency",
                unit="currency",
                labels=["tenant_id", "workflow_id", "impact_type"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.REVENUE_OPS],
                retention_days=365
            )
        }
    
    def _define_logs(self) -> Dict[str, LogDefinition]:
        """Define standard log entries for the RBA system"""
        
        return {
            "workflow_execution": LogDefinition(
                source="rba_orchestrator",
                level=LogLevel.INFO,
                message_template="Workflow {workflow_id} execution {status} for tenant {tenant_id}",
                structured_fields=["workflow_id", "execution_id", "tenant_id", "status", "duration_ms"],
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                compliance_relevant=True,
                retention_days=90
            ),
            
            "policy_violation": LogDefinition(
                source="policy_engine",
                level=LogLevel.WARN,
                message_template="Policy violation detected: {violation_type} in {workflow_id}",
                structured_fields=["policy_pack_id", "violation_type", "workflow_id", "tenant_id", "severity"],
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.EXECUTIVE],
                compliance_relevant=True,
                retention_days=2555  # 7 years
            ),
            
            "governance_check": LogDefinition(
                source="governance_engine",
                level=LogLevel.INFO,
                message_template="Governance check {status} for {entity_type} {entity_id}",
                structured_fields=["entity_type", "entity_id", "check_type", "status", "tenant_id"],
                persona_visibility=[PersonaType.COMPLIANCE],
                compliance_relevant=True,
                retention_days=365
            ),
            
            "override_applied": LogDefinition(
                source="override_ledger",
                level=LogLevel.WARN,
                message_template="Manual override applied by {user_id} for {workflow_id}",
                structured_fields=["user_id", "workflow_id", "override_reason", "approver_id", "tenant_id"],
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.EXECUTIVE],
                compliance_relevant=True,
                retention_days=2555  # 7 years
            ),
            
            "sla_breach": LogDefinition(
                source="sla_monitor",
                level=LogLevel.ERROR,
                message_template="SLA breach detected: {breach_type} for tenant {tenant_id}",
                structured_fields=["breach_type", "sla_tier", "threshold", "actual_value", "tenant_id"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.OPERATIONS],
                sla_relevant=True,
                retention_days=365
            ),
            
            "trust_score_update": LogDefinition(
                source="trust_engine",
                level=LogLevel.INFO,
                message_template="Trust score updated for {entity_type} {entity_id}: {old_score} -> {new_score}",
                structured_fields=["entity_type", "entity_id", "old_score", "new_score", "reason", "tenant_id"],
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.COMPLIANCE],
                compliance_relevant=True,
                retention_days=365
            ),
            
            "anomaly_detected": LogDefinition(
                source="anomaly_detector",
                level=LogLevel.WARN,
                message_template="Anomaly detected in {metric_name}: {anomaly_type}",
                structured_fields=["metric_name", "anomaly_type", "severity", "threshold", "actual_value", "tenant_id"],
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.COMPLIANCE],
                retention_days=90
            ),
            
            "evidence_generated": LogDefinition(
                source="evidence_engine",
                level=LogLevel.INFO,
                message_template="Evidence pack generated: {evidence_pack_id} for {workflow_id}",
                structured_fields=["evidence_pack_id", "workflow_id", "evidence_type", "compliance_framework", "tenant_id"],
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.REGULATOR],
                compliance_relevant=True,
                retention_days=2555  # 7 years
            )
        }
    
    def _define_traces(self) -> Dict[str, TraceDefinition]:
        """Define standard trace spans for the RBA system"""
        
        return {
            "workflow_execution": TraceDefinition(
                operation_name="execute_workflow",
                span_type=TraceSpanType.WORKFLOW,
                description="Complete workflow execution span",
                expected_duration_ms=30000,
                tags=["workflow_id", "tenant_id", "sla_tier"],
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                sla_relevant=True,
                compliance_relevant=True
            ),
            
            "step_execution": TraceDefinition(
                operation_name="execute_step",
                span_type=TraceSpanType.STEP,
                description="Individual step execution span",
                expected_duration_ms=5000,
                tags=["step_id", "step_type", "workflow_id", "tenant_id"],
                persona_visibility=[PersonaType.OPERATIONS],
                sla_relevant=True
            ),
            
            "policy_evaluation": TraceDefinition(
                operation_name="evaluate_policy",
                span_type=TraceSpanType.POLICY,
                description="Policy evaluation span",
                expected_duration_ms=1000,
                tags=["policy_pack_id", "policy_type", "tenant_id"],
                persona_visibility=[PersonaType.COMPLIANCE],
                compliance_relevant=True
            ),
            
            "governance_check": TraceDefinition(
                operation_name="governance_check",
                span_type=TraceSpanType.GOVERNANCE,
                description="Governance validation span",
                expected_duration_ms=2000,
                tags=["check_type", "entity_id", "tenant_id"],
                persona_visibility=[PersonaType.COMPLIANCE],
                compliance_relevant=True
            ),
            
            "agent_call": TraceDefinition(
                operation_name="agent_call",
                span_type=TraceSpanType.AGENT,
                description="AI agent invocation span",
                expected_duration_ms=15000,
                tags=["agent_type", "agent_id", "workflow_id", "tenant_id"],
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                sla_relevant=True
            ),
            
            "database_query": TraceDefinition(
                operation_name="database_query",
                span_type=TraceSpanType.DATABASE,
                description="Database query execution span",
                expected_duration_ms=3000,
                tags=["query_type", "table_name", "tenant_id"],
                persona_visibility=[PersonaType.OPERATIONS]
            ),
            
            "external_api_call": TraceDefinition(
                operation_name="external_api_call",
                span_type=TraceSpanType.EXTERNAL_API,
                description="External API call span",
                expected_duration_ms=10000,
                tags=["api_endpoint", "api_provider", "tenant_id"],
                persona_visibility=[PersonaType.OPERATIONS],
                sla_relevant=True
            )
        }
    
    def _define_events(self) -> Dict[str, EventDefinition]:
        """Define standard events for the RBA system"""
        
        return {
            "workflow_start": EventDefinition(
                event_type=EventType.WORKFLOW_START,
                description="Workflow execution started",
                severity=SeverityLevel.SEV_4,
                payload_schema={
                    "workflow_id": "string",
                    "execution_id": "string",
                    "tenant_id": "integer",
                    "user_id": "integer",
                    "trigger_type": "string"
                },
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                compliance_relevant=True
            ),
            
            "workflow_complete": EventDefinition(
                event_type=EventType.WORKFLOW_COMPLETE,
                description="Workflow execution completed successfully",
                severity=SeverityLevel.SEV_4,
                payload_schema={
                    "workflow_id": "string",
                    "execution_id": "string",
                    "tenant_id": "integer",
                    "duration_ms": "integer",
                    "steps_executed": "integer"
                },
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                compliance_relevant=True
            ),
            
            "workflow_failed": EventDefinition(
                event_type=EventType.WORKFLOW_FAILED,
                description="Workflow execution failed",
                severity=SeverityLevel.SEV_2,
                payload_schema={
                    "workflow_id": "string",
                    "execution_id": "string",
                    "tenant_id": "integer",
                    "error_message": "string",
                    "failed_step_id": "string"
                },
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.REVENUE_OPS],
                alert_rules=["workflow_failure_alert"],
                compliance_relevant=True
            ),
            
            "policy_violation": EventDefinition(
                event_type=EventType.POLICY_VIOLATION,
                description="Policy violation detected",
                severity=SeverityLevel.SEV_1,
                payload_schema={
                    "policy_pack_id": "string",
                    "violation_type": "string",
                    "workflow_id": "string",
                    "tenant_id": "integer",
                    "severity": "string"
                },
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.EXECUTIVE],
                alert_rules=["policy_violation_alert"],
                compliance_relevant=True,
                evidence_generation=True
            ),
            
            "sla_breach": EventDefinition(
                event_type=EventType.SLA_BREACH,
                description="SLA threshold breached",
                severity=SeverityLevel.SEV_1,
                payload_schema={
                    "sla_tier": "string",
                    "breach_type": "string",
                    "threshold": "number",
                    "actual_value": "number",
                    "tenant_id": "integer"
                },
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.OPERATIONS],
                alert_rules=["sla_breach_alert"],
                compliance_relevant=True
            ),
            
            "anomaly_detected": EventDefinition(
                event_type=EventType.ANOMALY_DETECTED,
                description="Anomaly detected in system behavior",
                severity=SeverityLevel.SEV_2,
                payload_schema={
                    "metric_name": "string",
                    "anomaly_type": "string",
                    "severity": "string",
                    "tenant_id": "integer",
                    "detection_algorithm": "string"
                },
                persona_visibility=[PersonaType.OPERATIONS, PersonaType.COMPLIANCE],
                alert_rules=["anomaly_detection_alert"]
            ),
            
            "trust_score_update": EventDefinition(
                event_type=EventType.TRUST_SCORE_UPDATE,
                description="Trust score updated for entity",
                severity=SeverityLevel.SEV_4,
                payload_schema={
                    "entity_type": "string",
                    "entity_id": "string",
                    "old_score": "number",
                    "new_score": "number",
                    "tenant_id": "integer"
                },
                persona_visibility=[PersonaType.EXECUTIVE, PersonaType.COMPLIANCE],
                compliance_relevant=True
            ),
            
            "override_applied": EventDefinition(
                event_type=EventType.OVERRIDE_APPLIED,
                description="Manual override applied to workflow",
                severity=SeverityLevel.SEV_2,
                payload_schema={
                    "workflow_id": "string",
                    "override_type": "string",
                    "user_id": "integer",
                    "approver_id": "integer",
                    "tenant_id": "integer",
                    "reason": "string"
                },
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.EXECUTIVE],
                alert_rules=["override_applied_alert"],
                compliance_relevant=True,
                evidence_generation=True
            ),
            
            "evidence_generated": EventDefinition(
                event_type=EventType.EVIDENCE_GENERATED,
                description="Evidence pack generated for compliance",
                severity=SeverityLevel.SEV_4,
                payload_schema={
                    "evidence_pack_id": "string",
                    "workflow_id": "string",
                    "compliance_framework": "string",
                    "tenant_id": "integer",
                    "evidence_type": "string"
                },
                persona_visibility=[PersonaType.COMPLIANCE, PersonaType.REGULATOR],
                compliance_relevant=True
            )
        }
    
    def _build_persona_metrics(self) -> Dict[PersonaType, List[str]]:
        """Build persona-specific metric views"""
        
        persona_metrics = {}
        
        for persona in PersonaType:
            persona_metrics[persona] = [
                name for name, definition in self.metrics.items()
                if persona in definition.persona_visibility
            ]
        
        return persona_metrics
    
    def _build_persona_logs(self) -> Dict[PersonaType, List[str]]:
        """Build persona-specific log views"""
        
        persona_logs = {}
        
        for persona in PersonaType:
            persona_logs[persona] = [
                name for name, definition in self.logs.items()
                if persona in definition.persona_visibility
            ]
        
        return persona_logs
    
    def _build_persona_traces(self) -> Dict[PersonaType, List[str]]:
        """Build persona-specific trace views"""
        
        persona_traces = {}
        
        for persona in PersonaType:
            persona_traces[persona] = [
                name for name, definition in self.traces.items()
                if persona in definition.persona_visibility
            ]
        
        return persona_traces
    
    def _build_persona_events(self) -> Dict[PersonaType, List[str]]:
        """Build persona-specific event views"""
        
        persona_events = {}
        
        for persona in PersonaType:
            persona_events[persona] = [
                name for name, definition in self.events.items()
                if persona in definition.persona_visibility
            ]
        
        return persona_events
    
    def get_metrics_for_persona(self, persona: PersonaType) -> List[MetricDefinition]:
        """Get metrics visible to a specific persona"""
        
        metric_names = self.persona_metrics.get(persona, [])
        return [self.metrics[name] for name in metric_names]
    
    def get_logs_for_persona(self, persona: PersonaType) -> List[LogDefinition]:
        """Get logs visible to a specific persona"""
        
        log_names = self.persona_logs.get(persona, [])
        return [self.logs[name] for name in log_names]
    
    def get_traces_for_persona(self, persona: PersonaType) -> List[TraceDefinition]:
        """Get traces visible to a specific persona"""
        
        trace_names = self.persona_traces.get(persona, [])
        return [self.traces[name] for name in trace_names]
    
    def get_events_for_persona(self, persona: PersonaType) -> List[EventDefinition]:
        """Get events visible to a specific persona"""
        
        event_names = self.persona_events.get(persona, [])
        return [self.events[name] for name in event_names]
    
    def get_sla_relevant_metrics(self) -> List[MetricDefinition]:
        """Get metrics relevant for SLA monitoring"""
        
        return [definition for definition in self.metrics.values() if definition.sla_relevant]
    
    def get_compliance_relevant_telemetry(self) -> Dict[str, List[Any]]:
        """Get all telemetry relevant for compliance"""
        
        return {
            "metrics": [definition for definition in self.metrics.values() if definition.compliance_relevant],
            "logs": [definition for definition in self.logs.values() if definition.compliance_relevant],
            "traces": [definition for definition in self.traces.values() if definition.compliance_relevant],
            "events": [definition for definition in self.events.values() if definition.compliance_relevant]
        }
    
    def validate_telemetry_context(self, context: ObservabilityContext) -> List[str]:
        """Validate observability context for completeness"""
        
        violations = []
        
        # Required fields
        if not context.tenant_id:
            violations.append("tenant_id is required")
        
        if not context.region_id:
            violations.append("region_id is required")
        
        if not context.sla_tier:
            violations.append("sla_tier is required")
        
        # Compliance requirements
        if context.region_id in ['EU', 'UK', 'IN'] and not context.consent_id:
            violations.append(f"consent_id required for region {context.region_id}")
        
        # SLA tier validation
        if context.sla_tier not in ['T0', 'T1', 'T2', 'T3']:
            violations.append(f"Invalid sla_tier: {context.sla_tier}")
        
        return violations
    
    def get_taxonomy_summary(self) -> Dict[str, Any]:
        """Get summary of the observability taxonomy"""
        
        return {
            "metrics": {
                "total": len(self.metrics),
                "by_type": {metric_type.value: len([m for m in self.metrics.values() if m.metric_type == metric_type]) 
                           for metric_type in MetricType},
                "sla_relevant": len([m for m in self.metrics.values() if m.sla_relevant]),
                "compliance_relevant": len([m for m in self.metrics.values() if m.compliance_relevant])
            },
            "logs": {
                "total": len(self.logs),
                "by_level": {level.value: len([l for l in self.logs.values() if l.level == level]) 
                            for level in LogLevel},
                "compliance_relevant": len([l for l in self.logs.values() if l.compliance_relevant])
            },
            "traces": {
                "total": len(self.traces),
                "by_type": {span_type.value: len([t for t in self.traces.values() if t.span_type == span_type]) 
                           for span_type in TraceSpanType},
                "sla_relevant": len([t for t in self.traces.values() if t.sla_relevant]),
                "compliance_relevant": len([t for t in self.traces.values() if t.compliance_relevant])
            },
            "events": {
                "total": len(self.events),
                "by_type": {event_type.value: len([e for e in self.events.values() if e.event_type == event_type]) 
                           for event_type in EventType},
                "by_severity": {severity.value: len([e for e in self.events.values() if e.severity == severity]) 
                               for severity in SeverityLevel},
                "compliance_relevant": len([e for e in self.events.values() if e.compliance_relevant]),
                "evidence_generating": len([e for e in self.events.values() if e.evidence_generation])
            },
            "personas": {
                "total": len(PersonaType),
                "metrics_per_persona": {persona.value: len(self.persona_metrics.get(persona, [])) 
                                       for persona in PersonaType},
                "logs_per_persona": {persona.value: len(self.persona_logs.get(persona, [])) 
                                    for persona in PersonaType},
                "traces_per_persona": {persona.value: len(self.persona_traces.get(persona, [])) 
                                      for persona in PersonaType},
                "events_per_persona": {persona.value: len(self.persona_events.get(persona, [])) 
                                      for persona in PersonaType}
            }
        }

# Global observability taxonomy instance
observability_taxonomy = ObservabilityTaxonomy()

def get_observability_taxonomy() -> ObservabilityTaxonomy:
    """Get the global observability taxonomy instance"""
    return observability_taxonomy
