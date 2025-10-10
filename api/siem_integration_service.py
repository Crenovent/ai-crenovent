"""
Task 3.2.21: SIEM/SOC integration for governance events
- Override, drift, kill, failed gate events
- Central security visibility
- Syslog/OpenTelemetry
- Correlate by tenant
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging
import json
import socket
import asyncio
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
import syslog
import time

app = FastAPI(title="RBIA SIEM/SOC Integration Service")
logger = logging.getLogger(__name__)

class GovernanceEventType(str, Enum):
    OVERRIDE_RECORDED = "override_recorded"
    DRIFT_DETECTED = "drift_detected"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    POLICY_GATE_FAILED = "policy_gate_failed"
    BIAS_VIOLATION = "bias_violation"
    TRUST_SCORE_DEGRADED = "trust_score_degraded"
    EVIDENCE_TAMPERED = "evidence_tampered"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SECURITY_INCIDENT = "security_incident"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SIEMEventStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"

class SIEMDestination(str, Enum):
    SYSLOG = "syslog"
    OPENTELEMETRY = "opentelemetry"
    SPLUNK = "splunk"
    ELASTIC = "elastic"
    SENTINEL = "sentinel"

class GovernanceSecurityEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: GovernanceEventType = Field(..., description="Type of governance event")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="User who triggered event")
    service_name: str = Field(..., description="Service that generated event")
    severity: SeverityLevel = Field(..., description="Event severity level")
    
    # Event details
    title: str = Field(..., description="Human-readable event title")
    description: str = Field(..., description="Detailed event description")
    affected_resource: str = Field(..., description="Resource affected by event")
    risk_score: float = Field(..., ge=0.0, le=10.0, description="Risk score (0-10)")
    
    # Context
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    model_id: Optional[str] = Field(None, description="Associated model ID")
    policy_id: Optional[str] = Field(None, description="Associated policy ID")
    
    # Metadata
    event_data: Dict[str, Any] = Field(default={}, description="Additional event data")
    tags: List[str] = Field(default=[], description="Event tags for correlation")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    
    # Timestamps
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    # SIEM integration
    status: SIEMEventStatus = Field(default=SIEMEventStatus.PENDING)
    destinations: List[SIEMDestination] = Field(default=[], description="SIEM destinations")
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class SIEMConfiguration(BaseModel):
    syslog_enabled: bool = Field(default=True)
    syslog_host: str = Field(default="localhost")
    syslog_port: int = Field(default=514)
    syslog_facility: int = Field(default=syslog.LOG_LOCAL0)
    
    opentelemetry_enabled: bool = Field(default=True)
    opentelemetry_endpoint: str = Field(default="http://localhost:4317")
    
    splunk_enabled: bool = Field(default=False)
    splunk_hec_url: Optional[str] = Field(None)
    splunk_token: Optional[str] = Field(None)
    
    elastic_enabled: bool = Field(default=False)
    elastic_url: Optional[str] = Field(None)
    elastic_index: str = Field(default="rbia-security-events")
    
    sentinel_enabled: bool = Field(default=False)
    sentinel_workspace_id: Optional[str] = Field(None)
    sentinel_shared_key: Optional[str] = Field(None)

class SIEMIntegrationService:
    """SIEM/SOC integration service for governance events"""
    
    def __init__(self, config: SIEMConfiguration):
        self.config = config
        self.events_queue: List[GovernanceSecurityEvent] = []
        self.correlation_cache: Dict[str, List[str]] = {}
        
        # Initialize OpenTelemetry if enabled
        if config.opentelemetry_enabled:
            self._setup_opentelemetry()
        
        # Initialize syslog if enabled
        if config.syslog_enabled:
            self._setup_syslog()
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": "rbia-governance",
            "service.version": "1.0.0"
        })
        
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.opentelemetry_endpoint,
            insecure=True
        )
        
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        logger.info("OpenTelemetry initialized")
    
    def _setup_syslog(self):
        """Setup syslog connection"""
        try:
            syslog.openlog("rbia-governance", syslog.LOG_PID, self.config.syslog_facility)
            logger.info(f"Syslog initialized - {self.config.syslog_host}:{self.config.syslog_port}")
        except Exception as e:
            logger.error(f"Failed to initialize syslog: {e}")
    
    async def send_governance_event(self, event: GovernanceSecurityEvent) -> bool:
        """Send governance event to configured SIEM destinations"""
        success = True
        
        try:
            # Add to correlation cache
            self._add_to_correlation_cache(event)
            
            # Send to each configured destination
            if self.config.syslog_enabled:
                await self._send_to_syslog(event)
                event.destinations.append(SIEMDestination.SYSLOG)
            
            if self.config.opentelemetry_enabled:
                await self._send_to_opentelemetry(event)
                event.destinations.append(SIEMDestination.OPENTELEMETRY)
            
            if self.config.splunk_enabled:
                await self._send_to_splunk(event)
                event.destinations.append(SIEMDestination.SPLUNK)
            
            if self.config.elastic_enabled:
                await self._send_to_elastic(event)
                event.destinations.append(SIEMDestination.ELASTIC)
            
            if self.config.sentinel_enabled:
                await self._send_to_sentinel(event)
                event.destinations.append(SIEMDestination.SENTINEL)
            
            event.status = SIEMEventStatus.SENT
            logger.info(f"Governance event sent to SIEM: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to send governance event: {e}")
            event.status = SIEMEventStatus.FAILED
            success = False
        
        return success
    
    def _add_to_correlation_cache(self, event: GovernanceSecurityEvent):
        """Add event to correlation cache for tenant-based correlation"""
        tenant_key = f"tenant_{event.tenant_id}"
        
        if tenant_key not in self.correlation_cache:
            self.correlation_cache[tenant_key] = []
        
        self.correlation_cache[tenant_key].append(event.event_id)
        
        # Keep only last 100 events per tenant
        if len(self.correlation_cache[tenant_key]) > 100:
            self.correlation_cache[tenant_key] = self.correlation_cache[tenant_key][-100:]
    
    async def _send_to_syslog(self, event: GovernanceSecurityEvent):
        """Send event to syslog"""
        # Map severity to syslog priority
        severity_map = {
            SeverityLevel.LOW: syslog.LOG_INFO,
            SeverityLevel.MEDIUM: syslog.LOG_WARNING,
            SeverityLevel.HIGH: syslog.LOG_ERR,
            SeverityLevel.CRITICAL: syslog.LOG_CRIT
        }
        
        priority = severity_map.get(event.severity, syslog.LOG_INFO)
        
        # Create structured syslog message
        message = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "tenant_id": event.tenant_id,
            "severity": event.severity.value,
            "title": event.title,
            "description": event.description,
            "affected_resource": event.affected_resource,
            "risk_score": event.risk_score,
            "correlation_id": event.correlation_id,
            "occurred_at": event.occurred_at.isoformat(),
            "service": "rbia-governance"
        }
        
        syslog_message = f"RBIA_GOVERNANCE: {json.dumps(message)}"
        syslog.syslog(priority, syslog_message)
    
    async def _send_to_opentelemetry(self, event: GovernanceSecurityEvent):
        """Send event to OpenTelemetry"""
        if not hasattr(self, 'tracer'):
            return
        
        with self.tracer.start_as_current_span("governance_event") as span:
            span.set_attribute("event.id", event.event_id)
            span.set_attribute("event.type", event.event_type.value)
            span.set_attribute("tenant.id", event.tenant_id)
            span.set_attribute("severity", event.severity.value)
            span.set_attribute("risk.score", event.risk_score)
            span.set_attribute("correlation.id", event.correlation_id)
            
            if event.user_id:
                span.set_attribute("user.id", event.user_id)
            
            if event.workflow_id:
                span.set_attribute("workflow.id", event.workflow_id)
            
            if event.model_id:
                span.set_attribute("model.id", event.model_id)
            
            # Add event data as span events
            span.add_event(
                name=event.title,
                attributes={
                    "description": event.description,
                    "affected_resource": event.affected_resource,
                    "event_data": json.dumps(event.event_data)
                }
            )
    
    async def _send_to_splunk(self, event: GovernanceSecurityEvent):
        """Send event to Splunk HEC"""
        if not self.config.splunk_hec_url or not self.config.splunk_token:
            return
        
        # Splunk HEC format
        splunk_event = {
            "time": int(event.occurred_at.timestamp()),
            "event": {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "tenant_id": event.tenant_id,
                "severity": event.severity.value,
                "title": event.title,
                "description": event.description,
                "affected_resource": event.affected_resource,
                "risk_score": event.risk_score,
                "correlation_id": event.correlation_id,
                "service": "rbia-governance",
                **event.event_data
            },
            "sourcetype": "rbia:governance",
            "source": "rbia-governance-service",
            "index": "security"
        }
        
        # In production, would use HTTP client to send to Splunk HEC
        logger.info(f"Would send to Splunk HEC: {json.dumps(splunk_event)}")
    
    async def _send_to_elastic(self, event: GovernanceSecurityEvent):
        """Send event to Elasticsearch"""
        if not self.config.elastic_url:
            return
        
        # Elasticsearch document
        elastic_doc = {
            "@timestamp": event.occurred_at.isoformat(),
            "event": {
                "id": event.event_id,
                "type": event.event_type.value,
                "severity": event.severity.value,
                "risk_score": event.risk_score
            },
            "tenant": {
                "id": event.tenant_id
            },
            "governance": {
                "title": event.title,
                "description": event.description,
                "affected_resource": event.affected_resource,
                "correlation_id": event.correlation_id,
                "service": "rbia-governance"
            },
            "tags": event.tags,
            **event.event_data
        }
        
        # In production, would use Elasticsearch client
        logger.info(f"Would send to Elasticsearch: {json.dumps(elastic_doc)}")
    
    async def _send_to_sentinel(self, event: GovernanceSecurityEvent):
        """Send event to Azure Sentinel"""
        if not self.config.sentinel_workspace_id or not self.config.sentinel_shared_key:
            return
        
        # Azure Sentinel format
        sentinel_event = {
            "TimeGenerated": event.occurred_at.isoformat(),
            "EventId": event.event_id,
            "EventType": event.event_type.value,
            "TenantId": event.tenant_id,
            "Severity": event.severity.value,
            "Title": event.title,
            "Description": event.description,
            "AffectedResource": event.affected_resource,
            "RiskScore": event.risk_score,
            "CorrelationId": event.correlation_id,
            "Service": "rbia-governance"
        }
        
        # In production, would use Azure Monitor API
        logger.info(f"Would send to Azure Sentinel: {json.dumps(sentinel_event)}")

# Global SIEM service instance
siem_config = SIEMConfiguration()
siem_service = SIEMIntegrationService(siem_config)

# In-memory storage for events (replace with actual database)
governance_events: Dict[str, GovernanceSecurityEvent] = {}

@app.post("/siem/events", response_model=GovernanceSecurityEvent)
async def create_governance_event(
    event_type: GovernanceEventType,
    tenant_id: str,
    title: str,
    description: str,
    affected_resource: str,
    severity: SeverityLevel = SeverityLevel.MEDIUM,
    user_id: Optional[str] = None,
    service_name: str = "rbia-governance",
    risk_score: float = 5.0,
    workflow_id: Optional[str] = None,
    model_id: Optional[str] = None,
    policy_id: Optional[str] = None,
    event_data: Dict[str, Any] = {},
    tags: List[str] = [],
    source_ip: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Create and send governance security event to SIEM
    """
    event = GovernanceSecurityEvent(
        event_type=event_type,
        tenant_id=tenant_id,
        user_id=user_id,
        service_name=service_name,
        severity=severity,
        title=title,
        description=description,
        affected_resource=affected_resource,
        risk_score=risk_score,
        workflow_id=workflow_id,
        model_id=model_id,
        policy_id=policy_id,
        event_data=event_data,
        tags=tags,
        source_ip=source_ip
    )
    
    governance_events[event.event_id] = event
    
    # Send to SIEM in background
    if background_tasks:
        background_tasks.add_task(siem_service.send_governance_event, event)
    else:
        await siem_service.send_governance_event(event)
    
    logger.info(f"Governance event created: {event.event_id} - {event.title}")
    
    return event

@app.post("/siem/events/override")
async def report_override_event(
    tenant_id: str,
    override_id: str,
    user_id: str,
    original_decision: Dict[str, Any],
    override_decision: Dict[str, Any],
    override_reason: str,
    workflow_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Report override recorded event"""
    
    # Determine severity based on override impact
    severity = SeverityLevel.HIGH if "critical" in override_reason.lower() else SeverityLevel.MEDIUM
    
    event_data = {
        "override_id": override_id,
        "original_decision": original_decision,
        "override_decision": override_decision,
        "override_reason": override_reason
    }
    
    return await create_governance_event(
        event_type=GovernanceEventType.OVERRIDE_RECORDED,
        tenant_id=tenant_id,
        title=f"Governance Override Recorded",
        description=f"User {user_id} overrode system decision: {override_reason}",
        affected_resource=f"workflow:{workflow_id}" if workflow_id else "system",
        severity=severity,
        user_id=user_id,
        risk_score=7.0 if severity == SeverityLevel.HIGH else 5.0,
        workflow_id=workflow_id,
        event_data=event_data,
        tags=["override", "governance", "audit"],
        background_tasks=background_tasks
    )

@app.post("/siem/events/drift")
async def report_drift_event(
    tenant_id: str,
    model_id: str,
    drift_type: str,
    drift_score: float,
    threshold: float,
    background_tasks: BackgroundTasks = None
):
    """Report model drift detected event"""
    
    severity = SeverityLevel.CRITICAL if drift_score > threshold * 2 else SeverityLevel.HIGH
    
    event_data = {
        "model_id": model_id,
        "drift_type": drift_type,
        "drift_score": drift_score,
        "threshold": threshold,
        "drift_ratio": drift_score / threshold
    }
    
    return await create_governance_event(
        event_type=GovernanceEventType.DRIFT_DETECTED,
        tenant_id=tenant_id,
        title=f"Model Drift Detected - {drift_type}",
        description=f"Model {model_id} drift score {drift_score} exceeds threshold {threshold}",
        affected_resource=f"model:{model_id}",
        severity=severity,
        risk_score=8.0 if severity == SeverityLevel.CRITICAL else 6.0,
        model_id=model_id,
        event_data=event_data,
        tags=["drift", "model", "performance"],
        background_tasks=background_tasks
    )

@app.post("/siem/events/kill-switch")
async def report_kill_switch_event(
    tenant_id: str,
    switch_id: str,
    target_resource: str,
    triggered_by: str,
    reason: str,
    scope: str = "model",
    background_tasks: BackgroundTasks = None
):
    """Report kill switch activation event"""
    
    event_data = {
        "switch_id": switch_id,
        "target_resource": target_resource,
        "triggered_by": triggered_by,
        "reason": reason,
        "scope": scope
    }
    
    return await create_governance_event(
        event_type=GovernanceEventType.KILL_SWITCH_ACTIVATED,
        tenant_id=tenant_id,
        title=f"Kill Switch Activated - {scope}",
        description=f"Kill switch {switch_id} activated by {triggered_by}: {reason}",
        affected_resource=target_resource,
        severity=SeverityLevel.CRITICAL,
        user_id=triggered_by,
        risk_score=9.0,
        event_data=event_data,
        tags=["kill-switch", "emergency", "safety"],
        background_tasks=background_tasks
    )

@app.post("/siem/events/gate-failed")
async def report_gate_failure_event(
    tenant_id: str,
    gate_type: str,
    policy_id: str,
    failure_reason: str,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Report policy gate failure event"""
    
    event_data = {
        "gate_type": gate_type,
        "policy_id": policy_id,
        "failure_reason": failure_reason
    }
    
    return await create_governance_event(
        event_type=GovernanceEventType.POLICY_GATE_FAILED,
        tenant_id=tenant_id,
        title=f"Policy Gate Failed - {gate_type}",
        description=f"Policy gate {gate_type} failed: {failure_reason}",
        affected_resource=f"policy:{policy_id}",
        severity=SeverityLevel.HIGH,
        user_id=user_id,
        risk_score=6.0,
        workflow_id=workflow_id,
        policy_id=policy_id,
        event_data=event_data,
        tags=["policy", "gate", "compliance"],
        background_tasks=background_tasks
    )

@app.get("/siem/events")
async def list_governance_events(
    tenant_id: Optional[str] = None,
    event_type: Optional[GovernanceEventType] = None,
    severity: Optional[SeverityLevel] = None,
    status: Optional[SIEMEventStatus] = None,
    limit: int = 100
):
    """List governance security events with filtering"""
    filtered_events = list(governance_events.values())
    
    if tenant_id:
        filtered_events = [e for e in filtered_events if e.tenant_id == tenant_id]
    
    if event_type:
        filtered_events = [e for e in filtered_events if e.event_type == event_type]
    
    if severity:
        filtered_events = [e for e in filtered_events if e.severity == severity]
    
    if status:
        filtered_events = [e for e in filtered_events if e.status == status]
    
    # Sort by occurred_at, most recent first
    filtered_events.sort(key=lambda x: x.occurred_at, reverse=True)
    
    return {
        "events": filtered_events[:limit],
        "total_count": len(filtered_events)
    }

@app.get("/siem/correlation/{tenant_id}")
async def get_tenant_correlation(tenant_id: str, hours: int = 24):
    """Get correlated events for tenant within time window"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    tenant_events = [
        e for e in governance_events.values()
        if e.tenant_id == tenant_id and e.occurred_at >= cutoff_time
    ]
    
    # Group by correlation patterns
    correlations = {
        "total_events": len(tenant_events),
        "by_type": {},
        "by_severity": {},
        "by_resource": {},
        "timeline": []
    }
    
    for event in tenant_events:
        # Count by type
        event_type = event.event_type.value
        correlations["by_type"][event_type] = correlations["by_type"].get(event_type, 0) + 1
        
        # Count by severity
        severity = event.severity.value
        correlations["by_severity"][severity] = correlations["by_severity"].get(severity, 0) + 1
        
        # Count by resource
        resource = event.affected_resource
        correlations["by_resource"][resource] = correlations["by_resource"].get(resource, 0) + 1
        
        # Add to timeline
        correlations["timeline"].append({
            "event_id": event.event_id,
            "event_type": event_type,
            "severity": severity,
            "occurred_at": event.occurred_at.isoformat(),
            "title": event.title
        })
    
    # Sort timeline by time
    correlations["timeline"].sort(key=lambda x: x["occurred_at"])
    
    return correlations

@app.get("/siem/config")
async def get_siem_config():
    """Get current SIEM configuration"""
    return {
        "syslog_enabled": siem_config.syslog_enabled,
        "syslog_host": siem_config.syslog_host,
        "syslog_port": siem_config.syslog_port,
        "opentelemetry_enabled": siem_config.opentelemetry_enabled,
        "opentelemetry_endpoint": siem_config.opentelemetry_endpoint,
        "splunk_enabled": siem_config.splunk_enabled,
        "elastic_enabled": siem_config.elastic_enabled,
        "sentinel_enabled": siem_config.sentinel_enabled,
        "supported_destinations": [dest.value for dest in SIEMDestination]
    }

@app.get("/health")
async def health_check():
    total_events = len(governance_events)
    sent_events = len([e for e in governance_events.values() if e.status == SIEMEventStatus.SENT])
    failed_events = len([e for e in governance_events.values() if e.status == SIEMEventStatus.FAILED])
    
    return {
        "status": "healthy",
        "service": "RBIA SIEM/SOC Integration Service",
        "task": "3.2.21",
        "total_events": total_events,
        "sent_events": sent_events,
        "failed_events": failed_events,
        "success_rate": (sent_events / total_events * 100) if total_events > 0 else 100,
        "siem_destinations": len([dest for dest in SIEMDestination]),
        "correlation_tenants": len(siem_service.correlation_cache)
    }

