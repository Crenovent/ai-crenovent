"""
Observability Taxonomy API
=========================

Task 7.4.1: Define observability taxonomy (metrics, logs, traces, events)
API endpoints for accessing and managing observability taxonomy
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Import observability taxonomy
try:
    from ..dsl.observability.observability_taxonomy import (
        get_observability_taxonomy,
        ObservabilityContext,
        PersonaType,
        TelemetryType,
        MetricType,
        LogLevel,
        TraceSpanType,
        EventType,
        SeverityLevel
    )
except ImportError:
    # Fallback if observability taxonomy is not available
    def get_observability_taxonomy():
        return None

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class ObservabilityContextRequest(BaseModel):
    """Request model for observability context"""
    tenant_id: int = Field(..., description="Tenant identifier")
    user_id: Optional[int] = Field(None, description="User identifier")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    execution_id: Optional[str] = Field(None, description="Execution identifier")
    step_id: Optional[str] = Field(None, description="Step identifier")
    region_id: str = Field("GLOBAL", description="Region identifier")
    sla_tier: str = Field("T2", description="SLA tier")
    industry_code: str = Field("SAAS", description="Industry code")
    policy_pack_id: Optional[str] = Field(None, description="Policy pack identifier")
    consent_id: Optional[str] = Field(None, description="Consent identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")

class MetricDefinitionResponse(BaseModel):
    """Response model for metric definition"""
    name: str
    metric_type: str
    description: str
    unit: str
    labels: List[str]
    help_text: str
    persona_visibility: List[str]
    sla_relevant: bool
    compliance_relevant: bool
    retention_days: int

class LogDefinitionResponse(BaseModel):
    """Response model for log definition"""
    source: str
    level: str
    message_template: str
    structured_fields: List[str]
    persona_visibility: List[str]
    pii_fields: List[str]
    compliance_relevant: bool
    retention_days: int

class TraceDefinitionResponse(BaseModel):
    """Response model for trace definition"""
    operation_name: str
    span_type: str
    description: str
    expected_duration_ms: int
    tags: List[str]
    persona_visibility: List[str]
    sla_relevant: bool
    compliance_relevant: bool

class EventDefinitionResponse(BaseModel):
    """Response model for event definition"""
    event_type: str
    description: str
    severity: str
    payload_schema: Dict[str, Any]
    persona_visibility: List[str]
    alert_rules: List[str]
    compliance_relevant: bool
    evidence_generation: bool

class TaxonomySummaryResponse(BaseModel):
    """Response model for taxonomy summary"""
    metrics: Dict[str, Any]
    logs: Dict[str, Any]
    traces: Dict[str, Any]
    events: Dict[str, Any]
    personas: Dict[str, Any]

@router.get("/summary", response_model=TaxonomySummaryResponse)
async def get_taxonomy_summary():
    """
    Get summary of the observability taxonomy
    
    Returns high-level statistics about metrics, logs, traces, and events
    """
    try:
        logger.info("📊 Getting observability taxonomy summary")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        summary = taxonomy.get_taxonomy_summary()
        
        logger.info("✅ Retrieved observability taxonomy summary")
        
        return TaxonomySummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"❌ Failed to get taxonomy summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@router.get("/metrics", response_model=List[MetricDefinitionResponse])
async def get_metrics(
    persona: Optional[str] = Query(None, description="Filter by persona type"),
    sla_relevant: Optional[bool] = Query(None, description="Filter by SLA relevance"),
    compliance_relevant: Optional[bool] = Query(None, description="Filter by compliance relevance")
):
    """
    Get metric definitions
    
    Returns all metric definitions, optionally filtered by persona or relevance
    """
    try:
        logger.info(f"📈 Getting metrics (persona: {persona}, sla: {sla_relevant}, compliance: {compliance_relevant})")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        # Get metrics based on filters
        if persona:
            try:
                persona_type = PersonaType(persona.lower())
                metrics = taxonomy.get_metrics_for_persona(persona_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid persona type: {persona}")
        else:
            metrics = list(taxonomy.metrics.values())
        
        # Apply additional filters
        if sla_relevant is not None:
            metrics = [m for m in metrics if m.sla_relevant == sla_relevant]
        
        if compliance_relevant is not None:
            metrics = [m for m in metrics if m.compliance_relevant == compliance_relevant]
        
        # Convert to response format
        response = []
        for metric in metrics:
            response.append(MetricDefinitionResponse(
                name=metric.name,
                metric_type=metric.metric_type.value,
                description=metric.description,
                unit=metric.unit,
                labels=metric.labels,
                help_text=metric.help_text,
                persona_visibility=[p.value for p in metric.persona_visibility],
                sla_relevant=metric.sla_relevant,
                compliance_relevant=metric.compliance_relevant,
                retention_days=metric.retention_days
            ))
        
        logger.info(f"✅ Retrieved {len(response)} metrics")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/logs", response_model=List[LogDefinitionResponse])
async def get_logs(
    persona: Optional[str] = Query(None, description="Filter by persona type"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    compliance_relevant: Optional[bool] = Query(None, description="Filter by compliance relevance")
):
    """
    Get log definitions
    
    Returns all log definitions, optionally filtered by persona, level, or relevance
    """
    try:
        logger.info(f"📝 Getting logs (persona: {persona}, level: {level}, compliance: {compliance_relevant})")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        # Get logs based on filters
        if persona:
            try:
                persona_type = PersonaType(persona.lower())
                logs = taxonomy.get_logs_for_persona(persona_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid persona type: {persona}")
        else:
            logs = list(taxonomy.logs.values())
        
        # Apply additional filters
        if level:
            try:
                log_level = LogLevel(level.lower())
                logs = [l for l in logs if l.level == log_level]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid log level: {level}")
        
        if compliance_relevant is not None:
            logs = [l for l in logs if l.compliance_relevant == compliance_relevant]
        
        # Convert to response format
        response = []
        for log in logs:
            response.append(LogDefinitionResponse(
                source=log.source,
                level=log.level.value,
                message_template=log.message_template,
                structured_fields=log.structured_fields,
                persona_visibility=[p.value for p in log.persona_visibility],
                pii_fields=log.pii_fields,
                compliance_relevant=log.compliance_relevant,
                retention_days=log.retention_days
            ))
        
        logger.info(f"✅ Retrieved {len(response)} logs")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@router.get("/traces", response_model=List[TraceDefinitionResponse])
async def get_traces(
    persona: Optional[str] = Query(None, description="Filter by persona type"),
    span_type: Optional[str] = Query(None, description="Filter by span type"),
    sla_relevant: Optional[bool] = Query(None, description="Filter by SLA relevance"),
    compliance_relevant: Optional[bool] = Query(None, description="Filter by compliance relevance")
):
    """
    Get trace definitions
    
    Returns all trace definitions, optionally filtered by persona, type, or relevance
    """
    try:
        logger.info(f"🔍 Getting traces (persona: {persona}, span_type: {span_type}, sla: {sla_relevant}, compliance: {compliance_relevant})")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        # Get traces based on filters
        if persona:
            try:
                persona_type = PersonaType(persona.lower())
                traces = taxonomy.get_traces_for_persona(persona_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid persona type: {persona}")
        else:
            traces = list(taxonomy.traces.values())
        
        # Apply additional filters
        if span_type:
            try:
                trace_span_type = TraceSpanType(span_type.lower())
                traces = [t for t in traces if t.span_type == trace_span_type]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid span type: {span_type}")
        
        if sla_relevant is not None:
            traces = [t for t in traces if t.sla_relevant == sla_relevant]
        
        if compliance_relevant is not None:
            traces = [t for t in traces if t.compliance_relevant == compliance_relevant]
        
        # Convert to response format
        response = []
        for trace in traces:
            response.append(TraceDefinitionResponse(
                operation_name=trace.operation_name,
                span_type=trace.span_type.value,
                description=trace.description,
                expected_duration_ms=trace.expected_duration_ms,
                tags=trace.tags,
                persona_visibility=[p.value for p in trace.persona_visibility],
                sla_relevant=trace.sla_relevant,
                compliance_relevant=trace.compliance_relevant
            ))
        
        logger.info(f"✅ Retrieved {len(response)} traces")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get traces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get traces: {str(e)}")

@router.get("/events", response_model=List[EventDefinitionResponse])
async def get_events(
    persona: Optional[str] = Query(None, description="Filter by persona type"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    compliance_relevant: Optional[bool] = Query(None, description="Filter by compliance relevance"),
    evidence_generation: Optional[bool] = Query(None, description="Filter by evidence generation")
):
    """
    Get event definitions
    
    Returns all event definitions, optionally filtered by persona, type, severity, or relevance
    """
    try:
        logger.info(f"📅 Getting events (persona: {persona}, event_type: {event_type}, severity: {severity})")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        # Get events based on filters
        if persona:
            try:
                persona_type = PersonaType(persona.lower())
                events = taxonomy.get_events_for_persona(persona_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid persona type: {persona}")
        else:
            events = list(taxonomy.events.values())
        
        # Apply additional filters
        if event_type:
            try:
                event_type_enum = EventType(event_type.lower())
                events = [e for e in events if e.event_type == event_type_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        if severity:
            try:
                severity_level = SeverityLevel(severity.lower())
                events = [e for e in events if e.severity == severity_level]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity level: {severity}")
        
        if compliance_relevant is not None:
            events = [e for e in events if e.compliance_relevant == compliance_relevant]
        
        if evidence_generation is not None:
            events = [e for e in events if e.evidence_generation == evidence_generation]
        
        # Convert to response format
        response = []
        for event in events:
            response.append(EventDefinitionResponse(
                event_type=event.event_type.value,
                description=event.description,
                severity=event.severity.value,
                payload_schema=event.payload_schema,
                persona_visibility=[p.value for p in event.persona_visibility],
                alert_rules=event.alert_rules,
                compliance_relevant=event.compliance_relevant,
                evidence_generation=event.evidence_generation
            ))
        
        logger.info(f"✅ Retrieved {len(response)} events")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get events: {str(e)}")

@router.post("/validate-context")
async def validate_observability_context(
    context: ObservabilityContextRequest
):
    """
    Validate observability context for completeness and compliance
    
    Checks if the provided context meets all requirements for telemetry
    """
    try:
        logger.info(f"✅ Validating observability context for tenant {context.tenant_id}")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        # Convert request to ObservabilityContext
        obs_context = ObservabilityContext(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            workflow_id=context.workflow_id,
            execution_id=context.execution_id,
            step_id=context.step_id,
            region_id=context.region_id,
            sla_tier=context.sla_tier,
            industry_code=context.industry_code,
            policy_pack_id=context.policy_pack_id,
            consent_id=context.consent_id,
            session_id=context.session_id,
            correlation_id=context.correlation_id
        )
        
        # Validate context
        violations = taxonomy.validate_telemetry_context(obs_context)
        
        result = {
            "valid": len(violations) == 0,
            "violations": violations,
            "context": {
                "tenant_id": context.tenant_id,
                "region_id": context.region_id,
                "sla_tier": context.sla_tier,
                "industry_code": context.industry_code
            },
            "recommendations": []
        }
        
        # Add recommendations for violations
        if violations:
            for violation in violations:
                if "consent_id" in violation:
                    result["recommendations"].append("Obtain user consent before processing personal data")
                elif "sla_tier" in violation:
                    result["recommendations"].append("Use valid SLA tier: T0, T1, T2, or T3")
                elif "tenant_id" in violation:
                    result["recommendations"].append("Provide valid tenant identifier")
                elif "region_id" in violation:
                    result["recommendations"].append("Specify region for compliance validation")
        
        logger.info(f"✅ Context validation completed: {'valid' if result['valid'] else 'invalid'}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to validate context: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/compliance-telemetry")
async def get_compliance_relevant_telemetry():
    """
    Get all telemetry definitions relevant for compliance
    
    Returns metrics, logs, traces, and events that are compliance-relevant
    """
    try:
        logger.info("📋 Getting compliance-relevant telemetry")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        compliance_telemetry = taxonomy.get_compliance_relevant_telemetry()
        
        result = {
            "metrics": [
                {
                    "name": metric.name,
                    "description": metric.description,
                    "retention_days": metric.retention_days,
                    "persona_visibility": [p.value for p in metric.persona_visibility]
                }
                for metric in compliance_telemetry["metrics"]
            ],
            "logs": [
                {
                    "source": log.source,
                    "message_template": log.message_template,
                    "retention_days": log.retention_days,
                    "persona_visibility": [p.value for p in log.persona_visibility]
                }
                for log in compliance_telemetry["logs"]
            ],
            "traces": [
                {
                    "operation_name": trace.operation_name,
                    "description": trace.description,
                    "persona_visibility": [p.value for p in trace.persona_visibility]
                }
                for trace in compliance_telemetry["traces"]
            ],
            "events": [
                {
                    "event_type": event.event_type.value,
                    "description": event.description,
                    "evidence_generation": event.evidence_generation,
                    "persona_visibility": [p.value for p in event.persona_visibility]
                }
                for event in compliance_telemetry["events"]
            ]
        }
        
        total_items = len(result["metrics"]) + len(result["logs"]) + len(result["traces"]) + len(result["events"])
        
        logger.info(f"✅ Retrieved {total_items} compliance-relevant telemetry items")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get compliance telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance telemetry: {str(e)}")

@router.get("/sla-relevant-metrics")
async def get_sla_relevant_metrics():
    """
    Get metrics relevant for SLA monitoring
    
    Returns all metrics that are used for SLA compliance tracking
    """
    try:
        logger.info("⏱️ Getting SLA-relevant metrics")
        
        taxonomy = get_observability_taxonomy()
        if not taxonomy:
            raise HTTPException(status_code=500, detail="Observability taxonomy not available")
        
        sla_metrics = taxonomy.get_sla_relevant_metrics()
        
        result = [
            {
                "name": metric.name,
                "description": metric.description,
                "unit": metric.unit,
                "labels": metric.labels,
                "persona_visibility": [p.value for p in metric.persona_visibility],
                "retention_days": metric.retention_days
            }
            for metric in sla_metrics
        ]
        
        logger.info(f"✅ Retrieved {len(result)} SLA-relevant metrics")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get SLA metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLA metrics: {str(e)}")

@router.get("/health")
async def observability_taxonomy_health_check():
    """
    Health check for observability taxonomy service
    """
    try:
        taxonomy = get_observability_taxonomy()
        
        return {
            "status": "healthy" if taxonomy else "unhealthy",
            "service": "Observability Taxonomy API",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "metrics_taxonomy": taxonomy is not None,
                "logs_taxonomy": taxonomy is not None,
                "traces_taxonomy": taxonomy is not None,
                "events_taxonomy": taxonomy is not None,
                "persona_filtering": taxonomy is not None,
                "compliance_filtering": taxonomy is not None,
                "context_validation": taxonomy is not None
            },
            "taxonomy_stats": taxonomy.get_taxonomy_summary() if taxonomy else {}
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
