"""
Task 4.3.30: Audit Trail Viewer API
Backend service for timeline of overrides, drift alerts, justifications
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

# Import existing services
from dsl.operators.override_ledger import OverrideLedger
from dsl.operators.drift_bias_monitor import DriftBiasMonitor
from dsl.operators.explainability_service import ExplainabilityService

app = FastAPI(title="RBIA Audit Trail Viewer API")
logger = logging.getLogger(__name__)

class EventType(str, Enum):
    OVERRIDE = "override"
    DRIFT_ALERT = "drift_alert"
    BIAS_ALERT = "bias_alert"
    EXPLAINABILITY = "explainability"
    QUARANTINE = "quarantine"
    APPROVAL = "approval"

class EventSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditEvent(BaseModel):
    event_id: str
    event_type: EventType
    timestamp: datetime
    severity: EventSeverity
    
    # Core event data
    title: str
    description: str
    model_id: Optional[str] = None
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    
    # User/approval data
    user_id: Optional[int] = None
    approver_id: Optional[int] = None
    justification: Optional[str] = None
    
    # Technical details
    technical_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    tenant_id: str
    tags: List[str] = Field(default_factory=list)

class AuditTrailQuery(BaseModel):
    tenant_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[EventType]] = None
    model_ids: Optional[List[str]] = None
    workflow_ids: Optional[List[str]] = None
    severity_levels: Optional[List[EventSeverity]] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)

class AuditTrailResponse(BaseModel):
    events: List[AuditEvent]
    total_count: int
    filtered_count: int
    query_metadata: Dict[str, Any]

class AuditTrailViewer:
    """Service for aggregating and viewing audit trail events"""
    
    def __init__(self):
        self.override_ledger = OverrideLedger()
        self.drift_monitor = DriftBiasMonitor()
        self.explainability_service = ExplainabilityService()
        self.logger = logging.getLogger(__name__)
    
    async def get_audit_trail(self, query: AuditTrailQuery) -> AuditTrailResponse:
        """Get comprehensive audit trail for a tenant"""
        try:
            all_events = []
            
            # Collect events from different sources
            if not query.event_types or EventType.OVERRIDE in query.event_types:
                override_events = await self._get_override_events(query)
                all_events.extend(override_events)
            
            if not query.event_types or EventType.DRIFT_ALERT in query.event_types:
                drift_events = await self._get_drift_events(query)
                all_events.extend(drift_events)
            
            if not query.event_types or EventType.BIAS_ALERT in query.event_types:
                bias_events = await self._get_bias_events(query)
                all_events.extend(bias_events)
            
            if not query.event_types or EventType.EXPLAINABILITY in query.event_types:
                explainability_events = await self._get_explainability_events(query)
                all_events.extend(explainability_events)
            
            if not query.event_types or EventType.QUARANTINE in query.event_types:
                quarantine_events = await self._get_quarantine_events(query)
                all_events.extend(quarantine_events)
            
            # Filter by date range
            if query.start_date or query.end_date:
                all_events = self._filter_by_date(all_events, query.start_date, query.end_date)
            
            # Filter by severity
            if query.severity_levels:
                all_events = [e for e in all_events if e.severity in query.severity_levels]
            
            # Filter by model/workflow IDs
            if query.model_ids:
                all_events = [e for e in all_events if e.model_id in query.model_ids]
            
            if query.workflow_ids:
                all_events = [e for e in all_events if e.workflow_id in query.workflow_ids]
            
            # Sort by timestamp (most recent first)
            all_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply pagination
            total_count = len(all_events)
            paginated_events = all_events[query.offset:query.offset + query.limit]
            
            return AuditTrailResponse(
                events=paginated_events,
                total_count=total_count,
                filtered_count=len(paginated_events),
                query_metadata={
                    "query_time": datetime.utcnow().isoformat(),
                    "tenant_id": query.tenant_id,
                    "filters_applied": {
                        "date_range": bool(query.start_date or query.end_date),
                        "event_types": bool(query.event_types),
                        "severity": bool(query.severity_levels),
                        "model_filter": bool(query.model_ids),
                        "workflow_filter": bool(query.workflow_ids)
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get audit trail for tenant {query.tenant_id}: {e}")
            raise
    
    async def _get_override_events(self, query: AuditTrailQuery) -> List[AuditEvent]:
        """Get override events from ledger"""
        try:
            ledger_entries = await self.override_ledger.get_entries_by_tenant(query.tenant_id)
            events = []
            
            for entry in ledger_entries:
                event = AuditEvent(
                    event_id=entry.entry_id,
                    event_type=EventType.OVERRIDE,
                    timestamp=entry.created_at,
                    severity=self._map_override_severity(entry.override_type),
                    title=f"Override: {entry.override_type}",
                    description=f"Model {entry.model_id} prediction overridden",
                    model_id=entry.model_id,
                    workflow_id=entry.workflow_id,
                    step_id=entry.step_id,
                    user_id=entry.requested_by,
                    approver_id=entry.approved_by,
                    justification=entry.justification,
                    technical_details={
                        "original_prediction": entry.original_prediction,
                        "override_prediction": entry.override_prediction,
                        "approval_status": entry.approval_status,
                        "approval_reason": entry.approval_reason
                    },
                    tenant_id=query.tenant_id,
                    tags=["override", entry.override_type]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to get override events: {e}")
            return []
    
    async def _get_drift_events(self, query: AuditTrailQuery) -> List[AuditEvent]:
        """Get drift alert events"""
        try:
            drift_alerts = await self.drift_monitor._get_drift_alerts(query.tenant_id)
            events = []
            
            for alert in drift_alerts:
                event = AuditEvent(
                    event_id=alert.alert_id,
                    event_type=EventType.DRIFT_ALERT,
                    timestamp=alert.detected_at,
                    severity=EventSeverity(alert.severity),
                    title=f"Drift Alert: {alert.drift_type}",
                    description=alert.description,
                    model_id=alert.model_id,
                    workflow_id=alert.workflow_id,
                    step_id=alert.step_id,
                    technical_details={
                        "drift_score": alert.drift_score,
                        "threshold": alert.threshold,
                        "affected_features": alert.affected_features,
                        "is_resolved": alert.is_resolved
                    },
                    tenant_id=query.tenant_id,
                    tags=["drift", alert.drift_type, alert.severity]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to get drift events: {e}")
            return []
    
    async def _get_bias_events(self, query: AuditTrailQuery) -> List[AuditEvent]:
        """Get bias alert events"""
        try:
            bias_alerts = await self.drift_monitor._get_bias_alerts(query.tenant_id)
            events = []
            
            for alert in bias_alerts:
                event = AuditEvent(
                    event_id=alert.alert_id,
                    event_type=EventType.BIAS_ALERT,
                    timestamp=alert.detected_at,
                    severity=EventSeverity(alert.severity),
                    title=f"Bias Alert: {alert.bias_type}",
                    description=alert.description,
                    model_id=alert.model_id,
                    workflow_id=alert.workflow_id,
                    step_id=alert.step_id,
                    technical_details={
                        "bias_score": alert.bias_score,
                        "threshold": alert.threshold,
                        "protected_attributes": alert.protected_attributes,
                        "affected_groups": alert.affected_groups,
                        "is_resolved": alert.is_resolved
                    },
                    tenant_id=query.tenant_id,
                    tags=["bias", alert.bias_type, alert.severity]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to get bias events: {e}")
            return []
    
    async def _get_explainability_events(self, query: AuditTrailQuery) -> List[AuditEvent]:
        """Get explainability events"""
        try:
            explanations = await self.explainability_service.get_explanations_by_tenant(query.tenant_id)
            events = []
            
            for explanation in explanations[:50]:  # Limit to avoid too many events
                event = AuditEvent(
                    event_id=explanation.log_id,
                    event_type=EventType.EXPLAINABILITY,
                    timestamp=explanation.created_at,
                    severity=EventSeverity.LOW,  # Explainability events are informational
                    title=f"Explanation Generated: {explanation.explanation_type}",
                    description=f"Explanation generated for model {explanation.model_id}",
                    model_id=explanation.model_id,
                    workflow_id=explanation.workflow_id,
                    step_id=explanation.step_id,
                    user_id=explanation.user_id,
                    technical_details={
                        "explanation_type": explanation.explanation_type,
                        "confidence": explanation.confidence,
                        "execution_time_ms": explanation.execution_time_ms
                    },
                    tenant_id=query.tenant_id,
                    tags=["explainability", explanation.explanation_type]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to get explainability events: {e}")
            return []
    
    async def _get_quarantine_events(self, query: AuditTrailQuery) -> List[AuditEvent]:
        """Get quarantine events"""
        try:
            quarantine_events = await self.drift_monitor.get_quarantine_events(query.tenant_id)
            events = []
            
            for qe in quarantine_events:
                event = AuditEvent(
                    event_id=qe['event_id'],
                    event_type=EventType.QUARANTINE,
                    timestamp=datetime.fromisoformat(qe['triggered_at']),
                    severity=EventSeverity.CRITICAL,  # Quarantine is always critical
                    title=f"Model Quarantined: {qe['alert_type']}",
                    description=qe['quarantine_reason'],
                    model_id=qe['model_id'],
                    technical_details={
                        "alert_id": qe['alert_id'],
                        "alert_type": qe['alert_type'],
                        "severity": qe['severity']
                    },
                    tenant_id=query.tenant_id,
                    tags=["quarantine", qe['alert_type'], "critical"]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to get quarantine events: {e}")
            return []
    
    def _filter_by_date(self, events: List[AuditEvent], start_date: Optional[datetime], end_date: Optional[datetime]) -> List[AuditEvent]:
        """Filter events by date range"""
        if not start_date and not end_date:
            return events
        
        filtered = []
        for event in events:
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            filtered.append(event)
        
        return filtered
    
    def _map_override_severity(self, override_type: str) -> EventSeverity:
        """Map override type to severity level"""
        severity_map = {
            "emergency": EventSeverity.CRITICAL,
            "policy": EventSeverity.HIGH,
            "manual": EventSeverity.MEDIUM,
            "compliance": EventSeverity.HIGH,
            "business_rule": EventSeverity.MEDIUM
        }
        return severity_map.get(override_type, EventSeverity.MEDIUM)

# Global service instance
audit_trail_viewer = AuditTrailViewer()

@app.post("/audit-trail/query", response_model=AuditTrailResponse)
async def get_audit_trail(query: AuditTrailQuery):
    """Get audit trail events for a tenant"""
    return await audit_trail_viewer.get_audit_trail(query)

@app.get("/audit-trail/timeline/{tenant_id}")
async def get_audit_timeline(
    tenant_id: str,
    hours: int = Query(default=24, description="Number of hours to look back"),
    event_types: Optional[str] = Query(default=None, description="Comma-separated event types"),
    severity: Optional[str] = Query(default=None, description="Minimum severity level")
):
    """Get audit timeline for the last N hours"""
    
    # Parse parameters
    start_date = datetime.utcnow() - timedelta(hours=hours)
    event_type_list = None
    if event_types:
        event_type_list = [EventType(t.strip()) for t in event_types.split(",")]
    
    severity_list = None
    if severity:
        severity_list = [EventSeverity(severity)]
    
    # Create query
    query = AuditTrailQuery(
        tenant_id=tenant_id,
        start_date=start_date,
        event_types=event_type_list,
        severity_levels=severity_list,
        limit=200
    )
    
    return await audit_trail_viewer.get_audit_trail(query)

@app.get("/audit-trail/summary/{tenant_id}")
async def get_audit_summary(
    tenant_id: str,
    days: int = Query(default=7, description="Number of days for summary")
):
    """Get audit trail summary statistics"""
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = AuditTrailQuery(
        tenant_id=tenant_id,
        start_date=start_date,
        limit=1000
    )
    
    trail = await audit_trail_viewer.get_audit_trail(query)
    
    # Calculate summary statistics
    summary = {
        "total_events": trail.total_count,
        "time_period_days": days,
        "events_by_type": {},
        "events_by_severity": {},
        "recent_critical_events": [],
        "top_models": {},
        "top_workflows": {}
    }
    
    # Analyze events
    for event in trail.events:
        # Count by type
        event_type = event.event_type.value
        summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
        
        # Count by severity
        severity = event.severity.value
        summary["events_by_severity"][severity] = summary["events_by_severity"].get(severity, 0) + 1
        
        # Track critical events
        if event.severity == EventSeverity.CRITICAL:
            summary["recent_critical_events"].append({
                "event_id": event.event_id,
                "title": event.title,
                "timestamp": event.timestamp.isoformat(),
                "model_id": event.model_id
            })
        
        # Count by model
        if event.model_id:
            summary["top_models"][event.model_id] = summary["top_models"].get(event.model_id, 0) + 1
        
        # Count by workflow
        if event.workflow_id:
            summary["top_workflows"][event.workflow_id] = summary["top_workflows"].get(event.workflow_id, 0) + 1
    
    # Sort and limit top items
    summary["top_models"] = dict(sorted(summary["top_models"].items(), key=lambda x: x[1], reverse=True)[:10])
    summary["top_workflows"] = dict(sorted(summary["top_workflows"].items(), key=lambda x: x[1], reverse=True)[:10])
    summary["recent_critical_events"] = summary["recent_critical_events"][:10]
    
    return summary

@app.get("/audit-trail/export/{tenant_id}")
async def export_audit_trail(
    tenant_id: str,
    format: str = Query(default="json", description="Export format: json, csv"),
    days: int = Query(default=30, description="Number of days to export")
):
    """Export audit trail data"""
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = AuditTrailQuery(
        tenant_id=tenant_id,
        start_date=start_date,
        limit=10000
    )
    
    trail = await audit_trail_viewer.get_audit_trail(query)
    
    if format.lower() == "csv":
        # Convert to CSV format
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Event ID", "Type", "Timestamp", "Severity", "Title", 
            "Description", "Model ID", "Workflow ID", "User ID", "Justification"
        ])
        
        # Write data
        for event in trail.events:
            writer.writerow([
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.severity.value,
                event.title,
                event.description,
                event.model_id or "",
                event.workflow_id or "",
                event.user_id or "",
                event.justification or ""
            ])
        
        return {"format": "csv", "data": output.getvalue(), "record_count": len(trail.events)}
    
    else:
        # Return JSON format
        return {
            "format": "json",
            "data": [event.dict() for event in trail.events],
            "record_count": len(trail.events),
            "export_metadata": {
                "tenant_id": tenant_id,
                "export_date": datetime.utcnow().isoformat(),
                "days_included": days
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
