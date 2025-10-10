"""
Task 4.4.21: Provide incident frequency metric (number of auto-fallbacks triggered)
- Risk transparency
- Orchestrator logs analysis
- CRO/Compliance persona
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import asyncpg

app = FastAPI(title="RBIA Incident Frequency Metrics")
logger = logging.getLogger(__name__)

class IncidentType(str, Enum):
    AUTO_FALLBACK = "auto_fallback"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    SECURITY_VIOLATION = "security_violation"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentRecord(BaseModel):
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Incident details
    incident_type: IncidentType
    severity: SeverityLevel
    workflow_id: Optional[str] = None
    capability_id: Optional[str] = None
    
    # Auto-fallback specific
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    fallback_target: Optional[str] = None  # What it fell back to
    
    # Timing
    incident_timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolution_timestamp: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    
    # Impact
    affected_users: int = 0
    failed_executions: int = 0
    
    # Context
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IncidentFrequencyMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Auto-fallback metrics
    auto_fallbacks_triggered: int = 0
    fallback_frequency_per_day: float = 0.0
    fallback_success_rate: float = 0.0
    
    # Incident metrics by type
    system_errors: int = 0
    performance_issues: int = 0
    data_quality_issues: int = 0
    security_violations: int = 0
    
    # Severity breakdown
    critical_incidents: int = 0
    high_severity_incidents: int = 0
    medium_severity_incidents: int = 0
    low_severity_incidents: int = 0
    
    # Time metrics
    avg_resolution_time_minutes: float = 0.0
    mttr: float = 0.0  # Mean Time To Resolution
    mtbf: float = 0.0  # Mean Time Between Failures
    
    # Impact metrics
    total_affected_users: int = 0
    total_failed_executions: int = 0
    
    # Risk score
    incident_risk_score: float = 0.0  # 0-100 scale
    
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    measurement_period: str = "daily"

# In-memory storage
incident_records_store: Dict[str, IncidentRecord] = {}
frequency_metrics_store: Dict[str, IncidentFrequencyMetric] = {}

class IncidentFrequencyTracker:
    def __init__(self):
        self.db_pool = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                database="rbia_metrics",
                user="postgres",
                password="password",
                min_size=5,
                max_size=20
            )
            logger.info("✅ Incident frequency tracker initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            return False
    
    async def collect_orchestrator_incidents(self, tenant_id: str, days: int = 1) -> List[IncidentRecord]:
        """Collect incidents from orchestrator logs"""
        
        incidents = []
        if not self.db_pool:
            return incidents
            
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get failed executions that triggered fallbacks
                rows = await conn.fetch("""
                    SELECT execution_id, tenant_id, workflow_id, status, error_message,
                           fallback_triggered, fallback_reason, created_at, completed_at
                    FROM orchestrator_executions 
                    WHERE tenant_id = $1 AND created_at >= $2 
                    AND (status = 'failed' OR fallback_triggered = true)
                """, tenant_id, cutoff_date)
                
                for row in rows:
                    # Determine incident type
                    incident_type = IncidentType.AUTO_FALLBACK if row['fallback_triggered'] else IncidentType.SYSTEM_ERROR
                    
                    # Determine severity based on error patterns
                    error_msg = row['error_message'] or ""
                    if "timeout" in error_msg.lower() or "performance" in error_msg.lower():
                        severity = SeverityLevel.MEDIUM
                        incident_type = IncidentType.PERFORMANCE_DEGRADATION
                    elif "security" in error_msg.lower() or "unauthorized" in error_msg.lower():
                        severity = SeverityLevel.HIGH
                        incident_type = IncidentType.SECURITY_VIOLATION
                    elif "data" in error_msg.lower() or "validation" in error_msg.lower():
                        severity = SeverityLevel.LOW
                        incident_type = IncidentType.DATA_QUALITY_ISSUE
                    else:
                        severity = SeverityLevel.MEDIUM
                    
                    # Calculate duration
                    duration = None
                    if row['completed_at']:
                        duration = (row['completed_at'] - row['created_at']).total_seconds() / 60
                    
                    incident = IncidentRecord(
                        tenant_id=tenant_id,
                        incident_type=incident_type,
                        severity=severity,
                        workflow_id=row['workflow_id'],
                        fallback_triggered=row['fallback_triggered'] or False,
                        fallback_reason=row['fallback_reason'],
                        incident_timestamp=row['created_at'],
                        resolution_timestamp=row['completed_at'],
                        duration_minutes=duration,
                        failed_executions=1,
                        error_message=row['error_message']
                    )
                    
                    incidents.append(incident)
                    
        except Exception as e:
            logger.error(f"Failed to collect orchestrator incidents: {e}")
            
        return incidents
    
    def calculate_frequency_metrics(self, tenant_id: str, incidents: List[IncidentRecord], days: int = 1) -> IncidentFrequencyMetric:
        """Calculate incident frequency metrics"""
        
        # Count incidents by type
        auto_fallbacks = len([i for i in incidents if i.incident_type == IncidentType.AUTO_FALLBACK])
        system_errors = len([i for i in incidents if i.incident_type == IncidentType.SYSTEM_ERROR])
        performance_issues = len([i for i in incidents if i.incident_type == IncidentType.PERFORMANCE_DEGRADATION])
        data_quality_issues = len([i for i in incidents if i.incident_type == IncidentType.DATA_QUALITY_ISSUE])
        security_violations = len([i for i in incidents if i.incident_type == IncidentType.SECURITY_VIOLATION])
        
        # Count by severity
        critical_incidents = len([i for i in incidents if i.severity == SeverityLevel.CRITICAL])
        high_severity = len([i for i in incidents if i.severity == SeverityLevel.HIGH])
        medium_severity = len([i for i in incidents if i.severity == SeverityLevel.MEDIUM])
        low_severity = len([i for i in incidents if i.severity == SeverityLevel.LOW])
        
        # Calculate fallback metrics
        fallback_frequency_per_day = auto_fallbacks / days if days > 0 else 0
        
        # Calculate fallback success rate (fallbacks that resolved vs total fallbacks)
        resolved_fallbacks = len([i for i in incidents if i.fallback_triggered and i.resolution_timestamp])
        fallback_success_rate = (resolved_fallbacks / auto_fallbacks * 100) if auto_fallbacks > 0 else 0
        
        # Calculate time metrics
        resolved_incidents = [i for i in incidents if i.duration_minutes is not None]
        avg_resolution_time = sum([i.duration_minutes for i in resolved_incidents]) / len(resolved_incidents) if resolved_incidents else 0
        
        # MTTR (Mean Time To Resolution)
        mttr = avg_resolution_time
        
        # MTBF (Mean Time Between Failures) - simplified calculation
        total_incidents = len(incidents)
        mtbf = (days * 24 * 60) / total_incidents if total_incidents > 0 else 0  # minutes between incidents
        
        # Calculate impact
        total_affected_users = sum([i.affected_users for i in incidents])
        total_failed_executions = sum([i.failed_executions for i in incidents])
        
        # Calculate risk score (0-100)
        risk_score = min(100, (
            (critical_incidents * 25) +
            (high_severity * 15) +
            (medium_severity * 10) +
            (low_severity * 5) +
            (auto_fallbacks * 2)
        ))
        
        return IncidentFrequencyMetric(
            tenant_id=tenant_id,
            auto_fallbacks_triggered=auto_fallbacks,
            fallback_frequency_per_day=fallback_frequency_per_day,
            fallback_success_rate=fallback_success_rate,
            system_errors=system_errors,
            performance_issues=performance_issues,
            data_quality_issues=data_quality_issues,
            security_violations=security_violations,
            critical_incidents=critical_incidents,
            high_severity_incidents=high_severity,
            medium_severity_incidents=medium_severity,
            low_severity_incidents=low_severity,
            avg_resolution_time_minutes=avg_resolution_time,
            mttr=mttr,
            mtbf=mtbf,
            total_affected_users=total_affected_users,
            total_failed_executions=total_failed_executions,
            incident_risk_score=risk_score
        )

# Global tracker instance
tracker = IncidentFrequencyTracker()

@app.on_event("startup")
async def startup_event():
    await tracker.initialize()

@app.post("/incidents/record", response_model=IncidentRecord)
async def record_incident(incident: IncidentRecord):
    """Record an incident"""
    
    incident_records_store[incident.incident_id] = incident
    
    logger.info(f"✅ Recorded incident {incident.incident_id} - Type: {incident.incident_type}, Severity: {incident.severity}")
    return incident

@app.post("/incidents/metrics/calculate", response_model=IncidentFrequencyMetric)
async def calculate_incident_metrics(tenant_id: str, days: int = 1):
    """Calculate incident frequency metrics for a tenant"""
    
    # Collect incidents from orchestrator logs
    orchestrator_incidents = await tracker.collect_orchestrator_incidents(tenant_id, days)
    
    # Get manually recorded incidents
    manual_incidents = [
        incident for incident in incident_records_store.values()
        if (incident.tenant_id == tenant_id and 
            incident.incident_timestamp >= datetime.utcnow() - timedelta(days=days))
    ]
    
    # Combine all incidents
    all_incidents = orchestrator_incidents + manual_incidents
    
    # Store incidents
    for incident in orchestrator_incidents:
        incident_records_store[incident.incident_id] = incident
    
    # Calculate metrics
    metric = tracker.calculate_frequency_metrics(tenant_id, all_incidents, days)
    
    # Store metric
    frequency_metrics_store[metric.metric_id] = metric
    
    logger.info(f"✅ Calculated incident metrics for tenant {tenant_id} - {len(all_incidents)} incidents, Risk Score: {metric.incident_risk_score}")
    return metric

@app.get("/incidents/tenant/{tenant_id}")
async def get_tenant_incidents(
    tenant_id: str,
    incident_type: Optional[IncidentType] = None,
    severity: Optional[SeverityLevel] = None,
    days: int = 7
):
    """Get incident records for a tenant"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    tenant_incidents = [
        incident for incident in incident_records_store.values()
        if (incident.tenant_id == tenant_id and 
            incident.incident_timestamp >= cutoff_date)
    ]
    
    if incident_type:
        tenant_incidents = [i for i in tenant_incidents if i.incident_type == incident_type]
    
    if severity:
        tenant_incidents = [i for i in tenant_incidents if i.severity == severity]
    
    return {
        "tenant_id": tenant_id,
        "incident_count": len(tenant_incidents),
        "incidents": tenant_incidents
    }

@app.get("/incidents/metrics/tenant/{tenant_id}")
async def get_tenant_incident_metrics(tenant_id: str):
    """Get incident frequency metrics for a tenant"""
    
    tenant_metrics = [
        metric for metric in frequency_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    if not tenant_metrics:
        raise HTTPException(status_code=404, detail="No incident metrics found for tenant")
    
    # Return most recent metric
    latest_metric = max(tenant_metrics, key=lambda x: x.measurement_date)
    
    return latest_metric

@app.get("/incidents/summary")
async def get_incident_summary():
    """Get overall incident frequency summary"""
    
    total_incidents = len(incident_records_store)
    total_metrics = len(frequency_metrics_store)
    
    if total_metrics == 0:
        return {
            "total_incidents": total_incidents,
            "total_tenants": 0,
            "average_risk_score": 0.0,
            "total_auto_fallbacks": 0
        }
    
    # Calculate averages
    all_metrics = list(frequency_metrics_store.values())
    avg_risk_score = sum([m.incident_risk_score for m in all_metrics]) / len(all_metrics)
    total_auto_fallbacks = sum([m.auto_fallbacks_triggered for m in all_metrics])
    avg_fallback_frequency = sum([m.fallback_frequency_per_day for m in all_metrics]) / len(all_metrics)
    
    return {
        "total_incidents": total_incidents,
        "total_tenants": len(set(m.tenant_id for m in all_metrics)),
        "average_risk_score": round(avg_risk_score, 2),
        "total_auto_fallbacks": total_auto_fallbacks,
        "average_fallback_frequency_per_day": round(avg_fallback_frequency, 2),
        "total_metrics": total_metrics
    }
