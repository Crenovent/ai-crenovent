"""
Observability Metrics Exporter Service
Exports metrics for mode adoption, override rates, trust drift, and system performance
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime, timedelta
import asyncio
import json

app = FastAPI(
    title="Observability Metrics Exporter Service",
    description="Exports metrics for mode adoption, override rates, trust drift, and system performance",
    version="1.0.0"
)

class MetricType(str, Enum):
    MODE_ADOPTION = "mode_adoption"
    OVERRIDE_RATE = "override_rate"
    TRUST_DRIFT = "trust_drift"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CONFIDENCE_SCORE = "confidence_score"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_HEALTH = "system_health"

class UIMode(str, Enum):
    UI = "ui"
    HYBRID = "hybrid"
    AGENT = "agent"

class MetricValue(BaseModel):
    metric_id: str
    tenant_id: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MetricAggregation(BaseModel):
    tenant_id: str
    metric_type: MetricType
    period: str  # 1m, 5m, 15m, 1h, 1d
    start_time: datetime
    end_time: datetime
    count: int
    sum: float
    avg: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float
    labels: Dict[str, str] = Field(default_factory=dict)

class ExportRequest(BaseModel):
    tenant_id: str
    metric_types: Optional[List[MetricType]] = None
    start_time: datetime
    end_time: datetime
    period: str = "1h"
    format: str = "prometheus"  # prometheus, json, csv
    labels: Optional[Dict[str, str]] = None

class DashboardMetric(BaseModel):
    metric_type: MetricType
    current_value: float
    trend: str  # up, down, stable
    status: str  # healthy, warning, critical
    threshold_warning: float
    threshold_critical: float
    last_updated: datetime

class ModeAdoptionEvent(BaseModel):
    tenant_id: str
    user_id: str
    session_id: str
    ui_mode: UIMode
    service_name: str
    operation_type: str
    confidence_score: float
    trust_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class OverrideEvent(BaseModel):
    tenant_id: str
    user_id: str
    session_id: str
    service_name: str
    operation_type: str
    original_action: Dict[str, Any]
    override_action: Dict[str, Any]
    override_reason: str
    confidence_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TrustDriftEvent(BaseModel):
    tenant_id: str
    user_id: str
    agent_id: str
    previous_trust_score: float
    current_trust_score: float
    drift_amount: float
    drift_reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage (replace with time-series database in production)
metrics_db: List[MetricValue] = []
mode_adoption_events_db: List[ModeAdoptionEvent] = []
override_events_db: List[OverrideEvent] = []
trust_drift_events_db: List[TrustDriftEvent] = []

def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from sorted values"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]

@app.post("/metrics/record")
async def record_metric(metric: MetricValue):
    """Record a metric value"""
    metric.metric_id = str(uuid.uuid4())
    metric.timestamp = datetime.utcnow()
    metrics_db.append(metric)
    return {"message": "Metric recorded successfully", "metric_id": metric.metric_id}

@app.post("/metrics/mode-adoption")
async def record_mode_adoption(event: ModeAdoptionEvent):
    """Record mode adoption event"""
    event.timestamp = datetime.utcnow()
    mode_adoption_events_db.append(event)
    
    # Record mode adoption metric
    metric = MetricValue(
        metric_id=str(uuid.uuid4()),
        tenant_id=event.tenant_id,
        metric_type=MetricType.MODE_ADOPTION,
        value=1.0,
        labels={
            "ui_mode": event.ui_mode.value,
            "service_name": event.service_name,
            "operation_type": event.operation_type
        },
        metadata={
            "user_id": event.user_id,
            "session_id": event.session_id,
            "confidence_score": event.confidence_score,
            "trust_score": event.trust_score
        }
    )
    metrics_db.append(metric)
    
    return {"message": "Mode adoption recorded successfully"}

@app.post("/metrics/override")
async def record_override(event: OverrideEvent):
    """Record override event"""
    event.timestamp = datetime.utcnow()
    override_events_db.append(event)
    
    # Record override rate metric
    metric = MetricValue(
        metric_id=str(uuid.uuid4()),
        tenant_id=event.tenant_id,
        metric_type=MetricType.OVERRIDE_RATE,
        value=1.0,
        labels={
            "service_name": event.service_name,
            "operation_type": event.operation_type
        },
        metadata={
            "user_id": event.user_id,
            "session_id": event.session_id,
            "confidence_score": event.confidence_score,
            "override_reason": event.override_reason
        }
    )
    metrics_db.append(metric)
    
    return {"message": "Override event recorded successfully"}

@app.post("/metrics/trust-drift")
async def record_trust_drift(event: TrustDriftEvent):
    """Record trust drift event"""
    event.timestamp = datetime.utcnow()
    trust_drift_events_db.append(event)
    
    # Record trust drift metric
    metric = MetricValue(
        metric_id=str(uuid.uuid4()),
        tenant_id=event.tenant_id,
        metric_type=MetricType.TRUST_DRIFT,
        value=event.drift_amount,
        labels={
            "agent_id": event.agent_id,
            "drift_direction": "positive" if event.drift_amount > 0 else "negative"
        },
        metadata={
            "user_id": event.user_id,
            "previous_trust_score": event.previous_trust_score,
            "current_trust_score": event.current_trust_score,
            "drift_reason": event.drift_reason
        }
    )
    metrics_db.append(metric)
    
    return {"message": "Trust drift recorded successfully"}

@app.get("/metrics/aggregations", response_model=List[MetricAggregation])
async def get_metric_aggregations(
    tenant_id: str,
    metric_type: Optional[MetricType] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    period: str = "1h",
    labels: Optional[Dict[str, str]] = None
):
    """Get metric aggregations"""
    
    # Filter metrics
    filtered_metrics = [m for m in metrics_db if m.tenant_id == tenant_id]
    
    if metric_type:
        filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]
    
    if start_time:
        filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
    
    if end_time:
        filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
    
    if labels:
        filtered_metrics = [m for m in filtered_metrics 
                           if all(m.labels.get(k) == v for k, v in labels.items())]
    
    # Group by metric type, period, and labels
    aggregations = {}
    
    for metric in filtered_metrics:
        # Create period key based on period type
        if period == "1m":
            period_key = metric.timestamp.replace(second=0, microsecond=0)
        elif period == "5m":
            minute = (metric.timestamp.minute // 5) * 5
            period_key = metric.timestamp.replace(minute=minute, second=0, microsecond=0)
        elif period == "15m":
            minute = (metric.timestamp.minute // 15) * 15
            period_key = metric.timestamp.replace(minute=minute, second=0, microsecond=0)
        elif period == "1h":
            period_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
        elif period == "1d":
            period_key = metric.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            period_key = metric.timestamp
        
        # Create aggregation key
        labels_key = tuple(sorted(metric.labels.items()))
        key = (metric.metric_type, period_key, labels_key)
        
        if key not in aggregations:
            aggregations[key] = []
        aggregations[key].append(metric.value)
    
    # Calculate aggregations
    result = []
    for (metric_type, period_start, labels_tuple), values in aggregations.items():
        period_end = period_start + timedelta(hours=1)  # Default to hourly
        
        if period == "1m":
            period_end = period_start + timedelta(minutes=1)
        elif period == "5m":
            period_end = period_start + timedelta(minutes=5)
        elif period == "15m":
            period_end = period_start + timedelta(minutes=15)
        elif period == "1d":
            period_end = period_start + timedelta(days=1)
        
        aggregation = MetricAggregation(
            tenant_id=tenant_id,
            metric_type=metric_type,
            period=period,
            start_time=period_start,
            end_time=period_end,
            count=len(values),
            sum=sum(values),
            avg=sum(values) / len(values) if values else 0,
            min=min(values) if values else 0,
            max=max(values) if values else 0,
            p50=calculate_percentile(values, 50),
            p95=calculate_percentile(values, 95),
            p99=calculate_percentile(values, 99),
            labels=dict(labels_tuple)
        )
        result.append(aggregation)
    
    return result

@app.post("/metrics/export")
async def export_metrics(request: ExportRequest):
    """Export metrics in specified format"""
    
    # Get aggregations
    aggregations = await get_metric_aggregations(
        tenant_id=request.tenant_id,
        metric_type=request.metric_types[0] if request.metric_types else None,
        start_time=request.start_time,
        end_time=request.end_time,
        period=request.period,
        labels=request.labels
    )
    
    if request.format == "prometheus":
        # Generate Prometheus format
        prometheus_data = ""
        for agg in aggregations:
            metric_name = f"revai_{agg.metric_type.value}"
            labels_str = ",".join([f'{k}="{v}"' for k, v in agg.labels.items()])
            if labels_str:
                labels_str = "{" + labels_str + "}"
            
            prometheus_data += f"{metric_name}{labels_str} {agg.avg} {int(agg.start_time.timestamp())}\n"
        
        return {
            "export_id": str(uuid.uuid4()),
            "format": "prometheus",
            "data": prometheus_data,
            "exported_at": datetime.utcnow().isoformat()
        }
    
    elif request.format == "json":
        return {
            "export_id": str(uuid.uuid4()),
            "format": "json",
            "data": [agg.dict() for agg in aggregations],
            "exported_at": datetime.utcnow().isoformat()
        }
    
    elif request.format == "csv":
        # Generate CSV format
        csv_data = "tenant_id,metric_type,period,start_time,end_time,count,sum,avg,min,max,p50,p95,p99,labels\n"
        for agg in aggregations:
            labels_str = json.dumps(agg.labels)
            csv_data += f"{agg.tenant_id},{agg.metric_type},{agg.period},{agg.start_time},{agg.end_time},{agg.count},{agg.sum},{agg.avg},{agg.min},{agg.max},{agg.p50},{agg.p95},{agg.p99},{labels_str}\n"
        
        return {
            "export_id": str(uuid.uuid4()),
            "format": "csv",
            "data": csv_data,
            "exported_at": datetime.utcnow().isoformat()
        }
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

@app.get("/metrics/dashboard")
async def get_dashboard_metrics(tenant_id: str):
    """Get dashboard metrics for the last 24 hours"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)
    
    # Get recent metrics
    recent_metrics = [m for m in metrics_db 
                     if m.tenant_id == tenant_id and start_time <= m.timestamp <= end_time]
    
    # Calculate dashboard metrics
    dashboard = {}
    
    # Mode adoption metrics
    mode_metrics = [m for m in recent_metrics if m.metric_type == MetricType.MODE_ADOPTION]
    if mode_metrics:
        mode_counts = {}
        for metric in mode_metrics:
            mode = metric.labels.get("ui_mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        total_modes = sum(mode_counts.values())
        dashboard["mode_adoption"] = {
            "ui_percentage": (mode_counts.get("ui", 0) / total_modes * 100) if total_modes > 0 else 0,
            "hybrid_percentage": (mode_counts.get("hybrid", 0) / total_modes * 100) if total_modes > 0 else 0,
            "agent_percentage": (mode_counts.get("agent", 0) / total_modes * 100) if total_modes > 0 else 0,
            "total_sessions": total_modes
        }
    
    # Override rate metrics
    override_metrics = [m for m in recent_metrics if m.metric_type == MetricType.OVERRIDE_RATE]
    if override_metrics:
        total_overrides = len(override_metrics)
        dashboard["override_rate"] = {
            "total_overrides": total_overrides,
            "override_percentage": (total_overrides / len(recent_metrics) * 100) if recent_metrics else 0,
            "avg_confidence_at_override": sum(m.metadata.get("confidence_score", 0) for m in override_metrics) / total_overrides if total_overrides > 0 else 0
        }
    
    # Trust drift metrics
    trust_drift_metrics = [m for m in recent_metrics if m.metric_type == MetricType.TRUST_DRIFT]
    if trust_drift_metrics:
        avg_drift = sum(m.value for m in trust_drift_metrics) / len(trust_drift_metrics)
        positive_drift = len([m for m in trust_drift_metrics if m.value > 0])
        negative_drift = len([m for m in trust_drift_metrics if m.value < 0])
        
        dashboard["trust_drift"] = {
            "avg_drift": avg_drift,
            "positive_drift_count": positive_drift,
            "negative_drift_count": negative_drift,
            "drift_trend": "positive" if avg_drift > 0 else "negative" if avg_drift < 0 else "stable"
        }
    
    # Response time metrics
    response_time_metrics = [m for m in recent_metrics if m.metric_type == MetricType.RESPONSE_TIME]
    if response_time_metrics:
        values = [m.value for m in response_time_metrics]
        dashboard["response_time"] = {
            "avg_ms": sum(values) / len(values),
            "p95_ms": calculate_percentile(values, 95),
            "p99_ms": calculate_percentile(values, 99),
            "max_ms": max(values)
        }
    
    # Error rate metrics
    error_metrics = [m for m in recent_metrics if m.metric_type == MetricType.ERROR_RATE]
    if error_metrics:
        total_errors = sum(m.value for m in error_metrics)
        dashboard["error_rate"] = {
            "total_errors": total_errors,
            "error_percentage": (total_errors / len(recent_metrics) * 100) if recent_metrics else 0
        }
    
    return {
        "tenant_id": tenant_id,
        "period": "24h",
        "metrics": dashboard,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/metrics/alerts")
async def get_metric_alerts(tenant_id: str):
    """Get metric alerts based on thresholds"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=15)  # Last 15 minutes
    
    # Get recent metrics
    recent_metrics = [m for m in metrics_db 
                     if m.tenant_id == tenant_id and start_time <= m.timestamp <= end_time]
    
    alerts = []
    
    # Check override rate threshold
    override_metrics = [m for m in recent_metrics if m.metric_type == MetricType.OVERRIDE_RATE]
    if override_metrics:
        override_rate = len(override_metrics) / len(recent_metrics) if recent_metrics else 0
        if override_rate > 0.3:  # 30% threshold
            alerts.append({
                "type": "override_rate_high",
                "severity": "warning",
                "message": f"Override rate is {override_rate:.1%}, above 30% threshold",
                "value": override_rate,
                "threshold": 0.3
            })
    
    # Check trust drift threshold
    trust_drift_metrics = [m for m in recent_metrics if m.metric_type == MetricType.TRUST_DRIFT]
    if trust_drift_metrics:
        avg_drift = sum(m.value for m in trust_drift_metrics) / len(trust_drift_metrics)
        if abs(avg_drift) > 0.1:  # 10% threshold
            alerts.append({
                "type": "trust_drift_high",
                "severity": "warning",
                "message": f"Trust drift is {avg_drift:.1%}, above 10% threshold",
                "value": avg_drift,
                "threshold": 0.1
            })
    
    # Check response time threshold
    response_time_metrics = [m for m in recent_metrics if m.metric_type == MetricType.RESPONSE_TIME]
    if response_time_metrics:
        p95_response_time = calculate_percentile([m.value for m in response_time_metrics], 95)
        if p95_response_time > 2000:  # 2 seconds threshold
            alerts.append({
                "type": "response_time_high",
                "severity": "critical",
                "message": f"P95 response time is {p95_response_time:.0f}ms, above 2000ms threshold",
                "value": p95_response_time,
                "threshold": 2000
            })
    
    return {
        "tenant_id": tenant_id,
        "alerts": alerts,
        "alert_count": len(alerts),
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "metrics-exporter",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics_count": len(metrics_db),
        "mode_adoption_events_count": len(mode_adoption_events_db),
        "override_events_count": len(override_events_db),
        "trust_drift_events_count": len(trust_drift_events_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
