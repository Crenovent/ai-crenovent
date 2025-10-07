"""
Experience KPIs Exporter Service
Tracks and exports key performance indicators for AI automation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime, timedelta
import asyncio

app = FastAPI(
    title="Experience KPIs Exporter Service",
    description="Tracks and exports key performance indicators for AI automation",
    version="1.0.0"
)

class KPIType(str, Enum):
    PROMPT_CONVERSION = "prompt_conversion"
    OVERRIDE_RATE = "override_rate"
    TRUST_DRIFT = "trust_drift"
    CONFIDENCE_SCORE = "confidence_score"
    AGENT_ADOPTION = "agent_adoption"
    USER_SATISFACTION = "user_satisfaction"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"

class KPIMetric(BaseModel):
    metric_id: str
    tenant_id: str
    user_id: Optional[str] = None
    kpi_type: KPIType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KPIAggregation(BaseModel):
    tenant_id: str
    kpi_type: KPIType
    period: str  # hourly, daily, weekly, monthly
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

class KPIExportRequest(BaseModel):
    tenant_id: str
    kpi_types: Optional[List[KPIType]] = None
    start_time: datetime
    end_time: datetime
    period: str = "daily"
    format: str = "json"  # json, csv, prometheus

class PromptConversionEvent(BaseModel):
    tenant_id: str
    user_id: str
    session_id: str
    prompt_type: str
    input_length: int
    output_length: int
    processing_time_ms: int
    success: bool
    confidence_score: float
    timestamp: datetime

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
    trust_score: float
    timestamp: datetime

class TrustDriftEvent(BaseModel):
    tenant_id: str
    user_id: str
    agent_id: str
    previous_trust_score: float
    current_trust_score: float
    drift_amount: float
    drift_reason: str
    timestamp: datetime

# In-memory storage (replace with time-series database in production)
kpi_metrics_db: List[KPIMetric] = []
prompt_events_db: List[PromptConversionEvent] = []
override_events_db: List[OverrideEvent] = []
trust_drift_db: List[TrustDriftEvent] = []

def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from sorted values"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]

@app.post("/kpis/metrics")
async def record_kpi_metric(metric: KPIMetric):
    """Record a KPI metric"""
    metric.metric_id = str(uuid.uuid4())
    metric.timestamp = datetime.utcnow()
    kpi_metrics_db.append(metric)
    return {"message": "KPI metric recorded successfully", "metric_id": metric.metric_id}

@app.post("/kpis/prompt-conversion")
async def record_prompt_conversion(event: PromptConversionEvent):
    """Record a prompt conversion event"""
    event.timestamp = datetime.utcnow()
    prompt_events_db.append(event)
    
    # Calculate conversion rate
    conversion_rate = event.output_length / event.input_length if event.input_length > 0 else 0
    
    # Record as KPI metric
    kpi_metric = KPIMetric(
        metric_id=str(uuid.uuid4()),
        tenant_id=event.tenant_id,
        user_id=event.user_id,
        kpi_type=KPIType.PROMPT_CONVERSION,
        value=conversion_rate,
        timestamp=event.timestamp,
        context={
            "prompt_type": event.prompt_type,
            "processing_time_ms": event.processing_time_ms,
            "success": event.success
        }
    )
    kpi_metrics_db.append(kpi_metric)
    
    return {"message": "Prompt conversion recorded successfully"}

@app.post("/kpis/override")
async def record_override(event: OverrideEvent):
    """Record an override event"""
    event.timestamp = datetime.utcnow()
    override_events_db.append(event)
    
    # Record override rate (1.0 for this event)
    kpi_metric = KPIMetric(
        metric_id=str(uuid.uuid4()),
        tenant_id=event.tenant_id,
        user_id=event.user_id,
        kpi_type=KPIType.OVERRIDE_RATE,
        value=1.0,
        timestamp=event.timestamp,
        context={
            "service_name": event.service_name,
            "operation_type": event.operation_type,
            "confidence_score": event.confidence_score,
            "trust_score": event.trust_score
        }
    )
    kpi_metrics_db.append(kpi_metric)
    
    return {"message": "Override event recorded successfully"}

@app.post("/kpis/trust-drift")
async def record_trust_drift(event: TrustDriftEvent):
    """Record a trust drift event"""
    event.timestamp = datetime.utcnow()
    trust_drift_db.append(event)
    
    # Record trust drift metric
    kpi_metric = KPIMetric(
        metric_id=str(uuid.uuid4()),
        tenant_id=event.tenant_id,
        user_id=event.user_id,
        kpi_type=KPIType.TRUST_DRIFT,
        value=event.drift_amount,
        timestamp=event.timestamp,
        context={
            "agent_id": event.agent_id,
            "previous_trust_score": event.previous_trust_score,
            "current_trust_score": event.current_trust_score,
            "drift_reason": event.drift_reason
        }
    )
    kpi_metrics_db.append(kpi_metric)
    
    return {"message": "Trust drift recorded successfully"}

@app.get("/kpis/aggregations", response_model=List[KPIAggregation])
async def get_kpi_aggregations(
    tenant_id: str,
    kpi_type: Optional[KPIType] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    period: str = "daily"
):
    """Get KPI aggregations"""
    
    # Filter metrics
    filtered_metrics = [m for m in kpi_metrics_db if m.tenant_id == tenant_id]
    
    if kpi_type:
        filtered_metrics = [m for m in filtered_metrics if m.kpi_type == kpi_type]
    
    if start_time:
        filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
    
    if end_time:
        filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
    
    # Group by KPI type and period
    aggregations = {}
    
    for metric in filtered_metrics:
        # Create period key based on period type
        if period == "hourly":
            period_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
        elif period == "daily":
            period_key = metric.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            # Start of week (Monday)
            days_since_monday = metric.timestamp.weekday()
            period_key = metric.timestamp.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        elif period == "monthly":
            period_key = metric.timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            period_key = metric.timestamp
        
        key = (metric.kpi_type, period_key)
        
        if key not in aggregations:
            aggregations[key] = []
        aggregations[key].append(metric.value)
    
    # Calculate aggregations
    result = []
    for (kpi_type, period_start), values in aggregations.items():
        period_end = period_start + timedelta(days=1)  # Default to daily
        
        if period == "hourly":
            period_end = period_start + timedelta(hours=1)
        elif period == "weekly":
            period_end = period_start + timedelta(weeks=1)
        elif period == "monthly":
            # Add one month
            if period_start.month == 12:
                period_end = period_start.replace(year=period_start.year + 1, month=1)
            else:
                period_end = period_start.replace(month=period_start.month + 1)
        
        aggregation = KPIAggregation(
            tenant_id=tenant_id,
            kpi_type=kpi_type,
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
            p99=calculate_percentile(values, 99)
        )
        result.append(aggregation)
    
    return result

@app.post("/kpis/export")
async def export_kpis(request: KPIExportRequest, background_tasks: BackgroundTasks):
    """Export KPIs in specified format"""
    
    # Get aggregations
    aggregations = await get_kpi_aggregations(
        tenant_id=request.tenant_id,
        kpi_type=request.kpi_types[0] if request.kpi_types else None,
        start_time=request.start_time,
        end_time=request.end_time,
        period=request.period
    )
    
    if request.format == "json":
        return {
            "export_id": str(uuid.uuid4()),
            "format": "json",
            "data": [agg.dict() for agg in aggregations],
            "exported_at": datetime.utcnow().isoformat()
        }
    elif request.format == "csv":
        # Generate CSV format
        csv_data = "tenant_id,kpi_type,period,start_time,end_time,count,sum,avg,min,max,p50,p95,p99\n"
        for agg in aggregations:
            csv_data += f"{agg.tenant_id},{agg.kpi_type},{agg.period},{agg.start_time},{agg.end_time},{agg.count},{agg.sum},{agg.avg},{agg.min},{agg.max},{agg.p50},{agg.p95},{agg.p99}\n"
        
        return {
            "export_id": str(uuid.uuid4()),
            "format": "csv",
            "data": csv_data,
            "exported_at": datetime.utcnow().isoformat()
        }
    elif request.format == "prometheus":
        # Generate Prometheus format
        prometheus_data = ""
        for agg in aggregations:
            metric_name = f"revai_kpi_{agg.kpi_type.value}"
            prometheus_data += f"{metric_name}{{tenant_id=\"{agg.tenant_id}\",period=\"{agg.period}\"}} {agg.avg} {int(agg.start_time.timestamp())}\n"
        
        return {
            "export_id": str(uuid.uuid4()),
            "format": "prometheus",
            "data": prometheus_data,
            "exported_at": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

@app.get("/kpis/dashboard")
async def get_dashboard_metrics(tenant_id: str):
    """Get dashboard metrics for the last 24 hours"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)
    
    # Get recent metrics
    recent_metrics = [m for m in kpi_metrics_db 
                     if m.tenant_id == tenant_id and start_time <= m.timestamp <= end_time]
    
    # Calculate key metrics
    metrics_by_type = {}
    for metric in recent_metrics:
        if metric.kpi_type not in metrics_by_type:
            metrics_by_type[metric.kpi_type] = []
        metrics_by_type[metric.kpi_type].append(metric.value)
    
    dashboard = {}
    for kpi_type, values in metrics_by_type.items():
        dashboard[kpi_type.value] = {
            "count": len(values),
            "avg": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "trend": "up" if len(values) > 1 and values[-1] > values[0] else "down"
        }
    
    return {
        "tenant_id": tenant_id,
        "period": "24h",
        "metrics": dashboard,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "kpi-exporter",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics_count": len(kpi_metrics_db),
        "prompt_events_count": len(prompt_events_db),
        "override_events_count": len(override_events_db),
        "trust_drift_count": len(trust_drift_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
