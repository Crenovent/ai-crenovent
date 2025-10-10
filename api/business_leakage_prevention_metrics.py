"""
Task 4.4.23: Add business leakage prevention metrics (revenue saved via fraud/anomaly detection)
- Prove financial impact
- Fraud ML + BI dashboards
- CFO value driver
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Business Leakage Prevention Metrics")
logger = logging.getLogger(__name__)

class LeakageType(str, Enum):
    FRAUD_PREVENTION = "fraud_prevention"
    REVENUE_LEAKAGE = "revenue_leakage"
    CHURN_PREVENTION = "churn_prevention"
    PRICING_OPTIMIZATION = "pricing_optimization"
    COMPLIANCE_VIOLATION = "compliance_violation"

class PreventionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Event details
    leakage_type: LeakageType
    amount_saved: float
    currency: str = "USD"
    
    # Detection details
    detection_method: str  # ml_model, rule_based, hybrid
    confidence_score: float = 0.0
    
    # Context
    workflow_id: Optional[str] = None
    customer_id: Optional[str] = None
    transaction_id: Optional[str] = None
    
    # Timing
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    prevented_at: Optional[datetime] = None
    
    # Additional details
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LeakagePreventionMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Financial impact
    total_amount_saved: float = 0.0
    fraud_prevention_savings: float = 0.0
    revenue_leakage_savings: float = 0.0
    churn_prevention_savings: float = 0.0
    pricing_optimization_savings: float = 0.0
    compliance_violation_savings: float = 0.0
    
    # Detection metrics
    total_prevention_events: int = 0
    ml_detected_events: int = 0
    rule_based_events: int = 0
    avg_confidence_score: float = 0.0
    
    # ROI metrics
    detection_cost: float = 0.0  # Cost of running detection systems
    net_savings: float = 0.0     # Total saved - detection cost
    roi_percentage: float = 0.0  # (Net savings / detection cost) * 100
    
    # Time metrics
    avg_detection_time_hours: float = 0.0
    prevention_rate: float = 0.0  # % of detected issues that were prevented
    
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    measurement_period: str = "monthly"
    currency: str = "USD"

class LeakagePreventionReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Report period
    start_date: datetime
    end_date: datetime
    
    # Summary metrics
    total_savings: float
    prevention_events: int
    roi_percentage: float
    
    # Breakdown by type
    savings_by_type: Dict[str, float] = Field(default_factory=dict)
    events_by_type: Dict[str, int] = Field(default_factory=dict)
    
    # Top prevention events
    top_savings_events: List[PreventionEvent] = Field(default_factory=list)
    
    # Trends
    monthly_savings_trend: List[Dict[str, Any]] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
prevention_events_store: Dict[str, PreventionEvent] = {}
leakage_metrics_store: Dict[str, LeakagePreventionMetric] = {}
prevention_reports_store: Dict[str, LeakagePreventionReport] = {}

class LeakagePreventionCalculator:
    def __init__(self):
        self.detection_cost_per_event = 5.0  # Average cost per detection
        
    def calculate_prevention_metrics(self, tenant_id: str, days: int = 30) -> LeakagePreventionMetric:
        """Calculate leakage prevention metrics for a tenant"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get prevention events for tenant
        tenant_events = [
            event for event in prevention_events_store.values()
            if (event.tenant_id == tenant_id and 
                event.detected_at >= cutoff_date)
        ]
        
        if not tenant_events:
            return LeakagePreventionMetric(tenant_id=tenant_id)
        
        # Calculate financial impact by type
        fraud_savings = sum([e.amount_saved for e in tenant_events if e.leakage_type == LeakageType.FRAUD_PREVENTION])
        revenue_savings = sum([e.amount_saved for e in tenant_events if e.leakage_type == LeakageType.REVENUE_LEAKAGE])
        churn_savings = sum([e.amount_saved for e in tenant_events if e.leakage_type == LeakageType.CHURN_PREVENTION])
        pricing_savings = sum([e.amount_saved for e in tenant_events if e.leakage_type == LeakageType.PRICING_OPTIMIZATION])
        compliance_savings = sum([e.amount_saved for e in tenant_events if e.leakage_type == LeakageType.COMPLIANCE_VIOLATION])
        
        total_savings = fraud_savings + revenue_savings + churn_savings + pricing_savings + compliance_savings
        
        # Calculate detection metrics
        total_events = len(tenant_events)
        ml_events = len([e for e in tenant_events if "ml" in e.detection_method.lower()])
        rule_events = len([e for e in tenant_events if "rule" in e.detection_method.lower()])
        
        # Calculate average confidence
        avg_confidence = sum([e.confidence_score for e in tenant_events]) / len(tenant_events) if tenant_events else 0.0
        
        # Calculate ROI metrics
        detection_cost = total_events * self.detection_cost_per_event
        net_savings = total_savings - detection_cost
        roi_percentage = (net_savings / detection_cost * 100) if detection_cost > 0 else 0.0
        
        # Calculate time metrics
        time_to_prevention = []
        for event in tenant_events:
            if event.prevented_at:
                time_diff = (event.prevented_at - event.detected_at).total_seconds() / 3600  # hours
                time_to_prevention.append(time_diff)
        
        avg_detection_time = sum(time_to_prevention) / len(time_to_prevention) if time_to_prevention else 0.0
        prevention_rate = (len(time_to_prevention) / total_events * 100) if total_events > 0 else 0.0
        
        return LeakagePreventionMetric(
            tenant_id=tenant_id,
            total_amount_saved=total_savings,
            fraud_prevention_savings=fraud_savings,
            revenue_leakage_savings=revenue_savings,
            churn_prevention_savings=churn_savings,
            pricing_optimization_savings=pricing_savings,
            compliance_violation_savings=compliance_savings,
            total_prevention_events=total_events,
            ml_detected_events=ml_events,
            rule_based_events=rule_events,
            avg_confidence_score=avg_confidence,
            detection_cost=detection_cost,
            net_savings=net_savings,
            roi_percentage=roi_percentage,
            avg_detection_time_hours=avg_detection_time,
            prevention_rate=prevention_rate
        )

# Global calculator instance
calculator = LeakagePreventionCalculator()

@app.post("/leakage-prevention/events/record", response_model=PreventionEvent)
async def record_prevention_event(event: PreventionEvent):
    """Record a business leakage prevention event"""
    
    prevention_events_store[event.event_id] = event
    
    logger.info(f"✅ Recorded prevention event - Type: {event.leakage_type}, Amount Saved: ${event.amount_saved:,.2f}")
    return event

@app.post("/leakage-prevention/metrics/calculate", response_model=LeakagePreventionMetric)
async def calculate_leakage_metrics(tenant_id: str, days: int = 30):
    """Calculate leakage prevention metrics for a tenant"""
    
    metric = calculator.calculate_prevention_metrics(tenant_id, days)
    
    # Store metric
    leakage_metrics_store[metric.metric_id] = metric
    
    logger.info(f"✅ Calculated leakage prevention metrics for tenant {tenant_id} - Total Saved: ${metric.total_amount_saved:,.2f}")
    return metric

@app.get("/leakage-prevention/metrics/tenant/{tenant_id}")
async def get_tenant_leakage_metrics(tenant_id: str):
    """Get leakage prevention metrics for a tenant"""
    
    tenant_metrics = [
        metric for metric in leakage_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    if not tenant_metrics:
        raise HTTPException(status_code=404, detail="No leakage prevention metrics found for tenant")
    
    # Return most recent metric
    latest_metric = max(tenant_metrics, key=lambda x: x.measurement_date)
    
    return latest_metric

@app.get("/leakage-prevention/events/tenant/{tenant_id}")
async def get_tenant_prevention_events(
    tenant_id: str,
    leakage_type: Optional[LeakageType] = None,
    days: int = 30
):
    """Get prevention events for a tenant"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    tenant_events = [
        event for event in prevention_events_store.values()
        if (event.tenant_id == tenant_id and 
            event.detected_at >= cutoff_date)
    ]
    
    if leakage_type:
        tenant_events = [e for e in tenant_events if e.leakage_type == leakage_type]
    
    # Sort by amount saved (descending)
    tenant_events.sort(key=lambda x: x.amount_saved, reverse=True)
    
    return {
        "tenant_id": tenant_id,
        "event_count": len(tenant_events),
        "total_amount_saved": sum([e.amount_saved for e in tenant_events]),
        "events": tenant_events
    }

@app.post("/leakage-prevention/reports/generate", response_model=LeakagePreventionReport)
async def generate_prevention_report(
    tenant_id: str,
    start_date: datetime,
    end_date: datetime
):
    """Generate leakage prevention report"""
    
    # Get events for the period
    period_events = [
        event for event in prevention_events_store.values()
        if (event.tenant_id == tenant_id and 
            start_date <= event.detected_at <= end_date)
    ]
    
    # Calculate summary metrics
    total_savings = sum([e.amount_saved for e in period_events])
    prevention_events_count = len(period_events)
    
    # Calculate ROI
    detection_cost = prevention_events_count * calculator.detection_cost_per_event
    net_savings = total_savings - detection_cost
    roi_percentage = (net_savings / detection_cost * 100) if detection_cost > 0 else 0.0
    
    # Breakdown by type
    savings_by_type = {}
    events_by_type = {}
    
    for leakage_type in LeakageType:
        type_events = [e for e in period_events if e.leakage_type == leakage_type]
        savings_by_type[leakage_type.value] = sum([e.amount_saved for e in type_events])
        events_by_type[leakage_type.value] = len(type_events)
    
    # Top savings events
    top_events = sorted(period_events, key=lambda x: x.amount_saved, reverse=True)[:10]
    
    # Monthly trend (simplified)
    monthly_trend = []
    current_date = start_date
    while current_date <= end_date:
        month_end = min(current_date + timedelta(days=30), end_date)
        month_events = [
            e for e in period_events 
            if current_date <= e.detected_at <= month_end
        ]
        monthly_trend.append({
            "month": current_date.strftime("%Y-%m"),
            "savings": sum([e.amount_saved for e in month_events]),
            "events": len(month_events)
        })
        current_date = month_end + timedelta(days=1)
    
    report = LeakagePreventionReport(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
        total_savings=total_savings,
        prevention_events=prevention_events_count,
        roi_percentage=roi_percentage,
        savings_by_type=savings_by_type,
        events_by_type=events_by_type,
        top_savings_events=top_events,
        monthly_savings_trend=monthly_trend
    )
    
    prevention_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated leakage prevention report for tenant {tenant_id} - Total Savings: ${total_savings:,.2f}")
    return report

@app.get("/leakage-prevention/summary")
async def get_leakage_prevention_summary():
    """Get overall leakage prevention summary"""
    
    total_events = len(prevention_events_store)
    total_metrics = len(leakage_metrics_store)
    
    if total_metrics == 0:
        return {
            "total_events": total_events,
            "total_tenants": 0,
            "total_savings": 0.0,
            "average_roi": 0.0
        }
    
    # Calculate totals
    all_events = list(prevention_events_store.values())
    total_savings = sum([e.amount_saved for e in all_events])
    
    all_metrics = list(leakage_metrics_store.values())
    avg_roi = sum([m.roi_percentage for m in all_metrics]) / len(all_metrics)
    
    # Breakdown by type
    type_breakdown = {}
    for leakage_type in LeakageType:
        type_events = [e for e in all_events if e.leakage_type == leakage_type]
        type_breakdown[leakage_type.value] = {
            "events": len(type_events),
            "savings": sum([e.amount_saved for e in type_events])
        }
    
    return {
        "total_events": total_events,
        "total_tenants": len(set(m.tenant_id for m in all_metrics)),
        "total_savings": round(total_savings, 2),
        "average_roi": round(avg_roi, 2),
        "breakdown_by_type": type_breakdown,
        "total_reports": len(prevention_reports_store)
    }
