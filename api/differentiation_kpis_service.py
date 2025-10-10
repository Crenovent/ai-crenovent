"""
Task 3.4.18: Establish differentiation KPIs
- Metrics pipeline service for tracking adoption rates, override reductions
- Database schema for KPI storage and historical tracking
- Analytics engine for calculating differentiation metrics
- Dashboard API endpoints for KPI visualization
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import asyncio
import statistics
from dataclasses import dataclass

app = FastAPI(title="RBIA Differentiation KPIs Service")
logger = logging.getLogger(__name__)

class KPICategory(str, Enum):
    ADOPTION = "adoption"
    GOVERNANCE = "governance"
    TRUST = "trust"
    EFFICIENCY = "efficiency"
    COMPLIANCE = "compliance"
    BUSINESS_IMPACT = "business_impact"

class KPIType(str, Enum):
    # Adoption KPIs
    USER_ADOPTION_RATE = "user_adoption_rate"
    FEATURE_ADOPTION_RATE = "feature_adoption_rate"
    TENANT_ONBOARDING_TIME = "tenant_onboarding_time"
    WORKFLOW_CREATION_RATE = "workflow_creation_rate"
    
    # Governance KPIs
    OVERRIDE_REDUCTION_RATE = "override_reduction_rate"
    POLICY_COMPLIANCE_RATE = "policy_compliance_rate"
    EVIDENCE_PACK_GENERATION_RATE = "evidence_pack_generation_rate"
    AUDIT_READINESS_SCORE = "audit_readiness_score"
    
    # Trust KPIs
    TRUST_SCORE_IMPROVEMENT = "trust_score_improvement"
    EXPLAINABILITY_USAGE_RATE = "explainability_usage_rate"
    SHADOW_MODE_SUCCESS_RATE = "shadow_mode_success_rate"
    
    # Efficiency KPIs
    WORKFLOW_EXECUTION_TIME = "workflow_execution_time"
    FALSE_POSITIVE_REDUCTION = "false_positive_reduction"
    MANUAL_INTERVENTION_REDUCTION = "manual_intervention_reduction"
    
    # Compliance KPIs
    REGULATOR_APPROVAL_RATE = "regulator_approval_rate"
    COMPLIANCE_VIOLATION_REDUCTION = "compliance_violation_reduction"
    DATA_RESIDENCY_COMPLIANCE_RATE = "data_residency_compliance_rate"
    
    # Business Impact KPIs
    REVENUE_IMPACT_POSITIVE = "revenue_impact_positive"
    COST_SAVINGS_REALIZED = "cost_savings_realized"
    DEAL_VELOCITY_IMPROVEMENT = "deal_velocity_improvement"
    CUSTOMER_SATISFACTION_SCORE = "customer_satisfaction_score"

class KPITrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

class KPIStatus(str, Enum):
    EXCELLENT = "excellent"  # > 90th percentile
    GOOD = "good"           # 70-90th percentile
    ACCEPTABLE = "acceptable" # 50-70th percentile
    NEEDS_IMPROVEMENT = "needs_improvement" # 25-50th percentile
    CRITICAL = "critical"    # < 25th percentile

class KPIMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: Optional[str] = None  # None for platform-wide metrics
    
    # KPI identification
    kpi_type: KPIType
    kpi_category: KPICategory
    metric_name: str
    metric_description: str
    
    # Metric value
    current_value: float
    previous_value: Optional[float] = None
    target_value: Optional[float] = None
    baseline_value: Optional[float] = None
    
    # Units and context
    unit: str  # percentage, count, minutes, score
    measurement_period: str  # daily, weekly, monthly, quarterly
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Analysis
    trend: KPITrend = KPITrend.STABLE
    status: KPIStatus = KPIStatus.ACCEPTABLE
    variance_from_target: Optional[float] = None
    percentile_rank: Optional[float] = None
    
    # Metadata
    data_sources: List[str] = Field(default_factory=list)
    calculation_method: str
    confidence_level: float = 0.8
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class KPIDashboard(BaseModel):
    dashboard_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dashboard_name: str
    dashboard_type: str  # executive, operational, detailed
    
    # Configuration
    included_kpis: List[KPIType] = Field(default_factory=list)
    included_categories: List[KPICategory] = Field(default_factory=list)
    tenant_filter: Optional[str] = None
    time_range_days: int = 30
    
    # Visualization settings
    chart_types: Dict[str, str] = Field(default_factory=dict)  # kpi_type -> chart_type
    refresh_interval_minutes: int = 15
    
    # Alerts
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    alert_recipients: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_refreshed: Optional[datetime] = None

class KPIAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kpi_type: KPIType
    tenant_id: Optional[str] = None
    
    # Alert details
    alert_type: str  # threshold_breach, trend_change, anomaly
    severity: str  # low, medium, high, critical
    message: str
    
    # Context
    current_value: float
    threshold_value: Optional[float] = None
    previous_value: Optional[float] = None
    
    # Status
    is_active: bool = True
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

class KPIReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str  # weekly, monthly, quarterly, annual
    report_period_start: datetime
    report_period_end: datetime
    
    # Summary metrics
    kpis_tracked: int
    kpis_improving: int
    kpis_declining: int
    overall_differentiation_score: float
    
    # Category performance
    category_scores: Dict[KPICategory, float] = Field(default_factory=dict)
    
    # Key insights
    top_performing_kpis: List[Dict[str, Any]] = Field(default_factory=list)
    underperforming_kpis: List[Dict[str, Any]] = Field(default_factory=list)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Recommendations
    improvement_recommendations: List[str] = Field(default_factory=list)
    strategic_priorities: List[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = "system"

# In-memory storage (replace with database in production)
kpi_metrics_store: Dict[str, KPIMetric] = {}
kpi_dashboards_store: Dict[str, KPIDashboard] = {}
kpi_alerts_store: Dict[str, KPIAlert] = {}
kpi_reports_store: Dict[str, KPIReport] = {}
kpi_historical_data: Dict[str, List[Dict[str, Any]]] = {}

def _initialize_default_kpis():
    """Initialize default KPI tracking"""
    
    # Adoption KPIs
    kpi_metrics_store["user_adoption"] = KPIMetric(
        kpi_type=KPIType.USER_ADOPTION_RATE,
        kpi_category=KPICategory.ADOPTION,
        metric_name="User Adoption Rate",
        metric_description="Percentage of users actively using RBIA features",
        current_value=78.5,
        target_value=85.0,
        baseline_value=65.0,
        unit="percentage",
        measurement_period="monthly",
        calculation_method="(active_users / total_users) * 100",
        trend=KPITrend.IMPROVING,
        status=KPIStatus.GOOD
    )
    
    # Governance KPIs
    kpi_metrics_store["override_reduction"] = KPIMetric(
        kpi_type=KPIType.OVERRIDE_REDUCTION_RATE,
        kpi_category=KPICategory.GOVERNANCE,
        metric_name="Override Reduction Rate",
        metric_description="Reduction in manual overrides due to improved automation",
        current_value=42.3,
        target_value=50.0,
        baseline_value=0.0,
        unit="percentage",
        measurement_period="monthly",
        calculation_method="((baseline_overrides - current_overrides) / baseline_overrides) * 100",
        trend=KPITrend.IMPROVING,
        status=KPIStatus.ACCEPTABLE
    )
    
    # Trust KPIs
    kpi_metrics_store["trust_improvement"] = KPIMetric(
        kpi_type=KPIType.TRUST_SCORE_IMPROVEMENT,
        kpi_category=KPICategory.TRUST,
        metric_name="Trust Score Improvement",
        metric_description="Average improvement in trust scores across all models",
        current_value=15.7,
        target_value=20.0,
        baseline_value=0.0,
        unit="percentage",
        measurement_period="monthly",
        calculation_method="((current_avg_trust - baseline_avg_trust) / baseline_avg_trust) * 100",
        trend=KPITrend.IMPROVING,
        status=KPIStatus.GOOD
    )
    
    # Compliance KPIs
    kpi_metrics_store["regulator_approval"] = KPIMetric(
        kpi_type=KPIType.REGULATOR_APPROVAL_RATE,
        kpi_category=KPICategory.COMPLIANCE,
        metric_name="Regulator Approval Rate",
        metric_description="Percentage of regulatory submissions approved without issues",
        current_value=94.2,
        target_value=95.0,
        baseline_value=85.0,
        unit="percentage",
        measurement_period="quarterly",
        calculation_method="(approved_submissions / total_submissions) * 100",
        trend=KPITrend.STABLE,
        status=KPIStatus.EXCELLENT
    )

@app.on_event("startup")
async def startup_event():
    """Initialize default KPIs on startup"""
    _initialize_default_kpis()
    logger.info("✅ Differentiation KPIs service initialized")

@app.post("/kpis/metrics", response_model=KPIMetric)
async def record_kpi_metric(metric: KPIMetric, background_tasks: BackgroundTasks):
    """Record a new KPI metric value"""
    
    # Store current metric
    kpi_metrics_store[metric.metric_id] = metric
    
    # Add to historical data
    kpi_key = f"{metric.kpi_type.value}_{metric.tenant_id or 'platform'}"
    if kpi_key not in kpi_historical_data:
        kpi_historical_data[kpi_key] = []
    
    kpi_historical_data[kpi_key].append({
        "timestamp": metric.measurement_date.isoformat(),
        "value": metric.current_value,
        "target": metric.target_value,
        "trend": metric.trend.value,
        "status": metric.status.value
    })
    
    # Check for alerts in background
    background_tasks.add_task(_check_kpi_alerts, metric)
    
    logger.info(f"📊 Recorded KPI metric: {metric.metric_name} = {metric.current_value} {metric.unit}")
    return metric

@app.get("/kpis/metrics", response_model=List[KPIMetric])
async def get_kpi_metrics(
    category: Optional[KPICategory] = None,
    kpi_type: Optional[KPIType] = None,
    tenant_id: Optional[str] = None,
    status: Optional[KPIStatus] = None
):
    """Get KPI metrics with optional filtering"""
    
    metrics = list(kpi_metrics_store.values())
    
    if category:
        metrics = [m for m in metrics if m.kpi_category == category]
    if kpi_type:
        metrics = [m for m in metrics if m.kpi_type == kpi_type]
    if tenant_id:
        metrics = [m for m in metrics if m.tenant_id == tenant_id]
    if status:
        metrics = [m for m in metrics if m.status == status]
    
    return metrics

@app.get("/kpis/metrics/{metric_id}/history")
async def get_kpi_history(metric_id: str, days: int = 30):
    """Get historical data for a specific KPI metric"""
    
    if metric_id not in kpi_metrics_store:
        raise HTTPException(status_code=404, detail="KPI metric not found")
    
    metric = kpi_metrics_store[metric_id]
    kpi_key = f"{metric.kpi_type.value}_{metric.tenant_id or 'platform'}"
    
    history = kpi_historical_data.get(kpi_key, [])
    
    # Filter by date range
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    filtered_history = [
        h for h in history 
        if datetime.fromisoformat(h["timestamp"]) >= cutoff_date
    ]
    
    return {
        "metric_id": metric_id,
        "kpi_type": metric.kpi_type.value,
        "metric_name": metric.metric_name,
        "unit": metric.unit,
        "history": filtered_history,
        "data_points": len(filtered_history)
    }

@app.post("/kpis/dashboards", response_model=KPIDashboard)
async def create_kpi_dashboard(dashboard: KPIDashboard):
    """Create KPI dashboard configuration"""
    
    kpi_dashboards_store[dashboard.dashboard_id] = dashboard
    logger.info(f"📊 Created KPI dashboard: {dashboard.dashboard_name}")
    return dashboard

@app.get("/kpis/dashboards/{dashboard_id}/data")
async def get_dashboard_data(dashboard_id: str):
    """Get dashboard data for KPI visualization"""
    
    if dashboard_id not in kpi_dashboards_store:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    dashboard = kpi_dashboards_store[dashboard_id]
    
    # Collect metrics for dashboard
    dashboard_metrics = []
    for metric in kpi_metrics_store.values():
        # Apply filters
        if dashboard.included_kpis and metric.kpi_type not in dashboard.included_kpis:
            continue
        if dashboard.included_categories and metric.kpi_category not in dashboard.included_categories:
            continue
        if dashboard.tenant_filter and metric.tenant_id != dashboard.tenant_filter:
            continue
        
        dashboard_metrics.append(metric)
    
    # Calculate summary statistics
    summary_stats = _calculate_dashboard_summary(dashboard_metrics)
    
    # Get trend analysis
    trend_analysis = _calculate_trend_analysis(dashboard_metrics, dashboard.time_range_days)
    
    # Update last refreshed
    dashboard.last_refreshed = datetime.utcnow()
    kpi_dashboards_store[dashboard_id] = dashboard
    
    return {
        "dashboard_config": dashboard.dict(),
        "summary_stats": summary_stats,
        "metrics": [m.dict() for m in dashboard_metrics],
        "trend_analysis": trend_analysis,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.post("/kpis/reports/generate", response_model=KPIReport)
async def generate_kpi_report(
    report_type: str,
    period_start: datetime,
    period_end: datetime,
    include_categories: Optional[List[KPICategory]] = None
):
    """Generate comprehensive KPI report"""
    
    # Filter metrics by date range and categories
    relevant_metrics = []
    for metric in kpi_metrics_store.values():
        if period_start <= metric.measurement_date <= period_end:
            if not include_categories or metric.kpi_category in include_categories:
                relevant_metrics.append(metric)
    
    # Calculate summary metrics
    kpis_improving = len([m for m in relevant_metrics if m.trend == KPITrend.IMPROVING])
    kpis_declining = len([m for m in relevant_metrics if m.trend == KPITrend.DECLINING])
    
    # Calculate overall differentiation score
    category_scores = {}
    for category in KPICategory:
        category_metrics = [m for m in relevant_metrics if m.kpi_category == category]
        if category_metrics:
            # Score based on status and trend
            score = _calculate_category_score(category_metrics)
            category_scores[category] = score
    
    overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0
    
    # Identify top performing and underperforming KPIs
    top_performing = sorted(relevant_metrics, key=lambda m: m.current_value, reverse=True)[:5]
    underperforming = [m for m in relevant_metrics if m.status in [KPIStatus.NEEDS_IMPROVEMENT, KPIStatus.CRITICAL]]
    
    # Generate recommendations
    recommendations = _generate_improvement_recommendations(relevant_metrics)
    strategic_priorities = _generate_strategic_priorities(category_scores)
    
    report = KPIReport(
        report_type=report_type,
        report_period_start=period_start,
        report_period_end=period_end,
        kpis_tracked=len(relevant_metrics),
        kpis_improving=kpis_improving,
        kpis_declining=kpis_declining,
        overall_differentiation_score=overall_score,
        category_scores=category_scores,
        top_performing_kpis=[{
            "kpi_type": m.kpi_type.value,
            "metric_name": m.metric_name,
            "current_value": m.current_value,
            "unit": m.unit,
            "status": m.status.value
        } for m in top_performing],
        underperforming_kpis=[{
            "kpi_type": m.kpi_type.value,
            "metric_name": m.metric_name,
            "current_value": m.current_value,
            "target_value": m.target_value,
            "unit": m.unit,
            "status": m.status.value
        } for m in underperforming],
        improvement_recommendations=recommendations,
        strategic_priorities=strategic_priorities
    )
    
    kpi_reports_store[report.report_id] = report
    logger.info(f"📋 Generated {report_type} KPI report for period {period_start} to {period_end}")
    
    return report

@app.get("/kpis/analytics/differentiation-score")
async def get_differentiation_score(tenant_id: Optional[str] = None):
    """Get overall RBIA differentiation score"""
    
    # Filter metrics by tenant if specified
    metrics = [m for m in kpi_metrics_store.values() if not tenant_id or m.tenant_id == tenant_id]
    
    if not metrics:
        return {"differentiation_score": 0, "message": "No metrics available"}
    
    # Calculate weighted differentiation score
    category_weights = {
        KPICategory.ADOPTION: 0.20,
        KPICategory.GOVERNANCE: 0.25,
        KPICategory.TRUST: 0.20,
        KPICategory.EFFICIENCY: 0.15,
        KPICategory.COMPLIANCE: 0.15,
        KPICategory.BUSINESS_IMPACT: 0.05
    }
    
    weighted_score = 0
    total_weight = 0
    
    for category, weight in category_weights.items():
        category_metrics = [m for m in metrics if m.kpi_category == category]
        if category_metrics:
            category_score = _calculate_category_score(category_metrics)
            weighted_score += category_score * weight
            total_weight += weight
    
    final_score = weighted_score / total_weight if total_weight > 0 else 0
    
    return {
        "differentiation_score": round(final_score, 2),
        "score_breakdown": {
            category.value: _calculate_category_score([m for m in metrics if m.kpi_category == category])
            for category in KPICategory
        },
        "metrics_analyzed": len(metrics),
        "tenant_id": tenant_id,
        "calculated_at": datetime.utcnow().isoformat()
    }

@app.get("/kpis/alerts", response_model=List[KPIAlert])
async def get_kpi_alerts(active_only: bool = True, severity: Optional[str] = None):
    """Get KPI alerts"""
    
    alerts = list(kpi_alerts_store.values())
    
    if active_only:
        alerts = [a for a in alerts if a.is_active and not a.acknowledged]
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    
    return sorted(alerts, key=lambda a: a.created_at, reverse=True)

@app.post("/kpis/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str):
    """Acknowledge a KPI alert"""
    
    if alert_id not in kpi_alerts_store:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert = kpi_alerts_store[alert_id]
    alert.acknowledged = True
    alert.acknowledged_by = acknowledged_by
    alert.acknowledged_at = datetime.utcnow()
    
    kpi_alerts_store[alert_id] = alert
    
    return {"message": "Alert acknowledged", "alert_id": alert_id}

async def _check_kpi_alerts(metric: KPIMetric):
    """Check if KPI metric triggers any alerts"""
    
    alerts_to_create = []
    
    # Check threshold breaches
    if metric.target_value and metric.current_value < metric.target_value * 0.8:  # 20% below target
        alerts_to_create.append(KPIAlert(
            kpi_type=metric.kpi_type,
            tenant_id=metric.tenant_id,
            alert_type="threshold_breach",
            severity="high" if metric.current_value < metric.target_value * 0.7 else "medium",
            message=f"{metric.metric_name} is significantly below target: {metric.current_value} {metric.unit} (target: {metric.target_value} {metric.unit})",
            current_value=metric.current_value,
            threshold_value=metric.target_value
        ))
    
    # Check trend changes
    if metric.trend == KPITrend.DECLINING and metric.status in [KPIStatus.NEEDS_IMPROVEMENT, KPIStatus.CRITICAL]:
        alerts_to_create.append(KPIAlert(
            kpi_type=metric.kpi_type,
            tenant_id=metric.tenant_id,
            alert_type="trend_change",
            severity="critical" if metric.status == KPIStatus.CRITICAL else "high",
            message=f"{metric.metric_name} is declining and needs immediate attention",
            current_value=metric.current_value,
            previous_value=metric.previous_value
        ))
    
    # Store alerts
    for alert in alerts_to_create:
        kpi_alerts_store[alert.alert_id] = alert
        logger.warning(f"🚨 KPI Alert created: {alert.message}")

def _calculate_dashboard_summary(metrics: List[KPIMetric]) -> Dict[str, Any]:
    """Calculate summary statistics for dashboard"""
    
    if not metrics:
        return {}
    
    return {
        "total_kpis": len(metrics),
        "excellent_kpis": len([m for m in metrics if m.status == KPIStatus.EXCELLENT]),
        "good_kpis": len([m for m in metrics if m.status == KPIStatus.GOOD]),
        "needs_improvement_kpis": len([m for m in metrics if m.status in [KPIStatus.NEEDS_IMPROVEMENT, KPIStatus.CRITICAL]]),
        "improving_trend": len([m for m in metrics if m.trend == KPITrend.IMPROVING]),
        "declining_trend": len([m for m in metrics if m.trend == KPITrend.DECLINING]),
        "average_performance": statistics.mean([_status_to_score(m.status) for m in metrics])
    }

def _calculate_trend_analysis(metrics: List[KPIMetric], days: int) -> Dict[str, Any]:
    """Calculate trend analysis for metrics"""
    
    trend_summary = {
        "period_days": days,
        "trends_by_category": {},
        "overall_trend": KPITrend.STABLE.value
    }
    
    # Group by category
    for category in KPICategory:
        category_metrics = [m for m in metrics if m.kpi_category == category]
        if category_metrics:
            improving = len([m for m in category_metrics if m.trend == KPITrend.IMPROVING])
            declining = len([m for m in category_metrics if m.trend == KPITrend.DECLINING])
            
            if improving > declining:
                overall_trend = "improving"
            elif declining > improving:
                overall_trend = "declining"
            else:
                overall_trend = "stable"
            
            trend_summary["trends_by_category"][category.value] = {
                "trend": overall_trend,
                "improving_count": improving,
                "declining_count": declining,
                "stable_count": len(category_metrics) - improving - declining
            }
    
    return trend_summary

def _calculate_category_score(metrics: List[KPIMetric]) -> float:
    """Calculate score for a category of metrics"""
    
    if not metrics:
        return 0.0
    
    status_scores = [_status_to_score(m.status) for m in metrics]
    return statistics.mean(status_scores)

def _status_to_score(status: KPIStatus) -> float:
    """Convert KPI status to numeric score"""
    
    status_mapping = {
        KPIStatus.EXCELLENT: 95.0,
        KPIStatus.GOOD: 80.0,
        KPIStatus.ACCEPTABLE: 65.0,
        KPIStatus.NEEDS_IMPROVEMENT: 40.0,
        KPIStatus.CRITICAL: 20.0
    }
    
    return status_mapping.get(status, 50.0)

def _generate_improvement_recommendations(metrics: List[KPIMetric]) -> List[str]:
    """Generate improvement recommendations based on KPI performance"""
    
    recommendations = []
    
    # Find underperforming areas
    underperforming = [m for m in metrics if m.status in [KPIStatus.NEEDS_IMPROVEMENT, KPIStatus.CRITICAL]]
    
    for metric in underperforming[:5]:  # Top 5 recommendations
        if metric.kpi_category == KPICategory.ADOPTION:
            recommendations.append(f"Improve {metric.metric_name}: Focus on user training and feature promotion")
        elif metric.kpi_category == KPICategory.GOVERNANCE:
            recommendations.append(f"Enhance {metric.metric_name}: Review policy enforcement and automation rules")
        elif metric.kpi_category == KPICategory.TRUST:
            recommendations.append(f"Strengthen {metric.metric_name}: Increase explainability and transparency features")
        elif metric.kpi_category == KPICategory.COMPLIANCE:
            recommendations.append(f"Address {metric.metric_name}: Review compliance processes and regulatory alignment")
    
    return recommendations

def _generate_strategic_priorities(category_scores: Dict[KPICategory, float]) -> List[str]:
    """Generate strategic priorities based on category performance"""
    
    priorities = []
    
    # Sort categories by score (lowest first - needs most attention)
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1])
    
    for category, score in sorted_categories[:3]:  # Top 3 priorities
        if score < 60:
            priorities.append(f"Critical focus needed on {category.value} - current score: {score:.1f}")
        elif score < 75:
            priorities.append(f"Moderate improvement needed in {category.value} - current score: {score:.1f}")
    
    return priorities

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Differentiation KPIs Service",
        "task": "3.4.18",
        "kpis_tracked": len(kpi_metrics_store),
        "dashboards_configured": len(kpi_dashboards_store),
        "active_alerts": len([a for a in kpi_alerts_store.values() if a.is_active])
    }
