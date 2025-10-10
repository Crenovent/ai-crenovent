"""
Task 3.3.48: Cross-Mode Trust Index Report Service
- Compare reliability per mode for CRO/CFO reports
- Dashboard integration with report generator
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Cross-Mode Trust Index Report Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class TrustMetric(str, Enum):
    ACCURACY = "accuracy"
    RELIABILITY = "reliability"
    CONSISTENCY = "consistency"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    AVAILABILITY = "availability"

class ReportPeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ModeTrustScore(BaseModel):
    mode: UXMode
    overall_trust_score: float  # 0-100
    
    # Individual metrics
    accuracy_score: float = 0.0
    reliability_score: float = 0.0
    consistency_score: float = 0.0
    user_satisfaction_score: float = 0.0
    
    # Performance metrics
    avg_error_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    availability_percentage: float = 0.0
    
    # Usage statistics
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    
    # Trend
    trend_direction: str = "stable"  # improving, declining, stable
    trend_percentage: float = 0.0

class TrustIndexReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Report metadata
    report_title: str
    report_period: ReportPeriod
    start_date: datetime
    end_date: datetime
    
    # Trust scores by mode
    mode_scores: List[ModeTrustScore] = Field(default_factory=list)
    
    # Comparative analysis
    highest_trust_mode: Optional[UXMode] = None
    lowest_trust_mode: Optional[UXMode] = None
    trust_score_variance: float = 0.0
    
    # Executive summary
    executive_summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Report generation
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = "system"

class TrustMetricData(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Metric details
    mode: UXMode
    metric_type: TrustMetric
    metric_value: float
    
    # Context
    workflow_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Timing
    recorded_at: datetime = Field(default_factory=datetime.utcnow)

class ReportConfiguration(BaseModel):
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Report settings
    auto_generate: bool = True
    report_frequency: ReportPeriod = ReportPeriod.WEEKLY
    
    # Recipients
    report_recipients: List[str] = Field(default_factory=list)  # Email addresses
    
    # Metrics to include
    included_metrics: List[TrustMetric] = Field(default_factory=lambda: [
        TrustMetric.ACCURACY, TrustMetric.RELIABILITY, TrustMetric.USER_SATISFACTION
    ])
    
    # Thresholds for alerts
    trust_score_threshold: float = 70.0  # Alert if below this
    variance_threshold: float = 20.0  # Alert if variance exceeds this
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
trust_reports_store: Dict[str, TrustIndexReport] = {}
trust_metrics_store: Dict[str, TrustMetricData] = {}
report_configs_store: Dict[str, ReportConfiguration] = {}

def _generate_sample_metrics(tenant_id: str):
    """Generate sample trust metrics for demonstration"""
    
    import random
    
    # Generate metrics for the last 30 days
    for days_ago in range(30):
        date = datetime.utcnow() - timedelta(days=days_ago)
        
        for mode in UXMode:
            # Generate different baseline scores for each mode
            if mode == UXMode.UI_LED:
                base_accuracy = 0.92
                base_reliability = 0.95
                base_satisfaction = 0.88
            elif mode == UXMode.ASSISTED:
                base_accuracy = 0.89
                base_reliability = 0.91
                base_satisfaction = 0.85
            else:  # CONVERSATIONAL
                base_accuracy = 0.85
                base_reliability = 0.87
                base_satisfaction = 0.82
            
            # Add some random variation
            accuracy = max(0.7, min(1.0, base_accuracy + random.uniform(-0.05, 0.05)))
            reliability = max(0.7, min(1.0, base_reliability + random.uniform(-0.03, 0.03)))
            satisfaction = max(0.6, min(1.0, base_satisfaction + random.uniform(-0.08, 0.08)))
            
            # Create metrics
            metrics = [
                TrustMetricData(
                    tenant_id=tenant_id,
                    mode=mode,
                    metric_type=TrustMetric.ACCURACY,
                    metric_value=accuracy * 100,
                    recorded_at=date
                ),
                TrustMetricData(
                    tenant_id=tenant_id,
                    mode=mode,
                    metric_type=TrustMetric.RELIABILITY,
                    metric_value=reliability * 100,
                    recorded_at=date
                ),
                TrustMetricData(
                    tenant_id=tenant_id,
                    mode=mode,
                    metric_type=TrustMetric.USER_SATISFACTION,
                    metric_value=satisfaction * 100,
                    recorded_at=date
                )
            ]
            
            for metric in metrics:
                trust_metrics_store[metric.metric_id] = metric

# Generate sample data for default tenant
_generate_sample_metrics("default")

@app.post("/trust-reports/config", response_model=ReportConfiguration)
async def configure_trust_reports(config: ReportConfiguration):
    """Configure trust index report settings"""
    report_configs_store[config.config_id] = config
    logger.info(f"Configured trust reports for tenant {config.tenant_id}")
    return config

@app.post("/trust-metrics", response_model=TrustMetricData)
async def record_trust_metric(metric: TrustMetricData):
    """Record a trust metric data point"""
    trust_metrics_store[metric.metric_id] = metric
    logger.info(f"Recorded {metric.metric_type.value} metric for {metric.mode.value}: {metric.metric_value}")
    return metric

@app.post("/trust-reports/generate", response_model=TrustIndexReport)
async def generate_trust_report(
    tenant_id: str,
    report_period: ReportPeriod,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate a cross-mode trust index report"""
    
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.utcnow()
    
    if not start_date:
        if report_period == ReportPeriod.DAILY:
            start_date = end_date - timedelta(days=1)
        elif report_period == ReportPeriod.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif report_period == ReportPeriod.MONTHLY:
            start_date = end_date - timedelta(days=30)
        else:  # QUARTERLY
            start_date = end_date - timedelta(days=90)
    
    # Generate report
    report = await _generate_report(tenant_id, report_period, start_date, end_date)
    
    # Store report
    trust_reports_store[report.report_id] = report
    
    # Send report in background if configured
    background_tasks.add_task(_send_report_if_configured, report)
    
    logger.info(f"Generated trust index report {report.report_id} for tenant {tenant_id}")
    return report

async def _generate_report(
    tenant_id: str, 
    report_period: ReportPeriod, 
    start_date: datetime, 
    end_date: datetime
) -> TrustIndexReport:
    """Generate the actual trust index report"""
    
    # Get metrics for the period
    period_metrics = [
        metric for metric in trust_metrics_store.values()
        if (metric.tenant_id == tenant_id and 
            start_date <= metric.recorded_at <= end_date)
    ]
    
    # Calculate trust scores for each mode
    mode_scores = []
    trust_scores = []
    
    for mode in UXMode:
        mode_metrics = [m for m in period_metrics if m.mode == mode]
        
        if not mode_metrics:
            continue
        
        # Calculate average scores for each metric type
        accuracy_metrics = [m.metric_value for m in mode_metrics if m.metric_type == TrustMetric.ACCURACY]
        reliability_metrics = [m.metric_value for m in mode_metrics if m.metric_type == TrustMetric.RELIABILITY]
        satisfaction_metrics = [m.metric_value for m in mode_metrics if m.metric_type == TrustMetric.USER_SATISFACTION]
        
        accuracy_score = sum(accuracy_metrics) / len(accuracy_metrics) if accuracy_metrics else 0
        reliability_score = sum(reliability_metrics) / len(reliability_metrics) if reliability_metrics else 0
        satisfaction_score = sum(satisfaction_metrics) / len(satisfaction_metrics) if satisfaction_metrics else 0
        
        # Calculate overall trust score (weighted average)
        overall_trust = (accuracy_score * 0.4 + reliability_score * 0.35 + satisfaction_score * 0.25)
        
        # Calculate trend (simplified)
        trend_direction, trend_percentage = _calculate_trend(mode_metrics, TrustMetric.ACCURACY)
        
        mode_score = ModeTrustScore(
            mode=mode,
            overall_trust_score=overall_trust,
            accuracy_score=accuracy_score,
            reliability_score=reliability_score,
            user_satisfaction_score=satisfaction_score,
            total_interactions=len(mode_metrics),
            successful_interactions=int(len(mode_metrics) * (reliability_score / 100)),
            failed_interactions=len(mode_metrics) - int(len(mode_metrics) * (reliability_score / 100)),
            trend_direction=trend_direction,
            trend_percentage=trend_percentage,
            avg_response_time_ms=200 + (100 - overall_trust) * 10,  # Simulated
            availability_percentage=95 + (overall_trust - 80) * 0.1,  # Simulated
            avg_error_rate=(100 - reliability_score) / 10  # Simulated
        )
        
        mode_scores.append(mode_score)
        trust_scores.append(overall_trust)
    
    # Find highest and lowest trust modes
    highest_trust_mode = None
    lowest_trust_mode = None
    
    if mode_scores:
        highest_trust_mode = max(mode_scores, key=lambda x: x.overall_trust_score).mode
        lowest_trust_mode = min(mode_scores, key=lambda x: x.overall_trust_score).mode
    
    # Calculate variance
    trust_score_variance = _calculate_variance(trust_scores) if trust_scores else 0
    
    # Generate executive summary and findings
    executive_summary, key_findings, recommendations = _generate_executive_content(
        mode_scores, highest_trust_mode, lowest_trust_mode, trust_score_variance
    )
    
    # Create report
    report = TrustIndexReport(
        tenant_id=tenant_id,
        report_title=f"Cross-Mode Trust Index Report - {report_period.value.title()}",
        report_period=report_period,
        start_date=start_date,
        end_date=end_date,
        mode_scores=mode_scores,
        highest_trust_mode=highest_trust_mode,
        lowest_trust_mode=lowest_trust_mode,
        trust_score_variance=trust_score_variance,
        executive_summary=executive_summary,
        key_findings=key_findings,
        recommendations=recommendations
    )
    
    return report

def _calculate_trend(metrics: List[TrustMetricData], metric_type: TrustMetric) -> tuple[str, float]:
    """Calculate trend direction and percentage for a metric"""
    
    type_metrics = [m for m in metrics if m.metric_type == metric_type]
    
    if len(type_metrics) < 2:
        return "stable", 0.0
    
    # Sort by date
    type_metrics.sort(key=lambda x: x.recorded_at)
    
    # Compare first half with second half
    mid_point = len(type_metrics) // 2
    first_half_avg = sum(m.metric_value for m in type_metrics[:mid_point]) / mid_point
    second_half_avg = sum(m.metric_value for m in type_metrics[mid_point:]) / (len(type_metrics) - mid_point)
    
    percentage_change = ((second_half_avg - first_half_avg) / first_half_avg) * 100
    
    if percentage_change > 2:
        return "improving", percentage_change
    elif percentage_change < -2:
        return "declining", abs(percentage_change)
    else:
        return "stable", abs(percentage_change)

def _calculate_variance(scores: List[float]) -> float:
    """Calculate variance of trust scores"""
    
    if len(scores) < 2:
        return 0.0
    
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    
    return variance ** 0.5  # Return standard deviation

def _generate_executive_content(
    mode_scores: List[ModeTrustScore],
    highest_trust_mode: Optional[UXMode],
    lowest_trust_mode: Optional[UXMode],
    variance: float
) -> tuple[str, List[str], List[str]]:
    """Generate executive summary, key findings, and recommendations"""
    
    if not mode_scores:
        return "No data available for the selected period.", [], []
    
    # Executive summary
    avg_trust = sum(score.overall_trust_score for score in mode_scores) / len(mode_scores)
    
    executive_summary = f"""
    Cross-mode trust analysis reveals an average trust score of {avg_trust:.1f} across all UX modes. 
    {highest_trust_mode.value.replace('_', ' ').title() if highest_trust_mode else 'N/A'} mode demonstrates 
    the highest reliability, while {lowest_trust_mode.value.replace('_', ' ').title() if lowest_trust_mode else 'N/A'} 
    mode shows the most opportunity for improvement. The variance of {variance:.1f} indicates 
    {'consistent' if variance < 10 else 'variable'} performance across modes.
    """
    
    # Key findings
    key_findings = []
    
    for score in mode_scores:
        mode_name = score.mode.value.replace('_', ' ').title()
        
        if score.overall_trust_score >= 90:
            key_findings.append(f"{mode_name} mode shows excellent trust scores ({score.overall_trust_score:.1f})")
        elif score.overall_trust_score >= 80:
            key_findings.append(f"{mode_name} mode demonstrates good reliability ({score.overall_trust_score:.1f})")
        elif score.overall_trust_score >= 70:
            key_findings.append(f"{mode_name} mode needs attention ({score.overall_trust_score:.1f})")
        else:
            key_findings.append(f"{mode_name} mode requires immediate improvement ({score.overall_trust_score:.1f})")
        
        if score.trend_direction == "improving":
            key_findings.append(f"{mode_name} mode shows improving trend (+{score.trend_percentage:.1f}%)")
        elif score.trend_direction == "declining":
            key_findings.append(f"{mode_name} mode shows declining trend (-{score.trend_percentage:.1f}%)")
    
    # Recommendations
    recommendations = []
    
    if variance > 15:
        recommendations.append("Standardize quality processes across all UX modes to reduce variance")
    
    for score in mode_scores:
        mode_name = score.mode.value.replace('_', ' ').title()
        
        if score.overall_trust_score < 80:
            recommendations.append(f"Improve {mode_name} mode reliability through enhanced testing and monitoring")
        
        if score.trend_direction == "declining":
            recommendations.append(f"Investigate root causes of declining performance in {mode_name} mode")
        
        if score.avg_error_rate > 5:
            recommendations.append(f"Reduce error rate in {mode_name} mode through better validation")
    
    if avg_trust < 85:
        recommendations.append("Implement comprehensive quality improvement program across all modes")
    
    return executive_summary.strip(), key_findings, recommendations

async def _send_report_if_configured(report: TrustIndexReport):
    """Send report to configured recipients if auto-send is enabled"""
    
    # Find configuration for this tenant
    config = None
    for cfg in report_configs_store.values():
        if cfg.tenant_id == report.tenant_id:
            config = cfg
            break
    
    if config and config.report_recipients:
        # In production, this would send actual emails
        logger.info(f"Would send report {report.report_id} to {len(config.report_recipients)} recipients")

@app.get("/trust-reports", response_model=List[TrustIndexReport])
async def get_trust_reports(
    tenant_id: str,
    report_period: Optional[ReportPeriod] = None,
    limit: int = 10
):
    """Get trust index reports for a tenant"""
    
    reports = [
        report for report in trust_reports_store.values()
        if report.tenant_id == tenant_id
    ]
    
    if report_period:
        reports = [r for r in reports if r.report_period == report_period]
    
    # Sort by generation time (most recent first)
    reports.sort(key=lambda x: x.generated_at, reverse=True)
    
    return reports[:limit]

@app.get("/trust-reports/{report_id}", response_model=TrustIndexReport)
async def get_trust_report(report_id: str):
    """Get a specific trust index report"""
    
    if report_id not in trust_reports_store:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return trust_reports_store[report_id]

@app.get("/trust-reports/{report_id}/dashboard-data")
async def get_report_dashboard_data(report_id: str):
    """Get dashboard-friendly data for a trust report"""
    
    if report_id not in trust_reports_store:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report = trust_reports_store[report_id]
    
    # Prepare dashboard data
    dashboard_data = {
        "report_id": report_id,
        "report_title": report.report_title,
        "period": f"{report.start_date.strftime('%Y-%m-%d')} to {report.end_date.strftime('%Y-%m-%d')}",
        "overall_metrics": {
            "avg_trust_score": sum(score.overall_trust_score for score in report.mode_scores) / len(report.mode_scores) if report.mode_scores else 0,
            "highest_trust_mode": report.highest_trust_mode.value if report.highest_trust_mode else None,
            "lowest_trust_mode": report.lowest_trust_mode.value if report.lowest_trust_mode else None,
            "trust_score_variance": report.trust_score_variance
        },
        "mode_comparison": [
            {
                "mode": score.mode.value,
                "trust_score": score.overall_trust_score,
                "accuracy": score.accuracy_score,
                "reliability": score.reliability_score,
                "satisfaction": score.user_satisfaction_score,
                "trend": score.trend_direction,
                "trend_percentage": score.trend_percentage
            }
            for score in report.mode_scores
        ],
        "executive_summary": report.executive_summary,
        "key_findings": report.key_findings,
        "recommendations": report.recommendations
    }
    
    return dashboard_data

@app.get("/analytics/trust-trends")
async def get_trust_trends(
    tenant_id: str,
    days: int = 30,
    mode: Optional[UXMode] = None
):
    """Get trust score trends over time"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get metrics for the period
    metrics = [
        metric for metric in trust_metrics_store.values()
        if (metric.tenant_id == tenant_id and 
            metric.recorded_at >= cutoff_date)
    ]
    
    if mode:
        metrics = [m for m in metrics if m.mode == mode]
    
    # Group by date and mode
    daily_scores = {}
    
    for metric in metrics:
        date_key = metric.recorded_at.strftime('%Y-%m-%d')
        mode_key = metric.mode.value
        
        if date_key not in daily_scores:
            daily_scores[date_key] = {}
        
        if mode_key not in daily_scores[date_key]:
            daily_scores[date_key][mode_key] = []
        
        daily_scores[date_key][mode_key].append(metric.metric_value)
    
    # Calculate daily averages
    trend_data = []
    
    for date_key in sorted(daily_scores.keys()):
        date_data = {"date": date_key}
        
        for mode_key, values in daily_scores[date_key].items():
            date_data[mode_key] = sum(values) / len(values)
        
        trend_data.append(date_data)
    
    return {"trend_data": trend_data, "period_days": days}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Cross-Mode Trust Index Report Service", "task": "3.3.48"}
