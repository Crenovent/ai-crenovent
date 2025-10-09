"""
Task 4.4.6: Define forecast improvement metrics (variance reduction, error rates)
- CFO adoption driver
- Forecast ML + reporting service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import statistics

app = FastAPI(title="RBIA Forecast Improvement Metrics")
logger = logging.getLogger(__name__)

class ForecastType(str, Enum):
    REVENUE = "revenue"
    PIPELINE = "pipeline"
    BOOKINGS = "bookings"
    CHURN = "churn"
    GROWTH = "growth"

class MetricType(str, Enum):
    VARIANCE_REDUCTION = "variance_reduction"
    ERROR_RATE = "error_rate"
    ACCURACY_IMPROVEMENT = "accuracy_improvement"
    BIAS_REDUCTION = "bias_reduction"

class ForecastMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    forecast_type: ForecastType
    metric_type: MetricType
    
    # Baseline (RBA) vs RBIA comparison
    baseline_value: float  # RBA performance
    rbia_value: float      # RBIA performance
    improvement_percentage: float
    
    # Statistical measures
    variance_reduction: Optional[float] = None
    error_rate_reduction: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    
    # Context
    forecast_period: str  # monthly, quarterly, yearly
    data_points: int
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ForecastImprovementReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Report period
    start_date: datetime
    end_date: datetime
    
    # Metrics by forecast type
    revenue_metrics: List[ForecastMetric] = Field(default_factory=list)
    pipeline_metrics: List[ForecastMetric] = Field(default_factory=list)
    bookings_metrics: List[ForecastMetric] = Field(default_factory=list)
    
    # Summary statistics
    overall_improvement: float = 0.0
    best_performing_type: Optional[ForecastType] = None
    total_variance_reduction: float = 0.0
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
forecast_metrics_store: Dict[str, ForecastMetric] = {}
improvement_reports_store: Dict[str, ForecastImprovementReport] = {}

class ForecastImprovementTracker:
    def __init__(self):
        self.baseline_forecasts: Dict[str, List[float]] = {}
        self.rbia_forecasts: Dict[str, List[float]] = {}
        
    def calculate_variance_reduction(self, baseline_values: List[float], 
                                   rbia_values: List[float]) -> float:
        """Calculate variance reduction between baseline and RBIA"""
        if len(baseline_values) < 2 or len(rbia_values) < 2:
            return 0.0
            
        baseline_variance = statistics.variance(baseline_values)
        rbia_variance = statistics.variance(rbia_values)
        
        if baseline_variance == 0:
            return 0.0
            
        return ((baseline_variance - rbia_variance) / baseline_variance) * 100
    
    def calculate_error_rate(self, predicted: List[float], actual: List[float]) -> float:
        """Calculate mean absolute percentage error"""
        if len(predicted) != len(actual) or len(predicted) == 0:
            return 0.0
            
        errors = []
        for p, a in zip(predicted, actual):
            if a != 0:
                errors.append(abs((a - p) / a))
                
        return (sum(errors) / len(errors)) * 100 if errors else 0.0
    
    def calculate_improvement_metrics(self, tenant_id: str, forecast_type: ForecastType,
                                    baseline_predictions: List[float],
                                    rbia_predictions: List[float],
                                    actual_values: List[float]) -> List[ForecastMetric]:
        """Calculate all improvement metrics"""
        metrics = []
        
        # Variance reduction metric
        variance_reduction = self.calculate_variance_reduction(
            baseline_predictions, rbia_predictions
        )
        
        baseline_error = self.calculate_error_rate(baseline_predictions, actual_values)
        rbia_error = self.calculate_error_rate(rbia_predictions, actual_values)
        error_rate_reduction = baseline_error - rbia_error
        
        # Variance reduction metric
        metrics.append(ForecastMetric(
            tenant_id=tenant_id,
            forecast_type=forecast_type,
            metric_type=MetricType.VARIANCE_REDUCTION,
            baseline_value=statistics.variance(baseline_predictions) if len(baseline_predictions) > 1 else 0.0,
            rbia_value=statistics.variance(rbia_predictions) if len(rbia_predictions) > 1 else 0.0,
            improvement_percentage=variance_reduction,
            variance_reduction=variance_reduction,
            data_points=len(actual_values),
            forecast_period="monthly"
        ))
        
        # Error rate metric
        metrics.append(ForecastMetric(
            tenant_id=tenant_id,
            forecast_type=forecast_type,
            metric_type=MetricType.ERROR_RATE,
            baseline_value=baseline_error,
            rbia_value=rbia_error,
            improvement_percentage=((baseline_error - rbia_error) / baseline_error * 100) if baseline_error > 0 else 0.0,
            error_rate_reduction=error_rate_reduction,
            data_points=len(actual_values),
            forecast_period="monthly"
        ))
        
        # Accuracy improvement
        baseline_accuracy = 100 - baseline_error
        rbia_accuracy = 100 - rbia_error
        accuracy_improvement = rbia_accuracy - baseline_accuracy
        
        metrics.append(ForecastMetric(
            tenant_id=tenant_id,
            forecast_type=forecast_type,
            metric_type=MetricType.ACCURACY_IMPROVEMENT,
            baseline_value=baseline_accuracy,
            rbia_value=rbia_accuracy,
            improvement_percentage=accuracy_improvement,
            data_points=len(actual_values),
            forecast_period="monthly"
        ))
        
        return metrics

# Global tracker instance
tracker = ForecastImprovementTracker()

@app.post("/forecast-metrics/calculate", response_model=List[ForecastMetric])
async def calculate_forecast_metrics(
    tenant_id: str,
    forecast_type: ForecastType,
    baseline_predictions: List[float],
    rbia_predictions: List[float],
    actual_values: List[float]
):
    """Calculate forecast improvement metrics"""
    
    if len(baseline_predictions) != len(rbia_predictions) or len(baseline_predictions) != len(actual_values):
        raise HTTPException(status_code=400, detail="All arrays must have the same length")
    
    if len(baseline_predictions) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 data points")
    
    metrics = tracker.calculate_improvement_metrics(
        tenant_id, forecast_type, baseline_predictions, rbia_predictions, actual_values
    )
    
    # Store metrics
    for metric in metrics:
        forecast_metrics_store[metric.metric_id] = metric
    
    logger.info(f"✅ Calculated {len(metrics)} forecast improvement metrics for {forecast_type}")
    return metrics

@app.get("/forecast-metrics/tenant/{tenant_id}")
async def get_tenant_forecast_metrics(
    tenant_id: str,
    forecast_type: Optional[ForecastType] = None,
    metric_type: Optional[MetricType] = None
):
    """Get forecast metrics for a tenant"""
    
    tenant_metrics = [
        metric for metric in forecast_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    if forecast_type:
        tenant_metrics = [m for m in tenant_metrics if m.forecast_type == forecast_type]
        
    if metric_type:
        tenant_metrics = [m for m in tenant_metrics if m.metric_type == metric_type]
    
    return {
        "tenant_id": tenant_id,
        "metric_count": len(tenant_metrics),
        "metrics": tenant_metrics
    }

@app.post("/forecast-metrics/report", response_model=ForecastImprovementReport)
async def generate_improvement_report(
    tenant_id: str,
    start_date: datetime,
    end_date: datetime
):
    """Generate forecast improvement report"""
    
    # Get metrics for the period
    period_metrics = [
        metric for metric in forecast_metrics_store.values()
        if (metric.tenant_id == tenant_id and 
            start_date <= metric.measurement_date <= end_date)
    ]
    
    # Group by forecast type
    revenue_metrics = [m for m in period_metrics if m.forecast_type == ForecastType.REVENUE]
    pipeline_metrics = [m for m in period_metrics if m.forecast_type == ForecastType.PIPELINE]
    bookings_metrics = [m for m in period_metrics if m.forecast_type == ForecastType.BOOKINGS]
    
    # Calculate overall improvement
    all_improvements = [m.improvement_percentage for m in period_metrics]
    overall_improvement = statistics.mean(all_improvements) if all_improvements else 0.0
    
    # Find best performing type
    type_improvements = {}
    for ftype in ForecastType:
        type_metrics = [m for m in period_metrics if m.forecast_type == ftype]
        if type_metrics:
            type_improvements[ftype] = statistics.mean([m.improvement_percentage for m in type_metrics])
    
    best_performing_type = max(type_improvements, key=type_improvements.get) if type_improvements else None
    
    # Calculate total variance reduction
    variance_metrics = [m for m in period_metrics if m.metric_type == MetricType.VARIANCE_REDUCTION]
    total_variance_reduction = sum([m.improvement_percentage for m in variance_metrics])
    
    report = ForecastImprovementReport(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
        revenue_metrics=revenue_metrics,
        pipeline_metrics=pipeline_metrics,
        bookings_metrics=bookings_metrics,
        overall_improvement=overall_improvement,
        best_performing_type=best_performing_type,
        total_variance_reduction=total_variance_reduction
    )
    
    improvement_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated forecast improvement report for tenant {tenant_id}")
    return report

@app.get("/forecast-metrics/summary")
async def get_metrics_summary():
    """Get summary of all forecast metrics"""
    
    total_metrics = len(forecast_metrics_store)
    
    # Group by type
    type_counts = {}
    for metric_type in MetricType:
        type_counts[metric_type.value] = len([
            m for m in forecast_metrics_store.values() 
            if m.metric_type == metric_type
        ])
    
    # Calculate average improvements
    avg_improvements = {}
    for forecast_type in ForecastType:
        type_metrics = [
            m for m in forecast_metrics_store.values() 
            if m.forecast_type == forecast_type
        ]
        if type_metrics:
            avg_improvements[forecast_type.value] = statistics.mean([
                m.improvement_percentage for m in type_metrics
            ])
    
    return {
        "total_metrics": total_metrics,
        "metrics_by_type": type_counts,
        "average_improvements": avg_improvements,
        "total_reports": len(improvement_reports_store)
    }
