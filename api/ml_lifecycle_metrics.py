"""
Task 4.4.41: Build ML lifecycle metric (model accuracy over time + retraining frequency)
- Ops monitoring
- Metrics pipeline
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import statistics

app = FastAPI(title="RBIA ML Lifecycle Metrics")
logger = logging.getLogger(__name__)

class ModelStage(str, Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    PRODUCTION = "production"
    RETIRED = "retired"

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    DRIFT_SCORE = "drift_score"

class ModelMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    model_id: str
    model_name: str
    model_version: str
    
    # Metric details
    metric_type: MetricType
    metric_value: float
    baseline_value: Optional[float] = None
    
    # Model context
    stage: ModelStage
    training_data_size: int
    evaluation_data_size: int
    
    # Timestamps
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    model_training_date: Optional[datetime] = None
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrainingEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    model_id: str
    model_name: str
    
    # Retraining details
    trigger_reason: str  # scheduled, drift_detected, performance_degradation
    old_version: str
    new_version: str
    
    # Performance comparison
    old_accuracy: float
    new_accuracy: float
    accuracy_improvement: float
    
    # Training details
    training_duration_hours: float
    training_data_size: int
    validation_accuracy: float
    
    # Timestamps
    retrain_started: datetime
    retrain_completed: datetime = Field(default_factory=datetime.utcnow)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MLLifecycleReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    model_id: str
    model_name: str
    
    # Report period
    start_date: datetime
    end_date: datetime
    
    # Accuracy over time
    accuracy_trend: List[Dict[str, Any]] = Field(default_factory=list)
    accuracy_degradation_rate: float = 0.0
    
    # Retraining frequency
    total_retrainings: int = 0
    avg_days_between_retraining: float = 0.0
    retraining_frequency_score: float = 0.0  # 0-100 score
    
    # Performance metrics
    current_accuracy: float = 0.0
    best_accuracy: float = 0.0
    worst_accuracy: float = 0.0
    accuracy_stability_score: float = 0.0
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
model_metrics_store: Dict[str, ModelMetric] = {}
retraining_events_store: Dict[str, RetrainingEvent] = {}
lifecycle_reports_store: Dict[str, MLLifecycleReport] = {}

class MLLifecycleTracker:
    def __init__(self):
        self.degradation_threshold = 0.05  # 5% accuracy drop triggers alert
        self.stability_window_days = 30
        
    def calculate_accuracy_trend(self, model_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """Calculate accuracy trend over time"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get accuracy metrics for the model
        accuracy_metrics = [
            metric for metric in model_metrics_store.values()
            if (metric.model_id == model_id and 
                metric.metric_type == MetricType.ACCURACY and
                metric.measurement_date >= cutoff_date)
        ]
        
        # Sort by date
        accuracy_metrics.sort(key=lambda x: x.measurement_date)
        
        trend = []
        for metric in accuracy_metrics:
            trend.append({
                "date": metric.measurement_date.isoformat(),
                "accuracy": metric.metric_value,
                "model_version": metric.model_version,
                "stage": metric.stage.value
            })
            
        return trend
    
    def calculate_degradation_rate(self, accuracy_trend: List[Dict[str, Any]]) -> float:
        """Calculate accuracy degradation rate per day"""
        
        if len(accuracy_trend) < 2:
            return 0.0
            
        # Calculate linear degradation rate
        values = [point["accuracy"] for point in accuracy_trend]
        dates = [datetime.fromisoformat(point["date"]) for point in accuracy_trend]
        
        if len(values) < 2:
            return 0.0
            
        # Simple linear regression for degradation rate
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Convert to degradation per day
        total_days = (dates[-1] - dates[0]).days
        degradation_per_day = slope / total_days if total_days > 0 else 0.0
        
        return degradation_per_day
    
    def calculate_retraining_frequency_score(self, model_id: str, days: int = 90) -> Dict[str, Any]:
        """Calculate retraining frequency score"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get retraining events
        retrainings = [
            event for event in retraining_events_store.values()
            if (event.model_id == model_id and 
                event.retrain_completed >= cutoff_date)
        ]
        
        total_retrainings = len(retrainings)
        
        if total_retrainings == 0:
            return {
                "total_retrainings": 0,
                "avg_days_between": 0,
                "frequency_score": 50  # Neutral score
            }
        
        # Calculate average days between retraining
        if total_retrainings > 1:
            retrainings.sort(key=lambda x: x.retrain_completed)
            intervals = []
            for i in range(1, len(retrainings)):
                interval = (retrainings[i].retrain_completed - retrainings[i-1].retrain_completed).days
                intervals.append(interval)
            avg_days_between = statistics.mean(intervals)
        else:
            avg_days_between = days  # Single retraining in period
        
        # Score based on retraining frequency (optimal is 30-60 days)
        if 30 <= avg_days_between <= 60:
            frequency_score = 100
        elif 15 <= avg_days_between < 30 or 60 < avg_days_between <= 90:
            frequency_score = 80
        elif avg_days_between < 15:
            frequency_score = 60  # Too frequent
        else:
            frequency_score = 40  # Too infrequent
        
        return {
            "total_retrainings": total_retrainings,
            "avg_days_between": avg_days_between,
            "frequency_score": frequency_score
        }
    
    def calculate_stability_score(self, accuracy_trend: List[Dict[str, Any]]) -> float:
        """Calculate accuracy stability score"""
        
        if len(accuracy_trend) < 3:
            return 50.0  # Neutral score for insufficient data
            
        accuracies = [point["accuracy"] for point in accuracy_trend]
        
        # Calculate coefficient of variation (lower is more stable)
        mean_accuracy = statistics.mean(accuracies)
        std_accuracy = statistics.stdev(accuracies)
        
        if mean_accuracy == 0:
            return 0.0
            
        cv = std_accuracy / mean_accuracy
        
        # Convert to stability score (0-100, higher is more stable)
        # CV of 0.05 (5%) = 90 score, CV of 0.20 (20%) = 10 score
        stability_score = max(0, min(100, 100 - (cv * 500)))
        
        return stability_score

# Global tracker instance
tracker = MLLifecycleTracker()

@app.post("/ml-lifecycle/metrics/record", response_model=ModelMetric)
async def record_model_metric(metric: ModelMetric):
    """Record a model performance metric"""
    
    model_metrics_store[metric.metric_id] = metric
    
    logger.info(f"✅ Recorded {metric.metric_type} metric for model {metric.model_name}: {metric.metric_value}")
    return metric

@app.post("/ml-lifecycle/retraining/record", response_model=RetrainingEvent)
async def record_retraining_event(event: RetrainingEvent):
    """Record a model retraining event"""
    
    # Calculate accuracy improvement
    event.accuracy_improvement = event.new_accuracy - event.old_accuracy
    
    retraining_events_store[event.event_id] = event
    
    logger.info(f"✅ Recorded retraining event for model {event.model_name}: {event.old_version} → {event.new_version}")
    return event

@app.get("/ml-lifecycle/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: str,
    metric_type: Optional[MetricType] = None,
    days: int = 30
):
    """Get metrics for a specific model"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    model_metrics = [
        metric for metric in model_metrics_store.values()
        if (metric.model_id == model_id and 
            metric.measurement_date >= cutoff_date)
    ]
    
    if metric_type:
        model_metrics = [m for m in model_metrics if m.metric_type == metric_type]
    
    return {
        "model_id": model_id,
        "metric_count": len(model_metrics),
        "metrics": model_metrics
    }

@app.get("/ml-lifecycle/models/{model_id}/retraining-history")
async def get_retraining_history(model_id: str, days: int = 90):
    """Get retraining history for a model"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    retrainings = [
        event for event in retraining_events_store.values()
        if (event.model_id == model_id and 
            event.retrain_completed >= cutoff_date)
    ]
    
    # Sort by completion date
    retrainings.sort(key=lambda x: x.retrain_completed, reverse=True)
    
    return {
        "model_id": model_id,
        "retraining_count": len(retrainings),
        "retrainings": retrainings
    }

@app.post("/ml-lifecycle/reports/generate", response_model=MLLifecycleReport)
async def generate_lifecycle_report(
    tenant_id: str,
    model_id: str,
    model_name: str,
    start_date: datetime,
    end_date: datetime
):
    """Generate ML lifecycle report"""
    
    # Calculate accuracy trend
    days_in_period = (end_date - start_date).days
    accuracy_trend = tracker.calculate_accuracy_trend(model_id, days_in_period)
    
    # Calculate degradation rate
    degradation_rate = tracker.calculate_degradation_rate(accuracy_trend)
    
    # Calculate retraining frequency
    freq_data = tracker.calculate_retraining_frequency_score(model_id, days_in_period)
    
    # Calculate stability score
    stability_score = tracker.calculate_stability_score(accuracy_trend)
    
    # Get current, best, worst accuracy
    accuracies = [point["accuracy"] for point in accuracy_trend]
    current_accuracy = accuracies[-1] if accuracies else 0.0
    best_accuracy = max(accuracies) if accuracies else 0.0
    worst_accuracy = min(accuracies) if accuracies else 0.0
    
    # Generate recommendations
    recommendations = []
    if degradation_rate < -0.01:  # Degrading by 1% per day
        recommendations.append("Model accuracy is degrading rapidly - consider immediate retraining")
    if freq_data["frequency_score"] < 60:
        recommendations.append("Retraining frequency is suboptimal - review retraining schedule")
    if stability_score < 70:
        recommendations.append("Model accuracy is unstable - investigate data quality and model robustness")
    if current_accuracy < best_accuracy * 0.95:
        recommendations.append("Current accuracy is significantly below peak - retraining recommended")
    
    report = MLLifecycleReport(
        tenant_id=tenant_id,
        model_id=model_id,
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
        accuracy_trend=accuracy_trend,
        accuracy_degradation_rate=degradation_rate,
        total_retrainings=freq_data["total_retrainings"],
        avg_days_between_retraining=freq_data["avg_days_between"],
        retraining_frequency_score=freq_data["frequency_score"],
        current_accuracy=current_accuracy,
        best_accuracy=best_accuracy,
        worst_accuracy=worst_accuracy,
        accuracy_stability_score=stability_score,
        recommendations=recommendations
    )
    
    lifecycle_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated ML lifecycle report for model {model_name}")
    return report

@app.get("/ml-lifecycle/tenant/{tenant_id}/summary")
async def get_tenant_ml_summary(tenant_id: str):
    """Get ML lifecycle summary for tenant"""
    
    # Get all models for tenant
    tenant_metrics = [
        metric for metric in model_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    tenant_retrainings = [
        event for event in retraining_events_store.values()
        if event.tenant_id == tenant_id
    ]
    
    # Get unique models
    unique_models = set(metric.model_id for metric in tenant_metrics)
    
    # Calculate averages
    if tenant_metrics:
        avg_accuracy = statistics.mean([
            m.metric_value for m in tenant_metrics 
            if m.metric_type == MetricType.ACCURACY
        ])
    else:
        avg_accuracy = 0.0
    
    return {
        "tenant_id": tenant_id,
        "total_models": len(unique_models),
        "total_metrics": len(tenant_metrics),
        "total_retrainings": len(tenant_retrainings),
        "average_accuracy": round(avg_accuracy, 4),
        "total_reports": len([
            r for r in lifecycle_reports_store.values() 
            if r.tenant_id == tenant_id
        ])
    }

@app.get("/ml-lifecycle/summary")
async def get_ml_lifecycle_summary():
    """Get overall ML lifecycle summary"""
    
    total_metrics = len(model_metrics_store)
    total_retrainings = len(retraining_events_store)
    total_reports = len(lifecycle_reports_store)
    
    # Get unique models and tenants
    unique_models = set(metric.model_id for metric in model_metrics_store.values())
    unique_tenants = set(metric.tenant_id for metric in model_metrics_store.values())
    
    return {
        "total_models": len(unique_models),
        "total_tenants": len(unique_tenants),
        "total_metrics": total_metrics,
        "total_retrainings": total_retrainings,
        "total_reports": total_reports
    }
