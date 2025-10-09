"""
Task 4.4.7: Define anomaly/fraud detection precision & recall metrics
- Fraud/ops KPIs
- ML metrics pipeline per industry overlay
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Anomaly/Fraud Detection Metrics")
logger = logging.getLogger(__name__)

class IndustryType(str, Enum):
    SAAS = "saas"
    BANKING = "banking"
    INSURANCE = "insurance"
    ECOMMERCE = "ecommerce"
    FINTECH = "fintech"

class AnomalyType(str, Enum):
    FRAUD_TRANSACTION = "fraud_transaction"
    REVENUE_ANOMALY = "revenue_anomaly"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_ANOMALY = "system_anomaly"
    DATA_QUALITY = "data_quality"

class DetectionMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    industry_type: IndustryType
    anomaly_type: AnomalyType
    
    # Core ML metrics
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Calculated metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    
    # Business impact
    detected_anomalies: int
    prevented_loss: Optional[float] = None
    false_alarm_cost: Optional[float] = None
    
    # Time context
    detection_period: str  # daily, weekly, monthly
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional context
    model_version: str = "v1.0"
    confidence_threshold: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DetectionReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    industry_type: IndustryType
    
    # Report period
    start_date: datetime
    end_date: datetime
    
    # Metrics by anomaly type
    fraud_metrics: List[DetectionMetric] = Field(default_factory=list)
    revenue_metrics: List[DetectionMetric] = Field(default_factory=list)
    behavior_metrics: List[DetectionMetric] = Field(default_factory=list)
    
    # Summary statistics
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    total_prevented_loss: float = 0.0
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
detection_metrics_store: Dict[str, DetectionMetric] = {}
detection_reports_store: Dict[str, DetectionReport] = {}

class AnomalyDetectionMetricsCalculator:
    def __init__(self):
        self.industry_thresholds = {
            IndustryType.BANKING: {"precision_min": 0.95, "recall_min": 0.80},
            IndustryType.FINTECH: {"precision_min": 0.90, "recall_min": 0.85},
            IndustryType.ECOMMERCE: {"precision_min": 0.85, "recall_min": 0.75},
            IndustryType.SAAS: {"precision_min": 0.80, "recall_min": 0.70},
            IndustryType.INSURANCE: {"precision_min": 0.92, "recall_min": 0.78}
        }
    
    def calculate_metrics(self, tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """Calculate precision, recall, F1, accuracy, specificity"""
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "accuracy": round(accuracy, 4),
            "specificity": round(specificity, 4)
        }
    
    def evaluate_performance(self, metric: DetectionMetric) -> Dict[str, Any]:
        """Evaluate performance against industry thresholds"""
        
        thresholds = self.industry_thresholds.get(metric.industry_type, {
            "precision_min": 0.80, "recall_min": 0.70
        })
        
        precision_meets_threshold = metric.precision >= thresholds["precision_min"]
        recall_meets_threshold = metric.recall >= thresholds["recall_min"]
        
        performance_grade = "A"
        if not precision_meets_threshold or not recall_meets_threshold:
            performance_grade = "B"
        if metric.precision < 0.70 or metric.recall < 0.60:
            performance_grade = "C"
        if metric.precision < 0.60 or metric.recall < 0.50:
            performance_grade = "D"
        
        return {
            "precision_meets_threshold": precision_meets_threshold,
            "recall_meets_threshold": recall_meets_threshold,
            "performance_grade": performance_grade,
            "improvement_needed": not (precision_meets_threshold and recall_meets_threshold)
        }

# Global calculator instance
calculator = AnomalyDetectionMetricsCalculator()

@app.post("/detection-metrics/calculate", response_model=DetectionMetric)
async def calculate_detection_metrics(
    tenant_id: str,
    industry_type: IndustryType,
    anomaly_type: AnomalyType,
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
    detection_period: str = "daily",
    model_version: str = "v1.0",
    confidence_threshold: float = 0.5,
    prevented_loss: Optional[float] = None,
    false_alarm_cost: Optional[float] = None
):
    """Calculate anomaly/fraud detection metrics"""
    
    # Calculate core metrics
    metrics = calculator.calculate_metrics(
        true_positives, false_positives, true_negatives, false_negatives
    )
    
    # Create metric record
    detection_metric = DetectionMetric(
        tenant_id=tenant_id,
        industry_type=industry_type,
        anomaly_type=anomaly_type,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        accuracy=metrics["accuracy"],
        specificity=metrics["specificity"],
        detected_anomalies=true_positives,
        prevented_loss=prevented_loss,
        false_alarm_cost=false_alarm_cost,
        detection_period=detection_period,
        model_version=model_version,
        confidence_threshold=confidence_threshold
    )
    
    # Store metric
    detection_metrics_store[detection_metric.metric_id] = detection_metric
    
    logger.info(f"✅ Calculated detection metrics for {anomaly_type} - Precision: {metrics['precision']}, Recall: {metrics['recall']}")
    return detection_metric

@app.get("/detection-metrics/tenant/{tenant_id}")
async def get_tenant_detection_metrics(
    tenant_id: str,
    industry_type: Optional[IndustryType] = None,
    anomaly_type: Optional[AnomalyType] = None
):
    """Get detection metrics for a tenant"""
    
    tenant_metrics = [
        metric for metric in detection_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    if industry_type:
        tenant_metrics = [m for m in tenant_metrics if m.industry_type == industry_type]
        
    if anomaly_type:
        tenant_metrics = [m for m in tenant_metrics if m.anomaly_type == anomaly_type]
    
    return {
        "tenant_id": tenant_id,
        "metric_count": len(tenant_metrics),
        "metrics": tenant_metrics
    }

@app.get("/detection-metrics/{metric_id}/evaluation")
async def evaluate_metric_performance(metric_id: str):
    """Evaluate metric performance against industry thresholds"""
    
    if metric_id not in detection_metrics_store:
        raise HTTPException(status_code=404, detail="Metric not found")
    
    metric = detection_metrics_store[metric_id]
    evaluation = calculator.evaluate_performance(metric)
    
    return {
        "metric_id": metric_id,
        "evaluation": evaluation,
        "metric_summary": {
            "precision": metric.precision,
            "recall": metric.recall,
            "f1_score": metric.f1_score,
            "industry_type": metric.industry_type,
            "anomaly_type": metric.anomaly_type
        }
    }

@app.post("/detection-metrics/report", response_model=DetectionReport)
async def generate_detection_report(
    tenant_id: str,
    industry_type: IndustryType,
    start_date: datetime,
    end_date: datetime
):
    """Generate detection metrics report"""
    
    # Get metrics for the period
    period_metrics = [
        metric for metric in detection_metrics_store.values()
        if (metric.tenant_id == tenant_id and 
            metric.industry_type == industry_type and
            start_date <= metric.measurement_date <= end_date)
    ]
    
    # Group by anomaly type
    fraud_metrics = [m for m in period_metrics if m.anomaly_type == AnomalyType.FRAUD_TRANSACTION]
    revenue_metrics = [m for m in period_metrics if m.anomaly_type == AnomalyType.REVENUE_ANOMALY]
    behavior_metrics = [m for m in period_metrics if m.anomaly_type == AnomalyType.USER_BEHAVIOR]
    
    # Calculate overall metrics
    if period_metrics:
        overall_precision = sum([m.precision for m in period_metrics]) / len(period_metrics)
        overall_recall = sum([m.recall for m in period_metrics]) / len(period_metrics)
        overall_f1 = sum([m.f1_score for m in period_metrics]) / len(period_metrics)
        total_prevented_loss = sum([m.prevented_loss for m in period_metrics if m.prevented_loss])
    else:
        overall_precision = overall_recall = overall_f1 = total_prevented_loss = 0.0
    
    report = DetectionReport(
        tenant_id=tenant_id,
        industry_type=industry_type,
        start_date=start_date,
        end_date=end_date,
        fraud_metrics=fraud_metrics,
        revenue_metrics=revenue_metrics,
        behavior_metrics=behavior_metrics,
        overall_precision=round(overall_precision, 4),
        overall_recall=round(overall_recall, 4),
        overall_f1=round(overall_f1, 4),
        total_prevented_loss=total_prevented_loss
    )
    
    detection_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated detection report for tenant {tenant_id} - {industry_type}")
    return report

@app.get("/detection-metrics/industry-benchmarks")
async def get_industry_benchmarks():
    """Get industry benchmark thresholds"""
    
    return {
        "industry_thresholds": calculator.industry_thresholds,
        "description": "Minimum precision and recall thresholds by industry"
    }

@app.get("/detection-metrics/summary")
async def get_detection_summary():
    """Get summary of all detection metrics"""
    
    total_metrics = len(detection_metrics_store)
    
    # Group by industry
    industry_counts = {}
    for industry in IndustryType:
        industry_counts[industry.value] = len([
            m for m in detection_metrics_store.values() 
            if m.industry_type == industry
        ])
    
    # Group by anomaly type
    anomaly_counts = {}
    for anomaly_type in AnomalyType:
        anomaly_counts[anomaly_type.value] = len([
            m for m in detection_metrics_store.values() 
            if m.anomaly_type == anomaly_type
        ])
    
    # Calculate averages
    if detection_metrics_store:
        avg_precision = sum([m.precision for m in detection_metrics_store.values()]) / total_metrics
        avg_recall = sum([m.recall for m in detection_metrics_store.values()]) / total_metrics
        avg_f1 = sum([m.f1_score for m in detection_metrics_store.values()]) / total_metrics
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
    
    return {
        "total_metrics": total_metrics,
        "metrics_by_industry": industry_counts,
        "metrics_by_anomaly_type": anomaly_counts,
        "average_metrics": {
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "f1_score": round(avg_f1, 4)
        },
        "total_reports": len(detection_reports_store)
    }
