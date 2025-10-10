"""
Task 3.2.31: Backtest framework (out-of-time tests; economic shifts)
- Robustness across regimes
- Offline eval jobs
- Signed results
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
import pandas as pd
import json
import logging
import hashlib
import hmac
from dataclasses import dataclass

app = FastAPI(title="RBIA Backtest Framework")
logger = logging.getLogger(__name__)

class BacktestType(str, Enum):
    OUT_OF_TIME = "out_of_time"
    ECONOMIC_REGIME = "economic_regime"
    SEASONAL_VARIATION = "seasonal_variation"
    MARKET_STRESS = "market_stress"
    CROSS_VALIDATION = "cross_validation"

class EconomicRegime(str, Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    RECESSION = "recession"
    RECOVERY = "recovery"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    HIGH_INFLATION = "high_inflation"
    DEFLATION = "deflation"

class BacktestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"

class BacktestRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier to backtest")
    model_version: str = Field(..., description="Model version")
    tenant_id: str = Field(..., description="Tenant identifier")
    backtest_type: BacktestType = Field(..., description="Type of backtest to perform")
    start_date: datetime = Field(..., description="Start date for backtest period")
    end_date: datetime = Field(..., description="End date for backtest period")
    economic_regimes: Optional[List[EconomicRegime]] = Field(None, description="Economic regimes to test")
    metrics_to_evaluate: List[MetricType] = Field(..., description="Metrics to calculate")
    benchmark_model_id: Optional[str] = Field(None, description="Benchmark model for comparison")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for statistical tests")

class TimeSeriesSplit(BaseModel):
    split_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    regime: Optional[EconomicRegime] = None
    regime_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class MetricResult(BaseModel):
    metric_type: MetricType
    value: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    statistical_significance: Optional[float] = None
    benchmark_comparison: Optional[float] = None

class BacktestSplit(BaseModel):
    split: TimeSeriesSplit
    metrics: List[MetricResult]
    sample_size: int
    regime_stability: Optional[float] = Field(None, description="Stability of economic regime during period")
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality of data in this split")

class BacktestResult(BaseModel):
    backtest_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version: str
    tenant_id: str
    backtest_type: BacktestType
    status: BacktestStatus = BacktestStatus.PENDING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Results
    splits: List[BacktestSplit] = Field(default=[], description="Results for each time split")
    overall_metrics: List[MetricResult] = Field(default=[], description="Overall aggregated metrics")
    regime_performance: Dict[str, List[MetricResult]] = Field(default={}, description="Performance by economic regime")
    
    # Statistical analysis
    performance_stability: Optional[float] = Field(None, description="Stability of performance across time")
    regime_sensitivity: Optional[float] = Field(None, description="Sensitivity to economic regime changes")
    out_of_sample_degradation: Optional[float] = Field(None, description="Performance degradation out-of-sample")
    
    # Signature for tamper-evidence
    result_hash: Optional[str] = Field(None, description="Hash of results for integrity")
    signature: Optional[str] = Field(None, description="Digital signature of results")
    
    # Metadata
    total_samples: int = Field(default=0)
    execution_time_seconds: Optional[float] = None
    warnings: List[str] = Field(default=[], description="Warnings during backtesting")
    recommendations: List[str] = Field(default=[], description="Recommendations based on results")

# In-memory storage for backtest results (replace with actual database)
backtest_results: Dict[str, BacktestResult] = {}

# Economic regime detection parameters
REGIME_INDICATORS = {
    "market_volatility": {"bull": (0, 0.15), "bear": (0.25, 1.0), "high_vol": (0.3, 1.0)},
    "market_return": {"bull": (0.1, 1.0), "bear": (-1.0, -0.1), "recession": (-1.0, -0.2)},
    "inflation_rate": {"high_inflation": (0.04, 1.0), "deflation": (-1.0, 0.0)},
    "interest_rate": {"rising": (0.02, 1.0), "falling": (-1.0, 0.02)}
}

@app.post("/backtest/start", response_model=BacktestResult)
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Start a comprehensive backtest of the model
    """
    backtest_result = BacktestResult(
        model_id=request.model_id,
        model_version=request.model_version,
        tenant_id=request.tenant_id,
        backtest_type=request.backtest_type,
        status=BacktestStatus.RUNNING
    )
    
    backtest_results[backtest_result.backtest_id] = backtest_result
    
    # Start background backtesting
    background_tasks.add_task(
        execute_backtest,
        backtest_result.backtest_id,
        request
    )
    
    return backtest_result

async def execute_backtest(backtest_id: str, request: BacktestRequest):
    """Execute backtest in background"""
    try:
        backtest_result = backtest_results[backtest_id]
        start_time = datetime.utcnow()
        
        # Generate time series splits
        splits = generate_time_series_splits(
            request.start_date,
            request.end_date,
            request.backtest_type,
            request.economic_regimes
        )
        
        # Execute backtest for each split
        for split in splits:
            split_result = await execute_split_backtest(split, request)
            backtest_result.splits.append(split_result)
        
        # Calculate overall metrics
        backtest_result.overall_metrics = calculate_overall_metrics(
            backtest_result.splits,
            request.metrics_to_evaluate
        )
        
        # Analyze regime performance
        backtest_result.regime_performance = analyze_regime_performance(
            backtest_result.splits
        )
        
        # Calculate stability metrics
        backtest_result.performance_stability = calculate_performance_stability(
            backtest_result.splits
        )
        backtest_result.regime_sensitivity = calculate_regime_sensitivity(
            backtest_result.splits
        )
        backtest_result.out_of_sample_degradation = calculate_oos_degradation(
            backtest_result.splits
        )
        
        # Generate warnings and recommendations
        backtest_result.warnings = generate_warnings(backtest_result)
        backtest_result.recommendations = generate_recommendations(backtest_result)
        
        # Calculate totals
        backtest_result.total_samples = sum(split.sample_size for split in backtest_result.splits)
        backtest_result.execution_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        
        # Sign results for tamper-evidence
        sign_backtest_results(backtest_result)
        
        backtest_result.status = BacktestStatus.COMPLETED
        backtest_result.completed_at = datetime.utcnow()
        
        logger.info(f"Backtest completed: {backtest_id} - Performance stability: {backtest_result.performance_stability}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {backtest_id} - {e}")
        backtest_result.status = BacktestStatus.FAILED
        backtest_result.warnings.append(f"Backtest execution failed: {str(e)}")

def generate_time_series_splits(
    start_date: datetime,
    end_date: datetime,
    backtest_type: BacktestType,
    economic_regimes: Optional[List[EconomicRegime]] = None
) -> List[TimeSeriesSplit]:
    """Generate time series splits for backtesting"""
    splits = []
    
    if backtest_type == BacktestType.OUT_OF_TIME:
        # Generate expanding window splits
        total_days = (end_date - start_date).days
        min_train_days = 365  # Minimum 1 year of training data
        test_period_days = 90  # 3 months test periods
        
        current_date = start_date + timedelta(days=min_train_days)
        
        while current_date + timedelta(days=test_period_days) <= end_date:
            split = TimeSeriesSplit(
                train_start=start_date,
                train_end=current_date,
                test_start=current_date + timedelta(days=1),
                test_end=current_date + timedelta(days=test_period_days)
            )
            
            # Detect economic regime for this period
            split.regime = detect_economic_regime(split.test_start, split.test_end)
            split.regime_confidence = np.random.uniform(0.7, 0.95)  # Simulate regime confidence
            
            splits.append(split)
            current_date += timedelta(days=test_period_days)
    
    elif backtest_type == BacktestType.ECONOMIC_REGIME:
        # Generate splits based on specific economic regimes
        if economic_regimes:
            for regime in economic_regimes:
                regime_periods = identify_regime_periods(start_date, end_date, regime)
                for period_start, period_end in regime_periods:
                    train_end = period_start - timedelta(days=1)
                    if train_end > start_date + timedelta(days=365):  # Ensure minimum training period
                        split = TimeSeriesSplit(
                            train_start=start_date,
                            train_end=train_end,
                            test_start=period_start,
                            test_end=period_end,
                            regime=regime,
                            regime_confidence=0.9
                        )
                        splits.append(split)
    
    return splits

def detect_economic_regime(start_date: datetime, end_date: datetime) -> EconomicRegime:
    """Detect economic regime for a given period"""
    # Simulate regime detection based on market indicators
    # In production, this would analyze actual economic data
    
    regimes = list(EconomicRegime)
    weights = [0.2, 0.2, 0.1, 0.15, 0.1, 0.1, 0.1, 0.05]  # Probability weights
    
    return np.random.choice(regimes, p=weights)

def identify_regime_periods(
    start_date: datetime,
    end_date: datetime,
    regime: EconomicRegime
) -> List[tuple[datetime, datetime]]:
    """Identify periods when a specific economic regime was active"""
    periods = []
    
    # Simulate regime periods
    current_date = start_date
    while current_date < end_date:
        # Random regime period length (30-180 days)
        period_length = np.random.randint(30, 180)
        period_end = min(current_date + timedelta(days=period_length), end_date)
        
        # 30% chance this period matches the target regime
        if np.random.random() < 0.3:
            periods.append((current_date, period_end))
        
        current_date = period_end + timedelta(days=1)
    
    return periods

async def execute_split_backtest(
    split: TimeSeriesSplit,
    request: BacktestRequest
) -> BacktestSplit:
    """Execute backtest for a single time split"""
    
    # Simulate model evaluation on the split
    sample_size = np.random.randint(1000, 10000)  # Simulate sample size
    
    metrics = []
    for metric_type in request.metrics_to_evaluate:
        metric_value = simulate_metric_calculation(metric_type, split.regime)
        
        metric_result = MetricResult(
            metric_type=metric_type,
            value=metric_value,
            confidence_interval_lower=metric_value - 0.05,
            confidence_interval_upper=metric_value + 0.05,
            statistical_significance=np.random.uniform(0.01, 0.05)
        )
        metrics.append(metric_result)
    
    # Simulate data quality assessment
    data_quality_score = np.random.uniform(0.8, 1.0)
    
    # Assess regime stability
    regime_stability = np.random.uniform(0.6, 0.95) if split.regime else None
    
    return BacktestSplit(
        split=split,
        metrics=metrics,
        sample_size=sample_size,
        regime_stability=regime_stability,
        data_quality_score=data_quality_score
    )

def simulate_metric_calculation(metric_type: MetricType, regime: Optional[EconomicRegime]) -> float:
    """Simulate metric calculation with regime-dependent performance"""
    
    base_performance = {
        MetricType.ACCURACY: 0.85,
        MetricType.PRECISION: 0.82,
        MetricType.RECALL: 0.78,
        MetricType.F1_SCORE: 0.80,
        MetricType.AUC_ROC: 0.88,
        MetricType.MSE: 0.15,
        MetricType.MAE: 0.12,
        MetricType.RMSE: 0.18,
        MetricType.SHARPE_RATIO: 1.2,
        MetricType.MAX_DRAWDOWN: 0.08
    }
    
    base_value = base_performance.get(metric_type, 0.8)
    
    # Adjust based on economic regime
    regime_adjustments = {
        EconomicRegime.BULL_MARKET: 0.05,
        EconomicRegime.BEAR_MARKET: -0.10,
        EconomicRegime.RECESSION: -0.15,
        EconomicRegime.HIGH_VOLATILITY: -0.08,
        EconomicRegime.LOW_VOLATILITY: 0.03
    }
    
    adjustment = regime_adjustments.get(regime, 0.0)
    
    # Add noise
    noise = np.random.normal(0, 0.02)
    
    final_value = base_value + adjustment + noise
    
    # Ensure reasonable bounds
    if metric_type in [MetricType.MSE, MetricType.MAE, MetricType.RMSE, MetricType.MAX_DRAWDOWN]:
        return max(0.01, final_value)
    else:
        return max(0.0, min(1.0, final_value))

def calculate_overall_metrics(
    splits: List[BacktestSplit],
    metric_types: List[MetricType]
) -> List[MetricResult]:
    """Calculate overall aggregated metrics across all splits"""
    overall_metrics = []
    
    for metric_type in metric_types:
        values = []
        for split in splits:
            for metric in split.metrics:
                if metric.metric_type == metric_type:
                    values.append(metric.value)
        
        if values:
            overall_metric = MetricResult(
                metric_type=metric_type,
                value=np.mean(values),
                confidence_interval_lower=np.percentile(values, 2.5),
                confidence_interval_upper=np.percentile(values, 97.5),
                statistical_significance=None  # Would calculate proper statistical test
            )
            overall_metrics.append(overall_metric)
    
    return overall_metrics

def analyze_regime_performance(splits: List[BacktestSplit]) -> Dict[str, List[MetricResult]]:
    """Analyze performance by economic regime"""
    regime_performance = {}
    
    # Group splits by regime
    regime_splits = {}
    for split in splits:
        if split.split.regime:
            regime_name = split.split.regime.value
            if regime_name not in regime_splits:
                regime_splits[regime_name] = []
            regime_splits[regime_name].append(split)
    
    # Calculate metrics for each regime
    for regime_name, regime_split_list in regime_splits.items():
        regime_metrics = []
        
        # Get all metric types from first split
        if regime_split_list:
            metric_types = [m.metric_type for m in regime_split_list[0].metrics]
            
            for metric_type in metric_types:
                values = []
                for split in regime_split_list:
                    for metric in split.metrics:
                        if metric.metric_type == metric_type:
                            values.append(metric.value)
                
                if values:
                    regime_metric = MetricResult(
                        metric_type=metric_type,
                        value=np.mean(values),
                        confidence_interval_lower=np.percentile(values, 2.5),
                        confidence_interval_upper=np.percentile(values, 97.5)
                    )
                    regime_metrics.append(regime_metric)
            
            regime_performance[regime_name] = regime_metrics
    
    return regime_performance

def calculate_performance_stability(splits: List[BacktestSplit]) -> float:
    """Calculate stability of performance across time splits"""
    if len(splits) < 2:
        return 1.0
    
    # Calculate coefficient of variation for primary metric (assume first metric)
    if not splits[0].metrics:
        return 1.0
    
    primary_metric_type = splits[0].metrics[0].metric_type
    values = []
    
    for split in splits:
        for metric in split.metrics:
            if metric.metric_type == primary_metric_type:
                values.append(metric.value)
                break
    
    if not values or np.mean(values) == 0:
        return 1.0
    
    cv = np.std(values) / np.mean(values)
    stability = max(0.0, 1.0 - cv)  # Higher stability = lower coefficient of variation
    
    return stability

def calculate_regime_sensitivity(splits: List[BacktestSplit]) -> float:
    """Calculate sensitivity to economic regime changes"""
    regime_performances = {}
    
    for split in splits:
        if split.split.regime and split.metrics:
            regime = split.split.regime.value
            primary_metric_value = split.metrics[0].value
            
            if regime not in regime_performances:
                regime_performances[regime] = []
            regime_performances[regime].append(primary_metric_value)
    
    if len(regime_performances) < 2:
        return 0.0  # No regime sensitivity if only one regime
    
    # Calculate variance across regime mean performances
    regime_means = [np.mean(values) for values in regime_performances.values()]
    sensitivity = np.std(regime_means) / np.mean(regime_means) if np.mean(regime_means) > 0 else 0.0
    
    return min(1.0, sensitivity)

def calculate_oos_degradation(splits: List[BacktestSplit]) -> float:
    """Calculate out-of-sample performance degradation"""
    if len(splits) < 2:
        return 0.0
    
    # Compare first split (earliest) with last split (latest)
    if not splits[0].metrics or not splits[-1].metrics:
        return 0.0
    
    early_performance = splits[0].metrics[0].value
    late_performance = splits[-1].metrics[0].value
    
    if early_performance == 0:
        return 0.0
    
    degradation = (early_performance - late_performance) / early_performance
    return max(0.0, degradation)

def generate_warnings(backtest_result: BacktestResult) -> List[str]:
    """Generate warnings based on backtest results"""
    warnings = []
    
    if backtest_result.performance_stability and backtest_result.performance_stability < 0.7:
        warnings.append("LOW STABILITY: Model performance is unstable across time periods")
    
    if backtest_result.regime_sensitivity and backtest_result.regime_sensitivity > 0.3:
        warnings.append("HIGH REGIME SENSITIVITY: Model performance varies significantly across economic regimes")
    
    if backtest_result.out_of_sample_degradation and backtest_result.out_of_sample_degradation > 0.15:
        warnings.append("PERFORMANCE DEGRADATION: Significant degradation in out-of-sample performance")
    
    # Check data quality
    low_quality_splits = [s for s in backtest_result.splits if s.data_quality_score < 0.8]
    if len(low_quality_splits) > len(backtest_result.splits) * 0.3:
        warnings.append("DATA QUALITY: More than 30% of splits have low data quality scores")
    
    return warnings

def generate_recommendations(backtest_result: BacktestResult) -> List[str]:
    """Generate recommendations based on backtest results"""
    recommendations = []
    
    if backtest_result.performance_stability and backtest_result.performance_stability < 0.8:
        recommendations.append("Consider ensemble methods or regularization to improve stability")
    
    if backtest_result.regime_sensitivity and backtest_result.regime_sensitivity > 0.2:
        recommendations.append("Implement regime-aware modeling or feature engineering")
    
    if backtest_result.out_of_sample_degradation and backtest_result.out_of_sample_degradation > 0.1:
        recommendations.append("Review model complexity and consider retraining frequency")
    
    # Regime-specific recommendations
    for regime, metrics in backtest_result.regime_performance.items():
        if metrics and metrics[0].value < 0.7:  # Assuming first metric is primary
            recommendations.append(f"Poor performance in {regime} regime - consider regime-specific adjustments")
    
    return recommendations

def sign_backtest_results(backtest_result: BacktestResult):
    """Sign backtest results for tamper-evidence"""
    # Create hash of results
    results_dict = {
        "model_id": backtest_result.model_id,
        "model_version": backtest_result.model_version,
        "backtest_type": backtest_result.backtest_type.value,
        "overall_metrics": [
            {"type": m.metric_type.value, "value": m.value} 
            for m in backtest_result.overall_metrics
        ],
        "performance_stability": backtest_result.performance_stability,
        "regime_sensitivity": backtest_result.regime_sensitivity,
        "total_samples": backtest_result.total_samples
    }
    
    results_json = json.dumps(results_dict, sort_keys=True)
    backtest_result.result_hash = hashlib.sha256(results_json.encode()).hexdigest()
    
    # Create HMAC signature (in production, use proper key management)
    secret_key = "backtest_signing_key"  # Replace with actual secret
    backtest_result.signature = hmac.new(
        secret_key.encode(),
        results_json.encode(),
        hashlib.sha256
    ).hexdigest()

@app.get("/backtest/{backtest_id}", response_model=BacktestResult)
async def get_backtest_result(backtest_id: str):
    """Get backtest results"""
    if backtest_id not in backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    return backtest_results[backtest_id]

@app.get("/backtest/{backtest_id}/verify")
async def verify_backtest_integrity(backtest_id: str):
    """Verify integrity of backtest results"""
    if backtest_id not in backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    backtest_result = backtest_results[backtest_id]
    
    if not backtest_result.result_hash or not backtest_result.signature:
        return {"verified": False, "reason": "Results not signed"}
    
    # Recreate hash
    results_dict = {
        "model_id": backtest_result.model_id,
        "model_version": backtest_result.model_version,
        "backtest_type": backtest_result.backtest_type.value,
        "overall_metrics": [
            {"type": m.metric_type.value, "value": m.value} 
            for m in backtest_result.overall_metrics
        ],
        "performance_stability": backtest_result.performance_stability,
        "regime_sensitivity": backtest_result.regime_sensitivity,
        "total_samples": backtest_result.total_samples
    }
    
    results_json = json.dumps(results_dict, sort_keys=True)
    expected_hash = hashlib.sha256(results_json.encode()).hexdigest()
    
    # Verify signature
    secret_key = "backtest_signing_key"
    expected_signature = hmac.new(
        secret_key.encode(),
        results_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    hash_valid = backtest_result.result_hash == expected_hash
    signature_valid = backtest_result.signature == expected_signature
    
    return {
        "verified": hash_valid and signature_valid,
        "hash_valid": hash_valid,
        "signature_valid": signature_valid,
        "result_hash": backtest_result.result_hash,
        "expected_hash": expected_hash
    }

@app.get("/backtests")
async def list_backtests(
    tenant_id: Optional[str] = None,
    model_id: Optional[str] = None,
    status: Optional[BacktestStatus] = None,
    limit: int = 50
):
    """List backtests with filtering"""
    filtered_results = list(backtest_results.values())
    
    if tenant_id:
        filtered_results = [r for r in filtered_results if r.tenant_id == tenant_id]
    
    if model_id:
        filtered_results = [r for r in filtered_results if r.model_id == model_id]
    
    if status:
        filtered_results = [r for r in filtered_results if r.status == status]
    
    # Sort by start time, most recent first
    filtered_results.sort(key=lambda x: x.started_at, reverse=True)
    
    return {
        "backtests": filtered_results[:limit],
        "total_count": len(filtered_results)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Backtest Framework",
        "task": "3.2.31",
        "total_backtests": len(backtest_results),
        "backtest_types_supported": [bt.value for bt in BacktestType],
        "economic_regimes_supported": [er.value for er in EconomicRegime],
        "metrics_supported": [mt.value for mt in MetricType]
    }

