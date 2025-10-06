"""
Task 3.2.9: Trust score computation
- Trust score = quality × explainability × drift × bias × ops SLO
- Comparable risk indicator per node
- Drives gates & UI badges
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import statistics
import math
from dataclasses import dataclass

app = FastAPI(title="RBIA Trust Score Service")

class TrustComponent(str, Enum):
    QUALITY = "quality"
    EXPLAINABILITY = "explainability"
    DRIFT = "drift"
    BIAS = "bias"
    OPS_SLO = "ops_slo"

class TrustLevel(str, Enum):
    EXCELLENT = "excellent"  # 0.9 - 1.0
    GOOD = "good"           # 0.7 - 0.89
    ACCEPTABLE = "acceptable" # 0.5 - 0.69
    POOR = "poor"           # 0.3 - 0.49
    CRITICAL = "critical"   # 0.0 - 0.29

class ComponentMetric(BaseModel):
    component: TrustComponent = Field(..., description="Trust component type")
    score: float = Field(..., ge=0.0, le=1.0, description="Component score (0-1)")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Component weight")
    metrics: Dict[str, float] = Field(default={}, description="Underlying metrics")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in score")

class TrustScore(BaseModel):
    trust_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_type: str = Field(..., description="Type of target (model, node, workflow)")
    target_id: str = Field(..., description="Target identifier")
    target_name: str = Field(..., description="Human-readable target name")
    tenant_id: str = Field(..., description="Tenant identifier")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall trust score")
    trust_level: TrustLevel = Field(..., description="Trust level category")
    components: List[ComponentMetric] = Field(..., description="Component scores")
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Score expiration time")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

class TrustScoreRequest(BaseModel):
    target_type: str = Field(..., description="Type of target")
    target_id: str = Field(..., description="Target identifier")
    target_name: str = Field(..., description="Target name")
    tenant_id: str = Field(..., description="Tenant identifier")
    component_weights: Dict[str, float] = Field(default={}, description="Custom component weights")
    force_refresh: bool = Field(default=False, description="Force score recalculation")

class TrustGate(BaseModel):
    gate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_type: str = Field(..., description="Target type")
    target_id: str = Field(..., description="Target identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    minimum_trust_score: float = Field(..., ge=0.0, le=1.0, description="Minimum required trust score")
    required_components: Dict[str, float] = Field(default={}, description="Minimum scores per component")
    gate_type: str = Field(..., description="Gate type (deployment, execution, etc.)")
    active: bool = Field(default=True, description="Whether gate is active")

# In-memory storage (replace with actual database)
trust_scores: Dict[str, TrustScore] = {}
trust_gates: Dict[str, TrustGate] = {}
component_history: Dict[str, List[ComponentMetric]] = {}

# Default component weights
DEFAULT_WEIGHTS = {
    TrustComponent.QUALITY.value: 0.25,
    TrustComponent.EXPLAINABILITY.value: 0.20,
    TrustComponent.DRIFT.value: 0.20,
    TrustComponent.BIAS.value: 0.20,
    TrustComponent.OPS_SLO.value: 0.15
}

@app.post("/trust-scores/compute", response_model=TrustScore)
async def compute_trust_score(request: TrustScoreRequest, background_tasks: BackgroundTasks):
    """
    Compute trust score for a target (model/node/workflow)
    """
    score_key = f"{request.target_type}_{request.target_id}_{request.tenant_id}"
    
    # Check if we have a recent score and don't need to refresh
    if not request.force_refresh and score_key in trust_scores:
        existing_score = trust_scores[score_key]
        if existing_score.computed_at > datetime.utcnow() - timedelta(minutes=30):
            return existing_score
    
    # Compute component scores
    components = await compute_component_scores(
        request.target_type, request.target_id, request.tenant_id
    )
    
    # Apply custom weights if provided
    weights = DEFAULT_WEIGHTS.copy()
    weights.update(request.component_weights)
    
    # Calculate weighted overall score
    weighted_sum = 0.0
    total_weight = 0.0
    
    for component in components:
        weight = weights.get(component.component.value, 1.0)
        component.weight = weight
        weighted_sum += component.score * weight
        total_weight += weight
    
    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Determine trust level
    trust_level = determine_trust_level(overall_score)
    
    # Create trust score
    trust_score = TrustScore(
        target_type=request.target_type,
        target_id=request.target_id,
        target_name=request.target_name,
        tenant_id=request.tenant_id,
        overall_score=overall_score,
        trust_level=trust_level,
        components=components,
        expires_at=datetime.utcnow() + timedelta(hours=1),  # Cache for 1 hour
        metadata={
            "weights_used": weights,
            "computation_method": "weighted_average",
            "component_count": len(components)
        }
    )
    
    # Store score
    trust_scores[score_key] = trust_score
    
    # Store component history
    for component in components:
        history_key = f"{score_key}_{component.component.value}"
        if history_key not in component_history:
            component_history[history_key] = []
        component_history[history_key].append(component)
        
        # Keep only last 100 entries
        component_history[history_key] = component_history[history_key][-100:]
    
    # Schedule background tasks for monitoring
    background_tasks.add_task(check_trust_gates, trust_score)
    
    return trust_score

async def compute_component_scores(target_type: str, target_id: str, tenant_id: str) -> List[ComponentMetric]:
    """
    Compute individual component scores
    """
    components = []
    
    # Quality Score (based on model performance metrics)
    quality_score = await compute_quality_score(target_type, target_id, tenant_id)
    components.append(quality_score)
    
    # Explainability Score (based on SHAP/LIME availability and quality)
    explainability_score = await compute_explainability_score(target_type, target_id, tenant_id)
    components.append(explainability_score)
    
    # Drift Score (based on data and prediction drift)
    drift_score = await compute_drift_score(target_type, target_id, tenant_id)
    components.append(drift_score)
    
    # Bias Score (based on fairness metrics)
    bias_score = await compute_bias_score(target_type, target_id, tenant_id)
    components.append(bias_score)
    
    # Ops SLO Score (based on operational metrics)
    ops_slo_score = await compute_ops_slo_score(target_type, target_id, tenant_id)
    components.append(ops_slo_score)
    
    return components

async def compute_quality_score(target_type: str, target_id: str, tenant_id: str) -> ComponentMetric:
    """
    Compute quality score based on model performance
    """
    # Simulate quality metrics (in real implementation, fetch from model registry/monitoring)
    metrics = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "auc_roc": 0.89,
        "calibration_error": 0.05
    }
    
    # Weighted quality score calculation
    quality_weights = {
        "accuracy": 0.25,
        "precision": 0.20,
        "recall": 0.20,
        "f1_score": 0.15,
        "auc_roc": 0.15,
        "calibration_error": 0.05  # Lower is better, so invert
    }
    
    weighted_score = 0.0
    for metric, value in metrics.items():
        weight = quality_weights.get(metric, 0.0)
        if metric == "calibration_error":
            # Lower calibration error is better
            score_contribution = (1.0 - value) * weight
        else:
            score_contribution = value * weight
        weighted_score += score_contribution
    
    return ComponentMetric(
        component=TrustComponent.QUALITY,
        score=min(1.0, max(0.0, weighted_score)),
        metrics=metrics,
        confidence=0.95
    )

async def compute_explainability_score(target_type: str, target_id: str, tenant_id: str) -> ComponentMetric:
    """
    Compute explainability score based on explanation quality
    """
    # Simulate explainability metrics
    metrics = {
        "shap_available": 1.0,  # SHAP explanations available
        "lime_available": 1.0,  # LIME explanations available
        "feature_importance_quality": 0.88,  # Quality of feature importance
        "explanation_coverage": 0.92,  # Percentage of predictions with explanations
        "explanation_stability": 0.85,  # Consistency of explanations
        "human_interpretability": 0.80  # Human interpretability score
    }
    
    # Calculate explainability score
    base_score = (
        metrics["shap_available"] * 0.2 +
        metrics["lime_available"] * 0.2 +
        metrics["feature_importance_quality"] * 0.2 +
        metrics["explanation_coverage"] * 0.15 +
        metrics["explanation_stability"] * 0.15 +
        metrics["human_interpretability"] * 0.1
    )
    
    return ComponentMetric(
        component=TrustComponent.EXPLAINABILITY,
        score=min(1.0, max(0.0, base_score)),
        metrics=metrics,
        confidence=0.90
    )

async def compute_drift_score(target_type: str, target_id: str, tenant_id: str) -> ComponentMetric:
    """
    Compute drift score based on data and prediction drift
    """
    # Simulate drift metrics
    metrics = {
        "data_drift_psi": 0.15,  # Population Stability Index (lower is better)
        "prediction_drift_ks": 0.08,  # Kolmogorov-Smirnov statistic (lower is better)
        "feature_drift_count": 2,  # Number of features with significant drift
        "total_features": 20,  # Total number of features
        "drift_detection_frequency": 1.0,  # How often drift is checked
        "time_since_last_retrain": 7  # Days since last retrain
    }
    
    # Calculate drift score (higher is better, so invert drift metrics)
    psi_score = max(0.0, 1.0 - (metrics["data_drift_psi"] / 0.25))  # PSI > 0.25 is concerning
    ks_score = max(0.0, 1.0 - (metrics["prediction_drift_ks"] / 0.2))  # KS > 0.2 is concerning
    feature_drift_score = 1.0 - (metrics["feature_drift_count"] / metrics["total_features"])
    
    # Penalize if too long since retrain
    retrain_penalty = max(0.0, 1.0 - (metrics["time_since_last_retrain"] / 30))  # 30 days max
    
    drift_score = (psi_score * 0.3 + ks_score * 0.3 + feature_drift_score * 0.2 + retrain_penalty * 0.2)
    
    return ComponentMetric(
        component=TrustComponent.DRIFT,
        score=min(1.0, max(0.0, drift_score)),
        metrics=metrics,
        confidence=0.85
    )

async def compute_bias_score(target_type: str, target_id: str, tenant_id: str) -> ComponentMetric:
    """
    Compute bias score based on fairness metrics
    """
    # Simulate bias/fairness metrics
    metrics = {
        "demographic_parity": 0.92,  # Demographic parity score
        "equalized_odds": 0.88,  # Equalized odds score
        "individual_fairness": 0.85,  # Individual fairness score
        "disparate_impact": 0.90,  # Disparate impact ratio
        "statistical_parity": 0.87,  # Statistical parity score
        "bias_audit_passed": 1.0  # Whether bias audit passed
    }
    
    # Calculate bias score (weighted average of fairness metrics)
    bias_score = (
        metrics["demographic_parity"] * 0.2 +
        metrics["equalized_odds"] * 0.2 +
        metrics["individual_fairness"] * 0.15 +
        metrics["disparate_impact"] * 0.15 +
        metrics["statistical_parity"] * 0.15 +
        metrics["bias_audit_passed"] * 0.15
    )
    
    return ComponentMetric(
        component=TrustComponent.BIAS,
        score=min(1.0, max(0.0, bias_score)),
        metrics=metrics,
        confidence=0.88
    )

async def compute_ops_slo_score(target_type: str, target_id: str, tenant_id: str) -> ComponentMetric:
    """
    Compute operational SLO score
    """
    # Simulate operational metrics
    metrics = {
        "uptime_percentage": 99.5,  # Uptime percentage
        "avg_latency_ms": 150,  # Average latency in milliseconds
        "p95_latency_ms": 250,  # 95th percentile latency
        "error_rate_percentage": 0.5,  # Error rate percentage
        "throughput_rps": 1000,  # Requests per second
        "resource_utilization": 0.65,  # Resource utilization (0-1)
        "sla_compliance": 0.98  # SLA compliance rate
    }
    
    # Calculate ops score based on SLO targets
    uptime_score = metrics["uptime_percentage"] / 100.0
    latency_score = max(0.0, 1.0 - (metrics["p95_latency_ms"] - 200) / 300)  # Target: 200ms, max: 500ms
    error_score = max(0.0, 1.0 - (metrics["error_rate_percentage"] / 5.0))  # Target: <5% error rate
    sla_score = metrics["sla_compliance"]
    
    ops_slo_score = (
        uptime_score * 0.3 +
        latency_score * 0.25 +
        error_score * 0.25 +
        sla_score * 0.2
    )
    
    return ComponentMetric(
        component=TrustComponent.OPS_SLO,
        score=min(1.0, max(0.0, ops_slo_score)),
        metrics=metrics,
        confidence=0.95
    )

def determine_trust_level(score: float) -> TrustLevel:
    """
    Determine trust level based on overall score
    """
    if score >= 0.9:
        return TrustLevel.EXCELLENT
    elif score >= 0.7:
        return TrustLevel.GOOD
    elif score >= 0.5:
        return TrustLevel.ACCEPTABLE
    elif score >= 0.3:
        return TrustLevel.POOR
    else:
        return TrustLevel.CRITICAL

async def check_trust_gates(trust_score: TrustScore):
    """
    Check if trust score passes configured gates
    """
    gate_key = f"{trust_score.target_type}_{trust_score.target_id}_{trust_score.tenant_id}"
    
    if gate_key in trust_gates:
        gate = trust_gates[gate_key]
        
        # Check overall score
        if trust_score.overall_score < gate.minimum_trust_score:
            print(f"TRUST GATE FAILED: {trust_score.target_name} score {trust_score.overall_score:.3f} < {gate.minimum_trust_score:.3f}")
        
        # Check component requirements
        for component in trust_score.components:
            required_score = gate.required_components.get(component.component.value)
            if required_score and component.score < required_score:
                print(f"COMPONENT GATE FAILED: {component.component.value} score {component.score:.3f} < {required_score:.3f}")

@app.get("/trust-scores/{target_type}/{target_id}")
async def get_trust_score(target_type: str, target_id: str, tenant_id: str):
    """
    Get trust score for a target
    """
    score_key = f"{target_type}_{target_id}_{tenant_id}"
    
    if score_key not in trust_scores:
        raise HTTPException(status_code=404, detail="Trust score not found")
    
    trust_score = trust_scores[score_key]
    
    # Check if score is expired
    if trust_score.expires_at and trust_score.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Trust score expired, please recompute")
    
    return trust_score

@app.get("/trust-scores")
async def list_trust_scores(
    tenant_id: Optional[str] = None,
    target_type: Optional[str] = None,
    min_score: Optional[float] = None,
    trust_level: Optional[TrustLevel] = None
):
    """
    List trust scores with optional filtering
    """
    scores = list(trust_scores.values())
    
    if tenant_id:
        scores = [s for s in scores if s.tenant_id == tenant_id]
    
    if target_type:
        scores = [s for s in scores if s.target_type == target_type]
    
    if min_score is not None:
        scores = [s for s in scores if s.overall_score >= min_score]
    
    if trust_level:
        scores = [s for s in scores if s.trust_level == trust_level]
    
    return {
        "scores": [s.dict() for s in scores],
        "total_count": len(scores)
    }

@app.post("/trust-gates")
async def create_trust_gate(gate: TrustGate):
    """
    Create trust gate with minimum score requirements
    """
    gate_key = f"{gate.target_type}_{gate.target_id}_{gate.tenant_id}"
    trust_gates[gate_key] = gate
    
    return {
        "gate_id": gate.gate_id,
        "message": "Trust gate created successfully"
    }

@app.get("/trust-gates/{target_type}/{target_id}")
async def get_trust_gate(target_type: str, target_id: str, tenant_id: str):
    """
    Get trust gate configuration
    """
    gate_key = f"{target_type}_{target_id}_{tenant_id}"
    
    if gate_key not in trust_gates:
        raise HTTPException(status_code=404, detail="Trust gate not found")
    
    return trust_gates[gate_key]

@app.get("/component-history/{target_type}/{target_id}")
async def get_component_history(
    target_type: str, 
    target_id: str, 
    tenant_id: str,
    component: Optional[TrustComponent] = None,
    days: int = 7
):
    """
    Get historical component scores
    """
    score_key = f"{target_type}_{target_id}_{tenant_id}"
    
    if component:
        history_key = f"{score_key}_{component.value}"
        history = component_history.get(history_key, [])
    else:
        # Get all components
        history = []
        for comp in TrustComponent:
            history_key = f"{score_key}_{comp.value}"
            history.extend(component_history.get(history_key, []))
    
    # Filter by date range
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    recent_history = [h for h in history if h.last_updated >= cutoff_date]
    
    return {
        "target_type": target_type,
        "target_id": target_id,
        "component": component.value if component else "all",
        "history": [h.dict() for h in recent_history],
        "count": len(recent_history)
    }

@app.get("/trust-summary/{tenant_id}")
async def get_trust_summary(tenant_id: str):
    """
    Get trust score summary for tenant
    """
    tenant_scores = [s for s in trust_scores.values() if s.tenant_id == tenant_id]
    
    if not tenant_scores:
        return {
            "tenant_id": tenant_id,
            "total_targets": 0,
            "summary": {}
        }
    
    # Calculate summary statistics
    overall_scores = [s.overall_score for s in tenant_scores]
    trust_levels = [s.trust_level.value for s in tenant_scores]
    
    summary = {
        "total_targets": len(tenant_scores),
        "average_trust_score": statistics.mean(overall_scores),
        "median_trust_score": statistics.median(overall_scores),
        "min_trust_score": min(overall_scores),
        "max_trust_score": max(overall_scores),
        "trust_level_distribution": {
            level.value: trust_levels.count(level.value) for level in TrustLevel
        },
        "component_averages": {}
    }
    
    # Calculate component averages
    for component in TrustComponent:
        component_scores = []
        for score in tenant_scores:
            for comp in score.components:
                if comp.component == component:
                    component_scores.append(comp.score)
        
        if component_scores:
            summary["component_averages"][component.value] = {
                "average": statistics.mean(component_scores),
                "count": len(component_scores)
            }
    
    return {
        "tenant_id": tenant_id,
        "summary": summary
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Trust Score",
        "task": "3.2.9 - Trust Score Computation",
        "total_scores": len(trust_scores),
        "total_gates": len(trust_gates)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
