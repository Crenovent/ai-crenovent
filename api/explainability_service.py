"""
Task 3.2.2 & 3.3.23: Enhanced Explainability service 
- SHAP/LIME explanations with inline viewer support
- Feature attribution and top drivers
- Reason codes generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging
import json
import numpy as np
from dataclasses import dataclass

app = FastAPI(title="RBIA Enhanced Explainability Service")
logger = logging.getLogger(__name__)

class ExplanationType(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    REASON_CODES = "reason_codes"
    FEATURE_IMPORTANCE = "feature_importance"

class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"

class FeatureAttribution(BaseModel):
    feature_name: str
    feature_value: Union[str, float, bool]
    feature_type: FeatureType
    attribution_value: float  # SHAP value or importance score
    absolute_attribution: float  # |attribution_value|
    contribution_percentage: float  # % contribution to final prediction
    rank: int  # Rank by absolute attribution (1 = most important)

class InlineExplanation(BaseModel):
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    prediction_value: float
    prediction_label: Optional[str] = None
    confidence_score: float
    
    # Top drivers for inline display
    top_positive_drivers: List[FeatureAttribution] = Field(default_factory=list)
    top_negative_drivers: List[FeatureAttribution] = Field(default_factory=list)
    
    # Complete attribution
    all_attributions: List[FeatureAttribution] = Field(default_factory=list)
    
    # Summary explanation
    explanation_text: str
    reason_codes: List[str] = Field(default_factory=list)
    
    # Metadata
    explanation_type: ExplanationType
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str

class SHAPExplanation(BaseModel):
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    instance_data: Dict[str, Any]
    
    # SHAP values
    shap_values: Dict[str, float]  # feature_name -> shap_value
    base_value: float  # Expected value when no features present
    
    # Feature details
    feature_attributions: List[FeatureAttribution]
    
    # Visualization data
    waterfall_data: List[Dict[str, Any]] = Field(default_factory=list)
    force_plot_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str

class LIMEExplanation(BaseModel):
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    instance_data: Dict[str, Any]
    
    # LIME explanations
    local_explanations: Dict[str, float]  # feature_name -> lime_score
    
    # Feature details
    feature_attributions: List[FeatureAttribution]
    
    # Local model info
    local_model_r2: float  # R² of local linear model
    intercept: float
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str

# In-memory stores (replace with proper database in production)
explanations_store: Dict[str, Union[SHAPExplanation, LIMEExplanation, InlineExplanation]] = {}

def _simulate_shap_values(feature_data: Dict[str, Any], model_id: str) -> Dict[str, float]:
    """Simulate SHAP value calculation for features."""
    
    # Mock SHAP values based on feature types and model
    shap_values = {}
    
    # Simulate different feature importance based on model type
    if "churn" in model_id.lower():
        # Churn model - customer behavior features are important
        feature_importance_map = {
            "days_since_last_activity": -0.15,
            "total_revenue": 0.12,
            "support_tickets": -0.08,
            "feature_usage_score": 0.10,
            "contract_length": 0.06,
            "discount_percentage": -0.04,
            "payment_delays": -0.07,
            "engagement_score": 0.09
        }
    elif "risk" in model_id.lower():
        # Risk model - financial and behavioral features
        feature_importance_map = {
            "credit_score": 0.18,
            "debt_to_income": -0.14,
            "payment_history": 0.11,
            "employment_length": 0.08,
            "loan_amount": -0.06,
            "collateral_value": 0.09,
            "previous_defaults": -0.12
        }
    else:
        # Generic model
        feature_importance_map = {
            "feature_1": 0.10,
            "feature_2": -0.08,
            "feature_3": 0.06,
            "feature_4": -0.04,
            "feature_5": 0.03
        }
    
    # Calculate SHAP values based on actual feature values
    for feature_name, feature_value in feature_data.items():
        base_importance = feature_importance_map.get(feature_name, np.random.uniform(-0.05, 0.05))
        
        # Adjust importance based on feature value
        if isinstance(feature_value, (int, float)):
            # Numerical features - scale by value
            normalized_value = min(max(feature_value / 100, -2), 2)  # Normalize to [-2, 2]
            shap_values[feature_name] = base_importance * normalized_value
        else:
            # Categorical features - use base importance
            shap_values[feature_name] = base_importance
    
    return shap_values

def _create_feature_attributions(shap_values: Dict[str, float], 
                               feature_data: Dict[str, Any]) -> List[FeatureAttribution]:
    """Create feature attribution objects from SHAP values."""
    
    attributions = []
    total_abs_attribution = sum(abs(v) for v in shap_values.values())
    
    for i, (feature_name, shap_value) in enumerate(sorted(shap_values.items(), 
                                                         key=lambda x: abs(x[1]), reverse=True)):
        feature_value = feature_data.get(feature_name, "Unknown")
        
        # Determine feature type
        if isinstance(feature_value, bool):
            feature_type = FeatureType.BOOLEAN
        elif isinstance(feature_value, (int, float)):
            feature_type = FeatureType.NUMERICAL
        else:
            feature_type = FeatureType.CATEGORICAL
        
        attribution = FeatureAttribution(
            feature_name=feature_name,
            feature_value=feature_value,
            feature_type=feature_type,
            attribution_value=shap_value,
            absolute_attribution=abs(shap_value),
            contribution_percentage=(abs(shap_value) / total_abs_attribution * 100) if total_abs_attribution > 0 else 0,
            rank=i + 1
        )
        attributions.append(attribution)
    
    return attributions

def _generate_explanation_text(prediction: float, top_drivers: List[FeatureAttribution], 
                             model_type: str = "generic") -> str:
    """Generate human-readable explanation text."""
    
    if not top_drivers:
        return f"Prediction: {prediction:.2f}. No feature attributions available."
    
    # Get top positive and negative drivers
    positive_drivers = [d for d in top_drivers[:3] if d.attribution_value > 0]
    negative_drivers = [d for d in top_drivers[:3] if d.attribution_value < 0]
    
    explanation_parts = [f"Prediction: {prediction:.2f}"]
    
    if positive_drivers:
        positive_text = ", ".join([
            f"{d.feature_name}={d.feature_value} (+{d.contribution_percentage:.1f}%)"
            for d in positive_drivers
        ])
        explanation_parts.append(f"Key positive factors: {positive_text}")
    
    if negative_drivers:
        negative_text = ", ".join([
            f"{d.feature_name}={d.feature_value} (-{d.contribution_percentage:.1f}%)"
            for d in negative_drivers
        ])
        explanation_parts.append(f"Key negative factors: {negative_text}")
    
    return ". ".join(explanation_parts) + "."

@app.post("/explanations/inline-explanation", response_model=InlineExplanation)
async def generate_inline_explanation(
    model_id: str,
    feature_data: Dict[str, Any],
    prediction_value: float,
    tenant_id: str,
    confidence_score: Optional[float] = None,
    top_k: int = 5
):
    """Generate inline explanation for immediate display in UX."""
    
    # Generate SHAP values
    shap_values = _simulate_shap_values(feature_data, model_id)
    
    # Create feature attributions
    all_attributions = _create_feature_attributions(shap_values, feature_data)
    
    # Get top positive and negative drivers
    positive_drivers = [attr for attr in all_attributions if attr.attribution_value > 0][:top_k]
    negative_drivers = [attr for attr in all_attributions if attr.attribution_value < 0][:top_k]
    
    # Generate explanation text
    explanation_text = _generate_explanation_text(prediction_value, all_attributions[:top_k], model_id)
    
    # Generate reason codes
    reason_codes = []
    for attr in all_attributions[:3]:  # Top 3 features
        if attr.attribution_value > 0:
            reason_codes.append(f"HIGH_{attr.feature_name.upper()}")
        else:
            reason_codes.append(f"LOW_{attr.feature_name.upper()}")
    
    inline_explanation = InlineExplanation(
        model_id=model_id,
        prediction_value=prediction_value,
        confidence_score=confidence_score or 0.85,
        top_positive_drivers=positive_drivers,
        top_negative_drivers=negative_drivers,
        all_attributions=all_attributions,
        explanation_text=explanation_text,
        reason_codes=reason_codes,
        explanation_type=ExplanationType.SHAP,
        tenant_id=tenant_id
    )
    
    explanations_store[inline_explanation.explanation_id] = inline_explanation
    
    logger.info(f"Generated inline explanation {inline_explanation.explanation_id} for model {model_id}")
    return inline_explanation

@app.post("/explanations/shap", response_model=SHAPExplanation)
async def generate_shap_explanation(
    model_id: str,
    instance_data: Dict[str, Any],
    tenant_id: str
):
    """Generate detailed SHAP explanation for ML node."""
    
    shap_values = _simulate_shap_values(instance_data, model_id)
    base_value = 0.5  # Simulated base value
    
    # Create feature attributions
    feature_attributions = _create_feature_attributions(shap_values, instance_data)
    
    # Create waterfall data for visualization
    waterfall_data = []
    cumulative_value = base_value
    
    for attr in sorted(feature_attributions, key=lambda x: abs(x.attribution_value), reverse=True):
        waterfall_data.append({
            "feature": attr.feature_name,
            "value": attr.feature_value,
            "shap_value": attr.attribution_value,
            "cumulative": cumulative_value + attr.attribution_value
        })
        cumulative_value += attr.attribution_value
    
    # Force plot data
    force_plot_data = {
        "base_value": base_value,
        "prediction": cumulative_value,
        "features": [
            {
                "name": attr.feature_name,
                "value": attr.feature_value,
                "effect": attr.attribution_value
            }
            for attr in feature_attributions
        ]
    }
    
    shap_explanation = SHAPExplanation(
        model_id=model_id,
        instance_data=instance_data,
        shap_values=shap_values,
        base_value=base_value,
        feature_attributions=feature_attributions,
        waterfall_data=waterfall_data,
        force_plot_data=force_plot_data,
        tenant_id=tenant_id
    )
    
    explanations_store[shap_explanation.explanation_id] = shap_explanation
    
    logger.info(f"Generated SHAP explanation {shap_explanation.explanation_id} for model {model_id}")
    return shap_explanation

@app.post("/explanations/lime", response_model=LIMEExplanation)
async def generate_lime_explanation(
    model_id: str,
    instance_data: Dict[str, Any],
    tenant_id: str
):
    """Generate LIME explanation for ML node."""
    
    # Simulate LIME local explanations (similar to SHAP but with different methodology)
    lime_scores = {}
    for feature_name, feature_value in instance_data.items():
        # LIME scores tend to be more localized
        base_score = np.random.uniform(-0.3, 0.3)
        if isinstance(feature_value, (int, float)):
            lime_scores[feature_name] = base_score * (feature_value / 50)  # Scale by value
        else:
            lime_scores[feature_name] = base_score
    
    # Create feature attributions
    feature_attributions = _create_feature_attributions(lime_scores, instance_data)
    
    lime_explanation = LIMEExplanation(
        model_id=model_id,
        instance_data=instance_data,
        local_explanations=lime_scores,
        feature_attributions=feature_attributions,
        local_model_r2=0.78,  # Simulated R²
        intercept=0.42,  # Simulated intercept
        tenant_id=tenant_id
    )
    
    explanations_store[lime_explanation.explanation_id] = lime_explanation
    
    logger.info(f"Generated LIME explanation {lime_explanation.explanation_id} for model {model_id}")
    return lime_explanation

@app.get("/explanations/{explanation_id}")
async def get_explanation(explanation_id: str):
    """Retrieve a stored explanation."""
    
    if explanation_id not in explanations_store:
        raise HTTPException(status_code=404, detail="Explanation not found")
    
    return explanations_store[explanation_id]

@app.get("/explanations/model/{model_id}")
async def get_model_explanations(
    model_id: str,
    tenant_id: str,
    limit: int = 10
):
    """Get recent explanations for a specific model."""
    
    model_explanations = [
        exp for exp in explanations_store.values()
        if exp.model_id == model_id and exp.tenant_id == tenant_id
    ]
    
    # Sort by generation time (most recent first)
    model_explanations.sort(key=lambda x: x.generated_at, reverse=True)
    
    return model_explanations[:limit]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Enhanced Explainability Service", "tasks": "3.2.2,3.3.23"}