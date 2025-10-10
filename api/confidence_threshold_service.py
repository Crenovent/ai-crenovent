"""
Task 3.3.22: Confidence thresholds in UX → below threshold, Assisted always required
- Risk gating based on confidence scores
- Mode switching recommendations
- Confidence-based workflow controls
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Confidence Threshold Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"  # 0.9 - 1.0
    HIGH = "high"           # 0.8 - 0.89
    MEDIUM = "medium"       # 0.6 - 0.79
    LOW = "low"             # 0.4 - 0.59
    VERY_LOW = "very_low"   # 0.0 - 0.39

class ThresholdAction(str, Enum):
    ALLOW = "allow"
    REQUIRE_ASSISTED = "require_assisted"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"
    WARN = "warn"

class ConfidenceThreshold(BaseModel):
    threshold_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Threshold configuration
    threshold_name: str
    model_id: Optional[str] = None  # None = applies to all models
    workflow_type: Optional[str] = None  # None = applies to all workflows
    
    # Threshold values
    ui_led_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    assisted_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    block_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Actions
    below_ui_threshold_action: ThresholdAction = ThresholdAction.REQUIRE_ASSISTED
    below_assisted_threshold_action: ThresholdAction = ThresholdAction.REQUIRE_APPROVAL
    below_block_threshold_action: ThresholdAction = ThresholdAction.BLOCK
    
    # Metadata
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True

class ConfidenceCheck(BaseModel):
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    model_id: str
    workflow_id: Optional[str] = None
    
    # Confidence data
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    
    # Current and recommended modes
    current_mode: UXMode
    recommended_mode: UXMode
    
    # Threshold results
    threshold_met: bool
    required_action: ThresholdAction
    action_reason: str
    
    # Additional context
    feature_confidence: Dict[str, float] = Field(default_factory=dict)
    uncertainty_sources: List[str] = Field(default_factory=list)
    
    # Metadata
    checked_at: datetime = Field(default_factory=datetime.utcnow)

class ModeRecommendation(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Current context
    current_mode: UXMode
    confidence_score: float
    model_id: str
    
    # Recommendation
    recommended_mode: UXMode
    recommendation_reason: str
    confidence_in_recommendation: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Actions required
    requires_mode_switch: bool
    requires_approval: bool
    requires_explanation: bool
    
    # UI guidance
    user_message: str
    suggested_actions: List[str] = Field(default_factory=list)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with proper database in production)
confidence_thresholds: Dict[str, ConfidenceThreshold] = {}
confidence_checks: List[ConfidenceCheck] = []
mode_recommendations: List[ModeRecommendation] = []

def _determine_confidence_level(confidence_score: float) -> ConfidenceLevel:
    """Determine confidence level from numeric score."""
    if confidence_score >= 0.9:
        return ConfidenceLevel.VERY_HIGH
    elif confidence_score >= 0.8:
        return ConfidenceLevel.HIGH
    elif confidence_score >= 0.6:
        return ConfidenceLevel.MEDIUM
    elif confidence_score >= 0.4:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW

def _get_applicable_threshold(tenant_id: str, model_id: str, 
                            workflow_type: Optional[str] = None) -> Optional[ConfidenceThreshold]:
    """Get the most specific applicable threshold for the context."""
    
    applicable_thresholds = [
        t for t in confidence_thresholds.values()
        if t.tenant_id == tenant_id and t.active
    ]
    
    # Find most specific match
    best_match = None
    specificity_score = -1
    
    for threshold in applicable_thresholds:
        score = 0
        
        # Check model match
        if threshold.model_id:
            if threshold.model_id == model_id:
                score += 2
            else:
                continue  # Skip if model specified but doesn't match
        
        # Check workflow type match
        if threshold.workflow_type:
            if threshold.workflow_type == workflow_type:
                score += 1
            else:
                continue  # Skip if workflow type specified but doesn't match
        
        if score > specificity_score:
            specificity_score = score
            best_match = threshold
    
    return best_match

def _determine_required_action(confidence_score: float, 
                             threshold: ConfidenceThreshold,
                             current_mode: UXMode) -> tuple[ThresholdAction, str]:
    """Determine required action based on confidence and thresholds."""
    
    if confidence_score < threshold.block_threshold:
        return threshold.below_block_threshold_action, f"Confidence {confidence_score:.2f} below block threshold {threshold.block_threshold:.2f}"
    
    elif confidence_score < threshold.assisted_threshold:
        return threshold.below_assisted_threshold_action, f"Confidence {confidence_score:.2f} below assisted threshold {threshold.assisted_threshold:.2f}"
    
    elif confidence_score < threshold.ui_led_threshold and current_mode == UXMode.UI_LED:
        return threshold.below_ui_threshold_action, f"Confidence {confidence_score:.2f} below UI-led threshold {threshold.ui_led_threshold:.2f}"
    
    else:
        return ThresholdAction.ALLOW, f"Confidence {confidence_score:.2f} meets all thresholds"

def _recommend_mode(confidence_score: float, threshold: ConfidenceThreshold) -> UXMode:
    """Recommend appropriate mode based on confidence score."""
    
    if confidence_score >= threshold.ui_led_threshold:
        return UXMode.UI_LED
    elif confidence_score >= threshold.assisted_threshold:
        return UXMode.ASSISTED
    else:
        return UXMode.CONVERSATIONAL  # Most guided mode for low confidence

@app.post("/confidence/check", response_model=ConfidenceCheck)
async def check_confidence_threshold(
    tenant_id: str,
    model_id: str,
    confidence_score: float,
    current_mode: UXMode,
    workflow_id: Optional[str] = None,
    workflow_type: Optional[str] = None,
    feature_confidence: Optional[Dict[str, float]] = None
):
    """Check confidence against thresholds and determine required actions."""
    
    # Get applicable threshold
    threshold = _get_applicable_threshold(tenant_id, model_id, workflow_type)
    
    if not threshold:
        # Create default threshold if none exists
        threshold = ConfidenceThreshold(
            tenant_id=tenant_id,
            threshold_name="Default Threshold",
            created_by="system"
        )
        confidence_thresholds[threshold.threshold_id] = threshold
    
    # Determine confidence level
    confidence_level = _determine_confidence_level(confidence_score)
    
    # Determine required action
    required_action, action_reason = _determine_required_action(
        confidence_score, threshold, current_mode
    )
    
    # Recommend appropriate mode
    recommended_mode = _recommend_mode(confidence_score, threshold)
    
    # Determine if threshold is met
    threshold_met = required_action == ThresholdAction.ALLOW
    
    # Identify uncertainty sources
    uncertainty_sources = []
    if feature_confidence:
        low_confidence_features = [
            feature for feature, conf in feature_confidence.items()
            if conf < 0.5
        ]
        if low_confidence_features:
            uncertainty_sources.append(f"Low confidence features: {', '.join(low_confidence_features)}")
    
    if confidence_score < 0.7:
        uncertainty_sources.append("Overall model confidence below 70%")
    
    confidence_check = ConfidenceCheck(
        tenant_id=tenant_id,
        model_id=model_id,
        workflow_id=workflow_id,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        current_mode=current_mode,
        recommended_mode=recommended_mode,
        threshold_met=threshold_met,
        required_action=required_action,
        action_reason=action_reason,
        feature_confidence=feature_confidence or {},
        uncertainty_sources=uncertainty_sources
    )
    
    confidence_checks.append(confidence_check)
    
    logger.info(f"Confidence check: {confidence_score:.2f} -> {required_action.value} (threshold: {threshold.threshold_name})")
    
    return confidence_check

@app.post("/confidence/recommend-mode", response_model=ModeRecommendation)
async def recommend_mode_switch(
    tenant_id: str,
    user_id: str,
    model_id: str,
    confidence_score: float,
    current_mode: UXMode,
    workflow_context: Optional[Dict[str, Any]] = None
):
    """Generate mode recommendation based on confidence score."""
    
    # Get applicable threshold
    threshold = _get_applicable_threshold(tenant_id, model_id)
    
    if not threshold:
        threshold = ConfidenceThreshold(
            tenant_id=tenant_id,
            threshold_name="Default Threshold",
            created_by="system"
        )
    
    # Determine recommended mode
    recommended_mode = _recommend_mode(confidence_score, threshold)
    
    # Check if mode switch is required
    requires_mode_switch = recommended_mode != current_mode
    
    # Determine if approval is required
    requires_approval = confidence_score < threshold.assisted_threshold
    
    # Determine if explanation is required
    requires_explanation = confidence_score < threshold.ui_led_threshold
    
    # Generate user message
    if not requires_mode_switch:
        user_message = f"Current {current_mode.value} mode is appropriate for this confidence level ({confidence_score:.2f})"
        recommendation_reason = "Confidence level supports current mode"
    elif recommended_mode == UXMode.ASSISTED:
        user_message = f"For better accuracy, we recommend switching to Assisted mode (confidence: {confidence_score:.2f})"
        recommendation_reason = f"Confidence {confidence_score:.2f} below UI-led threshold {threshold.ui_led_threshold:.2f}"
    elif recommended_mode == UXMode.CONVERSATIONAL:
        user_message = f"Due to low confidence ({confidence_score:.2f}), Conversational mode with guidance is recommended"
        recommendation_reason = f"Confidence {confidence_score:.2f} below assisted threshold {threshold.assisted_threshold:.2f}"
    else:
        user_message = f"You can use {recommended_mode.value} mode with confidence level {confidence_score:.2f}"
        recommendation_reason = f"Confidence {confidence_score:.2f} meets requirements for {recommended_mode.value} mode"
    
    # Generate suggested actions
    suggested_actions = []
    if requires_mode_switch:
        suggested_actions.append(f"Switch to {recommended_mode.value} mode")
    if requires_explanation:
        suggested_actions.append("Review explanation details before proceeding")
    if requires_approval:
        suggested_actions.append("Seek approval from supervisor")
    if confidence_score < 0.5:
        suggested_actions.append("Consider manual review of inputs")
    
    recommendation = ModeRecommendation(
        tenant_id=tenant_id,
        user_id=user_id,
        current_mode=current_mode,
        confidence_score=confidence_score,
        model_id=model_id,
        recommended_mode=recommended_mode,
        recommendation_reason=recommendation_reason,
        confidence_in_recommendation=0.9,  # High confidence in our recommendation logic
        requires_mode_switch=requires_mode_switch,
        requires_approval=requires_approval,
        requires_explanation=requires_explanation,
        user_message=user_message,
        suggested_actions=suggested_actions
    )
    
    mode_recommendations.append(recommendation)
    
    logger.info(f"Mode recommendation: {current_mode.value} -> {recommended_mode.value} (confidence: {confidence_score:.2f})")
    
    return recommendation

@app.post("/confidence/thresholds", response_model=ConfidenceThreshold)
async def create_confidence_threshold(threshold: ConfidenceThreshold):
    """Create or update confidence threshold configuration."""
    
    confidence_thresholds[threshold.threshold_id] = threshold
    
    logger.info(f"Created confidence threshold {threshold.threshold_id}: {threshold.threshold_name}")
    return threshold

@app.get("/confidence/thresholds/{tenant_id}")
async def get_confidence_thresholds(tenant_id: str):
    """Get all confidence thresholds for a tenant."""
    
    tenant_thresholds = [
        t for t in confidence_thresholds.values()
        if t.tenant_id == tenant_id
    ]
    
    return tenant_thresholds

@app.get("/confidence/checks/{tenant_id}")
async def get_confidence_checks(
    tenant_id: str,
    model_id: Optional[str] = None,
    limit: int = 100
):
    """Get recent confidence checks for a tenant."""
    
    filtered_checks = [
        c for c in confidence_checks
        if c.tenant_id == tenant_id
        and (model_id is None or c.model_id == model_id)
    ]
    
    # Sort by check time (most recent first)
    filtered_checks.sort(key=lambda x: x.checked_at, reverse=True)
    
    return filtered_checks[:limit]

@app.get("/confidence/recommendations/{tenant_id}")
async def get_mode_recommendations(
    tenant_id: str,
    user_id: Optional[str] = None,
    limit: int = 50
):
    """Get recent mode recommendations for a tenant."""
    
    filtered_recommendations = [
        r for r in mode_recommendations
        if r.tenant_id == tenant_id
        and (user_id is None or r.user_id == user_id)
    ]
    
    # Sort by generation time (most recent first)
    filtered_recommendations.sort(key=lambda x: x.generated_at, reverse=True)
    
    return filtered_recommendations[:limit]

@app.post("/confidence/initialize-defaults")
async def initialize_default_thresholds(tenant_id: str, created_by: str):
    """Initialize default confidence thresholds for a tenant."""
    
    default_thresholds = [
        ConfidenceThreshold(
            tenant_id=tenant_id,
            threshold_name="Default Global Threshold",
            ui_led_threshold=0.8,
            assisted_threshold=0.6,
            block_threshold=0.3,
            created_by=created_by
        ),
        ConfidenceThreshold(
            tenant_id=tenant_id,
            threshold_name="High-Risk Model Threshold",
            workflow_type="risk_assessment",
            ui_led_threshold=0.9,
            assisted_threshold=0.7,
            block_threshold=0.4,
            created_by=created_by
        ),
        ConfidenceThreshold(
            tenant_id=tenant_id,
            threshold_name="Financial Model Threshold",
            workflow_type="financial_analysis",
            ui_led_threshold=0.85,
            assisted_threshold=0.65,
            block_threshold=0.35,
            created_by=created_by
        )
    ]
    
    created_ids = []
    for threshold in default_thresholds:
        confidence_thresholds[threshold.threshold_id] = threshold
        created_ids.append(threshold.threshold_id)
    
    logger.info(f"Initialized {len(default_thresholds)} default thresholds for tenant {tenant_id}")
    
    return {
        "status": "initialized",
        "tenant_id": tenant_id,
        "created_thresholds": created_ids,
        "total_thresholds": len(default_thresholds)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Confidence Threshold Service", "task": "3.3.22"}