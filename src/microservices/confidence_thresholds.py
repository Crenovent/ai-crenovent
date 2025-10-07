"""
AI Confidence Thresholds Service
Manages confidence thresholds and auto-fallback logic
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import uuid
from datetime import datetime

app = FastAPI(
    title="AI Confidence Thresholds Service",
    description="Manages confidence thresholds and auto-fallback logic",
    version="1.0.0"
)

class FallbackMode(str, Enum):
    UI = "ui"
    HYBRID = "hybrid"
    HUMAN_REVIEW = "human_review"
    RETRY = "retry"

class ConfidenceLevel(str, Enum):
    CRITICAL = "critical"  # < 0.3
    LOW = "low"           # 0.3 - 0.5
    MEDIUM = "medium"     # 0.5 - 0.8
    HIGH = "high"         # > 0.8

class ThresholdConfig(BaseModel):
    config_id: str
    tenant_id: str
    service_name: str
    operation_type: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    trust_threshold: float = Field(ge=0.0, le=1.0)
    fallback_mode: FallbackMode
    retry_count: int = Field(default=3, ge=0, le=10)
    explainability_required: bool = True
    human_review_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    auto_approve_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ConfidenceEvaluationRequest(BaseModel):
    tenant_id: str
    service_name: str
    operation_type: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Optional[Dict[str, Any]] = None

class ConfidenceEvaluationResponse(BaseModel):
    action: Literal["proceed", "fallback", "human_review", "retry"]
    fallback_mode: FallbackMode
    confidence_level: ConfidenceLevel
    explanation: str
    retry_count: int = 0
    max_retries: int = 3
    ui_metadata: Dict[str, Any] = Field(default_factory=dict)

class ExplainabilityRequest(BaseModel):
    tenant_id: str
    service_name: str
    operation_type: str
    confidence_score: float
    trust_score: float
    context: Dict[str, Any]

class ExplainabilityResponse(BaseModel):
    explanation: str
    factors: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_breakdown: Dict[str, float]

# In-memory storage (replace with database in production)
threshold_configs_db: Dict[str, ThresholdConfig] = {}

# Default threshold configurations
DEFAULT_CONFIGS = [
    {
        "config_id": "calendar_default",
        "tenant_id": "default",
        "service_name": "calendar",
        "operation_type": "create_event",
        "confidence_threshold": 0.8,
        "trust_threshold": 0.7,
        "fallback_mode": FallbackMode.HYBRID,
        "retry_count": 3,
        "explainability_required": True,
        "human_review_threshold": 0.3,
        "auto_approve_threshold": 0.9
    },
    {
        "config_id": "letsmeet_default",
        "tenant_id": "default",
        "service_name": "letsmeet",
        "operation_type": "transcribe_audio",
        "confidence_threshold": 0.7,
        "trust_threshold": 0.6,
        "fallback_mode": FallbackMode.HYBRID,
        "retry_count": 2,
        "explainability_required": True,
        "human_review_threshold": 0.4,
        "auto_approve_threshold": 0.85
    },
    {
        "config_id": "cruxx_default",
        "tenant_id": "default",
        "service_name": "cruxx",
        "operation_type": "create_action",
        "confidence_threshold": 0.9,
        "trust_threshold": 0.8,
        "fallback_mode": FallbackMode.HUMAN_REVIEW,
        "retry_count": 1,
        "explainability_required": True,
        "human_review_threshold": 0.5,
        "auto_approve_threshold": 0.95
    }
]

# Initialize default configurations
for config_data in DEFAULT_CONFIGS:
    config = ThresholdConfig(**config_data)
    threshold_configs_db[config.config_id] = config

def get_confidence_level(confidence_score: float) -> ConfidenceLevel:
    """Determine confidence level from score"""
    if confidence_score < 0.3:
        return ConfidenceLevel.CRITICAL
    elif confidence_score < 0.5:
        return ConfidenceLevel.LOW
    elif confidence_score < 0.8:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.HIGH

def find_threshold_config(tenant_id: str, service_name: str, operation_type: str) -> Optional[ThresholdConfig]:
    """Find threshold configuration for given parameters"""
    # First try tenant-specific config
    for config in threshold_configs_db.values():
        if (config.is_active and 
            config.tenant_id == tenant_id and
            config.service_name == service_name and
            config.operation_type == operation_type):
            return config
    
    # Fallback to default config
    for config in threshold_configs_db.values():
        if (config.is_active and 
            config.tenant_id == "default" and
            config.service_name == service_name and
            config.operation_type == operation_type):
            return config
    
    return None

@app.post("/confidence/evaluate", response_model=ConfidenceEvaluationResponse)
async def evaluate_confidence(request: ConfidenceEvaluationRequest):
    """Evaluate confidence and determine action"""
    
    # Find threshold configuration
    config = find_threshold_config(
        request.tenant_id,
        request.service_name,
        request.operation_type
    )
    
    if not config:
        # Default evaluation if no config found
        if request.confidence_score >= 0.8 and request.trust_score >= 0.7:
            action = "proceed"
            fallback_mode = FallbackMode.UI
            explanation = "High confidence and trust scores - proceeding with automation"
        elif request.confidence_score >= 0.5:
            action = "fallback"
            fallback_mode = FallbackMode.HYBRID
            explanation = "Medium confidence - falling back to hybrid mode"
        else:
            action = "human_review"
            fallback_mode = FallbackMode.HUMAN_REVIEW
            explanation = "Low confidence - requiring human review"
    else:
        # Use configured thresholds
        if (request.confidence_score >= config.confidence_threshold and 
            request.trust_score >= config.trust_threshold):
            action = "proceed"
            fallback_mode = config.fallback_mode
            explanation = f"Confidence ({request.confidence_score:.2f}) and trust ({request.trust_score:.2f}) meet thresholds"
        elif request.confidence_score >= config.human_review_threshold:
            action = "fallback"
            fallback_mode = config.fallback_mode
            explanation = f"Confidence ({request.confidence_score:.2f}) below threshold ({config.confidence_threshold:.2f}) - falling back"
        else:
            action = "human_review"
            fallback_mode = FallbackMode.HUMAN_REVIEW
            explanation = f"Confidence ({request.confidence_score:.2f}) below human review threshold ({config.human_review_threshold:.2f})"
    
    confidence_level = get_confidence_level(request.confidence_score)
    
    # Generate UI metadata
    ui_metadata = {
        "confidence": request.confidence_score,
        "trust_score": request.trust_score,
        "confidence_level": confidence_level.value,
        "overrideable": action != "proceed",
        "agent_actionable": action == "proceed",
        "ui_badges": [],
        "suggested_ui_component": None,
        "explanation": explanation
    }
    
    # Add appropriate badges
    if confidence_level == ConfidenceLevel.HIGH:
        ui_metadata["ui_badges"].append("HIGH_CONFIDENCE")
    elif confidence_level == ConfidenceLevel.MEDIUM:
        ui_metadata["ui_badges"].append("MEDIUM_CONFIDENCE")
    elif confidence_level == ConfidenceLevel.LOW:
        ui_metadata["ui_badges"].append("LOW_CONFIDENCE")
    else:
        ui_metadata["ui_badges"].append("CRITICAL_CONFIDENCE")
    
    if request.trust_score >= 0.8:
        ui_metadata["ui_badges"].append("HIGH_TRUST")
    elif request.trust_score >= 0.6:
        ui_metadata["ui_badges"].append("MEDIUM_TRUST")
    else:
        ui_metadata["ui_badges"].append("LOW_TRUST")
    
    # Suggest UI components based on action
    if action == "proceed":
        ui_metadata["suggested_ui_component"] = "AGENT_CHAT"
    elif action == "fallback":
        ui_metadata["suggested_ui_component"] = "CONFIRMATION_MODAL"
    elif action == "human_review":
        ui_metadata["suggested_ui_component"] = "REVIEW_FORM"
    else:
        ui_metadata["suggested_ui_component"] = "RETRY_BUTTON"
    
    return ConfidenceEvaluationResponse(
        action=action,
        fallback_mode=fallback_mode,
        confidence_level=confidence_level,
        explanation=explanation,
        retry_count=config.retry_count if config else 3,
        max_retries=config.retry_count if config else 3,
        ui_metadata=ui_metadata
    )

@app.post("/explainability/generate", response_model=ExplainabilityResponse)
async def generate_explainability(request: ExplainabilityRequest):
    """Generate explainability for low confidence decisions"""
    
    # Simulate explainability generation
    factors = []
    recommendations = []
    confidence_breakdown = {}
    
    # Analyze confidence factors
    if request.confidence_score < 0.5:
        factors.append({
            "factor": "low_training_data",
            "impact": "negative",
            "weight": 0.3,
            "description": "Limited training data for this operation type"
        })
        recommendations.append("Collect more training data for this operation")
    
    if request.trust_score < 0.6:
        factors.append({
            "factor": "low_user_trust",
            "impact": "negative",
            "weight": 0.2,
            "description": "User has low trust in AI recommendations"
        })
        recommendations.append("Improve AI accuracy and user experience")
    
    if "complexity" in request.context and request.context["complexity"] == "high":
        factors.append({
            "factor": "high_complexity",
            "impact": "negative",
            "weight": 0.4,
            "description": "High complexity operation requiring human judgment"
        })
        recommendations.append("Consider breaking down into simpler operations")
    
    # Generate confidence breakdown
    confidence_breakdown = {
        "base_confidence": request.confidence_score * 0.6,
        "context_factor": request.confidence_score * 0.2,
        "user_feedback": request.confidence_score * 0.1,
        "system_reliability": request.confidence_score * 0.1
    }
    
    explanation = f"Confidence score of {request.confidence_score:.2f} is below threshold. "
    if factors:
        explanation += f"Key factors affecting confidence: {', '.join([f['description'] for f in factors])}. "
    if recommendations:
        explanation += f"Recommendations: {', '.join(recommendations)}."
    
    return ExplainabilityResponse(
        explanation=explanation,
        factors=factors,
        recommendations=recommendations,
        confidence_breakdown=confidence_breakdown
    )

@app.post("/thresholds", response_model=ThresholdConfig)
async def create_threshold_config(config: ThresholdConfig):
    """Create a new threshold configuration"""
    config.config_id = str(uuid.uuid4())
    config.created_at = datetime.utcnow()
    config.updated_at = datetime.utcnow()
    threshold_configs_db[config.config_id] = config
    return config

@app.get("/thresholds", response_model=List[ThresholdConfig])
async def list_threshold_configs(tenant_id: str):
    """List threshold configurations for a tenant"""
    return [config for config in threshold_configs_db.values() 
            if config.tenant_id == tenant_id or config.tenant_id == "default"]

@app.put("/thresholds/{config_id}", response_model=ThresholdConfig)
async def update_threshold_config(config_id: str, config_update: Dict[str, Any]):
    """Update a threshold configuration"""
    if config_id not in threshold_configs_db:
        raise HTTPException(status_code=404, detail="Threshold configuration not found")
    
    config = threshold_configs_db[config_id]
    for field, value in config_update.items():
        if hasattr(config, field):
            setattr(config, field, value)
    
    config.updated_at = datetime.utcnow()
    return config

@app.delete("/thresholds/{config_id}")
async def delete_threshold_config(config_id: str):
    """Delete a threshold configuration"""
    if config_id not in threshold_configs_db:
        raise HTTPException(status_code=404, detail="Threshold configuration not found")
    
    del threshold_configs_db[config_id]
    return {"message": "Threshold configuration deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "confidence-thresholds",
        "timestamp": datetime.utcnow().isoformat(),
        "configs_count": len(threshold_configs_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
