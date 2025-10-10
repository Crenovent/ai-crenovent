"""
Task 3.3.34: Dynamic Hints Service
- UI tooltips based on drift/bias detection
- Proactive education with monitoring integration
- Context-aware hints like "This model drifting"
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Dynamic Hints Service")
logger = logging.getLogger(__name__)

class HintType(str, Enum):
    DRIFT_WARNING = "drift_warning"
    BIAS_ALERT = "bias_alert"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONFIDENCE_LOW = "confidence_low"
    DATA_QUALITY = "data_quality"
    COMPLIANCE_RISK = "compliance_risk"
    BEST_PRACTICE = "best_practice"
    EDUCATIONAL = "educational"

class HintSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class HintTrigger(str, Enum):
    REAL_TIME_MONITORING = "real_time_monitoring"
    THRESHOLD_BREACH = "threshold_breach"
    PATTERN_DETECTION = "pattern_detection"
    USER_ACTION = "user_action"
    SCHEDULED_CHECK = "scheduled_check"

class UIComponent(str, Enum):
    TOOLTIP = "tooltip"
    POPOVER = "popover"
    BADGE = "badge"
    INLINE_MESSAGE = "inline_message"
    MODAL = "modal"
    BANNER = "banner"

class DynamicHint(BaseModel):
    hint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Hint classification
    hint_type: HintType
    severity: HintSeverity
    trigger: HintTrigger
    
    # Content
    title: str
    message: str
    detailed_explanation: Optional[str] = None
    action_suggestions: List[str] = Field(default_factory=list)
    learn_more_url: Optional[str] = None
    
    # Context
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    model_id: Optional[str] = None
    target_element: Optional[str] = None  # CSS selector for UI element
    
    # Display settings
    ui_component: UIComponent = UIComponent.TOOLTIP
    auto_show: bool = True
    dismissible: bool = True
    show_duration_seconds: Optional[int] = 10
    
    # Conditions
    show_conditions: Dict[str, Any] = Field(default_factory=dict)
    hide_after_acknowledgment: bool = True
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    # Analytics
    times_shown: int = 0
    times_acknowledged: int = 0
    times_dismissed: int = 0

class HintRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    rule_name: str
    
    # Trigger conditions
    hint_type: HintType
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Content template
    title_template: str
    message_template: str
    severity: HintSeverity
    
    # Display settings
    ui_component: UIComponent = UIComponent.TOOLTIP
    target_selector: Optional[str] = None  # CSS selector for where to show hint
    
    # Timing
    cooldown_minutes: int = 60  # Minimum time between same hints
    max_shows_per_day: int = 5
    
    # Status
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MonitoringData(BaseModel):
    """Data structure for monitoring system integration"""
    tenant_id: str
    model_id: Optional[str] = None
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    
    # Drift metrics
    psi_score: Optional[float] = None
    ks_statistic: Optional[float] = None
    drift_detected: bool = False
    
    # Bias metrics
    demographic_parity_diff: Optional[float] = None
    equalized_odds_diff: Optional[float] = None
    bias_detected: bool = False
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    confidence_score: Optional[float] = None
    
    # Data quality
    missing_data_percentage: Optional[float] = None
    data_quality_score: Optional[float] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HintInteraction(BaseModel):
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hint_id: str
    user_id: str
    tenant_id: str
    
    # Interaction details
    action: str  # "shown", "acknowledged", "dismissed", "clicked_action"
    interaction_time: datetime = Field(default_factory=datetime.utcnow)
    time_visible_seconds: Optional[int] = None
    
    # Context
    page_url: Optional[str] = None
    ui_element: Optional[str] = None

# In-memory stores (replace with database in production)
hints_store: Dict[str, DynamicHint] = {}
hint_rules_store: Dict[str, HintRule] = {}
interactions_store: Dict[str, HintInteraction] = {}
monitoring_data_store: Dict[str, MonitoringData] = {}

def _create_default_hint_rules():
    """Create default hint rules for common scenarios"""
    
    # Model drift warning rule
    drift_rule = HintRule(
        tenant_id="default",
        rule_name="Model Drift Detection",
        hint_type=HintType.DRIFT_WARNING,
        trigger_conditions={
            "psi_score_threshold": 0.2,
            "ks_statistic_threshold": 0.1
        },
        title_template="Model Drift Detected",
        message_template="This model is showing signs of drift (PSI: {psi_score:.3f}). Consider retraining or reviewing input data.",
        severity=HintSeverity.WARNING,
        ui_component=UIComponent.TOOLTIP,
        target_selector=".ml-node, .model-card"
    )
    
    # Bias alert rule
    bias_rule = HintRule(
        tenant_id="default",
        rule_name="Bias Detection Alert",
        hint_type=HintType.BIAS_ALERT,
        trigger_conditions={
            "demographic_parity_threshold": 0.1,
            "equalized_odds_threshold": 0.1
        },
        title_template="Potential Bias Detected",
        message_template="This model may have bias issues. Demographic parity difference: {demographic_parity_diff:.3f}",
        severity=HintSeverity.ERROR,
        ui_component=UIComponent.POPOVER,
        target_selector=".ml-node, .prediction-output"
    )
    
    # Low confidence rule
    confidence_rule = HintRule(
        tenant_id="default",
        rule_name="Low Confidence Warning",
        hint_type=HintType.CONFIDENCE_LOW,
        trigger_conditions={
            "confidence_threshold": 0.7
        },
        title_template="Low Prediction Confidence",
        message_template="This prediction has low confidence ({confidence_score:.2f}). Consider manual review or additional data validation.",
        severity=HintSeverity.WARNING,
        ui_component=UIComponent.BADGE,
        target_selector=".prediction-result, .confidence-indicator"
    )
    
    # Data quality rule
    data_quality_rule = HintRule(
        tenant_id="default",
        rule_name="Data Quality Issues",
        hint_type=HintType.DATA_QUALITY,
        trigger_conditions={
            "missing_data_threshold": 0.1,
            "data_quality_threshold": 0.8
        },
        title_template="Data Quality Concern",
        message_template="Input data has quality issues ({missing_data_percentage:.1f}% missing values). This may affect prediction accuracy.",
        severity=HintSeverity.WARNING,
        ui_component=UIComponent.INLINE_MESSAGE,
        target_selector=".data-input, .data-source-node"
    )
    
    # Best practice educational hint
    best_practice_rule = HintRule(
        tenant_id="default",
        rule_name="Best Practice Guidance",
        hint_type=HintType.BEST_PRACTICE,
        trigger_conditions={
            "user_action": "creating_ml_node"
        },
        title_template="Best Practice Tip",
        message_template="Consider setting up monitoring and alerts for this ML node to track performance over time.",
        severity=HintSeverity.INFO,
        ui_component=UIComponent.TOOLTIP,
        target_selector=".node-configuration-panel"
    )
    
    # Store default rules
    hint_rules_store[drift_rule.rule_id] = drift_rule
    hint_rules_store[bias_rule.rule_id] = bias_rule
    hint_rules_store[confidence_rule.rule_id] = confidence_rule
    hint_rules_store[data_quality_rule.rule_id] = data_quality_rule
    hint_rules_store[best_practice_rule.rule_id] = best_practice_rule
    
    logger.info("Created default hint rules")

# Initialize default rules
_create_default_hint_rules()

@app.post("/hints/rules", response_model=HintRule)
async def create_hint_rule(rule: HintRule):
    """Create a new hint rule"""
    hint_rules_store[rule.rule_id] = rule
    logger.info(f"Created hint rule: {rule.rule_name}")
    return rule

@app.get("/hints/rules", response_model=List[HintRule])
async def get_hint_rules(tenant_id: str, hint_type: Optional[HintType] = None):
    """Get hint rules for a tenant"""
    rules = [r for r in hint_rules_store.values() if r.tenant_id == tenant_id and r.is_active]
    
    if hint_type:
        rules = [r for r in rules if r.hint_type == hint_type]
    
    return rules

@app.post("/hints/monitoring-data")
async def submit_monitoring_data(
    data: MonitoringData,
    background_tasks: BackgroundTasks
):
    """Submit monitoring data that may trigger dynamic hints"""
    
    # Store monitoring data
    data_key = f"{data.tenant_id}_{data.model_id}_{data.workflow_id}_{data.node_id}"
    monitoring_data_store[data_key] = data
    
    # Process hints in background
    background_tasks.add_task(_process_monitoring_data, data)
    
    logger.info(f"Received monitoring data for {data_key}")
    return {"status": "received", "data_key": data_key}

async def _process_monitoring_data(data: MonitoringData):
    """Process monitoring data and generate hints if conditions are met"""
    
    # Find applicable rules
    applicable_rules = [
        rule for rule in hint_rules_store.values()
        if rule.tenant_id == data.tenant_id and rule.is_active
    ]
    
    for rule in applicable_rules:
        if _should_trigger_hint(rule, data):
            hint = _create_hint_from_rule(rule, data)
            hints_store[hint.hint_id] = hint
            logger.info(f"Generated hint {hint.hint_id} from rule {rule.rule_name}")

def _should_trigger_hint(rule: HintRule, data: MonitoringData) -> bool:
    """Check if monitoring data meets rule conditions"""
    
    conditions = rule.trigger_conditions
    
    # Check drift conditions
    if "psi_score_threshold" in conditions and data.psi_score:
        if data.psi_score > conditions["psi_score_threshold"]:
            return True
    
    if "ks_statistic_threshold" in conditions and data.ks_statistic:
        if data.ks_statistic > conditions["ks_statistic_threshold"]:
            return True
    
    # Check bias conditions
    if "demographic_parity_threshold" in conditions and data.demographic_parity_diff:
        if abs(data.demographic_parity_diff) > conditions["demographic_parity_threshold"]:
            return True
    
    if "equalized_odds_threshold" in conditions and data.equalized_odds_diff:
        if abs(data.equalized_odds_diff) > conditions["equalized_odds_threshold"]:
            return True
    
    # Check confidence conditions
    if "confidence_threshold" in conditions and data.confidence_score:
        if data.confidence_score < conditions["confidence_threshold"]:
            return True
    
    # Check data quality conditions
    if "missing_data_threshold" in conditions and data.missing_data_percentage:
        if data.missing_data_percentage > conditions["missing_data_threshold"]:
            return True
    
    if "data_quality_threshold" in conditions and data.data_quality_score:
        if data.data_quality_score < conditions["data_quality_threshold"]:
            return True
    
    return False

def _create_hint_from_rule(rule: HintRule, data: MonitoringData) -> DynamicHint:
    """Create a dynamic hint from a rule and monitoring data"""
    
    # Format templates with actual data
    title = rule.title_template.format(
        psi_score=data.psi_score or 0,
        ks_statistic=data.ks_statistic or 0,
        demographic_parity_diff=data.demographic_parity_diff or 0,
        equalized_odds_diff=data.equalized_odds_diff or 0,
        confidence_score=data.confidence_score or 0,
        missing_data_percentage=data.missing_data_percentage or 0,
        data_quality_score=data.data_quality_score or 0
    )
    
    message = rule.message_template.format(
        psi_score=data.psi_score or 0,
        ks_statistic=data.ks_statistic or 0,
        demographic_parity_diff=data.demographic_parity_diff or 0,
        equalized_odds_diff=data.equalized_odds_diff or 0,
        confidence_score=data.confidence_score or 0,
        missing_data_percentage=data.missing_data_percentage or 0,
        data_quality_score=data.data_quality_score or 0
    )
    
    # Generate action suggestions based on hint type
    action_suggestions = _generate_action_suggestions(rule.hint_type, data)
    
    hint = DynamicHint(
        tenant_id=data.tenant_id,
        hint_type=rule.hint_type,
        severity=rule.severity,
        trigger=HintTrigger.REAL_TIME_MONITORING,
        title=title,
        message=message,
        action_suggestions=action_suggestions,
        workflow_id=data.workflow_id,
        node_id=data.node_id,
        model_id=data.model_id,
        target_element=rule.target_selector,
        ui_component=rule.ui_component,
        expires_at=datetime.utcnow() + timedelta(hours=24)  # Default 24-hour expiry
    )
    
    return hint

def _generate_action_suggestions(hint_type: HintType, data: MonitoringData) -> List[str]:
    """Generate contextual action suggestions based on hint type"""
    
    suggestions = []
    
    if hint_type == HintType.DRIFT_WARNING:
        suggestions = [
            "Review recent input data for changes",
            "Consider retraining the model with recent data",
            "Check data pipeline for issues",
            "Set up automated drift monitoring"
        ]
    
    elif hint_type == HintType.BIAS_ALERT:
        suggestions = [
            "Review model training data for bias",
            "Apply bias mitigation techniques",
            "Conduct fairness audit",
            "Consider alternative modeling approaches"
        ]
    
    elif hint_type == HintType.CONFIDENCE_LOW:
        suggestions = [
            "Review input data quality",
            "Consider manual review for this prediction",
            "Gather additional features",
            "Use ensemble methods for better confidence"
        ]
    
    elif hint_type == HintType.DATA_QUALITY:
        suggestions = [
            "Investigate data source issues",
            "Implement data validation rules",
            "Consider data imputation strategies",
            "Set up data quality monitoring"
        ]
    
    elif hint_type == HintType.BEST_PRACTICE:
        suggestions = [
            "Review best practices documentation",
            "Set up monitoring and alerts",
            "Consider A/B testing",
            "Document model assumptions"
        ]
    
    return suggestions

@app.get("/hints/active")
async def get_active_hints(
    tenant_id: str,
    workflow_id: Optional[str] = None,
    node_id: Optional[str] = None,
    model_id: Optional[str] = None,
    hint_type: Optional[HintType] = None
):
    """Get active hints for display in UI"""
    
    current_time = datetime.utcnow()
    
    # Filter active, non-expired hints
    active_hints = [
        hint for hint in hints_store.values()
        if (hint.tenant_id == tenant_id and
            hint.is_active and
            (hint.expires_at is None or hint.expires_at > current_time))
    ]
    
    # Apply additional filters
    if workflow_id:
        active_hints = [h for h in active_hints if h.workflow_id == workflow_id]
    
    if node_id:
        active_hints = [h for h in active_hints if h.node_id == node_id]
    
    if model_id:
        active_hints = [h for h in active_hints if h.model_id == model_id]
    
    if hint_type:
        active_hints = [h for h in active_hints if h.hint_type == hint_type]
    
    # Sort by severity and creation time
    severity_order = {HintSeverity.CRITICAL: 4, HintSeverity.ERROR: 3, HintSeverity.WARNING: 2, HintSeverity.INFO: 1}
    active_hints.sort(key=lambda x: (severity_order[x.severity], x.created_at), reverse=True)
    
    return {"hints": active_hints, "count": len(active_hints)}

@app.post("/hints/{hint_id}/interaction")
async def record_hint_interaction(
    hint_id: str,
    user_id: str,
    action: str,
    time_visible_seconds: Optional[int] = None,
    page_url: Optional[str] = None
):
    """Record user interaction with a hint"""
    
    if hint_id not in hints_store:
        raise HTTPException(status_code=404, detail="Hint not found")
    
    hint = hints_store[hint_id]
    
    # Record interaction
    interaction = HintInteraction(
        hint_id=hint_id,
        user_id=user_id,
        tenant_id=hint.tenant_id,
        action=action,
        time_visible_seconds=time_visible_seconds,
        page_url=page_url,
        ui_element=hint.target_element
    )
    
    interactions_store[interaction.interaction_id] = interaction
    
    # Update hint analytics
    if action == "shown":
        hint.times_shown += 1
    elif action == "acknowledged":
        hint.times_acknowledged += 1
        if hint.hide_after_acknowledgment:
            hint.is_active = False
    elif action == "dismissed":
        hint.times_dismissed += 1
        hint.is_active = False
    
    logger.info(f"Recorded hint interaction: {action} for hint {hint_id}")
    return {"status": "recorded", "interaction_id": interaction.interaction_id}

@app.get("/hints/analytics/effectiveness")
async def get_hint_effectiveness_analytics(tenant_id: str, days: int = 30):
    """Get analytics on hint effectiveness"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get recent hints and interactions
    recent_hints = [
        hint for hint in hints_store.values()
        if hint.tenant_id == tenant_id and hint.created_at >= cutoff_date
    ]
    
    recent_interactions = [
        interaction for interaction in interactions_store.values()
        if interaction.tenant_id == tenant_id and interaction.interaction_time >= cutoff_date
    ]
    
    analytics = {
        "total_hints_generated": len(recent_hints),
        "by_hint_type": {},
        "by_severity": {},
        "by_ui_component": {},
        "interaction_rates": {},
        "avg_visibility_time": 0,
        "most_effective_hints": []
    }
    
    # Analyze by hint type
    for hint_type in HintType:
        type_hints = [h for h in recent_hints if h.hint_type == hint_type]
        analytics["by_hint_type"][hint_type.value] = {
            "count": len(type_hints),
            "avg_acknowledgment_rate": _calculate_acknowledgment_rate(type_hints)
        }
    
    # Analyze by severity
    for severity in HintSeverity:
        severity_hints = [h for h in recent_hints if h.severity == severity]
        analytics["by_severity"][severity.value] = len(severity_hints)
    
    # Analyze by UI component
    for component in UIComponent:
        component_hints = [h for h in recent_hints if h.ui_component == component]
        analytics["by_ui_component"][component.value] = len(component_hints)
    
    # Calculate interaction rates
    total_shown = sum(h.times_shown for h in recent_hints)
    total_acknowledged = sum(h.times_acknowledged for h in recent_hints)
    total_dismissed = sum(h.times_dismissed for h in recent_hints)
    
    if total_shown > 0:
        analytics["interaction_rates"] = {
            "acknowledgment_rate": (total_acknowledged / total_shown) * 100,
            "dismissal_rate": (total_dismissed / total_shown) * 100
        }
    
    # Calculate average visibility time
    visibility_times = [i.time_visible_seconds for i in recent_interactions if i.time_visible_seconds]
    if visibility_times:
        analytics["avg_visibility_time"] = sum(visibility_times) / len(visibility_times)
    
    # Find most effective hints (highest acknowledgment rate)
    effectiveness_scores = []
    for hint in recent_hints:
        if hint.times_shown > 0:
            effectiveness = (hint.times_acknowledged / hint.times_shown) * 100
            effectiveness_scores.append({
                "hint_id": hint.hint_id,
                "hint_type": hint.hint_type.value,
                "title": hint.title,
                "effectiveness_score": effectiveness,
                "times_shown": hint.times_shown
            })
    
    effectiveness_scores.sort(key=lambda x: x["effectiveness_score"], reverse=True)
    analytics["most_effective_hints"] = effectiveness_scores[:5]
    
    return analytics

def _calculate_acknowledgment_rate(hints: List[DynamicHint]) -> float:
    """Calculate acknowledgment rate for a list of hints"""
    total_shown = sum(h.times_shown for h in hints)
    total_acknowledged = sum(h.times_acknowledged for h in hints)
    
    if total_shown == 0:
        return 0.0
    
    return (total_acknowledged / total_shown) * 100

@app.delete("/hints/{hint_id}")
async def delete_hint(hint_id: str):
    """Delete a specific hint"""
    if hint_id not in hints_store:
        raise HTTPException(status_code=404, detail="Hint not found")
    
    del hints_store[hint_id]
    logger.info(f"Deleted hint {hint_id}")
    return {"status": "deleted"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Dynamic Hints Service", "task": "3.3.34"}
