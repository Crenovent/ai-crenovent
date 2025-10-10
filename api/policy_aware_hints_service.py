"""
Task 3.3.52: Policy-Aware Hints Service
- Proactive hints when compliance thresholds near breach
- Policy service integration with UI overlay support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Policy-Aware Hints Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class PolicyType(str, Enum):
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    BIAS_THRESHOLD = "bias_threshold"
    DRIFT_THRESHOLD = "drift_threshold"
    COMPLIANCE_RULE = "compliance_rule"
    DATA_QUALITY = "data_quality"
    PERFORMANCE_SLA = "performance_sla"
    GOVERNANCE_GATE = "governance_gate"

class HintSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"

class HintType(str, Enum):
    THRESHOLD_WARNING = "threshold_warning"
    COMPLIANCE_REMINDER = "compliance_reminder"
    BEST_PRACTICE = "best_practice"
    PREVENTIVE_ACTION = "preventive_action"
    REGULATORY_ALERT = "regulatory_alert"

class PolicyThreshold(BaseModel):
    threshold_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Policy details
    policy_type: PolicyType
    policy_name: str
    description: str
    
    # Threshold configuration
    warning_threshold: float  # When to start showing hints
    critical_threshold: float  # When to show critical hints
    breach_threshold: float   # When policy is actually breached
    
    # Applicable modes
    applicable_modes: List[UXMode] = Field(default_factory=lambda: [UXMode.UI_LED, UXMode.ASSISTED])
    
    # Hint configuration
    warning_message: str
    critical_message: str
    suggested_actions: List[str] = Field(default_factory=list)
    
    # Settings
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PolicyAwareHint(BaseModel):
    hint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Context
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    ux_mode: UXMode
    
    # Policy information
    policy_threshold_id: str
    policy_type: PolicyType
    current_value: float
    threshold_value: float
    
    # Hint details
    hint_type: HintType
    severity: HintSeverity
    title: str
    message: str
    detailed_explanation: str
    
    # Actions
    suggested_actions: List[str] = Field(default_factory=list)
    preventive_measures: List[str] = Field(default_factory=list)
    
    # UI display settings
    display_location: str = "top_banner"  # top_banner, sidebar, modal, inline
    auto_dismiss: bool = False
    dismiss_after_seconds: Optional[int] = None
    
    # Status
    is_active: bool = True
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

class PolicyMonitoringData(BaseModel):
    data_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Context
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    model_id: Optional[str] = None
    
    # Metrics
    confidence_score: Optional[float] = None
    bias_score: Optional[float] = None
    drift_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    performance_score: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HintConfiguration(BaseModel):
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Global settings
    enable_policy_hints: bool = True
    hint_frequency_limit: int = 5  # Max hints per hour per user
    
    # Display preferences
    default_display_location: str = "top_banner"
    enable_sound_alerts: bool = False
    enable_email_notifications: bool = False
    
    # Modes
    enabled_modes: List[UXMode] = Field(default_factory=lambda: [UXMode.UI_LED, UXMode.ASSISTED])
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
policy_thresholds_store: Dict[str, PolicyThreshold] = {}
policy_hints_store: Dict[str, PolicyAwareHint] = {}
monitoring_data_store: Dict[str, PolicyMonitoringData] = {}
hint_configs_store: Dict[str, HintConfiguration] = {}

def _create_default_policy_thresholds():
    """Create default policy thresholds"""
    
    # Confidence threshold policy
    confidence_threshold = PolicyThreshold(
        tenant_id="default",
        policy_type=PolicyType.CONFIDENCE_THRESHOLD,
        policy_name="Model Confidence Monitoring",
        description="Monitor model confidence scores to ensure reliable predictions",
        warning_threshold=0.75,
        critical_threshold=0.65,
        breach_threshold=0.50,
        applicable_modes=[UXMode.UI_LED, UXMode.ASSISTED],
        warning_message="Model confidence is below recommended level (< 0.75)",
        critical_message="Model confidence is critically low (< 0.65) - review required",
        suggested_actions=[
            "Review input data quality",
            "Consider manual validation",
            "Check for data drift",
            "Use alternative model if available"
        ]
    )
    
    # Bias threshold policy
    bias_threshold = PolicyThreshold(
        tenant_id="default",
        policy_type=PolicyType.BIAS_THRESHOLD,
        policy_name="Bias Detection Monitoring",
        description="Monitor model bias metrics to ensure fair outcomes",
        warning_threshold=0.10,
        critical_threshold=0.15,
        breach_threshold=0.20,
        warning_message="Potential bias detected in model predictions",
        critical_message="Significant bias detected - immediate review required",
        suggested_actions=[
            "Review training data for bias",
            "Apply bias mitigation techniques",
            "Consider alternative modeling approach",
            "Document bias assessment"
        ]
    )
    
    # Data quality threshold
    data_quality_threshold = PolicyThreshold(
        tenant_id="default",
        policy_type=PolicyType.DATA_QUALITY,
        policy_name="Data Quality Monitoring",
        description="Monitor data quality scores to ensure reliable inputs",
        warning_threshold=0.85,
        critical_threshold=0.75,
        breach_threshold=0.65,
        warning_message="Data quality below recommended standards",
        critical_message="Poor data quality detected - validation required",
        suggested_actions=[
            "Check data source integrity",
            "Validate data completeness",
            "Review data transformation pipeline",
            "Consider data cleansing"
        ]
    )
    
    # Store default thresholds
    policy_thresholds_store[confidence_threshold.threshold_id] = confidence_threshold
    policy_thresholds_store[bias_threshold.threshold_id] = bias_threshold
    policy_thresholds_store[data_quality_threshold.threshold_id] = data_quality_threshold
    
    # Create default hint configuration
    default_config = HintConfiguration(tenant_id="default")
    hint_configs_store[default_config.config_id] = default_config
    
    logger.info("Created default policy thresholds and configuration")

# Initialize defaults
_create_default_policy_thresholds()

@app.post("/policy-hints/thresholds", response_model=PolicyThreshold)
async def create_policy_threshold(threshold: PolicyThreshold):
    """Create a new policy threshold"""
    policy_thresholds_store[threshold.threshold_id] = threshold
    logger.info(f"Created policy threshold: {threshold.policy_name}")
    return threshold

@app.get("/policy-hints/thresholds", response_model=List[PolicyThreshold])
async def get_policy_thresholds(
    tenant_id: str,
    policy_type: Optional[PolicyType] = None,
    ux_mode: Optional[UXMode] = None
):
    """Get policy thresholds for a tenant"""
    
    thresholds = [
        t for t in policy_thresholds_store.values()
        if (t.tenant_id == tenant_id or t.tenant_id == "default") and t.is_active
    ]
    
    if policy_type:
        thresholds = [t for t in thresholds if t.policy_type == policy_type]
    
    if ux_mode:
        thresholds = [t for t in thresholds if ux_mode in t.applicable_modes]
    
    return thresholds

@app.post("/policy-hints/monitor", response_model=List[PolicyAwareHint])
async def monitor_policy_compliance(
    data: PolicyMonitoringData,
    user_id: str,
    ux_mode: UXMode,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Monitor policy compliance and generate hints if thresholds are approached"""
    
    # Store monitoring data
    monitoring_data_store[data.data_id] = data
    
    # Get applicable thresholds
    applicable_thresholds = [
        t for t in policy_thresholds_store.values()
        if ((t.tenant_id == data.tenant_id or t.tenant_id == "default") and
            t.is_active and ux_mode in t.applicable_modes)
    ]
    
    generated_hints = []
    
    # Check each threshold
    for threshold in applicable_thresholds:
        hint = _check_threshold_and_generate_hint(threshold, data, user_id, ux_mode)
        if hint:
            policy_hints_store[hint.hint_id] = hint
            generated_hints.append(hint)
    
    # Process hints in background
    if generated_hints:
        background_tasks.add_task(_process_generated_hints, generated_hints)
    
    logger.info(f"Generated {len(generated_hints)} policy hints for user {user_id}")
    return generated_hints

def _check_threshold_and_generate_hint(
    threshold: PolicyThreshold,
    data: PolicyMonitoringData,
    user_id: str,
    ux_mode: UXMode
) -> Optional[PolicyAwareHint]:
    """Check if threshold is approached and generate hint if needed"""
    
    # Get current value based on policy type
    current_value = None
    
    if threshold.policy_type == PolicyType.CONFIDENCE_THRESHOLD:
        current_value = data.confidence_score
    elif threshold.policy_type == PolicyType.BIAS_THRESHOLD:
        current_value = data.bias_score
    elif threshold.policy_type == PolicyType.DRIFT_THRESHOLD:
        current_value = data.drift_score
    elif threshold.policy_type == PolicyType.DATA_QUALITY:
        current_value = data.data_quality_score
    elif threshold.policy_type == PolicyType.PERFORMANCE_SLA:
        current_value = data.performance_score
    
    if current_value is None:
        return None
    
    # Determine if hint should be generated
    hint_needed = False
    severity = HintSeverity.INFO
    message = ""
    threshold_value = 0
    
    # For confidence and quality scores (higher is better)
    if threshold.policy_type in [PolicyType.CONFIDENCE_THRESHOLD, PolicyType.DATA_QUALITY, PolicyType.PERFORMANCE_SLA]:
        if current_value <= threshold.critical_threshold:
            hint_needed = True
            severity = HintSeverity.CRITICAL
            message = threshold.critical_message
            threshold_value = threshold.critical_threshold
        elif current_value <= threshold.warning_threshold:
            hint_needed = True
            severity = HintSeverity.WARNING
            message = threshold.warning_message
            threshold_value = threshold.warning_threshold
    
    # For bias and drift scores (lower is better)
    elif threshold.policy_type in [PolicyType.BIAS_THRESHOLD, PolicyType.DRIFT_THRESHOLD]:
        if current_value >= threshold.critical_threshold:
            hint_needed = True
            severity = HintSeverity.CRITICAL
            message = threshold.critical_message
            threshold_value = threshold.critical_threshold
        elif current_value >= threshold.warning_threshold:
            hint_needed = True
            severity = HintSeverity.WARNING
            message = threshold.warning_message
            threshold_value = threshold.warning_threshold
    
    if not hint_needed:
        return None
    
    # Generate hint
    hint = PolicyAwareHint(
        tenant_id=data.tenant_id,
        user_id=user_id,
        workflow_id=data.workflow_id,
        node_id=data.node_id,
        ux_mode=ux_mode,
        policy_threshold_id=threshold.threshold_id,
        policy_type=threshold.policy_type,
        current_value=current_value,
        threshold_value=threshold_value,
        hint_type=HintType.THRESHOLD_WARNING,
        severity=severity,
        title=f"Policy Alert: {threshold.policy_name}",
        message=message,
        detailed_explanation=_generate_detailed_explanation(threshold, current_value, threshold_value),
        suggested_actions=threshold.suggested_actions,
        preventive_measures=_generate_preventive_measures(threshold.policy_type),
        expires_at=datetime.utcnow() + timedelta(hours=1)  # Hints expire after 1 hour
    )
    
    return hint

def _generate_detailed_explanation(threshold: PolicyThreshold, current_value: float, threshold_value: float) -> str:
    """Generate detailed explanation for the hint"""
    
    if threshold.policy_type == PolicyType.CONFIDENCE_THRESHOLD:
        return f"The model's confidence score is {current_value:.2f}, which is below the recommended threshold of {threshold_value:.2f}. This may indicate uncertainty in the prediction and could affect decision reliability."
    
    elif threshold.policy_type == PolicyType.BIAS_THRESHOLD:
        return f"Bias metrics show a score of {current_value:.2f}, exceeding the acceptable threshold of {threshold_value:.2f}. This suggests potential unfair treatment of certain groups in the model's predictions."
    
    elif threshold.policy_type == PolicyType.DATA_QUALITY:
        return f"Data quality score is {current_value:.2f}, below the required standard of {threshold_value:.2f}. Poor data quality can lead to unreliable model predictions and business decisions."
    
    else:
        return f"Current value ({current_value:.2f}) has crossed the threshold ({threshold_value:.2f}) for {threshold.policy_name}."

def _generate_preventive_measures(policy_type: PolicyType) -> List[str]:
    """Generate preventive measures based on policy type"""
    
    measures = {
        PolicyType.CONFIDENCE_THRESHOLD: [
            "Implement confidence-based routing",
            "Set up automated model retraining",
            "Enable human-in-the-loop validation"
        ],
        PolicyType.BIAS_THRESHOLD: [
            "Implement bias monitoring dashboard",
            "Set up fairness constraints in model training",
            "Enable bias testing in CI/CD pipeline"
        ],
        PolicyType.DATA_QUALITY: [
            "Implement data validation rules",
            "Set up data quality monitoring alerts",
            "Enable automated data cleansing"
        ]
    }
    
    return measures.get(policy_type, ["Monitor regularly", "Set up alerts", "Review periodically"])

async def _process_generated_hints(hints: List[PolicyAwareHint]):
    """Process generated hints (notifications, logging, etc.)"""
    
    for hint in hints:
        # Log hint for audit trail
        logger.info(f"Policy hint generated: {hint.title} for user {hint.user_id}")
        
        # In production, this would:
        # - Send notifications if configured
        # - Update dashboards
        # - Trigger automated responses if configured

@app.get("/policy-hints/active", response_model=List[PolicyAwareHint])
async def get_active_hints(
    tenant_id: str,
    user_id: str,
    ux_mode: Optional[UXMode] = None,
    severity: Optional[HintSeverity] = None
):
    """Get active policy hints for a user"""
    
    current_time = datetime.utcnow()
    
    hints = [
        hint for hint in policy_hints_store.values()
        if (hint.tenant_id == tenant_id and
            hint.user_id == user_id and
            hint.is_active and
            not hint.acknowledged and
            (hint.expires_at is None or hint.expires_at > current_time))
    ]
    
    if ux_mode:
        hints = [h for h in hints if h.ux_mode == ux_mode]
    
    if severity:
        hints = [h for h in hints if h.severity == severity]
    
    # Sort by severity and creation time
    severity_order = {HintSeverity.URGENT: 4, HintSeverity.CRITICAL: 3, HintSeverity.WARNING: 2, HintSeverity.INFO: 1}
    hints.sort(key=lambda x: (severity_order[x.severity], x.created_at), reverse=True)
    
    return hints

@app.post("/policy-hints/{hint_id}/acknowledge")
async def acknowledge_hint(hint_id: str, user_id: str):
    """Acknowledge a policy hint"""
    
    if hint_id not in policy_hints_store:
        raise HTTPException(status_code=404, detail="Hint not found")
    
    hint = policy_hints_store[hint_id]
    
    if hint.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to acknowledge this hint")
    
    hint.acknowledged = True
    hint.acknowledged_at = datetime.utcnow()
    
    logger.info(f"Hint {hint_id} acknowledged by user {user_id}")
    return {"status": "acknowledged", "hint_id": hint_id}

@app.post("/policy-hints/config", response_model=HintConfiguration)
async def configure_policy_hints(config: HintConfiguration):
    """Configure policy hint settings"""
    hint_configs_store[config.config_id] = config
    logger.info(f"Configured policy hints for tenant {config.tenant_id}")
    return config

@app.get("/policy-hints/config/{tenant_id}", response_model=HintConfiguration)
async def get_hint_configuration(tenant_id: str):
    """Get hint configuration for a tenant"""
    
    # Find tenant-specific config
    tenant_config = None
    for config in hint_configs_store.values():
        if config.tenant_id == tenant_id:
            tenant_config = config
            break
    
    # Fallback to default config
    if not tenant_config:
        for config in hint_configs_store.values():
            if config.tenant_id == "default":
                tenant_config = config
                break
    
    if not tenant_config:
        raise HTTPException(status_code=404, detail="Hint configuration not found")
    
    return tenant_config

@app.get("/policy-hints/analytics/effectiveness")
async def get_hint_effectiveness_analytics(tenant_id: str):
    """Get analytics on policy hint effectiveness"""
    
    tenant_hints = [h for h in policy_hints_store.values() if h.tenant_id == tenant_id]
    
    if not tenant_hints:
        return {"message": "No policy hints found"}
    
    analytics = {
        "total_hints_generated": len(tenant_hints),
        "by_policy_type": {},
        "by_severity": {},
        "by_ux_mode": {},
        "acknowledgment_rate": 0,
        "avg_response_time_minutes": 0,
        "most_common_policies": []
    }
    
    # Analyze by policy type
    for policy_type in PolicyType:
        type_hints = [h for h in tenant_hints if h.policy_type == policy_type]
        analytics["by_policy_type"][policy_type.value] = len(type_hints)
    
    # Analyze by severity
    for severity in HintSeverity:
        severity_hints = [h for h in tenant_hints if h.severity == severity]
        analytics["by_severity"][severity.value] = len(severity_hints)
    
    # Analyze by UX mode
    for mode in UXMode:
        mode_hints = [h for h in tenant_hints if h.ux_mode == mode]
        analytics["by_ux_mode"][mode.value] = len(mode_hints)
    
    # Calculate acknowledgment rate
    acknowledged_hints = [h for h in tenant_hints if h.acknowledged]
    if tenant_hints:
        analytics["acknowledgment_rate"] = (len(acknowledged_hints) / len(tenant_hints)) * 100
    
    # Calculate average response time
    if acknowledged_hints:
        response_times = [
            (h.acknowledged_at - h.created_at).total_seconds() / 60
            for h in acknowledged_hints if h.acknowledged_at
        ]
        if response_times:
            analytics["avg_response_time_minutes"] = sum(response_times) / len(response_times)
    
    # Most common policies triggering hints
    policy_counts = {}
    for hint in tenant_hints:
        policy_id = hint.policy_threshold_id
        policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
    
    # Get policy names for top policies
    top_policies = []
    for policy_id, count in sorted(policy_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        policy = policy_thresholds_store.get(policy_id)
        if policy:
            top_policies.append({"policy_name": policy.policy_name, "hint_count": count})
    
    analytics["most_common_policies"] = top_policies
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Policy-Aware Hints Service", "task": "3.3.52"}
