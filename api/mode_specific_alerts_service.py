"""
Task 3.3.29: Mode-Specific Alerts Service
- Drift, bias, SLA alerts delivered in context
- Real-time governance with UI modals
- CRO/Compliance differentiated alerts
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Mode-Specific Alerts Service")
logger = logging.getLogger(__name__)

class AlertType(str, Enum):
    DRIFT = "drift"
    BIAS = "bias"
    SLA = "sla"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class UserRole(str, Enum):
    BUSINESS_USER = "business_user"
    ANALYST = "analyst"
    CRO = "cro"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMIN = "admin"

class AlertDeliveryMode(str, Enum):
    MODAL = "modal"
    TOAST = "toast"
    BANNER = "banner"
    EMAIL = "email"
    WEBHOOK = "webhook"

class ModeSpecificAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    alert_type: AlertType
    severity: AlertSeverity
    
    # Context
    ux_mode: UXMode
    target_roles: List[UserRole] = Field(default_factory=list)
    workflow_id: Optional[str] = None
    model_id: Optional[str] = None
    node_id: Optional[str] = None
    
    # Alert content
    title: str
    message: str
    detailed_message: Optional[str] = None
    action_required: bool = False
    suggested_actions: List[str] = Field(default_factory=list)
    
    # Delivery preferences
    delivery_modes: List[AlertDeliveryMode] = Field(default_factory=list)
    show_in_context: bool = True  # Show in the current UX mode
    persistent: bool = False  # Keep showing until acknowledged
    
    # Metadata
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    auto_resolve_after_minutes: Optional[int] = None
    
    # Alert data
    alert_data: Dict[str, Any] = Field(default_factory=dict)  # Specific metrics, thresholds, etc.

class AlertRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    rule_name: str
    alert_type: AlertType
    
    # Conditions
    ux_modes: List[UXMode] = Field(default_factory=list)  # Empty means all modes
    threshold_conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Alert configuration
    severity: AlertSeverity
    target_roles: List[UserRole] = Field(default_factory=list)
    delivery_modes: List[AlertDeliveryMode] = Field(default_factory=list)
    
    # Timing
    is_active: bool = True
    cooldown_minutes: int = 15  # Minimum time between same alerts
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None

class AlertAcknowledgment(BaseModel):
    acknowledgment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    user_id: str
    tenant_id: str
    acknowledged_at: datetime = Field(default_factory=datetime.utcnow)
    action_taken: Optional[str] = None
    notes: Optional[str] = None

# In-memory stores (replace with database in production)
alerts_store: Dict[str, ModeSpecificAlert] = {}
alert_rules_store: Dict[str, AlertRule] = {}
acknowledgments_store: Dict[str, AlertAcknowledgment] = {}

def _create_default_alert_rules():
    """Create default alert rules for different modes and scenarios"""
    
    # Drift alert for all modes
    drift_rule = AlertRule(
        tenant_id="default",
        rule_name="Model Drift Detection",
        alert_type=AlertType.DRIFT,
        ux_modes=[UXMode.UI_LED, UXMode.ASSISTED, UXMode.CONVERSATIONAL],
        threshold_conditions={
            "psi_threshold": 0.2,
            "ks_statistic": 0.1,
            "drift_score": 0.3
        },
        severity=AlertSeverity.HIGH,
        target_roles=[UserRole.ANALYST, UserRole.CRO, UserRole.COMPLIANCE_OFFICER],
        delivery_modes=[AlertDeliveryMode.MODAL, AlertDeliveryMode.EMAIL]
    )
    
    # Bias alert - more critical for Assisted/Conversational modes
    bias_rule = AlertRule(
        tenant_id="default",
        rule_name="Bias Threshold Breach",
        alert_type=AlertType.BIAS,
        ux_modes=[UXMode.ASSISTED, UXMode.CONVERSATIONAL],
        threshold_conditions={
            "demographic_parity_diff": 0.1,
            "equalized_odds_diff": 0.1,
            "tpr_gap": 0.15
        },
        severity=AlertSeverity.CRITICAL,
        target_roles=[UserRole.CRO, UserRole.COMPLIANCE_OFFICER],
        delivery_modes=[AlertDeliveryMode.MODAL, AlertDeliveryMode.EMAIL, AlertDeliveryMode.WEBHOOK]
    )
    
    # SLA breach - different thresholds per mode
    sla_rule = AlertRule(
        tenant_id="default",
        rule_name="SLA Performance Breach",
        alert_type=AlertType.SLA,
        ux_modes=[UXMode.UI_LED, UXMode.ASSISTED, UXMode.CONVERSATIONAL],
        threshold_conditions={
            "response_time_ms": 5000,  # 5 seconds
            "availability_percentage": 99.0,
            "error_rate_percentage": 5.0
        },
        severity=AlertSeverity.HIGH,
        target_roles=[UserRole.ADMIN, UserRole.CRO],
        delivery_modes=[AlertDeliveryMode.BANNER, AlertDeliveryMode.EMAIL]
    )
    
    # Store default rules
    alert_rules_store[drift_rule.rule_id] = drift_rule
    alert_rules_store[bias_rule.rule_id] = bias_rule
    alert_rules_store[sla_rule.rule_id] = sla_rule
    
    logger.info("Created default alert rules")

# Initialize default rules
_create_default_alert_rules()

@app.post("/alerts/rules", response_model=AlertRule)
async def create_alert_rule(rule: AlertRule):
    """Create a new alert rule"""
    alert_rules_store[rule.rule_id] = rule
    logger.info(f"Created alert rule {rule.rule_name} for tenant {rule.tenant_id}")
    return rule

@app.get("/alerts/rules", response_model=List[AlertRule])
async def get_alert_rules(tenant_id: str, alert_type: Optional[AlertType] = None):
    """Get alert rules for a tenant"""
    rules = [r for r in alert_rules_store.values() if r.tenant_id == tenant_id]
    
    if alert_type:
        rules = [r for r in rules if r.alert_type == alert_type]
    
    return rules

@app.post("/alerts/trigger")
async def trigger_alert(
    tenant_id: str,
    alert_type: AlertType,
    ux_mode: UXMode,
    severity: AlertSeverity,
    title: str,
    message: str,
    workflow_id: Optional[str] = None,
    model_id: Optional[str] = None,
    node_id: Optional[str] = None,
    alert_data: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Trigger an alert based on conditions"""
    
    # Find matching alert rules
    matching_rules = [
        rule for rule in alert_rules_store.values()
        if (rule.tenant_id == tenant_id and 
            rule.alert_type == alert_type and 
            rule.is_active and
            (not rule.ux_modes or ux_mode in rule.ux_modes))
    ]
    
    if not matching_rules:
        logger.warning(f"No matching alert rules for {alert_type} in {ux_mode} mode")
        return {"status": "no_matching_rules"}
    
    alerts_created = []
    
    for rule in matching_rules:
        # Check cooldown
        if rule.last_triggered:
            time_since_last = datetime.utcnow() - rule.last_triggered
            if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                continue
        
        # Create alert
        alert = ModeSpecificAlert(
            tenant_id=tenant_id,
            alert_type=alert_type,
            severity=severity,
            ux_mode=ux_mode,
            target_roles=rule.target_roles,
            workflow_id=workflow_id,
            model_id=model_id,
            node_id=node_id,
            title=title,
            message=message,
            delivery_modes=rule.delivery_modes,
            alert_data=alert_data or {}
        )
        
        # Customize message based on UX mode and role
        alert = _customize_alert_for_context(alert, rule)
        
        # Store alert
        alerts_store[alert.alert_id] = alert
        
        # Update rule last triggered time
        rule.last_triggered = datetime.utcnow()
        
        # Schedule delivery
        background_tasks.add_task(_deliver_alert, alert)
        
        alerts_created.append(alert.alert_id)
        logger.info(f"Triggered alert {alert.alert_id} for {alert_type} in {ux_mode} mode")
    
    return {"status": "success", "alerts_created": alerts_created}

def _customize_alert_for_context(alert: ModeSpecificAlert, rule: AlertRule) -> ModeSpecificAlert:
    """Customize alert content based on UX mode and target roles"""
    
    # Mode-specific customizations
    if alert.ux_mode == UXMode.CONVERSATIONAL:
        alert.detailed_message = f"While using Conversational Mode: {alert.message}"
        if alert.alert_type == AlertType.BIAS:
            alert.suggested_actions.append("Switch to Assisted Mode for manual review")
            alert.suggested_actions.append("Request human oversight for this decision")
    
    elif alert.ux_mode == UXMode.ASSISTED:
        alert.detailed_message = f"In Assisted Mode workflow: {alert.message}"
        if alert.alert_type == AlertType.DRIFT:
            alert.suggested_actions.append("Review suggested parameter adjustments")
            alert.suggested_actions.append("Consider model retraining")
    
    elif alert.ux_mode == UXMode.UI_LED:
        alert.detailed_message = f"In UI-Led workflow: {alert.message}"
        alert.suggested_actions.append("Review workflow configuration")
    
    # Role-specific customizations
    if UserRole.CRO in alert.target_roles:
        if alert.alert_type == AlertType.BIAS:
            alert.message += " - Immediate executive attention required for compliance."
            alert.action_required = True
        elif alert.alert_type == AlertType.SLA:
            alert.message += " - Customer impact possible."
    
    if UserRole.COMPLIANCE_OFFICER in alert.target_roles:
        alert.message += " - Regulatory implications may apply."
        alert.suggested_actions.append("Document incident for audit trail")
        alert.suggested_actions.append("Review compliance policy adherence")
    
    return alert

async def _deliver_alert(alert: ModeSpecificAlert):
    """Deliver alert through configured channels"""
    
    for delivery_mode in alert.delivery_modes:
        try:
            if delivery_mode == AlertDeliveryMode.MODAL:
                # In production, this would push to WebSocket or message queue
                logger.info(f"Delivering modal alert {alert.alert_id} to {alert.target_roles}")
            
            elif delivery_mode == AlertDeliveryMode.EMAIL:
                # In production, this would send actual emails
                logger.info(f"Sending email alert {alert.alert_id} to {alert.target_roles}")
            
            elif delivery_mode == AlertDeliveryMode.WEBHOOK:
                # In production, this would call external webhooks
                logger.info(f"Sending webhook alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to deliver alert {alert.alert_id} via {delivery_mode}: {e}")

@app.get("/alerts/active", response_model=List[ModeSpecificAlert])
async def get_active_alerts(
    tenant_id: str,
    ux_mode: Optional[UXMode] = None,
    user_role: Optional[UserRole] = None,
    acknowledged: bool = False
):
    """Get active alerts for a tenant, optionally filtered by mode and role"""
    
    alerts = [
        alert for alert in alerts_store.values()
        if (alert.tenant_id == tenant_id and 
            (acknowledged or alert.acknowledged_at is None))
    ]
    
    if ux_mode:
        alerts = [a for a in alerts if a.ux_mode == ux_mode]
    
    if user_role:
        alerts = [a for a in alerts if user_role in a.target_roles]
    
    # Sort by severity and time
    severity_order = {AlertSeverity.CRITICAL: 4, AlertSeverity.HIGH: 3, AlertSeverity.MEDIUM: 2, AlertSeverity.LOW: 1}
    alerts.sort(key=lambda x: (severity_order[x.severity], x.triggered_at), reverse=True)
    
    return alerts

@app.post("/alerts/{alert_id}/acknowledge", response_model=AlertAcknowledgment)
async def acknowledge_alert(
    alert_id: str,
    user_id: str,
    action_taken: Optional[str] = None,
    notes: Optional[str] = None
):
    """Acknowledge an alert"""
    
    if alert_id not in alerts_store:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert = alerts_store[alert_id]
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = user_id
    
    acknowledgment = AlertAcknowledgment(
        alert_id=alert_id,
        user_id=user_id,
        tenant_id=alert.tenant_id,
        action_taken=action_taken,
        notes=notes
    )
    
    acknowledgments_store[acknowledgment.acknowledgment_id] = acknowledgment
    
    logger.info(f"Alert {alert_id} acknowledged by user {user_id}")
    return acknowledgment

@app.get("/alerts/context/{workflow_id}")
async def get_contextual_alerts(
    workflow_id: str,
    tenant_id: str,
    ux_mode: UXMode,
    user_role: UserRole
):
    """Get alerts specific to a workflow execution context"""
    
    contextual_alerts = [
        alert for alert in alerts_store.values()
        if (alert.tenant_id == tenant_id and
            alert.workflow_id == workflow_id and
            alert.ux_mode == ux_mode and
            user_role in alert.target_roles and
            alert.acknowledged_at is None)
    ]
    
    # Add mode-specific presentation hints
    for alert in contextual_alerts:
        if alert.ux_mode == UXMode.CONVERSATIONAL:
            alert.alert_data["presentation_hint"] = "chat_bubble"
        elif alert.ux_mode == UXMode.ASSISTED:
            alert.alert_data["presentation_hint"] = "sidebar_notification"
        else:
            alert.alert_data["presentation_hint"] = "modal_dialog"
    
    return {"alerts": contextual_alerts, "context": {"workflow_id": workflow_id, "mode": ux_mode}}

@app.get("/alerts/analytics/mode-comparison")
async def get_mode_alert_analytics(tenant_id: str, days: int = 30):
    """Get alert analytics comparing different UX modes"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    recent_alerts = [
        alert for alert in alerts_store.values()
        if alert.tenant_id == tenant_id and alert.triggered_at >= cutoff_date
    ]
    
    analytics = {}
    
    for mode in UXMode:
        mode_alerts = [a for a in recent_alerts if a.ux_mode == mode]
        
        analytics[mode.value] = {
            "total_alerts": len(mode_alerts),
            "by_severity": {
                severity.value: len([a for a in mode_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "by_type": {
                alert_type.value: len([a for a in mode_alerts if a.alert_type == alert_type])
                for alert_type in AlertType
            },
            "avg_acknowledgment_time_minutes": _calculate_avg_acknowledgment_time(mode_alerts)
        }
    
    return analytics

def _calculate_avg_acknowledgment_time(alerts: List[ModeSpecificAlert]) -> float:
    """Calculate average time to acknowledge alerts"""
    acknowledged_alerts = [a for a in alerts if a.acknowledged_at]
    
    if not acknowledged_alerts:
        return 0.0
    
    total_time = sum(
        (alert.acknowledged_at - alert.triggered_at).total_seconds()
        for alert in acknowledged_alerts
    )
    
    return total_time / len(acknowledged_alerts) / 60  # Convert to minutes

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Mode-Specific Alerts Service", "task": "3.3.29"}
