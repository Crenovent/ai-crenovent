"""
Task 3.3.33: UX Fallback Flows Service
- Fallback from Conversational to Assisted mode when LLM fails
- Safety net with orchestrator integration
- Logs rationale for all fallback decisions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA UX Fallback Flows Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class FallbackTrigger(str, Enum):
    LLM_FAILURE = "llm_failure"
    LLM_TIMEOUT = "llm_timeout"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    PARSING_ERROR = "parsing_error"
    SAFETY_VIOLATION = "safety_violation"
    USER_REQUEST = "user_request"
    SYSTEM_OVERLOAD = "system_overload"

class FallbackSeverity(str, Enum):
    LOW = "low"           # Minor issue, suggest fallback
    MEDIUM = "medium"     # Moderate issue, recommend fallback
    HIGH = "high"         # Serious issue, strongly recommend fallback
    CRITICAL = "critical" # Critical failure, force fallback

class FallbackAction(str, Enum):
    SUGGEST = "suggest"           # Suggest user switch modes
    RECOMMEND = "recommend"       # Recommend user switch modes
    FORCE = "force"              # Automatically switch modes
    ESCALATE = "escalate"        # Escalate to human support

class PolicyBadgeType(str, Enum):
    COMPLIANCE = "compliance"     # Compliance policy triggered fallback
    SECURITY = "security"        # Security policy triggered fallback
    RISK = "risk"                # Risk management policy triggered fallback
    GOVERNANCE = "governance"    # Governance policy triggered fallback

class MessagePattern(str, Enum):
    PLAIN = "plain"               # Simple, direct message
    EXEC_SUMMARY = "exec_summary" # Executive-level summary
    VERBOSE = "verbose"           # Detailed technical explanation

class UserPersona(str, Enum):
    EXECUTIVE = "executive"       # CRO, CFO, VP-level
    MANAGER = "manager"          # RevOps Manager, Team Lead
    INDIVIDUAL = "individual"    # AE, SDR, individual contributor
    TECHNICAL = "technical"      # Developer, Admin

class FallbackRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    rule_name: str
    
    # Trigger conditions
    trigger: FallbackTrigger
    source_mode: UXMode
    target_mode: UXMode
    
    # Conditions
    threshold_conditions: Dict[str, Any] = Field(default_factory=dict)
    consecutive_failures: int = 1  # Number of failures before triggering
    time_window_minutes: int = 5   # Time window for counting failures
    
    # Response
    severity: FallbackSeverity
    action: FallbackAction
    user_message: str
    technical_message: str
    
    # Settings
    is_active: bool = True
    auto_fallback_enabled: bool = False  # Whether to auto-switch without user consent
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FallbackEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    session_id: str
    
    # Event details
    trigger: FallbackTrigger
    source_mode: UXMode
    target_mode: UXMode
    severity: FallbackSeverity
    
    # Context
    workflow_id: Optional[str] = None
    conversation_turn: Optional[int] = None
    original_request: str
    error_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Decision
    action_taken: FallbackAction
    user_message: str
    technical_rationale: str
    
    # User response
    user_accepted_fallback: Optional[bool] = None
    user_response_time_seconds: Optional[int] = None
    
    # Timing
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

class FallbackSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Session state
    current_mode: UXMode
    original_mode: UXMode
    fallback_chain: List[UXMode] = Field(default_factory=list)  # Track mode switches
    
    # Failure tracking
    consecutive_failures: Dict[FallbackTrigger, int] = Field(default_factory=dict)
    last_failure_times: Dict[FallbackTrigger, datetime] = Field(default_factory=dict)
    
    # Session metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

# In-memory stores (replace with database in production)
fallback_rules_store: Dict[str, FallbackRule] = {}
fallback_events_store: Dict[str, FallbackEvent] = {}
fallback_sessions_store: Dict[str, FallbackSession] = {}

def _create_default_fallback_rules():
    """Create default fallback rules for common scenarios"""
    
    # LLM failure fallback
    llm_failure_rule = FallbackRule(
        tenant_id="default",
        rule_name="LLM Failure Fallback",
        trigger=FallbackTrigger.LLM_FAILURE,
        source_mode=UXMode.CONVERSATIONAL,
        target_mode=UXMode.ASSISTED,
        consecutive_failures=1,
        severity=FallbackSeverity.HIGH,
        action=FallbackAction.RECOMMEND,
        user_message="I'm having trouble understanding your request. Would you like to switch to Assisted Mode where I can guide you step-by-step?",
        technical_message="LLM service unavailable or returned error response"
    )
    
    # LLM timeout fallback
    llm_timeout_rule = FallbackRule(
        tenant_id="default",
        rule_name="LLM Timeout Fallback",
        trigger=FallbackTrigger.LLM_TIMEOUT,
        source_mode=UXMode.CONVERSATIONAL,
        target_mode=UXMode.ASSISTED,
        threshold_conditions={"timeout_seconds": 30},
        severity=FallbackSeverity.MEDIUM,
        action=FallbackAction.SUGGEST,
        user_message="This is taking longer than expected. Would you like to try Assisted Mode for faster results?",
        technical_message="LLM response exceeded timeout threshold"
    )
    
    # Low confidence fallback
    confidence_rule = FallbackRule(
        tenant_id="default",
        rule_name="Low Confidence Fallback",
        trigger=FallbackTrigger.CONFIDENCE_TOO_LOW,
        source_mode=UXMode.CONVERSATIONAL,
        target_mode=UXMode.ASSISTED,
        threshold_conditions={"confidence_threshold": 0.3},
        consecutive_failures=2,
        severity=FallbackSeverity.MEDIUM,
        action=FallbackAction.SUGGEST,
        user_message="I'm not confident I understood your request correctly. Let me help you in Assisted Mode where we can work through this together.",
        technical_message="LLM confidence score below threshold"
    )
    
    # Safety violation fallback
    safety_rule = FallbackRule(
        tenant_id="default",
        rule_name="Safety Violation Fallback",
        trigger=FallbackTrigger.SAFETY_VIOLATION,
        source_mode=UXMode.CONVERSATIONAL,
        target_mode=UXMode.UI_LED,
        severity=FallbackSeverity.CRITICAL,
        action=FallbackAction.FORCE,
        auto_fallback_enabled=True,
        user_message="For security reasons, I'm switching you to the standard interface.",
        technical_message="Safety guardrails triggered automatic fallback"
    )
    
    # System overload fallback
    overload_rule = FallbackRule(
        tenant_id="default",
        rule_name="System Overload Fallback",
        trigger=FallbackTrigger.SYSTEM_OVERLOAD,
        source_mode=UXMode.CONVERSATIONAL,
        target_mode=UXMode.UI_LED,
        threshold_conditions={"cpu_usage_percent": 90, "queue_length": 100},
        severity=FallbackSeverity.HIGH,
        action=FallbackAction.RECOMMEND,
        user_message="Our conversational system is experiencing high load. I recommend using the standard interface for better performance.",
        technical_message="System resource utilization exceeded thresholds"
    )
    
    # Store default rules
    fallback_rules_store[llm_failure_rule.rule_id] = llm_failure_rule
    fallback_rules_store[llm_timeout_rule.rule_id] = llm_timeout_rule
    fallback_rules_store[confidence_rule.rule_id] = confidence_rule
    fallback_rules_store[safety_rule.rule_id] = safety_rule
    fallback_rules_store[overload_rule.rule_id] = overload_rule
    
    logger.info("Created default fallback rules")

# Initialize default rules
_create_default_fallback_rules()

@app.post("/fallback/rules", response_model=FallbackRule)
async def create_fallback_rule(rule: FallbackRule):
    """Create a new fallback rule"""
    fallback_rules_store[rule.rule_id] = rule
    logger.info(f"Created fallback rule: {rule.rule_name}")
    return rule

@app.get("/fallback/rules", response_model=List[FallbackRule])
async def get_fallback_rules(tenant_id: str, source_mode: Optional[UXMode] = None):
    """Get fallback rules for a tenant"""
    rules = [r for r in fallback_rules_store.values() if r.tenant_id == tenant_id and r.is_active]
    
    if source_mode:
        rules = [r for r in rules if r.source_mode == source_mode]
    
    return rules

@app.post("/fallback/sessions", response_model=FallbackSession)
async def create_fallback_session(
    tenant_id: str,
    user_id: str,
    initial_mode: UXMode
):
    """Create a new fallback session"""
    session = FallbackSession(
        tenant_id=tenant_id,
        user_id=user_id,
        current_mode=initial_mode,
        original_mode=initial_mode,
        fallback_chain=[initial_mode]
    )
    
    fallback_sessions_store[session.session_id] = session
    logger.info(f"Created fallback session {session.session_id} in {initial_mode} mode")
    return session

@app.post("/fallback/check-trigger")
async def check_fallback_trigger(
    session_id: str,
    trigger: FallbackTrigger,
    original_request: str,
    error_details: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Check if a fallback should be triggered and create fallback event"""
    
    if session_id not in fallback_sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = fallback_sessions_store[session_id]
    current_mode = session.current_mode
    
    # Find applicable fallback rules
    applicable_rules = [
        rule for rule in fallback_rules_store.values()
        if (rule.tenant_id == session.tenant_id and
            rule.trigger == trigger and
            rule.source_mode == current_mode and
            rule.is_active)
    ]
    
    if not applicable_rules:
        return {"fallback_triggered": False, "reason": "No applicable rules"}
    
    # Update failure tracking
    _update_failure_tracking(session, trigger)
    
    # Check if any rule conditions are met
    triggered_rule = None
    for rule in applicable_rules:
        if _should_trigger_fallback(session, rule, error_details or {}):
            triggered_rule = rule
            break
    
    if not triggered_rule:
        return {"fallback_triggered": False, "reason": "Conditions not met"}
    
    # Create fallback event
    fallback_event = FallbackEvent(
        tenant_id=session.tenant_id,
        user_id=session.user_id,
        session_id=session_id,
        trigger=trigger,
        source_mode=current_mode,
        target_mode=triggered_rule.target_mode,
        severity=triggered_rule.severity,
        original_request=original_request,
        error_details=error_details or {},
        action_taken=triggered_rule.action,
        user_message=triggered_rule.user_message,
        technical_rationale=triggered_rule.technical_message
    )
    
    fallback_events_store[fallback_event.event_id] = fallback_event
    
    # Execute fallback action
    background_tasks.add_task(_execute_fallback_action, fallback_event, session, triggered_rule)
    
    logger.info(f"Triggered fallback: {trigger.value} -> {triggered_rule.target_mode.value}")
    
    return {
        "fallback_triggered": True,
        "event_id": fallback_event.event_id,
        "target_mode": triggered_rule.target_mode.value,
        "action": triggered_rule.action.value,
        "user_message": triggered_rule.user_message,
        "severity": triggered_rule.severity.value,
        "auto_fallback": triggered_rule.auto_fallback_enabled
    }

def _update_failure_tracking(session: FallbackSession, trigger: FallbackTrigger):
    """Update failure tracking for the session"""
    current_time = datetime.utcnow()
    
    # Initialize if not exists
    if trigger not in session.consecutive_failures:
        session.consecutive_failures[trigger] = 0
        session.last_failure_times[trigger] = current_time
    
    # Check if within time window
    last_failure = session.last_failure_times[trigger]
    time_diff = (current_time - last_failure).total_seconds() / 60  # minutes
    
    # Reset counter if outside time window
    if time_diff > 5:  # Default 5-minute window
        session.consecutive_failures[trigger] = 1
    else:
        session.consecutive_failures[trigger] += 1
    
    session.last_failure_times[trigger] = current_time
    session.last_activity = current_time

def _should_trigger_fallback(
    session: FallbackSession,
    rule: FallbackRule,
    error_details: Dict[str, Any]
) -> bool:
    """Check if fallback rule conditions are met"""
    
    # Check consecutive failures
    consecutive_count = session.consecutive_failures.get(rule.trigger, 0)
    if consecutive_count < rule.consecutive_failures:
        return False
    
    # Check threshold conditions
    for condition_key, threshold_value in rule.threshold_conditions.items():
        actual_value = error_details.get(condition_key)
        
        if actual_value is None:
            continue
        
        # Handle different comparison types
        if condition_key.endswith("_threshold"):
            if actual_value > threshold_value:
                return True
        elif condition_key.endswith("_percent"):
            if actual_value > threshold_value:
                return True
        elif condition_key.endswith("_seconds"):
            if actual_value > threshold_value:
                return True
        else:
            if actual_value >= threshold_value:
                return True
    
    return True  # Default to trigger if no specific conditions

async def _execute_fallback_action(
    fallback_event: FallbackEvent,
    session: FallbackSession,
    rule: FallbackRule
):
    """Execute the fallback action"""
    
    try:
        if rule.action == FallbackAction.FORCE or rule.auto_fallback_enabled:
            # Automatically switch modes
            session.current_mode = rule.target_mode
            session.fallback_chain.append(rule.target_mode)
            
            logger.info(f"Auto-switched session {session.session_id} to {rule.target_mode.value}")
            
        elif rule.action in [FallbackAction.SUGGEST, FallbackAction.RECOMMEND]:
            # Log suggestion for UI to present to user
            logger.info(f"Suggested mode switch for session {session.session_id}")
            
        elif rule.action == FallbackAction.ESCALATE:
            # Log escalation request
            logger.warning(f"Escalating session {session.session_id} to human support")
        
        # Log the fallback decision
        _log_fallback_rationale(fallback_event, rule)
        
    except Exception as e:
        logger.error(f"Failed to execute fallback action: {e}")

def _log_fallback_rationale(fallback_event: FallbackEvent, rule: FallbackRule):
    """Log detailed rationale for fallback decision"""
    
    rationale = {
        "event_id": fallback_event.event_id,
        "session_id": fallback_event.session_id,
        "timestamp": fallback_event.triggered_at.isoformat(),
        "trigger": fallback_event.trigger.value,
        "source_mode": fallback_event.source_mode.value,
        "target_mode": fallback_event.target_mode.value,
        "rule_applied": rule.rule_name,
        "severity": fallback_event.severity.value,
        "action_taken": fallback_event.action_taken.value,
        "technical_rationale": fallback_event.technical_rationale,
        "error_context": fallback_event.error_details,
        "auto_fallback": rule.auto_fallback_enabled
    }
    
    logger.info(f"Fallback rationale: {json.dumps(rationale)}")

def generate_persona_aware_message(
    trigger: FallbackTrigger,
    severity: FallbackSeverity,
    persona: UserPersona,
    pattern: MessagePattern,
    context: Dict[str, Any] = None
) -> str:
    """Generate persona-aware fallback messages - Task 6.4.38"""
    
    context = context or {}
    
    # Message templates by persona and pattern
    message_templates = {
        UserPersona.EXECUTIVE: {
            MessagePattern.PLAIN: {
                FallbackTrigger.LLM_FAILURE: "System temporarily switched to standard interface for reliability.",
                FallbackTrigger.CONFIDENCE_TOO_LOW: "Switched to guided mode to ensure accuracy.",
                FallbackTrigger.SYSTEM_OVERLOAD: "Using optimized interface due to high system demand."
            },
            MessagePattern.EXEC_SUMMARY: {
                FallbackTrigger.LLM_FAILURE: "AI system encountered an issue. Automatically switched to proven workflow interface to maintain business continuity.",
                FallbackTrigger.CONFIDENCE_TOO_LOW: "AI confidence below business standards. Using guided interface to ensure decision quality.",
                FallbackTrigger.SYSTEM_OVERLOAD: "High system utilization detected. Optimized interface active to maintain performance SLAs."
            }
        },
        UserPersona.MANAGER: {
            MessagePattern.PLAIN: {
                FallbackTrigger.LLM_FAILURE: "Conversational AI unavailable. Switched to assisted workflow mode.",
                FallbackTrigger.CONFIDENCE_TOO_LOW: "AI confidence too low. Using step-by-step guided mode.",
                FallbackTrigger.SYSTEM_OVERLOAD: "System under high load. Using optimized workflow interface."
            },
            MessagePattern.VERBOSE: {
                FallbackTrigger.LLM_FAILURE: "The conversational AI system encountered a processing error. To ensure continuity, we've automatically switched to the assisted workflow mode where you can complete your tasks using guided steps.",
                FallbackTrigger.CONFIDENCE_TOO_LOW: "The AI system's confidence in understanding your request is below our quality threshold. We've switched to guided mode to ensure accurate task completion.",
                FallbackTrigger.SYSTEM_OVERLOAD: "Current system load is high, which may impact AI response times. We've switched to the optimized workflow interface to maintain performance."
            }
        },
        UserPersona.INDIVIDUAL: {
            MessagePattern.PLAIN: {
                FallbackTrigger.LLM_FAILURE: "Chat isn't working right now. Let me help you with the step-by-step interface.",
                FallbackTrigger.CONFIDENCE_TOO_LOW: "I'm not sure I understood correctly. Let's use the guided workflow instead.",
                FallbackTrigger.SYSTEM_OVERLOAD: "Things are running slow today. Using the faster workflow interface."
            },
            MessagePattern.VERBOSE: {
                FallbackTrigger.LLM_FAILURE: "I'm having trouble with the conversational interface right now. Don't worry - I'll switch you to the guided workflow where you can complete everything step-by-step.",
                FallbackTrigger.CONFIDENCE_TOO_LOW: "I want to make sure I get this right for you. Let me switch to the guided mode where we can work through this together.",
                FallbackTrigger.SYSTEM_OVERLOAD: "The system is pretty busy right now, so I'm switching you to the streamlined interface to keep things moving smoothly."
            }
        },
        UserPersona.TECHNICAL: {
            MessagePattern.VERBOSE: {
                FallbackTrigger.LLM_FAILURE: f"LLM service failure detected (Error: {context.get('error_code', 'UNKNOWN')}). Fallback routing to deterministic workflow engine. Session context preserved.",
                FallbackTrigger.CONFIDENCE_TOO_LOW: f"Model confidence score {context.get('confidence', 'N/A')} below threshold {context.get('threshold', '0.7')}. Initiating fallback to rule-based workflow.",
                FallbackTrigger.SYSTEM_OVERLOAD: f"System load at {context.get('cpu_usage', 'N/A')}%. Activating performance-optimized interface with reduced latency."
            }
        }
    }
    
    # Get appropriate message template
    persona_templates = message_templates.get(persona, message_templates[UserPersona.INDIVIDUAL])
    pattern_templates = persona_templates.get(pattern, persona_templates.get(MessagePattern.PLAIN, {}))
    message = pattern_templates.get(trigger, "System switched to alternative interface for optimal performance.")
    
    return message

def generate_policy_badge(
    policy_name: str,
    policy_type: str,
    trigger_reason: str,
    tenant_id: str = None
) -> Dict[str, Any]:
    """Generate policy badge for policy-driven fallbacks - Task 6.4.42"""
    
    # Determine badge type from policy type
    badge_type_mapping = {
        "sox_compliance": PolicyBadgeType.COMPLIANCE,
        "gdpr_compliance": PolicyBadgeType.COMPLIANCE,
        "hipaa_compliance": PolicyBadgeType.COMPLIANCE,
        "pci_compliance": PolicyBadgeType.COMPLIANCE,
        "security_policy": PolicyBadgeType.SECURITY,
        "data_security": PolicyBadgeType.SECURITY,
        "access_control": PolicyBadgeType.SECURITY,
        "risk_management": PolicyBadgeType.RISK,
        "operational_risk": PolicyBadgeType.RISK,
        "credit_risk": PolicyBadgeType.RISK,
        "governance_policy": PolicyBadgeType.GOVERNANCE,
        "approval_policy": PolicyBadgeType.GOVERNANCE,
        "workflow_governance": PolicyBadgeType.GOVERNANCE
    }
    
    badge_type = badge_type_mapping.get(policy_type.lower(), PolicyBadgeType.GOVERNANCE)
    
    # Generate badge display properties
    badge_config = {
        PolicyBadgeType.COMPLIANCE: {
            "color": "#1f77b4",      # Blue
            "icon": "shield-check",
            "priority": "high",
            "description": "Compliance policy enforced"
        },
        PolicyBadgeType.SECURITY: {
            "color": "#d62728",      # Red
            "icon": "security",
            "priority": "critical",
            "description": "Security policy enforced"
        },
        PolicyBadgeType.RISK: {
            "color": "#ff7f0e",      # Orange
            "icon": "warning",
            "priority": "high",
            "description": "Risk policy enforced"
        },
        PolicyBadgeType.GOVERNANCE: {
            "color": "#2ca02c",      # Green
            "icon": "gavel",
            "priority": "medium",
            "description": "Governance policy enforced"
        }
    }
    
    config = badge_config[badge_type]
    
    policy_badge = {
        "badge_id": f"policy_{policy_name.lower().replace(' ', '_')}",
        "badge_type": badge_type.value,
        "policy_name": policy_name,
        "policy_type": policy_type,
        "trigger_reason": trigger_reason,
        "display": {
            "text": f"{policy_name} Policy",
            "tooltip": f"Fallback triggered by {policy_name}: {trigger_reason}",
            "color": config["color"],
            "icon": config["icon"],
            "priority": config["priority"],
            "description": config["description"]
        },
        "metadata": {
            "triggered_at": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "is_policy_driven": True,
            "requires_acknowledgment": badge_type in [PolicyBadgeType.COMPLIANCE, PolicyBadgeType.SECURITY]
        }
    }
    
    return policy_badge

@app.post("/fallback/user-response/{event_id}")
async def record_user_response(
    event_id: str,
    accepted: bool,
    response_time_seconds: int
):
    """Record user's response to fallback suggestion"""
    
    if event_id not in fallback_events_store:
        raise HTTPException(status_code=404, detail="Fallback event not found")
    
    fallback_event = fallback_events_store[event_id]
    fallback_event.user_accepted_fallback = accepted
    fallback_event.user_response_time_seconds = response_time_seconds
    fallback_event.resolved_at = datetime.utcnow()
    
    # Update session if user accepted
    if accepted:
        session = fallback_sessions_store.get(fallback_event.session_id)
        if session:
            session.current_mode = fallback_event.target_mode
            session.fallback_chain.append(fallback_event.target_mode)
            session.last_activity = datetime.utcnow()
    
    logger.info(f"User {'accepted' if accepted else 'rejected'} fallback {event_id}")
    
    return {"status": "recorded", "accepted": accepted}

@app.get("/fallback/session/{session_id}", response_model=FallbackSession)
async def get_fallback_session(session_id: str):
    """Get current fallback session state"""
    if session_id not in fallback_sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return fallback_sessions_store[session_id]

@app.get("/fallback/events/{session_id}")
async def get_session_fallback_events(session_id: str):
    """Get all fallback events for a session"""
    events = [
        event for event in fallback_events_store.values()
        if event.session_id == session_id
    ]
    
    # Sort by timestamp
    events.sort(key=lambda x: x.triggered_at)
    
    return {"session_id": session_id, "events": events}

@app.get("/fallback/analytics/patterns")
async def get_fallback_analytics(tenant_id: str, days: int = 30):
    """Get analytics on fallback patterns and effectiveness"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    recent_events = [
        event for event in fallback_events_store.values()
        if event.tenant_id == tenant_id and event.triggered_at >= cutoff_date
    ]
    
    analytics = {
        "total_fallbacks": len(recent_events),
        "by_trigger": {},
        "by_source_mode": {},
        "by_target_mode": {},
        "by_severity": {},
        "user_acceptance_rate": 0,
        "avg_response_time_seconds": 0,
        "most_common_path": None
    }
    
    # Analyze by trigger
    for trigger in FallbackTrigger:
        trigger_events = [e for e in recent_events if e.trigger == trigger]
        analytics["by_trigger"][trigger.value] = len(trigger_events)
    
    # Analyze by mode transitions
    for event in recent_events:
        source = event.source_mode.value
        target = event.target_mode.value
        
        if source not in analytics["by_source_mode"]:
            analytics["by_source_mode"][source] = 0
        analytics["by_source_mode"][source] += 1
        
        if target not in analytics["by_target_mode"]:
            analytics["by_target_mode"][target] = 0
        analytics["by_target_mode"][target] += 1
    
    # Analyze by severity
    for severity in FallbackSeverity:
        severity_events = [e for e in recent_events if e.severity == severity]
        analytics["by_severity"][severity.value] = len(severity_events)
    
    # Calculate user acceptance rate
    user_responded_events = [e for e in recent_events if e.user_accepted_fallback is not None]
    if user_responded_events:
        accepted_count = len([e for e in user_responded_events if e.user_accepted_fallback])
        analytics["user_acceptance_rate"] = accepted_count / len(user_responded_events) * 100
        
        # Calculate average response time
        response_times = [e.user_response_time_seconds for e in user_responded_events if e.user_response_time_seconds]
        if response_times:
            analytics["avg_response_time_seconds"] = sum(response_times) / len(response_times)
    
    # Find most common fallback path
    paths = {}
    for event in recent_events:
        path = f"{event.source_mode.value}_to_{event.target_mode.value}"
        paths[path] = paths.get(path, 0) + 1
    
    if paths:
        analytics["most_common_path"] = max(paths.items(), key=lambda x: x[1])[0]
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA UX Fallback Flows Service", "task": "3.3.33"}
