"""
Task 3.3.51: AI Assistant Guardrails Service
- Safety compliance with configurable guardrails per tenant
- Prevent financial advice, HR decisions, and other restricted content
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging
import re

app = FastAPI(title="RBIA AI Assistant Guardrails Service")
logger = logging.getLogger(__name__)

class GuardrailType(str, Enum):
    FINANCIAL_ADVICE = "financial_advice"
    HR_DECISIONS = "hr_decisions"
    MEDICAL_ADVICE = "medical_advice"
    LEGAL_ADVICE = "legal_advice"
    PERSONAL_DATA = "personal_data"
    DISCRIMINATORY_CONTENT = "discriminatory_content"
    HARMFUL_CONTENT = "harmful_content"
    CONFIDENTIAL_INFO = "confidential_info"
    COMPETITIVE_INFO = "competitive_info"
    REGULATORY_RESTRICTED = "regulatory_restricted"

class ViolationSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class GuardrailAction(str, Enum):
    WARN = "warn"
    BLOCK = "block"
    REDIRECT = "redirect"
    ESCALATE = "escalate"
    LOG_ONLY = "log_only"

class GuardrailRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Rule definition
    rule_name: str
    guardrail_type: GuardrailType
    description: str
    
    # Detection patterns
    trigger_patterns: List[str] = Field(default_factory=list)  # Regex patterns
    trigger_keywords: List[str] = Field(default_factory=list)
    context_keywords: List[str] = Field(default_factory=list)  # Context that makes it more likely
    
    # Response configuration
    violation_severity: ViolationSeverity
    action: GuardrailAction
    
    # Messages
    user_message: str  # Message shown to user
    internal_message: str  # Message for logs/alerts
    redirect_suggestion: Optional[str] = None  # Alternative suggestion
    
    # Settings
    is_active: bool = True
    confidence_threshold: float = 0.7  # Confidence needed to trigger
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class GuardrailViolation(BaseModel):
    violation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    session_id: str
    
    # Violation details
    rule_id: str
    guardrail_type: GuardrailType
    severity: ViolationSeverity
    
    # Content
    original_query: str
    detected_patterns: List[str] = Field(default_factory=list)
    confidence_score: float
    
    # Response
    action_taken: GuardrailAction
    user_message: str
    alternative_response: Optional[str] = None
    
    # Context
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class GuardrailCheck(BaseModel):
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Input
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    violations_detected: List[GuardrailViolation] = Field(default_factory=list)
    highest_severity: Optional[ViolationSeverity] = None
    recommended_action: GuardrailAction = GuardrailAction.LOG_ONLY
    
    # Response
    is_allowed: bool = True
    response_message: str = ""
    alternative_suggestions: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
guardrail_rules_store: Dict[str, GuardrailRule] = {}
guardrail_violations_store: Dict[str, GuardrailViolation] = {}

def _create_default_guardrails():
    """Create default guardrail rules"""
    
    # Financial advice guardrail
    financial_rule = GuardrailRule(
        tenant_id="default",
        rule_name="Financial Advice Prevention",
        guardrail_type=GuardrailType.FINANCIAL_ADVICE,
        description="Prevent providing specific financial advice or investment recommendations",
        trigger_patterns=[
            r"\b(should I invest|buy.*stock|sell.*stock|financial advice)\b",
            r"\b(portfolio recommendation|investment strategy|stock pick)\b",
            r"\b(guaranteed return|sure bet|can't lose)\b"
        ],
        trigger_keywords=["invest", "stock", "portfolio", "trading", "financial advice", "money management"],
        context_keywords=["dollars", "profit", "loss", "market", "shares"],
        violation_severity=ViolationSeverity.HIGH,
        action=GuardrailAction.BLOCK,
        user_message="I cannot provide specific financial advice. Please consult with a qualified financial advisor for investment recommendations.",
        internal_message="Financial advice request blocked",
        redirect_suggestion="I can help you understand general business metrics and data analysis instead."
    )
    
    # HR decisions guardrail
    hr_rule = GuardrailRule(
        tenant_id="default",
        rule_name="HR Decision Prevention",
        guardrail_type=GuardrailType.HR_DECISIONS,
        description="Prevent making or recommending HR decisions like hiring, firing, or promotions",
        trigger_patterns=[
            r"\b(should I fire|should I hire|who to promote|salary recommendation)\b",
            r"\b(terminate.*employee|lay.*off|performance review score)\b",
            r"\b(discriminate|bias.*hiring|unfair treatment)\b"
        ],
        trigger_keywords=["fire", "hire", "promote", "terminate", "salary", "performance review"],
        context_keywords=["employee", "staff", "HR", "human resources", "personnel"],
        violation_severity=ViolationSeverity.CRITICAL,
        action=GuardrailAction.BLOCK,
        user_message="I cannot make HR decisions or recommendations about personnel matters. Please consult with your HR department.",
        internal_message="HR decision request blocked",
        redirect_suggestion="I can help you analyze workforce data and trends for informational purposes."
    )
    
    # Personal data guardrail
    personal_data_rule = GuardrailRule(
        tenant_id="default",
        rule_name="Personal Data Protection",
        guardrail_type=GuardrailType.PERSONAL_DATA,
        description="Prevent processing or exposing personal identifying information",
        trigger_patterns=[
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"  # Credit card pattern
        ],
        trigger_keywords=["SSN", "social security", "credit card", "personal information"],
        violation_severity=ViolationSeverity.CRITICAL,
        action=GuardrailAction.BLOCK,
        user_message="I cannot process personal identifying information. Please remove any PII from your request.",
        internal_message="PII detected and blocked",
        redirect_suggestion="Please rephrase your question without including personal information."
    )
    
    # Discriminatory content guardrail
    discrimination_rule = GuardrailRule(
        tenant_id="default",
        rule_name="Anti-Discrimination Filter",
        guardrail_type=GuardrailType.DISCRIMINATORY_CONTENT,
        description="Prevent discriminatory language or biased recommendations",
        trigger_patterns=[
            r"\b(race|gender|age|religion|disability).*bias\b",
            r"\b(discriminate.*against|prefer.*because.*of)\b"
        ],
        trigger_keywords=["discriminate", "bias", "prejudice", "stereotype"],
        context_keywords=["race", "gender", "age", "religion", "disability", "ethnicity"],
        violation_severity=ViolationSeverity.HIGH,
        action=GuardrailAction.BLOCK,
        user_message="I cannot provide biased or discriminatory responses. All analysis should be fair and objective.",
        internal_message="Discriminatory content blocked",
        redirect_suggestion="I can help you with objective data analysis that treats all groups fairly."
    )
    
    # Legal advice guardrail
    legal_rule = GuardrailRule(
        tenant_id="default",
        rule_name="Legal Advice Prevention",
        guardrail_type=GuardrailType.LEGAL_ADVICE,
        description="Prevent providing specific legal advice or legal interpretations",
        trigger_patterns=[
            r"\b(legal advice|should I sue|is this legal|court case)\b",
            r"\b(lawyer recommend|legal strategy|lawsuit)\b"
        ],
        trigger_keywords=["legal", "lawsuit", "court", "attorney", "lawyer", "litigation"],
        violation_severity=ViolationSeverity.HIGH,
        action=GuardrailAction.WARN,
        user_message="I cannot provide legal advice. Please consult with a qualified attorney for legal matters.",
        internal_message="Legal advice request warned",
        redirect_suggestion="I can help you with general business process analysis instead."
    )
    
    # Store default rules
    guardrail_rules_store[financial_rule.rule_id] = financial_rule
    guardrail_rules_store[hr_rule.rule_id] = hr_rule
    guardrail_rules_store[personal_data_rule.rule_id] = personal_data_rule
    guardrail_rules_store[discrimination_rule.rule_id] = discrimination_rule
    guardrail_rules_store[legal_rule.rule_id] = legal_rule
    
    logger.info("Created default guardrail rules")

# Initialize default guardrails
_create_default_guardrails()

@app.post("/guardrails/rules", response_model=GuardrailRule)
async def create_guardrail_rule(rule: GuardrailRule):
    """Create a new guardrail rule"""
    guardrail_rules_store[rule.rule_id] = rule
    logger.info(f"Created guardrail rule: {rule.rule_name}")
    return rule

@app.get("/guardrails/rules", response_model=List[GuardrailRule])
async def get_guardrail_rules(
    tenant_id: str,
    guardrail_type: Optional[GuardrailType] = None,
    active_only: bool = True
):
    """Get guardrail rules for a tenant"""
    
    rules = [
        rule for rule in guardrail_rules_store.values()
        if rule.tenant_id == tenant_id or rule.tenant_id == "default"
    ]
    
    if active_only:
        rules = [rule for rule in rules if rule.is_active]
    
    if guardrail_type:
        rules = [rule for rule in rules if rule.guardrail_type == guardrail_type]
    
    return rules

@app.post("/guardrails/check", response_model=GuardrailCheck)
async def check_guardrails(
    tenant_id: str,
    user_id: str,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
):
    """Check query against guardrails and return violations"""
    
    if context is None:
        context = {}
    
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Get applicable rules
    applicable_rules = [
        rule for rule in guardrail_rules_store.values()
        if (rule.tenant_id == tenant_id or rule.tenant_id == "default") and rule.is_active
    ]
    
    # Check each rule
    violations = []
    
    for rule in applicable_rules:
        violation = _check_rule_violation(rule, query, context, tenant_id, user_id, session_id)
        if violation:
            violations.append(violation)
            guardrail_violations_store[violation.violation_id] = violation
    
    # Determine overall response
    check_result = _determine_overall_response(violations, query, tenant_id, user_id)
    
    logger.info(f"Guardrail check for user {user_id}: {len(violations)} violations detected")
    return check_result

def _check_rule_violation(
    rule: GuardrailRule,
    query: str,
    context: Dict[str, Any],
    tenant_id: str,
    user_id: str,
    session_id: str
) -> Optional[GuardrailViolation]:
    """Check if a specific rule is violated"""
    
    query_lower = query.lower()
    detected_patterns = []
    confidence_score = 0.0
    
    # Check trigger patterns (regex)
    for pattern in rule.trigger_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            detected_patterns.append(pattern)
            confidence_score += 0.4
    
    # Check trigger keywords
    for keyword in rule.trigger_keywords:
        if keyword.lower() in query_lower:
            detected_patterns.append(f"keyword: {keyword}")
            confidence_score += 0.2
    
    # Check context keywords (boost confidence if present)
    context_boost = 0.0
    for context_keyword in rule.context_keywords:
        if context_keyword.lower() in query_lower:
            context_boost += 0.1
    
    confidence_score += min(context_boost, 0.3)  # Cap context boost
    
    # Normalize confidence score
    confidence_score = min(confidence_score, 1.0)
    
    # Check if violation threshold is met
    if confidence_score >= rule.confidence_threshold and detected_patterns:
        violation = GuardrailViolation(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            rule_id=rule.rule_id,
            guardrail_type=rule.guardrail_type,
            severity=rule.violation_severity,
            original_query=query,
            detected_patterns=detected_patterns,
            confidence_score=confidence_score,
            action_taken=rule.action,
            user_message=rule.user_message,
            alternative_response=rule.redirect_suggestion,
            conversation_context=context
        )
        
        return violation
    
    return None

def _determine_overall_response(
    violations: List[GuardrailViolation],
    query: str,
    tenant_id: str,
    user_id: str
) -> GuardrailCheck:
    """Determine overall response based on all violations"""
    
    check = GuardrailCheck(
        tenant_id=tenant_id,
        user_id=user_id,
        query=query,
        violations_detected=violations
    )
    
    if not violations:
        check.is_allowed = True
        check.response_message = "Query approved - no guardrail violations detected."
        return check
    
    # Find highest severity violation
    severity_order = {
        ViolationSeverity.CRITICAL: 4,
        ViolationSeverity.HIGH: 3,
        ViolationSeverity.MEDIUM: 2,
        ViolationSeverity.LOW: 1
    }
    
    highest_severity_violation = max(violations, key=lambda v: severity_order[v.severity])
    check.highest_severity = highest_severity_violation.severity
    
    # Determine action based on highest severity violation
    if highest_severity_violation.action_taken == GuardrailAction.BLOCK:
        check.is_allowed = False
        check.recommended_action = GuardrailAction.BLOCK
        check.response_message = highest_severity_violation.user_message
        
        # Collect alternative suggestions
        for violation in violations:
            if violation.alternative_response:
                check.alternative_suggestions.append(violation.alternative_response)
    
    elif highest_severity_violation.action_taken == GuardrailAction.WARN:
        check.is_allowed = True
        check.recommended_action = GuardrailAction.WARN
        check.response_message = f"Warning: {highest_severity_violation.user_message}"
    
    elif highest_severity_violation.action_taken == GuardrailAction.REDIRECT:
        check.is_allowed = False
        check.recommended_action = GuardrailAction.REDIRECT
        check.response_message = highest_severity_violation.user_message
        if highest_severity_violation.alternative_response:
            check.alternative_suggestions.append(highest_severity_violation.alternative_response)
    
    else:  # LOG_ONLY or ESCALATE
        check.is_allowed = True
        check.recommended_action = highest_severity_violation.action_taken
        check.response_message = "Query processed with monitoring."
    
    return check

@app.get("/guardrails/violations", response_model=List[GuardrailViolation])
async def get_violations(
    tenant_id: str,
    user_id: Optional[str] = None,
    guardrail_type: Optional[GuardrailType] = None,
    severity: Optional[ViolationSeverity] = None,
    limit: int = 100
):
    """Get guardrail violations"""
    
    violations = [
        violation for violation in guardrail_violations_store.values()
        if violation.tenant_id == tenant_id
    ]
    
    if user_id:
        violations = [v for v in violations if v.user_id == user_id]
    
    if guardrail_type:
        violations = [v for v in violations if v.guardrail_type == guardrail_type]
    
    if severity:
        violations = [v for v in violations if v.severity == severity]
    
    # Sort by timestamp (most recent first)
    violations.sort(key=lambda x: x.timestamp, reverse=True)
    
    return violations[:limit]

@app.put("/guardrails/rules/{rule_id}")
async def update_guardrail_rule(
    rule_id: str,
    updates: Dict[str, Any]
):
    """Update a guardrail rule"""
    
    if rule_id not in guardrail_rules_store:
        raise HTTPException(status_code=404, detail="Guardrail rule not found")
    
    rule = guardrail_rules_store[rule_id]
    
    # Update allowed fields
    allowed_updates = [
        "is_active", "confidence_threshold", "violation_severity", 
        "action", "user_message", "internal_message", "redirect_suggestion"
    ]
    
    for field, value in updates.items():
        if field in allowed_updates and hasattr(rule, field):
            setattr(rule, field, value)
    
    logger.info(f"Updated guardrail rule {rule_id}")
    return {"status": "updated", "rule_id": rule_id}

@app.delete("/guardrails/rules/{rule_id}")
async def delete_guardrail_rule(rule_id: str):
    """Delete (deactivate) a guardrail rule"""
    
    if rule_id not in guardrail_rules_store:
        raise HTTPException(status_code=404, detail="Guardrail rule not found")
    
    rule = guardrail_rules_store[rule_id]
    rule.is_active = False
    
    logger.info(f"Deactivated guardrail rule {rule_id}")
    return {"status": "deactivated", "rule_id": rule_id}

@app.get("/guardrails/analytics/violations")
async def get_violation_analytics(tenant_id: str, days: int = 30):
    """Get analytics on guardrail violations"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    recent_violations = [
        violation for violation in guardrail_violations_store.values()
        if violation.tenant_id == tenant_id and violation.timestamp >= cutoff_date
    ]
    
    if not recent_violations:
        return {"message": "No violations found in the specified period"}
    
    analytics = {
        "total_violations": len(recent_violations),
        "by_type": {},
        "by_severity": {},
        "by_action": {},
        "top_violating_users": {},
        "violation_trend": "stable",  # Simplified
        "most_common_patterns": []
    }
    
    # Analyze by type
    for guardrail_type in GuardrailType:
        type_violations = [v for v in recent_violations if v.guardrail_type == guardrail_type]
        analytics["by_type"][guardrail_type.value] = len(type_violations)
    
    # Analyze by severity
    for severity in ViolationSeverity:
        severity_violations = [v for v in recent_violations if v.severity == severity]
        analytics["by_severity"][severity.value] = len(severity_violations)
    
    # Analyze by action
    for action in GuardrailAction:
        action_violations = [v for v in recent_violations if v.action_taken == action]
        analytics["by_action"][action.value] = len(action_violations)
    
    # Top violating users
    user_counts = {}
    for violation in recent_violations:
        user_counts[violation.user_id] = user_counts.get(violation.user_id, 0) + 1
    
    analytics["top_violating_users"] = dict(sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Most common patterns
    pattern_counts = {}
    for violation in recent_violations:
        for pattern in violation.detected_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    analytics["most_common_patterns"] = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA AI Assistant Guardrails Service", "task": "3.3.51"}
