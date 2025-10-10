"""
Task 3.2.20: LLM/Prompt safety for Assisted UX nodes
- Prompt filters, output guards, pii scrub
- Safe human-in-loop UX
- Log safety decisions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import re
import logging

app = FastAPI(title="RBIA LLM/Prompt Safety Service")
logger = logging.getLogger(__name__)

class SafetyViolationType(str, Enum):
    PII_DETECTED = "pii_detected"
    PROMPT_INJECTION = "prompt_injection"
    HARMFUL_CONTENT = "harmful_content"
    POLICY_VIOLATION = "policy_violation"

class SafetyAction(str, Enum):
    BLOCK = "block"
    SANITIZE = "sanitize"
    FLAG = "flag"
    ALLOW = "allow"

class SafetyCheckRequest(BaseModel):
    content: str = Field(..., description="Content to check for safety")
    content_type: str = Field(..., description="Type: 'prompt' or 'output'")
    node_id: str = Field(..., description="Assisted UX node identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="User identifier")

class SafetyViolation(BaseModel):
    violation_type: SafetyViolationType
    severity: str = Field(..., description="low, medium, high, critical")
    detected_content: str = Field(..., description="Content that triggered violation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    suggested_action: SafetyAction

class SafetyCheckResult(BaseModel):
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    is_safe: bool = Field(..., description="Whether content is safe")
    action_taken: SafetyAction = Field(..., description="Action taken")
    violations: List[SafetyViolation] = Field(default=[], description="Detected violations")
    sanitized_content: Optional[str] = Field(None, description="Sanitized version if applicable")
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

# In-memory storage for safety decisions (replace with actual database)
safety_logs: List[Dict[str, Any]] = []

# PII detection patterns
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
}

# Prompt injection patterns
INJECTION_PATTERNS = [
    r'ignore\s+previous\s+instructions',
    r'forget\s+everything',
    r'new\s+instructions?:',
    r'system\s*:\s*you\s+are',
    r'<\s*script\s*>',
    r'javascript\s*:',
    r'eval\s*\(',
    r'exec\s*\('
]

# Harmful content patterns
HARMFUL_PATTERNS = [
    r'\b(kill|murder|suicide|bomb|weapon)\b',
    r'\b(hack|exploit|vulnerability)\b',
    r'\b(illegal|fraud|scam)\b'
]

@app.post("/safety/check", response_model=SafetyCheckResult)
async def check_content_safety(request: SafetyCheckRequest):
    """
    Check content for safety violations including PII, prompt injection, and harmful content
    """
    start_time = datetime.utcnow()
    
    violations = []
    action_taken = SafetyAction.ALLOW
    sanitized_content = None
    
    # Check for PII
    pii_violations = _detect_pii(request.content)
    violations.extend(pii_violations)
    
    # Check for prompt injection (only for prompts)
    if request.content_type == "prompt":
        injection_violations = _detect_prompt_injection(request.content)
        violations.extend(injection_violations)
    
    # Check for harmful content
    harmful_violations = _detect_harmful_content(request.content)
    violations.extend(harmful_violations)
    
    # Determine action based on violations
    if violations:
        max_severity = _get_max_severity(violations)
        if max_severity in ["critical", "high"]:
            action_taken = SafetyAction.BLOCK
        elif max_severity == "medium":
            action_taken = SafetyAction.SANITIZE
            sanitized_content = _sanitize_content(request.content, violations)
        else:
            action_taken = SafetyAction.FLAG
    
    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    
    result = SafetyCheckResult(
        is_safe=len(violations) == 0,
        action_taken=action_taken,
        violations=violations,
        sanitized_content=sanitized_content,
        processing_time_ms=processing_time
    )
    
    # Log safety decision
    await _log_safety_decision(request, result)
    
    return result

@app.post("/safety/scrub-pii")
async def scrub_pii_from_content(request: SafetyCheckRequest):
    """
    Remove or mask PII from content
    """
    scrubbed_content = request.content
    pii_found = []
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.finditer(pattern, scrubbed_content, re.IGNORECASE)
        for match in matches:
            pii_found.append({
                "type": pii_type,
                "value": match.group(),
                "position": match.span()
            })
            # Replace with masked version
            if pii_type == "email":
                scrubbed_content = scrubbed_content.replace(match.group(), "[EMAIL_REDACTED]")
            elif pii_type == "phone":
                scrubbed_content = scrubbed_content.replace(match.group(), "[PHONE_REDACTED]")
            elif pii_type == "ssn":
                scrubbed_content = scrubbed_content.replace(match.group(), "[SSN_REDACTED]")
            elif pii_type == "credit_card":
                scrubbed_content = scrubbed_content.replace(match.group(), "[CARD_REDACTED]")
            elif pii_type == "ip_address":
                scrubbed_content = scrubbed_content.replace(match.group(), "[IP_REDACTED]")
    
    return {
        "original_content": request.content,
        "scrubbed_content": scrubbed_content,
        "pii_found": pii_found,
        "scrub_count": len(pii_found)
    }

def _detect_pii(content: str) -> List[SafetyViolation]:
    """Detect PII in content"""
    violations = []
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.PII_DETECTED,
                severity="high",
                detected_content=match.group(),
                confidence=0.9,
                suggested_action=SafetyAction.SANITIZE
            ))
    
    return violations

def _detect_prompt_injection(content: str) -> List[SafetyViolation]:
    """Detect prompt injection attempts"""
    violations = []
    
    for pattern in INJECTION_PATTERNS:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.PROMPT_INJECTION,
                severity="critical",
                detected_content=match.group(),
                confidence=0.85,
                suggested_action=SafetyAction.BLOCK
            ))
    
    return violations

def _detect_harmful_content(content: str) -> List[SafetyViolation]:
    """Detect harmful content"""
    violations = []
    
    for pattern in HARMFUL_PATTERNS:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.HARMFUL_CONTENT,
                severity="high",
                detected_content=match.group(),
                confidence=0.8,
                suggested_action=SafetyAction.BLOCK
            ))
    
    return violations

def _get_max_severity(violations: List[SafetyViolation]) -> str:
    """Get maximum severity from violations"""
    severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    max_severity = "low"
    
    for violation in violations:
        if severity_order[violation.severity] > severity_order[max_severity]:
            max_severity = violation.severity
    
    return max_severity

def _sanitize_content(content: str, violations: List[SafetyViolation]) -> str:
    """Sanitize content by removing or masking violations"""
    sanitized = content
    
    for violation in violations:
        if violation.violation_type == SafetyViolationType.PII_DETECTED:
            sanitized = sanitized.replace(violation.detected_content, "[REDACTED]")
    
    return sanitized

async def _log_safety_decision(request: SafetyCheckRequest, result: SafetyCheckResult):
    """Log safety decision for audit trail"""
    log_entry = {
        "check_id": result.check_id,
        "timestamp": result.checked_at.isoformat(),
        "node_id": request.node_id,
        "tenant_id": request.tenant_id,
        "user_id": request.user_id,
        "content_type": request.content_type,
        "is_safe": result.is_safe,
        "action_taken": result.action_taken.value,
        "violations_count": len(result.violations),
        "violations": [
            {
                "type": v.violation_type.value,
                "severity": v.severity,
                "confidence": v.confidence
            } for v in result.violations
        ],
        "processing_time_ms": result.processing_time_ms
    }
    
    safety_logs.append(log_entry)
    logger.info(f"Safety decision logged: {result.check_id} - Action: {result.action_taken.value}")

@app.get("/safety/logs")
async def get_safety_logs(
    tenant_id: Optional[str] = None,
    node_id: Optional[str] = None,
    limit: int = 100
):
    """Get safety decision logs"""
    filtered_logs = safety_logs
    
    if tenant_id:
        filtered_logs = [log for log in filtered_logs if log.get("tenant_id") == tenant_id]
    
    if node_id:
        filtered_logs = [log for log in filtered_logs if log.get("node_id") == node_id]
    
    return {
        "logs": filtered_logs[-limit:],
        "total_count": len(filtered_logs)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA LLM/Prompt Safety Service",
        "task": "3.2.20",
        "safety_checks_logged": len(safety_logs)
    }

