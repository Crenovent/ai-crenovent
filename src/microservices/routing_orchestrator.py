"""
Routing Orchestrator Service
Handles mode selection based on policy and confidence
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import uuid
from datetime import datetime

app = FastAPI(
    title="Routing Orchestrator Service",
    description="Handles mode selection based on policy and confidence",
    version="1.0.0"
)

class UIMode(str, Enum):
    UI = "ui"
    HYBRID = "hybrid"
    AGENT = "agent"

class ConfidenceLevel(str, Enum):
    LOW = "low"      # < 0.5
    MEDIUM = "medium" # 0.5 - 0.8
    HIGH = "high"    # > 0.8

class PolicyRule(BaseModel):
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]  # JSON condition
    action: UIMode
    priority: int = Field(default=1, ge=1, le=10)
    tenant_id: str
    region: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RoutingRequest(BaseModel):
    tenant_id: str
    user_id: str
    session_id: str
    service_name: str  # calendar, letsmeet, cruxx
    operation_type: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Optional[Dict[str, Any]] = None

class RoutingResponse(BaseModel):
    recommended_mode: UIMode
    confidence_level: ConfidenceLevel
    applied_rule: Optional[str] = None
    reasoning: str
    fallback_mode: Optional[UIMode] = None
    ui_metadata: Dict[str, Any] = Field(default_factory=dict)

class PolicyEvaluationRequest(BaseModel):
    tenant_id: str
    region: str
    confidence_score: float
    trust_score: float
    service_name: str
    operation_type: str
    context: Dict[str, Any] = Field(default_factory=dict)

# In-memory storage (replace with database in production)
policy_rules_db: Dict[str, PolicyRule] = {}

# Default policy rules
DEFAULT_RULES = [
    {
        "rule_id": "high_confidence_agent",
        "name": "High Confidence Agent Mode",
        "description": "Route to agent mode for high confidence operations",
        "condition": {"confidence_score": {"gte": 0.8}, "trust_score": {"gte": 0.7}},
        "action": UIMode.AGENT,
        "priority": 1,
        "tenant_id": "default",
        "region": "global"
    },
    {
        "rule_id": "medium_confidence_hybrid",
        "name": "Medium Confidence Hybrid Mode",
        "description": "Route to hybrid mode for medium confidence operations",
        "condition": {"confidence_score": {"gte": 0.5, "lt": 0.8}},
        "action": UIMode.HYBRID,
        "priority": 2,
        "tenant_id": "default",
        "region": "global"
    },
    {
        "rule_id": "low_confidence_ui",
        "name": "Low Confidence UI Mode",
        "description": "Route to UI mode for low confidence operations",
        "condition": {"confidence_score": {"lt": 0.5}},
        "action": UIMode.UI,
        "priority": 3,
        "tenant_id": "default",
        "region": "global"
    }
]

# Initialize default rules
for rule_data in DEFAULT_RULES:
    rule = PolicyRule(**rule_data)
    policy_rules_db[rule.rule_id] = rule

def evaluate_condition(condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Evaluate a policy condition against context"""
    for field, constraint in condition.items():
        if field not in context:
            return False
        
        value = context[field]
        
        if isinstance(constraint, dict):
            for op, threshold in constraint.items():
                if op == "gte" and value < threshold:
                    return False
                elif op == "gt" and value <= threshold:
                    return False
                elif op == "lte" and value > threshold:
                    return False
                elif op == "lt" and value >= threshold:
                    return False
                elif op == "eq" and value != threshold:
                    return False
                elif op == "ne" and value == threshold:
                    return False
        elif value != constraint:
            return False
    
    return True

def get_confidence_level(confidence_score: float) -> ConfidenceLevel:
    """Determine confidence level from score"""
    if confidence_score < 0.5:
        return ConfidenceLevel.LOW
    elif confidence_score < 0.8:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.HIGH

@app.post("/routing/route", response_model=RoutingResponse)
async def route_request(request: RoutingRequest):
    """Route a request based on policy and confidence"""
    
    # Build evaluation context
    context = {
        "confidence_score": request.confidence_score,
        "trust_score": request.trust_score,
        "service_name": request.service_name,
        "operation_type": request.operation_type,
        "tenant_id": request.tenant_id,
        **request.context
    }
    
    # Find applicable rules (sorted by priority)
    applicable_rules = []
    for rule in policy_rules_db.values():
        if (rule.is_active and 
            (rule.tenant_id == request.tenant_id or rule.tenant_id == "default") and
            evaluate_condition(rule.condition, context)):
            applicable_rules.append(rule)
    
    # Sort by priority (lower number = higher priority)
    applicable_rules.sort(key=lambda r: r.priority)
    
    # Apply the highest priority rule
    if applicable_rules:
        applied_rule = applicable_rules[0]
        recommended_mode = applied_rule.action
        applied_rule_id = applied_rule.rule_id
        reasoning = f"Applied rule '{applied_rule.name}': {applied_rule.description}"
    else:
        # Default fallback based on confidence
        if request.confidence_score >= 0.8:
            recommended_mode = UIMode.AGENT
        elif request.confidence_score >= 0.5:
            recommended_mode = UIMode.HYBRID
        else:
            recommended_mode = UIMode.UI
        applied_rule_id = None
        reasoning = f"Default routing based on confidence score: {request.confidence_score}"
    
    # Determine fallback mode
    if recommended_mode == UIMode.AGENT:
        fallback_mode = UIMode.HYBRID
    elif recommended_mode == UIMode.HYBRID:
        fallback_mode = UIMode.UI
    else:
        fallback_mode = None
    
    # Generate UI metadata
    ui_metadata = {
        "confidence": request.confidence_score,
        "trust_score": request.trust_score,
        "overrideable": recommended_mode != UIMode.UI,
        "agent_actionable": recommended_mode == UIMode.AGENT,
        "ui_badges": [],
        "suggested_ui_component": None
    }
    
    # Add appropriate badges
    confidence_level = get_confidence_level(request.confidence_score)
    if confidence_level == ConfidenceLevel.HIGH:
        ui_metadata["ui_badges"].append("HIGH_CONFIDENCE")
    elif confidence_level == ConfidenceLevel.MEDIUM:
        ui_metadata["ui_badges"].append("MEDIUM_CONFIDENCE")
    else:
        ui_metadata["ui_badges"].append("LOW_CONFIDENCE")
    
    if request.trust_score >= 0.8:
        ui_metadata["ui_badges"].append("HIGH_TRUST")
    
    # Suggest UI components based on mode
    if recommended_mode == UIMode.AGENT:
        ui_metadata["suggested_ui_component"] = "AGENT_CHAT"
    elif recommended_mode == UIMode.HYBRID:
        ui_metadata["suggested_ui_component"] = "CONFIRMATION_MODAL"
    else:
        ui_metadata["suggested_ui_component"] = "EDIT_FORM"
    
    return RoutingResponse(
        recommended_mode=recommended_mode,
        confidence_level=confidence_level,
        applied_rule=applied_rule_id,
        reasoning=reasoning,
        fallback_mode=fallback_mode,
        ui_metadata=ui_metadata
    )

@app.post("/policies/evaluate", response_model=Dict[str, Any])
async def evaluate_policy(request: PolicyEvaluationRequest):
    """Evaluate policy rules against given context"""
    
    context = {
        "confidence_score": request.confidence_score,
        "trust_score": request.trust_score,
        "service_name": request.service_name,
        "operation_type": request.operation_type,
        "tenant_id": request.tenant_id,
        **request.context
    }
    
    matching_rules = []
    for rule in policy_rules_db.values():
        if (rule.is_active and 
            (rule.tenant_id == request.tenant_id or rule.tenant_id == "default")):
            if evaluate_condition(rule.condition, context):
                matching_rules.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "action": rule.action,
                    "priority": rule.priority
                })
    
    # Sort by priority
    matching_rules.sort(key=lambda r: r["priority"])
    
    return {
        "matching_rules": matching_rules,
        "recommended_action": matching_rules[0]["action"] if matching_rules else UIMode.UI,
        "context": context
    }

@app.post("/policies", response_model=PolicyRule)
async def create_policy_rule(rule: PolicyRule):
    """Create a new policy rule"""
    rule.rule_id = str(uuid.uuid4())
    policy_rules_db[rule.rule_id] = rule
    return rule

@app.get("/policies", response_model=List[PolicyRule])
async def list_policy_rules(tenant_id: str):
    """List policy rules for a tenant"""
    return [rule for rule in policy_rules_db.values() 
            if rule.tenant_id == tenant_id or rule.tenant_id == "default"]

@app.put("/policies/{rule_id}", response_model=PolicyRule)
async def update_policy_rule(rule_id: str, rule_update: Dict[str, Any]):
    """Update a policy rule"""
    if rule_id not in policy_rules_db:
        raise HTTPException(status_code=404, detail="Policy rule not found")
    
    rule = policy_rules_db[rule_id]
    for field, value in rule_update.items():
        if hasattr(rule, field):
            setattr(rule, field, value)
    
    return rule

@app.delete("/policies/{rule_id}")
async def delete_policy_rule(rule_id: str):
    """Delete a policy rule"""
    if rule_id not in policy_rules_db:
        raise HTTPException(status_code=404, detail="Policy rule not found")
    
    del policy_rules_db[rule_id]
    return {"message": "Policy rule deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "routing-orchestrator",
        "timestamp": datetime.utcnow().isoformat(),
        "policy_rules_count": len(policy_rules_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
