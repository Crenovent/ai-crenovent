"""
Task 3.3.32: Role-Based Conversational Restrictions Service
- Restrict conversational actions by user role/persona
- Safe conversational UX with orchestrator integration
- RBAC enforcement for ML node overrides and workflow modifications
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Role-Based Conversational Restrictions Service")
logger = logging.getLogger(__name__)

class UserRole(str, Enum):
    BUSINESS_USER = "business_user"
    ACCOUNT_EXECUTIVE = "account_executive"
    SALES_ANALYST = "sales_analyst"
    DATA_SCIENTIST = "data_scientist"
    ADMIN = "admin"
    COMPLIANCE_OFFICER = "compliance_officer"
    CRO = "cro"

class ConversationalAction(str, Enum):
    # Workflow actions
    CREATE_WORKFLOW = "create_workflow"
    MODIFY_WORKFLOW = "modify_workflow"
    DELETE_WORKFLOW = "delete_workflow"
    EXECUTE_WORKFLOW = "execute_workflow"
    
    # Node actions
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    CONFIGURE_NODE = "configure_node"
    
    # ML-specific actions
    OVERRIDE_ML_PREDICTION = "override_ml_prediction"
    MODIFY_ML_PARAMETERS = "modify_ml_parameters"
    RETRAIN_MODEL = "retrain_model"
    
    # Data actions
    ACCESS_SENSITIVE_DATA = "access_sensitive_data"
    EXPORT_DATA = "export_data"
    MODIFY_DATA_SOURCE = "modify_data_source"
    
    # Governance actions
    BYPASS_APPROVAL = "bypass_approval"
    MODIFY_COMPLIANCE_RULES = "modify_compliance_rules"
    ACCESS_AUDIT_LOGS = "access_audit_logs"

class NodeType(str, Enum):
    RBA_NODE = "rba_node"
    ML_NODE = "ml_node"
    DATA_SOURCE = "data_source"
    NOTIFICATION = "notification"
    GOVERNANCE = "governance"
    INTEGRATION = "integration"

class RestrictionLevel(str, Enum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    REQUIRES_APPROVAL = "requires_approval"
    FORBIDDEN = "forbidden"

class ConversationalRestriction(BaseModel):
    restriction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_role: UserRole
    
    # Action restrictions
    action: ConversationalAction
    node_types: List[NodeType] = Field(default_factory=list)  # Empty means all node types
    restriction_level: RestrictionLevel
    
    # Context conditions
    conditions: Dict[str, Any] = Field(default_factory=dict)  # Additional conditions
    
    # Restriction details
    reason: str
    alternative_suggestion: Optional[str] = None
    approval_required_from: List[UserRole] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class ConversationalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    user_role: UserRole
    
    # Request details
    action: ConversationalAction
    target_node_type: Optional[NodeType] = None
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    
    # Request context
    natural_language_request: str
    parsed_intent: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RestrictionCheckResult(BaseModel):
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    tenant_id: str
    user_role: UserRole
    
    # Check results
    is_allowed: bool
    restriction_level: RestrictionLevel
    applied_restrictions: List[ConversationalRestriction] = Field(default_factory=list)
    
    # Response details
    response_message: str
    alternative_suggestions: List[str] = Field(default_factory=list)
    approval_required: bool = False
    approval_required_from: List[UserRole] = Field(default_factory=list)
    
    # Audit
    checked_at: datetime = Field(default_factory=datetime.utcnow)

class ApprovalRequest(BaseModel):
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_request_id: str
    tenant_id: str
    
    # Requesting user
    requesting_user_id: str
    requesting_user_role: UserRole
    
    # Approval details
    required_approver_roles: List[UserRole]
    business_justification: str
    
    # Status
    status: str = "pending"  # pending, approved, rejected
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
restrictions_store: Dict[str, ConversationalRestriction] = {}
requests_store: Dict[str, ConversationalRequest] = {}
approval_requests_store: Dict[str, ApprovalRequest] = {}

def _create_default_restrictions():
    """Create default role-based restrictions"""
    
    # Account Executives cannot override ML predictions
    ae_ml_restriction = ConversationalRestriction(
        tenant_id="default",
        user_role=UserRole.ACCOUNT_EXECUTIVE,
        action=ConversationalAction.OVERRIDE_ML_PREDICTION,
        node_types=[NodeType.ML_NODE],
        restriction_level=RestrictionLevel.FORBIDDEN,
        reason="Account Executives are not authorized to override ML model predictions",
        alternative_suggestion="Request review from Sales Analyst or Data Scientist"
    )
    
    # Business users cannot modify ML parameters
    business_user_ml_restriction = ConversationalRestriction(
        tenant_id="default",
        user_role=UserRole.BUSINESS_USER,
        action=ConversationalAction.MODIFY_ML_PARAMETERS,
        node_types=[NodeType.ML_NODE],
        restriction_level=RestrictionLevel.REQUIRES_APPROVAL,
        reason="ML parameter changes require technical review",
        approval_required_from=[UserRole.DATA_SCIENTIST, UserRole.ADMIN]
    )
    
    # Sales Analysts can override but with approval for high-impact models
    analyst_override_restriction = ConversationalRestriction(
        tenant_id="default",
        user_role=UserRole.SALES_ANALYST,
        action=ConversationalAction.OVERRIDE_ML_PREDICTION,
        node_types=[NodeType.ML_NODE],
        restriction_level=RestrictionLevel.REQUIRES_APPROVAL,
        conditions={"model_risk_level": "high"},
        reason="High-risk model overrides require additional approval",
        approval_required_from=[UserRole.CRO, UserRole.COMPLIANCE_OFFICER]
    )
    
    # Only admins can modify compliance rules
    compliance_restriction = ConversationalRestriction(
        tenant_id="default",
        user_role=UserRole.BUSINESS_USER,
        action=ConversationalAction.MODIFY_COMPLIANCE_RULES,
        restriction_level=RestrictionLevel.FORBIDDEN,
        reason="Compliance rule modifications require administrative privileges"
    )
    
    # Data export restrictions
    data_export_restriction = ConversationalRestriction(
        tenant_id="default",
        user_role=UserRole.ACCOUNT_EXECUTIVE,
        action=ConversationalAction.EXPORT_DATA,
        restriction_level=RestrictionLevel.REQUIRES_APPROVAL,
        reason="Data export requires compliance approval",
        approval_required_from=[UserRole.COMPLIANCE_OFFICER]
    )
    
    # Store default restrictions
    restrictions_store[ae_ml_restriction.restriction_id] = ae_ml_restriction
    restrictions_store[business_user_ml_restriction.restriction_id] = business_user_ml_restriction
    restrictions_store[analyst_override_restriction.restriction_id] = analyst_override_restriction
    restrictions_store[compliance_restriction.restriction_id] = compliance_restriction
    restrictions_store[data_export_restriction.restriction_id] = data_export_restriction
    
    logger.info("Created default conversational restrictions")

# Initialize default restrictions
_create_default_restrictions()

@app.post("/conversational/restrictions", response_model=ConversationalRestriction)
async def create_restriction(restriction: ConversationalRestriction):
    """Create a new conversational restriction"""
    restrictions_store[restriction.restriction_id] = restriction
    logger.info(f"Created restriction for {restriction.user_role} on {restriction.action}")
    return restriction

@app.get("/conversational/restrictions", response_model=List[ConversationalRestriction])
async def get_restrictions(
    tenant_id: str,
    user_role: Optional[UserRole] = None,
    action: Optional[ConversationalAction] = None
):
    """Get conversational restrictions"""
    restrictions = [
        r for r in restrictions_store.values()
        if r.tenant_id == tenant_id and r.is_active
    ]
    
    if user_role:
        restrictions = [r for r in restrictions if r.user_role == user_role]
    
    if action:
        restrictions = [r for r in restrictions if r.action == action]
    
    return restrictions

@app.post("/conversational/check-request", response_model=RestrictionCheckResult)
async def check_conversational_request(request: ConversationalRequest):
    """Check if a conversational request is allowed based on user role"""
    
    # Find applicable restrictions
    applicable_restrictions = [
        r for r in restrictions_store.values()
        if (r.tenant_id == request.tenant_id and
            r.user_role == request.user_role and
            r.action == request.action and
            r.is_active and
            _matches_node_type(r, request.target_node_type) and
            _matches_conditions(r, request.context))
    ]
    
    # Determine most restrictive level
    restriction_level = RestrictionLevel.ALLOWED
    applied_restrictions = []
    alternative_suggestions = []
    approval_required_from = []
    
    for restriction in applicable_restrictions:
        applied_restrictions.append(restriction)
        
        if restriction.restriction_level == RestrictionLevel.FORBIDDEN:
            restriction_level = RestrictionLevel.FORBIDDEN
            break
        elif restriction.restriction_level == RestrictionLevel.REQUIRES_APPROVAL:
            restriction_level = RestrictionLevel.REQUIRES_APPROVAL
            approval_required_from.extend(restriction.approval_required_from)
        elif restriction.restriction_level == RestrictionLevel.RESTRICTED:
            if restriction_level == RestrictionLevel.ALLOWED:
                restriction_level = RestrictionLevel.RESTRICTED
        
        if restriction.alternative_suggestion:
            alternative_suggestions.append(restriction.alternative_suggestion)
    
    # Generate response message
    response_message = _generate_response_message(
        request, restriction_level, applied_restrictions
    )
    
    result = RestrictionCheckResult(
        request_id=request.request_id,
        tenant_id=request.tenant_id,
        user_role=request.user_role,
        is_allowed=(restriction_level == RestrictionLevel.ALLOWED),
        restriction_level=restriction_level,
        applied_restrictions=applied_restrictions,
        response_message=response_message,
        alternative_suggestions=list(set(alternative_suggestions)),
        approval_required=(restriction_level == RestrictionLevel.REQUIRES_APPROVAL),
        approval_required_from=list(set(approval_required_from))
    )
    
    # Store request for audit
    requests_store[request.request_id] = request
    
    logger.info(f"Checked request {request.request_id}: {restriction_level.value}")
    return result

def _matches_node_type(restriction: ConversationalRestriction, target_node_type: Optional[NodeType]) -> bool:
    """Check if restriction applies to the target node type"""
    if not restriction.node_types:  # Empty list means applies to all node types
        return True
    if target_node_type is None:
        return True
    return target_node_type in restriction.node_types

def _matches_conditions(restriction: ConversationalRestriction, context: Dict[str, Any]) -> bool:
    """Check if restriction conditions are met"""
    if not restriction.conditions:
        return True
    
    for condition_key, condition_value in restriction.conditions.items():
        context_value = context.get(condition_key)
        if context_value != condition_value:
            return False
    
    return True

def _generate_response_message(
    request: ConversationalRequest,
    restriction_level: RestrictionLevel,
    applied_restrictions: List[ConversationalRestriction]
) -> str:
    """Generate appropriate response message based on restriction level"""
    
    if restriction_level == RestrictionLevel.ALLOWED:
        return f"Your request to {request.action.value.replace('_', ' ')} is approved."
    
    elif restriction_level == RestrictionLevel.FORBIDDEN:
        primary_reason = applied_restrictions[0].reason if applied_restrictions else "Action not permitted"
        return f"I cannot {request.action.value.replace('_', ' ')} because: {primary_reason}"
    
    elif restriction_level == RestrictionLevel.REQUIRES_APPROVAL:
        approvers = []
        for restriction in applied_restrictions:
            approvers.extend([role.value.replace('_', ' ').title() for role in restriction.approval_required_from])
        approver_list = ", ".join(set(approvers))
        return f"Your request to {request.action.value.replace('_', ' ')} requires approval from: {approver_list}"
    
    elif restriction_level == RestrictionLevel.RESTRICTED:
        return f"Your request to {request.action.value.replace('_', ' ')} has restrictions. Please review the alternatives provided."
    
    return "Request processed."

@app.post("/conversational/request-approval", response_model=ApprovalRequest)
async def request_approval(
    request_id: str,
    user_id: str,
    business_justification: str
):
    """Request approval for a restricted conversational action"""
    
    if request_id not in requests_store:
        raise HTTPException(status_code=404, detail="Original request not found")
    
    original_request = requests_store[request_id]
    
    # Find restrictions that require approval
    approval_restrictions = [
        r for r in restrictions_store.values()
        if (r.tenant_id == original_request.tenant_id and
            r.user_role == original_request.user_role and
            r.action == original_request.action and
            r.restriction_level == RestrictionLevel.REQUIRES_APPROVAL)
    ]
    
    if not approval_restrictions:
        raise HTTPException(status_code=400, detail="No approval required for this request")
    
    # Collect all required approver roles
    required_approvers = []
    for restriction in approval_restrictions:
        required_approvers.extend(restriction.approval_required_from)
    
    approval_request = ApprovalRequest(
        original_request_id=request_id,
        tenant_id=original_request.tenant_id,
        requesting_user_id=user_id,
        requesting_user_role=original_request.user_role,
        required_approver_roles=list(set(required_approvers)),
        business_justification=business_justification
    )
    
    approval_requests_store[approval_request.approval_id] = approval_request
    
    logger.info(f"Created approval request {approval_request.approval_id} for {request_id}")
    return approval_request

@app.post("/conversational/approve/{approval_id}")
async def approve_request(
    approval_id: str,
    approver_user_id: str,
    approver_role: UserRole,
    approved: bool,
    notes: Optional[str] = None
):
    """Approve or reject an approval request"""
    
    if approval_id not in approval_requests_store:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    approval_request = approval_requests_store[approval_id]
    
    # Verify approver has required role
    if approver_role not in approval_request.required_approver_roles:
        raise HTTPException(
            status_code=403, 
            detail=f"Role {approver_role.value} is not authorized to approve this request"
        )
    
    # Update approval status
    approval_request.status = "approved" if approved else "rejected"
    approval_request.approved_by = approver_user_id
    approval_request.approved_at = datetime.utcnow()
    
    logger.info(f"Approval request {approval_id} {approval_request.status} by {approver_user_id}")
    
    return {
        "approval_id": approval_id,
        "status": approval_request.status,
        "approved_by": approver_user_id,
        "approved_at": approval_request.approved_at
    }

@app.get("/conversational/my-restrictions/{user_role}")
async def get_my_restrictions(user_role: UserRole, tenant_id: str):
    """Get restrictions that apply to a specific user role"""
    
    user_restrictions = [
        r for r in restrictions_store.values()
        if r.tenant_id == tenant_id and r.user_role == user_role and r.is_active
    ]
    
    # Group by action for easier understanding
    restrictions_by_action = {}
    for restriction in user_restrictions:
        action = restriction.action.value
        if action not in restrictions_by_action:
            restrictions_by_action[action] = []
        restrictions_by_action[action].append({
            "restriction_level": restriction.restriction_level.value,
            "reason": restriction.reason,
            "node_types": [nt.value for nt in restriction.node_types],
            "alternative_suggestion": restriction.alternative_suggestion
        })
    
    return {
        "user_role": user_role.value,
        "total_restrictions": len(user_restrictions),
        "restrictions_by_action": restrictions_by_action
    }

@app.get("/conversational/pending-approvals/{user_role}")
async def get_pending_approvals(user_role: UserRole, tenant_id: str):
    """Get pending approval requests that require this user role"""
    
    pending_approvals = [
        approval for approval in approval_requests_store.values()
        if (approval.tenant_id == tenant_id and
            approval.status == "pending" and
            user_role in approval.required_approver_roles)
    ]
    
    # Enrich with original request details
    enriched_approvals = []
    for approval in pending_approvals:
        original_request = requests_store.get(approval.original_request_id)
        enriched_approval = {
            "approval_id": approval.approval_id,
            "requesting_user": approval.requesting_user_id,
            "requesting_role": approval.requesting_user_role.value,
            "business_justification": approval.business_justification,
            "created_at": approval.created_at,
            "original_request": {
                "action": original_request.action.value if original_request else "unknown",
                "natural_language_request": original_request.natural_language_request if original_request else "unknown"
            }
        }
        enriched_approvals.append(enriched_approval)
    
    return {"pending_approvals": enriched_approvals}

@app.get("/conversational/analytics/restrictions")
async def get_restriction_analytics(tenant_id: str):
    """Get analytics on conversational restrictions and their usage"""
    
    tenant_restrictions = [r for r in restrictions_store.values() if r.tenant_id == tenant_id]
    tenant_requests = [r for r in requests_store.values() if r.tenant_id == tenant_id]
    
    analytics = {
        "total_restrictions": len(tenant_restrictions),
        "restrictions_by_role": {},
        "restrictions_by_action": {},
        "restriction_levels": {},
        "request_patterns": {}
    }
    
    # Analyze restrictions by role
    for role in UserRole:
        role_restrictions = [r for r in tenant_restrictions if r.user_role == role]
        analytics["restrictions_by_role"][role.value] = len(role_restrictions)
    
    # Analyze restrictions by action
    for action in ConversationalAction:
        action_restrictions = [r for r in tenant_restrictions if r.action == action]
        analytics["restrictions_by_action"][action.value] = len(action_restrictions)
    
    # Analyze restriction levels
    for level in RestrictionLevel:
        level_restrictions = [r for r in tenant_restrictions if r.restriction_level == level]
        analytics["restriction_levels"][level.value] = len(level_restrictions)
    
    # Analyze request patterns
    for request in tenant_requests:
        role = request.user_role.value
        if role not in analytics["request_patterns"]:
            analytics["request_patterns"][role] = {"total": 0, "actions": {}}
        
        analytics["request_patterns"][role]["total"] += 1
        action = request.action.value
        if action not in analytics["request_patterns"][role]["actions"]:
            analytics["request_patterns"][role]["actions"][action] = 0
        analytics["request_patterns"][role]["actions"][action] += 1
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Role-Based Conversational Restrictions Service", "task": "3.3.32"}
