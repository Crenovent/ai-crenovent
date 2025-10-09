"""
Task 3.2.34: UX governance in Builder (badges, warnings, tooltips, blockers when gates fail)
- Transparent UX for creators
- React + policy API
- Educates users
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA UX Governance Service")
logger = logging.getLogger(__name__)

class GovernanceElementType(str, Enum):
    BADGE = "badge"
    WARNING = "warning"
    TOOLTIP = "tooltip"
    BLOCKER = "blocker"
    INFO_PANEL = "info_panel"
    PROGRESS_BAR = "progress_bar"

class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"

class PolicyGateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    NOT_APPLICABLE = "not_applicable"

class GovernanceElement(BaseModel):
    element_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    element_type: GovernanceElementType
    severity: SeverityLevel
    title: str
    message: str
    
    # Visual properties
    color: Optional[str] = None  # hex color code
    icon: Optional[str] = None   # icon name/class
    position: Optional[str] = None  # "top-right", "bottom", etc.
    
    # Behavior properties
    dismissible: bool = True
    auto_hide_seconds: Optional[int] = None
    blocks_interaction: bool = False
    
    # Context
    related_policy: Optional[str] = None
    related_workflow: Optional[str] = None
    help_link: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str
    user_id: Optional[str] = None

class PolicyGateResult(BaseModel):
    gate_id: str
    gate_name: str
    policy_name: str
    status: PolicyGateStatus
    
    # Gate details
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    compliance_percentage: Optional[float] = None
    
    # Failure details
    failure_reason: Optional[str] = None
    required_actions: List[str] = Field(default_factory=list)
    
    # Metadata
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str
    workflow_id: Optional[str] = None

class WorkflowGovernanceState(BaseModel):
    workflow_id: str
    workflow_name: str
    tenant_id: str
    user_id: str
    
    # Overall state
    overall_compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    pending_gates: int = 0
    
    # Gate results
    gate_results: List[PolicyGateResult] = Field(default_factory=list)
    
    # UX elements to display
    governance_elements: List[GovernanceElement] = Field(default_factory=list)
    
    # Blocking status
    is_blocked: bool = False
    blocking_reasons: List[str] = Field(default_factory=list)
    
    # Progress tracking
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    next_required_action: Optional[str] = None
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with proper database in production)
workflow_governance_states: Dict[str, WorkflowGovernanceState] = {}
governance_elements_store: List[GovernanceElement] = []

def _generate_governance_elements(gate_results: List[PolicyGateResult], 
                                workflow_id: str, tenant_id: str) -> List[GovernanceElement]:
    """Generate UX governance elements based on policy gate results."""
    elements = []
    
    for gate in gate_results:
        if gate.status == PolicyGateStatus.FAILED:
            # Create blocker for failed critical gates
            if "critical" in gate.gate_name.lower() or "security" in gate.gate_name.lower():
                elements.append(GovernanceElement(
                    element_type=GovernanceElementType.BLOCKER,
                    severity=SeverityLevel.CRITICAL,
                    title=f"Critical Policy Gate Failed: {gate.gate_name}",
                    message=f"This workflow cannot proceed until the {gate.gate_name} policy gate passes. {gate.failure_reason or 'Please review and fix the issues.'}",
                    color="#dc3545",
                    icon="shield-x",
                    blocks_interaction=True,
                    dismissible=False,
                    related_policy=gate.policy_name,
                    related_workflow=workflow_id,
                    tenant_id=tenant_id
                ))
            else:
                # Create warning for non-critical failed gates
                elements.append(GovernanceElement(
                    element_type=GovernanceElementType.WARNING,
                    severity=SeverityLevel.ERROR,
                    title=f"Policy Gate Failed: {gate.gate_name}",
                    message=f"{gate.failure_reason or 'This gate needs attention before deployment.'}",
                    color="#ffc107",
                    icon="exclamation-triangle",
                    blocks_interaction=False,
                    dismissible=True,
                    related_policy=gate.policy_name,
                    related_workflow=workflow_id,
                    tenant_id=tenant_id
                ))
        
        elif gate.status == PolicyGateStatus.PASSED:
            # Create success badge for passed gates
            elements.append(GovernanceElement(
                element_type=GovernanceElementType.BADGE,
                severity=SeverityLevel.SUCCESS,
                title=f"✓ {gate.gate_name}",
                message=f"Policy gate passed with {gate.compliance_percentage or 100}% compliance",
                color="#28a745",
                icon="check-circle",
                position="top-right",
                auto_hide_seconds=5,
                related_policy=gate.policy_name,
                tenant_id=tenant_id
            ))
        
        elif gate.status == PolicyGateStatus.PENDING:
            # Create info tooltip for pending gates
            elements.append(GovernanceElement(
                element_type=GovernanceElementType.TOOLTIP,
                severity=SeverityLevel.INFO,
                title=f"Evaluating: {gate.gate_name}",
                message="This policy gate is currently being evaluated. Please wait...",
                color="#17a2b8",
                icon="clock",
                auto_hide_seconds=10,
                related_policy=gate.policy_name,
                tenant_id=tenant_id
            ))
    
    return elements

def _calculate_compliance_metrics(gate_results: List[PolicyGateResult]) -> Dict[str, Any]:
    """Calculate overall compliance metrics from gate results."""
    if not gate_results:
        return {
            "overall_score": 0.0,
            "total_gates": 0,
            "passed_gates": 0,
            "failed_gates": 0,
            "pending_gates": 0,
            "completion_percentage": 0.0
        }
    
    total_gates = len(gate_results)
    passed_gates = len([g for g in gate_results if g.status == PolicyGateStatus.PASSED])
    failed_gates = len([g for g in gate_results if g.status == PolicyGateStatus.FAILED])
    pending_gates = len([g for g in gate_results if g.status == PolicyGateStatus.PENDING])
    
    # Calculate weighted compliance score
    compliance_scores = []
    for gate in gate_results:
        if gate.status == PolicyGateStatus.PASSED:
            compliance_scores.append(gate.compliance_percentage or 100.0)
        elif gate.status == PolicyGateStatus.FAILED:
            compliance_scores.append(0.0)
        # Skip pending gates from score calculation
    
    overall_score = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.0
    completion_percentage = ((passed_gates + failed_gates) / total_gates) * 100 if total_gates > 0 else 0.0
    
    return {
        "overall_score": round(overall_score, 2),
        "total_gates": total_gates,
        "passed_gates": passed_gates,
        "failed_gates": failed_gates,
        "pending_gates": pending_gates,
        "completion_percentage": round(completion_percentage, 2)
    }

@app.post("/governance/workflow-state", response_model=WorkflowGovernanceState)
async def update_workflow_governance_state(
    workflow_id: str,
    workflow_name: str,
    tenant_id: str,
    user_id: str,
    gate_results: List[PolicyGateResult]
):
    """Update the governance state for a workflow and generate UX elements."""
    
    # Calculate compliance metrics
    metrics = _calculate_compliance_metrics(gate_results)
    
    # Generate governance elements
    governance_elements = _generate_governance_elements(gate_results, workflow_id, tenant_id)
    
    # Determine blocking status
    is_blocked = any(g.status == PolicyGateStatus.FAILED and 
                    ("critical" in g.gate_name.lower() or "security" in g.gate_name.lower()) 
                    for g in gate_results)
    
    blocking_reasons = [
        f"Critical gate failed: {g.gate_name} - {g.failure_reason}"
        for g in gate_results 
        if g.status == PolicyGateStatus.FAILED and 
           ("critical" in g.gate_name.lower() or "security" in g.gate_name.lower())
    ]
    
    # Determine next required action
    next_action = None
    failed_gates = [g for g in gate_results if g.status == PolicyGateStatus.FAILED]
    if failed_gates:
        next_action = f"Fix policy gate: {failed_gates[0].gate_name}"
    elif any(g.status == PolicyGateStatus.PENDING for g in gate_results):
        next_action = "Wait for policy gate evaluation to complete"
    
    # Create or update workflow governance state
    state = WorkflowGovernanceState(
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        tenant_id=tenant_id,
        user_id=user_id,
        overall_compliance_score=metrics["overall_score"],
        total_gates=metrics["total_gates"],
        passed_gates=metrics["passed_gates"],
        failed_gates=metrics["failed_gates"],
        pending_gates=metrics["pending_gates"],
        gate_results=gate_results,
        governance_elements=governance_elements,
        is_blocked=is_blocked,
        blocking_reasons=blocking_reasons,
        completion_percentage=metrics["completion_percentage"],
        next_required_action=next_action
    )
    
    workflow_governance_states[workflow_id] = state
    
    # Store elements for retrieval
    governance_elements_store.extend(governance_elements)
    
    logger.info(f"Updated governance state for workflow {workflow_id}: {metrics['passed_gates']}/{metrics['total_gates']} gates passed")
    
    return state

@app.get("/governance/workflow-state/{workflow_id}", response_model=WorkflowGovernanceState)
async def get_workflow_governance_state(workflow_id: str):
    """Get the current governance state for a workflow."""
    
    if workflow_id not in workflow_governance_states:
        raise HTTPException(status_code=404, detail="Workflow governance state not found")
    
    return workflow_governance_states[workflow_id]

@app.get("/governance/elements/{tenant_id}", response_model=List[GovernanceElement])
async def get_governance_elements(
    tenant_id: str,
    workflow_id: Optional[str] = None,
    element_type: Optional[GovernanceElementType] = None,
    severity: Optional[SeverityLevel] = None,
    active_only: bool = True
):
    """Get governance elements for display in the UI."""
    
    elements = [e for e in governance_elements_store if e.tenant_id == tenant_id]
    
    if workflow_id:
        elements = [e for e in elements if e.related_workflow == workflow_id]
    
    if element_type:
        elements = [e for e in elements if e.element_type == element_type]
    
    if severity:
        elements = [e for e in elements if e.severity == severity]
    
    if active_only:
        # Filter out auto-hidden elements
        current_time = datetime.utcnow()
        elements = [
            e for e in elements 
            if not e.auto_hide_seconds or 
               (current_time - e.created_at).seconds < e.auto_hide_seconds
        ]
    
    return elements

@app.post("/governance/elements/{element_id}/dismiss")
async def dismiss_governance_element(element_id: str, user_id: str):
    """Dismiss a governance element (if dismissible)."""
    
    element = next((e for e in governance_elements_store if e.element_id == element_id), None)
    if not element:
        raise HTTPException(status_code=404, detail="Governance element not found")
    
    if not element.dismissible:
        raise HTTPException(status_code=400, detail="This governance element cannot be dismissed")
    
    # Remove from active elements
    governance_elements_store[:] = [e for e in governance_elements_store if e.element_id != element_id]
    
    logger.info(f"User {user_id} dismissed governance element {element_id}")
    
    return {"status": "dismissed", "element_id": element_id}

@app.get("/governance/compliance-dashboard/{tenant_id}")
async def get_compliance_dashboard(tenant_id: str):
    """Get compliance dashboard data for a tenant."""
    
    tenant_workflows = {
        wid: state for wid, state in workflow_governance_states.items() 
        if state.tenant_id == tenant_id
    }
    
    if not tenant_workflows:
        return {
            "tenant_id": tenant_id,
            "total_workflows": 0,
            "average_compliance_score": 0.0,
            "workflows_blocked": 0,
            "total_policy_gates": 0,
            "gates_passed": 0,
            "gates_failed": 0,
            "gates_pending": 0,
            "recent_elements": []
        }
    
    # Calculate tenant-wide metrics
    total_workflows = len(tenant_workflows)
    average_compliance = sum(w.overall_compliance_score for w in tenant_workflows.values()) / total_workflows
    workflows_blocked = sum(1 for w in tenant_workflows.values() if w.is_blocked)
    
    total_gates = sum(w.total_gates for w in tenant_workflows.values())
    gates_passed = sum(w.passed_gates for w in tenant_workflows.values())
    gates_failed = sum(w.failed_gates for w in tenant_workflows.values())
    gates_pending = sum(w.pending_gates for w in tenant_workflows.values())
    
    # Get recent governance elements
    recent_elements = [
        e for e in governance_elements_store 
        if e.tenant_id == tenant_id
    ][-10:]  # Last 10 elements
    
    return {
        "tenant_id": tenant_id,
        "total_workflows": total_workflows,
        "average_compliance_score": round(average_compliance, 2),
        "workflows_blocked": workflows_blocked,
        "total_policy_gates": total_gates,
        "gates_passed": gates_passed,
        "gates_failed": gates_failed,
        "gates_pending": gates_pending,
        "recent_elements": recent_elements,
        "workflows": list(tenant_workflows.keys())
    }

@app.post("/governance/simulate-gate-failure")
async def simulate_gate_failure(
    workflow_id: str,
    tenant_id: str,
    gate_name: str,
    failure_reason: str,
    is_critical: bool = False
):
    """Simulate a policy gate failure for testing UX elements."""
    
    gate_result = PolicyGateResult(
        gate_id=str(uuid.uuid4()),
        gate_name=gate_name,
        policy_name=f"Policy for {gate_name}",
        status=PolicyGateStatus.FAILED,
        failure_reason=failure_reason,
        required_actions=[f"Fix issues in {gate_name}"],
        tenant_id=tenant_id,
        workflow_id=workflow_id
    )
    
    # Generate governance elements for this failure
    elements = _generate_governance_elements([gate_result], workflow_id, tenant_id)
    governance_elements_store.extend(elements)
    
    logger.info(f"Simulated gate failure: {gate_name} for workflow {workflow_id}")
    
    return {
        "status": "simulated",
        "gate_result": gate_result,
        "generated_elements": len(elements),
        "elements": elements
    }

@app.get("/governance/help/{policy_name}")
async def get_policy_help(policy_name: str, tenant_id: str):
    """Get help information for a specific policy."""
    
    # In production, this would query a knowledge base
    help_content = {
        "bias_fairness_gate": {
            "title": "Bias & Fairness Policy Gate",
            "description": "This gate ensures your model meets fairness criteria across different demographic groups.",
            "requirements": [
                "Demographic parity difference < 0.1",
                "Equal opportunity difference < 0.1",
                "Statistical parity ratio between 0.8 and 1.25"
            ],
            "how_to_fix": [
                "Review training data for demographic balance",
                "Apply fairness-aware training techniques",
                "Use bias mitigation algorithms",
                "Adjust decision thresholds per group"
            ],
            "documentation_link": "/docs/bias-fairness-policy"
        },
        "trust_score_gate": {
            "title": "Trust Score Policy Gate",
            "description": "This gate ensures your model has sufficient trust score for deployment.",
            "requirements": [
                "Overall trust score > 0.7",
                "Explainability score > 0.6",
                "Drift score < 0.3"
            ],
            "how_to_fix": [
                "Improve model explainability with SHAP/LIME",
                "Reduce model drift through retraining",
                "Enhance data quality and feature engineering",
                "Add more comprehensive test coverage"
            ],
            "documentation_link": "/docs/trust-score-policy"
        }
    }
    
    policy_help = help_content.get(policy_name.lower(), {
        "title": f"Policy: {policy_name}",
        "description": "Policy help information not available.",
        "requirements": ["Contact your compliance team for details"],
        "how_to_fix": ["Review policy documentation"],
        "documentation_link": "/docs/policies"
    })
    
    return {
        "policy_name": policy_name,
        "tenant_id": tenant_id,
        "help_content": policy_help,
        "contact_support": "governance-support@company.com"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA UX Governance Service", "task": "3.2.34"}
