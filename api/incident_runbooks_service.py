"""
Task 3.2.36: Runbooks: incident (sev ladder), drift, bias breach, privacy breach, regulator inquiry
- Consistent response
- Markdown + wiki
- Tested in drills
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import markdown
from dataclasses import dataclass

app = FastAPI(title="RBIA Incident Response Runbooks Service")
logger = logging.getLogger(__name__)

class RunbookType(str, Enum):
    INCIDENT_RESPONSE = "incident_response"
    DRIFT_DETECTION = "drift_detection"
    BIAS_BREACH = "bias_breach"
    PRIVACY_BREACH = "privacy_breach"
    REGULATOR_INQUIRY = "regulator_inquiry"
    SECURITY_INCIDENT = "security_incident"
    KILL_SWITCH_ACTIVATION = "kill_switch_activation"

class SeverityLevel(str, Enum):
    SEV1_CRITICAL = "sev1_critical"      # Complete service outage
    SEV2_HIGH = "sev2_high"              # Major functionality impaired
    SEV3_MEDIUM = "sev3_medium"          # Minor functionality impaired
    SEV4_LOW = "sev4_low"                # Minimal impact
    SEV5_INFO = "sev5_info"              # Informational only

class RunbookStatus(str, Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    ACTIVE = "active"
    DEPRECATED = "deprecated"

class DrillStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RunbookStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    title: str
    description: str
    
    # Execution details
    estimated_duration_minutes: int = 5
    required_roles: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    
    # Actions
    actions: List[str] = Field(default_factory=list)
    commands: List[str] = Field(default_factory=list)
    verification_steps: List[str] = Field(default_factory=list)
    
    # Decision points
    is_decision_point: bool = False
    decision_criteria: Optional[str] = None
    next_step_if_yes: Optional[str] = None
    next_step_if_no: Optional[str] = None
    
    # Escalation
    escalation_trigger: Optional[str] = None
    escalation_contact: Optional[str] = None

class Runbook(BaseModel):
    runbook_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    runbook_type: RunbookType
    title: str
    description: str
    
    # Scope and triggers
    applicable_severity: List[SeverityLevel] = Field(default_factory=list)
    trigger_conditions: List[str] = Field(default_factory=list)
    affected_systems: List[str] = Field(default_factory=list)
    
    # Content
    steps: List[RunbookStep] = Field(default_factory=list)
    markdown_content: Optional[str] = None
    
    # Metadata
    version: str = "1.0"
    status: RunbookStatus = RunbookStatus.DRAFT
    created_by: str
    approved_by: Optional[str] = None
    last_tested: Optional[datetime] = None
    
    # Timing
    estimated_total_duration_minutes: int = 30
    max_response_time_minutes: int = 15
    
    # Contacts and escalation
    primary_contact: str
    escalation_contacts: List[str] = Field(default_factory=list)
    external_contacts: List[str] = Field(default_factory=list)  # Regulators, vendors, etc.
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Tenant context
    tenant_id: str

class RunbookExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    runbook_id: str
    incident_id: Optional[str] = None
    
    # Execution context
    triggered_by: str
    trigger_reason: str
    severity: SeverityLevel
    
    # Progress tracking
    current_step: int = 1
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_duration_minutes: Optional[int] = None
    
    # Participants
    incident_commander: str
    participants: List[str] = Field(default_factory=list)
    
    # Status and notes
    status: str = "in_progress"  # in_progress, completed, failed, aborted
    execution_notes: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    
    # Tenant context
    tenant_id: str

class RunbookDrill(BaseModel):
    drill_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    runbook_id: str
    drill_name: str
    
    # Drill configuration
    drill_type: str = "table_top"  # table_top, simulation, live_fire
    scenario_description: str
    expected_duration_minutes: int = 60
    
    # Participants
    facilitator: str
    participants: List[str] = Field(default_factory=list)
    observers: List[str] = Field(default_factory=list)
    
    # Scheduling
    scheduled_at: datetime
    status: DrillStatus = DrillStatus.SCHEDULED
    
    # Results
    completed_at: Optional[datetime] = None
    actual_duration_minutes: Optional[int] = None
    success_rate: Optional[float] = None
    
    # Findings
    findings: List[str] = Field(default_factory=list)
    improvement_actions: List[str] = Field(default_factory=list)
    
    # Tenant context
    tenant_id: str

# In-memory stores (replace with proper database in production)
runbooks: Dict[str, Runbook] = {}
runbook_executions: Dict[str, RunbookExecution] = {}
runbook_drills: Dict[str, RunbookDrill] = {}

# Predefined runbook templates
def _create_incident_response_runbook(tenant_id: str, created_by: str) -> Runbook:
    """Create standard incident response runbook template."""
    steps = [
        RunbookStep(
            step_number=1,
            title="Incident Detection and Initial Assessment",
            description="Detect, acknowledge, and perform initial assessment of the incident",
            estimated_duration_minutes=5,
            required_roles=["on_call_engineer", "incident_commander"],
            actions=[
                "Acknowledge the alert in monitoring system",
                "Create incident ticket in ITSM system",
                "Perform initial impact assessment",
                "Determine preliminary severity level"
            ],
            verification_steps=[
                "Incident ticket created with unique ID",
                "Initial severity assigned",
                "Monitoring alerts acknowledged"
            ]
        ),
        RunbookStep(
            step_number=2,
            title="Incident Commander Assignment",
            description="Assign incident commander based on severity",
            estimated_duration_minutes=3,
            required_roles=["duty_manager"],
            actions=[
                "Assign incident commander based on severity matrix",
                "Notify incident commander via multiple channels",
                "Transfer incident ownership to commander"
            ],
            is_decision_point=True,
            decision_criteria="Is this SEV1 or SEV2?",
            next_step_if_yes="escalate_to_senior_commander",
            next_step_if_no="proceed_with_standard_commander"
        ),
        RunbookStep(
            step_number=3,
            title="Stakeholder Notification",
            description="Notify relevant stakeholders based on severity and impact",
            estimated_duration_minutes=10,
            required_roles=["incident_commander"],
            actions=[
                "Send initial notification to stakeholders",
                "Create communication channels (Slack, Teams)",
                "Notify customers if external impact",
                "Update status page if applicable"
            ],
            verification_steps=[
                "Stakeholder notification sent",
                "Communication channels established",
                "Status page updated if needed"
            ]
        ),
        RunbookStep(
            step_number=4,
            title="Investigation and Diagnosis",
            description="Investigate root cause and develop remediation plan",
            estimated_duration_minutes=30,
            required_roles=["technical_lead", "subject_matter_expert"],
            actions=[
                "Gather logs and diagnostic information",
                "Analyze system metrics and traces",
                "Identify root cause or contributing factors",
                "Develop remediation plan"
            ],
            verification_steps=[
                "Root cause identified or hypothesis formed",
                "Remediation plan documented",
                "Technical team aligned on approach"
            ]
        ),
        RunbookStep(
            step_number=5,
            title="Remediation and Recovery",
            description="Execute remediation plan and restore service",
            estimated_duration_minutes=45,
            required_roles=["technical_lead", "operations_team"],
            actions=[
                "Execute remediation steps",
                "Monitor system recovery",
                "Validate service restoration",
                "Perform smoke tests"
            ],
            verification_steps=[
                "Service functionality restored",
                "Monitoring shows normal operations",
                "Smoke tests passing"
            ]
        ),
        RunbookStep(
            step_number=6,
            title="Incident Closure",
            description="Close incident and initiate post-mortem process",
            estimated_duration_minutes=10,
            required_roles=["incident_commander"],
            actions=[
                "Confirm service restoration with stakeholders",
                "Send incident closure notification",
                "Schedule post-mortem meeting",
                "Update incident ticket to closed"
            ],
            verification_steps=[
                "Stakeholders confirm resolution",
                "Post-mortem scheduled",
                "Incident ticket closed"
            ]
        )
    ]
    
    return Runbook(
        runbook_type=RunbookType.INCIDENT_RESPONSE,
        title="Standard Incident Response Runbook",
        description="Standard operating procedure for incident response across all severity levels",
        applicable_severity=[SeverityLevel.SEV1_CRITICAL, SeverityLevel.SEV2_HIGH, SeverityLevel.SEV3_MEDIUM],
        trigger_conditions=[
            "System monitoring alert triggered",
            "Customer-reported service issue",
            "Automated health check failure",
            "Manual incident declaration"
        ],
        affected_systems=["RBIA Platform", "RBA System", "Data Pipeline", "API Gateway"],
        steps=steps,
        estimated_total_duration_minutes=103,
        max_response_time_minutes=15,
        primary_contact="incident-commander@company.com",
        escalation_contacts=["cto@company.com", "vp-engineering@company.com"],
        created_by=created_by,
        tenant_id=tenant_id
    )

def _create_bias_breach_runbook(tenant_id: str, created_by: str) -> Runbook:
    """Create bias breach response runbook."""
    steps = [
        RunbookStep(
            step_number=1,
            title="Bias Detection Confirmation",
            description="Confirm and validate the bias detection alert",
            estimated_duration_minutes=10,
            required_roles=["ml_engineer", "fairness_specialist"],
            actions=[
                "Review bias detection metrics and thresholds",
                "Validate statistical significance of bias",
                "Confirm affected demographic groups",
                "Assess scope and severity of bias"
            ]
        ),
        RunbookStep(
            step_number=2,
            title="Immediate Mitigation",
            description="Take immediate action to prevent further biased decisions",
            estimated_duration_minutes=15,
            required_roles=["ml_engineer", "operations_team"],
            actions=[
                "Activate kill switch for affected model if severe",
                "Route traffic to baseline/fallback model",
                "Flag affected decisions for review",
                "Document mitigation actions taken"
            ]
        ),
        RunbookStep(
            step_number=3,
            title="Stakeholder Notification",
            description="Notify legal, compliance, and business stakeholders",
            estimated_duration_minutes=20,
            required_roles=["compliance_officer", "legal_counsel"],
            actions=[
                "Notify legal team of potential bias issue",
                "Inform compliance officer for regulatory assessment",
                "Alert business stakeholders of affected processes",
                "Prepare initial bias incident report"
            ]
        ),
        RunbookStep(
            step_number=4,
            title="Root Cause Analysis",
            description="Investigate the source and cause of bias",
            estimated_duration_minutes=60,
            required_roles=["data_scientist", "ml_engineer"],
            actions=[
                "Analyze training data for bias sources",
                "Review feature engineering and selection",
                "Examine model architecture and parameters",
                "Identify specific bias mechanisms"
            ]
        ),
        RunbookStep(
            step_number=5,
            title="Bias Remediation",
            description="Implement bias mitigation measures",
            estimated_duration_minutes=120,
            required_roles=["data_scientist", "ml_engineer"],
            actions=[
                "Apply bias mitigation algorithms",
                "Retrain model with fairness constraints",
                "Implement post-processing corrections",
                "Validate bias reduction effectiveness"
            ]
        ),
        RunbookStep(
            step_number=6,
            title="Regulatory Reporting",
            description="Prepare and submit required regulatory reports",
            estimated_duration_minutes=30,
            required_roles=["compliance_officer", "legal_counsel"],
            actions=[
                "Prepare bias incident report for regulators",
                "Document remediation actions taken",
                "Submit reports to relevant authorities",
                "Update compliance documentation"
            ]
        )
    ]
    
    return Runbook(
        runbook_type=RunbookType.BIAS_BREACH,
        title="AI Bias Breach Response Runbook",
        description="Response procedure for detected bias in AI/ML models",
        applicable_severity=[SeverityLevel.SEV1_CRITICAL, SeverityLevel.SEV2_HIGH],
        trigger_conditions=[
            "Automated bias detection alert",
            "Fairness metric threshold exceeded",
            "External bias complaint received",
            "Audit finding of biased outcomes"
        ],
        steps=steps,
        estimated_total_duration_minutes=255,
        max_response_time_minutes=30,
        primary_contact="fairness-team@company.com",
        escalation_contacts=["legal@company.com", "compliance@company.com"],
        external_contacts=["regulatory-affairs@company.com"],
        created_by=created_by,
        tenant_id=tenant_id
    )

def _create_privacy_breach_runbook(tenant_id: str, created_by: str) -> Runbook:
    """Create privacy breach response runbook."""
    steps = [
        RunbookStep(
            step_number=1,
            title="Privacy Breach Assessment",
            description="Assess the scope and severity of the privacy breach",
            estimated_duration_minutes=15,
            required_roles=["privacy_officer", "security_team"],
            actions=[
                "Identify types of personal data involved",
                "Determine number of affected individuals",
                "Assess risk to individuals' rights and freedoms",
                "Classify breach severity under GDPR/CCPA"
            ]
        ),
        RunbookStep(
            step_number=2,
            title="Containment and Preservation",
            description="Contain the breach and preserve evidence",
            estimated_duration_minutes=20,
            required_roles=["security_team", "it_operations"],
            actions=[
                "Stop the data exposure or unauthorized access",
                "Preserve logs and evidence of the breach",
                "Secure affected systems and data",
                "Document containment actions taken"
            ]
        ),
        RunbookStep(
            step_number=3,
            title="Legal and Regulatory Assessment",
            description="Assess legal obligations and regulatory requirements",
            estimated_duration_minutes=30,
            required_roles=["legal_counsel", "privacy_officer"],
            actions=[
                "Determine notification obligations (72-hour rule)",
                "Assess need for supervisory authority notification",
                "Evaluate individual notification requirements",
                "Review contractual notification obligations"
            ]
        ),
        RunbookStep(
            step_number=4,
            title="Regulatory Notification",
            description="Notify supervisory authorities if required",
            estimated_duration_minutes=45,
            required_roles=["privacy_officer", "legal_counsel"],
            actions=[
                "Prepare breach notification for supervisory authority",
                "Submit notification within 72 hours if required",
                "Document submission confirmation",
                "Prepare for potential regulatory inquiry"
            ]
        ),
        RunbookStep(
            step_number=5,
            title="Individual Notification",
            description="Notify affected individuals if required",
            estimated_duration_minutes=60,
            required_roles=["privacy_officer", "communications_team"],
            actions=[
                "Prepare individual notification communications",
                "Send notifications via appropriate channels",
                "Set up support channels for affected individuals",
                "Document notification delivery"
            ]
        ),
        RunbookStep(
            step_number=6,
            title="Remediation and Recovery",
            description="Implement remediation measures and prevent recurrence",
            estimated_duration_minutes=90,
            required_roles=["security_team", "privacy_officer"],
            actions=[
                "Implement technical and organizational measures",
                "Update privacy controls and procedures",
                "Conduct security improvements",
                "Document remediation actions"
            ]
        )
    ]
    
    return Runbook(
        runbook_type=RunbookType.PRIVACY_BREACH,
        title="Privacy Breach Response Runbook",
        description="Response procedure for personal data breaches under GDPR/CCPA",
        applicable_severity=[SeverityLevel.SEV1_CRITICAL, SeverityLevel.SEV2_HIGH],
        trigger_conditions=[
            "Automated PII exposure detection",
            "Unauthorized access to personal data",
            "Data loss or theft incident",
            "Third-party data breach notification"
        ],
        steps=steps,
        estimated_total_duration_minutes=260,
        max_response_time_minutes=60,  # GDPR 72-hour requirement
        primary_contact="privacy-officer@company.com",
        escalation_contacts=["legal@company.com", "ciso@company.com"],
        external_contacts=["data-protection-authority@regulator.gov"],
        created_by=created_by,
        tenant_id=tenant_id
    )

@app.post("/runbooks", response_model=Runbook)
async def create_runbook(runbook: Runbook):
    """Create a new runbook."""
    
    runbook.updated_at = datetime.utcnow()
    runbooks[runbook.runbook_id] = runbook
    
    logger.info(f"Created runbook {runbook.runbook_id}: {runbook.title}")
    return runbook

@app.get("/runbooks", response_model=List[Runbook])
async def list_runbooks(
    tenant_id: Optional[str] = None,
    runbook_type: Optional[RunbookType] = None,
    status: Optional[RunbookStatus] = None
):
    """List runbooks with optional filters."""
    
    filtered_runbooks = list(runbooks.values())
    
    if tenant_id:
        filtered_runbooks = [r for r in filtered_runbooks if r.tenant_id == tenant_id]
    
    if runbook_type:
        filtered_runbooks = [r for r in filtered_runbooks if r.runbook_type == runbook_type]
    
    if status:
        filtered_runbooks = [r for r in filtered_runbooks if r.status == status]
    
    return filtered_runbooks

@app.post("/runbooks/initialize-defaults")
async def initialize_default_runbooks(tenant_id: str, created_by: str):
    """Initialize default runbooks for a tenant."""
    
    default_runbooks = [
        _create_incident_response_runbook(tenant_id, created_by),
        _create_bias_breach_runbook(tenant_id, created_by),
        _create_privacy_breach_runbook(tenant_id, created_by)
    ]
    
    created_runbook_ids = []
    for runbook in default_runbooks:
        runbooks[runbook.runbook_id] = runbook
        created_runbook_ids.append(runbook.runbook_id)
    
    logger.info(f"Initialized {len(default_runbooks)} default runbooks for tenant {tenant_id}")
    
    return {
        "status": "success",
        "tenant_id": tenant_id,
        "created_runbooks": created_runbook_ids,
        "total_runbooks": len(default_runbooks)
    }

@app.post("/runbooks/{runbook_id}/execute", response_model=RunbookExecution)
async def execute_runbook(
    runbook_id: str,
    triggered_by: str,
    trigger_reason: str,
    severity: SeverityLevel,
    incident_commander: str,
    incident_id: Optional[str] = None
):
    """Execute a runbook for an incident."""
    
    if runbook_id not in runbooks:
        raise HTTPException(status_code=404, detail="Runbook not found")
    
    runbook = runbooks[runbook_id]
    
    execution = RunbookExecution(
        runbook_id=runbook_id,
        incident_id=incident_id,
        triggered_by=triggered_by,
        trigger_reason=trigger_reason,
        severity=severity,
        incident_commander=incident_commander,
        tenant_id=runbook.tenant_id
    )
    
    runbook_executions[execution.execution_id] = execution
    
    logger.info(f"Started runbook execution {execution.execution_id} for runbook {runbook_id}")
    return execution

@app.post("/runbooks/executions/{execution_id}/complete-step")
async def complete_execution_step(
    execution_id: str,
    step_id: str,
    completed_by: str,
    notes: Optional[str] = None,
    success: bool = True
):
    """Mark a runbook execution step as completed."""
    
    if execution_id not in runbook_executions:
        raise HTTPException(status_code=404, detail="Runbook execution not found")
    
    execution = runbook_executions[execution_id]
    
    if success:
        execution.completed_steps.append(step_id)
        if notes:
            execution.execution_notes.append(f"Step {step_id} completed by {completed_by}: {notes}")
    else:
        execution.failed_steps.append(step_id)
        if notes:
            execution.execution_notes.append(f"Step {step_id} failed by {completed_by}: {notes}")
    
    # Advance to next step if successful
    if success:
        execution.current_step += 1
    
    logger.info(f"Step {step_id} {'completed' if success else 'failed'} in execution {execution_id}")
    
    return {
        "status": "updated",
        "execution_id": execution_id,
        "step_id": step_id,
        "success": success,
        "current_step": execution.current_step
    }

@app.post("/runbooks/{runbook_id}/schedule-drill", response_model=RunbookDrill)
async def schedule_runbook_drill(
    runbook_id: str,
    drill_name: str,
    scenario_description: str,
    facilitator: str,
    participants: List[str],
    scheduled_at: datetime,
    drill_type: str = "table_top"
):
    """Schedule a runbook drill."""
    
    if runbook_id not in runbooks:
        raise HTTPException(status_code=404, detail="Runbook not found")
    
    runbook = runbooks[runbook_id]
    
    drill = RunbookDrill(
        runbook_id=runbook_id,
        drill_name=drill_name,
        scenario_description=scenario_description,
        facilitator=facilitator,
        participants=participants,
        scheduled_at=scheduled_at,
        drill_type=drill_type,
        tenant_id=runbook.tenant_id
    )
    
    runbook_drills[drill.drill_id] = drill
    
    logger.info(f"Scheduled drill {drill.drill_id} for runbook {runbook_id}")
    return drill

@app.post("/runbooks/drills/{drill_id}/complete")
async def complete_runbook_drill(
    drill_id: str,
    actual_duration_minutes: int,
    success_rate: float,
    findings: List[str],
    improvement_actions: List[str]
):
    """Complete a runbook drill and record results."""
    
    if drill_id not in runbook_drills:
        raise HTTPException(status_code=404, detail="Runbook drill not found")
    
    drill = runbook_drills[drill_id]
    drill.status = DrillStatus.COMPLETED
    drill.completed_at = datetime.utcnow()
    drill.actual_duration_minutes = actual_duration_minutes
    drill.success_rate = success_rate
    drill.findings = findings
    drill.improvement_actions = improvement_actions
    
    # Update runbook last tested date
    if drill.runbook_id in runbooks:
        runbooks[drill.runbook_id].last_tested = drill.completed_at
    
    logger.info(f"Completed drill {drill_id} with {success_rate}% success rate")
    
    return {
        "status": "completed",
        "drill_id": drill_id,
        "success_rate": success_rate,
        "findings_count": len(findings),
        "improvement_actions_count": len(improvement_actions)
    }

@app.get("/runbooks/{runbook_id}/markdown")
async def get_runbook_markdown(runbook_id: str):
    """Get runbook content in Markdown format."""
    
    if runbook_id not in runbooks:
        raise HTTPException(status_code=404, detail="Runbook not found")
    
    runbook = runbooks[runbook_id]
    
    # Generate Markdown content if not already present
    if not runbook.markdown_content:
        markdown_content = f"""# {runbook.title}

## Description
{runbook.description}

## Severity Levels
{', '.join([s.value for s in runbook.applicable_severity])}

## Trigger Conditions
{chr(10).join([f"- {condition}" for condition in runbook.trigger_conditions])}

## Response Steps

"""
        
        for step in runbook.steps:
            markdown_content += f"""### Step {step.step_number}: {step.title}

**Description:** {step.description}

**Estimated Duration:** {step.estimated_duration_minutes} minutes

**Required Roles:** {', '.join(step.required_roles)}

**Actions:**
{chr(10).join([f"- {action}" for action in step.actions])}

**Verification:**
{chr(10).join([f"- {verification}" for verification in step.verification_steps])}

"""
            
            if step.is_decision_point:
                markdown_content += f"""**Decision Point:** {step.decision_criteria}
- If Yes: {step.next_step_if_yes or 'Continue to next step'}
- If No: {step.next_step_if_no or 'Escalate'}

"""
        
        markdown_content += f"""## Contacts

**Primary Contact:** {runbook.primary_contact}

**Escalation Contacts:**
{chr(10).join([f"- {contact}" for contact in runbook.escalation_contacts])}

**External Contacts:**
{chr(10).join([f"- {contact}" for contact in runbook.external_contacts])}

---
*Last Updated: {runbook.updated_at.strftime('%Y-%m-%d %H:%M:%S')}*
*Version: {runbook.version}*
"""
        
        runbook.markdown_content = markdown_content
    
    return {
        "runbook_id": runbook_id,
        "title": runbook.title,
        "markdown_content": runbook.markdown_content,
        "html_content": markdown.markdown(runbook.markdown_content)
    }

@app.get("/runbooks/drills/upcoming")
async def get_upcoming_drills(tenant_id: Optional[str] = None, days_ahead: int = 30):
    """Get upcoming runbook drills."""
    
    cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
    
    upcoming_drills = [
        drill for drill in runbook_drills.values()
        if drill.scheduled_at <= cutoff_date and drill.status == DrillStatus.SCHEDULED
        and (tenant_id is None or drill.tenant_id == tenant_id)
    ]
    
    # Sort by scheduled date
    upcoming_drills.sort(key=lambda x: x.scheduled_at)
    
    return upcoming_drills

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Incident Response Runbooks Service", "task": "3.2.36"}
