"""
Task 3.2.37: Table-top exercises (quarterly) and audit rehearsals
- Org readiness
- Scheduled drills
- Findings → backlog
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Table-Top Exercise & Audit Rehearsal Service")
logger = logging.getLogger(__name__)

class ExerciseType(str, Enum):
    TABLE_TOP = "table_top"
    SIMULATION = "simulation"
    AUDIT_REHEARSAL = "audit_rehearsal"
    CRISIS_DRILL = "crisis_drill"
    COMPLIANCE_REVIEW = "compliance_review"
    INCIDENT_RESPONSE = "incident_response"

class ExerciseStatus(str, Enum):
    PLANNED = "planned"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"

class ParticipantRole(str, Enum):
    FACILITATOR = "facilitator"
    OBSERVER = "observer"
    PARTICIPANT = "participant"
    SUBJECT_MATTER_EXPERT = "sme"
    AUDITOR = "auditor"
    REGULATOR = "regulator"

class FindingSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class ActionItemStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"

class ExerciseParticipant(BaseModel):
    participant_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    role: ParticipantRole
    department: str
    organization: str = "Internal"
    
    # Participation details
    attended: bool = True
    performance_rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 scale
    feedback: Optional[str] = None

class ExerciseScenario(BaseModel):
    scenario_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    
    # Scenario details
    scenario_type: str  # e.g., "data_breach", "model_bias", "regulatory_inquiry"
    complexity_level: str = "medium"  # low, medium, high
    estimated_duration_minutes: int = 120
    
    # Context and setup
    background_information: str
    initial_conditions: List[str] = Field(default_factory=list)
    injects: List[str] = Field(default_factory=list)  # Additional information injected during exercise
    
    # Expected outcomes
    learning_objectives: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)

class ExerciseFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    severity: FindingSeverity
    
    # Categorization
    category: str  # e.g., "process", "technology", "people", "documentation"
    affected_area: str  # e.g., "incident_response", "compliance", "risk_management"
    
    # Evidence and impact
    evidence: List[str] = Field(default_factory=list)
    potential_impact: str
    risk_rating: Optional[str] = None
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    identified_by: str
    identified_at: datetime = Field(default_factory=datetime.utcnow)

class ActionItem(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    
    # Assignment
    assigned_to: str
    assigned_by: str
    department: str
    
    # Timeline
    due_date: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Status and priority
    status: ActionItemStatus = ActionItemStatus.OPEN
    priority: str = "medium"  # low, medium, high, critical
    
    # Tracking
    progress_percentage: int = Field(default=0, ge=0, le=100)
    status_notes: List[str] = Field(default_factory=list)
    
    # Relationships
    related_finding_id: Optional[str] = None
    related_exercise_id: str

class TableTopExercise(BaseModel):
    exercise_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    exercise_type: ExerciseType
    
    # Scheduling
    scheduled_date: datetime
    estimated_duration_minutes: int = 120
    location: str = "Virtual"
    
    # Content
    scenario: ExerciseScenario
    participants: List[ExerciseParticipant] = Field(default_factory=list)
    
    # Execution
    status: ExerciseStatus = ExerciseStatus.PLANNED
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    actual_duration_minutes: Optional[int] = None
    
    # Results
    findings: List[ExerciseFinding] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    
    # Assessment
    overall_readiness_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    lessons_learned: List[str] = Field(default_factory=list)
    success_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Metadata
    facilitator: str
    created_by: str
    tenant_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Follow-up
    next_exercise_recommended_date: Optional[datetime] = None
    improvement_areas: List[str] = Field(default_factory=list)

class ExerciseReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exercise_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Executive summary
    executive_summary: str
    key_findings_summary: List[str] = Field(default_factory=list)
    
    # Detailed results
    participation_summary: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Recommendations
    immediate_actions: List[str] = Field(default_factory=list)
    long_term_improvements: List[str] = Field(default_factory=list)
    
    # Compliance and audit readiness
    compliance_gaps: List[str] = Field(default_factory=list)
    audit_readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Next steps
    follow_up_exercises: List[str] = Field(default_factory=list)
    training_recommendations: List[str] = Field(default_factory=list)

# In-memory stores (replace with proper database in production)
exercises: Dict[str, TableTopExercise] = {}
action_items: Dict[str, ActionItem] = {}
exercise_reports: Dict[str, ExerciseReport] = {}

def _create_data_breach_scenario() -> ExerciseScenario:
    """Create a data breach table-top exercise scenario."""
    return ExerciseScenario(
        title="AI Model Data Breach Response",
        description="A sophisticated attack has compromised our AI training data repository",
        scenario_type="data_breach",
        complexity_level="high",
        estimated_duration_minutes=180,
        background_information="""
        Your organization operates a multi-tenant AI platform processing sensitive customer data.
        At 2:30 AM, monitoring alerts indicate unusual data access patterns in the training data repository.
        Initial investigation suggests potential unauthorized access to customer PII used in model training.
        The affected data includes financial records, personal identifiers, and behavioral data from 50,000+ customers.
        """,
        initial_conditions=[
            "Monitoring system shows anomalous data access at 2:30 AM",
            "Security team on-call engineer has been notified",
            "Affected system contains training data for 15 production AI models",
            "Data includes PII from customers across 3 jurisdictions (US, EU, UK)",
            "Some affected models are currently serving live traffic"
        ],
        injects=[
            "30 minutes in: Customer reports receiving suspicious emails referencing their private data",
            "60 minutes in: Regulatory authority contacts your organization about potential breach",
            "90 minutes in: Media outlet publishes story about the breach",
            "120 minutes in: Class action lawsuit notice received"
        ],
        learning_objectives=[
            "Test incident response procedures for AI/ML systems",
            "Evaluate cross-functional coordination during crisis",
            "Assess regulatory notification processes",
            "Review customer communication protocols",
            "Validate technical containment procedures"
        ],
        success_criteria=[
            "Incident commander assigned within 15 minutes",
            "Technical containment achieved within 60 minutes",
            "Legal and compliance teams engaged within 30 minutes",
            "Customer communication plan activated within 2 hours",
            "Regulatory notifications prepared within 4 hours"
        ]
    )

def _create_bias_detection_scenario() -> ExerciseScenario:
    """Create a bias detection table-top exercise scenario."""
    return ExerciseScenario(
        title="AI Model Bias Detection and Response",
        description="Automated bias monitoring has detected significant fairness violations in a production credit scoring model",
        scenario_type="model_bias",
        complexity_level="medium",
        estimated_duration_minutes=120,
        background_information="""
        Your organization's credit scoring AI model, used by major financial institutions,
        has triggered multiple bias alerts. The model shows significant disparate impact
        against protected demographic groups, with approval rate differences exceeding regulatory thresholds.
        The model processes 10,000+ applications daily and has been in production for 6 months.
        """,
        initial_conditions=[
            "Bias monitoring system shows demographic parity violation of 0.15 (threshold: 0.10)",
            "Equal opportunity difference detected at 0.18 (threshold: 0.10)",
            "Affected groups: Women and minority ethnic groups",
            "Model is currently processing live applications",
            "Three major bank clients are using this model"
        ],
        injects=[
            "45 minutes in: Civil rights organization files complaint with regulatory authority",
            "75 minutes in: Major client bank threatens to suspend use of the model",
            "90 minutes in: Internal audit team requests immediate review"
        ],
        learning_objectives=[
            "Test bias incident response procedures",
            "Evaluate model kill-switch activation process",
            "Assess stakeholder communication for bias issues",
            "Review bias remediation technical procedures",
            "Validate regulatory reporting for fairness violations"
        ],
        success_criteria=[
            "Bias incident declared within 10 minutes",
            "Model traffic routing to fallback within 30 minutes",
            "Affected applications flagged for review within 60 minutes",
            "Client notifications sent within 90 minutes",
            "Remediation plan developed within 2 hours"
        ]
    )

def _create_regulatory_inquiry_scenario() -> ExerciseScenario:
    """Create a regulatory inquiry table-top exercise scenario."""
    return ExerciseScenario(
        title="Regulatory Authority AI Audit",
        description="Financial regulatory authority has initiated a comprehensive audit of AI governance practices",
        scenario_type="regulatory_inquiry",
        complexity_level="high",
        estimated_duration_minutes=150,
        background_information="""
        The Financial Conduct Authority (FCA) has sent a formal notice requesting comprehensive
        documentation of your AI governance practices, model risk management, and compliance procedures.
        They are particularly interested in explainability, bias testing, and consumer protection measures.
        The audit covers all AI models used in financial services over the past 18 months.
        """,
        initial_conditions=[
            "Formal audit notice received from FCA",
            "30-day deadline for initial response",
            "Request covers 25 production AI models",
            "Auditors want to interview key personnel",
            "Evidence packages required for each model"
        ],
        injects=[
            "Day 5: Auditors request additional documentation on model validation",
            "Day 10: Request for customer complaint data related to AI decisions",
            "Day 15: Auditors schedule on-site visit",
            "Day 20: Media reports on the regulatory investigation"
        ],
        learning_objectives=[
            "Test regulatory response procedures",
            "Evaluate evidence package generation process",
            "Assess cross-team coordination for audit response",
            "Review documentation completeness",
            "Validate stakeholder communication protocols"
        ],
        success_criteria=[
            "Audit response team assembled within 24 hours",
            "Initial response submitted within deadline",
            "Evidence packages generated for all models",
            "Key personnel prepared for interviews",
            "Legal strategy developed and approved"
        ]
    )

@app.post("/exercises", response_model=TableTopExercise)
async def create_exercise(exercise: TableTopExercise):
    """Create a new table-top exercise."""
    
    exercise.created_at = datetime.utcnow()
    exercises[exercise.exercise_id] = exercise
    
    logger.info(f"Created exercise {exercise.exercise_id}: {exercise.title}")
    return exercise

@app.post("/exercises/templates/create-from-template")
async def create_exercise_from_template(
    template_type: str,
    title: str,
    scheduled_date: datetime,
    facilitator: str,
    tenant_id: str,
    created_by: str,
    participants: List[ExerciseParticipant] = None
):
    """Create an exercise from a predefined template."""
    
    scenario_templates = {
        "data_breach": _create_data_breach_scenario(),
        "bias_detection": _create_bias_detection_scenario(),
        "regulatory_inquiry": _create_regulatory_inquiry_scenario()
    }
    
    if template_type not in scenario_templates:
        raise HTTPException(status_code=400, detail=f"Unknown template type: {template_type}")
    
    scenario = scenario_templates[template_type]
    
    exercise = TableTopExercise(
        title=title,
        description=f"Table-top exercise based on {template_type} scenario",
        exercise_type=ExerciseType.TABLE_TOP,
        scheduled_date=scheduled_date,
        estimated_duration_minutes=scenario.estimated_duration_minutes,
        scenario=scenario,
        participants=participants or [],
        facilitator=facilitator,
        created_by=created_by,
        tenant_id=tenant_id
    )
    
    exercises[exercise.exercise_id] = exercise
    
    logger.info(f"Created exercise from template {template_type}: {exercise.exercise_id}")
    return exercise

@app.post("/exercises/{exercise_id}/start")
async def start_exercise(exercise_id: str):
    """Start a table-top exercise."""
    
    if exercise_id not in exercises:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    exercise = exercises[exercise_id]
    exercise.status = ExerciseStatus.IN_PROGRESS
    exercise.actual_start_time = datetime.utcnow()
    
    logger.info(f"Started exercise {exercise_id}")
    return {"status": "started", "exercise_id": exercise_id, "start_time": exercise.actual_start_time}

@app.post("/exercises/{exercise_id}/findings", response_model=ExerciseFinding)
async def add_exercise_finding(exercise_id: str, finding: ExerciseFinding):
    """Add a finding to an exercise."""
    
    if exercise_id not in exercises:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    exercise = exercises[exercise_id]
    exercise.findings.append(finding)
    
    logger.info(f"Added finding {finding.finding_id} to exercise {exercise_id}")
    return finding

@app.post("/exercises/{exercise_id}/action-items", response_model=ActionItem)
async def create_action_item(exercise_id: str, action_item: ActionItem):
    """Create an action item from exercise findings."""
    
    if exercise_id not in exercises:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    action_item.related_exercise_id = exercise_id
    action_items[action_item.action_id] = action_item
    
    # Add to exercise
    exercise = exercises[exercise_id]
    exercise.action_items.append(action_item)
    
    logger.info(f"Created action item {action_item.action_id} for exercise {exercise_id}")
    return action_item

@app.post("/exercises/{exercise_id}/complete")
async def complete_exercise(
    exercise_id: str,
    overall_readiness_score: float,
    lessons_learned: List[str],
    success_rate: float,
    improvement_areas: List[str]
):
    """Complete a table-top exercise and record results."""
    
    if exercise_id not in exercises:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    exercise = exercises[exercise_id]
    exercise.status = ExerciseStatus.COMPLETED
    exercise.actual_end_time = datetime.utcnow()
    
    if exercise.actual_start_time:
        duration = exercise.actual_end_time - exercise.actual_start_time
        exercise.actual_duration_minutes = int(duration.total_seconds() / 60)
    
    exercise.overall_readiness_score = overall_readiness_score
    exercise.lessons_learned = lessons_learned
    exercise.success_rate = success_rate
    exercise.improvement_areas = improvement_areas
    
    # Schedule next exercise (quarterly)
    exercise.next_exercise_recommended_date = exercise.actual_end_time + timedelta(days=90)
    
    logger.info(f"Completed exercise {exercise_id} with {success_rate}% success rate")
    
    return {
        "status": "completed",
        "exercise_id": exercise_id,
        "duration_minutes": exercise.actual_duration_minutes,
        "readiness_score": overall_readiness_score,
        "findings_count": len(exercise.findings),
        "action_items_count": len(exercise.action_items)
    }

@app.post("/exercises/{exercise_id}/report", response_model=ExerciseReport)
async def generate_exercise_report(exercise_id: str):
    """Generate a comprehensive report for a completed exercise."""
    
    if exercise_id not in exercises:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    exercise = exercises[exercise_id]
    
    if exercise.status != ExerciseStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Exercise must be completed to generate report")
    
    # Calculate participation metrics
    total_participants = len(exercise.participants)
    attended_participants = len([p for p in exercise.participants if p.attended])
    attendance_rate = (attended_participants / total_participants * 100) if total_participants > 0 else 0
    
    # Calculate performance metrics
    performance_ratings = [p.performance_rating for p in exercise.participants if p.performance_rating]
    avg_performance = sum(performance_ratings) / len(performance_ratings) if performance_ratings else 0
    
    # Categorize findings
    critical_findings = [f for f in exercise.findings if f.severity == FindingSeverity.CRITICAL]
    high_findings = [f for f in exercise.findings if f.severity == FindingSeverity.HIGH]
    
    # Generate executive summary
    executive_summary = f"""
    Table-top exercise '{exercise.title}' was conducted on {exercise.scheduled_date.strftime('%Y-%m-%d')} 
    with {attended_participants} of {total_participants} participants attending ({attendance_rate:.1f}% attendance).
    
    The exercise achieved a {exercise.success_rate:.1f}% success rate and {exercise.overall_readiness_score:.1f}/100 
    organizational readiness score. {len(exercise.findings)} findings were identified, including 
    {len(critical_findings)} critical and {len(high_findings)} high-severity issues.
    
    {len(exercise.action_items)} action items were created to address identified gaps and improve 
    organizational readiness for similar real-world scenarios.
    """
    
    report = ExerciseReport(
        exercise_id=exercise_id,
        executive_summary=executive_summary.strip(),
        key_findings_summary=[f.title for f in exercise.findings if f.severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH]],
        participation_summary={
            "total_participants": total_participants,
            "attended_participants": attended_participants,
            "attendance_rate": attendance_rate,
            "average_performance_rating": avg_performance
        },
        performance_metrics={
            "success_rate": exercise.success_rate or 0,
            "readiness_score": exercise.overall_readiness_score or 0,
            "duration_vs_planned": (exercise.actual_duration_minutes / exercise.estimated_duration_minutes * 100) if exercise.actual_duration_minutes else 100
        },
        immediate_actions=[ai.title for ai in exercise.action_items if ai.priority in ["critical", "high"]],
        long_term_improvements=exercise.improvement_areas,
        compliance_gaps=[f.description for f in exercise.findings if "compliance" in f.category.lower()],
        audit_readiness_score=exercise.overall_readiness_score or 0,
        follow_up_exercises=[f"Address {area}" for area in exercise.improvement_areas[:3]],
        training_recommendations=[f"Training needed in {f.affected_area}" for f in exercise.findings if f.severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH]]
    )
    
    exercise_reports[report.report_id] = report
    
    logger.info(f"Generated report {report.report_id} for exercise {exercise_id}")
    return report

@app.get("/exercises", response_model=List[TableTopExercise])
async def list_exercises(
    tenant_id: Optional[str] = None,
    status: Optional[ExerciseStatus] = None,
    exercise_type: Optional[ExerciseType] = None
):
    """List exercises with optional filters."""
    
    filtered_exercises = list(exercises.values())
    
    if tenant_id:
        filtered_exercises = [e for e in filtered_exercises if e.tenant_id == tenant_id]
    
    if status:
        filtered_exercises = [e for e in filtered_exercises if e.status == status]
    
    if exercise_type:
        filtered_exercises = [e for e in filtered_exercises if e.exercise_type == exercise_type]
    
    # Sort by scheduled date
    filtered_exercises.sort(key=lambda x: x.scheduled_date, reverse=True)
    
    return filtered_exercises

@app.get("/action-items", response_model=List[ActionItem])
async def list_action_items(
    status: Optional[ActionItemStatus] = None,
    assigned_to: Optional[str] = None,
    priority: Optional[str] = None,
    overdue_only: bool = False
):
    """List action items with optional filters."""
    
    filtered_items = list(action_items.values())
    
    if status:
        filtered_items = [ai for ai in filtered_items if ai.status == status]
    
    if assigned_to:
        filtered_items = [ai for ai in filtered_items if ai.assigned_to == assigned_to]
    
    if priority:
        filtered_items = [ai for ai in filtered_items if ai.priority == priority]
    
    if overdue_only:
        now = datetime.utcnow()
        filtered_items = [ai for ai in filtered_items if ai.due_date < now and ai.status != ActionItemStatus.COMPLETED]
    
    # Sort by due date
    filtered_items.sort(key=lambda x: x.due_date)
    
    return filtered_items

@app.post("/action-items/{action_id}/update-status")
async def update_action_item_status(
    action_id: str,
    status: ActionItemStatus,
    progress_percentage: Optional[int] = None,
    status_note: Optional[str] = None
):
    """Update the status of an action item."""
    
    if action_id not in action_items:
        raise HTTPException(status_code=404, detail="Action item not found")
    
    action_item = action_items[action_id]
    action_item.status = status
    
    if progress_percentage is not None:
        action_item.progress_percentage = progress_percentage
    
    if status_note:
        action_item.status_notes.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}: {status_note}")
    
    if status == ActionItemStatus.COMPLETED:
        action_item.completed_at = datetime.utcnow()
        action_item.progress_percentage = 100
    
    logger.info(f"Updated action item {action_id} status to {status.value}")
    
    return {
        "status": "updated",
        "action_id": action_id,
        "new_status": status.value,
        "progress_percentage": action_item.progress_percentage
    }

@app.get("/exercises/upcoming-quarterly")
async def get_upcoming_quarterly_exercises(tenant_id: Optional[str] = None):
    """Get upcoming quarterly exercises and suggest scheduling if needed."""
    
    now = datetime.utcnow()
    three_months_ahead = now + timedelta(days=90)
    
    # Find upcoming exercises
    upcoming = [
        e for e in exercises.values()
        if e.scheduled_date >= now and e.scheduled_date <= three_months_ahead
        and (tenant_id is None or e.tenant_id == tenant_id)
    ]
    
    # Find exercises that need to be scheduled (completed exercises with recommended next date)
    needs_scheduling = [
        e for e in exercises.values()
        if e.status == ExerciseStatus.COMPLETED
        and e.next_exercise_recommended_date
        and e.next_exercise_recommended_date <= three_months_ahead
        and (tenant_id is None or e.tenant_id == tenant_id)
    ]
    
    return {
        "upcoming_exercises": upcoming,
        "exercises_needing_scheduling": needs_scheduling,
        "recommendation": f"Schedule {len(needs_scheduling)} follow-up exercises for quarterly compliance"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Table-Top Exercise & Audit Rehearsal Service", "task": "3.2.37"}
