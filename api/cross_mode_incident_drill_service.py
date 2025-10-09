"""
Task 3.4.29: Create cross-mode incident drill reports
- Incident drill simulation service
- Cross-mode execution tracking and comparison
- Report generation engine with evidence pack integration
- Database schema for drill results and analytics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import asyncio
import random
from dataclasses import dataclass

app = FastAPI(title="RBIA Cross-Mode Incident Drill Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class IncidentType(str, Enum):
    SLA_BREACH = "sla_breach"
    MODEL_FAILURE = "model_failure"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SECURITY_BREACH = "security_breach"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    API_OUTAGE = "api_outage"
    DATABASE_FAILURE = "database_failure"

class DrillSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DrillStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResponseAction(str, Enum):
    ESCALATE = "escalate"
    FALLBACK_TO_RBA = "fallback_to_rba"
    KILL_SWITCH = "kill_switch"
    MANUAL_OVERRIDE = "manual_override"
    RETRY_OPERATION = "retry_operation"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"
    GENERATE_EVIDENCE = "generate_evidence"

class IncidentDrill(BaseModel):
    drill_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Drill configuration
    drill_name: str
    incident_type: IncidentType
    severity: DrillSeverity
    description: str
    
    # Cross-mode testing
    target_modes: List[UXMode] = Field(default_factory=lambda: [UXMode.UI_LED, UXMode.ASSISTED, UXMode.CONVERSATIONAL])
    workflow_ids: List[str] = Field(default_factory=list)
    
    # Scenario details
    simulated_conditions: Dict[str, Any] = Field(default_factory=dict)
    expected_responses: List[ResponseAction] = Field(default_factory=list)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    
    # Scheduling
    scheduled_at: datetime
    duration_minutes: int = 30
    
    # Participants
    drill_participants: List[str] = Field(default_factory=list)  # User IDs
    observers: List[str] = Field(default_factory=list)
    
    # Status
    status: DrillStatus = DrillStatus.PLANNED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str

class DrillExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    drill_id: str
    mode: UXMode
    
    # Execution details
    workflow_id: str
    execution_start: datetime = Field(default_factory=datetime.utcnow)
    execution_end: Optional[datetime] = None
    
    # Results
    success: Optional[bool] = None
    response_time_ms: Optional[int] = None
    actions_taken: List[ResponseAction] = Field(default_factory=list)
    
    # Performance metrics
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    api_calls_made: int = 0
    errors_encountered: List[str] = Field(default_factory=list)
    
    # User interaction (for UI/Assisted modes)
    user_interactions: int = 0
    user_satisfaction_score: Optional[float] = None
    
    # Evidence generated
    evidence_packs_created: List[str] = Field(default_factory=list)
    audit_logs_generated: bool = False
    
    # Compliance
    compliance_violations: List[str] = Field(default_factory=list)
    policy_overrides_used: int = 0
    
    execution_logs: List[Dict[str, Any]] = Field(default_factory=list)

class DrillReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    drill_id: str
    
    # Report metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_type: str = "cross_mode_incident_drill"
    
    # Executive summary
    overall_success: bool
    total_executions: int
    successful_executions: int
    failed_executions: int
    
    # Cross-mode comparison
    mode_performance: Dict[UXMode, Dict[str, Any]] = Field(default_factory=dict)
    best_performing_mode: Optional[UXMode] = None
    worst_performing_mode: Optional[UXMode] = None
    
    # Response analysis
    response_effectiveness: Dict[ResponseAction, float] = Field(default_factory=dict)
    average_response_time_ms: float
    compliance_score: float
    
    # Key findings
    strengths_identified: List[str] = Field(default_factory=list)
    weaknesses_identified: List[str] = Field(default_factory=list)
    improvement_recommendations: List[str] = Field(default_factory=list)
    
    # Evidence and audit
    evidence_packs_generated: int
    audit_trail_complete: bool
    compliance_violations_found: int
    
    # Detailed analysis
    detailed_metrics: Dict[str, Any] = Field(default_factory=dict)
    cross_mode_insights: List[str] = Field(default_factory=list)
    
    # Follow-up actions
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    next_drill_recommendations: List[str] = Field(default_factory=list)

class DrillTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_name: str
    incident_type: IncidentType
    
    # Template configuration
    default_severity: DrillSeverity
    default_duration_minutes: int = 30
    recommended_frequency_days: int = 90
    
    # Scenario template
    scenario_template: Dict[str, Any] = Field(default_factory=dict)
    success_criteria_template: Dict[str, Any] = Field(default_factory=dict)
    
    # Industry-specific
    applicable_industries: List[str] = Field(default_factory=list)
    compliance_frameworks: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"

# In-memory storage (replace with database in production)
incident_drills_store: Dict[str, IncidentDrill] = {}
drill_executions_store: Dict[str, List[DrillExecution]] = {}
drill_reports_store: Dict[str, DrillReport] = {}
drill_templates_store: Dict[str, DrillTemplate] = {}

def _initialize_default_templates():
    """Initialize default drill templates"""
    
    # SLA Breach Template
    drill_templates_store["sla_breach_template"] = DrillTemplate(
        template_id="sla_breach_template",
        template_name="SLA Breach Response Drill",
        incident_type=IncidentType.SLA_BREACH,
        default_severity=DrillSeverity.HIGH,
        scenario_template={
            "simulated_latency_ms": 5000,
            "affected_workflows": ["forecast_approval", "pipeline_hygiene"],
            "customer_impact": "high_value_customers"
        },
        success_criteria_template={
            "max_response_time_ms": 30000,
            "fallback_activated": True,
            "evidence_generated": True,
            "stakeholders_notified": True
        },
        applicable_industries=["SaaS", "Financial Services"],
        compliance_frameworks=["SOX", "GDPR"]
    )
    
    # Model Failure Template
    drill_templates_store["model_failure_template"] = DrillTemplate(
        template_id="model_failure_template",
        template_name="AI Model Failure Response Drill",
        incident_type=IncidentType.MODEL_FAILURE,
        default_severity=DrillSeverity.CRITICAL,
        scenario_template={
            "failed_model_ids": ["churn_prediction_v2", "fraud_detection_v3"],
            "error_type": "prediction_timeout",
            "affected_tenants": ["tenant_001", "tenant_002"]
        },
        success_criteria_template={
            "fallback_to_rba": True,
            "kill_switch_available": True,
            "evidence_preserved": True,
            "audit_trail_complete": True
        },
        applicable_industries=["All"],
        compliance_frameworks=["SOX", "GDPR", "HIPAA"]
    )
    
    # Compliance Violation Template
    drill_templates_store["compliance_violation_template"] = DrillTemplate(
        template_id="compliance_violation_template",
        template_name="Compliance Violation Detection Drill",
        incident_type=IncidentType.COMPLIANCE_VIOLATION,
        default_severity=DrillSeverity.CRITICAL,
        scenario_template={
            "violation_type": "data_residency_breach",
            "affected_data_types": ["customer_pii", "financial_records"],
            "geographic_scope": "EU_to_US_transfer"
        },
        success_criteria_template={
            "violation_detected": True,
            "data_transfer_blocked": True,
            "regulator_notification": True,
            "evidence_pack_generated": True
        },
        applicable_industries=["Financial Services", "Healthcare"],
        compliance_frameworks=["GDPR", "HIPAA", "DPDP"]
    )

@app.on_event("startup")
async def startup_event():
    """Initialize default templates on startup"""
    _initialize_default_templates()
    logger.info("✅ Cross-mode incident drill service initialized")

@app.post("/drills", response_model=IncidentDrill)
async def create_incident_drill(drill: IncidentDrill):
    """Create a new incident drill"""
    
    incident_drills_store[drill.drill_id] = drill
    drill_executions_store[drill.drill_id] = []
    
    logger.info(f"🚨 Created incident drill: {drill.drill_name} ({drill.incident_type.value})")
    return drill

@app.post("/drills/from-template", response_model=IncidentDrill)
async def create_drill_from_template(
    template_id: str,
    tenant_id: str,
    drill_name: str,
    scheduled_at: datetime,
    created_by: str,
    customizations: Optional[Dict[str, Any]] = None
):
    """Create drill from template"""
    
    if template_id not in drill_templates_store:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template = drill_templates_store[template_id]
    
    # Create drill from template
    drill = IncidentDrill(
        tenant_id=tenant_id,
        drill_name=drill_name,
        incident_type=template.incident_type,
        severity=template.default_severity,
        description=f"Automated drill based on {template.template_name}",
        simulated_conditions=template.scenario_template.copy(),
        success_criteria=template.success_criteria_template.copy(),
        scheduled_at=scheduled_at,
        duration_minutes=template.default_duration_minutes,
        created_by=created_by
    )
    
    # Apply customizations
    if customizations:
        for key, value in customizations.items():
            if hasattr(drill, key):
                setattr(drill, key, value)
    
    incident_drills_store[drill.drill_id] = drill
    drill_executions_store[drill.drill_id] = []
    
    logger.info(f"📋 Created drill from template {template_id}: {drill_name}")
    return drill

@app.post("/drills/{drill_id}/execute")
async def execute_incident_drill(drill_id: str, background_tasks: BackgroundTasks):
    """Execute incident drill across all target modes"""
    
    if drill_id not in incident_drills_store:
        raise HTTPException(status_code=404, detail="Drill not found")
    
    drill = incident_drills_store[drill_id]
    
    if drill.status != DrillStatus.PLANNED:
        raise HTTPException(status_code=400, detail="Drill is not in planned status")
    
    # Update drill status
    drill.status = DrillStatus.RUNNING
    drill.started_at = datetime.utcnow()
    incident_drills_store[drill_id] = drill
    
    # Execute drill in background across all modes
    background_tasks.add_task(_execute_drill_across_modes, drill)
    
    return {
        "message": "Drill execution started",
        "drill_id": drill_id,
        "target_modes": [mode.value for mode in drill.target_modes],
        "estimated_duration_minutes": drill.duration_minutes
    }

@app.get("/drills/{drill_id}/status")
async def get_drill_status(drill_id: str):
    """Get current drill status and progress"""
    
    if drill_id not in incident_drills_store:
        raise HTTPException(status_code=404, detail="Drill not found")
    
    drill = incident_drills_store[drill_id]
    executions = drill_executions_store.get(drill_id, [])
    
    # Calculate progress
    total_planned = len(drill.target_modes) * len(drill.workflow_ids) if drill.workflow_ids else len(drill.target_modes)
    completed_executions = len([e for e in executions if e.execution_end is not None])
    
    progress_percentage = (completed_executions / total_planned * 100) if total_planned > 0 else 0
    
    return {
        "drill_id": drill_id,
        "status": drill.status.value,
        "progress_percentage": progress_percentage,
        "total_executions_planned": total_planned,
        "completed_executions": completed_executions,
        "running_executions": len([e for e in executions if e.execution_end is None]),
        "started_at": drill.started_at.isoformat() if drill.started_at else None,
        "estimated_completion": (drill.started_at + timedelta(minutes=drill.duration_minutes)).isoformat() if drill.started_at else None
    }

@app.post("/drills/{drill_id}/report", response_model=DrillReport)
async def generate_drill_report(drill_id: str):
    """Generate comprehensive drill report"""
    
    if drill_id not in incident_drills_store:
        raise HTTPException(status_code=404, detail="Drill not found")
    
    drill = incident_drills_store[drill_id]
    executions = drill_executions_store.get(drill_id, [])
    
    if drill.status != DrillStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Drill must be completed to generate report")
    
    # Generate comprehensive report
    report = await _generate_comprehensive_report(drill, executions)
    drill_reports_store[report.report_id] = report
    
    logger.info(f"📊 Generated drill report: {report.report_id}")
    return report

@app.get("/drills/{drill_id}/executions", response_model=List[DrillExecution])
async def get_drill_executions(drill_id: str, mode: Optional[UXMode] = None):
    """Get drill executions with optional mode filtering"""
    
    if drill_id not in drill_executions_store:
        raise HTTPException(status_code=404, detail="Drill not found")
    
    executions = drill_executions_store[drill_id]
    
    if mode:
        executions = [e for e in executions if e.mode == mode]
    
    return executions

@app.get("/templates", response_model=List[DrillTemplate])
async def list_drill_templates(incident_type: Optional[IncidentType] = None):
    """List available drill templates"""
    
    templates = list(drill_templates_store.values())
    
    if incident_type:
        templates = [t for t in templates if t.incident_type == incident_type]
    
    return templates

@app.get("/analytics/drill-performance")
async def get_drill_analytics(
    tenant_id: Optional[str] = None,
    days: int = 90,
    incident_type: Optional[IncidentType] = None
):
    """Get drill performance analytics"""
    
    # Filter drills by criteria
    relevant_drills = []
    for drill in incident_drills_store.values():
        if tenant_id and drill.tenant_id != tenant_id:
            continue
        if incident_type and drill.incident_type != incident_type:
            continue
        if drill.created_at < datetime.utcnow() - timedelta(days=days):
            continue
        relevant_drills.append(drill)
    
    # Calculate analytics
    total_drills = len(relevant_drills)
    completed_drills = len([d for d in relevant_drills if d.status == DrillStatus.COMPLETED])
    
    # Mode performance comparison
    mode_stats = {}
    for mode in UXMode:
        mode_executions = []
        for drill in relevant_drills:
            mode_executions.extend([e for e in drill_executions_store.get(drill.drill_id, []) if e.mode == mode])
        
        if mode_executions:
            successful = len([e for e in mode_executions if e.success])
            avg_response_time = sum([e.response_time_ms for e in mode_executions if e.response_time_ms]) / len(mode_executions)
            
            mode_stats[mode.value] = {
                "total_executions": len(mode_executions),
                "successful_executions": successful,
                "success_rate": successful / len(mode_executions) * 100,
                "average_response_time_ms": avg_response_time
            }
    
    return {
        "period_days": days,
        "total_drills": total_drills,
        "completed_drills": completed_drills,
        "completion_rate": completed_drills / total_drills * 100 if total_drills > 0 else 0,
        "mode_performance": mode_stats,
        "incident_type_distribution": _calculate_incident_type_distribution(relevant_drills),
        "recommendations": _generate_drill_recommendations(relevant_drills)
    }

async def _execute_drill_across_modes(drill: IncidentDrill):
    """Execute drill across all target modes"""
    
    try:
        executions = []
        
        # Execute for each mode
        for mode in drill.target_modes:
            # Execute for each workflow or default workflow
            workflows = drill.workflow_ids if drill.workflow_ids else ["default_workflow"]
            
            for workflow_id in workflows:
                execution = await _simulate_drill_execution(drill, mode, workflow_id)
                executions.append(execution)
                drill_executions_store[drill.drill_id].append(execution)
        
        # Mark drill as completed
        drill.status = DrillStatus.COMPLETED
        drill.completed_at = datetime.utcnow()
        incident_drills_store[drill.drill_id] = drill
        
        logger.info(f"✅ Completed drill {drill.drill_id} with {len(executions)} executions")
        
    except Exception as e:
        drill.status = DrillStatus.FAILED
        incident_drills_store[drill.drill_id] = drill
        logger.error(f"❌ Drill {drill.drill_id} failed: {e}")

async def _simulate_drill_execution(drill: IncidentDrill, mode: UXMode, workflow_id: str) -> DrillExecution:
    """Simulate drill execution for a specific mode"""
    
    execution = DrillExecution(
        drill_id=drill.drill_id,
        mode=mode,
        workflow_id=workflow_id
    )
    
    # Simulate execution based on incident type and mode
    await asyncio.sleep(random.uniform(1, 3))  # Simulate execution time
    
    # Simulate results based on mode characteristics
    if mode == UXMode.CONVERSATIONAL:
        # Conversational mode typically has higher user interaction
        execution.user_interactions = random.randint(5, 15)
        execution.response_time_ms = random.randint(2000, 8000)
        execution.success = random.choice([True, True, False])  # 67% success rate
        execution.user_satisfaction_score = random.uniform(3.5, 4.8)
        
    elif mode == UXMode.ASSISTED:
        # Assisted mode balances automation and user control
        execution.user_interactions = random.randint(2, 8)
        execution.response_time_ms = random.randint(1500, 5000)
        execution.success = random.choice([True, True, True, False])  # 75% success rate
        execution.user_satisfaction_score = random.uniform(4.0, 4.9)
        
    else:  # UI_LED
        # UI-led mode requires more manual steps but is more predictable
        execution.user_interactions = random.randint(8, 20)
        execution.response_time_ms = random.randint(3000, 10000)
        execution.success = random.choice([True, True, True, True, False])  # 80% success rate
        execution.user_satisfaction_score = random.uniform(3.8, 4.6)
    
    # Simulate actions taken based on incident type
    if drill.incident_type == IncidentType.SLA_BREACH:
        execution.actions_taken = [ResponseAction.ESCALATE, ResponseAction.NOTIFY_STAKEHOLDERS]
    elif drill.incident_type == IncidentType.MODEL_FAILURE:
        execution.actions_taken = [ResponseAction.FALLBACK_TO_RBA, ResponseAction.GENERATE_EVIDENCE]
    elif drill.incident_type == IncidentType.COMPLIANCE_VIOLATION:
        execution.actions_taken = [ResponseAction.KILL_SWITCH, ResponseAction.GENERATE_EVIDENCE, ResponseAction.NOTIFY_STAKEHOLDERS]
    
    # Simulate performance metrics
    execution.cpu_usage_percent = random.uniform(20, 80)
    execution.memory_usage_mb = random.uniform(100, 500)
    execution.api_calls_made = random.randint(5, 25)
    
    # Simulate evidence generation
    if ResponseAction.GENERATE_EVIDENCE in execution.actions_taken:
        execution.evidence_packs_created = [str(uuid.uuid4()) for _ in range(random.randint(1, 3))]
        execution.audit_logs_generated = True
    
    # Add some execution logs
    execution.execution_logs = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "drill_started",
            "mode": mode.value,
            "incident_type": drill.incident_type.value
        },
        {
            "timestamp": (datetime.utcnow() + timedelta(seconds=5)).isoformat(),
            "event": "incident_detected",
            "details": drill.simulated_conditions
        },
        {
            "timestamp": (datetime.utcnow() + timedelta(seconds=10)).isoformat(),
            "event": "response_initiated",
            "actions": [action.value for action in execution.actions_taken]
        }
    ]
    
    execution.execution_end = datetime.utcnow()
    
    return execution

async def _generate_comprehensive_report(drill: IncidentDrill, executions: List[DrillExecution]) -> DrillReport:
    """Generate comprehensive drill report"""
    
    successful_executions = len([e for e in executions if e.success])
    failed_executions = len(executions) - successful_executions
    
    # Calculate mode performance
    mode_performance = {}
    for mode in UXMode:
        mode_executions = [e for e in executions if e.mode == mode]
        if mode_executions:
            successful = len([e for e in mode_executions if e.success])
            avg_response_time = sum([e.response_time_ms for e in mode_executions if e.response_time_ms]) / len(mode_executions)
            avg_user_satisfaction = sum([e.user_satisfaction_score for e in mode_executions if e.user_satisfaction_score]) / len(mode_executions)
            
            mode_performance[mode] = {
                "total_executions": len(mode_executions),
                "successful_executions": successful,
                "success_rate": successful / len(mode_executions) * 100,
                "average_response_time_ms": avg_response_time,
                "average_user_satisfaction": avg_user_satisfaction,
                "total_user_interactions": sum([e.user_interactions for e in mode_executions])
            }
    
    # Determine best and worst performing modes
    best_mode = max(mode_performance.keys(), key=lambda m: mode_performance[m]["success_rate"]) if mode_performance else None
    worst_mode = min(mode_performance.keys(), key=lambda m: mode_performance[m]["success_rate"]) if mode_performance else None
    
    # Calculate response effectiveness
    response_effectiveness = {}
    all_actions = []
    for execution in executions:
        all_actions.extend(execution.actions_taken)
    
    for action in ResponseAction:
        action_count = all_actions.count(action)
        if action_count > 0:
            # Calculate effectiveness based on success rate when this action was used
            action_executions = [e for e in executions if action in e.actions_taken]
            effectiveness = len([e for e in action_executions if e.success]) / len(action_executions) * 100
            response_effectiveness[action] = effectiveness
    
    # Generate insights and recommendations
    strengths = _identify_drill_strengths(executions, mode_performance)
    weaknesses = _identify_drill_weaknesses(executions, mode_performance)
    recommendations = _generate_drill_improvement_recommendations(executions, mode_performance)
    
    report = DrillReport(
        drill_id=drill.drill_id,
        overall_success=successful_executions > failed_executions,
        total_executions=len(executions),
        successful_executions=successful_executions,
        failed_executions=failed_executions,
        mode_performance=mode_performance,
        best_performing_mode=best_mode,
        worst_performing_mode=worst_mode,
        response_effectiveness=response_effectiveness,
        average_response_time_ms=sum([e.response_time_ms for e in executions if e.response_time_ms]) / len(executions) if executions else 0,
        compliance_score=_calculate_compliance_score(executions),
        strengths_identified=strengths,
        weaknesses_identified=weaknesses,
        improvement_recommendations=recommendations,
        evidence_packs_generated=sum([len(e.evidence_packs_created) for e in executions]),
        audit_trail_complete=all([e.audit_logs_generated for e in executions]),
        compliance_violations_found=sum([len(e.compliance_violations) for e in executions])
    )
    
    return report

def _identify_drill_strengths(executions: List[DrillExecution], mode_performance: Dict[UXMode, Dict[str, Any]]) -> List[str]:
    """Identify strengths from drill execution"""
    
    strengths = []
    
    # High success rates
    for mode, perf in mode_performance.items():
        if perf["success_rate"] > 80:
            strengths.append(f"{mode.value} mode shows excellent reliability with {perf['success_rate']:.1f}% success rate")
    
    # Fast response times
    fast_executions = [e for e in executions if e.response_time_ms and e.response_time_ms < 3000]
    if len(fast_executions) > len(executions) * 0.7:
        strengths.append("Rapid incident response with 70%+ of executions completing under 3 seconds")
    
    # Evidence generation
    evidence_generated = [e for e in executions if e.evidence_packs_created]
    if len(evidence_generated) > len(executions) * 0.8:
        strengths.append("Strong evidence generation capability across incident scenarios")
    
    return strengths

def _identify_drill_weaknesses(executions: List[DrillExecution], mode_performance: Dict[UXMode, Dict[str, Any]]) -> List[str]:
    """Identify weaknesses from drill execution"""
    
    weaknesses = []
    
    # Low success rates
    for mode, perf in mode_performance.items():
        if perf["success_rate"] < 60:
            weaknesses.append(f"{mode.value} mode shows concerning failure rate with only {perf['success_rate']:.1f}% success")
    
    # Slow response times
    slow_executions = [e for e in executions if e.response_time_ms and e.response_time_ms > 8000]
    if len(slow_executions) > len(executions) * 0.3:
        weaknesses.append("Response time concerns with 30%+ of executions taking over 8 seconds")
    
    # Compliance violations
    violations = sum([len(e.compliance_violations) for e in executions])
    if violations > 0:
        weaknesses.append(f"Compliance violations detected: {violations} total violations across all executions")
    
    return weaknesses

def _generate_drill_improvement_recommendations(executions: List[DrillExecution], mode_performance: Dict[UXMode, Dict[str, Any]]) -> List[str]:
    """Generate improvement recommendations"""
    
    recommendations = []
    
    # Mode-specific recommendations
    worst_mode = min(mode_performance.keys(), key=lambda m: mode_performance[m]["success_rate"]) if mode_performance else None
    if worst_mode and mode_performance[worst_mode]["success_rate"] < 70:
        recommendations.append(f"Focus improvement efforts on {worst_mode.value} mode - consider additional training or process refinement")
    
    # Response time recommendations
    avg_response_time = sum([e.response_time_ms for e in executions if e.response_time_ms]) / len(executions) if executions else 0
    if avg_response_time > 5000:
        recommendations.append("Optimize incident response workflows to reduce average response time below 5 seconds")
    
    # Evidence generation recommendations
    no_evidence = [e for e in executions if not e.evidence_packs_created]
    if len(no_evidence) > 0:
        recommendations.append("Ensure evidence pack generation is triggered for all incident types to maintain audit compliance")
    
    return recommendations

def _calculate_compliance_score(executions: List[DrillExecution]) -> float:
    """Calculate overall compliance score for the drill"""
    
    if not executions:
        return 0.0
    
    compliance_factors = []
    
    # Evidence generation compliance
    evidence_compliance = len([e for e in executions if e.evidence_packs_created]) / len(executions)
    compliance_factors.append(evidence_compliance)
    
    # Audit trail compliance
    audit_compliance = len([e for e in executions if e.audit_logs_generated]) / len(executions)
    compliance_factors.append(audit_compliance)
    
    # Violation penalty
    violations = sum([len(e.compliance_violations) for e in executions])
    violation_penalty = max(0, 1 - (violations * 0.1))  # 10% penalty per violation
    compliance_factors.append(violation_penalty)
    
    return sum(compliance_factors) / len(compliance_factors) * 100

def _calculate_incident_type_distribution(drills: List[IncidentDrill]) -> Dict[str, int]:
    """Calculate distribution of incident types"""
    
    distribution = {}
    for drill in drills:
        incident_type = drill.incident_type.value
        distribution[incident_type] = distribution.get(incident_type, 0) + 1
    
    return distribution

def _generate_drill_recommendations(drills: List[IncidentDrill]) -> List[str]:
    """Generate recommendations based on drill history"""
    
    recommendations = []
    
    if len(drills) < 4:
        recommendations.append("Increase drill frequency - aim for at least monthly incident drills")
    
    # Check incident type coverage
    incident_types_covered = set([d.incident_type for d in drills])
    if len(incident_types_covered) < 4:
        recommendations.append("Expand incident type coverage - test more diverse failure scenarios")
    
    # Check mode coverage
    modes_tested = set()
    for drill in drills:
        modes_tested.update(drill.target_modes)
    
    if len(modes_tested) < 3:
        recommendations.append("Ensure all UX modes are tested in incident scenarios")
    
    return recommendations

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Cross-Mode Incident Drill Service",
        "task": "3.4.29",
        "active_drills": len([d for d in incident_drills_store.values() if d.status == DrillStatus.RUNNING]),
        "completed_drills": len([d for d in incident_drills_store.values() if d.status == DrillStatus.COMPLETED]),
        "available_templates": len(drill_templates_store)
    }