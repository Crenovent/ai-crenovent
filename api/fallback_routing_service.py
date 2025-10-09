"""
Task 3.2.39: Fallback matrices (AALA→RBIA→RBA; model→baseline) with auto-routing
- Predictable safety behavior
- Orchestrator rules
- Evidence logged
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Fallback Matrix & Auto-Routing Service")
logger = logging.getLogger(__name__)

class SystemType(str, Enum):
    AALA = "aala"           # AI Agent Language Assistant
    RBIA = "rbia"           # Rule-Based Intelligent Agent
    RBA = "rba"             # Rule-Based Agent
    BASELINE = "baseline"   # Simple deterministic rules

class FallbackTrigger(str, Enum):
    MODEL_FAILURE = "model_failure"
    CONFIDENCE_LOW = "confidence_low"
    LATENCY_HIGH = "latency_high"
    COST_LIMIT = "cost_limit"
    QUALITY_POOR = "quality_poor"
    SAFETY_VIOLATION = "safety_violation"
    MANUAL_OVERRIDE = "manual_override"
    KILL_SWITCH = "kill_switch"
    SYSTEM_OVERLOAD = "system_overload"

class FallbackStatus(str, Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"

class RoutingDecision(str, Enum):
    PROCEED = "proceed"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    BLOCK = "block"

class FallbackRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_name: str
    description: str
    
    # Trigger conditions
    trigger_type: FallbackTrigger
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Source and target systems
    source_system: SystemType
    target_system: SystemType
    
    # Routing logic
    evaluation_order: int = 1  # Lower numbers evaluated first
    enabled: bool = True
    
    # Thresholds and parameters
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    latency_threshold_ms: Optional[int] = None
    cost_threshold_usd: Optional[float] = None
    quality_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    tenant_id: str
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

class FallbackMatrix(BaseModel):
    matrix_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    matrix_name: str
    description: str
    
    # Matrix configuration
    tenant_id: str
    workflow_category: Optional[str] = None  # e.g., "credit_scoring", "fraud_detection"
    
    # Fallback chain definition
    primary_system: SystemType = SystemType.AALA
    fallback_chain: List[SystemType] = Field(default_factory=lambda: [SystemType.RBIA, SystemType.RBA, SystemType.BASELINE])
    
    # Rules for each transition
    fallback_rules: List[FallbackRule] = Field(default_factory=list)
    
    # Matrix status
    active: bool = True
    version: str = "1.0"
    
    # Metadata
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class RoutingRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    workflow_id: str
    user_input: str
    
    # Current system context
    current_system: SystemType
    attempted_systems: List[SystemType] = Field(default_factory=list)
    
    # Performance metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Error context
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RoutingResponse(BaseModel):
    request_id: str
    decision: RoutingDecision
    target_system: Optional[SystemType] = None
    
    # Decision context
    triggered_rule_id: Optional[str] = None
    fallback_reason: Optional[str] = None
    confidence_in_decision: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Evidence and logging
    evidence_logged: bool = False
    evidence_id: Optional[str] = None
    
    # Metadata
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[int] = None

class FallbackExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    matrix_id: str
    
    # Execution path
    attempted_systems: List[SystemType] = Field(default_factory=list)
    successful_system: Optional[SystemType] = None
    final_system: SystemType
    
    # Performance tracking
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    system_latencies: Dict[str, int] = Field(default_factory=dict)
    system_costs: Dict[str, float] = Field(default_factory=dict)
    
    # Results
    final_result: Optional[Dict[str, Any]] = None
    final_confidence: Optional[float] = None
    fallback_triggered: bool = False
    fallback_chain_exhausted: bool = False
    
    # Evidence
    evidence_trail: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    tenant_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: FallbackStatus = FallbackStatus.ACTIVE

# In-memory stores (replace with proper database in production)
fallback_matrices: Dict[str, FallbackMatrix] = {}
fallback_executions: Dict[str, FallbackExecution] = {}
routing_history: List[RoutingResponse] = []

def _create_default_fallback_matrix(tenant_id: str, created_by: str) -> FallbackMatrix:
    """Create a default fallback matrix with standard rules."""
    
    rules = [
        # AALA → RBIA fallback rules
        FallbackRule(
            rule_name="AALA Model Failure to RBIA",
            description="Fallback to RBIA when AALA model fails",
            trigger_type=FallbackTrigger.MODEL_FAILURE,
            source_system=SystemType.AALA,
            target_system=SystemType.RBIA,
            evaluation_order=1,
            tenant_id=tenant_id,
            created_by=created_by
        ),
        FallbackRule(
            rule_name="AALA Low Confidence to RBIA",
            description="Fallback to RBIA when AALA confidence is below threshold",
            trigger_type=FallbackTrigger.CONFIDENCE_LOW,
            source_system=SystemType.AALA,
            target_system=SystemType.RBIA,
            evaluation_order=2,
            confidence_threshold=0.7,
            tenant_id=tenant_id,
            created_by=created_by
        ),
        FallbackRule(
            rule_name="AALA High Latency to RBIA",
            description="Fallback to RBIA when AALA response time exceeds threshold",
            trigger_type=FallbackTrigger.LATENCY_HIGH,
            source_system=SystemType.AALA,
            target_system=SystemType.RBIA,
            evaluation_order=3,
            latency_threshold_ms=5000,
            tenant_id=tenant_id,
            created_by=created_by
        ),
        
        # RBIA → RBA fallback rules
        FallbackRule(
            rule_name="RBIA Model Failure to RBA",
            description="Fallback to RBA when RBIA model fails",
            trigger_type=FallbackTrigger.MODEL_FAILURE,
            source_system=SystemType.RBIA,
            target_system=SystemType.RBA,
            evaluation_order=1,
            tenant_id=tenant_id,
            created_by=created_by
        ),
        FallbackRule(
            rule_name="RBIA Low Confidence to RBA",
            description="Fallback to RBA when RBIA confidence is below threshold",
            trigger_type=FallbackTrigger.CONFIDENCE_LOW,
            source_system=SystemType.RBIA,
            target_system=SystemType.RBA,
            evaluation_order=2,
            confidence_threshold=0.6,
            tenant_id=tenant_id,
            created_by=created_by
        ),
        
        # RBA → Baseline fallback rules
        FallbackRule(
            rule_name="RBA Failure to Baseline",
            description="Fallback to baseline when RBA fails",
            trigger_type=FallbackTrigger.MODEL_FAILURE,
            source_system=SystemType.RBA,
            target_system=SystemType.BASELINE,
            evaluation_order=1,
            tenant_id=tenant_id,
            created_by=created_by
        ),
        
        # Kill switch rules (any system → baseline)
        FallbackRule(
            rule_name="Kill Switch to Baseline",
            description="Emergency fallback to baseline on kill switch activation",
            trigger_type=FallbackTrigger.KILL_SWITCH,
            source_system=SystemType.AALA,  # Will be applied to all systems
            target_system=SystemType.BASELINE,
            evaluation_order=0,  # Highest priority
            tenant_id=tenant_id,
            created_by=created_by
        )
    ]
    
    return FallbackMatrix(
        matrix_name="Default Fallback Matrix",
        description="Standard AALA→RBIA→RBA→Baseline fallback chain",
        tenant_id=tenant_id,
        fallback_rules=rules,
        created_by=created_by
    )

def _evaluate_fallback_rules(request: RoutingRequest, matrix: FallbackMatrix) -> Optional[FallbackRule]:
    """Evaluate fallback rules to determine if fallback is needed."""
    
    # Get applicable rules for current system
    applicable_rules = [
        rule for rule in matrix.fallback_rules
        if rule.source_system == request.current_system and rule.enabled
    ]
    
    # Sort by evaluation order
    applicable_rules.sort(key=lambda x: x.evaluation_order)
    
    for rule in applicable_rules:
        should_trigger = False
        
        if rule.trigger_type == FallbackTrigger.MODEL_FAILURE:
            should_trigger = request.error_message is not None
        
        elif rule.trigger_type == FallbackTrigger.CONFIDENCE_LOW:
            if request.confidence_score is not None and rule.confidence_threshold is not None:
                should_trigger = request.confidence_score < rule.confidence_threshold
        
        elif rule.trigger_type == FallbackTrigger.LATENCY_HIGH:
            if request.latency_ms is not None and rule.latency_threshold_ms is not None:
                should_trigger = request.latency_ms > rule.latency_threshold_ms
        
        elif rule.trigger_type == FallbackTrigger.COST_LIMIT:
            if request.cost_usd is not None and rule.cost_threshold_usd is not None:
                should_trigger = request.cost_usd > rule.cost_threshold_usd
        
        elif rule.trigger_type == FallbackTrigger.QUALITY_POOR:
            if request.quality_score is not None and rule.quality_threshold is not None:
                should_trigger = request.quality_score < rule.quality_threshold
        
        elif rule.trigger_type == FallbackTrigger.KILL_SWITCH:
            # Kill switch would be triggered externally
            should_trigger = request.error_type == "kill_switch_activated"
        
        elif rule.trigger_type == FallbackTrigger.MANUAL_OVERRIDE:
            should_trigger = request.error_type == "manual_override"
        
        if should_trigger:
            # Update rule statistics
            rule.last_triggered = datetime.utcnow()
            rule.trigger_count += 1
            return rule
    
    return None

def _log_routing_evidence(request: RoutingRequest, response: RoutingResponse, 
                         rule: Optional[FallbackRule] = None) -> str:
    """Log routing decision as evidence for audit trail."""
    
    evidence = {
        "request_id": request.request_id,
        "tenant_id": request.tenant_id,
        "workflow_id": request.workflow_id,
        "timestamp": datetime.utcnow().isoformat(),
        
        # Routing decision
        "decision": response.decision.value,
        "source_system": request.current_system.value,
        "target_system": response.target_system.value if response.target_system else None,
        
        # Context
        "confidence_score": request.confidence_score,
        "latency_ms": request.latency_ms,
        "cost_usd": request.cost_usd,
        "quality_score": request.quality_score,
        "error_message": request.error_message,
        
        # Rule details
        "triggered_rule": {
            "rule_id": rule.rule_id if rule else None,
            "rule_name": rule.rule_name if rule else None,
            "trigger_type": rule.trigger_type.value if rule else None
        } if rule else None,
        
        # Evidence integrity
        "evidence_hash": "placeholder_hash",  # Would be actual hash in production
        "logged_by": "fallback_routing_service"
    }
    
    evidence_id = str(uuid.uuid4())
    
    # In production, this would be stored in evidence database
    logger.info(f"Routing evidence logged: {evidence_id} - {response.decision.value}")
    
    return evidence_id

@app.post("/fallback-matrices", response_model=FallbackMatrix)
async def create_fallback_matrix(matrix: FallbackMatrix):
    """Create a new fallback matrix."""
    
    matrix.updated_at = datetime.utcnow()
    fallback_matrices[matrix.matrix_id] = matrix
    
    logger.info(f"Created fallback matrix {matrix.matrix_id}: {matrix.matrix_name}")
    return matrix

@app.post("/fallback-matrices/default")
async def create_default_matrix(tenant_id: str, created_by: str):
    """Create a default fallback matrix for a tenant."""
    
    matrix = _create_default_fallback_matrix(tenant_id, created_by)
    fallback_matrices[matrix.matrix_id] = matrix
    
    logger.info(f"Created default fallback matrix for tenant {tenant_id}")
    return {
        "status": "created",
        "matrix_id": matrix.matrix_id,
        "rules_count": len(matrix.fallback_rules)
    }

@app.post("/routing/evaluate", response_model=RoutingResponse)
async def evaluate_routing_decision(request: RoutingRequest, background_tasks: BackgroundTasks):
    """Evaluate whether to proceed with current system or fallback."""
    
    start_time = datetime.utcnow()
    
    # Find applicable fallback matrix
    matrix = None
    for m in fallback_matrices.values():
        if m.tenant_id == request.tenant_id and m.active:
            if not m.workflow_category or m.workflow_category in request.workflow_id:
                matrix = m
                break
    
    if not matrix:
        # No fallback matrix found, proceed with current system
        response = RoutingResponse(
            request_id=request.request_id,
            decision=RoutingDecision.PROCEED,
            fallback_reason="No applicable fallback matrix found"
        )
    else:
        # Evaluate fallback rules
        triggered_rule = _evaluate_fallback_rules(request, matrix)
        
        if triggered_rule:
            # Fallback triggered
            response = RoutingResponse(
                request_id=request.request_id,
                decision=RoutingDecision.FALLBACK,
                target_system=triggered_rule.target_system,
                triggered_rule_id=triggered_rule.rule_id,
                fallback_reason=f"Rule triggered: {triggered_rule.rule_name}"
            )
        else:
            # No fallback needed, proceed
            response = RoutingResponse(
                request_id=request.request_id,
                decision=RoutingDecision.PROCEED,
                fallback_reason="All fallback conditions passed"
            )
    
    # Calculate processing time
    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    response.processing_time_ms = int(processing_time)
    
    # Log evidence in background
    background_tasks.add_task(
        _log_routing_evidence, 
        request, 
        response, 
        triggered_rule if 'triggered_rule' in locals() else None
    )
    
    # Store in routing history
    routing_history.append(response)
    
    logger.info(f"Routing decision for {request.request_id}: {response.decision.value}")
    return response

@app.post("/routing/execute-fallback", response_model=FallbackExecution)
async def execute_fallback_chain(
    request: RoutingRequest,
    matrix_id: str,
    max_attempts: int = 4
):
    """Execute the complete fallback chain for a request."""
    
    if matrix_id not in fallback_matrices:
        raise HTTPException(status_code=404, detail="Fallback matrix not found")
    
    matrix = fallback_matrices[matrix_id]
    
    execution = FallbackExecution(
        request_id=request.request_id,
        matrix_id=matrix_id,
        final_system=request.current_system,
        tenant_id=request.tenant_id
    )
    
    current_system = request.current_system
    current_request = request.copy()
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        execution.attempted_systems.append(current_system)
        
        # Simulate system execution (in production, this would call actual systems)
        system_start = datetime.utcnow()
        
        # Mock execution based on system type
        if current_system == SystemType.AALA:
            # Simulate AALA execution
            success_rate = 0.85
            latency_ms = 2000
            cost_usd = 0.05
            confidence = 0.8
        elif current_system == SystemType.RBIA:
            success_rate = 0.95
            latency_ms = 500
            cost_usd = 0.01
            confidence = 0.7
        elif current_system == SystemType.RBA:
            success_rate = 0.98
            latency_ms = 100
            cost_usd = 0.001
            confidence = 0.6
        else:  # BASELINE
            success_rate = 1.0
            latency_ms = 50
            cost_usd = 0.0001
            confidence = 0.5
        
        # Simulate success/failure
        import random
        execution_successful = random.random() < success_rate
        
        system_end = datetime.utcnow()
        actual_latency = int((system_end - system_start).total_seconds() * 1000) + latency_ms
        
        # Track performance
        execution.system_latencies[current_system.value] = actual_latency
        execution.system_costs[current_system.value] = cost_usd
        execution.total_latency_ms += actual_latency
        execution.total_cost_usd += cost_usd
        
        # Log evidence
        evidence_entry = {
            "system": current_system.value,
            "attempt": attempts,
            "successful": execution_successful,
            "latency_ms": actual_latency,
            "cost_usd": cost_usd,
            "confidence": confidence if execution_successful else None,
            "timestamp": system_end.isoformat()
        }
        execution.evidence_trail.append(evidence_entry)
        
        if execution_successful:
            # System succeeded
            execution.successful_system = current_system
            execution.final_system = current_system
            execution.final_confidence = confidence
            execution.final_result = {
                "system_used": current_system.value,
                "confidence": confidence,
                "result": f"Mock result from {current_system.value}"
            }
            execution.status = FallbackStatus.COMPLETED
            break
        else:
            # System failed, evaluate fallback
            current_request.current_system = current_system
            current_request.confidence_score = None
            current_request.error_message = f"{current_system.value} execution failed"
            
            triggered_rule = _evaluate_fallback_rules(current_request, matrix)
            
            if triggered_rule:
                execution.fallback_triggered = True
                current_system = triggered_rule.target_system
                logger.info(f"Falling back from {current_request.current_system.value} to {current_system.value}")
            else:
                # No more fallback options
                execution.fallback_chain_exhausted = True
                execution.status = FallbackStatus.FAILED
                break
    
    execution.completed_at = datetime.utcnow()
    fallback_executions[execution.execution_id] = execution
    
    logger.info(f"Fallback execution {execution.execution_id} completed with status {execution.status.value}")
    return execution

@app.get("/fallback-matrices", response_model=List[FallbackMatrix])
async def list_fallback_matrices(tenant_id: Optional[str] = None):
    """List fallback matrices."""
    
    matrices = list(fallback_matrices.values())
    
    if tenant_id:
        matrices = [m for m in matrices if m.tenant_id == tenant_id]
    
    return matrices

@app.get("/routing/history")
async def get_routing_history(
    tenant_id: Optional[str] = None,
    decision: Optional[RoutingDecision] = None,
    limit: int = 100
):
    """Get routing decision history."""
    
    history = routing_history
    
    # Note: In production, would filter by tenant_id from request context
    if decision:
        history = [h for h in history if h.decision == decision]
    
    # Sort by timestamp (most recent first)
    history.sort(key=lambda x: x.decision_timestamp, reverse=True)
    
    return history[:limit]

@app.get("/routing/statistics/{tenant_id}")
async def get_routing_statistics(tenant_id: str, days: int = 30):
    """Get routing statistics for a tenant."""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Filter routing history for tenant and time period
    # Note: In production, would have tenant_id in routing history
    recent_history = [
        h for h in routing_history
        if h.decision_timestamp >= cutoff_date
    ]
    
    # Calculate statistics
    total_requests = len(recent_history)
    fallback_requests = len([h for h in recent_history if h.decision == RoutingDecision.FALLBACK])
    proceed_requests = len([h for h in recent_history if h.decision == RoutingDecision.PROCEED])
    
    fallback_rate = (fallback_requests / total_requests * 100) if total_requests > 0 else 0
    
    # Get fallback executions statistics
    tenant_executions = [e for e in fallback_executions.values() if e.tenant_id == tenant_id]
    successful_executions = len([e for e in tenant_executions if e.status == FallbackStatus.COMPLETED])
    failed_executions = len([e for e in tenant_executions if e.status == FallbackStatus.FAILED])
    
    return {
        "tenant_id": tenant_id,
        "period_days": days,
        "total_requests": total_requests,
        "fallback_rate_percent": round(fallback_rate, 2),
        "routing_decisions": {
            "proceed": proceed_requests,
            "fallback": fallback_requests
        },
        "fallback_executions": {
            "total": len(tenant_executions),
            "successful": successful_executions,
            "failed": failed_executions,
            "success_rate_percent": round((successful_executions / len(tenant_executions) * 100) if tenant_executions else 0, 2)
        },
        "average_processing_time_ms": round(sum(h.processing_time_ms or 0 for h in recent_history) / len(recent_history), 2) if recent_history else 0
    }

@app.post("/fallback-matrices/{matrix_id}/rules", response_model=FallbackRule)
async def add_fallback_rule(matrix_id: str, rule: FallbackRule):
    """Add a new fallback rule to a matrix."""
    
    if matrix_id not in fallback_matrices:
        raise HTTPException(status_code=404, detail="Fallback matrix not found")
    
    matrix = fallback_matrices[matrix_id]
    matrix.fallback_rules.append(rule)
    matrix.updated_at = datetime.utcnow()
    
    logger.info(f"Added fallback rule {rule.rule_id} to matrix {matrix_id}")
    return rule

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Fallback Matrix & Auto-Routing Service", "task": "3.2.39"}
