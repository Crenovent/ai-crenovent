"""
Run Trace Schema Service
Implements run trace schema for tracking inputs, decisions, outputs, and trust scores
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import json

app = FastAPI(
    title="Run Trace Schema Service",
    description="Implements run trace schema for tracking automation execution",
    version="1.0.0"
)

class TraceStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DecisionType(str, Enum):
    AUTOMATED = "automated"
    USER_OVERRIDE = "user_override"
    SYSTEM_FALLBACK = "system_fallback"
    HUMAN_REVIEW = "human_review"
    POLICY_VIOLATION = "policy_violation"

class TraceInput(BaseModel):
    input_id: str
    trace_id: str
    input_type: str  # user_request, system_event, external_api
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TraceDecision(BaseModel):
    decision_id: str
    trace_id: str
    decision_type: DecisionType
    decision_point: str  # routing, confidence_evaluation, action_selection
    input_data: Dict[str, Any]
    decision_criteria: Dict[str, Any]
    decision_result: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TraceOutput(BaseModel):
    output_id: str
    trace_id: str
    output_type: str  # api_response, action_result, notification
    data: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TraceSpan(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: TraceStatus
    tags: Dict[str, Any] = Field(default_factory=dict)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RunTrace(BaseModel):
    trace_id: str
    tenant_id: str
    user_id: str
    session_id: str
    service_name: str
    operation_type: str
    status: TraceStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # Trust and confidence metrics
    overall_confidence_score: float = Field(ge=0.0, le=1.0)
    overall_trust_score: float = Field(ge=0.0, le=1.0)
    trust_drift: float = Field(default=0.0)
    
    # Trace components
    inputs: List[TraceInput] = Field(default_factory=list)
    decisions: List[TraceDecision] = Field(default_factory=list)
    outputs: List[TraceOutput] = Field(default_factory=list)
    spans: List[TraceSpan] = Field(default_factory=list)
    
    # Compliance and audit
    compliance_flags: List[str] = Field(default_factory=list)
    audit_trail_id: Optional[str] = None
    evidence_pack_id: Optional[str] = None
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TraceQueryRequest(BaseModel):
    tenant_id: str
    start_time: datetime
    end_time: datetime
    user_id: Optional[str] = None
    service_name: Optional[str] = None
    operation_type: Optional[str] = None
    status: Optional[TraceStatus] = None
    min_confidence: Optional[float] = None
    min_trust: Optional[float] = None
    has_decisions: Optional[bool] = None
    has_overrides: Optional[bool] = None

class TraceAnalytics(BaseModel):
    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_traces: int
    successful_traces: int
    failed_traces: int
    avg_confidence_score: float
    avg_trust_score: float
    avg_duration_ms: float
    decision_count: int
    override_count: int
    trust_drift_count: int
    service_breakdown: Dict[str, int]
    operation_breakdown: Dict[str, int]

# In-memory storage (replace with time-series database in production)
run_traces_db: Dict[str, RunTrace] = {}
trace_inputs_db: List[TraceInput] = []
trace_decisions_db: List[TraceDecision] = []
trace_outputs_db: List[TraceOutput] = []
trace_spans_db: List[TraceSpan] = []

@app.post("/traces", response_model=RunTrace)
async def create_trace(trace: RunTrace):
    """Create a new run trace"""
    trace.trace_id = str(uuid.uuid4())
    trace.start_time = datetime.utcnow()
    trace.status = TraceStatus.STARTED
    run_traces_db[trace.trace_id] = trace
    return trace

@app.get("/traces/{trace_id}", response_model=RunTrace)
async def get_trace(trace_id: str):
    """Get a run trace by ID"""
    trace = run_traces_db.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace

@app.put("/traces/{trace_id}", response_model=RunTrace)
async def update_trace(trace_id: str, updates: Dict[str, Any]):
    """Update a run trace"""
    trace = run_traces_db.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    for field, value in updates.items():
        if hasattr(trace, field):
            setattr(trace, field, value)
    
    # Update end time and duration if status changed to completed/failed
    if updates.get("status") in [TraceStatus.COMPLETED, TraceStatus.FAILED]:
        trace.end_time = datetime.utcnow()
        trace.duration_ms = int((trace.end_time - trace.start_time).total_seconds() * 1000)
    
    return trace

@app.post("/traces/{trace_id}/inputs", response_model=TraceInput)
async def add_trace_input(trace_id: str, input_data: TraceInput):
    """Add an input to a trace"""
    trace = run_traces_db.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    input_data.input_id = str(uuid.uuid4())
    input_data.trace_id = trace_id
    input_data.timestamp = datetime.utcnow()
    
    trace_inputs_db.append(input_data)
    trace.inputs.append(input_data)
    
    return input_data

@app.post("/traces/{trace_id}/decisions", response_model=TraceDecision)
async def add_trace_decision(trace_id: str, decision: TraceDecision):
    """Add a decision to a trace"""
    trace = run_traces_db.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    decision.decision_id = str(uuid.uuid4())
    decision.trace_id = trace_id
    decision.timestamp = datetime.utcnow()
    
    trace_decisions_db.append(decision)
    trace.decisions.append(decision)
    
    # Update overall confidence and trust scores
    if trace.decisions:
        trace.overall_confidence_score = sum(d.confidence_score for d in trace.decisions) / len(trace.decisions)
        trace.overall_trust_score = sum(d.trust_score for d in trace.decisions) / len(trace.decisions)
    
    return decision

@app.post("/traces/{trace_id}/outputs", response_model=TraceOutput)
async def add_trace_output(trace_id: str, output: TraceOutput):
    """Add an output to a trace"""
    trace = run_traces_db.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    output.output_id = str(uuid.uuid4())
    output.trace_id = trace_id
    output.timestamp = datetime.utcnow()
    
    trace_outputs_db.append(output)
    trace.outputs.append(output)
    
    return output

@app.post("/traces/{trace_id}/spans", response_model=TraceSpan)
async def add_trace_span(trace_id: str, span: TraceSpan):
    """Add a span to a trace"""
    trace = run_traces_db.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    span.span_id = str(uuid.uuid4())
    span.trace_id = trace_id
    span.start_time = datetime.utcnow()
    
    trace_spans_db.append(span)
    trace.spans.append(span)
    
    return span

@app.put("/traces/{trace_id}/spans/{span_id}", response_model=TraceSpan)
async def update_trace_span(trace_id: str, span_id: str, updates: Dict[str, Any]):
    """Update a trace span"""
    span = next((s for s in trace_spans_db if s.span_id == span_id and s.trace_id == trace_id), None)
    if not span:
        raise HTTPException(status_code=404, detail="Span not found")
    
    for field, value in updates.items():
        if hasattr(span, field):
            setattr(span, field, value)
    
    # Update end time and duration if status changed
    if updates.get("status") in [TraceStatus.COMPLETED, TraceStatus.FAILED]:
        span.end_time = datetime.utcnow()
        span.duration_ms = int((span.end_time - span.start_time).total_seconds() * 1000)
    
    return span

@app.post("/traces/query", response_model=List[RunTrace])
async def query_traces(request: TraceQueryRequest):
    """Query traces based on criteria"""
    filtered_traces = []
    
    for trace in run_traces_db.values():
        # Filter by tenant
        if trace.tenant_id != request.tenant_id:
            continue
        
        # Filter by time range
        if trace.start_time < request.start_time or trace.start_time > request.end_time:
            continue
        
        # Filter by user
        if request.user_id and trace.user_id != request.user_id:
            continue
        
        # Filter by service
        if request.service_name and trace.service_name != request.service_name:
            continue
        
        # Filter by operation type
        if request.operation_type and trace.operation_type != request.operation_type:
            continue
        
        # Filter by status
        if request.status and trace.status != request.status:
            continue
        
        # Filter by minimum confidence
        if request.min_confidence and trace.overall_confidence_score < request.min_confidence:
            continue
        
        # Filter by minimum trust
        if request.min_trust and trace.overall_trust_score < request.min_trust:
            continue
        
        # Filter by decisions
        if request.has_decisions is not None:
            has_decisions = len(trace.decisions) > 0
            if has_decisions != request.has_decisions:
                continue
        
        # Filter by overrides
        if request.has_overrides is not None:
            has_overrides = any(d.decision_type == DecisionType.USER_OVERRIDE for d in trace.decisions)
            if has_overrides != request.has_overrides:
                continue
        
        filtered_traces.append(trace)
    
    # Sort by start time (most recent first)
    filtered_traces.sort(key=lambda t: t.start_time, reverse=True)
    
    return filtered_traces

@app.get("/traces/{trace_id}/analytics")
async def get_trace_analytics(trace_id: str):
    """Get analytics for a specific trace"""
    trace = run_traces_db.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    # Calculate trace analytics
    decision_count = len(trace.decisions)
    override_count = len([d for d in trace.decisions if d.decision_type == DecisionType.USER_OVERRIDE])
    automated_count = len([d for d in trace.decisions if d.decision_type == DecisionType.AUTOMATED])
    
    # Calculate trust drift
    if len(trace.decisions) > 1:
        initial_trust = trace.decisions[0].trust_score
        final_trust = trace.decisions[-1].trust_score
        trust_drift = final_trust - initial_trust
    else:
        trust_drift = 0.0
    
    # Performance metrics
    avg_decision_confidence = sum(d.confidence_score for d in trace.decisions) / len(trace.decisions) if trace.decisions else 0
    avg_decision_trust = sum(d.trust_score for d in trace.decisions) / len(trace.decisions) if trace.decisions else 0
    
    return {
        "trace_id": trace_id,
        "tenant_id": trace.tenant_id,
        "user_id": trace.user_id,
        "service_name": trace.service_name,
        "operation_type": trace.operation_type,
        "status": trace.status,
        "duration_ms": trace.duration_ms,
        "overall_confidence_score": trace.overall_confidence_score,
        "overall_trust_score": trace.overall_trust_score,
        "trust_drift": trust_drift,
        "decision_count": decision_count,
        "override_count": override_count,
        "automated_count": automated_count,
        "avg_decision_confidence": avg_decision_confidence,
        "avg_decision_trust": avg_decision_trust,
        "inputs_count": len(trace.inputs),
        "outputs_count": len(trace.outputs),
        "spans_count": len(trace.spans),
        "compliance_flags": trace.compliance_flags,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/analytics", response_model=TraceAnalytics)
async def get_analytics(
    tenant_id: str,
    start_time: datetime,
    end_time: datetime
):
    """Get analytics for traces in a time period"""
    
    # Filter traces by tenant and time range
    filtered_traces = [trace for trace in run_traces_db.values() 
                      if trace.tenant_id == tenant_id and 
                      start_time <= trace.start_time <= end_time]
    
    if not filtered_traces:
        return TraceAnalytics(
            tenant_id=tenant_id,
            period_start=start_time,
            period_end=end_time,
            total_traces=0,
            successful_traces=0,
            failed_traces=0,
            avg_confidence_score=0.0,
            avg_trust_score=0.0,
            avg_duration_ms=0.0,
            decision_count=0,
            override_count=0,
            trust_drift_count=0,
            service_breakdown={},
            operation_breakdown={}
        )
    
    # Calculate basic statistics
    total_traces = len(filtered_traces)
    successful_traces = len([t for t in filtered_traces if t.status == TraceStatus.COMPLETED])
    failed_traces = len([t for t in filtered_traces if t.status == TraceStatus.FAILED])
    
    # Calculate averages
    avg_confidence_score = sum(t.overall_confidence_score for t in filtered_traces) / total_traces
    avg_trust_score = sum(t.overall_trust_score for t in filtered_traces) / total_traces
    avg_duration_ms = sum(t.duration_ms or 0 for t in filtered_traces) / total_traces
    
    # Calculate decision statistics
    decision_count = sum(len(t.decisions) for t in filtered_traces)
    override_count = sum(len([d for d in t.decisions if d.decision_type == DecisionType.USER_OVERRIDE]) for t in filtered_traces)
    trust_drift_count = len([t for t in filtered_traces if abs(t.trust_drift) > 0.1])
    
    # Service and operation breakdown
    service_breakdown = {}
    operation_breakdown = {}
    
    for trace in filtered_traces:
        service_breakdown[trace.service_name] = service_breakdown.get(trace.service_name, 0) + 1
        operation_breakdown[trace.operation_type] = operation_breakdown.get(trace.operation_type, 0) + 1
    
    return TraceAnalytics(
        tenant_id=tenant_id,
        period_start=start_time,
        period_end=end_time,
        total_traces=total_traces,
        successful_traces=successful_traces,
        failed_traces=failed_traces,
        avg_confidence_score=avg_confidence_score,
        avg_trust_score=avg_trust_score,
        avg_duration_ms=avg_duration_ms,
        decision_count=decision_count,
        override_count=override_count,
        trust_drift_count=trust_drift_count,
        service_breakdown=service_breakdown,
        operation_breakdown=operation_breakdown
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "run-trace-schema",
        "timestamp": datetime.utcnow().isoformat(),
        "traces_count": len(run_traces_db),
        "inputs_count": len(trace_inputs_db),
        "decisions_count": len(trace_decisions_db),
        "outputs_count": len(trace_outputs_db),
        "spans_count": len(trace_spans_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
