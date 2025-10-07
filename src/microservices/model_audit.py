"""
Model/Agent Call Audit Service
Tracks and audits AI model calls, tools used, decisions, and spans
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import json

app = FastAPI(
    title="Model/Agent Call Audit Service",
    description="Tracks and audits AI model calls, tools used, decisions, and spans",
    version="1.0.0"
)

class CallStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    INVALID_INPUT = "invalid_input"

class ToolType(str, Enum):
    CALENDAR_API = "calendar_api"
    TRANSCRIPTION_SERVICE = "transcription_service"
    SUMMARY_GENERATOR = "summary_generator"
    ACTION_EXTRACTOR = "action_extractor"
    CRM_API = "crm_api"
    DATABASE_QUERY = "database_query"
    EXTERNAL_API = "external_api"

class DecisionType(str, Enum):
    AUTOMATED = "automated"
    USER_OVERRIDE = "user_override"
    SYSTEM_FALLBACK = "system_fallback"
    HUMAN_REVIEW = "human_review"

class ModelCall(BaseModel):
    call_id: str
    tenant_id: str
    user_id: str
    session_id: str
    agent_id: Optional[str] = None
    model_name: str
    model_version: str
    prompt: str
    response: str
    tokens_used: int
    cost_usd: float
    latency_ms: int
    status: CallStatus
    error_message: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolCall(BaseModel):
    tool_call_id: str
    call_id: str
    tool_type: ToolType
    tool_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    status: CallStatus
    latency_ms: int
    error_message: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DecisionRecord(BaseModel):
    decision_id: str
    call_id: str
    decision_type: DecisionType
    original_action: Dict[str, Any]
    final_action: Dict[str, Any]
    confidence_score: float
    trust_score: float
    reasoning: str
    user_feedback: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AuditSpan(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: int
    status: CallStatus
    tags: Dict[str, Any] = Field(default_factory=dict)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AuditQueryRequest(BaseModel):
    tenant_id: str
    start_time: datetime
    end_time: datetime
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    model_name: Optional[str] = None
    status: Optional[CallStatus] = None
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None

class AuditSummary(BaseModel):
    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    models_used: List[str]
    tools_used: List[str]
    decisions_made: int
    user_overrides: int
    system_fallbacks: int

# In-memory storage (replace with time-series database in production)
model_calls_db: List[ModelCall] = []
tool_calls_db: List[ToolCall] = []
decisions_db: List[DecisionRecord] = []
audit_spans_db: List[AuditSpan] = []

@app.post("/audit/model-call")
async def record_model_call(call: ModelCall):
    """Record a model call"""
    call.call_id = str(uuid.uuid4())
    call.timestamp = datetime.utcnow()
    model_calls_db.append(call)
    return {"message": "Model call recorded successfully", "call_id": call.call_id}

@app.post("/audit/tool-call")
async def record_tool_call(tool_call: ToolCall):
    """Record a tool call"""
    tool_call.tool_call_id = str(uuid.uuid4())
    tool_call.timestamp = datetime.utcnow()
    tool_calls_db.append(tool_call)
    return {"message": "Tool call recorded successfully", "tool_call_id": tool_call.tool_call_id}

@app.post("/audit/decision")
async def record_decision(decision: DecisionRecord):
    """Record a decision"""
    decision.decision_id = str(uuid.uuid4())
    decision.timestamp = datetime.utcnow()
    decisions_db.append(decision)
    return {"message": "Decision recorded successfully", "decision_id": decision.decision_id}

@app.post("/audit/span")
async def record_span(span: AuditSpan):
    """Record an audit span"""
    span.span_id = str(uuid.uuid4())
    audit_spans_db.append(span)
    return {"message": "Audit span recorded successfully", "span_id": span.span_id}

@app.post("/audit/query", response_model=List[ModelCall])
async def query_model_calls(request: AuditQueryRequest):
    """Query model calls based on criteria"""
    filtered_calls = []
    
    for call in model_calls_db:
        # Filter by tenant
        if call.tenant_id != request.tenant_id:
            continue
        
        # Filter by time range
        if call.timestamp < request.start_time or call.timestamp > request.end_time:
            continue
        
        # Filter by user
        if request.user_id and call.user_id != request.user_id:
            continue
        
        # Filter by agent
        if request.agent_id and call.agent_id != request.agent_id:
            continue
        
        # Filter by model
        if request.model_name and call.model_name != request.model_name:
            continue
        
        # Filter by status
        if request.status and call.status != request.status:
            continue
        
        # Filter by cost range
        if request.min_cost and call.cost_usd < request.min_cost:
            continue
        
        if request.max_cost and call.cost_usd > request.max_cost:
            continue
        
        filtered_calls.append(call)
    
    return filtered_calls

@app.get("/audit/summary", response_model=AuditSummary)
async def get_audit_summary(
    tenant_id: str,
    start_time: datetime,
    end_time: datetime
):
    """Get audit summary for a time period"""
    
    # Filter calls by tenant and time range
    filtered_calls = [call for call in model_calls_db 
                     if call.tenant_id == tenant_id and 
                     start_time <= call.timestamp <= end_time]
    
    # Calculate summary statistics
    total_calls = len(filtered_calls)
    successful_calls = len([call for call in filtered_calls if call.status == CallStatus.SUCCESS])
    failed_calls = total_calls - successful_calls
    total_tokens = sum(call.tokens_used for call in filtered_calls)
    total_cost = sum(call.cost_usd for call in filtered_calls)
    avg_latency = sum(call.latency_ms for call in filtered_calls) / total_calls if total_calls > 0 else 0
    
    # Get unique models and tools used
    models_used = list(set(call.model_name for call in filtered_calls))
    
    # Get tools used from tool calls
    filtered_tool_calls = [tc for tc in tool_calls_db 
                          if any(tc.call_id == call.call_id for call in filtered_calls)]
    tools_used = list(set(tc.tool_name for tc in filtered_tool_calls))
    
    # Get decision statistics
    filtered_decisions = [d for d in decisions_db 
                         if any(d.call_id == call.call_id for call in filtered_calls)]
    decisions_made = len(filtered_decisions)
    user_overrides = len([d for d in filtered_decisions if d.decision_type == DecisionType.USER_OVERRIDE])
    system_fallbacks = len([d for d in filtered_decisions if d.decision_type == DecisionType.SYSTEM_FALLBACK])
    
    return AuditSummary(
        tenant_id=tenant_id,
        period_start=start_time,
        period_end=end_time,
        total_calls=total_calls,
        successful_calls=successful_calls,
        failed_calls=failed_calls,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        avg_latency_ms=avg_latency,
        models_used=models_used,
        tools_used=tools_used,
        decisions_made=decisions_made,
        user_overrides=user_overrides,
        system_fallbacks=system_fallbacks
    )

@app.get("/audit/trace/{trace_id}", response_model=List[AuditSpan])
async def get_trace_spans(trace_id: str):
    """Get all spans for a trace"""
    return [span for span in audit_spans_db if span.trace_id == trace_id]

@app.get("/audit/call/{call_id}/details")
async def get_call_details(call_id: str):
    """Get detailed information for a model call"""
    # Find the model call
    model_call = next((call for call in model_calls_db if call.call_id == call_id), None)
    if not model_call:
        raise HTTPException(status_code=404, detail="Model call not found")
    
    # Find related tool calls
    tool_calls = [tc for tc in tool_calls_db if tc.call_id == call_id]
    
    # Find related decisions
    decisions = [d for d in decisions_db if d.call_id == call_id]
    
    # Find related spans
    spans = [span for span in audit_spans_db if call_id in str(span.metadata.get("call_ids", []))]
    
    return {
        "model_call": model_call.dict(),
        "tool_calls": [tc.dict() for tc in tool_calls],
        "decisions": [d.dict() for d in decisions],
        "spans": [span.dict() for span in spans]
    }

@app.get("/audit/export")
async def export_audit_data(
    tenant_id: str,
    start_time: datetime,
    end_time: datetime,
    format: str = "json"
):
    """Export audit data"""
    
    # Get filtered data
    filtered_calls = [call for call in model_calls_db 
                     if call.tenant_id == tenant_id and 
                     start_time <= call.timestamp <= end_time]
    
    filtered_tool_calls = [tc for tc in tool_calls_db 
                          if any(tc.call_id == call.call_id for call in filtered_calls)]
    
    filtered_decisions = [d for d in decisions_db 
                         if any(d.call_id == call.call_id for call in filtered_calls)]
    
    if format == "json":
        return {
            "export_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "model_calls": [call.dict() for call in filtered_calls],
            "tool_calls": [tc.dict() for tc in filtered_tool_calls],
            "decisions": [d.dict() for d in filtered_decisions],
            "exported_at": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "model-audit",
        "timestamp": datetime.utcnow().isoformat(),
        "model_calls_count": len(model_calls_db),
        "tool_calls_count": len(tool_calls_db),
        "decisions_count": len(decisions_db),
        "spans_count": len(audit_spans_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
