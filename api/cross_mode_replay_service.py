"""
Task 3.3.31: Cross-Mode Replay Service
- Replay workflows in different UX modes
- Cross-mode traceability with workflow hash linking
- Logs merged for complete audit trail
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging
import json
import hashlib

app = FastAPI(title="RBIA Cross-Mode Replay Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class ReplayStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    workflow_id: str
    workflow_hash: str  # Hash of workflow definition for linking
    
    # Execution context
    ux_mode: UXMode
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Workflow definition and parameters
    workflow_definition: Dict[str, Any] = Field(default_factory=dict)
    input_parameters: Dict[str, Any] = Field(default_factory=dict)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    output_results: Dict[str, Any] = Field(default_factory=dict)
    execution_logs: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance metrics
    execution_time_ms: Optional[int] = None
    nodes_executed: int = 0
    errors_encountered: List[Dict[str, Any]] = Field(default_factory=list)

class ReplayRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Source execution to replay
    source_execution_id: str
    target_ux_mode: UXMode
    
    # Replay options
    preserve_parameters: bool = True
    preserve_user_inputs: bool = True
    preserve_overrides: bool = False  # Whether to replay manual overrides
    
    # Mode-specific options
    replay_options: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ReplayExecution(BaseModel):
    replay_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    tenant_id: str
    user_id: str
    
    # Source and target
    source_execution_id: str
    source_ux_mode: UXMode
    target_execution_id: str
    target_ux_mode: UXMode
    
    # Linking
    workflow_hash: str  # Same hash links all replays of same workflow
    replay_chain_id: str  # Links all replays in a chain
    
    # Status
    status: ReplayStatus = ReplayStatus.PENDING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Results comparison
    results_comparison: Dict[str, Any] = Field(default_factory=dict)
    differences_detected: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Merged logs
    merged_execution_logs: List[Dict[str, Any]] = Field(default_factory=list)

# In-memory stores (replace with database in production)
executions_store: Dict[str, WorkflowExecution] = {}
replay_requests_store: Dict[str, ReplayRequest] = {}
replay_executions_store: Dict[str, ReplayExecution] = {}

def _generate_workflow_hash(workflow_definition: Dict[str, Any]) -> str:
    """Generate a consistent hash for workflow definition to enable linking"""
    # Remove execution-specific fields and sort for consistent hashing
    hashable_definition = {
        "nodes": workflow_definition.get("nodes", []),
        "connections": workflow_definition.get("connections", []),
        "metadata": {k: v for k, v in workflow_definition.get("metadata", {}).items() 
                    if k not in ["execution_id", "timestamp", "user_id"]}
    }
    
    definition_str = json.dumps(hashable_definition, sort_keys=True)
    return hashlib.sha256(definition_str.encode()).hexdigest()[:16]

@app.post("/replay/executions", response_model=WorkflowExecution)
async def record_execution(execution: WorkflowExecution):
    """Record a workflow execution for potential replay"""
    
    # Generate workflow hash for linking
    execution.workflow_hash = _generate_workflow_hash(execution.workflow_definition)
    
    executions_store[execution.execution_id] = execution
    logger.info(f"Recorded execution {execution.execution_id} in {execution.ux_mode} mode")
    return execution

@app.get("/replay/executions/{execution_id}", response_model=WorkflowExecution)
async def get_execution(execution_id: str):
    """Get a specific workflow execution"""
    if execution_id not in executions_store:
        raise HTTPException(status_code=404, detail="Execution not found")
    return executions_store[execution_id]

@app.get("/replay/executions", response_model=List[WorkflowExecution])
async def get_executions(
    tenant_id: str,
    user_id: Optional[str] = None,
    ux_mode: Optional[UXMode] = None,
    workflow_hash: Optional[str] = None,
    limit: int = 50
):
    """Get workflow executions with optional filtering"""
    
    executions = [
        exec for exec in executions_store.values()
        if exec.tenant_id == tenant_id
    ]
    
    if user_id:
        executions = [e for e in executions if e.user_id == user_id]
    
    if ux_mode:
        executions = [e for e in executions if e.ux_mode == ux_mode]
    
    if workflow_hash:
        executions = [e for e in executions if e.workflow_hash == workflow_hash]
    
    # Sort by timestamp, most recent first
    executions.sort(key=lambda x: x.execution_timestamp, reverse=True)
    
    return executions[:limit]

@app.post("/replay/request", response_model=ReplayRequest)
async def create_replay_request(request: ReplayRequest):
    """Create a request to replay an execution in a different mode"""
    
    # Validate source execution exists
    if request.source_execution_id not in executions_store:
        raise HTTPException(status_code=404, detail="Source execution not found")
    
    source_execution = executions_store[request.source_execution_id]
    
    # Validate not replaying in same mode
    if source_execution.ux_mode == request.target_ux_mode:
        raise HTTPException(status_code=400, detail="Cannot replay in the same UX mode")
    
    replay_requests_store[request.request_id] = request
    logger.info(f"Created replay request {request.request_id}: {source_execution.ux_mode} -> {request.target_ux_mode}")
    return request

@app.post("/replay/execute/{request_id}", response_model=ReplayExecution)
async def execute_replay(
    request_id: str,
    background_tasks: BackgroundTasks
):
    """Execute a replay request"""
    
    if request_id not in replay_requests_store:
        raise HTTPException(status_code=404, detail="Replay request not found")
    
    request = replay_requests_store[request_id]
    source_execution = executions_store[request.source_execution_id]
    
    # Create replay execution record
    replay_execution = ReplayExecution(
        request_id=request_id,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        source_execution_id=request.source_execution_id,
        source_ux_mode=source_execution.ux_mode,
        target_execution_id=str(uuid.uuid4()),
        target_ux_mode=request.target_ux_mode,
        workflow_hash=source_execution.workflow_hash,
        replay_chain_id=_get_or_create_replay_chain_id(source_execution.workflow_hash)
    )
    
    replay_executions_store[replay_execution.replay_id] = replay_execution
    
    # Execute replay in background
    background_tasks.add_task(_perform_replay, replay_execution, source_execution, request)
    
    logger.info(f"Started replay execution {replay_execution.replay_id}")
    return replay_execution

def _get_or_create_replay_chain_id(workflow_hash: str) -> str:
    """Get existing replay chain ID or create new one for workflow hash"""
    
    # Find existing replay chain for this workflow hash
    existing_replays = [
        r for r in replay_executions_store.values()
        if r.workflow_hash == workflow_hash
    ]
    
    if existing_replays:
        return existing_replays[0].replay_chain_id
    else:
        return str(uuid.uuid4())

async def _perform_replay(
    replay_execution: ReplayExecution,
    source_execution: WorkflowExecution,
    request: ReplayRequest
):
    """Perform the actual replay execution"""
    
    try:
        replay_execution.status = ReplayStatus.IN_PROGRESS
        
        # Prepare execution parameters for target mode
        target_execution = WorkflowExecution(
            execution_id=replay_execution.target_execution_id,
            tenant_id=source_execution.tenant_id,
            user_id=request.user_id,
            workflow_id=source_execution.workflow_id,
            workflow_hash=source_execution.workflow_hash,
            ux_mode=request.target_ux_mode,
            workflow_definition=source_execution.workflow_definition.copy()
        )
        
        # Apply replay options
        if request.preserve_parameters:
            target_execution.input_parameters = source_execution.input_parameters.copy()
        
        if request.preserve_user_inputs:
            target_execution.execution_context = source_execution.execution_context.copy()
        
        # Mode-specific adaptations
        target_execution = _adapt_execution_for_target_mode(
            target_execution, source_execution, request.target_ux_mode, request.replay_options
        )
        
        # Simulate execution (in production, this would call actual workflow engine)
        execution_result = _simulate_workflow_execution(target_execution)
        
        # Store target execution
        executions_store[target_execution.execution_id] = execution_result
        
        # Compare results
        comparison = _compare_execution_results(source_execution, execution_result)
        replay_execution.results_comparison = comparison["comparison"]
        replay_execution.differences_detected = comparison["differences"]
        
        # Merge logs for complete audit trail
        replay_execution.merged_execution_logs = _merge_execution_logs(
            source_execution, execution_result
        )
        
        # Complete replay
        replay_execution.status = ReplayStatus.COMPLETED
        replay_execution.completed_at = datetime.utcnow()
        
        logger.info(f"Completed replay execution {replay_execution.replay_id}")
        
    except Exception as e:
        replay_execution.status = ReplayStatus.FAILED
        replay_execution.completed_at = datetime.utcnow()
        logger.error(f"Replay execution {replay_execution.replay_id} failed: {e}")

def _adapt_execution_for_target_mode(
    target_execution: WorkflowExecution,
    source_execution: WorkflowExecution,
    target_mode: UXMode,
    replay_options: Dict[str, Any]
) -> WorkflowExecution:
    """Adapt execution parameters for the target UX mode"""
    
    if target_mode == UXMode.CONVERSATIONAL:
        # For conversational mode, add natural language context
        target_execution.execution_context["conversation_context"] = {
            "original_mode": source_execution.ux_mode.value,
            "user_intent": replay_options.get("user_intent", "Replay previous workflow"),
            "conversation_history": []
        }
        
    elif target_mode == UXMode.ASSISTED:
        # For assisted mode, enable suggestion mechanisms
        target_execution.execution_context["assisted_mode_config"] = {
            "enable_suggestions": True,
            "confidence_threshold": replay_options.get("confidence_threshold", 0.7),
            "allow_overrides": replay_options.get("allow_overrides", True)
        }
        
    elif target_mode == UXMode.UI_LED:
        # For UI mode, ensure all parameters are explicitly set
        target_execution.execution_context["ui_mode_config"] = {
            "require_explicit_params": True,
            "show_advanced_options": replay_options.get("show_advanced", False)
        }
    
    return target_execution

def _simulate_workflow_execution(execution: WorkflowExecution) -> WorkflowExecution:
    """Simulate workflow execution (placeholder for actual execution engine)"""
    
    import time
    import random
    
    start_time = time.time()
    
    # Simulate execution based on mode
    if execution.ux_mode == UXMode.CONVERSATIONAL:
        # Simulate conversational interaction
        execution.execution_logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "conversation_started",
            "message": "Starting workflow from natural language description"
        })
        execution.nodes_executed = len(execution.workflow_definition.get("nodes", []))
        
    elif execution.ux_mode == UXMode.ASSISTED:
        # Simulate assisted execution with suggestions
        execution.execution_logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "assisted_execution_started",
            "suggestions_provided": random.randint(2, 5)
        })
        execution.nodes_executed = len(execution.workflow_definition.get("nodes", []))
        
    else:  # UI_LED
        # Simulate UI-led execution
        execution.execution_logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "ui_execution_started",
            "user_interactions": random.randint(5, 10)
        })
        execution.nodes_executed = len(execution.workflow_definition.get("nodes", []))
    
    # Simulate results with some variation
    base_results = {"score": random.uniform(0.7, 0.95), "confidence": random.uniform(0.8, 0.99)}
    
    # Add mode-specific result variations
    if execution.ux_mode == UXMode.CONVERSATIONAL:
        base_results["natural_language_summary"] = "Workflow completed successfully via conversation"
    elif execution.ux_mode == UXMode.ASSISTED:
        base_results["suggestions_accepted"] = random.randint(1, 3)
        base_results["overrides_made"] = random.randint(0, 1)
    
    execution.output_results = base_results
    execution.execution_time_ms = int((time.time() - start_time) * 1000)
    
    return execution

def _compare_execution_results(
    source_execution: WorkflowExecution,
    target_execution: WorkflowExecution
) -> Dict[str, Any]:
    """Compare results between source and target executions"""
    
    comparison = {
        "source_mode": source_execution.ux_mode.value,
        "target_mode": target_execution.ux_mode.value,
        "execution_time_diff_ms": target_execution.execution_time_ms - source_execution.execution_time_ms,
        "results_similarity": _calculate_results_similarity(
            source_execution.output_results, target_execution.output_results
        )
    }
    
    differences = []
    
    # Compare key result metrics
    for key in source_execution.output_results:
        if key in target_execution.output_results:
            source_val = source_execution.output_results[key]
            target_val = target_execution.output_results[key]
            
            if isinstance(source_val, (int, float)) and isinstance(target_val, (int, float)):
                diff_percentage = abs(source_val - target_val) / source_val * 100 if source_val != 0 else 0
                if diff_percentage > 5:  # 5% threshold for significant difference
                    differences.append({
                        "metric": key,
                        "source_value": source_val,
                        "target_value": target_val,
                        "difference_percentage": diff_percentage
                    })
    
    return {"comparison": comparison, "differences": differences}

def _calculate_results_similarity(source_results: Dict, target_results: Dict) -> float:
    """Calculate similarity score between two result sets"""
    
    common_keys = set(source_results.keys()) & set(target_results.keys())
    if not common_keys:
        return 0.0
    
    similarities = []
    for key in common_keys:
        source_val = source_results[key]
        target_val = target_results[key]
        
        if isinstance(source_val, (int, float)) and isinstance(target_val, (int, float)):
            if source_val == 0 and target_val == 0:
                similarities.append(1.0)
            elif source_val == 0 or target_val == 0:
                similarities.append(0.0)
            else:
                similarity = 1 - abs(source_val - target_val) / max(abs(source_val), abs(target_val))
                similarities.append(max(0, similarity))
    
    return sum(similarities) / len(similarities) if similarities else 0.0

def _merge_execution_logs(
    source_execution: WorkflowExecution,
    target_execution: WorkflowExecution
) -> List[Dict[str, Any]]:
    """Merge execution logs from both executions for complete audit trail"""
    
    merged_logs = []
    
    # Add source execution logs with context
    for log in source_execution.execution_logs:
        merged_log = log.copy()
        merged_log["execution_context"] = {
            "execution_id": source_execution.execution_id,
            "ux_mode": source_execution.ux_mode.value,
            "role": "source"
        }
        merged_logs.append(merged_log)
    
    # Add target execution logs with context
    for log in target_execution.execution_logs:
        merged_log = log.copy()
        merged_log["execution_context"] = {
            "execution_id": target_execution.execution_id,
            "ux_mode": target_execution.ux_mode.value,
            "role": "replay_target"
        }
        merged_logs.append(merged_log)
    
    # Sort by timestamp
    merged_logs.sort(key=lambda x: x.get("timestamp", ""))
    
    return merged_logs

@app.get("/replay/executions/{replay_id}", response_model=ReplayExecution)
async def get_replay_execution(replay_id: str):
    """Get a specific replay execution"""
    if replay_id not in replay_executions_store:
        raise HTTPException(status_code=404, detail="Replay execution not found")
    return replay_executions_store[replay_id]

@app.get("/replay/chain/{workflow_hash}")
async def get_replay_chain(workflow_hash: str, tenant_id: str):
    """Get all replays in a chain for a specific workflow hash"""
    
    # Get all executions with this workflow hash
    chain_executions = [
        exec for exec in executions_store.values()
        if exec.workflow_hash == workflow_hash and exec.tenant_id == tenant_id
    ]
    
    # Get all replays with this workflow hash
    chain_replays = [
        replay for replay in replay_executions_store.values()
        if replay.workflow_hash == workflow_hash and replay.tenant_id == tenant_id
    ]
    
    # Build chain visualization
    chain_data = {
        "workflow_hash": workflow_hash,
        "original_executions": len(chain_executions),
        "total_replays": len(chain_replays),
        "executions_by_mode": {},
        "replay_history": []
    }
    
    # Group executions by mode
    for mode in UXMode:
        mode_executions = [e for e in chain_executions if e.ux_mode == mode]
        chain_data["executions_by_mode"][mode.value] = len(mode_executions)
    
    # Add replay history
    for replay in sorted(chain_replays, key=lambda x: x.started_at):
        chain_data["replay_history"].append({
            "replay_id": replay.replay_id,
            "source_mode": replay.source_ux_mode.value,
            "target_mode": replay.target_ux_mode.value,
            "status": replay.status.value,
            "started_at": replay.started_at.isoformat(),
            "differences_count": len(replay.differences_detected)
        })
    
    return chain_data

@app.get("/replay/analytics/mode-transitions")
async def get_mode_transition_analytics(tenant_id: str):
    """Get analytics on mode transitions and replay patterns"""
    
    replays = [r for r in replay_executions_store.values() if r.tenant_id == tenant_id]
    
    # Analyze transition patterns
    transitions = {}
    for replay in replays:
        transition_key = f"{replay.source_ux_mode.value}_to_{replay.target_ux_mode.value}"
        if transition_key not in transitions:
            transitions[transition_key] = {
                "count": 0,
                "success_rate": 0,
                "avg_similarity": 0,
                "common_differences": []
            }
        
        transitions[transition_key]["count"] += 1
        
        if replay.status == ReplayStatus.COMPLETED:
            similarity = replay.results_comparison.get("results_similarity", 0)
            transitions[transition_key]["avg_similarity"] += similarity
    
    # Calculate averages
    for transition in transitions.values():
        if transition["count"] > 0:
            transition["avg_similarity"] /= transition["count"]
    
    return {
        "total_replays": len(replays),
        "transition_patterns": transitions,
        "most_common_transition": max(transitions.items(), key=lambda x: x[1]["count"])[0] if transitions else None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Cross-Mode Replay Service", "task": "3.3.31"}
