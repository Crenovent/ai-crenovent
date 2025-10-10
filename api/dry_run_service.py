"""
Dry-Run Service API - Task 6.1.21
==================================
FastAPI service for dry-run workflow execution
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl.execution.dry_run_executor import (
    DryRunExecutor, DryRunConfig, DryRunResult, DryRunMode,
    execute_workflow_dry_run
)

app = FastAPI(
    title="RBIA Dry-Run Service",
    description="Execute workflows in dry-run mode for testing and validation",
    version="1.0.0"
)


class DryRunRequest(BaseModel):
    """Request model for dry-run execution"""
    workflow_definition: Dict[str, Any]
    input_data: Dict[str, Any]
    mode: str = Field(default="stub", description="Dry-run mode: stub, validate, trace, performance")
    stub_values: Optional[Dict[str, Any]] = Field(default=None, description="Custom stub values for specific steps")
    validate_policies: bool = Field(default=True, description="Validate governance policies")
    validate_schemas: bool = Field(default=True, description="Validate input/output schemas")
    tenant_id: str = Field(default="1300", description="Tenant ID")
    user_id: int = Field(default=1319, description="User ID")


class DryRunResponse(BaseModel):
    """Response model for dry-run execution"""
    execution_id: str
    workflow_id: str
    mode: str
    success: bool
    
    # Validation
    structure_valid: bool
    validation_errors: List[str]
    validation_warnings: List[str]
    
    # Execution
    steps_executed: List[Dict[str, Any]]
    stub_results: Dict[str, Any]
    
    # Metrics
    execution_time_ms: int
    steps_count: int
    nodes_validated: int
    
    # Governance
    policy_checks_passed: int
    policy_checks_failed: int
    policy_violations: List[str]
    
    # Metadata
    timestamp: datetime
    error_message: Optional[str] = None


# In-memory storage for dry-run history
dry_run_history: Dict[str, DryRunResult] = {}


@app.post("/dry-run/execute", response_model=DryRunResponse)
async def execute_dry_run(request: DryRunRequest, background_tasks: BackgroundTasks):
    """
    Execute a workflow in dry-run mode
    
    Dry-run modes:
    - stub: Execute with stub results (no actual ML inference)
    - validate: Only validate structure and parameters
    - trace: Execute and trace without side effects
    - performance: Test performance without actual compute
    """
    try:
        # Validate mode
        if request.mode not in ['stub', 'validate', 'trace', 'performance']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Must be one of: stub, validate, trace, performance"
            )
        
        # Execute dry-run
        result = await execute_workflow_dry_run(
            workflow_definition=request.workflow_definition,
            input_data=request.input_data,
            mode=request.mode,
            stub_values=request.stub_values,
            tenant_id=request.tenant_id,
            user_id=request.user_id
        )
        
        # Store in history
        dry_run_history[result.execution_id] = result
        
        # Convert to response model
        response = DryRunResponse(
            execution_id=result.execution_id,
            workflow_id=result.workflow_id,
            mode=result.mode.value,
            success=result.success,
            structure_valid=result.structure_valid,
            validation_errors=result.validation_errors,
            validation_warnings=result.validation_warnings,
            steps_executed=result.steps_executed,
            stub_results=result.stub_results,
            execution_time_ms=result.execution_time_ms,
            steps_count=result.steps_count,
            nodes_validated=result.nodes_validated,
            policy_checks_passed=result.policy_checks_passed,
            policy_checks_failed=result.policy_checks_failed,
            policy_violations=result.policy_violations,
            timestamp=result.timestamp,
            error_message=result.error_message
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dry-run execution failed: {str(e)}")


@app.get("/dry-run/{execution_id}", response_model=DryRunResponse)
async def get_dry_run_result(execution_id: str):
    """Get dry-run execution result by ID"""
    if execution_id not in dry_run_history:
        raise HTTPException(status_code=404, detail=f"Dry-run execution not found: {execution_id}")
    
    result = dry_run_history[execution_id]
    
    return DryRunResponse(
        execution_id=result.execution_id,
        workflow_id=result.workflow_id,
        mode=result.mode.value,
        success=result.success,
        structure_valid=result.structure_valid,
        validation_errors=result.validation_errors,
        validation_warnings=result.validation_warnings,
        steps_executed=result.steps_executed,
        stub_results=result.stub_results,
        execution_time_ms=result.execution_time_ms,
        steps_count=result.steps_count,
        nodes_validated=result.nodes_validated,
        policy_checks_passed=result.policy_checks_passed,
        policy_checks_failed=result.policy_checks_failed,
        policy_violations=result.policy_violations,
        timestamp=result.timestamp,
        error_message=result.error_message
    )


@app.get("/dry-run/history/{tenant_id}")
async def get_dry_run_history(tenant_id: str, limit: int = 50):
    """Get dry-run execution history for a tenant"""
    # Filter by tenant (in production, this would use DB query)
    tenant_executions = [
        {
            'execution_id': result.execution_id,
            'workflow_id': result.workflow_id,
            'mode': result.mode.value,
            'success': result.success,
            'execution_time_ms': result.execution_time_ms,
            'timestamp': result.timestamp.isoformat()
        }
        for result in dry_run_history.values()
    ][:limit]
    
    return {
        'tenant_id': tenant_id,
        'total_executions': len(tenant_executions),
        'executions': tenant_executions
    }


@app.post("/dry-run/validate-workflow")
async def validate_workflow_structure(workflow_definition: Dict[str, Any]):
    """
    Validate workflow structure without execution
    Quick validation for workflow builder
    """
    try:
        # Create minimal request
        result = await execute_workflow_dry_run(
            workflow_definition=workflow_definition,
            input_data={},
            mode="validate"
        )
        
        return {
            'valid': result.success,
            'errors': result.validation_errors,
            'warnings': result.validation_warnings,
            'steps_validated': result.nodes_validated
        }
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [str(e)],
            'warnings': []
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'RBIA Dry-Run Service',
        'version': '1.0.0',
        'dry_runs_executed': len(dry_run_history)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)

