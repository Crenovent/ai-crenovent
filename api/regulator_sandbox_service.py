"""
Dedicated Regulator Sandbox - Task 6.1.36
=========================================
Safe environment for regulators to test and audit RBIA
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid

app = FastAPI(title="Regulator Sandbox Service")


class SandboxStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"


class RegulatorSandbox(BaseModel):
    """Regulator sandbox environment"""
    sandbox_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    regulator_id: str
    regulator_name: str
    
    # Access control
    allowed_workflows: List[str] = Field(default_factory=list)
    allowed_datasets: List[str] = Field(default_factory=list)
    read_only: bool = True
    
    # Status
    status: SandboxStatus = SandboxStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    
    # Audit trail
    access_log: List[Dict[str, Any]] = Field(default_factory=list)


# Sandbox storage
sandboxes: Dict[str, RegulatorSandbox] = {}


# Pre-configured safe datasets for regulators
SAFE_DATASETS = {
    'demo_opportunities': {
        'description': 'Anonymized opportunity data for testing',
        'records': 100,
        'anonymized': True
    },
    'demo_forecasts': {
        'description': 'Historical forecast data (anonymized)',
        'records': 50,
        'anonymized': True
    }
}


# Pre-configured workflow templates
TEMPLATE_WORKFLOWS = {
    'churn_prediction_demo': {
        'id': 'churn_prediction_demo',
        'name': 'Churn Prediction Demo',
        'description': 'Demonstrate ML-powered churn prediction',
        'type': 'RBIA',
        'steps': [
            {'id': 'load_data', 'type': 'data_load', 'dataset': 'demo_opportunities'},
            {'id': 'predict_churn', 'type': 'ml_predict', 'model_id': 'demo_churn_model'},
            {'id': 'explain', 'type': 'ml_explain', 'method': 'shap'}
        ]
    },
    'governance_audit_demo': {
        'id': 'governance_audit_demo',
        'name': 'Governance Audit Demo',
        'description': 'Demonstrate governance controls',
        'type': 'RBIA',
        'governance': {
            'confidence_threshold': 0.7,
            'explainability_required': True,
            'evidence_capture': True
        }
    }
}


@app.post("/sandboxes/create", response_model=RegulatorSandbox)
async def create_regulator_sandbox(
    regulator_id: str,
    regulator_name: str,
    duration_days: int = 30
):
    """Create a sandbox environment for a regulator"""
    from datetime import timedelta
    
    sandbox = RegulatorSandbox(
        regulator_id=regulator_id,
        regulator_name=regulator_name,
        allowed_workflows=list(TEMPLATE_WORKFLOWS.keys()),
        allowed_datasets=list(SAFE_DATASETS.keys()),
        expires_at=datetime.utcnow() + timedelta(days=duration_days)
    )
    
    # Log creation
    sandbox.access_log.append({
        'action': 'sandbox_created',
        'timestamp': datetime.utcnow().isoformat(),
        'user': regulator_name
    })
    
    sandboxes[sandbox.sandbox_id] = sandbox
    
    return sandbox


@app.post("/sandboxes/{sandbox_id}/execute")
async def execute_workflow_in_sandbox(
    sandbox_id: str,
    workflow_id: str,
    regulator_user: str
):
    """Execute a workflow in sandbox environment"""
    if sandbox_id not in sandboxes:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox = sandboxes[sandbox_id]
    
    # Check sandbox is active
    if sandbox.status != SandboxStatus.ACTIVE:
        raise HTTPException(status_code=403, detail=f"Sandbox is {sandbox.status}")
    
    # Check workflow allowed
    if workflow_id not in sandbox.allowed_workflows:
        raise HTTPException(status_code=403, detail="Workflow not allowed in sandbox")
    
    # Log access
    sandbox.access_log.append({
        'action': 'workflow_executed',
        'workflow_id': workflow_id,
        'timestamp': datetime.utcnow().isoformat(),
        'user': regulator_user
    })
    
    # Return workflow template
    workflow = TEMPLATE_WORKFLOWS.get(workflow_id, {})
    
    return {
        'status': 'success',
        'sandbox_id': sandbox_id,
        'workflow': workflow,
        'execution_id': str(uuid.uuid4()),
        'note': 'Sandbox execution - no real data processed'
    }


@app.get("/sandboxes/{sandbox_id}/datasets")
async def list_sandbox_datasets(sandbox_id: str):
    """List available datasets in sandbox"""
    if sandbox_id not in sandboxes:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox = sandboxes[sandbox_id]
    datasets = {k: v for k, v in SAFE_DATASETS.items() if k in sandbox.allowed_datasets}
    
    return {'datasets': datasets}


@app.get("/sandboxes/{sandbox_id}/workflows")
async def list_sandbox_workflows(sandbox_id: str):
    """List available workflows in sandbox"""
    if sandbox_id not in sandboxes:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox = sandboxes[sandbox_id]
    workflows = {k: v for k, v in TEMPLATE_WORKFLOWS.items() if k in sandbox.allowed_workflows}
    
    return {'workflows': workflows}


@app.get("/sandboxes/{sandbox_id}/audit-log")
async def get_sandbox_audit_log(sandbox_id: str):
    """Get audit log for sandbox"""
    if sandbox_id not in sandboxes:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    sandbox = sandboxes[sandbox_id]
    return {'access_log': sandbox.access_log}

