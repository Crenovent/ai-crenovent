"""
Task 3.2.6: SoD & approvals implementation
- Model dev ≠ approver
- CAB sign-off required
- Per-tenant risk owners
- Evidence stored for all decisions
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import json

app = FastAPI(title="RBIA Governance Approvals Service")

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ModelApprovalRequest(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    model_version: str = Field(..., description="Model version (semver)")
    tenant_id: str = Field(..., description="Tenant identifier")
    industry: str = Field(..., description="Industry overlay (SaaS, Banking, Insurance, etc.)")
    developer_id: str = Field(..., description="Model developer user ID")
    risk_level: RiskLevel = Field(..., description="Assessed risk level")
    model_card_url: str = Field(..., description="URL to model card documentation")
    test_results: Dict[str, Any] = Field(..., description="Model performance test results")
    compliance_overlays: List[str] = Field(default=[], description="Required compliance frameworks")
    justification: str = Field(..., description="Business justification for model")
    
class CABApproval(BaseModel):
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(..., description="Reference to approval request")
    cab_member_id: str = Field(..., description="CAB member approving")
    cab_member_role: str = Field(..., description="CAB member role/title")
    decision: ApprovalStatus = Field(..., description="Approval decision")
    decision_rationale: str = Field(..., description="Reason for decision")
    conditions: List[str] = Field(default=[], description="Approval conditions if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    evidence_hash: str = Field(..., description="Hash of evidence reviewed")

class TenantRiskOwner(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier")
    risk_owner_id: str = Field(..., description="Risk owner user ID")
    risk_owner_name: str = Field(..., description="Risk owner full name")
    risk_owner_role: str = Field(..., description="Risk owner role/title")
    approval_authority: List[RiskLevel] = Field(..., description="Risk levels they can approve")
    active: bool = Field(default=True, description="Whether risk owner is active")
    appointed_date: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage (replace with actual database)
approval_requests: Dict[str, Dict] = {}
cab_approvals: Dict[str, List[CABApproval]] = {}
tenant_risk_owners: Dict[str, List[TenantRiskOwner]] = {}
evidence_store: Dict[str, Dict] = {}

@app.post("/approval-requests", response_model=Dict[str, str])
async def create_approval_request(request: ModelApprovalRequest):
    """
    Create new model approval request with SoD validation
    """
    request_id = str(uuid.uuid4())
    
    # SoD Check: Ensure developer cannot approve their own model
    tenant_owners = tenant_risk_owners.get(request.tenant_id, [])
    developer_is_approver = any(
        owner.risk_owner_id == request.developer_id 
        for owner in tenant_owners
    )
    
    if developer_is_approver:
        raise HTTPException(
            status_code=400, 
            detail="SoD Violation: Model developer cannot be the approver"
        )
    
    # Validate risk owner exists for tenant
    if not tenant_owners:
        raise HTTPException(
            status_code=400,
            detail=f"No risk owners configured for tenant {request.tenant_id}"
        )
    
    # Store request
    approval_requests[request_id] = {
        **request.dict(),
        "request_id": request_id,
        "status": ApprovalStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "sod_validated": True
    }
    
    # Store evidence
    evidence_store[request_id] = {
        "request_data": request.dict(),
        "sod_check": {
            "developer_id": request.developer_id,
            "tenant_risk_owners": [owner.dict() for owner in tenant_owners],
            "sod_violation": False,
            "validated_at": datetime.utcnow().isoformat()
        }
    }
    
    return {
        "request_id": request_id,
        "status": "pending",
        "message": "Approval request created successfully"
    }

@app.post("/cab-approval/{request_id}")
async def submit_cab_approval(request_id: str, approval: CABApproval):
    """
    Submit CAB (Change Advisory Board) approval decision
    """
    if request_id not in approval_requests:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    request_data = approval_requests[request_id]
    
    # Validate CAB member has authority
    tenant_id = request_data["tenant_id"]
    risk_level = RiskLevel(request_data["risk_level"])
    
    # Check if CAB member is authorized risk owner
    tenant_owners = tenant_risk_owners.get(tenant_id, [])
    authorized_owner = None
    for owner in tenant_owners:
        if owner.risk_owner_id == approval.cab_member_id:
            if risk_level in owner.approval_authority:
                authorized_owner = owner
                break
    
    if not authorized_owner:
        raise HTTPException(
            status_code=403,
            detail="CAB member not authorized for this risk level"
        )
    
    # SoD Check: Ensure approver is not the developer
    if approval.cab_member_id == request_data["developer_id"]:
        raise HTTPException(
            status_code=400,
            detail="SoD Violation: Developer cannot approve their own model"
        )
    
    # Store CAB approval
    approval.request_id = request_id
    if request_id not in cab_approvals:
        cab_approvals[request_id] = []
    cab_approvals[request_id].append(approval)
    
    # Update request status
    approval_requests[request_id]["status"] = approval.decision.value
    approval_requests[request_id]["approved_by"] = approval.cab_member_id
    approval_requests[request_id]["approved_at"] = datetime.utcnow().isoformat()
    
    # Store evidence
    evidence_store[f"{request_id}_approval"] = {
        "approval_data": approval.dict(),
        "sod_check": {
            "approver_id": approval.cab_member_id,
            "developer_id": request_data["developer_id"],
            "sod_violation": False,
            "authorized_owner": authorized_owner.dict()
        },
        "decision_timestamp": datetime.utcnow().isoformat()
    }
    
    return {
        "message": "CAB approval recorded",
        "approval_id": approval.approval_id,
        "status": approval.decision.value
    }

@app.post("/tenant-risk-owners/{tenant_id}")
async def configure_tenant_risk_owner(tenant_id: str, owner: TenantRiskOwner):
    """
    Configure risk owner for tenant
    """
    owner.tenant_id = tenant_id
    
    if tenant_id not in tenant_risk_owners:
        tenant_risk_owners[tenant_id] = []
    
    tenant_risk_owners[tenant_id].append(owner)
    
    return {
        "message": "Risk owner configured successfully",
        "tenant_id": tenant_id,
        "risk_owner_id": owner.risk_owner_id
    }

@app.get("/approval-requests/{request_id}")
async def get_approval_request(request_id: str):
    """
    Get approval request details
    """
    if request_id not in approval_requests:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    request_data = approval_requests[request_id]
    approvals = cab_approvals.get(request_id, [])
    
    return {
        "request": request_data,
        "approvals": [approval.dict() for approval in approvals],
        "evidence_available": request_id in evidence_store
    }

@app.get("/tenant-risk-owners/{tenant_id}")
async def get_tenant_risk_owners(tenant_id: str):
    """
    Get risk owners for tenant
    """
    owners = tenant_risk_owners.get(tenant_id, [])
    return {
        "tenant_id": tenant_id,
        "risk_owners": [owner.dict() for owner in owners]
    }

@app.get("/evidence/{request_id}")
async def get_evidence_pack(request_id: str):
    """
    Get evidence pack for approval request
    """
    evidence = evidence_store.get(request_id)
    approval_evidence = evidence_store.get(f"{request_id}_approval")
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    return {
        "request_evidence": evidence,
        "approval_evidence": approval_evidence,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Governance Approvals",
        "task": "3.2.6 - SoD & Approvals"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
