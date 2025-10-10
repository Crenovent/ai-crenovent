"""
Task 3.3.45: UX Rollback Flow Service
- Revert workflows to deterministic UI mode for rollback safety
- Orchestrator config with evidence stored
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA UX Rollback Flow Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class RollbackReason(str, Enum):
    SAFETY_CONCERN = "safety_concern"
    MODEL_FAILURE = "model_failure"
    COMPLIANCE_ISSUE = "compliance_issue"
    USER_REQUEST = "user_request"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    REGULATORY_REQUIREMENT = "regulatory_requirement"

class RollbackStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowRollback(BaseModel):
    rollback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Rollback details
    workflow_id: str
    current_mode: UXMode
    target_mode: UXMode = UXMode.UI_LED  # Always rollback to deterministic UI mode
    
    # Reason and context
    rollback_reason: RollbackReason
    reason_description: str
    business_justification: Optional[str] = None
    
    # Workflow state
    current_workflow_config: Dict[str, Any] = Field(default_factory=dict)
    rollback_workflow_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution
    status: RollbackStatus = RollbackStatus.PENDING
    initiated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Evidence
    evidence_pack_id: Optional[str] = None
    audit_trail: List[str] = Field(default_factory=list)
    
    # Impact assessment
    affected_users: List[str] = Field(default_factory=list)
    data_preservation_confirmed: bool = False
    
    # Approval (if required)
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

class RollbackConfiguration(BaseModel):
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Rollback policies
    auto_rollback_enabled: bool = False
    require_approval_for_rollback: bool = True
    preserve_user_data: bool = True
    create_backup_before_rollback: bool = True
    
    # Triggers
    auto_rollback_triggers: List[RollbackReason] = Field(default_factory=lambda: [
        RollbackReason.SAFETY_CONCERN,
        RollbackReason.COMPLIANCE_ISSUE
    ])
    
    # Notification settings
    notify_users_on_rollback: bool = True
    notification_message_template: str = "Your workflow has been reverted to UI mode for safety reasons."
    
    # Retention
    evidence_retention_days: int = 365
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RollbackEvidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rollback_id: str
    tenant_id: str
    
    # Evidence content
    pre_rollback_state: Dict[str, Any] = Field(default_factory=dict)
    post_rollback_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration changes
    config_changes: List[Dict[str, Any]] = Field(default_factory=list)
    
    # User impact
    user_sessions_affected: int = 0
    workflows_modified: int = 0
    
    # Compliance
    regulatory_compliance_check: bool = True
    data_integrity_verified: bool = True
    
    # Audit
    rollback_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
rollback_requests_store: Dict[str, WorkflowRollback] = {}
rollback_configs_store: Dict[str, RollbackConfiguration] = {}
rollback_evidence_store: Dict[str, RollbackEvidence] = {}

def _create_default_rollback_config():
    """Create default rollback configuration"""
    
    default_config = RollbackConfiguration(
        tenant_id="default",
        auto_rollback_enabled=False,
        require_approval_for_rollback=True,
        preserve_user_data=True,
        create_backup_before_rollback=True,
        auto_rollback_triggers=[
            RollbackReason.SAFETY_CONCERN,
            RollbackReason.COMPLIANCE_ISSUE,
            RollbackReason.REGULATORY_REQUIREMENT
        ],
        notify_users_on_rollback=True,
        notification_message_template="Your workflow has been safely reverted to the standard UI mode to ensure reliable operation."
    )
    
    rollback_configs_store[default_config.config_id] = default_config
    logger.info("Created default rollback configuration")

# Initialize default configuration
_create_default_rollback_config()

@app.post("/rollback/config", response_model=RollbackConfiguration)
async def configure_rollback_settings(config: RollbackConfiguration):
    """Configure rollback settings for a tenant"""
    rollback_configs_store[config.config_id] = config
    logger.info(f"Configured rollback settings for tenant {config.tenant_id}")
    return config

@app.get("/rollback/config/{tenant_id}", response_model=RollbackConfiguration)
async def get_rollback_config(tenant_id: str):
    """Get rollback configuration for a tenant"""
    
    # Find tenant-specific config
    tenant_config = None
    for config in rollback_configs_store.values():
        if config.tenant_id == tenant_id:
            tenant_config = config
            break
    
    # Fallback to default config
    if not tenant_config:
        for config in rollback_configs_store.values():
            if config.tenant_id == "default":
                tenant_config = config
                break
    
    if not tenant_config:
        raise HTTPException(status_code=404, detail="Rollback configuration not found")
    
    return tenant_config

@app.post("/rollback/initiate", response_model=WorkflowRollback)
async def initiate_rollback(
    tenant_id: str,
    user_id: str,
    workflow_id: str,
    current_mode: UXMode,
    rollback_reason: RollbackReason,
    reason_description: str,
    business_justification: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Initiate a workflow rollback to UI mode"""
    
    # Get rollback configuration
    config = await get_rollback_config(tenant_id)
    
    # Create rollback request
    rollback = WorkflowRollback(
        tenant_id=tenant_id,
        user_id=user_id,
        workflow_id=workflow_id,
        current_mode=current_mode,
        rollback_reason=rollback_reason,
        reason_description=reason_description,
        business_justification=business_justification,
        requires_approval=config.require_approval_for_rollback
    )
    
    # Add initial audit trail entry
    rollback.audit_trail.append(f"Rollback initiated by user {user_id} at {rollback.initiated_at.isoformat()}")
    rollback.audit_trail.append(f"Reason: {rollback_reason.value} - {reason_description}")
    
    # Store rollback request
    rollback_requests_store[rollback.rollback_id] = rollback
    
    # Execute rollback in background if auto-approved
    if not rollback.requires_approval or rollback_reason in config.auto_rollback_triggers:
        rollback.status = RollbackStatus.IN_PROGRESS
        background_tasks.add_task(_execute_rollback, rollback, config)
    
    logger.info(f"Initiated rollback {rollback.rollback_id} for workflow {workflow_id}")
    return rollback

@app.post("/rollback/{rollback_id}/approve")
async def approve_rollback(
    rollback_id: str,
    approver_user_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Approve a pending rollback request"""
    
    if rollback_id not in rollback_requests_store:
        raise HTTPException(status_code=404, detail="Rollback request not found")
    
    rollback = rollback_requests_store[rollback_id]
    
    if rollback.status != RollbackStatus.PENDING:
        raise HTTPException(status_code=400, detail="Rollback is not in pending state")
    
    # Approve rollback
    rollback.approved_by = approver_user_id
    rollback.approved_at = datetime.utcnow()
    rollback.status = RollbackStatus.IN_PROGRESS
    
    rollback.audit_trail.append(f"Rollback approved by {approver_user_id} at {rollback.approved_at.isoformat()}")
    
    # Get configuration and execute rollback
    config = await get_rollback_config(rollback.tenant_id)
    background_tasks.add_task(_execute_rollback, rollback, config)
    
    logger.info(f"Approved rollback {rollback_id} by user {approver_user_id}")
    return {"status": "approved", "rollback_id": rollback_id}

async def _execute_rollback(rollback: WorkflowRollback, config: RollbackConfiguration):
    """Execute the actual rollback process"""
    
    try:
        # Step 1: Create backup if required
        if config.create_backup_before_rollback:
            await _create_workflow_backup(rollback)
        
        # Step 2: Capture pre-rollback state
        pre_rollback_state = await _capture_workflow_state(rollback.workflow_id, rollback.current_mode)
        
        # Step 3: Perform rollback to UI mode
        rollback_config = await _generate_ui_mode_config(rollback.workflow_id, rollback.current_workflow_config)
        rollback.rollback_workflow_config = rollback_config
        
        # Step 4: Apply rollback configuration
        await _apply_rollback_configuration(rollback.workflow_id, rollback_config)
        
        # Step 5: Capture post-rollback state
        post_rollback_state = await _capture_workflow_state(rollback.workflow_id, UXMode.UI_LED)
        
        # Step 6: Generate evidence pack
        evidence = await _generate_rollback_evidence(rollback, pre_rollback_state, post_rollback_state)
        rollback_evidence_store[evidence.evidence_id] = evidence
        rollback.evidence_pack_id = evidence.evidence_id
        
        # Step 7: Notify affected users if configured
        if config.notify_users_on_rollback:
            await _notify_affected_users(rollback, config.notification_message_template)
        
        # Step 8: Complete rollback
        rollback.status = RollbackStatus.COMPLETED
        rollback.completed_at = datetime.utcnow()
        rollback.data_preservation_confirmed = True
        
        rollback.audit_trail.append(f"Rollback completed successfully at {rollback.completed_at.isoformat()}")
        
        logger.info(f"Successfully completed rollback {rollback.rollback_id}")
        
    except Exception as e:
        rollback.status = RollbackStatus.FAILED
        rollback.completed_at = datetime.utcnow()
        rollback.audit_trail.append(f"Rollback failed at {rollback.completed_at.isoformat()}: {str(e)}")
        
        logger.error(f"Rollback {rollback.rollback_id} failed: {e}")

async def _create_workflow_backup(rollback: WorkflowRollback):
    """Create backup of current workflow state"""
    rollback.audit_trail.append("Workflow backup created")
    # In production, this would create actual backup

async def _capture_workflow_state(workflow_id: str, mode: UXMode) -> Dict[str, Any]:
    """Capture current workflow state"""
    # Simulate workflow state capture
    return {
        "workflow_id": workflow_id,
        "mode": mode.value,
        "configuration": {"nodes": [], "connections": []},
        "captured_at": datetime.utcnow().isoformat()
    }

async def _generate_ui_mode_config(workflow_id: str, current_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deterministic UI mode configuration"""
    
    # Convert current configuration to deterministic UI mode
    ui_config = {
        "workflow_id": workflow_id,
        "mode": UXMode.UI_LED.value,
        "deterministic": True,
        "nodes": [],
        "connections": [],
        "ui_settings": {
            "show_advanced_options": False,
            "require_explicit_confirmation": True,
            "disable_ai_suggestions": True,
            "enable_manual_validation": True
        }
    }
    
    # Convert nodes to UI-compatible format
    if "nodes" in current_config:
        for node in current_config["nodes"]:
            ui_node = {
                "id": node.get("id"),
                "type": node.get("type"),
                "config": node.get("config", {}),
                "ui_mode": "manual",  # Force manual configuration
                "ai_assistance": "disabled"  # Disable AI assistance
            }
            ui_config["nodes"].append(ui_node)
    
    return ui_config

async def _apply_rollback_configuration(workflow_id: str, rollback_config: Dict[str, Any]):
    """Apply the rollback configuration to the workflow"""
    # In production, this would update the actual workflow configuration
    pass

async def _generate_rollback_evidence(
    rollback: WorkflowRollback,
    pre_state: Dict[str, Any],
    post_state: Dict[str, Any]
) -> RollbackEvidence:
    """Generate evidence pack for the rollback"""
    
    config_changes = [
        {
            "field": "mode",
            "old_value": rollback.current_mode.value,
            "new_value": rollback.target_mode.value,
            "timestamp": datetime.utcnow().isoformat()
        },
        {
            "field": "ai_assistance",
            "old_value": "enabled",
            "new_value": "disabled",
            "timestamp": datetime.utcnow().isoformat()
        }
    ]
    
    timeline = [
        {
            "timestamp": rollback.initiated_at.isoformat(),
            "event": "Rollback initiated",
            "details": f"Reason: {rollback.rollback_reason.value}"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Configuration applied",
            "details": "Workflow reverted to deterministic UI mode"
        }
    ]
    
    evidence = RollbackEvidence(
        rollback_id=rollback.rollback_id,
        tenant_id=rollback.tenant_id,
        pre_rollback_state=pre_state,
        post_rollback_state=post_state,
        config_changes=config_changes,
        user_sessions_affected=len(rollback.affected_users),
        workflows_modified=1,
        rollback_timeline=timeline
    )
    
    return evidence

async def _notify_affected_users(rollback: WorkflowRollback, message_template: str):
    """Notify users affected by the rollback"""
    rollback.audit_trail.append(f"Notified {len(rollback.affected_users)} affected users")
    # In production, this would send actual notifications

@app.get("/rollback/requests", response_model=List[WorkflowRollback])
async def get_rollback_requests(
    tenant_id: str,
    status: Optional[RollbackStatus] = None,
    user_id: Optional[str] = None
):
    """Get rollback requests for a tenant"""
    
    requests = [r for r in rollback_requests_store.values() if r.tenant_id == tenant_id]
    
    if status:
        requests = [r for r in requests if r.status == status]
    
    if user_id:
        requests = [r for r in requests if r.user_id == user_id]
    
    # Sort by initiation time (most recent first)
    requests.sort(key=lambda x: x.initiated_at, reverse=True)
    
    return requests

@app.get("/rollback/requests/{rollback_id}", response_model=WorkflowRollback)
async def get_rollback_request(rollback_id: str):
    """Get a specific rollback request"""
    
    if rollback_id not in rollback_requests_store:
        raise HTTPException(status_code=404, detail="Rollback request not found")
    
    return rollback_requests_store[rollback_id]

@app.get("/rollback/evidence/{rollback_id}", response_model=RollbackEvidence)
async def get_rollback_evidence(rollback_id: str):
    """Get evidence pack for a rollback"""
    
    evidence = None
    for ev in rollback_evidence_store.values():
        if ev.rollback_id == rollback_id:
            evidence = ev
            break
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Rollback evidence not found")
    
    return evidence

@app.post("/rollback/{rollback_id}/cancel")
async def cancel_rollback(rollback_id: str, cancelled_by: str):
    """Cancel a pending rollback request"""
    
    if rollback_id not in rollback_requests_store:
        raise HTTPException(status_code=404, detail="Rollback request not found")
    
    rollback = rollback_requests_store[rollback_id]
    
    if rollback.status not in [RollbackStatus.PENDING, RollbackStatus.IN_PROGRESS]:
        raise HTTPException(status_code=400, detail="Cannot cancel rollback in current state")
    
    rollback.status = RollbackStatus.CANCELLED
    rollback.completed_at = datetime.utcnow()
    rollback.audit_trail.append(f"Rollback cancelled by {cancelled_by} at {rollback.completed_at.isoformat()}")
    
    logger.info(f"Cancelled rollback {rollback_id} by user {cancelled_by}")
    return {"status": "cancelled", "rollback_id": rollback_id}

@app.get("/rollback/analytics/statistics")
async def get_rollback_analytics(tenant_id: str):
    """Get rollback analytics for a tenant"""
    
    tenant_rollbacks = [r for r in rollback_requests_store.values() if r.tenant_id == tenant_id]
    
    if not tenant_rollbacks:
        return {"message": "No rollback requests found"}
    
    analytics = {
        "total_rollbacks": len(tenant_rollbacks),
        "by_status": {},
        "by_reason": {},
        "by_source_mode": {},
        "success_rate": 0,
        "avg_completion_time_minutes": 0
    }
    
    # Analyze by status
    for status in RollbackStatus:
        status_count = len([r for r in tenant_rollbacks if r.status == status])
        analytics["by_status"][status.value] = status_count
    
    # Analyze by reason
    for reason in RollbackReason:
        reason_count = len([r for r in tenant_rollbacks if r.rollback_reason == reason])
        analytics["by_reason"][reason.value] = reason_count
    
    # Analyze by source mode
    for mode in UXMode:
        mode_count = len([r for r in tenant_rollbacks if r.current_mode == mode])
        analytics["by_source_mode"][mode.value] = mode_count
    
    # Calculate success rate
    completed_rollbacks = [r for r in tenant_rollbacks if r.status == RollbackStatus.COMPLETED]
    analytics["success_rate"] = (len(completed_rollbacks) / len(tenant_rollbacks)) * 100
    
    # Calculate average completion time
    if completed_rollbacks:
        total_time = sum(
            (r.completed_at - r.initiated_at).total_seconds() / 60
            for r in completed_rollbacks if r.completed_at
        )
        analytics["avg_completion_time_minutes"] = total_time / len(completed_rollbacks)
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA UX Rollback Flow Service", "task": "3.3.45"}
