"""
Task 3.2.8: Kill switch per model/node/workflow
- Kill switch per model/node/workflow
- Global "intelligence off" → RBA fallback
- Instant containment capability
- Must be audited
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import json
import asyncio

app = FastAPI(title="RBIA Kill Switch Service")

class KillSwitchScope(str, Enum):
    MODEL = "model"
    NODE = "node"
    WORKFLOW = "workflow"
    TENANT = "tenant"
    GLOBAL = "global"

class KillSwitchStatus(str, Enum):
    ACTIVE = "active"        # Intelligence is active
    KILLED = "killed"        # Intelligence is disabled, fallback to RBA
    MAINTENANCE = "maintenance"  # Temporarily disabled for maintenance

class KillSwitchTrigger(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    DRIFT_DETECTED = "drift_detected"
    BIAS_DETECTED = "bias_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_VIOLATION = "compliance_violation"

class KillSwitch(BaseModel):
    switch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scope: KillSwitchScope = Field(..., description="Scope of the kill switch")
    target_id: str = Field(..., description="ID of the target (model/node/workflow/tenant)")
    target_name: str = Field(..., description="Human-readable name of target")
    tenant_id: str = Field(..., description="Tenant identifier")
    status: KillSwitchStatus = Field(default=KillSwitchStatus.ACTIVE, description="Current status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="User who created the kill switch")

class KillSwitchActivation(BaseModel):
    activation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    switch_id: str = Field(..., description="Kill switch identifier")
    trigger_type: KillSwitchTrigger = Field(..., description="What triggered the kill switch")
    triggered_by: str = Field(..., description="User or system that triggered")
    reason: str = Field(..., description="Reason for activation")
    evidence: Dict[str, Any] = Field(default={}, description="Supporting evidence")
    activated_at: datetime = Field(default_factory=datetime.utcnow)
    auto_revert_at: Optional[datetime] = Field(None, description="Automatic revert time if applicable")

class FallbackConfiguration(BaseModel):
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scope: KillSwitchScope = Field(..., description="Scope of fallback")
    target_id: str = Field(..., description="Target identifier")
    fallback_type: str = Field(..., description="Type of fallback (RBA, baseline_model, etc.)")
    fallback_config: Dict[str, Any] = Field(..., description="Fallback configuration")
    tenant_id: str = Field(..., description="Tenant identifier")
    active: bool = Field(default=True, description="Whether fallback is active")

# In-memory storage (replace with actual database)
kill_switches: Dict[str, KillSwitch] = {}
activations: Dict[str, List[KillSwitchActivation]] = {}
fallback_configs: Dict[str, FallbackConfiguration] = {}
audit_log: List[Dict[str, Any]] = []

# Global kill switch state
global_intelligence_enabled = True

@app.post("/kill-switches", response_model=Dict[str, str])
async def create_kill_switch(switch: KillSwitch):
    """
    Create a new kill switch for model/node/workflow
    """
    switch_id = switch.switch_id
    kill_switches[switch_id] = switch
    
    # Audit log
    audit_entry = {
        "event": "kill_switch_created",
        "switch_id": switch_id,
        "scope": switch.scope.value,
        "target_id": switch.target_id,
        "tenant_id": switch.tenant_id,
        "created_by": switch.created_by,
        "timestamp": datetime.utcnow().isoformat()
    }
    audit_log.append(audit_entry)
    
    return {
        "switch_id": switch_id,
        "status": "created",
        "scope": switch.scope.value,
        "target": switch.target_name
    }

@app.post("/kill-switches/{switch_id}/activate")
async def activate_kill_switch(switch_id: str, activation: KillSwitchActivation):
    """
    Activate kill switch - disable intelligence and fallback to RBA
    """
    if switch_id not in kill_switches:
        raise HTTPException(status_code=404, detail="Kill switch not found")
    
    switch = kill_switches[switch_id]
    activation.switch_id = switch_id
    
    # Update switch status
    switch.status = KillSwitchStatus.KILLED
    switch.updated_at = datetime.utcnow()
    
    # Store activation
    if switch_id not in activations:
        activations[switch_id] = []
    activations[switch_id].append(activation)
    
    # Handle global kill switch
    global global_intelligence_enabled
    if switch.scope == KillSwitchScope.GLOBAL:
        global_intelligence_enabled = False
    
    # Trigger fallback mechanism
    await trigger_fallback(switch, activation)
    
    # Audit log
    audit_entry = {
        "event": "kill_switch_activated",
        "switch_id": switch_id,
        "activation_id": activation.activation_id,
        "scope": switch.scope.value,
        "target_id": switch.target_id,
        "tenant_id": switch.tenant_id,
        "trigger_type": activation.trigger_type.value,
        "triggered_by": activation.triggered_by,
        "reason": activation.reason,
        "timestamp": activation.activated_at.isoformat()
    }
    audit_log.append(audit_entry)
    
    return {
        "activation_id": activation.activation_id,
        "switch_id": switch_id,
        "status": "activated",
        "fallback_triggered": True,
        "message": f"Intelligence disabled for {switch.scope.value}: {switch.target_name}"
    }

@app.post("/kill-switches/{switch_id}/deactivate")
async def deactivate_kill_switch(switch_id: str, deactivated_by: str, reason: str):
    """
    Deactivate kill switch - re-enable intelligence
    """
    if switch_id not in kill_switches:
        raise HTTPException(status_code=404, detail="Kill switch not found")
    
    switch = kill_switches[switch_id]
    
    # Check if switch is currently killed
    if switch.status != KillSwitchStatus.KILLED:
        raise HTTPException(
            status_code=400, 
            detail=f"Kill switch is not active (current status: {switch.status.value})"
        )
    
    # Update switch status
    switch.status = KillSwitchStatus.ACTIVE
    switch.updated_at = datetime.utcnow()
    
    # Handle global kill switch
    global global_intelligence_enabled
    if switch.scope == KillSwitchScope.GLOBAL:
        global_intelligence_enabled = True
    
    # Restore intelligence (remove fallback)
    await restore_intelligence(switch)
    
    # Audit log
    audit_entry = {
        "event": "kill_switch_deactivated",
        "switch_id": switch_id,
        "scope": switch.scope.value,
        "target_id": switch.target_id,
        "tenant_id": switch.tenant_id,
        "deactivated_by": deactivated_by,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    }
    audit_log.append(audit_entry)
    
    return {
        "switch_id": switch_id,
        "status": "deactivated",
        "intelligence_restored": True,
        "message": f"Intelligence restored for {switch.scope.value}: {switch.target_name}"
    }

@app.post("/global-kill-switch")
async def global_kill_switch(
    triggered_by: str, 
    reason: str, 
    trigger_type: KillSwitchTrigger = KillSwitchTrigger.MANUAL
):
    """
    Global kill switch - disable ALL intelligence across platform
    """
    global global_intelligence_enabled
    global_intelligence_enabled = False
    
    # Create global kill switch if not exists
    global_switch_id = "global_kill_switch"
    if global_switch_id not in kill_switches:
        global_switch = KillSwitch(
            switch_id=global_switch_id,
            scope=KillSwitchScope.GLOBAL,
            target_id="platform",
            target_name="Global Platform Intelligence",
            tenant_id="system",
            created_by="system"
        )
        kill_switches[global_switch_id] = global_switch
    
    # Activate global kill switch
    activation = KillSwitchActivation(
        switch_id=global_switch_id,
        trigger_type=trigger_type,
        triggered_by=triggered_by,
        reason=reason
    )
    
    await activate_kill_switch(global_switch_id, activation)
    
    # Disable all active switches
    disabled_switches = []
    for switch_id, switch in kill_switches.items():
        if switch.status == KillSwitchStatus.ACTIVE and switch_id != global_switch_id:
            switch.status = KillSwitchStatus.KILLED
            disabled_switches.append(switch_id)
    
    return {
        "global_kill_activated": True,
        "intelligence_enabled": False,
        "disabled_switches": len(disabled_switches),
        "message": "ALL INTELLIGENCE DISABLED - PLATFORM IN RBA FALLBACK MODE"
    }

async def trigger_fallback(switch: KillSwitch, activation: KillSwitchActivation):
    """
    Trigger fallback mechanism for killed switch
    """
    # Get fallback configuration
    fallback_key = f"{switch.scope.value}_{switch.target_id}"
    fallback_config = fallback_configs.get(fallback_key)
    
    if not fallback_config:
        # Create default RBA fallback
        fallback_config = FallbackConfiguration(
            scope=switch.scope,
            target_id=switch.target_id,
            fallback_type="RBA",
            fallback_config={
                "mode": "deterministic_rules_only",
                "disable_ml_nodes": True,
                "use_baseline_logic": True
            },
            tenant_id=switch.tenant_id
        )
        fallback_configs[fallback_key] = fallback_config
    
    # Simulate triggering fallback (in real implementation, this would call orchestrator)
    print(f"FALLBACK TRIGGERED: {switch.scope.value} {switch.target_id} -> {fallback_config.fallback_type}")

async def restore_intelligence(switch: KillSwitch):
    """
    Restore intelligence after kill switch deactivation
    """
    # Simulate restoring intelligence (in real implementation, this would call orchestrator)
    print(f"INTELLIGENCE RESTORED: {switch.scope.value} {switch.target_id}")

@app.get("/kill-switches")
async def list_kill_switches(
    tenant_id: Optional[str] = None,
    scope: Optional[KillSwitchScope] = None,
    status: Optional[KillSwitchStatus] = None
):
    """
    List kill switches with optional filtering
    """
    switches = list(kill_switches.values())
    
    if tenant_id:
        switches = [s for s in switches if s.tenant_id == tenant_id]
    
    if scope:
        switches = [s for s in switches if s.scope == scope]
    
    if status:
        switches = [s for s in switches if s.status == status]
    
    return {
        "switches": [s.dict() for s in switches],
        "total_count": len(switches),
        "global_intelligence_enabled": global_intelligence_enabled
    }

@app.get("/kill-switches/{switch_id}")
async def get_kill_switch(switch_id: str):
    """
    Get kill switch details including activation history
    """
    if switch_id not in kill_switches:
        raise HTTPException(status_code=404, detail="Kill switch not found")
    
    switch = kill_switches[switch_id]
    switch_activations = activations.get(switch_id, [])
    
    return {
        "switch": switch.dict(),
        "activations": [a.dict() for a in switch_activations],
        "activation_count": len(switch_activations)
    }

@app.get("/intelligence-status")
async def get_intelligence_status():
    """
    Get overall intelligence status across platform
    """
    killed_switches = [s for s in kill_switches.values() if s.status == KillSwitchStatus.KILLED]
    active_switches = [s for s in kill_switches.values() if s.status == KillSwitchStatus.ACTIVE]
    
    # Count by scope
    scope_status = {}
    for scope in KillSwitchScope:
        scope_switches = [s for s in kill_switches.values() if s.scope == scope]
        killed_count = len([s for s in scope_switches if s.status == KillSwitchStatus.KILLED])
        scope_status[scope.value] = {
            "total": len(scope_switches),
            "killed": killed_count,
            "active": len(scope_switches) - killed_count
        }
    
    return {
        "global_intelligence_enabled": global_intelligence_enabled,
        "total_switches": len(kill_switches),
        "killed_switches": len(killed_switches),
        "active_switches": len(active_switches),
        "scope_breakdown": scope_status,
        "platform_status": "INTELLIGENCE_DISABLED" if not global_intelligence_enabled else "INTELLIGENCE_ENABLED"
    }

@app.post("/fallback-config")
async def create_fallback_config(config: FallbackConfiguration):
    """
    Create fallback configuration for specific scope/target
    """
    config_key = f"{config.scope.value}_{config.target_id}"
    fallback_configs[config_key] = config
    
    return {
        "config_id": config.config_id,
        "scope": config.scope.value,
        "target_id": config.target_id,
        "fallback_type": config.fallback_type,
        "message": "Fallback configuration created"
    }

@app.get("/audit-log")
async def get_audit_log(
    limit: int = 100,
    switch_id: Optional[str] = None,
    tenant_id: Optional[str] = None
):
    """
    Get kill switch audit log
    """
    filtered_log = audit_log
    
    if switch_id:
        filtered_log = [entry for entry in filtered_log if entry.get("switch_id") == switch_id]
    
    if tenant_id:
        filtered_log = [entry for entry in filtered_log if entry.get("tenant_id") == tenant_id]
    
    # Sort by timestamp (most recent first) and limit
    filtered_log = sorted(filtered_log, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    return {
        "audit_entries": filtered_log,
        "total_entries": len(filtered_log)
    }

@app.post("/emergency-shutdown")
async def emergency_shutdown(
    triggered_by: str,
    reason: str,
    authorization_code: str  # In real implementation, this would be properly validated
):
    """
    Emergency shutdown of ALL intelligence - highest priority kill switch
    """
    if authorization_code != "EMERGENCY_OVERRIDE":  # Placeholder validation
        raise HTTPException(status_code=403, detail="Invalid authorization code")
    
    global global_intelligence_enabled
    global_intelligence_enabled = False
    
    # Kill ALL switches
    for switch in kill_switches.values():
        switch.status = KillSwitchStatus.KILLED
        switch.updated_at = datetime.utcnow()
    
    # Emergency audit log
    emergency_entry = {
        "event": "emergency_shutdown",
        "triggered_by": triggered_by,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
        "affected_switches": len(kill_switches),
        "severity": "CRITICAL"
    }
    audit_log.append(emergency_entry)
    
    return {
        "emergency_shutdown": True,
        "intelligence_enabled": False,
        "affected_switches": len(kill_switches),
        "message": "EMERGENCY SHUTDOWN ACTIVATED - ALL INTELLIGENCE DISABLED"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Kill Switch",
        "task": "3.2.8 - Kill Switch & RBA Fallback",
        "global_intelligence_enabled": global_intelligence_enabled,
        "total_switches": len(kill_switches)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
