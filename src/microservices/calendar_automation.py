"""
Calendar Automation Hooks
Implements automation hooks for Calendar service: auto-join, sync overlays, close loops
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import asyncio

app = FastAPI(
    title="Calendar Automation Hooks",
    description="Implements automation hooks for Calendar service",
    version="1.0.0"
)

class AutomationTrigger(str, Enum):
    EVENT_CREATED = "event_created"
    EVENT_UPDATED = "event_updated"
    EVENT_DELETED = "event_deleted"
    MEETING_STARTED = "meeting_started"
    MEETING_ENDED = "meeting_ended"
    OVERLAY_CREATED = "overlay_created"
    OVERLAY_UPDATED = "overlay_updated"
    CRUXX_LINKED = "cruxx_linked"
    CRUXX_COMPLETED = "cruxx_completed"

class AutomationAction(str, Enum):
    AUTO_JOIN_MEETING = "auto_join_meeting"
    SYNC_OVERLAYS = "sync_overlays"
    CLOSE_LOOPS = "close_loops"
    UPDATE_FREE_BUSY = "update_free_busy"
    SEND_REMINDERS = "send_reminders"
    CREATE_FOLLOW_UP = "create_follow_up"

class AutomationRule(BaseModel):
    rule_id: str
    tenant_id: str
    trigger: AutomationTrigger
    conditions: Dict[str, Any]
    actions: List[AutomationAction]
    priority: int = Field(default=1, ge=1, le=10)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AutomationEvent(BaseModel):
    event_id: str
    tenant_id: str
    user_id: str
    trigger: AutomationTrigger
    event_data: Dict[str, Any]
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AutomationExecution(BaseModel):
    execution_id: str
    event_id: str
    rule_id: str
    actions: List[AutomationAction]
    status: str  # pending, running, completed, failed
    results: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

# In-memory storage (replace with database in production)
automation_rules_db: Dict[str, AutomationRule] = {}
automation_events_db: List[AutomationEvent] = []
automation_executions_db: List[AutomationExecution] = []

# Default automation rules
DEFAULT_RULES = [
    {
        "rule_id": "auto_join_meetings",
        "tenant_id": "default",
        "trigger": AutomationTrigger.MEETING_STARTED,
        "conditions": {
            "event_type": "meeting",
            "auto_join_enabled": True,
            "user_preference": "auto_join"
        },
        "actions": [AutomationAction.AUTO_JOIN_MEETING],
        "priority": 1
    },
    {
        "rule_id": "sync_cruxx_overlays",
        "tenant_id": "default",
        "trigger": AutomationTrigger.CRUXX_LINKED,
        "conditions": {
            "cruxx_status": "open",
            "overlay_enabled": True
        },
        "actions": [AutomationAction.SYNC_OVERLAYS],
        "priority": 2
    },
    {
        "rule_id": "close_completed_cruxx",
        "tenant_id": "default",
        "trigger": AutomationTrigger.CRUXX_COMPLETED,
        "conditions": {
            "auto_close_enabled": True
        },
        "actions": [AutomationAction.CLOSE_LOOPS],
        "priority": 3
    }
]

# Initialize default rules
for rule_data in DEFAULT_RULES:
    rule = AutomationRule(**rule_data)
    automation_rules_db[rule.rule_id] = rule

def evaluate_conditions(conditions: Dict[str, Any], event_data: Dict[str, Any]) -> bool:
    """Evaluate automation rule conditions against event data"""
    for field, expected_value in conditions.items():
        if field not in event_data:
            return False
        
        actual_value = event_data[field]
        
        if isinstance(expected_value, dict):
            # Handle complex conditions
            for op, threshold in expected_value.items():
                if op == "gte" and actual_value < threshold:
                    return False
                elif op == "gt" and actual_value <= threshold:
                    return False
                elif op == "lte" and actual_value > threshold:
                    return False
                elif op == "lt" and actual_value >= threshold:
                    return False
                elif op == "eq" and actual_value != threshold:
                    return False
                elif op == "ne" and actual_value == threshold:
                    return False
        elif actual_value != expected_value:
            return False
    
    return True

async def execute_automation_action(action: AutomationAction, event_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a specific automation action"""
    
    if action == AutomationAction.AUTO_JOIN_MEETING:
        # Simulate auto-joining a meeting
        meeting_url = event_data.get("meeting_url")
        if meeting_url:
            # In a real implementation, this would integrate with meeting platform APIs
            return {
                "action": "auto_join_meeting",
                "meeting_url": meeting_url,
                "status": "joined",
                "joined_at": datetime.utcnow().isoformat()
            }
        else:
            return {
                "action": "auto_join_meeting",
                "status": "failed",
                "error": "No meeting URL provided"
            }
    
    elif action == AutomationAction.SYNC_OVERLAYS:
        # Simulate syncing overlays
        event_id = event_data.get("event_id")
        cruxx_id = event_data.get("cruxx_id")
        
        if event_id and cruxx_id:
            # In a real implementation, this would update calendar overlays
            return {
                "action": "sync_overlays",
                "event_id": event_id,
                "cruxx_id": cruxx_id,
                "overlays_created": 1,
                "status": "synced"
            }
        else:
            return {
                "action": "sync_overlays",
                "status": "failed",
                "error": "Missing event_id or cruxx_id"
            }
    
    elif action == AutomationAction.CLOSE_LOOPS:
        # Simulate closing automation loops
        cruxx_id = event_data.get("cruxx_id")
        
        if cruxx_id:
            # In a real implementation, this would close related automation loops
            return {
                "action": "close_loops",
                "cruxx_id": cruxx_id,
                "loops_closed": 1,
                "status": "closed"
            }
        else:
            return {
                "action": "close_loops",
                "status": "failed",
                "error": "Missing cruxx_id"
            }
    
    elif action == AutomationAction.UPDATE_FREE_BUSY:
        # Simulate updating free/busy status
        user_id = event_data.get("user_id")
        
        if user_id:
            return {
                "action": "update_free_busy",
                "user_id": user_id,
                "cache_updated": True,
                "status": "updated"
            }
        else:
            return {
                "action": "update_free_busy",
                "status": "failed",
                "error": "Missing user_id"
            }
    
    elif action == AutomationAction.SEND_REMINDERS:
        # Simulate sending reminders
        event_id = event_data.get("event_id")
        
        if event_id:
            return {
                "action": "send_reminders",
                "event_id": event_id,
                "reminders_sent": 1,
                "status": "sent"
            }
        else:
            return {
                "action": "send_reminders",
                "status": "failed",
                "error": "Missing event_id"
            }
    
    elif action == AutomationAction.CREATE_FOLLOW_UP:
        # Simulate creating follow-up events
        original_event_id = event_data.get("event_id")
        
        if original_event_id:
            follow_up_id = str(uuid.uuid4())
            return {
                "action": "create_follow_up",
                "original_event_id": original_event_id,
                "follow_up_id": follow_up_id,
                "status": "created"
            }
        else:
            return {
                "action": "create_follow_up",
                "status": "failed",
                "error": "Missing event_id"
            }
    
    else:
        return {
            "action": str(action),
            "status": "not_implemented",
            "error": f"Action {action} not implemented"
        }

@app.post("/automation/trigger")
async def trigger_automation(event: AutomationEvent, background_tasks: BackgroundTasks):
    """Trigger automation based on an event"""
    
    event.event_id = str(uuid.uuid4())
    event.timestamp = datetime.utcnow()
    automation_events_db.append(event)
    
    # Find applicable rules
    applicable_rules = []
    for rule in automation_rules_db.values():
        if (rule.is_active and 
            rule.trigger == event.trigger and
            (rule.tenant_id == event.tenant_id or rule.tenant_id == "default")):
            
            if evaluate_conditions(rule.conditions, event.event_data):
                applicable_rules.append(rule)
    
    # Sort by priority
    applicable_rules.sort(key=lambda r: r.priority)
    
    # Execute automation for each applicable rule
    for rule in applicable_rules:
        execution = AutomationExecution(
            execution_id=str(uuid.uuid4()),
            event_id=event.event_id,
            rule_id=rule.rule_id,
            actions=rule.actions,
            status="pending",
            started_at=datetime.utcnow()
        )
        automation_executions_db.append(execution)
        
        # Execute automation in background
        background_tasks.add_task(execute_automation_rule, execution, event.event_data, event.context)
    
    return {
        "message": "Automation triggered successfully",
        "event_id": event.event_id,
        "rules_applied": len(applicable_rules),
        "executions_started": len(applicable_rules)
    }

async def execute_automation_rule(execution: AutomationExecution, event_data: Dict[str, Any], context: Dict[str, Any]):
    """Execute automation rule actions"""
    
    execution.status = "running"
    results = {}
    
    try:
        for action in execution.actions:
            result = await execute_automation_action(action, event_data, context)
            results[str(action)] = result
        
        execution.results = results
        execution.status = "completed"
        execution.completed_at = datetime.utcnow()
        
    except Exception as e:
        execution.status = "failed"
        execution.error_message = str(e)
        execution.completed_at = datetime.utcnow()

@app.post("/automation/rules", response_model=AutomationRule)
async def create_automation_rule(rule: AutomationRule):
    """Create a new automation rule"""
    rule.rule_id = str(uuid.uuid4())
    rule.created_at = datetime.utcnow()
    rule.updated_at = datetime.utcnow()
    automation_rules_db[rule.rule_id] = rule
    return rule

@app.get("/automation/rules", response_model=List[AutomationRule])
async def list_automation_rules(tenant_id: str):
    """List automation rules for a tenant"""
    return [rule for rule in automation_rules_db.values() 
            if rule.tenant_id == tenant_id or rule.tenant_id == "default"]

@app.put("/automation/rules/{rule_id}", response_model=AutomationRule)
async def update_automation_rule(rule_id: str, rule_update: Dict[str, Any]):
    """Update an automation rule"""
    if rule_id not in automation_rules_db:
        raise HTTPException(status_code=404, detail="Automation rule not found")
    
    rule = automation_rules_db[rule_id]
    for field, value in rule_update.items():
        if hasattr(rule, field):
            setattr(rule, field, value)
    
    rule.updated_at = datetime.utcnow()
    return rule

@app.delete("/automation/rules/{rule_id}")
async def delete_automation_rule(rule_id: str):
    """Delete an automation rule"""
    if rule_id not in automation_rules_db:
        raise HTTPException(status_code=404, detail="Automation rule not found")
    
    del automation_rules_db[rule_id]
    return {"message": "Automation rule deleted successfully"}

@app.get("/automation/executions", response_model=List[AutomationExecution])
async def list_automation_executions(
    tenant_id: str,
    status: Optional[str] = None,
    limit: int = 100
):
    """List automation executions for a tenant"""
    executions = []
    
    for execution in automation_executions_db:
        # Find the associated event
        event = next((e for e in automation_events_db if e.event_id == execution.event_id), None)
        if event and event.tenant_id == tenant_id:
            if status is None or execution.status == status:
                executions.append(execution)
    
    return executions[-limit:]  # Return most recent executions

@app.get("/automation/events", response_model=List[AutomationEvent])
async def list_automation_events(
    tenant_id: str,
    trigger: Optional[AutomationTrigger] = None,
    limit: int = 100
):
    """List automation events for a tenant"""
    events = [event for event in automation_events_db if event.tenant_id == tenant_id]
    
    if trigger:
        events = [event for event in events if event.trigger == trigger]
    
    return events[-limit:]  # Return most recent events

@app.get("/automation/stats")
async def get_automation_stats(tenant_id: str):
    """Get automation statistics for a tenant"""
    
    # Filter events and executions by tenant
    tenant_events = [e for e in automation_events_db if e.tenant_id == tenant_id]
    tenant_executions = []
    
    for execution in automation_executions_db:
        event = next((e for e in tenant_events if e.event_id == execution.event_id), None)
        if event:
            tenant_executions.append(execution)
    
    # Calculate statistics
    total_events = len(tenant_events)
    total_executions = len(tenant_executions)
    successful_executions = len([e for e in tenant_executions if e.status == "completed"])
    failed_executions = len([e for e in tenant_executions if e.status == "failed"])
    
    # Group by trigger
    trigger_stats = {}
    for event in tenant_events:
        trigger = event.trigger
        if trigger not in trigger_stats:
            trigger_stats[trigger] = 0
        trigger_stats[trigger] += 1
    
    # Group by action
    action_stats = {}
    for execution in tenant_executions:
        for action in execution.actions:
            if action not in action_stats:
                action_stats[action] = 0
            action_stats[action] += 1
    
    return {
        "tenant_id": tenant_id,
        "total_events": total_events,
        "total_executions": total_executions,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
        "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
        "trigger_stats": trigger_stats,
        "action_stats": action_stats,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "calendar-automation",
        "timestamp": datetime.utcnow().isoformat(),
        "rules_count": len(automation_rules_db),
        "events_count": len(automation_events_db),
        "executions_count": len(automation_executions_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
