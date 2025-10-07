"""
Cruxx Automation Hooks
Implements automation hooks for Cruxx service: auto-create from summary, close when opp closes
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import asyncio

app = FastAPI(
    title="Cruxx Automation Hooks",
    description="Implements automation hooks for Cruxx service",
    version="1.0.0"
)

class AutomationTrigger(str, Enum):
    SUMMARY_CREATED = "summary_created"
    ACTION_EXTRACTED = "action_extracted"
    OPPORTUNITY_UPDATED = "opportunity_updated"
    OPPORTUNITY_CLOSED = "opportunity_closed"
    OPPORTUNITY_WON = "opportunity_won"
    OPPORTUNITY_LOST = "opportunity_lost"
    SLA_WARNING = "sla_warning"
    SLA_VIOLATED = "sla_violated"
    CRUXX_CREATED = "cruxx_created"
    CRUXX_UPDATED = "cruxx_updated"

class AutomationAction(str, Enum):
    AUTO_CREATE_CRUXX = "auto_create_cruxx"
    AUTO_CLOSE_CRUXX = "auto_close_cruxx"
    AUTO_UPDATE_SLA = "auto_update_sla"
    SEND_SLA_NOTIFICATIONS = "send_sla_notifications"
    CREATE_FOLLOW_UP = "create_follow_up"
    UPDATE_CALENDAR = "update_calendar"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"
    ESCALATE_TO_MANAGER = "escalate_to_manager"

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
    cruxx_id: Optional[str] = None
    opportunity_id: Optional[str] = None
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

class CruxxAction(BaseModel):
    cruxx_id: str
    tenant_id: str
    opportunity_id: str
    title: str
    description: str
    assignee: str
    due_date: datetime
    priority: str
    status: str
    sla_status: str
    source: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage (replace with database in production)
automation_rules_db: Dict[str, AutomationRule] = {}
automation_events_db: List[AutomationEvent] = []
automation_executions_db: List[AutomationExecution] = []
cruxx_actions_db: List[CruxxAction] = []

# Default automation rules
DEFAULT_RULES = [
    {
        "rule_id": "auto_create_cruxx_from_summary",
        "tenant_id": "default",
        "trigger": AutomationTrigger.SUMMARY_CREATED,
        "conditions": {
            "has_action_items": True,
            "min_confidence": 0.7,
            "auto_create_enabled": True
        },
        "actions": [AutomationAction.AUTO_CREATE_CRUXX],
        "priority": 1
    },
    {
        "rule_id": "auto_create_cruxx_from_actions",
        "tenant_id": "default",
        "trigger": AutomationTrigger.ACTION_EXTRACTED,
        "conditions": {
            "has_opportunities": True,
            "min_confidence": 0.8,
            "auto_create_enabled": True
        },
        "actions": [AutomationAction.AUTO_CREATE_CRUXX],
        "priority": 2
    },
    {
        "rule_id": "auto_close_cruxx_on_opp_close",
        "tenant_id": "default",
        "trigger": AutomationTrigger.OPPORTUNITY_CLOSED,
        "conditions": {
            "auto_close_enabled": True,
            "close_reason": ["won", "lost"]
        },
        "actions": [AutomationAction.AUTO_CLOSE_CRUXX],
        "priority": 3
    },
    {
        "rule_id": "sla_warning_notifications",
        "tenant_id": "default",
        "trigger": AutomationTrigger.SLA_WARNING,
        "conditions": {
            "notification_enabled": True,
            "warning_threshold_hours": 24
        },
        "actions": [AutomationAction.SEND_SLA_NOTIFICATIONS],
        "priority": 4
    },
    {
        "rule_id": "sla_violation_escalation",
        "tenant_id": "default",
        "trigger": AutomationTrigger.SLA_VIOLATED,
        "conditions": {
            "escalation_enabled": True,
            "escalate_to_manager": True
        },
        "actions": [AutomationAction.ESCALATE_TO_MANAGER, AutomationAction.NOTIFY_STAKEHOLDERS],
        "priority": 5
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
        elif isinstance(expected_value, list):
            # Handle list conditions
            if actual_value not in expected_value:
                return False
        elif actual_value != expected_value:
            return False
    
    return True

async def execute_automation_action(action: AutomationAction, event_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a specific automation action"""
    
    if action == AutomationAction.AUTO_CREATE_CRUXX:
        # Simulate automatic Cruxx creation
        opportunity_id = event_data.get("opportunity_id")
        actions = event_data.get("actions", [])
        summary = event_data.get("summary")
        
        if opportunity_id and (actions or summary):
            cruxx_actions = []
            
            # Create Cruxx actions from extracted actions
            if actions:
                for action_item in actions:
                    cruxx_action = CruxxAction(
                        cruxx_id=str(uuid.uuid4()),
                        tenant_id=context.get("tenant_id", "default"),
                        opportunity_id=opportunity_id,
                        title=action_item.get("action", "Action Item"),
                        description=action_item.get("description", ""),
                        assignee=action_item.get("assignee", "Unassigned"),
                        due_date=datetime.fromisoformat(action_item.get("due_date", "2024-01-31")),
                        priority=action_item.get("priority", "medium"),
                        status="open",
                        sla_status="met",
                        source="automation"
                    )
                    cruxx_actions.append(cruxx_action)
                    cruxx_actions_db.append(cruxx_action)
            
            # Create Cruxx action from summary if no specific actions
            elif summary:
                cruxx_action = CruxxAction(
                    cruxx_id=str(uuid.uuid4()),
                    tenant_id=context.get("tenant_id", "default"),
                    opportunity_id=opportunity_id,
                    title="Follow up on meeting",
                    description=f"Follow up based on meeting summary: {summary[:100]}...",
                    assignee="Unassigned",
                    due_date=datetime.utcnow().replace(day=datetime.utcnow().day + 7),
                    priority="medium",
                    status="open",
                    sla_status="met",
                    source="automation"
                )
                cruxx_actions.append(cruxx_action)
                cruxx_actions_db.append(cruxx_action)
            
            return {
                "action": "auto_create_cruxx",
                "opportunity_id": opportunity_id,
                "cruxx_actions_created": len(cruxx_actions),
                "cruxx_ids": [action.cruxx_id for action in cruxx_actions],
                "status": "completed"
            }
        else:
            return {
                "action": "auto_create_cruxx",
                "status": "failed",
                "error": "Missing opportunity_id or actions/summary"
            }
    
    elif action == AutomationAction.AUTO_CLOSE_CRUXX:
        # Simulate automatic Cruxx closure
        opportunity_id = event_data.get("opportunity_id")
        close_reason = event_data.get("close_reason")
        
        if opportunity_id:
            # Find and close related Cruxx actions
            closed_count = 0
            for cruxx_action in cruxx_actions_db:
                if cruxx_action.opportunity_id == opportunity_id and cruxx_action.status == "open":
                    cruxx_action.status = "closed"
                    cruxx_action.sla_status = "met" if close_reason == "won" else "missed"
                    closed_count += 1
            
            return {
                "action": "auto_close_cruxx",
                "opportunity_id": opportunity_id,
                "close_reason": close_reason,
                "cruxx_actions_closed": closed_count,
                "status": "completed"
            }
        else:
            return {
                "action": "auto_close_cruxx",
                "status": "failed",
                "error": "Missing opportunity_id"
            }
    
    elif action == AutomationAction.AUTO_UPDATE_SLA:
        # Simulate automatic SLA updates
        cruxx_id = event_data.get("cruxx_id")
        sla_status = event_data.get("sla_status")
        
        if cruxx_id and sla_status:
            # Find and update Cruxx action SLA
            for cruxx_action in cruxx_actions_db:
                if cruxx_action.cruxx_id == cruxx_id:
                    cruxx_action.sla_status = sla_status
                    break
            
            return {
                "action": "auto_update_sla",
                "cruxx_id": cruxx_id,
                "sla_status": sla_status,
                "status": "completed"
            }
        else:
            return {
                "action": "auto_update_sla",
                "status": "failed",
                "error": "Missing cruxx_id or sla_status"
            }
    
    elif action == AutomationAction.SEND_SLA_NOTIFICATIONS:
        # Simulate SLA notification sending
        cruxx_id = event_data.get("cruxx_id")
        assignee = event_data.get("assignee")
        sla_status = event_data.get("sla_status")
        
        if cruxx_id and assignee:
            return {
                "action": "send_sla_notifications",
                "cruxx_id": cruxx_id,
                "assignee": assignee,
                "sla_status": sla_status,
                "notification_sent": True,
                "status": "completed"
            }
        else:
            return {
                "action": "send_sla_notifications",
                "status": "failed",
                "error": "Missing cruxx_id or assignee"
            }
    
    elif action == AutomationAction.CREATE_FOLLOW_UP:
        # Simulate creating follow-up actions
        opportunity_id = event_data.get("opportunity_id")
        follow_up_type = event_data.get("follow_up_type", "general")
        
        if opportunity_id:
            follow_up_action = CruxxAction(
                cruxx_id=str(uuid.uuid4()),
                tenant_id=context.get("tenant_id", "default"),
                opportunity_id=opportunity_id,
                title=f"Follow-up: {follow_up_type}",
                description=f"Automated follow-up action for {follow_up_type}",
                assignee="Unassigned",
                due_date=datetime.utcnow().replace(day=datetime.utcnow().day + 14),
                priority="low",
                status="open",
                sla_status="met",
                source="automation_followup"
            )
            cruxx_actions_db.append(follow_up_action)
            
            return {
                "action": "create_follow_up",
                "opportunity_id": opportunity_id,
                "follow_up_id": follow_up_action.cruxx_id,
                "follow_up_type": follow_up_type,
                "status": "completed"
            }
        else:
            return {
                "action": "create_follow_up",
                "status": "failed",
                "error": "Missing opportunity_id"
            }
    
    elif action == AutomationAction.UPDATE_CALENDAR:
        # Simulate updating calendar with Cruxx reminders
        cruxx_id = event_data.get("cruxx_id")
        due_date = event_data.get("due_date")
        
        if cruxx_id and due_date:
            return {
                "action": "update_calendar",
                "cruxx_id": cruxx_id,
                "due_date": due_date,
                "calendar_updated": True,
                "reminder_created": True,
                "status": "completed"
            }
        else:
            return {
                "action": "update_calendar",
                "status": "failed",
                "error": "Missing cruxx_id or due_date"
            }
    
    elif action == AutomationAction.NOTIFY_STAKEHOLDERS:
        # Simulate notifying stakeholders
        cruxx_id = event_data.get("cruxx_id")
        stakeholders = event_data.get("stakeholders", [])
        
        if cruxx_id and stakeholders:
            return {
                "action": "notify_stakeholders",
                "cruxx_id": cruxx_id,
                "stakeholders_notified": len(stakeholders),
                "notification_sent": True,
                "status": "completed"
            }
        else:
            return {
                "action": "notify_stakeholders",
                "status": "failed",
                "error": "Missing cruxx_id or stakeholders"
            }
    
    elif action == AutomationAction.ESCALATE_TO_MANAGER:
        # Simulate escalation to manager
        cruxx_id = event_data.get("cruxx_id")
        manager_id = event_data.get("manager_id")
        
        if cruxx_id and manager_id:
            return {
                "action": "escalate_to_manager",
                "cruxx_id": cruxx_id,
                "manager_id": manager_id,
                "escalation_sent": True,
                "priority": "high",
                "status": "completed"
            }
        else:
            return {
                "action": "escalate_to_manager",
                "status": "failed",
                "error": "Missing cruxx_id or manager_id"
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

@app.get("/cruxx/{cruxx_id}")
async def get_cruxx_action(cruxx_id: str):
    """Get Cruxx action by ID"""
    cruxx_action = next((c for c in cruxx_actions_db if c.cruxx_id == cruxx_id), None)
    if not cruxx_action:
        raise HTTPException(status_code=404, detail="Cruxx action not found")
    return cruxx_action

@app.get("/cruxx/opportunity/{opportunity_id}")
async def get_cruxx_actions_for_opportunity(opportunity_id: str):
    """Get all Cruxx actions for an opportunity"""
    actions = [c for c in cruxx_actions_db if c.opportunity_id == opportunity_id]
    return {
        "opportunity_id": opportunity_id,
        "actions": actions,
        "count": len(actions)
    }

@app.post("/cruxx", response_model=CruxxAction)
async def create_cruxx_action(cruxx_action: CruxxAction):
    """Create a new Cruxx action"""
    cruxx_action.cruxx_id = str(uuid.uuid4())
    cruxx_action.created_at = datetime.utcnow()
    cruxx_actions_db.append(cruxx_action)
    return cruxx_action

@app.put("/cruxx/{cruxx_id}", response_model=CruxxAction)
async def update_cruxx_action(cruxx_id: str, updates: Dict[str, Any]):
    """Update a Cruxx action"""
    cruxx_action = next((c for c in cruxx_actions_db if c.cruxx_id == cruxx_id), None)
    if not cruxx_action:
        raise HTTPException(status_code=404, detail="Cruxx action not found")
    
    for field, value in updates.items():
        if hasattr(cruxx_action, field):
            setattr(cruxx_action, field, value)
    
    return cruxx_action

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
    
    # Cruxx statistics
    tenant_cruxx_actions = [c for c in cruxx_actions_db if c.tenant_id == tenant_id]
    open_actions = len([c for c in tenant_cruxx_actions if c.status == "open"])
    closed_actions = len([c for c in tenant_cruxx_actions if c.status == "closed"])
    sla_violations = len([c for c in tenant_cruxx_actions if c.sla_status == "missed"])
    
    return {
        "tenant_id": tenant_id,
        "total_events": total_events,
        "total_executions": total_executions,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
        "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
        "cruxx_actions_total": len(tenant_cruxx_actions),
        "cruxx_actions_open": open_actions,
        "cruxx_actions_closed": closed_actions,
        "sla_violations": sla_violations,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "cruxx-automation",
        "timestamp": datetime.utcnow().isoformat(),
        "rules_count": len(automation_rules_db),
        "events_count": len(automation_events_db),
        "executions_count": len(automation_executions_db),
        "cruxx_actions_count": len(cruxx_actions_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
