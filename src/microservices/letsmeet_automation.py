"""
Let's Meet Automation Hooks
Implements automation hooks for Let's Meet service: always transcribe, summarize, action extract
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import asyncio

app = FastAPI(
    title="Let's Meet Automation Hooks",
    description="Implements automation hooks for Let's Meet service",
    version="1.0.0"
)

class AutomationTrigger(str, Enum):
    AUDIO_UPLOADED = "audio_uploaded"
    AUDIO_PROCESSED = "audio_processed"
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    SUMMARY_GENERATED = "summary_generated"
    ACTION_EXTRACTED = "action_extracted"
    MEETING_STARTED = "meeting_started"
    MEETING_ENDED = "meeting_ended"

class AutomationAction(str, Enum):
    AUTO_TRANSCRIBE = "auto_transcribe"
    AUTO_SUMMARIZE = "auto_summarize"
    AUTO_EXTRACT_ACTIONS = "auto_extract_actions"
    GENERATE_EMBEDDINGS = "generate_embeddings"
    CREATE_CRUXX_ACTIONS = "create_cruxx_actions"
    SEND_NOTIFICATIONS = "send_notifications"
    UPDATE_CALENDAR = "update_calendar"

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
    meeting_id: str
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

class TranscriptionResult(BaseModel):
    meeting_id: str
    transcript: str
    segments: List[Dict[str, Any]]
    confidence_score: float
    language: str
    duration_seconds: float

class SummaryResult(BaseModel):
    meeting_id: str
    summary_text: str
    key_points: List[str]
    action_items: List[str]
    decisions: List[str]
    confidence_score: float

class ActionExtractionResult(BaseModel):
    meeting_id: str
    actions: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    follow_ups: List[Dict[str, Any]]
    confidence_score: float

# In-memory storage (replace with database in production)
automation_rules_db: Dict[str, AutomationRule] = {}
automation_events_db: List[AutomationEvent] = []
automation_executions_db: List[AutomationExecution] = []
transcription_results_db: List[TranscriptionResult] = []
summary_results_db: List[SummaryResult] = []
action_extraction_results_db: List[ActionExtractionResult] = []

# Default automation rules
DEFAULT_RULES = [
    {
        "rule_id": "auto_transcribe_audio",
        "tenant_id": "default",
        "trigger": AutomationTrigger.AUDIO_UPLOADED,
        "conditions": {
            "auto_transcribe": True,
            "audio_format": ["wav", "mp3", "mp4", "ogg"]
        },
        "actions": [AutomationAction.AUTO_TRANSCRIBE],
        "priority": 1
    },
    {
        "rule_id": "auto_summarize_transcript",
        "tenant_id": "default",
        "trigger": AutomationTrigger.TRANSCRIPTION_COMPLETED,
        "conditions": {
            "auto_summarize": True,
            "min_duration_seconds": 60
        },
        "actions": [AutomationAction.AUTO_SUMMARIZE],
        "priority": 2
    },
    {
        "rule_id": "auto_extract_actions",
        "tenant_id": "default",
        "trigger": AutomationTrigger.SUMMARY_GENERATED,
        "conditions": {
            "auto_extract_actions": True,
            "min_confidence": 0.7
        },
        "actions": [AutomationAction.AUTO_EXTRACT_ACTIONS],
        "priority": 3
    },
    {
        "rule_id": "create_cruxx_from_actions",
        "tenant_id": "default",
        "trigger": AutomationTrigger.ACTION_EXTRACTED,
        "conditions": {
            "auto_create_cruxx": True,
            "has_opportunities": True
        },
        "actions": [AutomationAction.CREATE_CRUXX_ACTIONS],
        "priority": 4
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
            # Handle list conditions (e.g., audio format)
            if actual_value not in expected_value:
                return False
        elif actual_value != expected_value:
            return False
    
    return True

async def execute_automation_action(action: AutomationAction, event_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a specific automation action"""
    
    if action == AutomationAction.AUTO_TRANSCRIBE:
        # Simulate automatic transcription
        meeting_id = event_data.get("meeting_id")
        audio_url = event_data.get("audio_url")
        
        if meeting_id and audio_url:
            # In a real implementation, this would call a transcription service
            transcript_result = TranscriptionResult(
                meeting_id=meeting_id,
                transcript="This is a simulated transcript of the meeting. The team discussed project updates and next steps.",
                segments=[
                    {"speaker": "John", "text": "Let's start with project updates", "start_time": 0, "end_time": 3},
                    {"speaker": "Jane", "text": "We've completed the first phase", "start_time": 3, "end_time": 7}
                ],
                confidence_score=0.95,
                language="en-US",
                duration_seconds=300
            )
            transcription_results_db.append(transcript_result)
            
            return {
                "action": "auto_transcribe",
                "meeting_id": meeting_id,
                "transcript_id": str(uuid.uuid4()),
                "confidence_score": transcript_result.confidence_score,
                "duration_seconds": transcript_result.duration_seconds,
                "status": "completed"
            }
        else:
            return {
                "action": "auto_transcribe",
                "status": "failed",
                "error": "Missing meeting_id or audio_url"
            }
    
    elif action == AutomationAction.AUTO_SUMMARIZE:
        # Simulate automatic summarization
        meeting_id = event_data.get("meeting_id")
        transcript = event_data.get("transcript")
        
        if meeting_id and transcript:
            # In a real implementation, this would call an AI summarization service
            summary_result = SummaryResult(
                meeting_id=meeting_id,
                summary_text="The team discussed project updates and next steps. Key decisions were made regarding timeline and resource allocation.",
                key_points=[
                    "Project phase 1 completed successfully",
                    "Timeline adjusted for phase 2",
                    "Additional resources allocated"
                ],
                action_items=[
                    "Update project documentation",
                    "Schedule follow-up meeting",
                    "Review resource requirements"
                ],
                decisions=[
                    "Approved timeline extension",
                    "Approved additional budget"
                ],
                confidence_score=0.88
            )
            summary_results_db.append(summary_result)
            
            return {
                "action": "auto_summarize",
                "meeting_id": meeting_id,
                "summary_id": str(uuid.uuid4()),
                "confidence_score": summary_result.confidence_score,
                "key_points_count": len(summary_result.key_points),
                "action_items_count": len(summary_result.action_items),
                "status": "completed"
            }
        else:
            return {
                "action": "auto_summarize",
                "status": "failed",
                "error": "Missing meeting_id or transcript"
            }
    
    elif action == AutomationAction.AUTO_EXTRACT_ACTIONS:
        # Simulate automatic action extraction
        meeting_id = event_data.get("meeting_id")
        summary = event_data.get("summary")
        
        if meeting_id and summary:
            # In a real implementation, this would call an AI action extraction service
            action_result = ActionExtractionResult(
                meeting_id=meeting_id,
                actions=[
                    {
                        "action": "Update project documentation",
                        "assignee": "John Doe",
                        "due_date": "2024-01-20",
                        "priority": "high",
                        "confidence": 0.9
                    },
                    {
                        "action": "Schedule follow-up meeting",
                        "assignee": "Jane Smith",
                        "due_date": "2024-01-18",
                        "priority": "medium",
                        "confidence": 0.85
                    }
                ],
                opportunities=[
                    {
                        "opportunity": "Expand to new market",
                        "probability": 0.7,
                        "value": 50000,
                        "confidence": 0.8
                    }
                ],
                follow_ups=[
                    {
                        "follow_up": "Review budget allocation",
                        "due_date": "2024-01-25",
                        "confidence": 0.75
                    }
                ],
                confidence_score=0.82
            )
            action_extraction_results_db.append(action_result)
            
            return {
                "action": "auto_extract_actions",
                "meeting_id": meeting_id,
                "actions_count": len(action_result.actions),
                "opportunities_count": len(action_result.opportunities),
                "follow_ups_count": len(action_result.follow_ups),
                "confidence_score": action_result.confidence_score,
                "status": "completed"
            }
        else:
            return {
                "action": "auto_extract_actions",
                "status": "failed",
                "error": "Missing meeting_id or summary"
            }
    
    elif action == AutomationAction.GENERATE_EMBEDDINGS:
        # Simulate embedding generation
        meeting_id = event_data.get("meeting_id")
        transcript = event_data.get("transcript")
        
        if meeting_id and transcript:
            # In a real implementation, this would generate embeddings using pgvector
            return {
                "action": "generate_embeddings",
                "meeting_id": meeting_id,
                "embeddings_count": len(transcript.split()) // 10,  # Simulate chunking
                "vector_dimensions": 1536,
                "status": "completed"
            }
        else:
            return {
                "action": "generate_embeddings",
                "status": "failed",
                "error": "Missing meeting_id or transcript"
            }
    
    elif action == AutomationAction.CREATE_CRUXX_ACTIONS:
        # Simulate creating Cruxx actions
        meeting_id = event_data.get("meeting_id")
        actions = event_data.get("actions", [])
        
        if meeting_id and actions:
            cruxx_actions = []
            for action_item in actions:
                cruxx_action = {
                    "cruxx_id": str(uuid.uuid4()),
                    "meeting_id": meeting_id,
                    "title": action_item.get("action"),
                    "assignee": action_item.get("assignee"),
                    "due_date": action_item.get("due_date"),
                    "priority": action_item.get("priority"),
                    "source": "letsmeet_automation"
                }
                cruxx_actions.append(cruxx_action)
            
            return {
                "action": "create_cruxx_actions",
                "meeting_id": meeting_id,
                "cruxx_actions_created": len(cruxx_actions),
                "cruxx_ids": [action["cruxx_id"] for action in cruxx_actions],
                "status": "completed"
            }
        else:
            return {
                "action": "create_cruxx_actions",
                "status": "failed",
                "error": "Missing meeting_id or actions"
            }
    
    elif action == AutomationAction.SEND_NOTIFICATIONS:
        # Simulate sending notifications
        meeting_id = event_data.get("meeting_id")
        participants = event_data.get("participants", [])
        
        if meeting_id and participants:
            return {
                "action": "send_notifications",
                "meeting_id": meeting_id,
                "notifications_sent": len(participants),
                "status": "completed"
            }
        else:
            return {
                "action": "send_notifications",
                "status": "failed",
                "error": "Missing meeting_id or participants"
            }
    
    elif action == AutomationAction.UPDATE_CALENDAR:
        # Simulate updating calendar
        meeting_id = event_data.get("meeting_id")
        event_id = event_data.get("event_id")
        
        if meeting_id and event_id:
            return {
                "action": "update_calendar",
                "meeting_id": meeting_id,
                "event_id": event_id,
                "calendar_updated": True,
                "status": "completed"
            }
        else:
            return {
                "action": "update_calendar",
                "status": "failed",
                "error": "Missing meeting_id or event_id"
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

@app.get("/transcriptions/{meeting_id}")
async def get_transcription(meeting_id: str):
    """Get transcription result for a meeting"""
    transcription = next((t for t in transcription_results_db if t.meeting_id == meeting_id), None)
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return transcription

@app.get("/summaries/{meeting_id}")
async def get_summary(meeting_id: str):
    """Get summary result for a meeting"""
    summary = next((s for s in summary_results_db if s.meeting_id == meeting_id), None)
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    return summary

@app.get("/actions/{meeting_id}")
async def get_extracted_actions(meeting_id: str):
    """Get extracted actions for a meeting"""
    actions = next((a for a in action_extraction_results_db if a.meeting_id == meeting_id), None)
    if not actions:
        raise HTTPException(status_code=404, detail="Actions not found")
    return actions

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
    
    # Processing statistics
    transcriptions_count = len([t for t in transcription_results_db if any(e.meeting_id == t.meeting_id for e in tenant_events)])
    summaries_count = len([s for s in summary_results_db if any(e.meeting_id == s.meeting_id for e in tenant_events)])
    actions_count = len([a for a in action_extraction_results_db if any(e.meeting_id == a.meeting_id for e in tenant_events)])
    
    return {
        "tenant_id": tenant_id,
        "total_events": total_events,
        "total_executions": total_executions,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
        "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
        "transcriptions_processed": transcriptions_count,
        "summaries_generated": summaries_count,
        "actions_extracted": actions_count,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "letsmeet-automation",
        "timestamp": datetime.utcnow().isoformat(),
        "rules_count": len(automation_rules_db),
        "events_count": len(automation_events_db),
        "executions_count": len(automation_executions_db),
        "transcriptions_count": len(transcription_results_db),
        "summaries_count": len(summary_results_db),
        "actions_count": len(action_extraction_results_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
