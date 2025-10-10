"""
Task 3.3.28: Training Walkthroughs Service
- UI demos, Assisted explainers, Conversational examples
- Adoption enablement with persona-specific content
- LMS integration ready
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Training Walkthrough Service")
logger = logging.getLogger(__name__)

class WalkthroughType(str, Enum):
    UI_DEMO = "ui_demo"
    ASSISTED_EXPLAINER = "assisted_explainer"
    CONVERSATIONAL_EXAMPLE = "conversational_example"

class PersonaType(str, Enum):
    BUSINESS_USER = "business_user"
    ANALYST = "analyst"
    ADMIN = "admin"
    DEVELOPER = "developer"
    COMPLIANCE_OFFICER = "compliance_officer"

class WalkthroughStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    title: str
    description: str
    action_type: str  # "click", "input", "observe", "validate"
    target_element: Optional[str] = None  # CSS selector or element ID
    expected_result: str
    screenshot_url: Optional[str] = None
    video_url: Optional[str] = None

class TrainingWalkthrough(BaseModel):
    walkthrough_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    walkthrough_type: WalkthroughType
    target_persona: PersonaType
    tenant_id: str
    
    # Content
    steps: List[WalkthroughStep] = Field(default_factory=list)
    estimated_duration_minutes: int = 5
    
    # Prerequisites
    required_permissions: List[str] = Field(default_factory=list)
    prerequisite_walkthroughs: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    completion_rate: float = 0.0  # Percentage of users who complete

class WalkthroughProgress(BaseModel):
    progress_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    tenant_id: str
    walkthrough_id: str
    
    # Progress tracking
    current_step: int = 1
    completed_steps: List[int] = Field(default_factory=list)
    is_completed: bool = False
    completion_percentage: float = 0.0
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    
    # Analytics
    time_spent_seconds: int = 0
    help_requests: int = 0
    errors_encountered: int = 0

# In-memory stores (replace with database in production)
walkthroughs_store: Dict[str, TrainingWalkthrough] = {}
progress_store: Dict[str, WalkthroughProgress] = {}

def _create_default_walkthroughs():
    """Create default walkthroughs for each mode and persona"""
    
    # UI Demo for Business Users
    ui_demo = TrainingWalkthrough(
        title="Building Your First Workflow - UI Mode",
        description="Learn how to create a simple workflow using the visual workflow builder",
        walkthrough_type=WalkthroughType.UI_DEMO,
        target_persona=PersonaType.BUSINESS_USER,
        tenant_id="default",
        estimated_duration_minutes=10,
        steps=[
            WalkthroughStep(
                step_number=1,
                title="Access Workflow Builder",
                description="Navigate to the workflow builder from the main dashboard",
                action_type="click",
                target_element="#workflow-builder-btn",
                expected_result="Workflow builder interface opens"
            ),
            WalkthroughStep(
                step_number=2,
                title="Drag RBA Node",
                description="Drag an RBA node from the palette to the canvas",
                action_type="click",
                target_element=".rba-node-template",
                expected_result="RBA node appears on canvas"
            ),
            WalkthroughStep(
                step_number=3,
                title="Configure Node",
                description="Click on the node to configure its parameters",
                action_type="click",
                target_element=".workflow-node",
                expected_result="Configuration panel opens on the right"
            ),
            WalkthroughStep(
                step_number=4,
                title="Save Workflow",
                description="Click the save button to store your workflow",
                action_type="click",
                target_element="#save-workflow-btn",
                expected_result="Workflow saved successfully message appears"
            )
        ]
    )
    
    # Assisted Explainer for Analysts
    assisted_explainer = TrainingWalkthrough(
        title="Understanding Assisted Mode Suggestions",
        description="Learn how Assisted Mode provides intelligent suggestions and overrides",
        walkthrough_type=WalkthroughType.ASSISTED_EXPLAINER,
        target_persona=PersonaType.ANALYST,
        tenant_id="default",
        estimated_duration_minutes=8,
        steps=[
            WalkthroughStep(
                step_number=1,
                title="Enable Assisted Mode",
                description="Toggle Assisted Mode in the workflow execution panel",
                action_type="click",
                target_element="#assisted-mode-toggle",
                expected_result="Assisted Mode indicator shows as active"
            ),
            WalkthroughStep(
                step_number=2,
                title="Review Suggestions",
                description="Observe how the system provides parameter suggestions",
                action_type="observe",
                expected_result="Suggestion tooltips appear next to configurable parameters"
            ),
            WalkthroughStep(
                step_number=3,
                title="Accept Suggestion",
                description="Click the checkmark to accept a suggested value",
                action_type="click",
                target_element=".suggestion-accept-btn",
                expected_result="Parameter value updates with suggested value"
            ),
            WalkthroughStep(
                step_number=4,
                title="Override with Reason",
                description="Override a suggestion and provide a business reason",
                action_type="input",
                target_element=".override-reason-input",
                expected_result="Override is logged with your reason for audit trail"
            )
        ]
    )
    
    # Conversational Example for All Users
    conversational_example = TrainingWalkthrough(
        title="Natural Language Workflow Creation",
        description="Learn how to describe workflows in plain English using Conversational Mode",
        walkthrough_type=WalkthroughType.CONVERSATIONAL_EXAMPLE,
        target_persona=PersonaType.BUSINESS_USER,
        tenant_id="default",
        estimated_duration_minutes=12,
        steps=[
            WalkthroughStep(
                step_number=1,
                title="Open Conversational Interface",
                description="Click on the chat icon to open Conversational Mode",
                action_type="click",
                target_element="#conversational-mode-btn",
                expected_result="Chat interface opens in sidebar"
            ),
            WalkthroughStep(
                step_number=2,
                title="Describe Your Goal",
                description="Type: 'I want to identify deals at risk of churning'",
                action_type="input",
                target_element="#chat-input",
                expected_result="System acknowledges your request and asks clarifying questions"
            ),
            WalkthroughStep(
                step_number=3,
                title="Provide Context",
                description="Answer follow-up questions about your data and criteria",
                action_type="input",
                target_element="#chat-input",
                expected_result="System builds workflow components based on your responses"
            ),
            WalkthroughStep(
                step_number=4,
                title="Review Generated Workflow",
                description="Examine the automatically generated workflow",
                action_type="observe",
                expected_result="Complete workflow appears on canvas with appropriate nodes"
            ),
            WalkthroughStep(
                step_number=5,
                title="Refine with Conversation",
                description="Ask to modify: 'Can you add email notification for high-risk deals?'",
                action_type="input",
                target_element="#chat-input",
                expected_result="Notification node is added to the workflow"
            )
        ]
    )
    
    # Store default walkthroughs
    walkthroughs_store[ui_demo.walkthrough_id] = ui_demo
    walkthroughs_store[assisted_explainer.walkthrough_id] = assisted_explainer
    walkthroughs_store[conversational_example.walkthrough_id] = conversational_example
    
    logger.info("Created default training walkthroughs")

# Initialize default walkthroughs
_create_default_walkthroughs()

@app.post("/training/walkthroughs", response_model=TrainingWalkthrough)
async def create_walkthrough(walkthrough: TrainingWalkthrough):
    """Create a new training walkthrough"""
    walkthroughs_store[walkthrough.walkthrough_id] = walkthrough
    logger.info(f"Created walkthrough {walkthrough.walkthrough_id} for {walkthrough.target_persona}")
    return walkthrough

@app.get("/training/walkthroughs", response_model=List[TrainingWalkthrough])
async def get_walkthroughs(
    tenant_id: str,
    persona: Optional[PersonaType] = None,
    walkthrough_type: Optional[WalkthroughType] = None
):
    """Get available walkthroughs for a tenant and persona"""
    walkthroughs = [
        w for w in walkthroughs_store.values()
        if w.tenant_id == tenant_id and w.is_active
    ]
    
    if persona:
        walkthroughs = [w for w in walkthroughs if w.target_persona == persona]
    
    if walkthrough_type:
        walkthroughs = [w for w in walkthroughs if w.walkthrough_type == walkthrough_type]
    
    return walkthroughs

@app.get("/training/walkthroughs/{walkthrough_id}", response_model=TrainingWalkthrough)
async def get_walkthrough(walkthrough_id: str):
    """Get a specific walkthrough by ID"""
    if walkthrough_id not in walkthroughs_store:
        raise HTTPException(status_code=404, detail="Walkthrough not found")
    return walkthroughs_store[walkthrough_id]

@app.post("/training/progress/start", response_model=WalkthroughProgress)
async def start_walkthrough(
    user_id: str,
    tenant_id: str,
    walkthrough_id: str
):
    """Start a walkthrough for a user"""
    if walkthrough_id not in walkthroughs_store:
        raise HTTPException(status_code=404, detail="Walkthrough not found")
    
    progress = WalkthroughProgress(
        user_id=user_id,
        tenant_id=tenant_id,
        walkthrough_id=walkthrough_id
    )
    
    progress_store[progress.progress_id] = progress
    logger.info(f"User {user_id} started walkthrough {walkthrough_id}")
    return progress

@app.put("/training/progress/{progress_id}/step", response_model=WalkthroughProgress)
async def update_progress(
    progress_id: str,
    step_number: int,
    completed: bool = True,
    time_spent_seconds: Optional[int] = None,
    errors_encountered: Optional[int] = None
):
    """Update progress for a walkthrough step"""
    if progress_id not in progress_store:
        raise HTTPException(status_code=404, detail="Progress not found")
    
    progress = progress_store[progress_id]
    walkthrough = walkthroughs_store[progress.walkthrough_id]
    
    if completed and step_number not in progress.completed_steps:
        progress.completed_steps.append(step_number)
        progress.current_step = min(step_number + 1, len(walkthrough.steps))
    
    if time_spent_seconds:
        progress.time_spent_seconds += time_spent_seconds
    
    if errors_encountered:
        progress.errors_encountered += errors_encountered
    
    # Calculate completion percentage
    progress.completion_percentage = len(progress.completed_steps) / len(walkthrough.steps) * 100
    
    # Check if walkthrough is completed
    if len(progress.completed_steps) == len(walkthrough.steps):
        progress.is_completed = True
        progress.completed_at = datetime.utcnow()
    
    progress.last_accessed = datetime.utcnow()
    
    logger.info(f"Updated progress for user {progress.user_id}, step {step_number}, completion: {progress.completion_percentage}%")
    return progress

@app.get("/training/progress/{user_id}", response_model=List[WalkthroughProgress])
async def get_user_progress(user_id: str, tenant_id: str):
    """Get all walkthrough progress for a user"""
    progress_list = [
        p for p in progress_store.values()
        if p.user_id == user_id and p.tenant_id == tenant_id
    ]
    return progress_list

@app.get("/training/analytics/completion-rates")
async def get_completion_analytics(tenant_id: str):
    """Get completion rate analytics for walkthroughs"""
    analytics = {}
    
    for walkthrough_id, walkthrough in walkthroughs_store.items():
        if walkthrough.tenant_id != tenant_id:
            continue
        
        # Calculate completion rates
        user_progress = [p for p in progress_store.values() if p.walkthrough_id == walkthrough_id]
        total_started = len(user_progress)
        total_completed = len([p for p in user_progress if p.is_completed])
        
        completion_rate = (total_completed / total_started * 100) if total_started > 0 else 0
        avg_time = sum(p.time_spent_seconds for p in user_progress) / total_started if total_started > 0 else 0
        
        analytics[walkthrough_id] = {
            "title": walkthrough.title,
            "type": walkthrough.walkthrough_type,
            "persona": walkthrough.target_persona,
            "total_started": total_started,
            "total_completed": total_completed,
            "completion_rate": completion_rate,
            "avg_completion_time_seconds": avg_time,
            "avg_errors_per_user": sum(p.errors_encountered for p in user_progress) / total_started if total_started > 0 else 0
        }
    
    return analytics

@app.post("/training/lms-integration/sync")
async def sync_with_lms(tenant_id: str, lms_endpoint: str, api_key: str):
    """Sync walkthrough completion data with external LMS (placeholder for integration)"""
    # This is a placeholder for LMS integration
    # In production, this would sync completion data with external learning management systems
    
    completed_progress = [
        p for p in progress_store.values()
        if p.tenant_id == tenant_id and p.is_completed
    ]
    
    sync_data = {
        "tenant_id": tenant_id,
        "sync_timestamp": datetime.utcnow().isoformat(),
        "completed_walkthroughs": len(completed_progress),
        "users_trained": len(set(p.user_id for p in completed_progress)),
        "lms_endpoint": lms_endpoint  # In production, would make actual API call
    }
    
    logger.info(f"LMS sync completed for tenant {tenant_id}: {sync_data}")
    return {"status": "success", "sync_data": sync_data}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Training Walkthrough Service", "task": "3.3.28"}
