"""
Task 3.3.43: Compliance Training Badge Service
- Users must complete training before enabling Assisted/Conversational modes
- LMS integration with IAM gating and logged records
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Compliance Training Badge Service")
logger = logging.getLogger(__name__)

class UXMode(str, Enum):
    UI_LED = "ui_led"
    ASSISTED = "assisted"
    CONVERSATIONAL = "conversational"

class TrainingModule(str, Enum):
    BASIC_COMPLIANCE = "basic_compliance"
    DATA_PRIVACY = "data_privacy"
    AI_ETHICS = "ai_ethics"
    BIAS_AWARENESS = "bias_awareness"
    ASSISTED_MODE_SAFETY = "assisted_mode_safety"
    CONVERSATIONAL_SAFETY = "conversational_safety"
    REGULATORY_OVERVIEW = "regulatory_overview"

class BadgeStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    REVOKED = "revoked"

class ComplianceTraining(BaseModel):
    training_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Training details
    module: TrainingModule
    title: str
    description: str
    required_for_modes: List[UXMode]
    
    # Content
    training_content_url: Optional[str] = None
    quiz_questions: List[Dict[str, Any]] = Field(default_factory=list)
    passing_score: int = 80  # Percentage
    
    # Validity
    validity_days: int = 365  # Training valid for 1 year
    
    # Requirements
    prerequisites: List[TrainingModule] = Field(default_factory=list)
    mandatory: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class UserTrainingRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    training_id: str
    
    # Progress
    status: BadgeStatus = BadgeStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Results
    quiz_score: Optional[int] = None
    quiz_attempts: int = 0
    max_attempts: int = 3
    
    # Compliance
    certificate_id: Optional[str] = None
    logged_completion: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class ComplianceBadge(BaseModel):
    badge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Badge details
    badge_name: str
    enabled_modes: List[UXMode]
    required_trainings: List[str] = Field(default_factory=list)  # training_ids
    
    # Status
    is_valid: bool = True
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    
    # Audit
    issued_by: str = "system"
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revoked_reason: Optional[str] = None

class ModeAccessCheck(BaseModel):
    user_id: str
    tenant_id: str
    requested_mode: UXMode
    
    # Result
    access_granted: bool
    missing_trainings: List[str] = Field(default_factory=list)
    expired_badges: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
compliance_trainings_store: Dict[str, ComplianceTraining] = {}
user_training_records_store: Dict[str, UserTrainingRecord] = {}
compliance_badges_store: Dict[str, ComplianceBadge] = {}

def _create_default_trainings():
    """Create default compliance training modules"""
    
    # Basic compliance training
    basic_training = ComplianceTraining(
        tenant_id="default",
        module=TrainingModule.BASIC_COMPLIANCE,
        title="Basic Compliance and Ethics",
        description="Fundamental compliance principles and ethical guidelines",
        required_for_modes=[UXMode.ASSISTED, UXMode.CONVERSATIONAL],
        quiz_questions=[
            {
                "question": "What is the primary purpose of compliance training?",
                "options": ["Legal protection", "User safety", "Both", "Neither"],
                "correct_answer": "Both"
            }
        ]
    )
    
    # Assisted mode safety training
    assisted_training = ComplianceTraining(
        tenant_id="default",
        module=TrainingModule.ASSISTED_MODE_SAFETY,
        title="Assisted Mode Safety Guidelines",
        description="Safe usage of AI-assisted decision making",
        required_for_modes=[UXMode.ASSISTED],
        prerequisites=[TrainingModule.BASIC_COMPLIANCE],
        quiz_questions=[
            {
                "question": "When should you override an AI suggestion?",
                "options": ["Never", "When you have better information", "Always", "Only with approval"],
                "correct_answer": "When you have better information"
            }
        ]
    )
    
    # Conversational safety training
    conversational_training = ComplianceTraining(
        tenant_id="default",
        module=TrainingModule.CONVERSATIONAL_SAFETY,
        title="Conversational AI Safety and Best Practices",
        description="Safe and effective use of conversational AI interfaces",
        required_for_modes=[UXMode.CONVERSATIONAL],
        prerequisites=[TrainingModule.BASIC_COMPLIANCE, TrainingModule.AI_ETHICS],
        quiz_questions=[
            {
                "question": "What should you do if the AI provides potentially biased results?",
                "options": ["Ignore it", "Report it", "Use it anyway", "Modify the data"],
                "correct_answer": "Report it"
            }
        ]
    )
    
    # AI Ethics training
    ethics_training = ComplianceTraining(
        tenant_id="default",
        module=TrainingModule.AI_ETHICS,
        title="AI Ethics and Responsible Use",
        description="Ethical considerations when using AI systems",
        required_for_modes=[UXMode.ASSISTED, UXMode.CONVERSATIONAL],
        quiz_questions=[
            {
                "question": "What is algorithmic bias?",
                "options": ["A programming error", "Unfair treatment of certain groups", "Slow performance", "High accuracy"],
                "correct_answer": "Unfair treatment of certain groups"
            }
        ]
    )
    
    # Store trainings
    compliance_trainings_store[basic_training.training_id] = basic_training
    compliance_trainings_store[assisted_training.training_id] = assisted_training
    compliance_trainings_store[conversational_training.training_id] = conversational_training
    compliance_trainings_store[ethics_training.training_id] = ethics_training
    
    logger.info("Created default compliance training modules")

# Initialize default trainings
_create_default_trainings()

@app.post("/training/modules", response_model=ComplianceTraining)
async def create_training_module(training: ComplianceTraining):
    """Create a new compliance training module"""
    compliance_trainings_store[training.training_id] = training
    logger.info(f"Created training module: {training.title}")
    return training

@app.get("/training/modules", response_model=List[ComplianceTraining])
async def get_training_modules(
    tenant_id: str,
    required_for_mode: Optional[UXMode] = None
):
    """Get available training modules"""
    modules = [t for t in compliance_trainings_store.values() if t.tenant_id == tenant_id and t.is_active]
    
    if required_for_mode:
        modules = [t for t in modules if required_for_mode in t.required_for_modes]
    
    return modules

@app.post("/training/enroll")
async def enroll_user_in_training(
    tenant_id: str,
    user_id: str,
    training_id: str
):
    """Enroll user in a training module"""
    
    if training_id not in compliance_trainings_store:
        raise HTTPException(status_code=404, detail="Training module not found")
    
    training = compliance_trainings_store[training_id]
    
    # Check if user already has a record for this training
    existing_record = None
    for record in user_training_records_store.values():
        if (record.tenant_id == tenant_id and 
            record.user_id == user_id and 
            record.training_id == training_id):
            existing_record = record
            break
    
    if existing_record and existing_record.status == BadgeStatus.COMPLETED:
        return {"message": "User already completed this training", "record_id": existing_record.record_id}
    
    # Create new training record
    record = UserTrainingRecord(
        tenant_id=tenant_id,
        user_id=user_id,
        training_id=training_id,
        status=BadgeStatus.IN_PROGRESS,
        started_at=datetime.utcnow()
    )
    
    user_training_records_store[record.record_id] = record
    logger.info(f"Enrolled user {user_id} in training {training.title}")
    
    return {"message": "User enrolled successfully", "record_id": record.record_id}

@app.post("/training/complete-quiz")
async def complete_training_quiz(
    tenant_id: str,
    user_id: str,
    training_id: str,
    quiz_answers: Dict[str, str]
):
    """Complete training quiz and update badge status"""
    
    # Find user's training record
    training_record = None
    for record in user_training_records_store.values():
        if (record.tenant_id == tenant_id and 
            record.user_id == user_id and 
            record.training_id == training_id):
            training_record = record
            break
    
    if not training_record:
        raise HTTPException(status_code=404, detail="Training record not found")
    
    if training_record.quiz_attempts >= training_record.max_attempts:
        raise HTTPException(status_code=400, detail="Maximum quiz attempts exceeded")
    
    # Get training module
    training = compliance_trainings_store[training_id]
    
    # Calculate quiz score
    correct_answers = 0
    total_questions = len(training.quiz_questions)
    
    for question_data in training.quiz_questions:
        question_id = str(hash(question_data["question"]))  # Simple question ID
        user_answer = quiz_answers.get(question_id, "")
        if user_answer == question_data["correct_answer"]:
            correct_answers += 1
    
    score = int((correct_answers / total_questions) * 100) if total_questions > 0 else 0
    
    # Update training record
    training_record.quiz_attempts += 1
    training_record.quiz_score = score
    training_record.last_updated = datetime.utcnow()
    
    # Check if passed
    if score >= training.passing_score:
        training_record.status = BadgeStatus.COMPLETED
        training_record.completed_at = datetime.utcnow()
        training_record.expires_at = datetime.utcnow() + timedelta(days=training.validity_days)
        training_record.certificate_id = str(uuid.uuid4())
        training_record.logged_completion = True
        
        # Log completion for compliance
        logger.info(f"Training completed: User {user_id}, Training {training.title}, Score {score}%")
        
        # Check if user now qualifies for badges
        await _check_and_issue_badges(tenant_id, user_id)
        
        return {
            "status": "passed",
            "score": score,
            "certificate_id": training_record.certificate_id,
            "expires_at": training_record.expires_at
        }
    else:
        return {
            "status": "failed",
            "score": score,
            "attempts_remaining": training_record.max_attempts - training_record.quiz_attempts,
            "passing_score": training.passing_score
        }

async def _check_and_issue_badges(tenant_id: str, user_id: str):
    """Check if user qualifies for compliance badges and issue them"""
    
    # Get user's completed trainings
    completed_trainings = [
        record for record in user_training_records_store.values()
        if (record.tenant_id == tenant_id and 
            record.user_id == user_id and 
            record.status == BadgeStatus.COMPLETED and
            record.expires_at and record.expires_at > datetime.utcnow())
    ]
    
    completed_training_ids = [record.training_id for record in completed_trainings]
    
    # Check for Assisted Mode badge
    assisted_requirements = []
    conversational_requirements = []
    
    for training in compliance_trainings_store.values():
        if training.tenant_id == tenant_id:
            if UXMode.ASSISTED in training.required_for_modes:
                assisted_requirements.append(training.training_id)
            if UXMode.CONVERSATIONAL in training.required_for_modes:
                conversational_requirements.append(training.training_id)
    
    # Issue Assisted Mode badge if qualified
    if all(req in completed_training_ids for req in assisted_requirements):
        await _issue_badge(tenant_id, user_id, "Assisted Mode Compliance", [UXMode.ASSISTED], assisted_requirements)
    
    # Issue Conversational Mode badge if qualified
    if all(req in completed_training_ids for req in conversational_requirements):
        await _issue_badge(tenant_id, user_id, "Conversational Mode Compliance", [UXMode.CONVERSATIONAL], conversational_requirements)

async def _issue_badge(tenant_id: str, user_id: str, badge_name: str, enabled_modes: List[UXMode], required_trainings: List[str]):
    """Issue a compliance badge to a user"""
    
    # Check if badge already exists
    existing_badge = None
    for badge in compliance_badges_store.values():
        if (badge.tenant_id == tenant_id and 
            badge.user_id == user_id and 
            badge.badge_name == badge_name and
            badge.is_valid):
            existing_badge = badge
            break
    
    if existing_badge:
        logger.info(f"Badge {badge_name} already exists for user {user_id}")
        return
    
    # Create new badge
    badge = ComplianceBadge(
        tenant_id=tenant_id,
        user_id=user_id,
        badge_name=badge_name,
        enabled_modes=enabled_modes,
        required_trainings=required_trainings,
        expires_at=datetime.utcnow() + timedelta(days=365)  # 1 year validity
    )
    
    compliance_badges_store[badge.badge_id] = badge
    logger.info(f"Issued badge '{badge_name}' to user {user_id}")

@app.get("/training/user-progress/{user_id}")
async def get_user_training_progress(user_id: str, tenant_id: str):
    """Get user's training progress and badge status"""
    
    # Get user's training records
    user_records = [
        record for record in user_training_records_store.values()
        if record.tenant_id == tenant_id and record.user_id == user_id
    ]
    
    # Get user's badges
    user_badges = [
        badge for badge in compliance_badges_store.values()
        if badge.tenant_id == tenant_id and badge.user_id == user_id and badge.is_valid
    ]
    
    # Calculate progress
    progress = {
        "user_id": user_id,
        "total_trainings": len(user_records),
        "completed_trainings": len([r for r in user_records if r.status == BadgeStatus.COMPLETED]),
        "active_badges": len(user_badges),
        "training_records": user_records,
        "badges": user_badges
    }
    
    return progress

@app.post("/access/check-mode-access", response_model=ModeAccessCheck)
async def check_mode_access(
    user_id: str,
    tenant_id: str,
    requested_mode: UXMode
):
    """Check if user has required training to access a UX mode"""
    
    # Get required trainings for this mode
    required_trainings = [
        training for training in compliance_trainings_store.values()
        if (training.tenant_id == tenant_id and 
            requested_mode in training.required_for_modes and
            training.mandatory)
    ]
    
    # Get user's valid badges
    user_badges = [
        badge for badge in compliance_badges_store.values()
        if (badge.tenant_id == tenant_id and 
            badge.user_id == user_id and 
            badge.is_valid and
            badge.expires_at > datetime.utcnow() and
            requested_mode in badge.enabled_modes)
    ]
    
    # Check if user has completed required trainings
    missing_trainings = []
    expired_badges = []
    
    if not user_badges:
        # No valid badges, check individual training completion
        for training in required_trainings:
            user_record = None
            for record in user_training_records_store.values():
                if (record.tenant_id == tenant_id and 
                    record.user_id == user_id and 
                    record.training_id == training.training_id):
                    user_record = record
                    break
            
            if not user_record or user_record.status != BadgeStatus.COMPLETED:
                missing_trainings.append(training.title)
            elif user_record.expires_at and user_record.expires_at <= datetime.utcnow():
                expired_badges.append(training.title)
    
    access_granted = len(missing_trainings) == 0 and len(expired_badges) == 0
    
    result = ModeAccessCheck(
        user_id=user_id,
        tenant_id=tenant_id,
        requested_mode=requested_mode,
        access_granted=access_granted,
        missing_trainings=missing_trainings,
        expired_badges=expired_badges
    )
    
    # Log access check for compliance
    logger.info(f"Mode access check: User {user_id}, Mode {requested_mode.value}, Granted: {access_granted}")
    
    return result

@app.get("/badges/{user_id}")
async def get_user_badges(user_id: str, tenant_id: str):
    """Get all badges for a user"""
    
    user_badges = [
        badge for badge in compliance_badges_store.values()
        if badge.tenant_id == tenant_id and badge.user_id == user_id
    ]
    
    return {"user_id": user_id, "badges": user_badges}

@app.post("/badges/{badge_id}/revoke")
async def revoke_badge(
    badge_id: str,
    revoked_by: str,
    reason: str
):
    """Revoke a compliance badge"""
    
    if badge_id not in compliance_badges_store:
        raise HTTPException(status_code=404, detail="Badge not found")
    
    badge = compliance_badges_store[badge_id]
    badge.is_valid = False
    badge.revoked_at = datetime.utcnow()
    badge.revoked_by = revoked_by
    badge.revoked_reason = reason
    
    logger.info(f"Revoked badge {badge.badge_name} for user {badge.user_id}: {reason}")
    
    return {"status": "revoked", "badge_id": badge_id}

@app.get("/analytics/compliance-status")
async def get_compliance_analytics(tenant_id: str):
    """Get compliance training analytics for the tenant"""
    
    tenant_records = [r for r in user_training_records_store.values() if r.tenant_id == tenant_id]
    tenant_badges = [b for b in compliance_badges_store.values() if b.tenant_id == tenant_id and b.is_valid]
    
    analytics = {
        "total_users_enrolled": len(set(r.user_id for r in tenant_records)),
        "total_completions": len([r for r in tenant_records if r.status == BadgeStatus.COMPLETED]),
        "total_active_badges": len(tenant_badges),
        "completion_rate": 0,
        "by_training_module": {},
        "by_mode_access": {}
    }
    
    if tenant_records:
        analytics["completion_rate"] = (analytics["total_completions"] / len(tenant_records)) * 100
    
    # Analyze by training module
    for module in TrainingModule:
        module_records = [r for r in tenant_records if any(
            t.module == module and t.training_id == r.training_id 
            for t in compliance_trainings_store.values()
        )]
        completed_count = len([r for r in module_records if r.status == BadgeStatus.COMPLETED])
        
        analytics["by_training_module"][module.value] = {
            "enrolled": len(module_records),
            "completed": completed_count,
            "completion_rate": (completed_count / len(module_records) * 100) if module_records else 0
        }
    
    # Analyze by mode access
    for mode in UXMode:
        mode_badges = [b for b in tenant_badges if mode in b.enabled_modes]
        analytics["by_mode_access"][mode.value] = len(mode_badges)
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Compliance Training Badge Service", "task": "3.3.43"}
