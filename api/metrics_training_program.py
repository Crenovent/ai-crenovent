"""
Task 4.4.38: Create training program to teach interpreting success metrics
- Training modules for different personas
- Interactive tutorials
- Certification tracking
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Success Metrics Training Program")
logger = logging.getLogger(__name__)

class PersonaType(str, Enum):
    CRO = "cro"
    CFO = "cfo"
    COMPLIANCE = "compliance"
    REVOPS = "revops"
    DATA_SCIENTIST = "data_scientist"
    BUSINESS_ANALYST = "business_analyst"

class TrainingModuleType(str, Enum):
    INTRODUCTION = "introduction"
    TRUST_SCORES = "trust_scores"
    ROI_CALCULATION = "roi_calculation"
    GOVERNANCE_METRICS = "governance_metrics"
    FORECAST_METRICS = "forecast_metrics"
    DRIFT_BIAS = "drift_bias"
    SLA_MONITORING = "sla_monitoring"
    ADVANCED_ANALYTICS = "advanced_analytics"

class TrainingModule(BaseModel):
    module_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    module_type: TrainingModuleType
    title: str
    description: str
    
    # Content
    learning_objectives: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    example_scenarios: List[str] = Field(default_factory=list)
    
    # Duration
    estimated_duration_minutes: int
    
    # Target personas
    target_personas: List[PersonaType] = Field(default_factory=list)
    
    # Prerequisites
    prerequisite_modules: List[str] = Field(default_factory=list)
    
    # Assessment
    has_quiz: bool = True
    passing_score: int = 80
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrainingProgress(BaseModel):
    progress_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    persona: PersonaType
    
    # Completed modules
    completed_modules: List[str] = Field(default_factory=list)
    
    # Module scores
    module_scores: Dict[str, int] = Field(default_factory=dict)
    
    # Overall progress
    completion_percentage: float = 0.0
    
    # Certification
    is_certified: bool = False
    certification_date: Optional[datetime] = None
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

class TrainingCertification(BaseModel):
    certification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    persona: PersonaType
    
    # Certification details
    certification_name: str
    modules_completed: List[str] = Field(default_factory=list)
    overall_score: float
    
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

# In-memory storage
training_modules_store: Dict[str, TrainingModule] = {}
training_progress_store: Dict[str, TrainingProgress] = {}
certifications_store: Dict[str, TrainingCertification] = {}

class MetricsTrainingEngine:
    def __init__(self):
        self._initialize_training_modules()
    
    def _initialize_training_modules(self):
        """Initialize default training modules"""
        
        modules = [
            {
                "type": TrainingModuleType.INTRODUCTION,
                "title": "Introduction to RBIA Success Metrics",
                "description": "Learn the fundamentals of RBIA success metrics and why they matter",
                "objectives": [
                    "Understand the purpose of success metrics",
                    "Learn the different categories of metrics",
                    "Know where metrics are collected from"
                ],
                "concepts": [
                    "Metrics pipeline",
                    "Trust scores",
                    "KPI categories"
                ],
                "scenarios": [
                    "Reviewing monthly trust score reports",
                    "Identifying declining metrics"
                ],
                "duration": 30,
                "personas": [p for p in PersonaType]
            },
            {
                "type": TrainingModuleType.TRUST_SCORES,
                "title": "Understanding Trust Scores",
                "description": "Deep dive into trust score components and interpretation",
                "objectives": [
                    "Understand trust score calculation",
                    "Identify trust score components",
                    "Interpret trust score trends"
                ],
                "concepts": [
                    "Accuracy component",
                    "Explainability component",
                    "Drift health component",
                    "Composite scoring"
                ],
                "scenarios": [
                    "Investigating low trust scores",
                    "Comparing trust across tenants"
                ],
                "duration": 45,
                "personas": [PersonaType.CRO, PersonaType.COMPLIANCE, PersonaType.DATA_SCIENTIST]
            },
            {
                "type": TrainingModuleType.ROI_CALCULATION,
                "title": "ROI and Cost Efficiency Metrics",
                "description": "Learn how to calculate and interpret ROI metrics",
                "objectives": [
                    "Calculate ROI from automation",
                    "Understand cost savings metrics",
                    "Interpret payback period"
                ],
                "concepts": [
                    "Cost per prediction",
                    "Manual vs automated costs",
                    "Net benefit calculation",
                    "ROI percentage"
                ],
                "scenarios": [
                    "Building CFO-ready ROI reports",
                    "Justifying AI investment"
                ],
                "duration": 50,
                "personas": [PersonaType.CFO, PersonaType.BUSINESS_ANALYST]
            },
            {
                "type": TrainingModuleType.GOVERNANCE_METRICS,
                "title": "Governance and Compliance Metrics",
                "description": "Master governance metrics for regulatory compliance",
                "objectives": [
                    "Track governance compliance rates",
                    "Monitor audit trail completeness",
                    "Interpret regulatory acceptance metrics"
                ],
                "concepts": [
                    "Governance compliance score",
                    "Evidence pack generation",
                    "Audit pass rates",
                    "Regulator access logs"
                ],
                "scenarios": [
                    "Preparing for regulatory audit",
                    "Demonstrating compliance"
                ],
                "duration": 60,
                "personas": [PersonaType.COMPLIANCE, PersonaType.CRO]
            },
            {
                "type": TrainingModuleType.FORECAST_METRICS,
                "title": "Forecast Improvement Metrics",
                "description": "Learn to measure and improve forecast accuracy",
                "objectives": [
                    "Measure forecast variance reduction",
                    "Calculate forecast error rates",
                    "Use what-if simulators"
                ],
                "concepts": [
                    "Forecast accuracy",
                    "Variance reduction",
                    "Error rate tracking",
                    "Scenario simulation"
                ],
                "scenarios": [
                    "Improving sales forecasts",
                    "Reducing forecast errors"
                ],
                "duration": 45,
                "personas": [PersonaType.REVOPS, PersonaType.DATA_SCIENTIST, PersonaType.BUSINESS_ANALYST]
            },
            {
                "type": TrainingModuleType.DRIFT_BIAS,
                "title": "Drift and Bias Monitoring",
                "description": "Detect and respond to drift and bias in AI systems",
                "objectives": [
                    "Identify data drift",
                    "Detect prediction drift",
                    "Monitor fairness metrics"
                ],
                "concepts": [
                    "Data drift indicators",
                    "Prediction drift detection",
                    "Bias metrics",
                    "Fairness thresholds"
                ],
                "scenarios": [
                    "Responding to drift alerts",
                    "Investigating bias reports"
                ],
                "duration": 55,
                "personas": [PersonaType.DATA_SCIENTIST, PersonaType.CRO, PersonaType.COMPLIANCE]
            },
            {
                "type": TrainingModuleType.SLA_MONITORING,
                "title": "SLA Compliance and Performance",
                "description": "Monitor and maintain SLA compliance",
                "objectives": [
                    "Track SLA compliance rates",
                    "Monitor latency and uptime",
                    "Respond to SLA violations"
                ],
                "concepts": [
                    "SLA targets",
                    "Latency monitoring",
                    "Availability tracking",
                    "Cost predictability"
                ],
                "scenarios": [
                    "Investigating SLA breaches",
                    "Optimizing performance"
                ],
                "duration": 40,
                "personas": [PersonaType.REVOPS, PersonaType.DATA_SCIENTIST]
            },
            {
                "type": TrainingModuleType.ADVANCED_ANALYTICS,
                "title": "Advanced Metrics Analytics",
                "description": "Master advanced analytics and benchmarking",
                "objectives": [
                    "Perform cross-tenant benchmarking",
                    "Analyze industry comparisons",
                    "Build custom reports"
                ],
                "concepts": [
                    "Tenant benchmarking",
                    "Industry standards",
                    "Custom KPI weighting",
                    "Correlation analysis"
                ],
                "scenarios": [
                    "Building executive dashboards",
                    "Competitive analysis"
                ],
                "duration": 70,
                "personas": [PersonaType.DATA_SCIENTIST, PersonaType.BUSINESS_ANALYST]
            }
        ]
        
        for module_data in modules:
            module = TrainingModule(
                module_type=module_data["type"],
                title=module_data["title"],
                description=module_data["description"],
                learning_objectives=module_data["objectives"],
                key_concepts=module_data["concepts"],
                example_scenarios=module_data["scenarios"],
                estimated_duration_minutes=module_data["duration"],
                target_personas=module_data["personas"]
            )
            training_modules_store[module.module_id] = module
    
    def get_modules_for_persona(self, persona: PersonaType) -> List[TrainingModule]:
        """Get training modules for a specific persona"""
        
        return [
            module for module in training_modules_store.values()
            if persona in module.target_personas
        ]
    
    def create_training_progress(self, user_id: str, persona: PersonaType) -> TrainingProgress:
        """Create training progress tracker for a user"""
        
        progress = TrainingProgress(
            user_id=user_id,
            persona=persona
        )
        
        training_progress_store[progress.progress_id] = progress
        return progress
    
    def complete_module(
        self,
        progress_id: str,
        module_id: str,
        score: int
    ) -> TrainingProgress:
        """Mark a module as completed"""
        
        progress = training_progress_store.get(progress_id)
        if not progress:
            raise ValueError("Progress not found")
        
        module = training_modules_store.get(module_id)
        if not module:
            raise ValueError("Module not found")
        
        # Add to completed modules
        if module_id not in progress.completed_modules:
            progress.completed_modules.append(module_id)
        
        # Record score
        progress.module_scores[module_id] = score
        
        # Update completion percentage
        persona_modules = self.get_modules_for_persona(progress.persona)
        total_modules = len(persona_modules)
        completed = len(progress.completed_modules)
        progress.completion_percentage = (completed / total_modules * 100) if total_modules > 0 else 0
        
        # Update timestamp
        progress.last_activity = datetime.utcnow()
        
        # Check for certification eligibility
        if progress.completion_percentage == 100:
            avg_score = sum(progress.module_scores.values()) / len(progress.module_scores)
            if avg_score >= 80:
                self._issue_certification(progress)
        
        return progress
    
    def _issue_certification(self, progress: TrainingProgress):
        """Issue certification to user"""
        
        avg_score = sum(progress.module_scores.values()) / len(progress.module_scores)
        
        certification = TrainingCertification(
            user_id=progress.user_id,
            persona=progress.persona,
            certification_name=f"RBIA Success Metrics - {progress.persona.value.upper()}",
            modules_completed=progress.completed_modules,
            overall_score=avg_score
        )
        
        certifications_store[certification.certification_id] = certification
        
        progress.is_certified = True
        progress.certification_date = datetime.utcnow()
        
        logger.info(f"✅ Issued certification to user {progress.user_id}")

# Global training engine
training_engine = MetricsTrainingEngine()

@app.get("/training/modules", response_model=List[TrainingModule])
async def get_all_training_modules():
    """Get all training modules"""
    
    return list(training_modules_store.values())

@app.get("/training/modules/persona/{persona}")
async def get_modules_for_persona(persona: PersonaType):
    """Get training modules for a specific persona"""
    
    modules = training_engine.get_modules_for_persona(persona)
    
    return {
        "persona": persona.value,
        "total_modules": len(modules),
        "total_duration_minutes": sum(m.estimated_duration_minutes for m in modules),
        "modules": modules
    }

@app.post("/training/progress/create", response_model=TrainingProgress)
async def create_training_progress(user_id: str, persona: PersonaType):
    """Create training progress for a user"""
    
    progress = training_engine.create_training_progress(user_id, persona)
    
    logger.info(f"✅ Created training progress for user {user_id}")
    return progress

@app.put("/training/progress/{progress_id}/complete-module")
async def complete_module(
    progress_id: str,
    module_id: str,
    score: int = Field(ge=0, le=100)
):
    """Mark a module as completed with score"""
    
    try:
        progress = training_engine.complete_module(progress_id, module_id, score)
        
        logger.info(f"✅ Module {module_id} completed with score {score}")
        return progress
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/training/progress/{progress_id}")
async def get_training_progress(progress_id: str):
    """Get training progress"""
    
    progress = training_progress_store.get(progress_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Progress not found")
    
    return progress

@app.get("/training/progress/user/{user_id}")
async def get_user_progress(user_id: str):
    """Get all training progress for a user"""
    
    user_progress = [
        p for p in training_progress_store.values()
        if p.user_id == user_id
    ]
    
    return {
        "user_id": user_id,
        "total_tracks": len(user_progress),
        "progress": user_progress
    }

@app.get("/training/certifications/{user_id}")
async def get_user_certifications(user_id: str):
    """Get certifications for a user"""
    
    user_certs = [
        c for c in certifications_store.values()
        if c.user_id == user_id
    ]
    
    return {
        "user_id": user_id,
        "total_certifications": len(user_certs),
        "certifications": user_certs
    }

@app.get("/training/summary")
async def get_training_summary():
    """Get training program summary"""
    
    total_modules = len(training_modules_store)
    total_learners = len(set(p.user_id for p in training_progress_store.values()))
    total_certifications = len(certifications_store)
    
    return {
        "total_modules": total_modules,
        "total_learners": total_learners,
        "total_certifications": total_certifications,
        "service_status": "active"
    }

