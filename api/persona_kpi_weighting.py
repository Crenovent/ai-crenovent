"""
Task 4.4.37: Add per-persona KPI weighting system
- Different personas care about different metrics
- Customizable KPI weights per role
- Persona-specific dashboard priorities
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Persona KPI Weighting System")
logger = logging.getLogger(__name__)

class PersonaType(str, Enum):
    CRO = "cro"  # Chief Risk Officer
    CFO = "cfo"  # Chief Financial Officer
    COMPLIANCE = "compliance"
    REVOPS = "revops"  # Revenue Operations
    REGULATOR = "regulator"
    DATA_SCIENTIST = "data_scientist"
    BUSINESS_ANALYST = "business_analyst"
    EXECUTIVE = "executive"

class KPIMetric(str, Enum):
    ACCURACY = "accuracy"
    OVERRIDE_REDUCTION = "override_reduction"
    FORECAST_IMPROVEMENT = "forecast_improvement"
    ANOMALY_DETECTION = "anomaly_detection"
    COST_EFFICIENCY = "cost_efficiency"
    TRUST_SCORE = "trust_score"
    SLA_COMPLIANCE = "sla_compliance"
    ADOPTION_RATE = "adoption_rate"
    ROI = "roi"
    FAIRNESS = "fairness"
    EXPLAINABILITY = "explainability"
    DRIFT_STABILITY = "drift_stability"
    GOVERNANCE_COMPLIANCE = "governance_compliance"
    SECURITY = "security"
    TIME_TO_VALUE = "time_to_value"

class KPIWeight(BaseModel):
    metric: KPIMetric
    weight: float = Field(ge=0.0, le=1.0, description="Weight between 0.0 and 1.0")
    priority: int = Field(ge=1, le=10, description="Display priority (1=highest)")
    display_name: str
    description: str

class PersonaKPIProfile(BaseModel):
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona: PersonaType
    
    # KPI weights (sum should = 1.0)
    kpi_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Priority ordering for dashboard
    kpi_priorities: Dict[str, int] = Field(default_factory=dict)
    
    # Top KPIs for this persona
    top_kpis: List[str] = Field(default_factory=list)
    
    # Dashboard configuration
    dashboard_title: str
    key_questions: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PersonaKPIWeighting(BaseModel):
    weighting_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona: PersonaType
    weights: List[KPIWeight] = Field(default_factory=list)
    total_weight: float = Field(default=1.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
persona_profiles_store: Dict[str, PersonaKPIProfile] = {}
persona_weightings_store: Dict[str, PersonaKPIWeighting] = {}

class PersonaKPIEngine:
    def __init__(self):
        # Define default KPI weights for each persona
        self.default_profiles = {
            PersonaType.CRO: {
                "weights": {
                    KPIMetric.TRUST_SCORE.value: 0.25,
                    KPIMetric.FAIRNESS.value: 0.20,
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 0.20,
                    KPIMetric.DRIFT_STABILITY.value: 0.15,
                    KPIMetric.ACCURACY.value: 0.10,
                    KPIMetric.EXPLAINABILITY.value: 0.10
                },
                "priorities": {
                    KPIMetric.TRUST_SCORE.value: 1,
                    KPIMetric.FAIRNESS.value: 2,
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 3,
                    KPIMetric.DRIFT_STABILITY.value: 4,
                    KPIMetric.EXPLAINABILITY.value: 5
                },
                "title": "Risk Officer Dashboard",
                "questions": [
                    "Is our AI trustworthy and fair?",
                    "Are we compliant with regulations?",
                    "What are our drift and bias levels?"
                ]
            },
            PersonaType.CFO: {
                "weights": {
                    KPIMetric.ROI.value: 0.30,
                    KPIMetric.COST_EFFICIENCY.value: 0.25,
                    KPIMetric.FORECAST_IMPROVEMENT.value: 0.20,
                    KPIMetric.TIME_TO_VALUE.value: 0.15,
                    KPIMetric.ADOPTION_RATE.value: 0.10
                },
                "priorities": {
                    KPIMetric.ROI.value: 1,
                    KPIMetric.COST_EFFICIENCY.value: 2,
                    KPIMetric.FORECAST_IMPROVEMENT.value: 3,
                    KPIMetric.TIME_TO_VALUE.value: 4
                },
                "title": "Financial Impact Dashboard",
                "questions": [
                    "What's the ROI of our AI investment?",
                    "How much are we saving on operations?",
                    "What's the payback period?"
                ]
            },
            PersonaType.COMPLIANCE: {
                "weights": {
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 0.30,
                    KPIMetric.EXPLAINABILITY.value: 0.25,
                    KPIMetric.FAIRNESS.value: 0.20,
                    KPIMetric.SECURITY.value: 0.15,
                    KPIMetric.TRUST_SCORE.value: 0.10
                },
                "priorities": {
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 1,
                    KPIMetric.EXPLAINABILITY.value: 2,
                    KPIMetric.FAIRNESS.value: 3,
                    KPIMetric.SECURITY.value: 4
                },
                "title": "Compliance & Governance Dashboard",
                "questions": [
                    "Are all decisions auditable?",
                    "Do we pass regulatory audits?",
                    "Is our AI explainable to regulators?"
                ]
            },
            PersonaType.REVOPS: {
                "weights": {
                    KPIMetric.FORECAST_IMPROVEMENT.value: 0.30,
                    KPIMetric.ACCURACY.value: 0.25,
                    KPIMetric.ADOPTION_RATE.value: 0.20,
                    KPIMetric.OVERRIDE_REDUCTION.value: 0.15,
                    KPIMetric.SLA_COMPLIANCE.value: 0.10
                },
                "priorities": {
                    KPIMetric.FORECAST_IMPROVEMENT.value: 1,
                    KPIMetric.ACCURACY.value: 2,
                    KPIMetric.ADOPTION_RATE.value: 3,
                    KPIMetric.OVERRIDE_REDUCTION.value: 4
                },
                "title": "Revenue Operations Dashboard",
                "questions": [
                    "How accurate are our forecasts?",
                    "Are teams adopting the AI?",
                    "What's our prediction accuracy?"
                ]
            },
            PersonaType.REGULATOR: {
                "weights": {
                    KPIMetric.EXPLAINABILITY.value: 0.30,
                    KPIMetric.FAIRNESS.value: 0.25,
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 0.25,
                    KPIMetric.SECURITY.value: 0.15,
                    KPIMetric.DRIFT_STABILITY.value: 0.05
                },
                "priorities": {
                    KPIMetric.EXPLAINABILITY.value: 1,
                    KPIMetric.FAIRNESS.value: 2,
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 3,
                    KPIMetric.SECURITY.value: 4
                },
                "title": "Regulatory Oversight Dashboard",
                "questions": [
                    "Can all decisions be explained?",
                    "Is the AI free from bias?",
                    "Are governance policies enforced?"
                ]
            },
            PersonaType.DATA_SCIENTIST: {
                "weights": {
                    KPIMetric.ACCURACY.value: 0.30,
                    KPIMetric.DRIFT_STABILITY.value: 0.25,
                    KPIMetric.ANOMALY_DETECTION.value: 0.20,
                    KPIMetric.FAIRNESS.value: 0.15,
                    KPIMetric.EXPLAINABILITY.value: 0.10
                },
                "priorities": {
                    KPIMetric.ACCURACY.value: 1,
                    KPIMetric.DRIFT_STABILITY.value: 2,
                    KPIMetric.ANOMALY_DETECTION.value: 3,
                    KPIMetric.FAIRNESS.value: 4
                },
                "title": "Model Performance Dashboard",
                "questions": [
                    "How accurate are our models?",
                    "Is there any drift or bias?",
                    "What's our anomaly detection rate?"
                ]
            },
            PersonaType.BUSINESS_ANALYST: {
                "weights": {
                    KPIMetric.ADOPTION_RATE.value: 0.25,
                    KPIMetric.ACCURACY.value: 0.20,
                    KPIMetric.EXPLAINABILITY.value: 0.20,
                    KPIMetric.ROI.value: 0.20,
                    KPIMetric.TIME_TO_VALUE.value: 0.15
                },
                "priorities": {
                    KPIMetric.ADOPTION_RATE.value: 1,
                    KPIMetric.ACCURACY.value: 2,
                    KPIMetric.EXPLAINABILITY.value: 3,
                    KPIMetric.ROI.value: 4
                },
                "title": "Business Impact Dashboard",
                "questions": [
                    "How widely is the AI being used?",
                    "What's the business value?",
                    "Can users understand the predictions?"
                ]
            },
            PersonaType.EXECUTIVE: {
                "weights": {
                    KPIMetric.ROI.value: 0.25,
                    KPIMetric.TRUST_SCORE.value: 0.20,
                    KPIMetric.ADOPTION_RATE.value: 0.20,
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 0.15,
                    KPIMetric.COST_EFFICIENCY.value: 0.10,
                    KPIMetric.SLA_COMPLIANCE.value: 0.10
                },
                "priorities": {
                    KPIMetric.ROI.value: 1,
                    KPIMetric.TRUST_SCORE.value: 2,
                    KPIMetric.ADOPTION_RATE.value: 3,
                    KPIMetric.GOVERNANCE_COMPLIANCE.value: 4
                },
                "title": "Executive Summary Dashboard",
                "questions": [
                    "What's the overall business impact?",
                    "Is our AI trustworthy and compliant?",
                    "Are we achieving strategic goals?"
                ]
            }
        }
    
    def create_persona_profile(self, persona: PersonaType) -> PersonaKPIProfile:
        """Create KPI profile for a persona"""
        
        default = self.default_profiles.get(persona)
        if not default:
            raise ValueError(f"No default profile for persona: {persona}")
        
        top_kpis = sorted(
            default["weights"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        profile = PersonaKPIProfile(
            persona=persona,
            kpi_weights=default["weights"],
            kpi_priorities=default["priorities"],
            top_kpis=[kpi for kpi, _ in top_kpis],
            dashboard_title=default["title"],
            key_questions=default["questions"]
        )
        
        persona_profiles_store[profile.profile_id] = profile
        return profile
    
    def create_persona_weighting(self, persona: PersonaType) -> PersonaKPIWeighting:
        """Create detailed KPI weighting for a persona"""
        
        default = self.default_profiles.get(persona)
        if not default:
            raise ValueError(f"No default profile for persona: {persona}")
        
        weights = []
        for metric_str, weight in default["weights"].items():
            metric = KPIMetric(metric_str)
            priority = default["priorities"].get(metric_str, 10)
            
            kpi_weight = KPIWeight(
                metric=metric,
                weight=weight,
                priority=priority,
                display_name=metric.value.replace("_", " ").title(),
                description=self._get_metric_description(metric)
            )
            weights.append(kpi_weight)
        
        # Sort by weight (descending)
        weights.sort(key=lambda x: x.weight, reverse=True)
        
        weighting = PersonaKPIWeighting(
            persona=persona,
            weights=weights,
            total_weight=sum(w.weight for w in weights)
        )
        
        persona_weightings_store[weighting.weighting_id] = weighting
        return weighting
    
    def _get_metric_description(self, metric: KPIMetric) -> str:
        """Get description for a KPI metric"""
        
        descriptions = {
            KPIMetric.ACCURACY: "Model prediction accuracy",
            KPIMetric.OVERRIDE_REDUCTION: "Reduction in manual overrides",
            KPIMetric.FORECAST_IMPROVEMENT: "Improvement in forecast accuracy",
            KPIMetric.ANOMALY_DETECTION: "Precision and recall for anomaly detection",
            KPIMetric.COST_EFFICIENCY: "Cost savings from automation",
            KPIMetric.TRUST_SCORE: "Composite trust index",
            KPIMetric.SLA_COMPLIANCE: "Service level agreement compliance",
            KPIMetric.ADOPTION_RATE: "User and tenant adoption rate",
            KPIMetric.ROI: "Return on investment",
            KPIMetric.FAIRNESS: "Bias and fairness metrics",
            KPIMetric.EXPLAINABILITY: "Explainability and interpretability",
            KPIMetric.DRIFT_STABILITY: "Data and prediction drift stability",
            KPIMetric.GOVERNANCE_COMPLIANCE: "Governance and regulatory compliance",
            KPIMetric.SECURITY: "Security and data protection",
            KPIMetric.TIME_TO_VALUE: "Time to realize business value"
        }
        
        return descriptions.get(metric, "No description available")

# Global persona KPI engine
persona_engine = PersonaKPIEngine()

@app.post("/persona-kpi/profile/create", response_model=PersonaKPIProfile)
async def create_persona_profile(persona: PersonaType):
    """Create KPI profile for a persona"""
    
    profile = persona_engine.create_persona_profile(persona)
    
    logger.info(f"✅ Created KPI profile for persona: {persona.value}")
    return profile

@app.post("/persona-kpi/weighting/create", response_model=PersonaKPIWeighting)
async def create_persona_weighting(persona: PersonaType):
    """Create KPI weighting for a persona"""
    
    weighting = persona_engine.create_persona_weighting(persona)
    
    logger.info(f"✅ Created KPI weighting for persona: {persona.value}")
    return weighting

@app.get("/persona-kpi/profile/{profile_id}")
async def get_persona_profile(profile_id: str):
    """Get specific persona KPI profile"""
    
    profile = persona_profiles_store.get(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile

@app.get("/persona-kpi/profile/by-persona/{persona}")
async def get_profile_by_persona(persona: PersonaType):
    """Get KPI profile by persona type"""
    
    # Find existing profile or create new one
    for profile in persona_profiles_store.values():
        if profile.persona == persona:
            return profile
    
    # Create new profile if not exists
    return persona_engine.create_persona_profile(persona)

@app.put("/persona-kpi/profile/{profile_id}/weights")
async def update_kpi_weights(
    profile_id: str,
    weights: Dict[str, float]
):
    """Update KPI weights for a profile"""
    
    profile = persona_profiles_store.get(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Weights must sum to 1.0 (current sum: {total_weight})"
        )
    
    profile.kpi_weights = weights
    profile.updated_at = datetime.utcnow()
    
    # Update top KPIs
    top_kpis = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    profile.top_kpis = [kpi for kpi, _ in top_kpis]
    
    logger.info(f"✅ Updated KPI weights for profile: {profile_id}")
    return profile

@app.get("/persona-kpi/all-personas")
async def get_all_persona_profiles():
    """Get KPI profiles for all personas"""
    
    profiles = {}
    for persona in PersonaType:
        profile = persona_engine.create_persona_profile(persona)
        profiles[persona.value] = profile
    
    return profiles

@app.get("/persona-kpi/compare/{persona1}/{persona2}")
async def compare_persona_weights(
    persona1: PersonaType,
    persona2: PersonaType
):
    """Compare KPI weights between two personas"""
    
    profile1 = persona_engine.create_persona_profile(persona1)
    profile2 = persona_engine.create_persona_profile(persona2)
    
    # Find common metrics
    common_metrics = set(profile1.kpi_weights.keys()) & set(profile2.kpi_weights.keys())
    
    comparison = {
        "persona_1": {
            "name": persona1.value,
            "weights": profile1.kpi_weights,
            "top_kpis": profile1.top_kpis
        },
        "persona_2": {
            "name": persona2.value,
            "weights": profile2.kpi_weights,
            "top_kpis": profile2.top_kpis
        },
        "differences": {
            metric: {
                "persona_1_weight": profile1.kpi_weights.get(metric, 0.0),
                "persona_2_weight": profile2.kpi_weights.get(metric, 0.0),
                "difference": profile1.kpi_weights.get(metric, 0.0) - profile2.kpi_weights.get(metric, 0.0)
            }
            for metric in common_metrics
        }
    }
    
    return comparison

@app.get("/persona-kpi/summary")
async def get_persona_kpi_summary():
    """Get persona KPI system summary"""
    
    total_profiles = len(persona_profiles_store)
    total_weightings = len(persona_weightings_store)
    
    return {
        "total_profiles": total_profiles,
        "total_weightings": total_weightings,
        "supported_personas": [p.value for p in PersonaType],
        "service_status": "active"
    }

