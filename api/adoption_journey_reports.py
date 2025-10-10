"""
Task 4.4.27: Add adoption journey reports (RBA-only → RBIA adoption rate)
- Transition metric
- Adoption tracker
- CRO-level adoption index
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Adoption Journey Reports")
logger = logging.getLogger(__name__)

class AdoptionStage(str, Enum):
    RBA_ONLY = "rba_only"
    RBIA_TRIAL = "rbia_trial"
    RBIA_PARTIAL = "rbia_partial"
    RBIA_FULL = "rbia_full"
    RBIA_ADVANCED = "rbia_advanced"

class JourneyMilestone(BaseModel):
    milestone_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    stage: AdoptionStage
    achieved_date: datetime
    duration_days: int = 0
    key_metrics: Dict[str, float] = Field(default_factory=dict)

class AdoptionJourney(BaseModel):
    journey_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    tenant_name: str
    
    # Journey timeline
    journey_start: datetime
    current_stage: AdoptionStage
    milestones: List[JourneyMilestone] = Field(default_factory=list)
    
    # Progress metrics
    total_journey_days: int = 0
    rba_to_trial_days: int = 0
    trial_to_partial_days: int = 0
    partial_to_full_days: int = 0
    
    # Adoption metrics
    rbia_adoption_percentage: float = 0.0
    active_rbia_workflows: int = 0
    total_workflows: int = 0
    
    # Success indicators
    journey_velocity_score: float = 0.0  # 0-100 scale
    adoption_health_score: float = 0.0   # 0-100 scale
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class AdoptionJourneyReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Report metadata
    report_title: str = "RBIA Adoption Journey Analysis"
    report_period: str
    total_tenants: int = 0
    
    # Journey analytics
    stage_distribution: Dict[str, int] = Field(default_factory=dict)
    average_journey_duration: Dict[str, float] = Field(default_factory=dict)
    conversion_rates: Dict[str, float] = Field(default_factory=dict)
    
    # Performance metrics
    fastest_adopters: List[AdoptionJourney] = Field(default_factory=list)
    struggling_adopters: List[AdoptionJourney] = Field(default_factory=list)
    
    # Insights
    key_insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
adoption_journeys_store: Dict[str, AdoptionJourney] = {}
journey_reports_store: Dict[str, AdoptionJourneyReport] = {}

class AdoptionJourneyTracker:
    def __init__(self):
        # Mock journey data (in production, would track actual tenant progression)
        self.mock_journeys = {
            "tenant_1": {
                "name": "Acme Corp",
                "start_date": datetime.utcnow() - timedelta(days=180),
                "current_stage": AdoptionStage.RBIA_FULL,
                "milestones": [
                    {"stage": AdoptionStage.RBA_ONLY, "days_ago": 180, "metrics": {"workflows": 5, "executions": 200}},
                    {"stage": AdoptionStage.RBIA_TRIAL, "days_ago": 150, "metrics": {"rbia_workflows": 2, "adoption_rate": 15.0}},
                    {"stage": AdoptionStage.RBIA_PARTIAL, "days_ago": 90, "metrics": {"rbia_workflows": 8, "adoption_rate": 45.0}},
                    {"stage": AdoptionStage.RBIA_FULL, "days_ago": 30, "metrics": {"rbia_workflows": 15, "adoption_rate": 85.0}}
                ]
            },
            "tenant_2": {
                "name": "Beta Industries",
                "start_date": datetime.utcnow() - timedelta(days=120),
                "current_stage": AdoptionStage.RBIA_PARTIAL,
                "milestones": [
                    {"stage": AdoptionStage.RBA_ONLY, "days_ago": 120, "metrics": {"workflows": 8, "executions": 350}},
                    {"stage": AdoptionStage.RBIA_TRIAL, "days_ago": 90, "metrics": {"rbia_workflows": 3, "adoption_rate": 25.0}},
                    {"stage": AdoptionStage.RBIA_PARTIAL, "days_ago": 45, "metrics": {"rbia_workflows": 6, "adoption_rate": 55.0}}
                ]
            },
            "tenant_3": {
                "name": "Gamma Solutions",
                "start_date": datetime.utcnow() - timedelta(days=90),
                "current_stage": AdoptionStage.RBIA_TRIAL,
                "milestones": [
                    {"stage": AdoptionStage.RBA_ONLY, "days_ago": 90, "metrics": {"workflows": 3, "executions": 150}},
                    {"stage": AdoptionStage.RBIA_TRIAL, "days_ago": 60, "metrics": {"rbia_workflows": 1, "adoption_rate": 10.0}}
                ]
            }
        }
    
    def calculate_journey_velocity(self, journey: AdoptionJourney) -> float:
        """Calculate journey velocity score (0-100)"""
        
        if not journey.milestones:
            return 0.0
        
        # Expected timeline benchmarks (in days)
        expected_timelines = {
            AdoptionStage.RBIA_TRIAL: 30,      # 30 days from RBA-only to trial
            AdoptionStage.RBIA_PARTIAL: 90,    # 90 days to partial adoption
            AdoptionStage.RBIA_FULL: 150,      # 150 days to full adoption
            AdoptionStage.RBIA_ADVANCED: 200   # 200 days to advanced
        }
        
        # Calculate actual vs expected timeline
        current_stage_timeline = expected_timelines.get(journey.current_stage, 180)
        actual_timeline = journey.total_journey_days
        
        if actual_timeline <= current_stage_timeline:
            # Ahead of or on schedule
            velocity_score = min(100, (current_stage_timeline / actual_timeline) * 100)
        else:
            # Behind schedule
            velocity_score = max(0, 100 - ((actual_timeline - current_stage_timeline) / current_stage_timeline * 50))
        
        return round(velocity_score, 1)
    
    def calculate_adoption_health(self, journey: AdoptionJourney) -> float:
        """Calculate adoption health score (0-100)"""
        
        # Health factors
        stage_progression_score = {
            AdoptionStage.RBA_ONLY: 20,
            AdoptionStage.RBIA_TRIAL: 40,
            AdoptionStage.RBIA_PARTIAL: 60,
            AdoptionStage.RBIA_FULL: 85,
            AdoptionStage.RBIA_ADVANCED: 100
        }.get(journey.current_stage, 0)
        
        # Adoption percentage bonus
        adoption_bonus = min(20, journey.rbia_adoption_percentage / 5)  # Up to 20 points for 100% adoption
        
        # Workflow diversity bonus
        workflow_bonus = min(15, journey.active_rbia_workflows * 2)  # Up to 15 points for workflow count
        
        health_score = stage_progression_score + adoption_bonus + workflow_bonus
        
        return min(100, round(health_score, 1))
    
    def generate_adoption_journey(self, tenant_id: str) -> AdoptionJourney:
        """Generate adoption journey for a tenant"""
        
        tenant_data = self.mock_journeys.get(tenant_id)
        if not tenant_data:
            raise ValueError(f"No journey data found for tenant {tenant_id}")
        
        # Create milestones
        milestones = []
        for i, milestone_data in enumerate(tenant_data["milestones"]):
            milestone_date = datetime.utcnow() - timedelta(days=milestone_data["days_ago"])
            duration = milestone_data["days_ago"] - (tenant_data["milestones"][i+1]["days_ago"] if i+1 < len(tenant_data["milestones"]) else 0)
            
            milestone = JourneyMilestone(
                tenant_id=tenant_id,
                stage=AdoptionStage(milestone_data["stage"]),
                achieved_date=milestone_date,
                duration_days=duration,
                key_metrics=milestone_data["metrics"]
            )
            milestones.append(milestone)
        
        # Calculate journey metrics
        total_days = (datetime.utcnow() - tenant_data["start_date"]).days
        current_metrics = tenant_data["milestones"][-1]["metrics"]
        
        journey = AdoptionJourney(
            tenant_id=tenant_id,
            tenant_name=tenant_data["name"],
            journey_start=tenant_data["start_date"],
            current_stage=AdoptionStage(tenant_data["current_stage"]),
            milestones=milestones,
            total_journey_days=total_days,
            rbia_adoption_percentage=current_metrics.get("adoption_rate", 0.0),
            active_rbia_workflows=current_metrics.get("rbia_workflows", 0),
            total_workflows=current_metrics.get("workflows", 0) + current_metrics.get("rbia_workflows", 0)
        )
        
        # Calculate derived metrics
        journey.journey_velocity_score = self.calculate_journey_velocity(journey)
        journey.adoption_health_score = self.calculate_adoption_health(journey)
        
        return journey
    
    def generate_journey_report(self, report_period: str = "quarterly") -> AdoptionJourneyReport:
        """Generate comprehensive adoption journey report"""
        
        # Generate journeys for all tenants
        journeys = []
        for tenant_id in self.mock_journeys.keys():
            journey = self.generate_adoption_journey(tenant_id)
            journeys.append(journey)
            adoption_journeys_store[journey.journey_id] = journey
        
        # Calculate stage distribution
        stage_distribution = {}
        for stage in AdoptionStage:
            stage_distribution[stage.value] = len([j for j in journeys if j.current_stage == stage])
        
        # Calculate average journey durations
        average_durations = {}
        for stage in AdoptionStage:
            stage_journeys = [j for j in journeys if j.current_stage == stage]
            if stage_journeys:
                avg_duration = sum([j.total_journey_days for j in stage_journeys]) / len(stage_journeys)
                average_durations[stage.value] = round(avg_duration, 1)
        
        # Calculate conversion rates
        conversion_rates = {}
        total_started = len(journeys)
        for stage in AdoptionStage:
            if stage != AdoptionStage.RBA_ONLY:
                reached_stage = len([j for j in journeys if j.current_stage.value >= stage.value or any(m.stage == stage for m in j.milestones)])
                conversion_rates[f"to_{stage.value}"] = (reached_stage / total_started * 100) if total_started > 0 else 0.0
        
        # Identify fastest and struggling adopters
        fastest_adopters = sorted(journeys, key=lambda x: x.journey_velocity_score, reverse=True)[:3]
        struggling_adopters = sorted(journeys, key=lambda x: x.adoption_health_score)[:2]
        
        # Generate insights
        key_insights = [
            f"{stage_distribution.get('rbia_full', 0)} tenants have achieved full RBIA adoption",
            f"Average time to trial: {average_durations.get('rbia_trial', 0)} days",
            f"Full adoption conversion rate: {conversion_rates.get('to_rbia_full', 0):.1f}%",
            f"Fastest adopter achieved full adoption in {min([j.total_journey_days for j in journeys if j.current_stage == AdoptionStage.RBIA_FULL], default=0)} days"
        ]
        
        # Generate recommendations
        recommendations = [
            "Focus onboarding resources on tenants in trial stage to accelerate progression",
            "Develop success playbooks based on fastest adopter patterns",
            "Implement early warning system for struggling adopters",
            "Create milestone celebration program to maintain adoption momentum"
        ]
        
        return AdoptionJourneyReport(
            report_period=report_period,
            total_tenants=len(journeys),
            stage_distribution=stage_distribution,
            average_journey_duration=average_durations,
            conversion_rates=conversion_rates,
            fastest_adopters=fastest_adopters,
            struggling_adopters=struggling_adopters,
            key_insights=key_insights,
            recommendations=recommendations
        )

# Global journey tracker
journey_tracker = AdoptionJourneyTracker()

@app.post("/adoption-journey/report", response_model=AdoptionJourneyReport)
async def generate_adoption_journey_report(report_period: str = "quarterly"):
    """Generate adoption journey report"""
    
    report = journey_tracker.generate_journey_report(report_period)
    
    # Store report
    journey_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated adoption journey report - {report.total_tenants} tenants analyzed")
    return report

@app.get("/adoption-journey/tenant/{tenant_id}")
async def get_tenant_journey(tenant_id: str):
    """Get adoption journey for a specific tenant"""
    
    try:
        journey = journey_tracker.generate_adoption_journey(tenant_id)
        
        # Store journey
        adoption_journeys_store[journey.journey_id] = journey
        
        return journey
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/adoption-journey/stage-analysis")
async def get_stage_analysis():
    """Get analysis of adoption stages"""
    
    # Generate current journeys
    journeys = []
    for tenant_id in journey_tracker.mock_journeys.keys():
        journey = journey_tracker.generate_adoption_journey(tenant_id)
        journeys.append(journey)
    
    # Analyze by stage
    stage_analysis = {}
    for stage in AdoptionStage:
        stage_journeys = [j for j in journeys if j.current_stage == stage]
        
        if stage_journeys:
            stage_analysis[stage.value] = {
                "tenant_count": len(stage_journeys),
                "avg_velocity_score": round(sum([j.journey_velocity_score for j in stage_journeys]) / len(stage_journeys), 1),
                "avg_health_score": round(sum([j.adoption_health_score for j in stage_journeys]) / len(stage_journeys), 1),
                "avg_days_in_stage": round(sum([j.total_journey_days for j in stage_journeys]) / len(stage_journeys), 1)
            }
    
    return {
        "stage_analysis": stage_analysis,
        "total_tenants": len(journeys)
    }

@app.get("/adoption-journey/conversion-funnel")
async def get_conversion_funnel():
    """Get adoption conversion funnel"""
    
    # Generate current journeys
    journeys = []
    for tenant_id in journey_tracker.mock_journeys.keys():
        journey = journey_tracker.generate_adoption_journey(tenant_id)
        journeys.append(journey)
    
    # Build conversion funnel
    funnel_stages = [
        {"stage": "RBA Only", "count": len(journeys), "percentage": 100.0},
        {"stage": "RBIA Trial", "count": len([j for j in journeys if j.current_stage.value != "rba_only"]), "percentage": 0},
        {"stage": "RBIA Partial", "count": len([j for j in journeys if j.current_stage.value in ["rbia_partial", "rbia_full", "rbia_advanced"]]), "percentage": 0},
        {"stage": "RBIA Full", "count": len([j for j in journeys if j.current_stage.value in ["rbia_full", "rbia_advanced"]]), "percentage": 0},
        {"stage": "RBIA Advanced", "count": len([j for j in journeys if j.current_stage == AdoptionStage.RBIA_ADVANCED]), "percentage": 0}
    ]
    
    # Calculate percentages
    total_tenants = len(journeys)
    for stage in funnel_stages[1:]:
        stage["percentage"] = (stage["count"] / total_tenants * 100) if total_tenants > 0 else 0.0
    
    return {
        "conversion_funnel": funnel_stages,
        "total_tenants": total_tenants,
        "overall_conversion_rate": funnel_stages[-2]["percentage"]  # Full adoption rate
    }

@app.get("/adoption-journey/summary")
async def get_adoption_journey_summary():
    """Get adoption journey service summary"""
    
    total_journeys = len(adoption_journeys_store)
    total_reports = len(journey_reports_store)
    
    # Calculate summary stats
    if adoption_journeys_store:
        all_journeys = list(adoption_journeys_store.values())
        avg_velocity = sum([j.journey_velocity_score for j in all_journeys]) / len(all_journeys)
        avg_health = sum([j.adoption_health_score for j in all_journeys]) / len(all_journeys)
        
        # Stage distribution
        stage_counts = {}
        for journey in all_journeys:
            stage = journey.current_stage.value
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    else:
        avg_velocity = avg_health = 0.0
        stage_counts = {}
    
    return {
        "total_journeys_tracked": total_journeys,
        "total_reports_generated": total_reports,
        "average_velocity_score": round(avg_velocity, 1),
        "average_health_score": round(avg_health, 1),
        "stage_distribution": stage_counts
    }
