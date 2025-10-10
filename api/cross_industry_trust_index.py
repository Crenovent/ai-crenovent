"""
Task 4.4.34: Provide cross-industry trust index report
- Compare trust scores across industries
- Identify industry leaders
- Benchmark against industry averages
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging
import statistics

app = FastAPI(title="RBIA Cross-Industry Trust Index")
logger = logging.getLogger(__name__)

class IndustryType(str, Enum):
    BANKING = "banking"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    SAAS = "saas"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TELECOM = "telecom"

class TrustComponent(str, Enum):
    ACCURACY = "accuracy"
    EXPLAINABILITY = "explainability"
    DRIFT_STABILITY = "drift_stability"
    BIAS_FAIRNESS = "bias_fairness"
    GOVERNANCE_COMPLIANCE = "governance_compliance"
    OPERATIONAL_SLA = "operational_sla"
    SECURITY = "security"

class IndustryTrustScore(BaseModel):
    industry: IndustryType
    
    # Component scores (0-100)
    accuracy_score: float
    explainability_score: float
    drift_stability_score: float
    bias_fairness_score: float
    governance_compliance_score: float
    operational_sla_score: float
    security_score: float
    
    # Composite trust index (0-100)
    trust_index: float
    
    # Tenant statistics
    total_tenants: int
    active_tenants: int
    
    # Performance metrics
    avg_predictions_per_day: float
    avg_override_rate: float
    
    calculated_at: datetime = Field(default_factory=datetime.utcnow)

class CrossIndustryTrustReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Industry scores
    industry_scores: Dict[str, IndustryTrustScore] = Field(default_factory=dict)
    
    # Rankings
    top_industries: List[str] = Field(default_factory=list)
    bottom_industries: List[str] = Field(default_factory=list)
    
    # Overall statistics
    overall_avg_trust_index: float
    highest_trust_index: float
    lowest_trust_index: float
    
    # Component-wise rankings
    accuracy_leader: str
    explainability_leader: str
    governance_leader: str
    
    # Insights
    key_findings: List[str] = Field(default_factory=list)
    industry_trends: Dict[str, Any] = Field(default_factory=dict)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
industry_trust_scores_store: Dict[str, IndustryTrustScore] = {}
cross_industry_reports_store: Dict[str, CrossIndustryTrustReport] = {}

class CrossIndustryTrustEngine:
    def __init__(self):
        # Baseline industry trust scores
        self.baseline_scores = {
            IndustryType.BANKING: {
                "accuracy": 88.5,
                "explainability": 92.0,
                "drift_stability": 86.0,
                "bias_fairness": 90.0,
                "governance": 95.0,
                "sla": 91.0,
                "security": 96.0,
                "tenants": 45,
                "active": 42,
                "predictions_per_day": 125000,
                "override_rate": 3.2
            },
            IndustryType.INSURANCE: {
                "accuracy": 85.2,
                "explainability": 88.0,
                "drift_stability": 84.0,
                "bias_fairness": 87.0,
                "governance": 93.0,
                "sla": 89.0,
                "security": 94.0,
                "tenants": 38,
                "active": 35,
                "predictions_per_day": 98000,
                "override_rate": 4.1
            },
            IndustryType.HEALTHCARE: {
                "accuracy": 90.1,
                "explainability": 94.0,
                "drift_stability": 88.0,
                "bias_fairness": 92.0,
                "governance": 96.0,
                "sla": 93.0,
                "security": 97.0,
                "tenants": 32,
                "active": 30,
                "predictions_per_day": 87000,
                "override_rate": 2.8
            },
            IndustryType.FINTECH: {
                "accuracy": 87.8,
                "explainability": 85.0,
                "drift_stability": 82.0,
                "bias_fairness": 86.0,
                "governance": 89.0,
                "sla": 92.0,
                "security": 93.0,
                "tenants": 56,
                "active": 52,
                "predictions_per_day": 145000,
                "override_rate": 3.7
            },
            IndustryType.SAAS: {
                "accuracy": 84.3,
                "explainability": 82.0,
                "drift_stability": 80.0,
                "bias_fairness": 83.0,
                "governance": 86.0,
                "sla": 94.0,
                "security": 91.0,
                "tenants": 78,
                "active": 73,
                "predictions_per_day": 178000,
                "override_rate": 4.8
            },
            IndustryType.RETAIL: {
                "accuracy": 82.1,
                "explainability": 79.0,
                "drift_stability": 78.0,
                "bias_fairness": 81.0,
                "governance": 82.0,
                "sla": 90.0,
                "security": 88.0,
                "tenants": 64,
                "active": 59,
                "predictions_per_day": 165000,
                "override_rate": 5.2
            },
            IndustryType.MANUFACTURING: {
                "accuracy": 86.4,
                "explainability": 87.0,
                "drift_stability": 85.0,
                "bias_fairness": 88.0,
                "governance": 90.0,
                "sla": 88.0,
                "security": 92.0,
                "tenants": 28,
                "active": 26,
                "predictions_per_day": 72000,
                "override_rate": 3.5
            },
            IndustryType.TELECOM: {
                "accuracy": 83.7,
                "explainability": 81.0,
                "drift_stability": 79.0,
                "bias_fairness": 84.0,
                "governance": 87.0,
                "sla": 95.0,
                "security": 90.0,
                "tenants": 41,
                "active": 38,
                "predictions_per_day": 132000,
                "override_rate": 4.4
            }
        }
    
    def calculate_trust_index(self, scores: Dict[str, float]) -> float:
        """Calculate composite trust index with weighted components"""
        
        weights = {
            "accuracy": 0.20,
            "explainability": 0.18,
            "drift_stability": 0.15,
            "bias_fairness": 0.15,
            "governance": 0.18,
            "sla": 0.10,
            "security": 0.04
        }
        
        trust_index = (
            scores["accuracy"] * weights["accuracy"] +
            scores["explainability"] * weights["explainability"] +
            scores["drift_stability"] * weights["drift_stability"] +
            scores["bias_fairness"] * weights["bias_fairness"] +
            scores["governance"] * weights["governance"] +
            scores["sla"] * weights["sla"] +
            scores["security"] * weights["security"]
        )
        
        return round(trust_index, 2)
    
    def generate_industry_trust_scores(self) -> Dict[str, IndustryTrustScore]:
        """Generate trust scores for all industries"""
        
        industry_scores = {}
        
        for industry, baseline in self.baseline_scores.items():
            trust_index = self.calculate_trust_index(baseline)
            
            industry_score = IndustryTrustScore(
                industry=industry,
                accuracy_score=baseline["accuracy"],
                explainability_score=baseline["explainability"],
                drift_stability_score=baseline["drift_stability"],
                bias_fairness_score=baseline["bias_fairness"],
                governance_compliance_score=baseline["governance"],
                operational_sla_score=baseline["sla"],
                security_score=baseline["security"],
                trust_index=trust_index,
                total_tenants=baseline["tenants"],
                active_tenants=baseline["active"],
                avg_predictions_per_day=baseline["predictions_per_day"],
                avg_override_rate=baseline["override_rate"]
            )
            
            industry_scores[industry.value] = industry_score
            industry_trust_scores_store[industry.value] = industry_score
        
        return industry_scores
    
    def generate_cross_industry_report(self) -> CrossIndustryTrustReport:
        """Generate comprehensive cross-industry trust report"""
        
        # Generate industry scores
        industry_scores = self.generate_industry_trust_scores()
        
        # Calculate rankings
        sorted_industries = sorted(
            industry_scores.items(),
            key=lambda x: x[1].trust_index,
            reverse=True
        )
        
        top_industries = [ind[0] for ind in sorted_industries[:3]]
        bottom_industries = [ind[0] for ind in sorted_industries[-3:]]
        
        # Overall statistics
        all_trust_indices = [score.trust_index for score in industry_scores.values()]
        overall_avg = statistics.mean(all_trust_indices)
        highest = max(all_trust_indices)
        lowest = min(all_trust_indices)
        
        # Component-wise leaders
        accuracy_leader = max(
            industry_scores.items(),
            key=lambda x: x[1].accuracy_score
        )[0]
        
        explainability_leader = max(
            industry_scores.items(),
            key=lambda x: x[1].explainability_score
        )[0]
        
        governance_leader = max(
            industry_scores.items(),
            key=lambda x: x[1].governance_compliance_score
        )[0]
        
        # Key findings
        top_industry = sorted_industries[0]
        key_findings = [
            f"{top_industry[0]} leads with trust index of {top_industry[1].trust_index}",
            f"Healthcare shows highest governance compliance at {industry_scores['healthcare'].governance_compliance_score}%",
            f"SaaS has the highest adoption with {industry_scores['saas'].total_tenants} tenants",
            f"Banking demonstrates strong security at {industry_scores['banking'].security_score}%",
            f"Overall average trust index is {overall_avg:.2f} across all industries"
        ]
        
        # Industry trends
        industry_trends = {
            "high_growth_industries": ["fintech", "saas"],
            "most_regulated": ["banking", "healthcare", "insurance"],
            "highest_volume": ["saas", "fintech", "retail"],
            "trust_gap": f"{highest - lowest:.2f} points between highest and lowest"
        }
        
        report = CrossIndustryTrustReport(
            industry_scores=industry_scores,
            top_industries=top_industries,
            bottom_industries=bottom_industries,
            overall_avg_trust_index=overall_avg,
            highest_trust_index=highest,
            lowest_trust_index=lowest,
            accuracy_leader=accuracy_leader,
            explainability_leader=explainability_leader,
            governance_leader=governance_leader,
            key_findings=key_findings,
            industry_trends=industry_trends
        )
        
        cross_industry_reports_store[report.report_id] = report
        return report

# Global trust engine
trust_engine = CrossIndustryTrustEngine()

@app.post("/cross-industry-trust/generate", response_model=CrossIndustryTrustReport)
async def generate_cross_industry_trust_report():
    """Generate cross-industry trust index report"""
    
    report = trust_engine.generate_cross_industry_report()
    
    logger.info(f"✅ Generated cross-industry trust report - Top industry: {report.top_industries[0]}")
    return report

@app.get("/cross-industry-trust/industry/{industry}")
async def get_industry_trust_score(industry: IndustryType):
    """Get trust score for specific industry"""
    
    score = industry_trust_scores_store.get(industry.value)
    if not score:
        # Generate if not exists
        trust_engine.generate_industry_trust_scores()
        score = industry_trust_scores_store.get(industry.value)
    
    if not score:
        raise HTTPException(status_code=404, detail="Industry not found")
    
    return score

@app.get("/cross-industry-trust/compare")
async def compare_industries(
    industry1: IndustryType,
    industry2: IndustryType
):
    """Compare trust scores between two industries"""
    
    # Ensure scores exist
    trust_engine.generate_industry_trust_scores()
    
    score1 = industry_trust_scores_store.get(industry1.value)
    score2 = industry_trust_scores_store.get(industry2.value)
    
    if not score1 or not score2:
        raise HTTPException(status_code=404, detail="Industry comparison failed")
    
    comparison = {
        "industry_1": {
            "name": industry1.value,
            "trust_index": score1.trust_index,
            "accuracy": score1.accuracy_score,
            "governance": score1.governance_compliance_score,
            "tenants": score1.total_tenants
        },
        "industry_2": {
            "name": industry2.value,
            "trust_index": score2.trust_index,
            "accuracy": score2.accuracy_score,
            "governance": score2.governance_compliance_score,
            "tenants": score2.total_tenants
        },
        "differences": {
            "trust_index_diff": score1.trust_index - score2.trust_index,
            "accuracy_diff": score1.accuracy_score - score2.accuracy_score,
            "governance_diff": score1.governance_compliance_score - score2.governance_compliance_score
        },
        "winner": industry1.value if score1.trust_index > score2.trust_index else industry2.value
    }
    
    return comparison

@app.get("/cross-industry-trust/rankings")
async def get_industry_rankings():
    """Get industry rankings by trust index"""
    
    # Generate scores
    trust_engine.generate_industry_trust_scores()
    
    rankings = sorted(
        industry_trust_scores_store.items(),
        key=lambda x: x[1].trust_index,
        reverse=True
    )
    
    return {
        "rankings": [
            {
                "rank": idx + 1,
                "industry": industry,
                "trust_index": score.trust_index,
                "accuracy": score.accuracy_score,
                "governance": score.governance_compliance_score,
                "tenants": score.total_tenants
            }
            for idx, (industry, score) in enumerate(rankings)
        ],
        "total_industries": len(rankings)
    }

@app.get("/cross-industry-trust/component-leaders")
async def get_component_leaders():
    """Get leaders for each trust component"""
    
    # Generate scores
    trust_engine.generate_industry_trust_scores()
    
    leaders = {
        "accuracy": max(
            industry_trust_scores_store.items(),
            key=lambda x: x[1].accuracy_score
        ),
        "explainability": max(
            industry_trust_scores_store.items(),
            key=lambda x: x[1].explainability_score
        ),
        "drift_stability": max(
            industry_trust_scores_store.items(),
            key=lambda x: x[1].drift_stability_score
        ),
        "bias_fairness": max(
            industry_trust_scores_store.items(),
            key=lambda x: x[1].bias_fairness_score
        ),
        "governance": max(
            industry_trust_scores_store.items(),
            key=lambda x: x[1].governance_compliance_score
        ),
        "sla": max(
            industry_trust_scores_store.items(),
            key=lambda x: x[1].operational_sla_score
        ),
        "security": max(
            industry_trust_scores_store.items(),
            key=lambda x: x[1].security_score
        )
    }
    
    return {
        component: {
            "leader": industry,
            "score": score.dict()[f"{component}_score"]
        }
        for component, (industry, score) in leaders.items()
    }

@app.get("/cross-industry-trust/summary")
async def get_trust_index_summary():
    """Get cross-industry trust index summary"""
    
    total_scores = len(industry_trust_scores_store)
    total_reports = len(cross_industry_reports_store)
    
    return {
        "total_industries_tracked": total_scores,
        "total_reports_generated": total_reports,
        "latest_report_date": max(
            [r.generated_at for r in cross_industry_reports_store.values()],
            default=datetime.utcnow()
        ).isoformat() if cross_industry_reports_store else None,
        "service_status": "active"
    }

