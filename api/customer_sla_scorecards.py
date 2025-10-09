"""
Task 4.4.26: Provide customer-specific SLA scorecards
- Transparency to customers
- Dashboard + export
- Contractual trust
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Customer SLA Scorecards")
logger = logging.getLogger(__name__)

class SLAMetric(str, Enum):
    AVAILABILITY = "availability"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    ACCURACY = "accuracy"

class SLAStatus(str, Enum):
    MET = "met"
    BREACHED = "breached"
    AT_RISK = "at_risk"

class SLATarget(BaseModel):
    metric: SLAMetric
    target_value: float
    unit: str
    measurement_period: str = "monthly"

class SLAMeasurement(BaseModel):
    measurement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric: SLAMetric
    measured_value: float
    target_value: float
    unit: str
    status: SLAStatus
    variance_percentage: float = 0.0
    measurement_date: datetime = Field(default_factory=datetime.utcnow)

class CustomerSLAScorecard(BaseModel):
    scorecard_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    customer_name: str
    
    # Scorecard period
    period_start: datetime
    period_end: datetime
    
    # SLA measurements
    measurements: List[SLAMeasurement] = Field(default_factory=list)
    
    # Overall SLA performance
    overall_sla_score: float = 0.0  # 0-100 percentage
    sla_compliance_rate: float = 0.0  # Percentage of SLAs met
    
    # Summary by metric
    availability_summary: Dict[str, Any] = Field(default_factory=dict)
    performance_summary: Dict[str, Any] = Field(default_factory=dict)
    quality_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Breach information
    total_breaches: int = 0
    breach_details: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Credits and remediation
    sla_credits_earned: float = 0.0
    remediation_actions: List[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class SLAContract(BaseModel):
    contract_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    customer_name: str
    
    # SLA targets
    sla_targets: List[SLATarget] = Field(default_factory=list)
    
    # Contract terms
    contract_start: datetime
    contract_end: datetime
    credit_percentage_per_breach: float = 5.0  # 5% credit per breach
    
    # Contact information
    customer_contact: str
    technical_contact: str

# In-memory storage
sla_scorecards_store: Dict[str, CustomerSLAScorecard] = {}
sla_contracts_store: Dict[str, SLAContract] = {}
sla_measurements_store: Dict[str, List[SLAMeasurement]] = {}

class SLAScorecardGenerator:
    def __init__(self):
        # Default SLA targets
        self.default_sla_targets = [
            SLATarget(metric=SLAMetric.AVAILABILITY, target_value=99.9, unit="percentage"),
            SLATarget(metric=SLAMetric.RESPONSE_TIME, target_value=2000, unit="milliseconds"),
            SLATarget(metric=SLAMetric.THROUGHPUT, target_value=1000, unit="requests_per_minute"),
            SLATarget(metric=SLAMetric.ERROR_RATE, target_value=0.1, unit="percentage"),
            SLATarget(metric=SLAMetric.ACCURACY, target_value=95.0, unit="percentage")
        ]
        
        # Mock measurement data (in production, would come from monitoring systems)
        self.mock_measurements = {
            "tenant_1": [
                SLAMeasurement(
                    metric=SLAMetric.AVAILABILITY,
                    measured_value=99.95,
                    target_value=99.9,
                    unit="percentage",
                    status=SLAStatus.MET,
                    variance_percentage=0.05
                ),
                SLAMeasurement(
                    metric=SLAMetric.RESPONSE_TIME,
                    measured_value=1850,
                    target_value=2000,
                    unit="milliseconds",
                    status=SLAStatus.MET,
                    variance_percentage=-7.5
                ),
                SLAMeasurement(
                    metric=SLAMetric.ERROR_RATE,
                    measured_value=0.08,
                    target_value=0.1,
                    unit="percentage",
                    status=SLAStatus.MET,
                    variance_percentage=-20.0
                ),
                SLAMeasurement(
                    metric=SLAMetric.ACCURACY,
                    measured_value=96.2,
                    target_value=95.0,
                    unit="percentage",
                    status=SLAStatus.MET,
                    variance_percentage=1.26
                )
            ]
        }
    
    def calculate_sla_score(self, measurements: List[SLAMeasurement]) -> float:
        """Calculate overall SLA score from measurements"""
        
        if not measurements:
            return 0.0
        
        # Weight different metrics
        weights = {
            SLAMetric.AVAILABILITY: 0.3,
            SLAMetric.RESPONSE_TIME: 0.25,
            SLAMetric.THROUGHPUT: 0.2,
            SLAMetric.ERROR_RATE: 0.15,
            SLAMetric.ACCURACY: 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for measurement in measurements:
            weight = weights.get(measurement.metric, 0.1)
            
            # Calculate metric score (100 if met, proportional if breached)
            if measurement.status == SLAStatus.MET:
                metric_score = 100.0
            elif measurement.status == SLAStatus.AT_RISK:
                metric_score = 85.0
            else:  # BREACHED
                # Calculate proportional score based on how far off target
                if measurement.variance_percentage < 0:
                    metric_score = 100.0  # Better than target
                else:
                    # Worse than target - calculate proportional penalty
                    penalty = min(50.0, measurement.variance_percentage * 2)
                    metric_score = max(0.0, 100.0 - penalty)
            
            weighted_score += metric_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def generate_scorecard(self, tenant_id: str, customer_name: str,
                          period_start: datetime, period_end: datetime) -> CustomerSLAScorecard:
        """Generate SLA scorecard for a customer"""
        
        # Get measurements for the period
        measurements = self.mock_measurements.get(tenant_id, [])
        
        # Calculate overall SLA score
        overall_score = self.calculate_sla_score(measurements)
        
        # Calculate compliance rate
        met_slas = len([m for m in measurements if m.status == SLAStatus.MET])
        compliance_rate = (met_slas / len(measurements) * 100) if measurements else 0.0
        
        # Generate summaries
        availability_measurements = [m for m in measurements if m.metric == SLAMetric.AVAILABILITY]
        performance_measurements = [m for m in measurements if m.metric in [SLAMetric.RESPONSE_TIME, SLAMetric.THROUGHPUT]]
        quality_measurements = [m for m in measurements if m.metric in [SLAMetric.ERROR_RATE, SLAMetric.ACCURACY]]
        
        availability_summary = {
            "average_availability": sum([m.measured_value for m in availability_measurements]) / len(availability_measurements) if availability_measurements else 0.0,
            "target_availability": 99.9,
            "status": "met" if availability_measurements and all(m.status == SLAStatus.MET for m in availability_measurements) else "breached"
        }
        
        performance_summary = {
            "average_response_time": sum([m.measured_value for m in performance_measurements if m.metric == SLAMetric.RESPONSE_TIME]) / len([m for m in performance_measurements if m.metric == SLAMetric.RESPONSE_TIME]) if performance_measurements else 0.0,
            "target_response_time": 2000,
            "status": "met" if performance_measurements and all(m.status == SLAStatus.MET for m in performance_measurements) else "breached"
        }
        
        quality_summary = {
            "average_accuracy": sum([m.measured_value for m in quality_measurements if m.metric == SLAMetric.ACCURACY]) / len([m for m in quality_measurements if m.metric == SLAMetric.ACCURACY]) if quality_measurements else 0.0,
            "average_error_rate": sum([m.measured_value for m in quality_measurements if m.metric == SLAMetric.ERROR_RATE]) / len([m for m in quality_measurements if m.metric == SLAMetric.ERROR_RATE]) if quality_measurements else 0.0,
            "status": "met" if quality_measurements and all(m.status == SLAStatus.MET for m in quality_measurements) else "breached"
        }
        
        # Calculate breaches
        breached_measurements = [m for m in measurements if m.status == SLAStatus.BREACHED]
        total_breaches = len(breached_measurements)
        
        breach_details = [
            {
                "metric": breach.metric.value,
                "measured_value": breach.measured_value,
                "target_value": breach.target_value,
                "variance": breach.variance_percentage,
                "date": breach.measurement_date.isoformat()
            }
            for breach in breached_measurements
        ]
        
        # Calculate SLA credits
        sla_credits = total_breaches * 5.0  # 5% credit per breach
        
        # Generate remediation actions
        remediation_actions = []
        if total_breaches > 0:
            remediation_actions.extend([
                "Enhanced monitoring implemented for affected metrics",
                "Root cause analysis completed for each breach",
                "Process improvements deployed to prevent recurrence"
            ])
        
        return CustomerSLAScorecard(
            tenant_id=tenant_id,
            customer_name=customer_name,
            period_start=period_start,
            period_end=period_end,
            measurements=measurements,
            overall_sla_score=overall_score,
            sla_compliance_rate=compliance_rate,
            availability_summary=availability_summary,
            performance_summary=performance_summary,
            quality_summary=quality_summary,
            total_breaches=total_breaches,
            breach_details=breach_details,
            sla_credits_earned=sla_credits,
            remediation_actions=remediation_actions
        )

# Global scorecard generator
scorecard_generator = SLAScorecardGenerator()

@app.post("/sla-scorecards/generate", response_model=CustomerSLAScorecard)
async def generate_sla_scorecard(
    tenant_id: str,
    customer_name: str,
    period_start: datetime,
    period_end: datetime
):
    """Generate SLA scorecard for a customer"""
    
    scorecard = scorecard_generator.generate_scorecard(
        tenant_id, customer_name, period_start, period_end
    )
    
    # Store scorecard
    sla_scorecards_store[scorecard.scorecard_id] = scorecard
    
    logger.info(f"✅ Generated SLA scorecard for {customer_name} - Score: {scorecard.overall_sla_score:.1f}%")
    return scorecard

@app.get("/sla-scorecards/tenant/{tenant_id}")
async def get_tenant_scorecards(tenant_id: str, limit: int = 5):
    """Get SLA scorecards for a tenant"""
    
    tenant_scorecards = [
        scorecard for scorecard in sla_scorecards_store.values()
        if scorecard.tenant_id == tenant_id
    ]
    
    # Sort by generation date (most recent first)
    tenant_scorecards.sort(key=lambda x: x.generated_at, reverse=True)
    
    return {
        "tenant_id": tenant_id,
        "scorecard_count": len(tenant_scorecards),
        "scorecards": tenant_scorecards[:limit]
    }

@app.get("/sla-scorecards/{scorecard_id}")
async def get_scorecard(scorecard_id: str):
    """Get specific SLA scorecard"""
    
    scorecard = sla_scorecards_store.get(scorecard_id)
    
    if not scorecard:
        raise HTTPException(status_code=404, detail="SLA scorecard not found")
    
    return scorecard

@app.get("/sla-scorecards/{scorecard_id}/export")
async def export_scorecard(scorecard_id: str, format: str = "json"):
    """Export SLA scorecard in specified format"""
    
    scorecard = sla_scorecards_store.get(scorecard_id)
    
    if not scorecard:
        raise HTTPException(status_code=404, detail="SLA scorecard not found")
    
    if format.lower() == "csv":
        # Generate CSV content
        csv_content = "Metric,Measured Value,Target Value,Unit,Status,Variance %\n"
        
        for measurement in scorecard.measurements:
            csv_content += f"{measurement.metric.value},{measurement.measured_value},{measurement.target_value},"
            csv_content += f"{measurement.unit},{measurement.status.value},{measurement.variance_percentage}\n"
        
        from fastapi.responses import Response
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=sla_scorecard_{scorecard_id}.csv"}
        )
    
    else:
        # Return JSON format
        return scorecard

@app.post("/sla-contracts/create", response_model=SLAContract)
async def create_sla_contract(contract: SLAContract):
    """Create SLA contract for a customer"""
    
    # Use default targets if none provided
    if not contract.sla_targets:
        contract.sla_targets = scorecard_generator.default_sla_targets
    
    # Store contract
    sla_contracts_store[contract.contract_id] = contract
    
    logger.info(f"✅ Created SLA contract for {contract.customer_name}")
    return contract

@app.get("/sla-contracts/tenant/{tenant_id}")
async def get_tenant_contracts(tenant_id: str):
    """Get SLA contracts for a tenant"""
    
    tenant_contracts = [
        contract for contract in sla_contracts_store.values()
        if contract.tenant_id == tenant_id
    ]
    
    return {
        "tenant_id": tenant_id,
        "contract_count": len(tenant_contracts),
        "contracts": tenant_contracts
    }

@app.get("/sla-scorecards/summary")
async def get_sla_summary():
    """Get SLA scorecard service summary"""
    
    total_scorecards = len(sla_scorecards_store)
    total_contracts = len(sla_contracts_store)
    
    if total_scorecards == 0:
        return {
            "total_scorecards": 0,
            "total_contracts": 0,
            "average_sla_score": 0.0,
            "breach_rate": 0.0
        }
    
    # Calculate averages
    all_scorecards = list(sla_scorecards_store.values())
    avg_sla_score = sum([s.overall_sla_score for s in all_scorecards]) / len(all_scorecards)
    total_measurements = sum([len(s.measurements) for s in all_scorecards])
    total_breaches = sum([s.total_breaches for s in all_scorecards])
    breach_rate = (total_breaches / total_measurements * 100) if total_measurements > 0 else 0.0
    
    return {
        "total_scorecards": total_scorecards,
        "total_contracts": total_contracts,
        "unique_tenants": len(set(s.tenant_id for s in all_scorecards)),
        "average_sla_score": round(avg_sla_score, 2),
        "breach_rate": round(breach_rate, 2),
        "total_credits_issued": sum([s.sla_credits_earned for s in all_scorecards])
    }
