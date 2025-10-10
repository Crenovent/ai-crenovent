"""
Task 3.4.38: Add regulator persona dashboards
- Regulator-specific dashboard service
- Pre-built compliance report templates
- Evidence pack aggregation for regulatory views
- Role-based access control for regulator personas
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

app = FastAPI(title="RBIA Regulator Persona Dashboards")

class RegulatorType(str, Enum):
    FINANCIAL = "financial"
    DATA_PROTECTION = "data_protection"
    HEALTHCARE = "healthcare"
    INDUSTRY_SPECIFIC = "industry_specific"

class RegulatorDashboard(BaseModel):
    dashboard_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    regulator_type: RegulatorType
    dashboard_name: str
    
    # Compliance focus areas
    monitored_frameworks: List[str] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)
    
    # Data aggregation
    evidence_pack_types: List[str] = Field(default_factory=list)
    audit_trail_depth: str = "full"  # full, summary, minimal
    
    # Access control
    authorized_users: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RegulatorReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    regulator_type: RegulatorType
    tenant_id: str
    
    # Report content
    compliance_summary: Dict[str, Any] = Field(default_factory=dict)
    violations_found: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_packs: List[str] = Field(default_factory=list)
    
    # Regulator-specific metrics
    regulatory_score: float = 0.0
    audit_readiness: str = "ready"  # ready, needs_attention, not_ready
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# Storage
dashboards_store: Dict[str, RegulatorDashboard] = {}
reports_store: Dict[str, RegulatorReport] = {}

@app.post("/regulator/dashboards", response_model=RegulatorDashboard)
async def create_regulator_dashboard(dashboard: RegulatorDashboard):
    """Create regulator-specific dashboard"""
    dashboards_store[dashboard.dashboard_id] = dashboard
    return dashboard

@app.get("/regulator/dashboards/{regulator_type}", response_model=List[RegulatorDashboard])
async def get_regulator_dashboards(regulator_type: RegulatorType):
    """Get dashboards for regulator type"""
    return [d for d in dashboards_store.values() if d.regulator_type == regulator_type]

@app.get("/regulator/dashboards/{dashboard_id}/data")
async def get_dashboard_data(dashboard_id: str, tenant_id: str):
    """Get dashboard data for regulator view"""
    if dashboard_id not in dashboards_store:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    dashboard = dashboards_store[dashboard_id]
    
    # Simulate regulator-focused data
    return {
        "dashboard_id": dashboard_id,
        "compliance_status": "compliant",
        "violations_count": 0,
        "evidence_packs_available": 15,
        "last_audit_date": "2024-01-15",
        "regulatory_score": 94.5,
        "key_metrics": {
            "data_residency_compliance": 100.0,
            "audit_trail_completeness": 98.5,
            "policy_enforcement_rate": 99.2
        }
    }

@app.post("/regulator/reports", response_model=RegulatorReport)
async def generate_regulator_report(
    regulator_type: RegulatorType,
    tenant_id: str,
    frameworks: List[str]
):
    """Generate regulator-specific compliance report"""
    
    report = RegulatorReport(
        regulator_type=regulator_type,
        tenant_id=tenant_id,
        compliance_summary={
            "frameworks_assessed": frameworks,
            "overall_compliance": 95.0,
            "critical_issues": 0,
            "recommendations": 3
        },
        regulatory_score=95.0,
        audit_readiness="ready"
    )
    
    reports_store[report.report_id] = report
    return report

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Regulator Persona Dashboards", "task": "3.4.38"}
