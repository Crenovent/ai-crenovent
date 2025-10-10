#!/usr/bin/env python3
"""
Governance Dashboard API - Chapter 14 Integration
================================================
Tasks 14.1-T17, T18, 14.2-T19 to T26, 14.3-T22 to T25, 14.4-T20 to T23

Features:
- Executive governance dashboards (CRO, CFO, Compliance Officer)
- Operational dashboards (RevOps, Ops, Finance)
- Real-time approval workflow monitoring
- Trust scoring visualization and trends
- Evidence pack coverage and audit readiness
- Compliance posture and risk assessment
- Industry-specific regulatory dashboards
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

# Database and dependencies
from src.services.connection_pool_manager import get_pool_manager
from dsl.governance.rbac_dashboard_manager import RBACDashboardManager
from dsl.governance.industry_overlay_dashboard_manager import IndustryOverlayDashboardManager

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# PYDANTIC MODELS
# =====================================================

class DashboardType(str, Enum):
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    FINANCE = "finance"
    SECURITY = "security"

class PersonaType(str, Enum):
    CRO = "cro"
    CFO = "cfo"
    COMPLIANCE_OFFICER = "compliance_officer"
    REVOPS_MANAGER = "revops_manager"
    OPS_MANAGER = "ops_manager"
    FINANCE_MANAGER = "finance_manager"
    SECURITY_ADMIN = "security_admin"

class TimeRange(str, Enum):
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    LAST_6_MONTHS = "6m"
    LAST_12_MONTHS = "12m"

class MetricType(str, Enum):
    APPROVAL_SLA = "approval_sla"
    TRUST_SCORE = "trust_score"
    EVIDENCE_COVERAGE = "evidence_coverage"
    COMPLIANCE_POSTURE = "compliance_posture"
    OVERRIDE_FREQUENCY = "override_frequency"
    AUDIT_READINESS = "audit_readiness"

class DashboardRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    dashboard_type: DashboardType = Field(..., description="Type of dashboard")
    persona: PersonaType = Field(..., description="User persona")
    time_range: TimeRange = Field(TimeRange.LAST_30_DAYS, description="Time range for data")
    industry_filter: Optional[str] = Field(None, description="Industry filter")
    compliance_framework_filter: Optional[str] = Field(None, description="Compliance framework filter")

# =====================================================
# GOVERNANCE DASHBOARD SERVICE
# =====================================================

class GovernanceDashboardService:
    """
    Comprehensive governance dashboard service
    Integrates all Chapter 14 components for unified governance visibility
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.rbac_dashboard_manager = RBACDashboardManager(pool_manager)
        self.industry_overlay_manager = IndustryOverlayDashboardManager(pool_manager)
        
        # Dashboard configurations by persona
        self.persona_configs = {
            PersonaType.CRO: {
                "primary_metrics": ["trust_score", "approval_sla", "compliance_posture"],
                "widgets": ["trust_ladder", "approval_trends", "business_impact", "risk_summary"],
                "refresh_interval": 300  # 5 minutes
            },
            PersonaType.CFO: {
                "primary_metrics": ["approval_sla", "override_frequency", "audit_readiness"],
                "widgets": ["financial_controls", "sox_compliance", "approval_costs", "override_impact"],
                "refresh_interval": 600  # 10 minutes
            },
            PersonaType.COMPLIANCE_OFFICER: {
                "primary_metrics": ["compliance_posture", "evidence_coverage", "audit_readiness"],
                "widgets": ["policy_violations", "evidence_gaps", "regulatory_status", "audit_timeline"],
                "refresh_interval": 180  # 3 minutes
            },
            PersonaType.REVOPS_MANAGER: {
                "primary_metrics": ["approval_sla", "trust_score", "workflow_efficiency"],
                "widgets": ["pending_approvals", "workflow_performance", "trust_trends", "adoption_metrics"],
                "refresh_interval": 120  # 2 minutes
            }
        }
    
    async def get_executive_dashboard(self, request: DashboardRequest) -> Dict[str, Any]:
        """Get executive governance dashboard"""
        
        try:
            dashboard_data = {
                "dashboard_id": f"exec_{request.persona.value}_{request.tenant_id}",
                "persona": request.persona.value,
                "tenant_id": request.tenant_id,
                "generated_at": datetime.utcnow(),
                "time_range": request.time_range.value,
                "widgets": {}
            }
            
            # Get persona-specific widgets
            if request.persona == PersonaType.CRO:
                dashboard_data["widgets"] = await self._get_cro_widgets(request)
            elif request.persona == PersonaType.CFO:
                dashboard_data["widgets"] = await self._get_cfo_widgets(request)
            elif request.persona == PersonaType.COMPLIANCE_OFFICER:
                dashboard_data["widgets"] = await self._get_compliance_widgets(request)
            
            # Add common executive metrics
            dashboard_data["summary_metrics"] = await self._get_executive_summary_metrics(request)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get executive dashboard: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_operational_dashboard(self, request: DashboardRequest) -> Dict[str, Any]:
        """Get operational governance dashboard"""
        
        try:
            dashboard_data = {
                "dashboard_id": f"ops_{request.persona.value}_{request.tenant_id}",
                "persona": request.persona.value,
                "tenant_id": request.tenant_id,
                "generated_at": datetime.utcnow(),
                "time_range": request.time_range.value,
                "widgets": {}
            }
            
            # Get operational widgets
            if request.persona == PersonaType.REVOPS_MANAGER:
                dashboard_data["widgets"] = await self._get_revops_widgets(request)
            elif request.persona == PersonaType.OPS_MANAGER:
                dashboard_data["widgets"] = await self._get_ops_widgets(request)
            elif request.persona == PersonaType.FINANCE_MANAGER:
                dashboard_data["widgets"] = await self._get_finance_widgets(request)
            
            # Add operational metrics
            dashboard_data["operational_metrics"] = await self._get_operational_metrics(request)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get operational dashboard: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_real_time_metrics(self, tenant_id: int, metrics: List[MetricType]) -> Dict[str, Any]:
        """Get real-time governance metrics"""
        
        try:
            real_time_data = {
                "tenant_id": tenant_id,
                "timestamp": datetime.utcnow(),
                "metrics": {}
            }
            
            async with self.pool_manager.get_connection() as conn:
                for metric in metrics:
                    if metric == MetricType.APPROVAL_SLA:
                        real_time_data["metrics"]["approval_sla"] = await self._get_approval_sla_metrics(conn, tenant_id)
                    elif metric == MetricType.TRUST_SCORE:
                        real_time_data["metrics"]["trust_score"] = await self._get_trust_score_metrics(conn, tenant_id)
                    elif metric == MetricType.EVIDENCE_COVERAGE:
                        real_time_data["metrics"]["evidence_coverage"] = await self._get_evidence_coverage_metrics(conn, tenant_id)
                    elif metric == MetricType.COMPLIANCE_POSTURE:
                        real_time_data["metrics"]["compliance_posture"] = await self._get_compliance_posture_metrics(conn, tenant_id)
                    elif metric == MetricType.OVERRIDE_FREQUENCY:
                        real_time_data["metrics"]["override_frequency"] = await self._get_override_frequency_metrics(conn, tenant_id)
                    elif metric == MetricType.AUDIT_READINESS:
                        real_time_data["metrics"]["audit_readiness"] = await self._get_audit_readiness_metrics(conn, tenant_id)
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get real-time metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Persona-specific widget methods
    async def _get_cro_widgets(self, request: DashboardRequest) -> Dict[str, Any]:
        """CRO dashboard widgets - Trust ladder, business impact, risk summary"""
        
        async with self.pool_manager.get_connection() as conn:
            return {
                "trust_ladder": {
                    "title": "Trust Ladder Overview",
                    "data": await self._get_trust_ladder_data(conn, request.tenant_id),
                    "chart_type": "ladder"
                },
                "business_impact": {
                    "title": "Business Impact Summary",
                    "data": await self._get_business_impact_data(conn, request.tenant_id),
                    "chart_type": "metrics"
                },
                "approval_trends": {
                    "title": "Approval Trends",
                    "data": await self._get_approval_trends_data(conn, request.tenant_id, request.time_range),
                    "chart_type": "line"
                },
                "risk_summary": {
                    "title": "Risk Summary",
                    "data": await self._get_risk_summary_data(conn, request.tenant_id),
                    "chart_type": "gauge"
                }
            }
    
    async def _get_cfo_widgets(self, request: DashboardRequest) -> Dict[str, Any]:
        """CFO dashboard widgets - Financial controls, SOX compliance, costs"""
        
        async with self.pool_manager.get_connection() as conn:
            return {
                "financial_controls": {
                    "title": "Financial Controls Status",
                    "data": await self._get_financial_controls_data(conn, request.tenant_id),
                    "chart_type": "status"
                },
                "sox_compliance": {
                    "title": "SOX Compliance Posture",
                    "data": await self._get_sox_compliance_data(conn, request.tenant_id),
                    "chart_type": "compliance"
                },
                "approval_costs": {
                    "title": "Approval Process Costs",
                    "data": await self._get_approval_costs_data(conn, request.tenant_id, request.time_range),
                    "chart_type": "cost"
                },
                "override_impact": {
                    "title": "Override Financial Impact",
                    "data": await self._get_override_impact_data(conn, request.tenant_id),
                    "chart_type": "impact"
                }
            }
    
    async def _get_compliance_widgets(self, request: DashboardRequest) -> Dict[str, Any]:
        """Compliance Officer dashboard widgets - Policy violations, evidence gaps, regulatory status"""
        
        async with self.pool_manager.get_connection() as conn:
            return {
                "policy_violations": {
                    "title": "Policy Violations",
                    "data": await self._get_policy_violations_data(conn, request.tenant_id),
                    "chart_type": "violations"
                },
                "evidence_gaps": {
                    "title": "Evidence Coverage Gaps",
                    "data": await self._get_evidence_gaps_data(conn, request.tenant_id),
                    "chart_type": "gaps"
                },
                "regulatory_status": {
                    "title": "Regulatory Compliance Status",
                    "data": await self._get_regulatory_status_data(conn, request.tenant_id),
                    "chart_type": "regulatory"
                },
                "audit_timeline": {
                    "title": "Audit Readiness Timeline",
                    "data": await self._get_audit_timeline_data(conn, request.tenant_id),
                    "chart_type": "timeline"
                }
            }
    
    async def _get_revops_widgets(self, request: DashboardRequest) -> Dict[str, Any]:
        """RevOps Manager dashboard widgets"""
        
        async with self.pool_manager.get_connection() as conn:
            return {
                "pending_approvals": {
                    "title": "Pending Approvals",
                    "data": await self._get_pending_approvals_data(conn, request.tenant_id),
                    "chart_type": "list"
                },
                "workflow_performance": {
                    "title": "Workflow Performance",
                    "data": await self._get_workflow_performance_data(conn, request.tenant_id),
                    "chart_type": "performance"
                },
                "trust_trends": {
                    "title": "Trust Score Trends",
                    "data": await self._get_trust_trends_data(conn, request.tenant_id, request.time_range),
                    "chart_type": "trend"
                }
            }
    
    # Metric calculation methods
    async def _get_approval_sla_metrics(self, conn, tenant_id: int) -> Dict[str, Any]:
        """Get approval SLA metrics"""
        
        # Get SLA adherence data
        sla_data = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_approvals,
                COUNT(CASE WHEN completed_at <= expires_at THEN 1 END) as on_time_approvals,
                AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_resolution_hours,
                COUNT(CASE WHEN status = 'pending' AND expires_at < NOW() THEN 1 END) as overdue_approvals
            FROM approval_ledger
            WHERE tenant_id = $1 AND created_at >= $2
        """, tenant_id, datetime.utcnow() - timedelta(days=30))
        
        sla_adherence = (sla_data['on_time_approvals'] / sla_data['total_approvals'] * 100) if sla_data['total_approvals'] > 0 else 0
        
        return {
            "sla_adherence_percentage": round(sla_adherence, 1),
            "total_approvals": sla_data['total_approvals'],
            "on_time_approvals": sla_data['on_time_approvals'],
            "avg_resolution_hours": round(float(sla_data['avg_resolution_hours']) if sla_data['avg_resolution_hours'] else 0, 1),
            "overdue_approvals": sla_data['overdue_approvals'],
            "status": "good" if sla_adherence >= 90 else "warning" if sla_adherence >= 75 else "critical"
        }
    
    async def _get_trust_score_metrics(self, conn, tenant_id: int) -> Dict[str, Any]:
        """Get trust score metrics"""
        
        trust_data = await conn.fetchrow("""
            SELECT 
                AVG(overall_score) as avg_trust_score,
                COUNT(CASE WHEN trust_level = 'excellent' THEN 1 END) as excellent_count,
                COUNT(CASE WHEN trust_level = 'good' THEN 1 END) as good_count,
                COUNT(CASE WHEN trust_level = 'concerning' THEN 1 END) as concerning_count,
                COUNT(*) as total_scores
            FROM trust_scores
            WHERE tenant_id = $1 AND calculated_at >= $2
        """, tenant_id, datetime.utcnow() - timedelta(days=30))
        
        avg_score = float(trust_data['avg_trust_score']) if trust_data['avg_trust_score'] else 0
        
        return {
            "average_trust_score": round(avg_score, 3),
            "trust_distribution": {
                "excellent": trust_data['excellent_count'],
                "good": trust_data['good_count'],
                "concerning": trust_data['concerning_count']
            },
            "total_assessments": trust_data['total_scores'],
            "trend": "improving" if avg_score > 0.8 else "stable" if avg_score > 0.7 else "declining"
        }
    
    async def _get_evidence_coverage_metrics(self, conn, tenant_id: int) -> Dict[str, Any]:
        """Get evidence coverage metrics"""
        
        evidence_data = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_evidence_packs,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_packs,
                SUM(evidence_size_bytes) as total_size_bytes,
                COUNT(CASE WHEN export_count > 0 THEN 1 END) as exported_packs
            FROM evidence_packs
            WHERE tenant_id = $1 AND created_at >= $2
        """, tenant_id, datetime.utcnow() - timedelta(days=30))
        
        coverage_percentage = (evidence_data['active_packs'] / evidence_data['total_evidence_packs'] * 100) if evidence_data['total_evidence_packs'] > 0 else 0
        
        return {
            "coverage_percentage": round(coverage_percentage, 1),
            "total_evidence_packs": evidence_data['total_evidence_packs'],
            "active_packs": evidence_data['active_packs'],
            "total_size_mb": round((evidence_data['total_size_bytes'] or 0) / 1024 / 1024, 2),
            "exported_packs": evidence_data['exported_packs'],
            "status": "complete" if coverage_percentage >= 95 else "partial" if coverage_percentage >= 80 else "incomplete"
        }
    
    async def _get_compliance_posture_metrics(self, conn, tenant_id: int) -> Dict[str, Any]:
        """Get compliance posture metrics"""
        
        # This would integrate with policy engine and compliance monitoring
        return {
            "overall_compliance_score": 87.5,
            "framework_scores": {
                "SOX": 92.0,
                "GDPR": 85.0,
                "RBI": 88.0
            },
            "violations_count": 3,
            "remediation_in_progress": 2,
            "status": "compliant"
        }
    
    async def _get_override_frequency_metrics(self, conn, tenant_id: int) -> Dict[str, Any]:
        """Get override frequency metrics"""
        
        override_data = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_overrides,
                COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_risk_overrides,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_overrides,
                AVG(estimated_business_impact) as avg_business_impact
            FROM override_ledger
            WHERE tenant_id = $1 AND created_at >= $2
        """, tenant_id, datetime.utcnow() - timedelta(days=30))
        
        return {
            "total_overrides": override_data['total_overrides'],
            "high_risk_overrides": override_data['high_risk_overrides'],
            "active_overrides": override_data['active_overrides'],
            "avg_business_impact": round(float(override_data['avg_business_impact']) if override_data['avg_business_impact'] else 0, 2),
            "frequency_trend": "increasing" if override_data['total_overrides'] > 10 else "stable"
        }
    
    async def _get_audit_readiness_metrics(self, conn, tenant_id: int) -> Dict[str, Any]:
        """Get audit readiness metrics"""
        
        audit_data = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_audit_packs,
                COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_packs,
                COUNT(CASE WHEN submitted_to_regulator THEN 1 END) as submitted_packs
            FROM audit_packs
            WHERE tenant_id = $1
        """, tenant_id)
        
        readiness_score = (audit_data['approved_packs'] / audit_data['total_audit_packs'] * 100) if audit_data['total_audit_packs'] > 0 else 0
        
        return {
            "readiness_score": round(readiness_score, 1),
            "total_audit_packs": audit_data['total_audit_packs'],
            "approved_packs": audit_data['approved_packs'],
            "submitted_packs": audit_data['submitted_packs'],
            "status": "ready" if readiness_score >= 90 else "preparing" if readiness_score >= 70 else "not_ready"
        }
    
    # Additional helper methods for widget data
    async def _get_executive_summary_metrics(self, request: DashboardRequest) -> Dict[str, Any]:
        """Get executive summary metrics"""
        
        return {
            "governance_health_score": 88.5,
            "risk_level": "Medium",
            "compliance_status": "Compliant",
            "pending_actions": 7,
            "last_updated": datetime.utcnow()
        }
    
    async def _get_operational_metrics(self, request: DashboardRequest) -> Dict[str, Any]:
        """Get operational metrics"""
        
        return {
            "workflow_efficiency": 92.3,
            "automation_rate": 78.5,
            "error_rate": 2.1,
            "sla_adherence": 94.7
        }
    
    # Placeholder methods for specific widget data (would be implemented based on actual requirements)
    async def _get_trust_ladder_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"levels": ["Draft", "Reviewed", "Approved", "Trusted", "Autonomous"], "current_distribution": [10, 25, 40, 20, 5]}
    
    async def _get_business_impact_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"revenue_impact": 125000, "cost_savings": 45000, "efficiency_gain": 15.2}
    
    async def _get_approval_trends_data(self, conn, tenant_id: int, time_range: TimeRange) -> Dict[str, Any]:
        return {"trend": "improving", "data_points": [85, 87, 89, 91, 93]}
    
    async def _get_risk_summary_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"overall_risk": "Medium", "risk_factors": {"operational": "Low", "compliance": "Medium", "financial": "Low"}}
    
    async def _get_financial_controls_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"controls_active": 45, "controls_total": 50, "effectiveness": 90.0}
    
    async def _get_sox_compliance_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"compliance_score": 92.0, "controls_tested": 48, "deficiencies": 2}
    
    async def _get_approval_costs_data(self, conn, tenant_id: int, time_range: TimeRange) -> Dict[str, Any]:
        return {"total_cost": 15000, "cost_per_approval": 125, "trend": "decreasing"}
    
    async def _get_override_impact_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"financial_impact": 25000, "risk_exposure": "Medium", "remediation_cost": 8000}
    
    async def _get_policy_violations_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"total_violations": 5, "critical": 1, "high": 2, "medium": 2}
    
    async def _get_evidence_gaps_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"gaps_identified": 3, "coverage_percentage": 94.5, "missing_evidence_types": ["approval_workflows"]}
    
    async def _get_regulatory_status_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"frameworks": {"SOX": "Compliant", "GDPR": "Compliant", "RBI": "Under Review"}}
    
    async def _get_audit_timeline_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"next_audit": "2024-03-15", "preparation_status": "On Track", "days_remaining": 45}
    
    async def _get_pending_approvals_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"count": 12, "urgent": 3, "overdue": 1, "avg_age_hours": 18.5}
    
    async def _get_workflow_performance_data(self, conn, tenant_id: int) -> Dict[str, Any]:
        return {"success_rate": 96.5, "avg_execution_time": 45.2, "error_rate": 3.5}
    
    async def _get_trust_trends_data(self, conn, tenant_id: int, time_range: TimeRange) -> Dict[str, Any]:
        return {"trend": "improving", "current_score": 0.875, "change": "+0.025"}

# =====================================================
# API ENDPOINTS
# =====================================================

# Initialize service
dashboard_service = None

def get_dashboard_service(pool_manager=Depends(get_pool_manager)) -> GovernanceDashboardService:
    global dashboard_service
    if dashboard_service is None:
        dashboard_service = GovernanceDashboardService(pool_manager)
    return dashboard_service

@router.post("/executive")
async def get_executive_dashboard(
    request: DashboardRequest,
    service: GovernanceDashboardService = Depends(get_dashboard_service)
):
    """
    Get executive governance dashboard
    Tasks 14.1-T17, 14.2-T19, 14.3-T22: Executive dashboards
    """
    return await service.get_executive_dashboard(request)

@router.post("/operational")
async def get_operational_dashboard(
    request: DashboardRequest,
    service: GovernanceDashboardService = Depends(get_dashboard_service)
):
    """
    Get operational governance dashboard
    Tasks 14.1-T18, 14.2-T21, 14.3-T24: Operational dashboards
    """
    return await service.get_operational_dashboard(request)

@router.get("/real-time-metrics")
async def get_real_time_metrics(
    tenant_id: int = Query(..., description="Tenant ID"),
    metrics: str = Query(..., description="Comma-separated metric types"),
    service: GovernanceDashboardService = Depends(get_dashboard_service)
):
    """Get real-time governance metrics"""
    
    try:
        metric_types = [MetricType(m.strip()) for m in metrics.split(",")]
        return await service.get_real_time_metrics(tenant_id, metric_types)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metric type: {e}")

@router.get("/persona-config/{persona}")
async def get_persona_config(
    persona: PersonaType,
    service: GovernanceDashboardService = Depends(get_dashboard_service)
):
    """Get dashboard configuration for persona"""
    
    config = service.persona_configs.get(persona)
    if not config:
        raise HTTPException(status_code=404, detail="Persona configuration not found")
    
    return {
        "persona": persona.value,
        "config": config
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "governance_dashboard_api", "timestamp": datetime.utcnow()}
