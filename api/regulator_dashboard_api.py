"""
CHAPTER 16.5: REGULATOR DASHBOARD API
Tasks 16.5.1-16.5.20: Regulator-facing dashboard backend with compliance reporting
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field
from enum import Enum

from src.database.connection_pool import get_pool_manager, ConnectionPoolManager
from src.auth.auth_utils import get_current_user, get_tenant_id_from_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/regulator-dashboard", tags=["Regulator Dashboard"])

class ComplianceFramework(str, Enum):
    SOX = "SOX"
    GDPR = "GDPR"
    RBI = "RBI"
    HIPAA = "HIPAA"
    NAIC = "NAIC"
    PCI_DSS = "PCI_DSS"

class ReportType(str, Enum):
    COMPLIANCE_SUMMARY = "compliance_summary"
    VIOLATION_REPORT = "violation_report"
    AUDIT_TRAIL = "audit_trail"
    OVERRIDE_ANALYSIS = "override_analysis"
    POLICY_ENFORCEMENT = "policy_enforcement"

class RegulatorDashboardService:
    """
    Task 16.5.1: Regulator Dashboard Service
    Provides compliance reporting and regulatory oversight capabilities
    """
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.logger = logger
    
    async def get_compliance_overview(self, tenant_id: int, framework: str, period_days: int = 30) -> Dict[str, Any]:
        """Task 16.5.2: Get compliance overview for regulator"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                # Get compliance enforcement metrics
                compliance_query = """
                    SELECT 
                        COUNT(*) as total_enforcements,
                        COUNT(CASE WHEN enforcement_result = 'compliant' THEN 1 END) as compliant_count,
                        COUNT(CASE WHEN enforcement_result = 'violation' THEN 1 END) as violation_count,
                        AVG(compliance_score) as avg_compliance_score,
                        SUM(critical_violations) as total_critical_violations,
                        SUM(high_violations) as total_high_violations
                    FROM compliance_overlay_enforcement coe
                    JOIN compliance_overlay co ON coe.overlay_id = co.overlay_id
                    WHERE coe.tenant_id = $1 
                    AND co.compliance_framework = $2
                    AND coe.enforced_at >= NOW() - INTERVAL '%s days'
                """ % period_days
                
                compliance_row = await conn.fetchrow(compliance_query, tenant_id, framework)
                
                # Get override statistics
                override_query = """
                    SELECT 
                        COUNT(*) as total_overrides,
                        COUNT(CASE WHEN approval_status = 'approved' THEN 1 END) as approved_overrides,
                        COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_overrides,
                        SUM(COALESCE(financial_impact_usd, 0)) as total_financial_impact
                    FROM override_ledger_entry
                    WHERE tenant_id = $1 
                    AND compliance_framework = $2
                    AND created_at >= NOW() - INTERVAL '%s days'
                """ % period_days
                
                override_row = await conn.fetchrow(override_query, tenant_id, framework)
                
                # Get audit pack statistics
                audit_query = """
                    SELECT 
                        COUNT(*) as total_audit_packs,
                        COUNT(CASE WHEN status = 'submitted' THEN 1 END) as submitted_packs,
                        COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_packs
                    FROM audit_packs
                    WHERE tenant_id = $1 
                    AND compliance_framework = $2
                    AND created_at >= NOW() - INTERVAL '%s days'
                """ % period_days
                
                audit_row = await conn.fetchrow(audit_query, tenant_id, framework)
                
                return {
                    "compliance_metrics": dict(compliance_row) if compliance_row else {},
                    "override_metrics": dict(override_row) if override_row else {},
                    "audit_metrics": dict(audit_row) if audit_row else {},
                    "framework": framework,
                    "period_days": period_days,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get compliance overview: {e}")
            raise
    
    async def get_violation_details(self, tenant_id: int, framework: str, severity: str = None) -> List[Dict[str, Any]]:
        """Task 16.5.3: Get detailed violation information"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                query = """
                    SELECT 
                        coe.enforcement_id,
                        coe.workflow_id,
                        coe.enforcement_result,
                        coe.violation_details,
                        coe.compliance_score,
                        coe.critical_violations,
                        coe.high_violations,
                        coe.enforced_at,
                        co.overlay_name,
                        co.regulatory_authority
                    FROM compliance_overlay_enforcement coe
                    JOIN compliance_overlay co ON coe.overlay_id = co.overlay_id
                    WHERE coe.tenant_id = $1 
                    AND co.compliance_framework = $2
                    AND coe.enforcement_result = 'violation'
                """
                params = [tenant_id, framework]
                
                if severity == "critical":
                    query += " AND coe.critical_violations > 0"
                elif severity == "high":
                    query += " AND coe.high_violations > 0"
                
                query += " ORDER BY coe.enforced_at DESC LIMIT 100"
                
                rows = await conn.fetch(query, *params)
                
                violations = []
                for row in rows:
                    violation_dict = dict(row)
                    violation_dict['violation_details'] = row['violation_details'] if row['violation_details'] else {}
                    violations.append(violation_dict)
                
                return violations
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get violation details: {e}")
            raise
    
    async def get_override_analysis(self, tenant_id: int, framework: str, period_days: int = 30) -> Dict[str, Any]:
        """Task 16.5.4: Get override analysis for regulatory review"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                # Get override trends
                trend_query = """
                    SELECT 
                        DATE_TRUNC('day', created_at) as date,
                        COUNT(*) as override_count,
                        COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_count,
                        SUM(COALESCE(financial_impact_usd, 0)) as daily_financial_impact
                    FROM override_ledger_entry
                    WHERE tenant_id = $1 
                    AND compliance_framework = $2
                    AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY DATE_TRUNC('day', created_at)
                    ORDER BY date
                """ % period_days
                
                trend_rows = await conn.fetch(trend_query, tenant_id, framework)
                
                # Get override categories
                category_query = """
                    SELECT 
                        override_category,
                        COUNT(*) as count,
                        AVG(COALESCE(financial_impact_usd, 0)) as avg_financial_impact
                    FROM override_ledger_entry
                    WHERE tenant_id = $1 
                    AND compliance_framework = $2
                    AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY override_category
                """ % period_days
                
                category_rows = await conn.fetch(category_query, tenant_id, framework)
                
                return {
                    "daily_trends": [dict(row) for row in trend_rows],
                    "category_breakdown": [dict(row) for row in category_rows],
                    "framework": framework,
                    "period_days": period_days,
                    "analysis_date": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get override analysis: {e}")
            raise
    
    async def generate_regulatory_report(self, tenant_id: int, framework: str, 
                                       report_type: str, period_start: datetime, 
                                       period_end: datetime) -> Dict[str, Any]:
        """Task 16.5.5: Generate regulatory compliance report"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                report_id = f"REG_{framework}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if report_type == "compliance_summary":
                    report_data = await self._generate_compliance_summary(conn, tenant_id, framework, period_start, period_end)
                elif report_type == "violation_report":
                    report_data = await self._generate_violation_report(conn, tenant_id, framework, period_start, period_end)
                elif report_type == "override_analysis":
                    report_data = await self._generate_override_report(conn, tenant_id, framework, period_start, period_end)
                else:
                    raise ValueError(f"Unsupported report type: {report_type}")
                
                # Store report metadata
                await conn.execute("""
                    INSERT INTO compliance_overlay_report (
                        tenant_id, report_name, report_type, compliance_framework,
                        report_period_start, report_period_end, report_data,
                        generated_by_user_id, regulatory_filing_required
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, tenant_id, f"{framework} {report_type.title()} Report", report_type, framework,
                period_start, period_end, report_data, None, True)
                
                return {
                    "report_id": report_id,
                    "report_type": report_type,
                    "framework": framework,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "report_data": report_data,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate regulatory report: {e}")
            raise
    
    async def _generate_compliance_summary(self, conn, tenant_id: int, framework: str, 
                                         period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate compliance summary report"""
        # Get overall compliance metrics
        summary_query = """
            SELECT 
                COUNT(*) as total_evaluations,
                COUNT(CASE WHEN enforcement_result = 'compliant' THEN 1 END) as compliant_evaluations,
                COUNT(CASE WHEN enforcement_result = 'violation' THEN 1 END) as violations,
                AVG(compliance_score) as avg_compliance_score,
                MIN(compliance_score) as min_compliance_score,
                MAX(compliance_score) as max_compliance_score
            FROM compliance_overlay_enforcement coe
            JOIN compliance_overlay co ON coe.overlay_id = co.overlay_id
            WHERE coe.tenant_id = $1 
            AND co.compliance_framework = $2
            AND coe.enforced_at BETWEEN $3 AND $4
        """
        
        summary_row = await conn.fetchrow(summary_query, tenant_id, framework, period_start, period_end)
        
        return {
            "summary_metrics": dict(summary_row) if summary_row else {},
            "compliance_rate": (summary_row["compliant_evaluations"] / max(1, summary_row["total_evaluations"]) * 100) if summary_row else 0,
            "report_period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            }
        }
    
    async def _generate_violation_report(self, conn, tenant_id: int, framework: str,
                                       period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate detailed violation report"""
        violations_query = """
            SELECT 
                coe.enforcement_id,
                coe.workflow_id,
                coe.violation_details,
                coe.critical_violations,
                coe.high_violations,
                coe.medium_violations,
                coe.low_violations,
                coe.enforced_at,
                co.overlay_name
            FROM compliance_overlay_enforcement coe
            JOIN compliance_overlay co ON coe.overlay_id = co.overlay_id
            WHERE coe.tenant_id = $1 
            AND co.compliance_framework = $2
            AND coe.enforcement_result = 'violation'
            AND coe.enforced_at BETWEEN $3 AND $4
            ORDER BY coe.enforced_at DESC
        """
        
        violation_rows = await conn.fetch(violations_query, tenant_id, framework, period_start, period_end)
        
        return {
            "total_violations": len(violation_rows),
            "violations": [dict(row) for row in violation_rows],
            "severity_breakdown": {
                "critical": sum(row["critical_violations"] for row in violation_rows),
                "high": sum(row["high_violations"] for row in violation_rows),
                "medium": sum(row["medium_violations"] for row in violation_rows),
                "low": sum(row["low_violations"] for row in violation_rows)
            }
        }
    
    async def _generate_override_report(self, conn, tenant_id: int, framework: str,
                                      period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate override analysis report"""
        overrides_query = """
            SELECT 
                override_id,
                override_type,
                override_category,
                risk_level,
                financial_impact_usd,
                approval_status,
                created_at,
                approved_at
            FROM override_ledger_entry
            WHERE tenant_id = $1 
            AND compliance_framework = $2
            AND created_at BETWEEN $3 AND $4
            ORDER BY created_at DESC
        """
        
        override_rows = await conn.fetch(overrides_query, tenant_id, framework, period_start, period_end)
        
        return {
            "total_overrides": len(override_rows),
            "overrides": [dict(row) for row in override_rows],
            "total_financial_impact": sum(row["financial_impact_usd"] or 0 for row in override_rows),
            "approval_rate": len([r for r in override_rows if r["approval_status"] == "approved"]) / max(1, len(override_rows)) * 100
        }

@router.get("/compliance-overview", response_model=Dict[str, Any])
async def get_compliance_overview(
    framework: ComplianceFramework,
    period_days: int = Query(30, description="Period in days"),
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.5.2: Get compliance overview for regulator
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        dashboard_service = RegulatorDashboardService(pool_manager)
        overview = await dashboard_service.get_compliance_overview(tenant_id, framework.value, period_days)
        
        return {
            "status": "success",
            "compliance_overview": overview,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get compliance overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance overview: {str(e)}")

@router.get("/violations", response_model=Dict[str, Any])
async def get_violation_details(
    framework: ComplianceFramework,
    severity: Optional[str] = Query(None, description="Filter by severity: critical, high"),
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.5.3: Get detailed violation information
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        dashboard_service = RegulatorDashboardService(pool_manager)
        violations = await dashboard_service.get_violation_details(tenant_id, framework.value, severity)
        
        return {
            "status": "success",
            "violations": violations,
            "total_violations": len(violations),
            "framework": framework.value,
            "severity_filter": severity,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get violation details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get violation details: {str(e)}")

@router.get("/override-analysis", response_model=Dict[str, Any])
async def get_override_analysis(
    framework: ComplianceFramework,
    period_days: int = Query(30, description="Analysis period in days"),
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.5.4: Get override analysis for regulatory review
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        dashboard_service = RegulatorDashboardService(pool_manager)
        analysis = await dashboard_service.get_override_analysis(tenant_id, framework.value, period_days)
        
        return {
            "status": "success",
            "override_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get override analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get override analysis: {str(e)}")

@router.post("/reports/generate", response_model=Dict[str, Any])
async def generate_regulatory_report(
    framework: ComplianceFramework,
    report_type: ReportType,
    period_start: datetime,
    period_end: datetime,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.5.5: Generate regulatory compliance report
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        dashboard_service = RegulatorDashboardService(pool_manager)
        report = await dashboard_service.generate_regulatory_report(
            tenant_id, framework.value, report_type.value, period_start, period_end
        )
        
        return {
            "status": "success",
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate regulatory report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate regulatory report: {str(e)}")

@router.get("/frameworks", response_model=Dict[str, Any])
async def get_supported_frameworks(
    current_user: dict = Depends(get_current_user)
):
    """
    Get supported compliance frameworks for regulator dashboard
    """
    try:
        frameworks = {
            "SOX": {
                "name": "Sarbanes-Oxley Act",
                "regulator": "SEC",
                "jurisdiction": "US",
                "reporting_frequency": "quarterly",
                "key_requirements": ["Internal Controls", "Financial Reporting", "Audit Trail"]
            },
            "GDPR": {
                "name": "General Data Protection Regulation",
                "regulator": "Data Protection Authorities",
                "jurisdiction": "EU",
                "reporting_frequency": "incident_based",
                "key_requirements": ["Consent Management", "Data Minimization", "Breach Notification"]
            },
            "RBI": {
                "name": "Reserve Bank of India Guidelines",
                "regulator": "RBI",
                "jurisdiction": "India",
                "reporting_frequency": "monthly",
                "key_requirements": ["Risk Management", "Operational Resilience", "Customer Protection"]
            },
            "HIPAA": {
                "name": "Health Insurance Portability and Accountability Act",
                "regulator": "HHS",
                "jurisdiction": "US",
                "reporting_frequency": "incident_based",
                "key_requirements": ["Privacy Rule", "Security Rule", "Breach Notification"]
            },
            "NAIC": {
                "name": "National Association of Insurance Commissioners",
                "regulator": "State Insurance Commissioners",
                "jurisdiction": "US",
                "reporting_frequency": "quarterly",
                "key_requirements": ["Solvency", "Market Conduct", "Consumer Protection"]
            }
        }
        
        return {
            "status": "success",
            "supported_frameworks": frameworks,
            "framework_count": len(frameworks),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get supported frameworks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported frameworks: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def regulator_dashboard_health(
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Health check for regulator dashboard service
    """
    try:
        health_status = {
            "service": "regulator_dashboard",
            "status": "healthy",
            "database_connected": pool_manager is not None,
            "features": {
                "compliance_overview": True,
                "violation_tracking": True,
                "override_analysis": True,
                "regulatory_reporting": True,
                "multi_framework_support": True
            },
            "supported_frameworks": [framework.value for framework in ComplianceFramework],
            "supported_reports": [report.value for report in ReportType],
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "service": "regulator_dashboard",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
