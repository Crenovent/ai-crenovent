"""
Task 8.5-T08: Implement compliance reporting
Automated generation of compliance reports for regulatory frameworks
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import io
import base64

import asyncpg

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of compliance reports"""
    ASSESSMENT_SUMMARY = "assessment_summary"
    DETAILED_FINDINGS = "detailed_findings"
    GAP_ANALYSIS = "gap_analysis"
    REMEDIATION_PLAN = "remediation_plan"
    EXECUTIVE_DASHBOARD = "executive_dashboard"
    REGULATORY_SUBMISSION = "regulatory_submission"
    AUDIT_TRAIL = "audit_trail"
    RISK_REGISTER = "risk_register"


class ReportFormat(Enum):
    """Report output formats"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"


class ReportStatus(Enum):
    """Report generation status"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    RBI = "rbi"
    IRDAI = "irdai"
    PCI_DSS = "pci_dss"
    BASEL_III = "basel_iii"
    DPDP = "dpdp"
    CCPA = "ccpa"
    ISO_27001 = "iso_27001"


@dataclass
class ReportRequest:
    """Compliance report generation request"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Report configuration
    report_type: ReportType = ReportType.ASSESSMENT_SUMMARY
    report_format: ReportFormat = ReportFormat.PDF
    framework: ComplianceFramework = ComplianceFramework.SOX
    
    # Scope and filters
    tenant_id: int = 0
    object_ids: List[str] = field(default_factory=list)  # Specific objects to include
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    
    # Report options
    include_executive_summary: bool = True
    include_detailed_findings: bool = True
    include_recommendations: bool = True
    include_charts: bool = True
    include_raw_data: bool = False
    
    # Delivery options
    requested_by: str = "system"
    delivery_email: Optional[str] = None
    auto_schedule: bool = False
    schedule_frequency: Optional[str] = None  # daily, weekly, monthly
    
    # Timestamps
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "report_type": self.report_type.value,
            "report_format": self.report_format.value,
            "framework": self.framework.value,
            "tenant_id": self.tenant_id,
            "object_ids": self.object_ids,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "include_executive_summary": self.include_executive_summary,
            "include_detailed_findings": self.include_detailed_findings,
            "include_recommendations": self.include_recommendations,
            "include_charts": self.include_charts,
            "include_raw_data": self.include_raw_data,
            "requested_by": self.requested_by,
            "delivery_email": self.delivery_email,
            "auto_schedule": self.auto_schedule,
            "schedule_frequency": self.schedule_frequency,
            "requested_at": self.requested_at.isoformat()
        }


@dataclass
class ComplianceReport:
    """Generated compliance report"""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    
    # Report metadata
    report_type: ReportType = ReportType.ASSESSMENT_SUMMARY
    report_format: ReportFormat = ReportFormat.PDF
    framework: ComplianceFramework = ComplianceFramework.SOX
    
    # Report content
    title: str = ""
    executive_summary: str = ""
    content_sections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Report data
    report_data: Dict[str, Any] = field(default_factory=dict)
    charts_data: List[Dict[str, Any]] = field(default_factory=list)
    
    # File information
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    file_hash: Optional[str] = None
    
    # Status and metadata
    status: ReportStatus = ReportStatus.PENDING
    tenant_id: int = 0
    generated_by: str = "system"
    
    # Statistics
    total_assessments: int = 0
    compliant_percentage: float = 0.0
    critical_gaps: int = 0
    
    # Timestamps
    generation_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generation_completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "request_id": self.request_id,
            "report_type": self.report_type.value,
            "report_format": self.report_format.value,
            "framework": self.framework.value,
            "title": self.title,
            "executive_summary": self.executive_summary,
            "content_sections": self.content_sections,
            "report_data": self.report_data,
            "charts_data": self.charts_data,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "generated_by": self.generated_by,
            "total_assessments": self.total_assessments,
            "compliant_percentage": self.compliant_percentage,
            "critical_gaps": self.critical_gaps,
            "generation_started_at": self.generation_started_at.isoformat(),
            "generation_completed_at": self.generation_completed_at.isoformat() if self.generation_completed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


class ComplianceReportingEngine:
    """
    Compliance Reporting Engine - Task 8.5-T08
    
    Automated generation of compliance reports for regulatory frameworks
    Supports multiple report types and formats for different stakeholders
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'enable_automated_reporting': True,
            'report_generation_timeout_minutes': 30,
            'max_concurrent_reports': 5,
            'report_retention_days': 90,
            'enable_report_scheduling': True,
            'default_report_expiry_days': 30,
            'enable_email_delivery': False,  # Would require email service integration
            'report_storage_path': '/tmp/compliance_reports'
        }
        
        # Report generation queue
        self.report_queue: List[ReportRequest] = []
        self.active_reports: Dict[str, ComplianceReport] = {}
        self.queue_lock = asyncio.Lock()
        
        # Statistics
        self.reporting_stats = {
            'total_reports_generated': 0,
            'reports_successful': 0,
            'reports_failed': 0,
            'average_generation_time_ms': 0.0,
            'total_file_size_mb': 0.0,
            'most_requested_report_type': None
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Report templates
        self.report_templates = self._initialize_report_templates()
    
    async def initialize(self) -> bool:
        """Initialize compliance reporting engine"""
        try:
            await self._create_compliance_reporting_tables()
            if self.config['enable_automated_reporting']:
                await self._start_background_tasks()
            self.is_running = True
            self.logger.info("✅ Compliance reporting engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize compliance reporting engine: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown reporting engine"""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("🛑 Compliance reporting engine shutdown")
    
    async def generate_compliance_report(
        self,
        request: ReportRequest
    ) -> ComplianceReport:
        """
        Generate compliance report based on request
        
        This is the main entry point for report generation
        """
        
        try:
            # Validate request
            await self._validate_report_request(request)
            
            # Create report object
            report = ComplianceReport(
                request_id=request.request_id,
                report_type=request.report_type,
                report_format=request.report_format,
                framework=request.framework,
                tenant_id=request.tenant_id,
                expires_at=datetime.now(timezone.utc) + timedelta(days=self.config['default_report_expiry_days'])
            )
            
            # Add to active reports
            self.active_reports[report.report_id] = report
            
            # Generate report content based on type
            if request.report_type == ReportType.ASSESSMENT_SUMMARY:
                await self._generate_assessment_summary_report(request, report)
            elif request.report_type == ReportType.DETAILED_FINDINGS:
                await self._generate_detailed_findings_report(request, report)
            elif request.report_type == ReportType.GAP_ANALYSIS:
                await self._generate_gap_analysis_report(request, report)
            elif request.report_type == ReportType.REMEDIATION_PLAN:
                await self._generate_remediation_plan_report(request, report)
            elif request.report_type == ReportType.EXECUTIVE_DASHBOARD:
                await self._generate_executive_dashboard_report(request, report)
            elif request.report_type == ReportType.REGULATORY_SUBMISSION:
                await self._generate_regulatory_submission_report(request, report)
            elif request.report_type == ReportType.AUDIT_TRAIL:
                await self._generate_audit_trail_report(request, report)
            elif request.report_type == ReportType.RISK_REGISTER:
                await self._generate_risk_register_report(request, report)
            else:
                raise ValueError(f"Unsupported report type: {request.report_type}")
            
            # Generate file output
            await self._generate_report_file(report, request)
            
            # Complete report
            report.status = ReportStatus.COMPLETED
            report.generation_completed_at = datetime.now(timezone.utc)
            
            # Update statistics
            self._update_reporting_stats(report)
            
            # Store report
            if self.db_pool:
                await self._store_report(report)
            
            # Remove from active reports
            self.active_reports.pop(report.report_id, None)
            
            self.logger.info(f"✅ Generated {request.report_type.value} report: {report.report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate compliance report: {e}")
            
            # Mark report as failed
            if 'report' in locals():
                report.status = ReportStatus.FAILED
                report.generation_completed_at = datetime.now(timezone.utc)
                self.active_reports.pop(report.report_id, None)
                return report
            
            # Create failed report
            failed_report = ComplianceReport(
                request_id=request.request_id,
                report_type=request.report_type,
                report_format=request.report_format,
                framework=request.framework,
                tenant_id=request.tenant_id,
                status=ReportStatus.FAILED,
                generation_completed_at=datetime.now(timezone.utc)
            )
            failed_report.executive_summary = f"Report generation failed: {str(e)}"
            
            return failed_report
    
    async def schedule_recurring_report(
        self,
        request: ReportRequest,
        frequency: str = "monthly"  # daily, weekly, monthly
    ) -> str:
        """Schedule recurring report generation"""
        
        if not self.config['enable_report_scheduling']:
            raise ValueError("Report scheduling is disabled")
        
        request.auto_schedule = True
        request.schedule_frequency = frequency
        
        # Store scheduled report request
        if self.db_pool:
            await self._store_scheduled_report(request)
        
        self.logger.info(f"📅 Scheduled {frequency} {request.report_type.value} report for tenant {request.tenant_id}")
        return request.request_id
    
    async def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get status of report generation"""
        
        # Check active reports
        if report_id in self.active_reports:
            report = self.active_reports[report_id]
            return {
                "report_id": report_id,
                "status": report.status.value,
                "progress": "generating",
                "started_at": report.generation_started_at.isoformat()
            }
        
        # Check database for completed reports
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT status, generation_completed_at FROM compliance_reports WHERE report_id = $1",
                        report_id
                    )
                    
                    if row:
                        return {
                            "report_id": report_id,
                            "status": row['status'],
                            "completed_at": row['generation_completed_at'].isoformat() if row['generation_completed_at'] else None
                        }
            except Exception as e:
                self.logger.error(f"Error checking report status: {e}")
        
        return {
            "report_id": report_id,
            "status": "not_found"
        }
    
    async def _validate_report_request(self, request: ReportRequest):
        """Validate report generation request"""
        
        if not request.tenant_id:
            raise ValueError("Tenant ID is required")
        
        if request.date_range_start and request.date_range_end:
            if request.date_range_start >= request.date_range_end:
                raise ValueError("Invalid date range")
        
        # Check concurrent report limit
        if len(self.active_reports) >= self.config['max_concurrent_reports']:
            raise ValueError("Maximum concurrent reports limit reached")
    
    async def _generate_assessment_summary_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate assessment summary report"""
        
        report.title = f"{request.framework.value.upper()} Compliance Assessment Summary"
        
        # Get assessment data
        assessment_data = await self._get_assessment_data(
            request.tenant_id, request.framework, request.date_range_start, request.date_range_end
        )
        
        # Calculate summary statistics
        total_assessments = len(assessment_data)
        compliant_count = sum(1 for a in assessment_data if a.get('overall_status') == 'compliant')
        compliance_percentage = (compliant_count / total_assessments * 100) if total_assessments > 0 else 0
        
        critical_gaps = sum(a.get('critical_gaps', 0) for a in assessment_data)
        high_risk_gaps = sum(a.get('high_risk_gaps', 0) for a in assessment_data)
        
        # Update report statistics
        report.total_assessments = total_assessments
        report.compliant_percentage = compliance_percentage
        report.critical_gaps = critical_gaps
        
        # Generate executive summary
        report.executive_summary = f"""
        This report provides a comprehensive summary of {request.framework.value.upper()} compliance assessments 
        for the period {request.date_range_start or 'inception'} to {request.date_range_end or 'present'}.
        
        Key Findings:
        • Total Assessments: {total_assessments}
        • Compliance Rate: {compliance_percentage:.1f}%
        • Critical Gaps: {critical_gaps}
        • High-Risk Gaps: {high_risk_gaps}
        
        {"✅ Overall compliance status is GOOD" if compliance_percentage >= 80 else "⚠️ Compliance gaps require immediate attention"}
        """
        
        # Generate content sections
        report.content_sections = [
            {
                "title": "Executive Summary",
                "content": report.executive_summary,
                "type": "text"
            },
            {
                "title": "Compliance Overview",
                "content": {
                    "total_assessments": total_assessments,
                    "compliant_assessments": compliant_count,
                    "compliance_percentage": compliance_percentage,
                    "critical_gaps": critical_gaps,
                    "high_risk_gaps": high_risk_gaps
                },
                "type": "metrics"
            },
            {
                "title": "Assessment Details",
                "content": assessment_data[:10],  # Top 10 assessments
                "type": "table"
            }
        ]
        
        # Generate charts data
        if request.include_charts:
            report.charts_data = [
                {
                    "title": "Compliance Status Distribution",
                    "type": "pie",
                    "data": {
                        "compliant": compliant_count,
                        "non_compliant": total_assessments - compliant_count
                    }
                },
                {
                    "title": "Gap Severity Distribution",
                    "type": "bar",
                    "data": {
                        "critical": critical_gaps,
                        "high": high_risk_gaps,
                        "medium": sum(a.get('medium_risk_gaps', 0) for a in assessment_data),
                        "low": sum(a.get('low_risk_gaps', 0) for a in assessment_data)
                    }
                }
            ]
        
        # Store raw data if requested
        if request.include_raw_data:
            report.report_data = {
                "assessments": assessment_data,
                "summary_statistics": {
                    "total_assessments": total_assessments,
                    "compliance_percentage": compliance_percentage,
                    "critical_gaps": critical_gaps,
                    "high_risk_gaps": high_risk_gaps
                }
            }
    
    async def _generate_detailed_findings_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate detailed findings report"""
        
        report.title = f"{request.framework.value.upper()} Detailed Compliance Findings"
        
        # Get detailed assessment data with control-level findings
        detailed_data = await self._get_detailed_assessment_data(
            request.tenant_id, request.framework, request.object_ids
        )
        
        # Generate executive summary
        total_controls = sum(len(a.get('control_assessments', [])) for a in detailed_data)
        non_compliant_controls = sum(
            len([c for c in a.get('control_assessments', []) if c.get('status') != 'compliant'])
            for a in detailed_data
        )
        
        report.executive_summary = f"""
        Detailed compliance findings for {len(detailed_data)} assessments covering {total_controls} controls.
        
        Control-Level Analysis:
        • Total Controls Evaluated: {total_controls}
        • Non-Compliant Controls: {non_compliant_controls}
        • Control Compliance Rate: {((total_controls - non_compliant_controls) / total_controls * 100) if total_controls > 0 else 0:.1f}%
        """
        
        # Generate detailed content sections
        report.content_sections = []
        
        for assessment in detailed_data:
            assessment_section = {
                "title": f"Assessment: {assessment.get('object_id', 'Unknown')}",
                "content": {
                    "overall_status": assessment.get('overall_status'),
                    "compliance_percentage": assessment.get('compliance_percentage'),
                    "control_details": assessment.get('control_assessments', [])
                },
                "type": "detailed_assessment"
            }
            report.content_sections.append(assessment_section)
        
        report.report_data = {"detailed_assessments": detailed_data}
    
    async def _generate_gap_analysis_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate gap analysis report"""
        
        report.title = f"{request.framework.value.upper()} Compliance Gap Analysis"
        
        # Get gap analysis data
        gap_data = await self._get_gap_analysis_data(
            request.tenant_id, request.framework
        )
        
        # Analyze gaps by severity and control
        gaps_by_severity = {}
        gaps_by_control = {}
        
        for gap in gap_data:
            severity = gap.get('severity', 'medium')
            control_id = gap.get('control_id', 'unknown')
            
            gaps_by_severity[severity] = gaps_by_severity.get(severity, 0) + 1
            gaps_by_control[control_id] = gaps_by_control.get(control_id, 0) + 1
        
        report.executive_summary = f"""
        Gap analysis identifies {len(gap_data)} compliance gaps requiring remediation.
        
        Gap Distribution:
        • Critical: {gaps_by_severity.get('critical', 0)}
        • High: {gaps_by_severity.get('high', 0)}
        • Medium: {gaps_by_severity.get('medium', 0)}
        • Low: {gaps_by_severity.get('low', 0)}
        """
        
        report.content_sections = [
            {
                "title": "Gap Summary",
                "content": gaps_by_severity,
                "type": "metrics"
            },
            {
                "title": "Gaps by Control",
                "content": gaps_by_control,
                "type": "table"
            },
            {
                "title": "Detailed Gap Analysis",
                "content": gap_data,
                "type": "gap_details"
            }
        ]
    
    async def _generate_remediation_plan_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate remediation plan report"""
        
        report.title = f"{request.framework.value.upper()} Compliance Remediation Plan"
        
        # Get remediation data
        remediation_data = await self._get_remediation_data(
            request.tenant_id, request.framework
        )
        
        # Prioritize remediation actions
        critical_actions = [r for r in remediation_data if r.get('priority') == 'critical']
        high_actions = [r for r in remediation_data if r.get('priority') == 'high']
        
        report.executive_summary = f"""
        Remediation plan for addressing {len(remediation_data)} compliance gaps.
        
        Immediate Actions Required:
        • Critical Priority: {len(critical_actions)} actions
        • High Priority: {len(high_actions)} actions
        
        Estimated Timeline: {self._calculate_remediation_timeline(remediation_data)}
        """
        
        report.content_sections = [
            {
                "title": "Remediation Overview",
                "content": {
                    "total_actions": len(remediation_data),
                    "critical_actions": len(critical_actions),
                    "high_actions": len(high_actions)
                },
                "type": "metrics"
            },
            {
                "title": "Priority Actions",
                "content": critical_actions + high_actions,
                "type": "action_plan"
            },
            {
                "title": "Implementation Timeline",
                "content": self._generate_timeline(remediation_data),
                "type": "timeline"
            }
        ]
    
    async def _generate_executive_dashboard_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate executive dashboard report"""
        
        report.title = f"Executive Compliance Dashboard - {request.framework.value.upper()}"
        
        # Get executive-level metrics
        exec_data = await self._get_executive_metrics(
            request.tenant_id, request.framework
        )
        
        report.executive_summary = f"""
        Executive overview of compliance posture across all {request.framework.value.upper()} requirements.
        
        Key Metrics:
        • Overall Compliance Score: {exec_data.get('compliance_score', 0):.1f}%
        • Risk Exposure: {exec_data.get('risk_level', 'Medium')}
        • Trend: {exec_data.get('trend', 'Stable')}
        """
        
        # Generate executive-focused content
        report.content_sections = [
            {
                "title": "Compliance Scorecard",
                "content": exec_data.get('scorecard', {}),
                "type": "scorecard"
            },
            {
                "title": "Risk Heat Map",
                "content": exec_data.get('risk_heatmap', {}),
                "type": "heatmap"
            },
            {
                "title": "Trend Analysis",
                "content": exec_data.get('trends', {}),
                "type": "trends"
            }
        ]
    
    async def _generate_regulatory_submission_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate regulatory submission report"""
        
        report.title = f"Regulatory Submission - {request.framework.value.upper()}"
        
        # Get regulatory-specific data
        regulatory_data = await self._get_regulatory_submission_data(
            request.tenant_id, request.framework
        )
        
        report.executive_summary = "Formal compliance report prepared for regulatory submission."
        
        # Generate regulatory-compliant content
        report.content_sections = [
            {
                "title": "Regulatory Declaration",
                "content": self._generate_regulatory_declaration(request.framework),
                "type": "declaration"
            },
            {
                "title": "Compliance Evidence",
                "content": regulatory_data.get('evidence', []),
                "type": "evidence"
            },
            {
                "title": "Attestations",
                "content": regulatory_data.get('attestations', []),
                "type": "attestations"
            }
        ]
    
    async def _generate_audit_trail_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate audit trail report"""
        
        report.title = f"Compliance Audit Trail - {request.framework.value.upper()}"
        
        # Get audit trail data
        audit_data = await self._get_audit_trail_data(
            request.tenant_id, request.date_range_start, request.date_range_end
        )
        
        report.executive_summary = f"""
        Comprehensive audit trail covering {len(audit_data)} compliance-related events.
        
        Audit Summary:
        • Total Events: {len(audit_data)}
        • Time Period: {request.date_range_start} to {request.date_range_end}
        • Event Types: {len(set(e.get('event_type') for e in audit_data))}
        """
        
        report.content_sections = [
            {
                "title": "Audit Event Summary",
                "content": self._summarize_audit_events(audit_data),
                "type": "audit_summary"
            },
            {
                "title": "Detailed Audit Log",
                "content": audit_data,
                "type": "audit_log"
            }
        ]
    
    async def _generate_risk_register_report(
        self, request: ReportRequest, report: ComplianceReport
    ):
        """Generate risk register report"""
        
        report.title = f"Compliance Risk Register - {request.framework.value.upper()}"
        
        # Get risk data
        risk_data = await self._get_risk_register_data(
            request.tenant_id, request.framework
        )
        
        # Analyze risk distribution
        risks_by_level = {}
        for risk in risk_data:
            level = risk.get('risk_level', 'medium')
            risks_by_level[level] = risks_by_level.get(level, 0) + 1
        
        report.executive_summary = f"""
        Risk register identifies {len(risk_data)} compliance-related risks.
        
        Risk Distribution:
        • Critical: {risks_by_level.get('critical', 0)}
        • High: {risks_by_level.get('high', 0)}
        • Medium: {risks_by_level.get('medium', 0)}
        • Low: {risks_by_level.get('low', 0)}
        """
        
        report.content_sections = [
            {
                "title": "Risk Overview",
                "content": risks_by_level,
                "type": "risk_summary"
            },
            {
                "title": "Risk Register",
                "content": risk_data,
                "type": "risk_register"
            }
        ]
    
    async def _generate_report_file(self, report: ComplianceReport, request: ReportRequest):
        """Generate report file in requested format"""
        
        try:
            if request.report_format == ReportFormat.JSON:
                await self._generate_json_report(report)
            elif request.report_format == ReportFormat.HTML:
                await self._generate_html_report(report)
            elif request.report_format == ReportFormat.CSV:
                await self._generate_csv_report(report)
            elif request.report_format == ReportFormat.PDF:
                await self._generate_pdf_report(report)
            elif request.report_format == ReportFormat.XLSX:
                await self._generate_xlsx_report(report)
            else:
                raise ValueError(f"Unsupported report format: {request.report_format}")
                
        except Exception as e:
            self.logger.error(f"Error generating report file: {e}")
            raise
    
    async def _generate_json_report(self, report: ComplianceReport):
        """Generate JSON format report"""
        
        json_content = report.to_dict()
        
        # Save to file
        file_path = f"{self.config['report_storage_path']}/{report.report_id}.json"
        
        # In production, this would save to actual file system or cloud storage
        report.file_path = file_path
        report.file_size_bytes = len(json.dumps(json_content))
        report.file_hash = "mock_hash_" + report.report_id[:8]
    
    async def _generate_html_report(self, report: ComplianceReport):
        """Generate HTML format report"""
        
        html_template = self.report_templates.get('html_template', '')
        
        # Generate HTML content
        html_content = html_template.format(
            title=report.title,
            executive_summary=report.executive_summary,
            content_sections=self._render_html_sections(report.content_sections),
            generated_at=report.generation_started_at.strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        
        # Save to file
        file_path = f"{self.config['report_storage_path']}/{report.report_id}.html"
        
        report.file_path = file_path
        report.file_size_bytes = len(html_content.encode('utf-8'))
        report.file_hash = "mock_hash_" + report.report_id[:8]
    
    async def _generate_csv_report(self, report: ComplianceReport):
        """Generate CSV format report"""
        
        # Convert report data to CSV format
        csv_content = self._convert_to_csv(report.report_data)
        
        # Save to file
        file_path = f"{self.config['report_storage_path']}/{report.report_id}.csv"
        
        report.file_path = file_path
        report.file_size_bytes = len(csv_content.encode('utf-8'))
        report.file_hash = "mock_hash_" + report.report_id[:8]
    
    async def _generate_pdf_report(self, report: ComplianceReport):
        """Generate PDF format report"""
        
        # In production, this would use a PDF generation library like ReportLab
        pdf_content = f"PDF Report: {report.title}\n\n{report.executive_summary}"
        
        # Save to file
        file_path = f"{self.config['report_storage_path']}/{report.report_id}.pdf"
        
        report.file_path = file_path
        report.file_size_bytes = len(pdf_content.encode('utf-8'))
        report.file_hash = "mock_hash_" + report.report_id[:8]
    
    async def _generate_xlsx_report(self, report: ComplianceReport):
        """Generate Excel format report"""
        
        # In production, this would use openpyxl or xlsxwriter
        xlsx_content = f"Excel Report: {report.title}"
        
        # Save to file
        file_path = f"{self.config['report_storage_path']}/{report.report_id}.xlsx"
        
        report.file_path = file_path
        report.file_size_bytes = len(xlsx_content.encode('utf-8'))
        report.file_hash = "mock_hash_" + report.report_id[:8]
    
    # Helper methods for data retrieval (would integrate with actual data sources)
    
    async def _get_assessment_data(
        self, tenant_id: int, framework: ComplianceFramework, 
        start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Get assessment data from database"""
        
        # Mock data - in production would query compliance_assessments table
        return [
            {
                "assessment_id": "assess_001",
                "object_id": "workflow_001",
                "overall_status": "compliant",
                "compliance_percentage": 85.0,
                "critical_gaps": 0,
                "high_risk_gaps": 1,
                "medium_risk_gaps": 2,
                "low_risk_gaps": 1
            },
            {
                "assessment_id": "assess_002",
                "object_id": "workflow_002",
                "overall_status": "partially_compliant",
                "compliance_percentage": 65.0,
                "critical_gaps": 1,
                "high_risk_gaps": 2,
                "medium_risk_gaps": 1,
                "low_risk_gaps": 0
            }
        ]
    
    async def _get_detailed_assessment_data(
        self, tenant_id: int, framework: ComplianceFramework, object_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get detailed assessment data with control-level information"""
        
        # Mock detailed data
        return [
            {
                "assessment_id": "assess_001",
                "object_id": "workflow_001",
                "overall_status": "compliant",
                "compliance_percentage": 85.0,
                "control_assessments": [
                    {
                        "control_id": "SOX-001",
                        "status": "compliant",
                        "score": 90,
                        "findings": ["Proper segregation of duties implemented"]
                    },
                    {
                        "control_id": "SOX-002",
                        "status": "partially_compliant",
                        "score": 70,
                        "findings": ["Audit trail partially configured"]
                    }
                ]
            }
        ]
    
    async def _get_gap_analysis_data(
        self, tenant_id: int, framework: ComplianceFramework
    ) -> List[Dict[str, Any]]:
        """Get gap analysis data"""
        
        return [
            {
                "gap_id": "gap_001",
                "control_id": "SOX-002",
                "severity": "high",
                "description": "Incomplete audit trail configuration",
                "impact": "Regulatory compliance risk"
            }
        ]
    
    async def _get_remediation_data(
        self, tenant_id: int, framework: ComplianceFramework
    ) -> List[Dict[str, Any]]:
        """Get remediation action data"""
        
        return [
            {
                "action_id": "action_001",
                "priority": "critical",
                "description": "Enable comprehensive audit logging",
                "estimated_effort": "2 weeks",
                "owner": "IT Team"
            }
        ]
    
    async def _get_executive_metrics(
        self, tenant_id: int, framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Get executive-level compliance metrics"""
        
        return {
            "compliance_score": 78.5,
            "risk_level": "Medium",
            "trend": "Improving",
            "scorecard": {
                "controls_compliant": 15,
                "controls_total": 20,
                "assessments_passed": 8,
                "assessments_total": 10
            },
            "risk_heatmap": {
                "critical": 1,
                "high": 3,
                "medium": 5,
                "low": 2
            },
            "trends": {
                "last_month": 75.0,
                "current_month": 78.5,
                "direction": "up"
            }
        }
    
    async def _get_regulatory_submission_data(
        self, tenant_id: int, framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Get regulatory submission data"""
        
        return {
            "evidence": [
                {
                    "evidence_id": "ev_001",
                    "type": "audit_log",
                    "description": "Comprehensive audit trail evidence"
                }
            ],
            "attestations": [
                {
                    "attestation_id": "att_001",
                    "signatory": "Chief Compliance Officer",
                    "statement": "I attest that the compliance controls are operating effectively"
                }
            ]
        }
    
    async def _get_audit_trail_data(
        self, tenant_id: int, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Get audit trail data"""
        
        return [
            {
                "event_id": "evt_001",
                "event_type": "compliance_assessment",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "actor": "system",
                "description": "Automated compliance assessment completed"
            }
        ]
    
    async def _get_risk_register_data(
        self, tenant_id: int, framework: ComplianceFramework
    ) -> List[Dict[str, Any]]:
        """Get risk register data"""
        
        return [
            {
                "risk_id": "risk_001",
                "risk_level": "high",
                "description": "Regulatory compliance violation risk",
                "mitigation": "Implement additional controls",
                "owner": "Compliance Team"
            }
        ]
    
    # Utility methods
    
    def _initialize_report_templates(self) -> Dict[str, str]:
        """Initialize report templates"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; }}
                .section {{ margin: 20px 0; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated: {generated_at}</p>
            </div>
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>{executive_summary}</p>
            </div>
            <div class="content">
                {content_sections}
            </div>
        </body>
        </html>
        """
        
        return {
            "html_template": html_template
        }
    
    def _render_html_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Render content sections as HTML"""
        
        html = ""
        for section in sections:
            html += f"<div class='section'><h3>{section['title']}</h3>"
            
            if section['type'] == 'text':
                html += f"<p>{section['content']}</p>"
            elif section['type'] == 'metrics':
                html += "<ul>"
                for key, value in section['content'].items():
                    html += f"<li>{key}: {value}</li>"
                html += "</ul>"
            elif section['type'] == 'table':
                html += "<table border='1'><tr><th>Item</th><th>Details</th></tr>"
                for item in section['content']:
                    html += f"<tr><td>{item.get('id', 'N/A')}</td><td>{item.get('status', 'N/A')}</td></tr>"
                html += "</table>"
            
            html += "</div>"
        
        return html
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert report data to CSV format"""
        
        # Simple CSV conversion - in production would use proper CSV library
        csv_lines = ["Field,Value"]
        
        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    yield from flatten_dict(value, f"{prefix}{key}.")
                else:
                    yield f"{prefix}{key},{value}"
        
        csv_lines.extend(flatten_dict(data))
        return "\n".join(csv_lines)
    
    def _calculate_remediation_timeline(self, remediation_data: List[Dict[str, Any]]) -> str:
        """Calculate estimated remediation timeline"""
        
        critical_count = sum(1 for r in remediation_data if r.get('priority') == 'critical')
        high_count = sum(1 for r in remediation_data if r.get('priority') == 'high')
        
        if critical_count > 0:
            return f"{critical_count * 2} weeks (critical items)"
        elif high_count > 0:
            return f"{high_count * 1} weeks (high priority items)"
        else:
            return "4-6 weeks (standard timeline)"
    
    def _generate_timeline(self, remediation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate implementation timeline"""
        
        timeline = []
        week = 1
        
        for action in remediation_data:
            if action.get('priority') == 'critical':
                timeline.append({
                    "week": week,
                    "action": action.get('description'),
                    "owner": action.get('owner')
                })
                week += 2
        
        return timeline
    
    def _generate_regulatory_declaration(self, framework: ComplianceFramework) -> str:
        """Generate regulatory declaration text"""
        
        declarations = {
            ComplianceFramework.SOX: "This report certifies compliance with Sarbanes-Oxley Act requirements for internal controls over financial reporting.",
            ComplianceFramework.GDPR: "This report demonstrates compliance with General Data Protection Regulation requirements for personal data processing.",
            ComplianceFramework.HIPAA: "This report validates compliance with Health Insurance Portability and Accountability Act requirements for protected health information.",
            ComplianceFramework.RBI: "This report confirms adherence to Reserve Bank of India guidelines and regulatory requirements."
        }
        
        return declarations.get(framework, "This report demonstrates compliance with applicable regulatory requirements.")
    
    def _summarize_audit_events(self, audit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize audit events"""
        
        event_types = {}
        for event in audit_data:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": len(audit_data),
            "event_types": event_types,
            "time_span": f"{audit_data[0].get('timestamp')} to {audit_data[-1].get('timestamp')}" if audit_data else "No events"
        }
    
    def _update_reporting_stats(self, report: ComplianceReport):
        """Update reporting statistics"""
        
        self.reporting_stats['total_reports_generated'] += 1
        
        if report.status == ReportStatus.COMPLETED:
            self.reporting_stats['reports_successful'] += 1
        else:
            self.reporting_stats['reports_failed'] += 1
        
        # Update file size statistics
        if report.file_size_bytes > 0:
            self.reporting_stats['total_file_size_mb'] += report.file_size_bytes / (1024 * 1024)
        
        # Update generation time
        if report.generation_completed_at and report.generation_started_at:
            generation_time = (report.generation_completed_at - report.generation_started_at).total_seconds() * 1000
            current_avg = self.reporting_stats['average_generation_time_ms']
            total_reports = self.reporting_stats['total_reports_generated']
            self.reporting_stats['average_generation_time_ms'] = (
                (current_avg * (total_reports - 1) + generation_time) / total_reports
            )
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        
        # Report queue processor
        queue_task = asyncio.create_task(self._process_report_queue())
        self.background_tasks.append(queue_task)
        
        # Scheduled report processor
        schedule_task = asyncio.create_task(self._process_scheduled_reports())
        self.background_tasks.append(schedule_task)
        
        # Report cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_expired_reports())
        self.background_tasks.append(cleanup_task)
    
    async def _process_report_queue(self):
        """Background task for processing report generation queue"""
        
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                async with self.queue_lock:
                    if not self.report_queue:
                        continue
                    
                    # Process next report in queue
                    request = self.report_queue.pop(0)
                
                # Generate report
                await self.generate_compliance_report(request)
                
            except Exception as e:
                self.logger.error(f"❌ Report queue processor error: {e}")
    
    async def _process_scheduled_reports(self):
        """Background task for processing scheduled reports"""
        
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Get due scheduled reports from database
                # This would query scheduled_reports table in production
                
            except Exception as e:
                self.logger.error(f"❌ Scheduled report processor error: {e}")
    
    async def _cleanup_expired_reports(self):
        """Background task for cleaning up expired reports"""
        
        while self.is_running:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                # Clean up expired reports
                expiry_date = datetime.now(timezone.utc) - timedelta(days=self.config['report_retention_days'])
                
                if self.db_pool:
                    async with self.db_pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE compliance_reports SET status = 'expired' WHERE generation_completed_at < $1 AND status = 'completed'",
                            expiry_date
                        )
                
            except Exception as e:
                self.logger.error(f"❌ Report cleanup error: {e}")
    
    async def _create_compliance_reporting_tables(self):
        """Create compliance reporting database tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Compliance reports
        CREATE TABLE IF NOT EXISTS compliance_reports (
            report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            request_id UUID NOT NULL,
            
            -- Report metadata
            report_type VARCHAR(50) NOT NULL,
            report_format VARCHAR(20) NOT NULL,
            framework VARCHAR(50) NOT NULL,
            title VARCHAR(500),
            
            -- Report content
            executive_summary TEXT,
            content_sections JSONB DEFAULT '[]',
            report_data JSONB DEFAULT '{}',
            charts_data JSONB DEFAULT '[]',
            
            -- File information
            file_path VARCHAR(1000),
            file_size_bytes BIGINT DEFAULT 0,
            file_hash VARCHAR(128),
            
            -- Status and metadata
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            tenant_id INTEGER NOT NULL,
            generated_by VARCHAR(100) DEFAULT 'system',
            
            -- Statistics
            total_assessments INTEGER DEFAULT 0,
            compliant_percentage FLOAT DEFAULT 0,
            critical_gaps INTEGER DEFAULT 0,
            
            -- Timestamps
            generation_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            generation_completed_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            
            -- Constraints
            CONSTRAINT chk_report_type CHECK (report_type IN ('assessment_summary', 'detailed_findings', 'gap_analysis', 'remediation_plan', 'executive_dashboard', 'regulatory_submission', 'audit_trail', 'risk_register')),
            CONSTRAINT chk_report_format CHECK (report_format IN ('pdf', 'html', 'json', 'csv', 'xlsx')),
            CONSTRAINT chk_report_status CHECK (status IN ('pending', 'generating', 'completed', 'failed', 'expired'))
        );
        
        -- Report requests
        CREATE TABLE IF NOT EXISTS compliance_report_requests (
            request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            
            -- Request configuration
            report_type VARCHAR(50) NOT NULL,
            report_format VARCHAR(20) NOT NULL,
            framework VARCHAR(50) NOT NULL,
            
            -- Scope and filters
            tenant_id INTEGER NOT NULL,
            object_ids TEXT[] DEFAULT ARRAY[],
            date_range_start TIMESTAMPTZ,
            date_range_end TIMESTAMPTZ,
            
            -- Report options
            include_executive_summary BOOLEAN DEFAULT TRUE,
            include_detailed_findings BOOLEAN DEFAULT TRUE,
            include_recommendations BOOLEAN DEFAULT TRUE,
            include_charts BOOLEAN DEFAULT TRUE,
            include_raw_data BOOLEAN DEFAULT FALSE,
            
            -- Delivery options
            requested_by VARCHAR(100) NOT NULL,
            delivery_email VARCHAR(255),
            auto_schedule BOOLEAN DEFAULT FALSE,
            schedule_frequency VARCHAR(20),
            
            -- Timestamps
            requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_request_schedule_frequency CHECK (schedule_frequency IS NULL OR schedule_frequency IN ('daily', 'weekly', 'monthly'))
        );
        
        -- Scheduled reports
        CREATE TABLE IF NOT EXISTS compliance_scheduled_reports (
            schedule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            request_id UUID REFERENCES compliance_report_requests(request_id) ON DELETE CASCADE,
            
            -- Schedule configuration
            frequency VARCHAR(20) NOT NULL,
            next_run_at TIMESTAMPTZ NOT NULL,
            last_run_at TIMESTAMPTZ,
            
            -- Status
            is_active BOOLEAN DEFAULT TRUE,
            
            -- Timestamps
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_schedule_frequency CHECK (frequency IN ('daily', 'weekly', 'monthly'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_compliance_reports_tenant ON compliance_reports(tenant_id, generation_completed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compliance_reports_status ON compliance_reports(status, generation_started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compliance_reports_framework ON compliance_reports(framework, report_type);
        CREATE INDEX IF NOT EXISTS idx_compliance_reports_expires ON compliance_reports(expires_at) WHERE status = 'completed';
        
        CREATE INDEX IF NOT EXISTS idx_compliance_report_requests_tenant ON compliance_report_requests(tenant_id, requested_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compliance_report_requests_type ON compliance_report_requests(report_type, framework);
        
        CREATE INDEX IF NOT EXISTS idx_compliance_scheduled_reports_next_run ON compliance_scheduled_reports(next_run_at) WHERE is_active = TRUE;
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Compliance reporting tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create compliance reporting tables: {e}")
            raise
    
    async def _store_report(self, report: ComplianceReport):
        """Store generated report in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_reports (
                        report_id, request_id, report_type, report_format, framework,
                        title, executive_summary, content_sections, report_data,
                        charts_data, file_path, file_size_bytes, file_hash,
                        status, tenant_id, generated_by, total_assessments,
                        compliant_percentage, critical_gaps, generation_started_at,
                        generation_completed_at, expires_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
                """,
                report.report_id, report.request_id, report.report_type.value,
                report.report_format.value, report.framework.value, report.title,
                report.executive_summary, json.dumps(report.content_sections),
                json.dumps(report.report_data), json.dumps(report.charts_data),
                report.file_path, report.file_size_bytes, report.file_hash,
                report.status.value, report.tenant_id, report.generated_by,
                report.total_assessments, report.compliant_percentage,
                report.critical_gaps, report.generation_started_at,
                report.generation_completed_at, report.expires_at)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store compliance report: {e}")
    
    async def _store_scheduled_report(self, request: ReportRequest):
        """Store scheduled report request"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Store request
                await conn.execute("""
                    INSERT INTO compliance_report_requests (
                        request_id, report_type, report_format, framework, tenant_id,
                        object_ids, date_range_start, date_range_end,
                        include_executive_summary, include_detailed_findings,
                        include_recommendations, include_charts, include_raw_data,
                        requested_by, delivery_email, auto_schedule, schedule_frequency
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                request.request_id, request.report_type.value, request.report_format.value,
                request.framework.value, request.tenant_id, request.object_ids,
                request.date_range_start, request.date_range_end,
                request.include_executive_summary, request.include_detailed_findings,
                request.include_recommendations, request.include_charts,
                request.include_raw_data, request.requested_by, request.delivery_email,
                request.auto_schedule, request.schedule_frequency)
                
                # Calculate next run time
                if request.schedule_frequency == "daily":
                    next_run = datetime.now(timezone.utc) + timedelta(days=1)
                elif request.schedule_frequency == "weekly":
                    next_run = datetime.now(timezone.utc) + timedelta(weeks=1)
                elif request.schedule_frequency == "monthly":
                    next_run = datetime.now(timezone.utc) + timedelta(days=30)
                else:
                    next_run = datetime.now(timezone.utc) + timedelta(days=1)
                
                # Store schedule
                await conn.execute("""
                    INSERT INTO compliance_scheduled_reports (
                        request_id, frequency, next_run_at
                    ) VALUES ($1, $2, $3)
                """, request.request_id, request.schedule_frequency, next_run)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store scheduled report: {e}")
    
    def get_reporting_statistics(self) -> Dict[str, Any]:
        """Get reporting statistics"""
        
        success_rate = 0.0
        if self.reporting_stats['total_reports_generated'] > 0:
            success_rate = (
                self.reporting_stats['reports_successful'] / 
                self.reporting_stats['total_reports_generated']
            ) * 100
        
        return {
            'total_reports_generated': self.reporting_stats['total_reports_generated'],
            'reports_successful': self.reporting_stats['reports_successful'],
            'reports_failed': self.reporting_stats['reports_failed'],
            'success_rate_percentage': round(success_rate, 2),
            'average_generation_time_ms': round(self.reporting_stats['average_generation_time_ms'], 2),
            'total_file_size_mb': round(self.reporting_stats['total_file_size_mb'], 2),
            'active_reports_count': len(self.active_reports),
            'queue_size': len(self.report_queue),
            'automated_reporting_enabled': self.config['enable_automated_reporting'],
            'supported_formats': [f.value for f in ReportFormat],
            'supported_report_types': [t.value for t in ReportType]
        }


# Global compliance reporting engine instance
compliance_reporting_engine = ComplianceReportingEngine()
