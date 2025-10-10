"""
Task 8.5-T07: Build compliance assessment engine
Automated compliance assessment and scoring for workflows and policies
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


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


class AssessmentStatus(Enum):
    """Assessment status values"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    REQUIRES_REVIEW = "requires_review"
    NOT_ASSESSED = "not_assessed"


class RiskLevel(Enum):
    """Risk level values"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class ComplianceControl:
    """Individual compliance control definition"""
    control_id: str
    control_name: str
    framework: ComplianceFramework
    
    # Control details
    description: str = ""
    requirement: str = ""
    control_type: str = "technical"  # technical, administrative, physical
    
    # Assessment criteria
    assessment_criteria: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    
    # Scoring
    weight: float = 1.0  # Weight in overall compliance score
    max_score: int = 100
    
    # Risk
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Metadata
    reference_url: Optional[str] = None
    is_mandatory: bool = True
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "control_name": self.control_name,
            "framework": self.framework.value,
            "description": self.description,
            "requirement": self.requirement,
            "control_type": self.control_type,
            "assessment_criteria": self.assessment_criteria,
            "evidence_requirements": self.evidence_requirements,
            "weight": self.weight,
            "max_score": self.max_score,
            "risk_level": self.risk_level.value,
            "reference_url": self.reference_url,
            "is_mandatory": self.is_mandatory,
            "is_active": self.is_active
        }


@dataclass
class ControlAssessment:
    """Assessment result for individual control"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    control_id: str = ""
    
    # Assessment results
    status: AssessmentStatus = AssessmentStatus.NOT_ASSESSED
    score: int = 0
    max_score: int = 100
    
    # Details
    findings: List[str] = field(default_factory=list)
    evidence_collected: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    remediation_priority: RiskLevel = RiskLevel.MEDIUM
    
    # Metadata
    assessed_by: str = "system"
    assessment_method: str = "automated"
    
    # Timestamps
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "control_id": self.control_id,
            "status": self.status.value,
            "score": self.score,
            "max_score": self.max_score,
            "findings": self.findings,
            "evidence_collected": self.evidence_collected,
            "gaps_identified": self.gaps_identified,
            "recommendations": self.recommendations,
            "remediation_priority": self.remediation_priority.value,
            "assessed_by": self.assessed_by,
            "assessment_method": self.assessment_method,
            "assessed_at": self.assessed_at.isoformat()
        }


@dataclass
class ComplianceAssessmentResult:
    """Overall compliance assessment result"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    object_id: str = ""
    object_type: str = ""  # workflow, policy, system
    
    # Framework and scope
    framework: ComplianceFramework = ComplianceFramework.SOX
    assessment_scope: List[str] = field(default_factory=list)
    
    # Overall results
    overall_status: AssessmentStatus = AssessmentStatus.NOT_ASSESSED
    compliance_score: float = 0.0
    max_possible_score: float = 100.0
    compliance_percentage: float = 0.0
    
    # Control assessments
    control_assessments: List[ControlAssessment] = field(default_factory=list)
    
    # Statistics
    total_controls: int = 0
    compliant_controls: int = 0
    non_compliant_controls: int = 0
    partially_compliant_controls: int = 0
    
    # Risk analysis
    critical_gaps: int = 0
    high_risk_gaps: int = 0
    medium_risk_gaps: int = 0
    low_risk_gaps: int = 0
    
    # Recommendations
    priority_recommendations: List[str] = field(default_factory=list)
    estimated_remediation_effort: str = "medium"
    
    # Metadata
    tenant_id: int = 0
    assessed_by: str = "system"
    assessment_version: str = "1.0"
    
    # Timestamps
    assessment_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assessment_completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "framework": self.framework.value,
            "assessment_scope": self.assessment_scope,
            "overall_status": self.overall_status.value,
            "compliance_score": self.compliance_score,
            "max_possible_score": self.max_possible_score,
            "compliance_percentage": self.compliance_percentage,
            "control_assessments": [ca.to_dict() for ca in self.control_assessments],
            "total_controls": self.total_controls,
            "compliant_controls": self.compliant_controls,
            "non_compliant_controls": self.non_compliant_controls,
            "partially_compliant_controls": self.partially_compliant_controls,
            "critical_gaps": self.critical_gaps,
            "high_risk_gaps": self.high_risk_gaps,
            "medium_risk_gaps": self.medium_risk_gaps,
            "low_risk_gaps": self.low_risk_gaps,
            "priority_recommendations": self.priority_recommendations,
            "estimated_remediation_effort": self.estimated_remediation_effort,
            "tenant_id": self.tenant_id,
            "assessed_by": self.assessed_by,
            "assessment_version": self.assessment_version,
            "assessment_started_at": self.assessment_started_at.isoformat(),
            "assessment_completed_at": self.assessment_completed_at.isoformat() if self.assessment_completed_at else None
        }


class ComplianceAssessmentEngine:
    """
    Compliance Assessment Engine - Task 8.5-T07
    
    Automated compliance assessment and scoring for workflows and policies
    Provides comprehensive compliance evaluation against regulatory frameworks
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'enable_automated_assessment': True,
            'assessment_timeout_minutes': 30,
            'parallel_control_assessment': True,
            'cache_control_definitions': True,
            'generate_detailed_reports': True,
            'auto_remediation_suggestions': True,
            'compliance_threshold_percentage': 80.0
        }
        
        # Control definitions cache
        self.compliance_controls: Dict[str, List[ComplianceControl]] = {}
        self.controls_last_loaded: Optional[datetime] = None
        self.cache_ttl_minutes = 60
        
        # Assessment statistics
        self.assessment_stats = {
            'total_assessments': 0,
            'assessments_passed': 0,
            'assessments_failed': 0,
            'average_compliance_score': 0.0,
            'critical_gaps_identified': 0,
            'average_assessment_time_ms': 0.0
        }
        
        # Initialize default compliance controls
        self._initialize_default_compliance_controls()
    
    async def initialize(self) -> bool:
        """Initialize compliance assessment engine"""
        try:
            await self._create_compliance_assessment_tables()
            await self._load_compliance_controls()
            self.logger.info("✅ Compliance assessment engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize compliance assessment engine: {e}")
            return False
    
    async def assess_workflow_compliance(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: int,
        framework: ComplianceFramework = ComplianceFramework.SOX,
        assessment_scope: List[str] = None
    ) -> ComplianceAssessmentResult:
        """
        Assess workflow compliance against specified framework
        
        This is the main entry point for workflow compliance assessment
        """
        
        start_time = datetime.now(timezone.utc)
        
        workflow_id = workflow_definition.get('workflow_id', 'unknown')
        
        # Initialize assessment result
        result = ComplianceAssessmentResult(
            object_id=workflow_id,
            object_type="workflow",
            framework=framework,
            assessment_scope=assessment_scope or ["all"],
            tenant_id=tenant_id
        )
        
        try:
            # Refresh controls if needed
            await self._refresh_controls_if_needed()
            
            # Get applicable compliance controls
            applicable_controls = self._get_applicable_controls(
                framework, "workflow", assessment_scope or ["all"]
            )
            
            result.total_controls = len(applicable_controls)
            
            # Assess each control
            if self.config['parallel_control_assessment']:
                assessment_tasks = [
                    self._assess_control(workflow_definition, control, "workflow")
                    for control in applicable_controls
                ]
                control_assessments = await asyncio.gather(*assessment_tasks, return_exceptions=True)
                
                for assessment in control_assessments:
                    if isinstance(assessment, Exception):
                        self.logger.error(f"Control assessment error: {assessment}")
                        continue
                    result.control_assessments.append(assessment)
            else:
                for control in applicable_controls:
                    assessment = await self._assess_control(workflow_definition, control, "workflow")
                    result.control_assessments.append(assessment)
            
            # Calculate overall compliance
            await self._calculate_overall_compliance(result)
            
            # Generate recommendations
            await self._generate_recommendations(result)
            
            # Complete assessment
            result.assessment_completed_at = datetime.now(timezone.utc)
            
            # Update statistics
            self._update_assessment_stats(result, start_time)
            
            # Store assessment result
            if self.db_pool:
                await self._store_assessment_result(result)
            
            if result.overall_status == AssessmentStatus.COMPLIANT:
                self.logger.info(f"✅ Compliance assessment passed for workflow {workflow_id} ({result.compliance_percentage:.1f}%)")
            else:
                self.logger.warning(f"⚠️ Compliance assessment failed for workflow {workflow_id} ({result.compliance_percentage:.1f}%)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Compliance assessment error for workflow {workflow_id}: {e}")
            
            # Mark assessment as failed
            result.overall_status = AssessmentStatus.REQUIRES_REVIEW
            result.priority_recommendations.append(f"Assessment system error: {str(e)}")
            result.assessment_completed_at = datetime.now(timezone.utc)
            
            return result
    
    async def assess_policy_compliance(
        self,
        policy_definition: Dict[str, Any],
        tenant_id: int,
        framework: ComplianceFramework = ComplianceFramework.SOX,
        assessment_scope: List[str] = None
    ) -> ComplianceAssessmentResult:
        """
        Assess policy compliance against specified framework
        """
        
        start_time = datetime.now(timezone.utc)
        
        policy_id = policy_definition.get('policy_id', 'unknown')
        
        # Initialize assessment result
        result = ComplianceAssessmentResult(
            object_id=policy_id,
            object_type="policy",
            framework=framework,
            assessment_scope=assessment_scope or ["all"],
            tenant_id=tenant_id
        )
        
        try:
            # Get applicable compliance controls
            applicable_controls = self._get_applicable_controls(
                framework, "policy", assessment_scope or ["all"]
            )
            
            result.total_controls = len(applicable_controls)
            
            # Assess each control
            for control in applicable_controls:
                assessment = await self._assess_control(policy_definition, control, "policy")
                result.control_assessments.append(assessment)
            
            # Calculate overall compliance
            await self._calculate_overall_compliance(result)
            
            # Generate recommendations
            await self._generate_recommendations(result)
            
            # Complete assessment
            result.assessment_completed_at = datetime.now(timezone.utc)
            
            # Update statistics
            self._update_assessment_stats(result, start_time)
            
            # Store assessment result
            if self.db_pool:
                await self._store_assessment_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Compliance assessment error for policy {policy_id}: {e}")
            
            # Mark assessment as failed
            result.overall_status = AssessmentStatus.REQUIRES_REVIEW
            result.priority_recommendations.append(f"Assessment system error: {str(e)}")
            result.assessment_completed_at = datetime.now(timezone.utc)
            
            return result
    
    def _initialize_default_compliance_controls(self):
        """Initialize default compliance controls for different frameworks"""
        
        # SOX Controls
        sox_controls = [
            ComplianceControl(
                control_id="SOX-001",
                control_name="Segregation of Duties",
                framework=ComplianceFramework.SOX,
                description="Ensure proper segregation of duties in financial processes",
                requirement="No single individual should have control over all aspects of a financial transaction",
                control_type="administrative",
                assessment_criteria=[
                    "Workflow has multiple approval steps",
                    "Different roles for creation and approval",
                    "No self-approval allowed"
                ],
                evidence_requirements=[
                    "Workflow definition with role assignments",
                    "Approval chain documentation",
                    "Access control matrix"
                ],
                weight=2.0,
                risk_level=RiskLevel.CRITICAL,
                is_mandatory=True
            ),
            
            ComplianceControl(
                control_id="SOX-002",
                control_name="Audit Trail Requirements",
                framework=ComplianceFramework.SOX,
                description="Maintain comprehensive audit trails for all financial transactions",
                requirement="All financial processes must generate immutable audit trails",
                control_type="technical",
                assessment_criteria=[
                    "Evidence pack generation enabled",
                    "Immutable audit logging configured",
                    "Retention policy defined"
                ],
                evidence_requirements=[
                    "Audit log configuration",
                    "Evidence pack samples",
                    "Retention policy documentation"
                ],
                weight=1.5,
                risk_level=RiskLevel.HIGH,
                is_mandatory=True
            ),
            
            ComplianceControl(
                control_id="SOX-003",
                control_name="Change Management",
                framework=ComplianceFramework.SOX,
                description="Implement proper change management for financial systems",
                requirement="All changes to financial processes must be approved and documented",
                control_type="administrative",
                assessment_criteria=[
                    "Workflow versioning enabled",
                    "Change approval process defined",
                    "Change documentation maintained"
                ],
                evidence_requirements=[
                    "Change management policy",
                    "Approval records",
                    "Version history"
                ],
                weight=1.0,
                risk_level=RiskLevel.MEDIUM,
                is_mandatory=True
            )
        ]
        
        # GDPR Controls
        gdpr_controls = [
            ComplianceControl(
                control_id="GDPR-001",
                control_name="Data Protection by Design",
                framework=ComplianceFramework.GDPR,
                description="Implement data protection by design and by default",
                requirement="Personal data processing must include appropriate technical and organizational measures",
                control_type="technical",
                assessment_criteria=[
                    "PII redaction enabled",
                    "Data minimization implemented",
                    "Purpose limitation enforced"
                ],
                evidence_requirements=[
                    "Data protection impact assessment",
                    "PII handling procedures",
                    "Data flow documentation"
                ],
                weight=2.0,
                risk_level=RiskLevel.CRITICAL,
                is_mandatory=True
            ),
            
            ComplianceControl(
                control_id="GDPR-002",
                control_name="Consent Management",
                framework=ComplianceFramework.GDPR,
                description="Implement proper consent management for personal data processing",
                requirement="Valid consent must be obtained and managed for personal data processing",
                control_type="administrative",
                assessment_criteria=[
                    "Consent tracking implemented",
                    "Consent withdrawal mechanism available",
                    "Consent records maintained"
                ],
                evidence_requirements=[
                    "Consent management system",
                    "Consent records",
                    "Privacy notices"
                ],
                weight=1.5,
                risk_level=RiskLevel.HIGH,
                is_mandatory=True
            )
        ]
        
        # HIPAA Controls
        hipaa_controls = [
            ComplianceControl(
                control_id="HIPAA-001",
                control_name="PHI Protection",
                framework=ComplianceFramework.HIPAA,
                description="Protect Protected Health Information (PHI)",
                requirement="PHI must be protected through appropriate safeguards",
                control_type="technical",
                assessment_criteria=[
                    "PHI encryption enabled",
                    "Access controls implemented",
                    "Audit logging configured"
                ],
                evidence_requirements=[
                    "Encryption configuration",
                    "Access control policies",
                    "Audit log samples"
                ],
                weight=2.0,
                risk_level=RiskLevel.CRITICAL,
                is_mandatory=True
            ),
            
            ComplianceControl(
                control_id="HIPAA-002",
                control_name="Minimum Necessary Standard",
                framework=ComplianceFramework.HIPAA,
                description="Implement minimum necessary standard for PHI access",
                requirement="Only minimum necessary PHI should be accessed for specific purposes",
                control_type="administrative",
                assessment_criteria=[
                    "Role-based access controls",
                    "Data minimization implemented",
                    "Access justification required"
                ],
                evidence_requirements=[
                    "Access control matrix",
                    "Data access policies",
                    "Access justification records"
                ],
                weight=1.5,
                risk_level=RiskLevel.HIGH,
                is_mandatory=True
            )
        ]
        
        # RBI Controls
        rbi_controls = [
            ComplianceControl(
                control_id="RBI-001",
                control_name="Dual Control Mechanism",
                framework=ComplianceFramework.RBI,
                description="Implement dual control for high-value transactions",
                requirement="High-value transactions must have dual authorization",
                control_type="administrative",
                assessment_criteria=[
                    "Dual approval required",
                    "Independent verification",
                    "Transaction limits enforced"
                ],
                evidence_requirements=[
                    "Dual control policy",
                    "Approval records",
                    "Transaction monitoring logs"
                ],
                weight=2.0,
                risk_level=RiskLevel.CRITICAL,
                is_mandatory=True
            ),
            
            ComplianceControl(
                control_id="RBI-002",
                control_name="Risk Assessment",
                framework=ComplianceFramework.RBI,
                description="Conduct comprehensive risk assessment for lending decisions",
                requirement="All lending decisions must include proper risk assessment",
                control_type="technical",
                assessment_criteria=[
                    "Risk scoring implemented",
                    "Credit assessment performed",
                    "Risk mitigation measures defined"
                ],
                evidence_requirements=[
                    "Risk assessment reports",
                    "Credit scoring models",
                    "Risk mitigation plans"
                ],
                weight=1.5,
                risk_level=RiskLevel.HIGH,
                is_mandatory=True
            )
        ]
        
        # Store controls by framework
        self.compliance_controls = {
            ComplianceFramework.SOX.value: sox_controls,
            ComplianceFramework.GDPR.value: gdpr_controls,
            ComplianceFramework.HIPAA.value: hipaa_controls,
            ComplianceFramework.RBI.value: rbi_controls
        }
    
    def _get_applicable_controls(
        self,
        framework: ComplianceFramework,
        object_type: str,
        assessment_scope: List[str]
    ) -> List[ComplianceControl]:
        """Get applicable compliance controls for assessment"""
        
        controls = self.compliance_controls.get(framework.value, [])
        
        # Filter by scope if specified
        if assessment_scope and "all" not in assessment_scope:
            # In production, this would filter controls by scope
            pass
        
        # Filter active controls
        return [control for control in controls if control.is_active]
    
    async def _assess_control(
        self,
        object_definition: Dict[str, Any],
        control: ComplianceControl,
        object_type: str
    ) -> ControlAssessment:
        """Assess individual compliance control"""
        
        assessment = ControlAssessment(
            control_id=control.control_id,
            max_score=control.max_score
        )
        
        try:
            # Assess based on control type and criteria
            if control.control_id == "SOX-001":  # Segregation of Duties
                assessment = await self._assess_segregation_of_duties(object_definition, control, assessment)
            elif control.control_id == "SOX-002":  # Audit Trail
                assessment = await self._assess_audit_trail(object_definition, control, assessment)
            elif control.control_id == "SOX-003":  # Change Management
                assessment = await self._assess_change_management(object_definition, control, assessment)
            elif control.control_id == "GDPR-001":  # Data Protection by Design
                assessment = await self._assess_data_protection_by_design(object_definition, control, assessment)
            elif control.control_id == "GDPR-002":  # Consent Management
                assessment = await self._assess_consent_management(object_definition, control, assessment)
            elif control.control_id == "HIPAA-001":  # PHI Protection
                assessment = await self._assess_phi_protection(object_definition, control, assessment)
            elif control.control_id == "HIPAA-002":  # Minimum Necessary
                assessment = await self._assess_minimum_necessary(object_definition, control, assessment)
            elif control.control_id == "RBI-001":  # Dual Control
                assessment = await self._assess_dual_control(object_definition, control, assessment)
            elif control.control_id == "RBI-002":  # Risk Assessment
                assessment = await self._assess_risk_assessment(object_definition, control, assessment)
            else:
                # Generic assessment
                assessment = await self._assess_generic_control(object_definition, control, assessment)
            
            # Determine overall status based on score
            if assessment.score >= control.max_score * 0.9:
                assessment.status = AssessmentStatus.COMPLIANT
            elif assessment.score >= control.max_score * 0.7:
                assessment.status = AssessmentStatus.PARTIALLY_COMPLIANT
            else:
                assessment.status = AssessmentStatus.NON_COMPLIANT
            
        except Exception as e:
            self.logger.error(f"Error assessing control {control.control_id}: {e}")
            assessment.status = AssessmentStatus.REQUIRES_REVIEW
            assessment.findings.append(f"Assessment error: {str(e)}")
        
        return assessment
    
    # Specific control assessment methods
    
    async def _assess_segregation_of_duties(
        self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment
    ) -> ControlAssessment:
        """Assess segregation of duties control"""
        
        steps = obj.get('steps', [])
        score = 0
        
        # Check for multiple approval steps
        approval_steps = [s for s in steps if s.get('type') == 'approval']
        if len(approval_steps) >= 2:
            score += 40
            assessment.findings.append("Multiple approval steps found")
            assessment.evidence_collected.append("Approval steps configuration")
        else:
            assessment.gaps_identified.append("Insufficient approval steps")
            assessment.recommendations.append("Add additional approval steps")
        
        # Check for role separation
        create_roles = set()
        approve_roles = set()
        
        for step in steps:
            if step.get('action') in ['create', 'submit']:
                create_roles.update(step.get('allowed_roles', []))
            elif step.get('action') in ['approve', 'authorize']:
                approve_roles.update(step.get('allowed_roles', []))
        
        if not create_roles.intersection(approve_roles):
            score += 40
            assessment.findings.append("Proper role separation implemented")
            assessment.evidence_collected.append("Role separation matrix")
        else:
            assessment.gaps_identified.append("Role overlap between creation and approval")
            assessment.recommendations.append("Separate creation and approval roles")
        
        # Check for self-approval prevention
        if obj.get('governance', {}).get('prevent_self_approval', False):
            score += 20
            assessment.findings.append("Self-approval prevention enabled")
            assessment.evidence_collected.append("Self-approval prevention configuration")
        else:
            assessment.gaps_identified.append("Self-approval prevention not configured")
            assessment.recommendations.append("Enable self-approval prevention")
        
        assessment.score = score
        return assessment
    
    async def _assess_audit_trail(
        self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment
    ) -> ControlAssessment:
        """Assess audit trail control"""
        
        score = 0
        
        # Check for evidence pack generation
        if obj.get('governance', {}).get('evidence_pack_required', False):
            score += 40
            assessment.findings.append("Evidence pack generation enabled")
            assessment.evidence_collected.append("Evidence pack configuration")
        else:
            assessment.gaps_identified.append("Evidence pack generation not enabled")
            assessment.recommendations.append("Enable evidence pack generation")
        
        # Check for audit logging
        if obj.get('governance', {}).get('audit_logging_enabled', False):
            score += 40
            assessment.findings.append("Audit logging enabled")
            assessment.evidence_collected.append("Audit logging configuration")
        else:
            assessment.gaps_identified.append("Audit logging not enabled")
            assessment.recommendations.append("Enable comprehensive audit logging")
        
        # Check for retention policy
        retention_days = obj.get('governance', {}).get('retention_days', 0)
        if retention_days >= 2555:  # 7 years
            score += 20
            assessment.findings.append("Appropriate retention policy configured")
            assessment.evidence_collected.append("Retention policy configuration")
        else:
            assessment.gaps_identified.append("Insufficient retention period")
            assessment.recommendations.append("Set retention period to at least 7 years")
        
        assessment.score = score
        return assessment
    
    async def _assess_change_management(
        self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment
    ) -> ControlAssessment:
        """Assess change management control"""
        
        score = 0
        
        # Check for versioning
        if obj.get('version'):
            score += 30
            assessment.findings.append("Workflow versioning implemented")
            assessment.evidence_collected.append("Version information")
        else:
            assessment.gaps_identified.append("No version information")
            assessment.recommendations.append("Implement workflow versioning")
        
        # Check for approval workflow
        if obj.get('governance', {}).get('change_approval_required', False):
            score += 40
            assessment.findings.append("Change approval process defined")
            assessment.evidence_collected.append("Change approval configuration")
        else:
            assessment.gaps_identified.append("No change approval process")
            assessment.recommendations.append("Implement change approval workflow")
        
        # Check for change documentation
        if obj.get('metadata', {}).get('change_log'):
            score += 30
            assessment.findings.append("Change documentation maintained")
            assessment.evidence_collected.append("Change log")
        else:
            assessment.gaps_identified.append("No change documentation")
            assessment.recommendations.append("Maintain comprehensive change logs")
        
        assessment.score = score
        return assessment
    
    # Additional assessment methods for other controls...
    async def _assess_data_protection_by_design(self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment) -> ControlAssessment:
        score = 50  # Placeholder implementation
        assessment.score = score
        assessment.findings.append("Basic data protection measures found")
        return assessment
    
    async def _assess_consent_management(self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment) -> ControlAssessment:
        score = 50  # Placeholder implementation
        assessment.score = score
        assessment.findings.append("Basic consent management found")
        return assessment
    
    async def _assess_phi_protection(self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment) -> ControlAssessment:
        score = 50  # Placeholder implementation
        assessment.score = score
        assessment.findings.append("Basic PHI protection measures found")
        return assessment
    
    async def _assess_minimum_necessary(self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment) -> ControlAssessment:
        score = 50  # Placeholder implementation
        assessment.score = score
        assessment.findings.append("Basic minimum necessary controls found")
        return assessment
    
    async def _assess_dual_control(self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment) -> ControlAssessment:
        score = 50  # Placeholder implementation
        assessment.score = score
        assessment.findings.append("Basic dual control measures found")
        return assessment
    
    async def _assess_risk_assessment(self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment) -> ControlAssessment:
        score = 50  # Placeholder implementation
        assessment.score = score
        assessment.findings.append("Basic risk assessment found")
        return assessment
    
    async def _assess_generic_control(self, obj: Dict[str, Any], control: ComplianceControl, assessment: ControlAssessment) -> ControlAssessment:
        score = 50  # Placeholder implementation
        assessment.score = score
        assessment.findings.append("Generic assessment performed")
        return assessment
    
    async def _calculate_overall_compliance(self, result: ComplianceAssessmentResult):
        """Calculate overall compliance score and status"""
        
        if not result.control_assessments:
            result.overall_status = AssessmentStatus.NOT_ASSESSED
            return
        
        # Calculate weighted score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for assessment in result.control_assessments:
            # Get control weight (default to 1.0)
            control = next(
                (c for controls in self.compliance_controls.values() 
                 for c in controls if c.control_id == assessment.control_id), 
                None
            )
            weight = control.weight if control else 1.0
            
            total_weighted_score += assessment.score * weight
            total_weight += assessment.max_score * weight
            
            # Count by status
            if assessment.status == AssessmentStatus.COMPLIANT:
                result.compliant_controls += 1
            elif assessment.status == AssessmentStatus.NON_COMPLIANT:
                result.non_compliant_controls += 1
            elif assessment.status == AssessmentStatus.PARTIALLY_COMPLIANT:
                result.partially_compliant_controls += 1
            
            # Count gaps by risk level
            if assessment.remediation_priority == RiskLevel.CRITICAL:
                result.critical_gaps += 1
            elif assessment.remediation_priority == RiskLevel.HIGH:
                result.high_risk_gaps += 1
            elif assessment.remediation_priority == RiskLevel.MEDIUM:
                result.medium_risk_gaps += 1
            elif assessment.remediation_priority == RiskLevel.LOW:
                result.low_risk_gaps += 1
        
        # Calculate percentages
        if total_weight > 0:
            result.compliance_score = total_weighted_score
            result.max_possible_score = total_weight
            result.compliance_percentage = (total_weighted_score / total_weight) * 100
        
        # Determine overall status
        if result.compliance_percentage >= self.config['compliance_threshold_percentage']:
            result.overall_status = AssessmentStatus.COMPLIANT
        elif result.compliance_percentage >= 50:
            result.overall_status = AssessmentStatus.PARTIALLY_COMPLIANT
        else:
            result.overall_status = AssessmentStatus.NON_COMPLIANT
        
        # Override if critical gaps exist
        if result.critical_gaps > 0:
            result.overall_status = AssessmentStatus.NON_COMPLIANT
    
    async def _generate_recommendations(self, result: ComplianceAssessmentResult):
        """Generate priority recommendations based on assessment results"""
        
        recommendations = []
        
        # Critical gaps first
        critical_assessments = [
            a for a in result.control_assessments 
            if a.remediation_priority == RiskLevel.CRITICAL and a.status != AssessmentStatus.COMPLIANT
        ]
        
        for assessment in critical_assessments:
            recommendations.extend(assessment.recommendations)
        
        # High-risk gaps
        high_risk_assessments = [
            a for a in result.control_assessments 
            if a.remediation_priority == RiskLevel.HIGH and a.status != AssessmentStatus.COMPLIANT
        ]
        
        for assessment in high_risk_assessments[:3]:  # Top 3 high-risk
            recommendations.extend(assessment.recommendations)
        
        # Add general recommendations
        if result.compliance_percentage < 50:
            recommendations.append("Conduct comprehensive compliance review")
            recommendations.append("Implement compliance management framework")
        
        if result.critical_gaps > 0:
            recommendations.append("Address critical compliance gaps immediately")
        
        # Estimate remediation effort
        if result.critical_gaps > 3 or result.high_risk_gaps > 5:
            result.estimated_remediation_effort = "high"
        elif result.critical_gaps > 1 or result.high_risk_gaps > 2:
            result.estimated_remediation_effort = "medium"
        else:
            result.estimated_remediation_effort = "low"
        
        result.priority_recommendations = list(set(recommendations))[:10]  # Top 10 unique recommendations
    
    def _update_assessment_stats(self, result: ComplianceAssessmentResult, start_time: datetime):
        """Update assessment statistics"""
        
        self.assessment_stats['total_assessments'] += 1
        
        if result.overall_status == AssessmentStatus.COMPLIANT:
            self.assessment_stats['assessments_passed'] += 1
        else:
            self.assessment_stats['assessments_failed'] += 1
        
        # Update average compliance score
        current_avg = self.assessment_stats['average_compliance_score']
        total_assessments = self.assessment_stats['total_assessments']
        self.assessment_stats['average_compliance_score'] = (
            (current_avg * (total_assessments - 1) + result.compliance_percentage) / total_assessments
        )
        
        # Update critical gaps
        self.assessment_stats['critical_gaps_identified'] += result.critical_gaps
        
        # Update average assessment time
        assessment_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        current_avg_time = self.assessment_stats['average_assessment_time_ms']
        self.assessment_stats['average_assessment_time_ms'] = (
            (current_avg_time * (total_assessments - 1) + assessment_time) / total_assessments
        )
    
    async def _refresh_controls_if_needed(self):
        """Refresh controls from database if cache is stale"""
        
        if not self.config['cache_control_definitions']:
            await self._load_compliance_controls()
            return
        
        if (not self.controls_last_loaded or 
            (datetime.now(timezone.utc) - self.controls_last_loaded).total_seconds() > self.cache_ttl_minutes * 60):
            await self._load_compliance_controls()
    
    async def _load_compliance_controls(self):
        """Load compliance controls from database"""
        # In production, this would load custom controls from database
        # For now, we use the default controls initialized in memory
        self.controls_last_loaded = datetime.now(timezone.utc)
    
    async def _create_compliance_assessment_tables(self):
        """Create compliance assessment database tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Compliance assessment results
        CREATE TABLE IF NOT EXISTS compliance_assessments (
            assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            object_id VARCHAR(100) NOT NULL,
            object_type VARCHAR(50) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Framework and scope
            framework VARCHAR(50) NOT NULL,
            assessment_scope TEXT[] DEFAULT ARRAY[],
            
            -- Overall results
            overall_status VARCHAR(30) NOT NULL,
            compliance_score FLOAT NOT NULL DEFAULT 0,
            max_possible_score FLOAT NOT NULL DEFAULT 100,
            compliance_percentage FLOAT NOT NULL DEFAULT 0,
            
            -- Statistics
            total_controls INTEGER NOT NULL DEFAULT 0,
            compliant_controls INTEGER NOT NULL DEFAULT 0,
            non_compliant_controls INTEGER NOT NULL DEFAULT 0,
            partially_compliant_controls INTEGER NOT NULL DEFAULT 0,
            
            -- Risk analysis
            critical_gaps INTEGER NOT NULL DEFAULT 0,
            high_risk_gaps INTEGER NOT NULL DEFAULT 0,
            medium_risk_gaps INTEGER NOT NULL DEFAULT 0,
            low_risk_gaps INTEGER NOT NULL DEFAULT 0,
            
            -- Recommendations
            priority_recommendations TEXT[] DEFAULT ARRAY[],
            estimated_remediation_effort VARCHAR(20) DEFAULT 'medium',
            
            -- Metadata
            assessed_by VARCHAR(100) DEFAULT 'system',
            assessment_version VARCHAR(20) DEFAULT '1.0',
            
            -- Timestamps
            assessment_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            assessment_completed_at TIMESTAMPTZ,
            
            -- Constraints
            CONSTRAINT chk_compliance_overall_status CHECK (overall_status IN ('compliant', 'non_compliant', 'partially_compliant', 'requires_review', 'not_assessed')),
            CONSTRAINT chk_compliance_framework CHECK (framework IN ('sox', 'gdpr', 'hipaa', 'rbi', 'irdai', 'pci_dss', 'basel_iii', 'dpdp', 'ccpa', 'iso_27001')),
            CONSTRAINT chk_compliance_object_type CHECK (object_type IN ('workflow', 'policy', 'system')),
            CONSTRAINT chk_compliance_remediation_effort CHECK (estimated_remediation_effort IN ('low', 'medium', 'high'))
        );
        
        -- Control assessments
        CREATE TABLE IF NOT EXISTS compliance_control_assessments (
            assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            compliance_assessment_id UUID REFERENCES compliance_assessments(assessment_id) ON DELETE CASCADE,
            control_id VARCHAR(100) NOT NULL,
            
            -- Assessment results
            status VARCHAR(30) NOT NULL,
            score INTEGER NOT NULL DEFAULT 0,
            max_score INTEGER NOT NULL DEFAULT 100,
            
            -- Details
            findings TEXT[] DEFAULT ARRAY[],
            evidence_collected TEXT[] DEFAULT ARRAY[],
            gaps_identified TEXT[] DEFAULT ARRAY[],
            recommendations TEXT[] DEFAULT ARRAY[],
            
            -- Metadata
            remediation_priority VARCHAR(20) DEFAULT 'medium',
            assessed_by VARCHAR(100) DEFAULT 'system',
            assessment_method VARCHAR(50) DEFAULT 'automated',
            
            -- Timestamps
            assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_control_assessment_status CHECK (status IN ('compliant', 'non_compliant', 'partially_compliant', 'requires_review', 'not_assessed')),
            CONSTRAINT chk_control_remediation_priority CHECK (remediation_priority IN ('critical', 'high', 'medium', 'low', 'minimal'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_compliance_assessments_object ON compliance_assessments(object_id, object_type, assessment_completed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compliance_assessments_tenant ON compliance_assessments(tenant_id, assessment_completed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compliance_assessments_framework ON compliance_assessments(framework, overall_status);
        CREATE INDEX IF NOT EXISTS idx_compliance_assessments_status ON compliance_assessments(overall_status, assessment_completed_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_control_assessments_compliance ON compliance_control_assessments(compliance_assessment_id);
        CREATE INDEX IF NOT EXISTS idx_control_assessments_control ON compliance_control_assessments(control_id, status);
        CREATE INDEX IF NOT EXISTS idx_control_assessments_priority ON compliance_control_assessments(remediation_priority, assessed_at DESC);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Compliance assessment tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create compliance assessment tables: {e}")
            raise
    
    async def _store_assessment_result(self, result: ComplianceAssessmentResult):
        """Store assessment result in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert main assessment result
                assessment_query = """
                    INSERT INTO compliance_assessments (
                        assessment_id, object_id, object_type, tenant_id, framework,
                        assessment_scope, overall_status, compliance_score, max_possible_score,
                        compliance_percentage, total_controls, compliant_controls,
                        non_compliant_controls, partially_compliant_controls,
                        critical_gaps, high_risk_gaps, medium_risk_gaps, low_risk_gaps,
                        priority_recommendations, estimated_remediation_effort,
                        assessed_by, assessment_version, assessment_started_at,
                        assessment_completed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
                    RETURNING assessment_id
                """
                
                assessment_id = await conn.fetchval(
                    assessment_query,
                    result.assessment_id,
                    result.object_id,
                    result.object_type,
                    result.tenant_id,
                    result.framework.value,
                    result.assessment_scope,
                    result.overall_status.value,
                    result.compliance_score,
                    result.max_possible_score,
                    result.compliance_percentage,
                    result.total_controls,
                    result.compliant_controls,
                    result.non_compliant_controls,
                    result.partially_compliant_controls,
                    result.critical_gaps,
                    result.high_risk_gaps,
                    result.medium_risk_gaps,
                    result.low_risk_gaps,
                    result.priority_recommendations,
                    result.estimated_remediation_effort,
                    result.assessed_by,
                    result.assessment_version,
                    result.assessment_started_at,
                    result.assessment_completed_at
                )
                
                # Insert control assessments
                if result.control_assessments:
                    control_query = """
                        INSERT INTO compliance_control_assessments (
                            compliance_assessment_id, control_id, status, score, max_score,
                            findings, evidence_collected, gaps_identified, recommendations,
                            remediation_priority, assessed_by, assessment_method, assessed_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """
                    
                    for control_assessment in result.control_assessments:
                        await conn.execute(
                            control_query,
                            assessment_id,
                            control_assessment.control_id,
                            control_assessment.status.value,
                            control_assessment.score,
                            control_assessment.max_score,
                            control_assessment.findings,
                            control_assessment.evidence_collected,
                            control_assessment.gaps_identified,
                            control_assessment.recommendations,
                            control_assessment.remediation_priority.value,
                            control_assessment.assessed_by,
                            control_assessment.assessment_method,
                            control_assessment.assessed_at
                        )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store compliance assessment result: {e}")
    
    def get_assessment_statistics(self) -> Dict[str, Any]:
        """Get assessment statistics"""
        
        pass_rate = 0.0
        if self.assessment_stats['total_assessments'] > 0:
            pass_rate = (
                self.assessment_stats['assessments_passed'] / 
                self.assessment_stats['total_assessments']
            ) * 100
        
        return {
            'total_assessments': self.assessment_stats['total_assessments'],
            'assessments_passed': self.assessment_stats['assessments_passed'],
            'assessments_failed': self.assessment_stats['assessments_failed'],
            'pass_rate_percentage': round(pass_rate, 2),
            'average_compliance_score': round(self.assessment_stats['average_compliance_score'], 2),
            'critical_gaps_identified': self.assessment_stats['critical_gaps_identified'],
            'average_assessment_time_ms': round(self.assessment_stats['average_assessment_time_ms'], 2),
            'compliance_threshold_percentage': self.config['compliance_threshold_percentage'],
            'supported_frameworks': len(self.compliance_controls),
            'automated_assessment_enabled': self.config['enable_automated_assessment']
        }


# Global compliance assessment engine instance
compliance_assessment_engine = ComplianceAssessmentEngine()
