# Security & Compliance Checklist - Chapter 8.5
# Tasks 8.5-T01 to T30: GDPR, HIPAA, SOX, RBI, IRDAI compliance checklists

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Compliance frameworks"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    RBI = "rbi"
    IRDAI = "irdai"
    PCI_DSS = "pci_dss"
    BASEL_III = "basel_iii"
    SOLVENCY_II = "solvency_ii"
    DPDP = "dpdp"
    CCPA = "ccpa"

class ChecklistItemType(Enum):
    """Types of checklist items"""
    POLICY = "policy"
    TECHNICAL_CONTROL = "technical_control"
    PROCESS = "process"
    DOCUMENTATION = "documentation"
    TRAINING = "training"
    AUDIT = "audit"
    MONITORING = "monitoring"
    REPORTING = "reporting"

class ComplianceStatus(Enum):
    """Compliance status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NON_COMPLIANT = "non_compliant"
    EXEMPT = "exempt"
    DEFERRED = "deferred"

class RiskLevel(Enum):
    """Risk levels for non-compliance"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ChecklistItem:
    """Individual compliance checklist item"""
    item_id: str
    title: str
    description: str
    
    # Classification
    framework: ComplianceFramework
    item_type: ChecklistItemType
    category: str
    
    # Requirements
    mandatory: bool = True
    applicable_industries: List[str] = field(default_factory=list)
    applicable_regions: List[str] = field(default_factory=list)
    
    # Implementation
    implementation_guidance: str = ""
    technical_requirements: List[str] = field(default_factory=list)
    documentation_requirements: List[str] = field(default_factory=list)
    
    # Validation
    validation_criteria: List[str] = field(default_factory=list)
    testing_procedures: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    
    # Risk and impact
    risk_level: RiskLevel = RiskLevel.MEDIUM
    non_compliance_impact: str = ""
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    related_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['framework'] = self.framework.value
        data['item_type'] = self.item_type.value
        data['risk_level'] = self.risk_level.value
        return data

@dataclass
class ComplianceAssessment:
    """Compliance assessment for a checklist item"""
    assessment_id: str
    item_id: str
    
    # Assessment details
    assessed_by: str
    assessed_at: str
    status: ComplianceStatus
    
    # Evidence
    evidence_provided: List[str] = field(default_factory=list)
    evidence_quality: str = "adequate"  # inadequate, adequate, strong
    
    # Findings
    findings: str = ""
    gaps_identified: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Remediation
    remediation_required: bool = False
    remediation_plan: str = ""
    remediation_deadline: Optional[str] = None
    remediation_owner: Optional[str] = None
    
    # Follow-up
    next_assessment_date: Optional[str] = None
    monitoring_frequency: str = "annual"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data

@dataclass
class ComplianceChecklist:
    """Complete compliance checklist for a framework"""
    checklist_id: str
    framework: ComplianceFramework
    checklist_name: str
    description: str
    version: str
    
    # Scope
    applicable_industries: List[str] = field(default_factory=list)
    applicable_regions: List[str] = field(default_factory=list)
    
    # Items
    items: List[ChecklistItem] = field(default_factory=list)
    
    # Metadata
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Validation
    total_items: int = 0
    mandatory_items: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['framework'] = self.framework.value
        data['items'] = [item.to_dict() for item in self.items]
        return data
    
    def calculate_statistics(self) -> None:
        """Calculate checklist statistics"""
        self.total_items = len(self.items)
        self.mandatory_items = len([item for item in self.items if item.mandatory])

class SecurityComplianceChecklistManager:
    """
    Security & Compliance Checklist Manager - Tasks 8.5-T01 to T30
    Manages comprehensive compliance checklists for various regulatory frameworks
    """
    
    def __init__(self):
        self.checklists: Dict[ComplianceFramework, ComplianceChecklist] = {}
        self.assessments: Dict[str, ComplianceAssessment] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize all compliance checklists
        self._initialize_all_checklists()
    
    def _initialize_all_checklists(self) -> None:
        """Initialize all compliance checklists"""
        
        self._initialize_gdpr_checklist()
        self._initialize_hipaa_checklist()
        self._initialize_sox_checklist()
        self._initialize_rbi_checklist()
        self._initialize_irdai_checklist()
        self._initialize_pci_dss_checklist()
        
        self.logger.info(f"✅ Initialized {len(self.checklists)} compliance checklists")
    
    def _initialize_gdpr_checklist(self) -> None:
        """Initialize GDPR compliance checklist"""
        
        gdpr_items = [
            ChecklistItem(
                item_id="gdpr_001",
                title="Lawful Basis for Processing",
                description="Establish and document lawful basis for all personal data processing activities",
                framework=ComplianceFramework.GDPR,
                item_type=ChecklistItemType.POLICY,
                category="Data Processing",
                applicable_industries=["saas", "healthcare", "banking", "insurance"],
                applicable_regions=["EU", "EEA"],
                implementation_guidance="Document lawful basis for each processing activity and ensure it's communicated to data subjects",
                technical_requirements=[
                    "Consent management system",
                    "Data processing inventory",
                    "Legal basis tracking"
                ],
                documentation_requirements=[
                    "Data processing impact assessment",
                    "Legal basis documentation",
                    "Privacy policy updates"
                ],
                validation_criteria=[
                    "All processing activities have documented lawful basis",
                    "Consent is freely given, specific, informed, and unambiguous",
                    "Legal basis is communicated to data subjects"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="Fines up to 4% of annual global revenue"
            ),
            
            ChecklistItem(
                item_id="gdpr_002",
                title="Data Subject Rights Implementation",
                description="Implement processes to handle data subject rights requests",
                framework=ComplianceFramework.GDPR,
                item_type=ChecklistItemType.PROCESS,
                category="Data Subject Rights",
                applicable_industries=["saas", "healthcare", "banking", "insurance"],
                applicable_regions=["EU", "EEA"],
                implementation_guidance="Establish procedures for handling access, rectification, erasure, portability, and objection requests",
                technical_requirements=[
                    "Data subject request portal",
                    "Data discovery and retrieval tools",
                    "Automated data deletion capabilities"
                ],
                documentation_requirements=[
                    "Data subject rights procedures",
                    "Request handling workflows",
                    "Response templates and timelines"
                ],
                validation_criteria=[
                    "Requests handled within 30 days",
                    "Identity verification procedures in place",
                    "Data portability in machine-readable format"
                ],
                risk_level=RiskLevel.HIGH,
                non_compliance_impact="Regulatory enforcement action, reputational damage"
            ),
            
            ChecklistItem(
                item_id="gdpr_003",
                title="Data Protection by Design and Default",
                description="Implement privacy by design principles in all systems and processes",
                framework=ComplianceFramework.GDPR,
                item_type=ChecklistItemType.TECHNICAL_CONTROL,
                category="Privacy Engineering",
                applicable_industries=["saas", "healthcare", "banking", "insurance"],
                applicable_regions=["EU", "EEA"],
                implementation_guidance="Integrate privacy considerations into system design and default settings",
                technical_requirements=[
                    "Privacy-preserving system architecture",
                    "Data minimization controls",
                    "Purpose limitation enforcement"
                ],
                documentation_requirements=[
                    "Privacy impact assessments",
                    "System design documentation",
                    "Default privacy settings documentation"
                ],
                validation_criteria=[
                    "Privacy considerations documented in system design",
                    "Default settings maximize privacy protection",
                    "Data minimization principles applied"
                ],
                risk_level=RiskLevel.HIGH,
                non_compliance_impact="Regulatory scrutiny, potential fines"
            ),
            
            ChecklistItem(
                item_id="gdpr_004",
                title="Breach Notification Procedures",
                description="Establish procedures for detecting, reporting, and communicating data breaches",
                framework=ComplianceFramework.GDPR,
                item_type=ChecklistItemType.PROCESS,
                category="Incident Response",
                applicable_industries=["saas", "healthcare", "banking", "insurance"],
                applicable_regions=["EU", "EEA"],
                implementation_guidance="Implement 72-hour breach notification to supervisory authority and timely notification to data subjects",
                technical_requirements=[
                    "Breach detection systems",
                    "Incident response automation",
                    "Notification tracking system"
                ],
                documentation_requirements=[
                    "Breach response procedures",
                    "Notification templates",
                    "Breach register and logs"
                ],
                validation_criteria=[
                    "Breaches detected within 24 hours",
                    "Supervisory authority notified within 72 hours",
                    "Data subjects notified when required"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="Additional fines for late notification"
            )
        ]
        
        gdpr_checklist = ComplianceChecklist(
            checklist_id="gdpr_checklist_v1",
            framework=ComplianceFramework.GDPR,
            checklist_name="GDPR Compliance Checklist",
            description="Comprehensive checklist for GDPR compliance covering data protection requirements",
            version="1.0.0",
            applicable_industries=["saas", "healthcare", "banking", "insurance"],
            applicable_regions=["EU", "EEA"],
            items=gdpr_items,
            created_by="compliance_team"
        )
        
        gdpr_checklist.calculate_statistics()
        self.checklists[ComplianceFramework.GDPR] = gdpr_checklist
    
    def _initialize_hipaa_checklist(self) -> None:
        """Initialize HIPAA compliance checklist"""
        
        hipaa_items = [
            ChecklistItem(
                item_id="hipaa_001",
                title="Administrative Safeguards",
                description="Implement administrative safeguards to protect PHI",
                framework=ComplianceFramework.HIPAA,
                item_type=ChecklistItemType.POLICY,
                category="Administrative Safeguards",
                applicable_industries=["healthcare", "insurance"],
                applicable_regions=["USA"],
                implementation_guidance="Establish policies and procedures for PHI access, workforce training, and incident response",
                technical_requirements=[
                    "Access control systems",
                    "Audit logging",
                    "User authentication"
                ],
                documentation_requirements=[
                    "HIPAA policies and procedures",
                    "Workforce training records",
                    "Business associate agreements"
                ],
                validation_criteria=[
                    "All workforce members trained on HIPAA",
                    "Access controls properly configured",
                    "Incident response procedures tested"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="Civil monetary penalties up to $1.5M per incident"
            ),
            
            ChecklistItem(
                item_id="hipaa_002",
                title="Physical Safeguards",
                description="Implement physical safeguards to protect PHI and systems",
                framework=ComplianceFramework.HIPAA,
                item_type=ChecklistItemType.TECHNICAL_CONTROL,
                category="Physical Safeguards",
                applicable_industries=["healthcare", "insurance"],
                applicable_regions=["USA"],
                implementation_guidance="Secure physical access to facilities, workstations, and media containing PHI",
                technical_requirements=[
                    "Physical access controls",
                    "Workstation security",
                    "Media disposal procedures"
                ],
                documentation_requirements=[
                    "Physical security policies",
                    "Access control procedures",
                    "Media disposal logs"
                ],
                validation_criteria=[
                    "Restricted access to PHI storage areas",
                    "Workstations secured when unattended",
                    "Media properly disposed of"
                ],
                risk_level=RiskLevel.HIGH,
                non_compliance_impact="Regulatory enforcement, breach notification requirements"
            ),
            
            ChecklistItem(
                item_id="hipaa_003",
                title="Technical Safeguards",
                description="Implement technical safeguards to protect electronic PHI",
                framework=ComplianceFramework.HIPAA,
                item_type=ChecklistItemType.TECHNICAL_CONTROL,
                category="Technical Safeguards",
                applicable_industries=["healthcare", "insurance"],
                applicable_regions=["USA"],
                implementation_guidance="Implement encryption, access controls, and audit logging for electronic PHI",
                technical_requirements=[
                    "Encryption at rest and in transit",
                    "Access control and authentication",
                    "Audit logging and monitoring"
                ],
                documentation_requirements=[
                    "Technical safeguards documentation",
                    "Encryption implementation guide",
                    "Audit log procedures"
                ],
                validation_criteria=[
                    "All PHI encrypted at rest and in transit",
                    "Strong authentication implemented",
                    "Comprehensive audit logging in place"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="Breach notification requirements, regulatory penalties"
            )
        ]
        
        hipaa_checklist = ComplianceChecklist(
            checklist_id="hipaa_checklist_v1",
            framework=ComplianceFramework.HIPAA,
            checklist_name="HIPAA Compliance Checklist",
            description="Comprehensive checklist for HIPAA compliance covering PHI protection requirements",
            version="1.0.0",
            applicable_industries=["healthcare", "insurance"],
            applicable_regions=["USA"],
            items=hipaa_items,
            created_by="compliance_team"
        )
        
        hipaa_checklist.calculate_statistics()
        self.checklists[ComplianceFramework.HIPAA] = hipaa_checklist
    
    def _initialize_sox_checklist(self) -> None:
        """Initialize SOX compliance checklist"""
        
        sox_items = [
            ChecklistItem(
                item_id="sox_001",
                title="Internal Control Over Financial Reporting",
                description="Establish and maintain internal controls over financial reporting",
                framework=ComplianceFramework.SOX,
                item_type=ChecklistItemType.PROCESS,
                category="Internal Controls",
                applicable_industries=["saas", "banking", "insurance"],
                applicable_regions=["USA"],
                implementation_guidance="Design and implement controls to ensure accurate financial reporting",
                technical_requirements=[
                    "Automated financial controls",
                    "Segregation of duties enforcement",
                    "Change management controls"
                ],
                documentation_requirements=[
                    "Control documentation",
                    "Process narratives",
                    "Testing evidence"
                ],
                validation_criteria=[
                    "Controls designed to address financial reporting risks",
                    "Controls operating effectively",
                    "Deficiencies remediated timely"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="SEC enforcement action, delisting risk"
            ),
            
            ChecklistItem(
                item_id="sox_002",
                title="Management Assessment",
                description="Management assessment of internal control effectiveness",
                framework=ComplianceFramework.SOX,
                item_type=ChecklistItemType.AUDIT,
                category="Management Assessment",
                applicable_industries=["saas", "banking", "insurance"],
                applicable_regions=["USA"],
                implementation_guidance="Annual management assessment of internal control effectiveness",
                technical_requirements=[
                    "Control testing automation",
                    "Deficiency tracking system",
                    "Management reporting tools"
                ],
                documentation_requirements=[
                    "Management assessment report",
                    "Control testing documentation",
                    "Deficiency remediation plans"
                ],
                validation_criteria=[
                    "Annual management assessment completed",
                    "Material weaknesses identified and disclosed",
                    "Remediation plans implemented"
                ],
                risk_level=RiskLevel.HIGH,
                non_compliance_impact="Audit qualifications, investor confidence loss"
            )
        ]
        
        sox_checklist = ComplianceChecklist(
            checklist_id="sox_checklist_v1",
            framework=ComplianceFramework.SOX,
            checklist_name="SOX Compliance Checklist",
            description="Comprehensive checklist for SOX compliance covering internal controls",
            version="1.0.0",
            applicable_industries=["saas", "banking", "insurance"],
            applicable_regions=["USA"],
            items=sox_items,
            created_by="compliance_team"
        )
        
        sox_checklist.calculate_statistics()
        self.checklists[ComplianceFramework.SOX] = sox_checklist
    
    def _initialize_rbi_checklist(self) -> None:
        """Initialize RBI compliance checklist"""
        
        rbi_items = [
            ChecklistItem(
                item_id="rbi_001",
                title="Know Your Customer (KYC) Compliance",
                description="Implement comprehensive KYC procedures as per RBI guidelines",
                framework=ComplianceFramework.RBI,
                item_type=ChecklistItemType.PROCESS,
                category="Customer Due Diligence",
                applicable_industries=["banking", "fintech"],
                applicable_regions=["India"],
                implementation_guidance="Establish risk-based KYC procedures for customer onboarding and monitoring",
                technical_requirements=[
                    "KYC document verification system",
                    "Risk assessment algorithms",
                    "Ongoing monitoring tools"
                ],
                documentation_requirements=[
                    "KYC policy document",
                    "Customer risk assessment procedures",
                    "Ongoing monitoring procedures"
                ],
                validation_criteria=[
                    "All customers have completed KYC",
                    "Risk categorization properly applied",
                    "Periodic KYC updates performed"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="Monetary penalties, license restrictions"
            ),
            
            ChecklistItem(
                item_id="rbi_002",
                title="Anti-Money Laundering (AML) Controls",
                description="Implement AML controls and suspicious transaction monitoring",
                framework=ComplianceFramework.RBI,
                item_type=ChecklistItemType.MONITORING,
                category="AML Compliance",
                applicable_industries=["banking", "fintech"],
                applicable_regions=["India"],
                implementation_guidance="Establish transaction monitoring and suspicious activity reporting",
                technical_requirements=[
                    "Transaction monitoring system",
                    "Sanctions screening",
                    "Suspicious activity reporting"
                ],
                documentation_requirements=[
                    "AML policy and procedures",
                    "Suspicious transaction reports",
                    "Training records"
                ],
                validation_criteria=[
                    "All transactions monitored for suspicious activity",
                    "Suspicious transactions reported timely",
                    "Staff trained on AML requirements"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="Regulatory penalties, reputational damage"
            )
        ]
        
        rbi_checklist = ComplianceChecklist(
            checklist_id="rbi_checklist_v1",
            framework=ComplianceFramework.RBI,
            checklist_name="RBI Compliance Checklist",
            description="Comprehensive checklist for RBI compliance covering banking regulations",
            version="1.0.0",
            applicable_industries=["banking", "fintech"],
            applicable_regions=["India"],
            items=rbi_items,
            created_by="compliance_team"
        )
        
        rbi_checklist.calculate_statistics()
        self.checklists[ComplianceFramework.RBI] = rbi_checklist
    
    def _initialize_irdai_checklist(self) -> None:
        """Initialize IRDAI compliance checklist"""
        
        irdai_items = [
            ChecklistItem(
                item_id="irdai_001",
                title="Solvency Margin Maintenance",
                description="Maintain required solvency margin as per IRDAI regulations",
                framework=ComplianceFramework.IRDAI,
                item_type=ChecklistItemType.MONITORING,
                category="Solvency Management",
                applicable_industries=["insurance"],
                applicable_regions=["India"],
                implementation_guidance="Monitor and maintain solvency margin above regulatory requirements",
                technical_requirements=[
                    "Solvency calculation system",
                    "Real-time monitoring dashboard",
                    "Regulatory reporting automation"
                ],
                documentation_requirements=[
                    "Solvency margin calculations",
                    "Actuarial valuations",
                    "Capital adequacy reports"
                ],
                validation_criteria=[
                    "Solvency margin above minimum requirement",
                    "Regular actuarial valuations performed",
                    "Regulatory returns filed timely"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="License suspension, regulatory intervention"
            ),
            
            ChecklistItem(
                item_id="irdai_002",
                title="Claims Settlement Procedures",
                description="Implement fair and transparent claims settlement procedures",
                framework=ComplianceFramework.IRDAI,
                item_type=ChecklistItemType.PROCESS,
                category="Claims Management",
                applicable_industries=["insurance"],
                applicable_regions=["India"],
                implementation_guidance="Establish procedures for timely and fair claims settlement",
                technical_requirements=[
                    "Claims management system",
                    "Fraud detection algorithms",
                    "Settlement tracking tools"
                ],
                documentation_requirements=[
                    "Claims settlement procedures",
                    "Fraud investigation reports",
                    "Settlement ratio reports"
                ],
                validation_criteria=[
                    "Claims settled within prescribed timelines",
                    "Settlement ratios meet regulatory benchmarks",
                    "Fraud detection mechanisms in place"
                ],
                risk_level=RiskLevel.HIGH,
                non_compliance_impact="Regulatory penalties, customer complaints"
            )
        ]
        
        irdai_checklist = ComplianceChecklist(
            checklist_id="irdai_checklist_v1",
            framework=ComplianceFramework.IRDAI,
            checklist_name="IRDAI Compliance Checklist",
            description="Comprehensive checklist for IRDAI compliance covering insurance regulations",
            version="1.0.0",
            applicable_industries=["insurance"],
            applicable_regions=["India"],
            items=irdai_items,
            created_by="compliance_team"
        )
        
        irdai_checklist.calculate_statistics()
        self.checklists[ComplianceFramework.IRDAI] = irdai_checklist
    
    def _initialize_pci_dss_checklist(self) -> None:
        """Initialize PCI DSS compliance checklist"""
        
        pci_items = [
            ChecklistItem(
                item_id="pci_001",
                title="Secure Network and Systems",
                description="Install and maintain a firewall configuration to protect cardholder data",
                framework=ComplianceFramework.PCI_DSS,
                item_type=ChecklistItemType.TECHNICAL_CONTROL,
                category="Network Security",
                applicable_industries=["banking", "ecommerce", "fintech"],
                applicable_regions=["Global"],
                implementation_guidance="Implement network segmentation and firewall rules to protect cardholder data",
                technical_requirements=[
                    "Network firewalls",
                    "Network segmentation",
                    "Intrusion detection systems"
                ],
                documentation_requirements=[
                    "Network architecture diagrams",
                    "Firewall rule documentation",
                    "Security configuration standards"
                ],
                validation_criteria=[
                    "Firewalls properly configured",
                    "Network segmentation implemented",
                    "Regular security assessments performed"
                ],
                risk_level=RiskLevel.CRITICAL,
                non_compliance_impact="Card brand penalties, loss of processing privileges"
            )
        ]
        
        pci_checklist = ComplianceChecklist(
            checklist_id="pci_dss_checklist_v1",
            framework=ComplianceFramework.PCI_DSS,
            checklist_name="PCI DSS Compliance Checklist",
            description="Comprehensive checklist for PCI DSS compliance covering payment card security",
            version="1.0.0",
            applicable_industries=["banking", "ecommerce", "fintech"],
            applicable_regions=["Global"],
            items=pci_items,
            created_by="compliance_team"
        )
        
        pci_checklist.calculate_statistics()
        self.checklists[ComplianceFramework.PCI_DSS] = pci_checklist
    
    def get_checklist(self, framework: ComplianceFramework) -> Optional[ComplianceChecklist]:
        """Get compliance checklist for a framework"""
        return self.checklists.get(framework)
    
    def get_applicable_checklists(
        self,
        industry: str,
        region: str = "Global"
    ) -> List[ComplianceChecklist]:
        """Get applicable checklists for industry and region"""
        
        applicable_checklists = []
        
        for checklist in self.checklists.values():
            if (not checklist.applicable_industries or 
                industry.lower() in [ind.lower() for ind in checklist.applicable_industries]):
                if (not checklist.applicable_regions or
                    region in checklist.applicable_regions or
                    "Global" in checklist.applicable_regions):
                    applicable_checklists.append(checklist)
        
        return applicable_checklists
    
    def create_assessment(
        self,
        item_id: str,
        assessed_by: str,
        status: ComplianceStatus,
        findings: str = "",
        evidence_provided: List[str] = None,
        gaps_identified: List[str] = None,
        recommendations: List[str] = None
    ) -> ComplianceAssessment:
        """Create compliance assessment for a checklist item"""
        
        assessment_id = str(uuid.uuid4())
        
        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            item_id=item_id,
            assessed_by=assessed_by,
            assessed_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            findings=findings,
            evidence_provided=evidence_provided or [],
            gaps_identified=gaps_identified or [],
            recommendations=recommendations or []
        )
        
        # Set remediation requirements for non-compliant items
        if status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.IN_PROGRESS]:
            assessment.remediation_required = True
            assessment.next_assessment_date = (
                datetime.now(timezone.utc) + timedelta(days=30)
            ).isoformat()
        
        self.assessments[assessment_id] = assessment
        
        self.logger.info(f"✅ Created assessment: {assessment_id} for item {item_id}")
        return assessment
    
    def get_compliance_status_summary(
        self,
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Get compliance status summary for a framework"""
        
        checklist = self.checklists.get(framework)
        if not checklist:
            return {"error": "Framework not found"}
        
        # Get assessments for this framework
        framework_assessments = [
            assessment for assessment in self.assessments.values()
            if any(item.item_id == assessment.item_id for item in checklist.items)
        ]
        
        # Calculate statistics
        total_items = len(checklist.items)
        assessed_items = len(framework_assessments)
        
        status_counts = {}
        for status in ComplianceStatus:
            status_counts[status.value] = len([
                a for a in framework_assessments if a.status == status
            ])
        
        compliance_percentage = (
            status_counts.get(ComplianceStatus.COMPLETED.value, 0) / total_items * 100
            if total_items > 0 else 0
        )
        
        return {
            "framework": framework.value,
            "total_items": total_items,
            "assessed_items": assessed_items,
            "compliance_percentage": round(compliance_percentage, 1),
            "status_breakdown": status_counts,
            "mandatory_items": checklist.mandatory_items,
            "high_risk_items": len([
                item for item in checklist.items
                if item.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ])
        }
    
    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        include_evidence: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        checklist = self.checklists.get(framework)
        if not checklist:
            return {"error": "Framework not found"}
        
        # Get assessments for this framework
        framework_assessments = {
            assessment.item_id: assessment
            for assessment in self.assessments.values()
            if any(item.item_id == assessment.item_id for item in checklist.items)
        }
        
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "framework": framework.value,
            "checklist_info": {
                "name": checklist.checklist_name,
                "version": checklist.version,
                "total_items": checklist.total_items,
                "mandatory_items": checklist.mandatory_items
            },
            "compliance_summary": self.get_compliance_status_summary(framework),
            "items": []
        }
        
        # Add item details
        for item in checklist.items:
            item_data = item.to_dict()
            
            # Add assessment data if available
            assessment = framework_assessments.get(item.item_id)
            if assessment:
                item_data["assessment"] = assessment.to_dict()
                if not include_evidence:
                    # Remove detailed evidence for summary reports
                    item_data["assessment"].pop("evidence_provided", None)
            else:
                item_data["assessment"] = {
                    "status": ComplianceStatus.NOT_STARTED.value,
                    "assessed_by": None,
                    "assessed_at": None
                }
            
            report["items"].append(item_data)
        
        return report
    
    def get_remediation_plan(
        self,
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Get remediation plan for non-compliant items"""
        
        checklist = self.checklists.get(framework)
        if not checklist:
            return {"error": "Framework not found"}
        
        # Get non-compliant assessments
        non_compliant_assessments = [
            assessment for assessment in self.assessments.values()
            if assessment.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.IN_PROGRESS]
            and any(item.item_id == assessment.item_id for item in checklist.items)
        ]
        
        remediation_plan = {
            "plan_id": str(uuid.uuid4()),
            "framework": framework.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_items_requiring_remediation": len(non_compliant_assessments),
            "remediation_items": []
        }
        
        for assessment in non_compliant_assessments:
            # Find the corresponding checklist item
            item = next(
                (item for item in checklist.items if item.item_id == assessment.item_id),
                None
            )
            
            if item:
                remediation_item = {
                    "item_id": item.item_id,
                    "title": item.title,
                    "risk_level": item.risk_level.value,
                    "current_status": assessment.status.value,
                    "gaps_identified": assessment.gaps_identified,
                    "recommendations": assessment.recommendations,
                    "remediation_deadline": assessment.remediation_deadline,
                    "remediation_owner": assessment.remediation_owner,
                    "priority": "high" if item.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else "medium"
                }
                
                remediation_plan["remediation_items"].append(remediation_item)
        
        # Sort by priority and risk level
        remediation_plan["remediation_items"].sort(
            key=lambda x: (x["priority"] == "high", x["risk_level"] == "critical"),
            reverse=True
        )
        
        return remediation_plan
    
    def get_all_frameworks_summary(self) -> Dict[str, Any]:
        """Get summary of all compliance frameworks"""
        
        summary = {
            "total_frameworks": len(self.checklists),
            "total_checklist_items": sum(checklist.total_items for checklist in self.checklists.values()),
            "total_assessments": len(self.assessments),
            "frameworks": {}
        }
        
        for framework in self.checklists.keys():
            summary["frameworks"][framework.value] = self.get_compliance_status_summary(framework)
        
        return summary

# Global instance
security_compliance_checklist = SecurityComplianceChecklistManager()
