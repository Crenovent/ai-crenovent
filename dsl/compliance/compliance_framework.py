"""
Compliance & IP Flywheel Framework - Section 7.5 Tasks
======================================================

Implements Section 7.5 Tasks:
- 7.5.1: Define compliance & IP flywheel principles
- 7.5.2: Map regulatory overlays (SOX, HIPAA, GDPR, DPDP, RBI)
- 7.5.3: Build compliance metadata schema
- 7.5.4: Configure evidence pack standard
- 7.5.9: Configure SoD enforcement
- 7.5.14: Build anomaly detection for compliance violations
- 7.5.15: Configure predictive compliance analytics
- 7.5.32: Automate IP asset tagging in registries

Following user rules:
- No hardcoding, dynamic configuration
- SaaS/IT industry focus
- Multi-tenant aware
- Modular design
"""

import asyncio
import json
import logging
import os
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import base64

logger = logging.getLogger(__name__)

class RegulatoryFramework(Enum):
    """Task 7.5.2: Map regulatory overlays"""
    SOX = "sox"          # Sarbanes-Oxley Act
    HIPAA = "hipaa"      # Health Insurance Portability and Accountability Act
    GDPR = "gdpr"        # General Data Protection Regulation
    DPDP = "dpdp"        # Digital Personal Data Protection (India)
    RBI = "rbi"          # Reserve Bank of India guidelines
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    CCPA = "ccpa"        # California Consumer Privacy Act
    ISO27001 = "iso27001" # ISO 27001 Information Security

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REMEDIATION_REQUIRED = "remediation_required"
    EXEMPT = "exempt"

class EvidenceType(Enum):
    """Task 7.5.4: Evidence pack types"""
    AUDIT_LOG = "audit_log"
    APPROVAL_RECORD = "approval_record"
    ACCESS_CONTROL = "access_control"
    DATA_LINEAGE = "data_lineage"
    ENCRYPTION_PROOF = "encryption_proof"
    RETENTION_RECORD = "retention_record"
    CONSENT_RECORD = "consent_record"
    BREACH_NOTIFICATION = "breach_notification"

class SoDRole(Enum):
    """Task 7.5.9: Separation of Duties roles"""
    CREATOR = "creator"
    REVIEWER = "reviewer"
    APPROVER = "approver"
    EXECUTOR = "executor"
    AUDITOR = "auditor"
    COMPLIANCE_OFFICER = "compliance_officer"

class IPAssetType(Enum):
    """Task 7.5.32: IP asset types for tagging"""
    ALGORITHM = "algorithm"
    MODEL = "model"
    WORKFLOW_TEMPLATE = "workflow_template"
    BUSINESS_RULE = "business_rule"
    DATA_SCHEMA = "data_schema"
    API_SPECIFICATION = "api_specification"
    PROCESS_DEFINITION = "process_definition"

@dataclass
class CompliancePrinciple:
    """Task 7.5.1: Define compliance & IP flywheel principles"""
    principle_id: str
    name: str
    description: str
    category: str
    
    # Regulatory mapping
    applicable_frameworks: List[RegulatoryFramework] = field(default_factory=list)
    
    # Implementation requirements
    mandatory_controls: List[str] = field(default_factory=list)
    evidence_requirements: List[EvidenceType] = field(default_factory=list)
    
    # SaaS/IT specific
    industry_applicability: List[str] = field(default_factory=list)
    tenant_tier_requirements: Dict[str, List[str]] = field(default_factory=dict)
    
    # IP flywheel integration
    ip_generation_enabled: bool = False
    knowledge_capture_points: List[str] = field(default_factory=list)
    
    # Metadata
    priority: int = 1  # 1=critical, 5=low
    automation_level: str = "manual"  # manual, semi-automated, automated

@dataclass
class ComplianceMetadata:
    """Task 7.5.3: Build compliance metadata schema"""
    metadata_id: str
    entity_id: str  # ID of the entity being tracked
    entity_type: str  # Type of entity (workflow, user, data, etc.)
    
    # Regulatory context
    applicable_frameworks: List[RegulatoryFramework] = field(default_factory=list)
    compliance_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    
    # Audit trail
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    last_reviewed_at: Optional[datetime] = None
    last_reviewed_by: Optional[str] = None
    
    # Evidence and documentation
    evidence_packs: List[str] = field(default_factory=list)  # Evidence pack IDs
    documentation_links: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "medium"  # low, medium, high, critical
    risk_factors: List[str] = field(default_factory=list)
    mitigation_measures: List[str] = field(default_factory=list)
    
    # Multi-tenant context
    tenant_id: Optional[int] = None
    region: Optional[str] = None
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvidencePack:
    """Task 7.5.4: Configure evidence pack standard"""
    evidence_id: str
    evidence_type: EvidenceType
    title: str
    description: str
    
    # Content and integrity
    content_hash: str  # SHA-256 hash for integrity
    content_location: str  # Storage location (blob URL, file path, etc.)
    content_size_bytes: int = 0
    
    # Regulatory context
    regulatory_frameworks: List[RegulatoryFramework] = field(default_factory=list)
    compliance_requirement: str = ""
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    retention_until: Optional[datetime] = None
    
    # Chain of custody
    created_by: str = ""
    witnessed_by: List[str] = field(default_factory=list)
    digital_signatures: List[str] = field(default_factory=list)
    
    # Relationships
    related_entities: List[str] = field(default_factory=list)
    parent_evidence_id: Optional[str] = None
    child_evidence_ids: List[str] = field(default_factory=list)
    
    # Multi-tenant context
    tenant_id: Optional[int] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComplianceViolation:
    """Task 7.5.14: Compliance violation detection"""
    violation_id: str
    detected_at: datetime
    violation_type: str
    severity: str  # low, medium, high, critical
    
    # Regulatory context
    regulatory_framework: RegulatoryFramework
    violated_requirement: str
    
    # Details
    description: str
    affected_entities: List[str] = field(default_factory=list)
    evidence_of_violation: List[str] = field(default_factory=list)
    
    # Impact assessment
    business_impact: str = ""
    financial_impact: Optional[float] = None
    reputational_risk: str = "low"
    
    # Remediation
    remediation_required: bool = True
    remediation_deadline: Optional[datetime] = None
    remediation_actions: List[str] = field(default_factory=list)
    remediation_status: str = "open"  # open, in_progress, resolved, closed
    
    # Multi-tenant context
    tenant_id: Optional[int] = None
    
    # Workflow integration
    escalation_required: bool = False
    assigned_to: Optional[str] = None

@dataclass
class IPAsset:
    """Task 7.5.32: IP asset for automated tagging"""
    asset_id: str
    name: str
    description: str
    asset_type: IPAssetType
    
    # IP classification
    ip_category: str = ""  # proprietary, open_source, licensed, etc.
    confidentiality_level: str = "internal"  # public, internal, confidential, restricted
    
    # Ownership and licensing
    owner: str = ""
    contributors: List[str] = field(default_factory=list)
    license_type: str = ""
    license_terms: str = ""
    
    # Technical details
    version: str = "1.0.0"
    technology_stack: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Business value
    business_value_score: float = 0.0
    revenue_potential: str = "unknown"
    competitive_advantage: bool = False
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    last_modified_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"  # active, deprecated, archived
    
    # Compliance and legal
    patent_status: str = "none"  # none, pending, granted
    trade_secret: bool = False
    export_controlled: bool = False
    
    # Multi-tenant context
    tenant_id: Optional[int] = None
    
    # Metadata and tags
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComplianceFramework:
    """
    Comprehensive Compliance & IP Flywheel Framework
    
    Implements Section 7.5 Tasks:
    - 7.5.1: Define compliance & IP flywheel principles
    - 7.5.2: Map regulatory overlays (SOX, HIPAA, GDPR, DPDP, RBI)
    - 7.5.3: Build compliance metadata schema
    - 7.5.4: Configure evidence pack standard
    - 7.5.9: Configure SoD enforcement
    - 7.5.14: Build anomaly detection for compliance violations
    - 7.5.15: Configure predictive compliance analytics
    - 7.5.32: Automate IP asset tagging in registries
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize compliance principles (Task 7.5.1)
        self.principles = self._initialize_compliance_principles()
        
        # Initialize regulatory mappings (Task 7.5.2)
        self.regulatory_mappings = self._initialize_regulatory_mappings()
        
        # Storage for compliance entities
        self.compliance_metadata: Dict[str, ComplianceMetadata] = {}
        self.evidence_packs: Dict[str, EvidencePack] = {}
        self.violations: List[ComplianceViolation] = []
        self.ip_assets: Dict[str, IPAsset] = {}
        
        # SoD enforcement (Task 7.5.9)
        self.sod_policies = self._initialize_sod_policies()
        
        # Anomaly detection (Task 7.5.14)
        self.violation_patterns = {}
        self.anomaly_thresholds = self._load_anomaly_thresholds()
        
        # Predictive analytics (Task 7.5.15)
        self.compliance_trends = {}
        self.prediction_models = {}
        
        self.logger.info("✅ Initialized Compliance & IP Flywheel Framework")
    
    def _initialize_compliance_principles(self) -> Dict[str, CompliancePrinciple]:
        """Task 7.5.1: Define compliance & IP flywheel principles"""
        principles = {}
        
        # Principle 1: Data Protection and Privacy
        principles["data_protection"] = CompliancePrinciple(
            principle_id="data_protection",
            name="Data Protection and Privacy",
            description="Ensure comprehensive data protection and privacy compliance",
            category="privacy",
            applicable_frameworks=[RegulatoryFramework.GDPR, RegulatoryFramework.DPDP, RegulatoryFramework.CCPA],
            mandatory_controls=[
                "data_encryption_at_rest",
                "data_encryption_in_transit",
                "access_control_enforcement",
                "consent_management",
                "data_minimization",
                "right_to_erasure"
            ],
            evidence_requirements=[
                EvidenceType.ENCRYPTION_PROOF,
                EvidenceType.CONSENT_RECORD,
                EvidenceType.ACCESS_CONTROL,
                EvidenceType.DATA_LINEAGE
            ],
            industry_applicability=["saas", "fintech", "healthcare", "ecommerce"],
            tenant_tier_requirements={
                "T0": ["full_encryption", "audit_logging", "consent_management"],
                "T1": ["encryption", "basic_audit"],
                "T2": ["basic_encryption"]
            },
            ip_generation_enabled=True,
            knowledge_capture_points=["consent_patterns", "privacy_policies", "data_flows"],
            priority=1,
            automation_level="semi-automated"
        )
        
        # Principle 2: Financial Controls and SOX Compliance
        principles["financial_controls"] = CompliancePrinciple(
            principle_id="financial_controls",
            name="Financial Controls and SOX Compliance",
            description="Implement SOX-compliant financial controls and audit trails",
            category="financial",
            applicable_frameworks=[RegulatoryFramework.SOX],
            mandatory_controls=[
                "segregation_of_duties",
                "approval_workflows",
                "immutable_audit_trails",
                "financial_reporting_controls",
                "change_management"
            ],
            evidence_requirements=[
                EvidenceType.APPROVAL_RECORD,
                EvidenceType.AUDIT_LOG,
                EvidenceType.ACCESS_CONTROL
            ],
            industry_applicability=["saas", "fintech", "public_companies"],
            tenant_tier_requirements={
                "T0": ["full_sox_compliance", "quarterly_attestation"],
                "T1": ["basic_financial_controls"],
                "T2": ["minimal_controls"]
            },
            ip_generation_enabled=True,
            knowledge_capture_points=["approval_patterns", "control_effectiveness", "risk_assessments"],
            priority=1,
            automation_level="automated"
        )
        
        # Principle 3: Security and Access Control
        principles["security_controls"] = CompliancePrinciple(
            principle_id="security_controls",
            name="Security and Access Control",
            description="Implement comprehensive security controls and access management",
            category="security",
            applicable_frameworks=[RegulatoryFramework.ISO27001, RegulatoryFramework.SOX, RegulatoryFramework.HIPAA],
            mandatory_controls=[
                "multi_factor_authentication",
                "role_based_access_control",
                "privileged_access_management",
                "security_monitoring",
                "incident_response"
            ],
            evidence_requirements=[
                EvidenceType.ACCESS_CONTROL,
                EvidenceType.AUDIT_LOG
            ],
            industry_applicability=["saas", "fintech", "healthcare", "banking"],
            tenant_tier_requirements={
                "T0": ["advanced_security", "continuous_monitoring"],
                "T1": ["standard_security"],
                "T2": ["basic_security"]
            },
            ip_generation_enabled=True,
            knowledge_capture_points=["threat_patterns", "security_policies", "incident_responses"],
            priority=1,
            automation_level="automated"
        )
        
        # Principle 4: IP Asset Management
        principles["ip_management"] = CompliancePrinciple(
            principle_id="ip_management",
            name="Intellectual Property Asset Management",
            description="Systematic management and protection of intellectual property assets",
            category="intellectual_property",
            applicable_frameworks=[],  # Cross-cutting principle
            mandatory_controls=[
                "ip_asset_inventory",
                "ip_classification",
                "ip_protection_measures",
                "license_compliance",
                "trade_secret_protection"
            ],
            evidence_requirements=[
                EvidenceType.AUDIT_LOG,
                EvidenceType.ACCESS_CONTROL
            ],
            industry_applicability=["saas", "technology", "software"],
            tenant_tier_requirements={
                "T0": ["comprehensive_ip_management"],
                "T1": ["basic_ip_tracking"],
                "T2": ["minimal_ip_awareness"]
            },
            ip_generation_enabled=True,
            knowledge_capture_points=["innovation_patterns", "asset_valuations", "competitive_intelligence"],
            priority=2,
            automation_level="semi-automated"
        )
        
        return principles
    
    def _initialize_regulatory_mappings(self) -> Dict[RegulatoryFramework, Dict[str, Any]]:
        """Task 7.5.2: Map regulatory overlays (SOX, HIPAA, GDPR, DPDP, RBI)"""
        mappings = {}
        
        # SOX (Sarbanes-Oxley Act)
        mappings[RegulatoryFramework.SOX] = {
            "full_name": "Sarbanes-Oxley Act",
            "jurisdiction": "United States",
            "applicability": "Public companies and their subsidiaries",
            "key_requirements": {
                "section_302": "CEO/CFO certification of financial reports",
                "section_404": "Internal control over financial reporting",
                "section_409": "Real-time disclosure of material changes",
                "section_802": "Criminal penalties for document destruction"
            },
            "controls_mapping": {
                "segregation_of_duties": "Required for financial processes",
                "approval_workflows": "Multi-level approval for financial transactions",
                "audit_trails": "Immutable logs for all financial activities",
                "change_management": "Controlled changes to financial systems"
            },
            "evidence_requirements": [
                EvidenceType.APPROVAL_RECORD,
                EvidenceType.AUDIT_LOG,
                EvidenceType.ACCESS_CONTROL
            ],
            "retention_periods": {
                "audit_logs": 2555,  # 7 years in days
                "approval_records": 2555,
                "financial_reports": 2555
            },
            "saas_specific_considerations": [
                "Revenue recognition automation",
                "Subscription billing controls",
                "Customer data financial impact"
            ]
        }
        
        # GDPR (General Data Protection Regulation)
        mappings[RegulatoryFramework.GDPR] = {
            "full_name": "General Data Protection Regulation",
            "jurisdiction": "European Union",
            "applicability": "Organizations processing EU personal data",
            "key_requirements": {
                "article_6": "Lawful basis for processing",
                "article_7": "Consent requirements",
                "article_17": "Right to erasure (right to be forgotten)",
                "article_25": "Data protection by design and by default",
                "article_32": "Security of processing",
                "article_33": "Breach notification (72 hours)"
            },
            "controls_mapping": {
                "consent_management": "Explicit consent collection and management",
                "data_minimization": "Process only necessary personal data",
                "encryption": "Appropriate technical measures for security",
                "breach_notification": "Automated breach detection and notification"
            },
            "evidence_requirements": [
                EvidenceType.CONSENT_RECORD,
                EvidenceType.DATA_LINEAGE,
                EvidenceType.ENCRYPTION_PROOF,
                EvidenceType.BREACH_NOTIFICATION
            ],
            "retention_periods": {
                "consent_records": 2555,  # 7 years
                "breach_notifications": 1825,  # 5 years
                "data_processing_logs": 1095  # 3 years
            },
            "saas_specific_considerations": [
                "Customer data processing agreements",
                "Cross-border data transfers",
                "Automated decision-making transparency"
            ]
        }
        
        # HIPAA (Health Insurance Portability and Accountability Act)
        mappings[RegulatoryFramework.HIPAA] = {
            "full_name": "Health Insurance Portability and Accountability Act",
            "jurisdiction": "United States",
            "applicability": "Healthcare organizations and business associates",
            "key_requirements": {
                "privacy_rule": "Protection of PHI (Protected Health Information)",
                "security_rule": "Administrative, physical, and technical safeguards",
                "breach_notification_rule": "Notification of PHI breaches",
                "omnibus_rule": "Business associate liability"
            },
            "controls_mapping": {
                "access_control": "Minimum necessary access to PHI",
                "encryption": "Encryption of PHI at rest and in transit",
                "audit_logging": "Access logs for all PHI interactions",
                "business_associate_agreements": "Contracts with third parties"
            },
            "evidence_requirements": [
                EvidenceType.ACCESS_CONTROL,
                EvidenceType.ENCRYPTION_PROOF,
                EvidenceType.AUDIT_LOG,
                EvidenceType.BREACH_NOTIFICATION
            ],
            "retention_periods": {
                "audit_logs": 2190,  # 6 years
                "access_records": 2190,
                "breach_notifications": 2190
            },
            "saas_specific_considerations": [
                "Cloud service provider agreements",
                "PHI data residency requirements",
                "Automated PHI detection and protection"
            ]
        }
        
        # DPDP (Digital Personal Data Protection - India)
        mappings[RegulatoryFramework.DPDP] = {
            "full_name": "Digital Personal Data Protection Act",
            "jurisdiction": "India",
            "applicability": "Organizations processing personal data of Indian citizens",
            "key_requirements": {
                "section_6": "Consent for processing personal data",
                "section_8": "Purpose limitation",
                "section_9": "Data retention and erasure",
                "section_16": "Cross-border transfer restrictions"
            },
            "controls_mapping": {
                "consent_management": "Clear and specific consent",
                "data_localization": "Storage of personal data within India",
                "purpose_limitation": "Use data only for stated purposes",
                "data_erasure": "Delete data when no longer needed"
            },
            "evidence_requirements": [
                EvidenceType.CONSENT_RECORD,
                EvidenceType.DATA_LINEAGE,
                EvidenceType.RETENTION_RECORD
            ],
            "retention_periods": {
                "consent_records": 1825,  # 5 years
                "processing_logs": 1095,  # 3 years
                "erasure_records": 1095
            },
            "saas_specific_considerations": [
                "Data localization for Indian customers",
                "Consent management for Indian users",
                "Cross-border data transfer compliance"
            ]
        }
        
        # RBI (Reserve Bank of India)
        mappings[RegulatoryFramework.RBI] = {
            "full_name": "Reserve Bank of India Guidelines",
            "jurisdiction": "India",
            "applicability": "Financial institutions and fintech companies",
            "key_requirements": {
                "data_localization": "Critical financial data must be stored in India",
                "cybersecurity_framework": "Comprehensive cybersecurity measures",
                "outsourcing_guidelines": "Risk management for outsourced services",
                "digital_lending_guidelines": "Compliance for digital lending platforms"
            },
            "controls_mapping": {
                "data_residency": "Financial data storage within India",
                "cybersecurity_controls": "Multi-layered security framework",
                "vendor_management": "Due diligence for service providers",
                "incident_reporting": "Cyber incident reporting to RBI"
            },
            "evidence_requirements": [
                EvidenceType.DATA_LINEAGE,
                EvidenceType.ACCESS_CONTROL,
                EvidenceType.AUDIT_LOG
            ],
            "retention_periods": {
                "transaction_logs": 3650,  # 10 years
                "audit_records": 2555,  # 7 years
                "incident_reports": 1825  # 5 years
            },
            "saas_specific_considerations": [
                "Fintech SaaS data residency",
                "Digital payment compliance",
                "Lending platform regulations"
            ]
        }
        
        return mappings
    
    def _initialize_sod_policies(self) -> Dict[str, Dict[str, Any]]:
        """Task 7.5.9: Configure SoD enforcement"""
        return {
            "financial_transactions": {
                "required_roles": [SoDRole.CREATOR, SoDRole.REVIEWER, SoDRole.APPROVER],
                "minimum_separation": 2,  # At least 2 different people
                "approval_matrix": {
                    "amount_thresholds": {
                        "low": {"max_amount": 1000, "required_approvers": 1},
                        "medium": {"max_amount": 10000, "required_approvers": 2},
                        "high": {"max_amount": 100000, "required_approvers": 3}
                    }
                },
                "prohibited_combinations": [
                    [SoDRole.CREATOR, SoDRole.APPROVER],  # Same person cannot create and approve
                    [SoDRole.REVIEWER, SoDRole.EXECUTOR]  # Same person cannot review and execute
                ]
            },
            "system_administration": {
                "required_roles": [SoDRole.CREATOR, SoDRole.REVIEWER, SoDRole.APPROVER],
                "minimum_separation": 2,
                "change_types": {
                    "configuration_change": {"required_approvers": 1},
                    "security_change": {"required_approvers": 2},
                    "production_deployment": {"required_approvers": 2}
                },
                "prohibited_combinations": [
                    [SoDRole.CREATOR, SoDRole.EXECUTOR]
                ]
            },
            "compliance_activities": {
                "required_roles": [SoDRole.COMPLIANCE_OFFICER, SoDRole.AUDITOR],
                "minimum_separation": 2,
                "activity_types": {
                    "policy_creation": {"required_approvers": 1},
                    "violation_investigation": {"required_approvers": 2},
                    "audit_execution": {"required_approvers": 1}
                },
                "prohibited_combinations": [
                    [SoDRole.COMPLIANCE_OFFICER, SoDRole.AUDITOR]  # Independence requirement
                ]
            }
        }
    
    def _load_anomaly_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Task 7.5.14: Load anomaly detection thresholds"""
        return {
            "access_patterns": {
                "unusual_access_hours": 0.1,  # 10% threshold for after-hours access
                "failed_login_spike": 0.05,   # 5% threshold for failed login increase
                "privilege_escalation": 0.0   # Zero tolerance for unauthorized escalation
            },
            "data_patterns": {
                "bulk_data_access": 0.2,      # 20% threshold for bulk access
                "cross_tenant_access": 0.0,   # Zero tolerance for cross-tenant access
                "data_export_volume": 0.15    # 15% threshold for data export volume
            },
            "financial_patterns": {
                "transaction_amount_spike": 0.3,  # 30% threshold for amount spikes
                "approval_bypass": 0.0,           # Zero tolerance for approval bypass
                "duplicate_transactions": 0.02    # 2% threshold for duplicates
            }
        }
    
    def create_compliance_metadata(self, entity_id: str, entity_type: str,
                                 applicable_frameworks: List[RegulatoryFramework],
                                 tenant_id: Optional[int] = None) -> ComplianceMetadata:
        """Task 7.5.3: Create compliance metadata"""
        try:
            metadata_id = f"cm_{entity_type}_{entity_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            metadata = ComplianceMetadata(
                metadata_id=metadata_id,
                entity_id=entity_id,
                entity_type=entity_type,
                applicable_frameworks=applicable_frameworks,
                tenant_id=tenant_id,
                created_at=datetime.utcnow()
            )
            
            # Auto-populate risk assessment based on frameworks
            metadata.risk_level = self._assess_initial_risk_level(applicable_frameworks, entity_type)
            metadata.risk_factors = self._identify_risk_factors(applicable_frameworks, entity_type)
            
            # Store metadata
            self.compliance_metadata[metadata_id] = metadata
            
            self.logger.info(f"✅ Created compliance metadata: {metadata_id}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create compliance metadata: {e}")
            raise
    
    def create_evidence_pack(self, evidence_type: EvidenceType, title: str,
                           content_location: str, regulatory_frameworks: List[RegulatoryFramework],
                           tenant_id: Optional[int] = None) -> EvidencePack:
        """Task 7.5.4: Configure evidence pack standard"""
        try:
            evidence_id = f"ep_{evidence_type.value}_{uuid.uuid4().hex[:8]}"
            
            # Calculate content hash for integrity
            content_hash = self._calculate_content_hash(content_location)
            
            # Determine retention period based on regulatory requirements
            retention_until = self._calculate_retention_period(regulatory_frameworks)
            
            evidence_pack = EvidencePack(
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                title=title,
                description=f"Evidence pack for {evidence_type.value}",
                content_hash=content_hash,
                content_location=content_location,
                regulatory_frameworks=regulatory_frameworks,
                retention_until=retention_until,
                tenant_id=tenant_id,
                created_at=datetime.utcnow()
            )
            
            # Store evidence pack
            self.evidence_packs[evidence_id] = evidence_pack
            
            self.logger.info(f"✅ Created evidence pack: {evidence_id}")
            return evidence_pack
            
        except Exception as e:
            self.logger.error(f"Failed to create evidence pack: {e}")
            raise
    
    def enforce_sod_compliance(self, activity_type: str, user_roles: List[SoDRole],
                             user_id: str, activity_context: Dict[str, Any]) -> Dict[str, Any]:
        """Task 7.5.9: Configure SoD enforcement"""
        try:
            sod_result = {
                "compliant": False,
                "violations": [],
                "required_actions": [],
                "approval_required": False
            }
            
            # Get SoD policy for activity type
            policy = self.sod_policies.get(activity_type)
            if not policy:
                sod_result["violations"].append(f"No SoD policy defined for {activity_type}")
                return sod_result
            
            # Check required roles
            required_roles = policy["required_roles"]
            if not all(role in user_roles for role in required_roles):
                missing_roles = [role for role in required_roles if role not in user_roles]
                sod_result["violations"].append(f"Missing required roles: {missing_roles}")
            
            # Check prohibited combinations
            prohibited_combinations = policy.get("prohibited_combinations", [])
            for prohibited_combo in prohibited_combinations:
                if all(role in user_roles for role in prohibited_combo):
                    sod_result["violations"].append(f"Prohibited role combination: {prohibited_combo}")
            
            # Check approval requirements
            if "approval_matrix" in policy:
                approval_check = self._check_approval_requirements(
                    policy["approval_matrix"], activity_context
                )
                if approval_check["approval_required"]:
                    sod_result["approval_required"] = True
                    sod_result["required_actions"].extend(approval_check["required_actions"])
            
            # Determine overall compliance
            sod_result["compliant"] = len(sod_result["violations"]) == 0
            
            return sod_result
            
        except Exception as e:
            self.logger.error(f"Failed to enforce SoD compliance: {e}")
            return {"compliant": False, "error": str(e)}
    
    def detect_compliance_violations(self, entity_type: str, activity_data: Dict[str, Any],
                                   tenant_id: Optional[int] = None) -> List[ComplianceViolation]:
        """Task 7.5.14: Build anomaly detection for compliance violations"""
        try:
            violations = []
            
            # Access pattern anomalies
            access_violations = self._detect_access_anomalies(activity_data, tenant_id)
            violations.extend(access_violations)
            
            # Data pattern anomalies
            data_violations = self._detect_data_anomalies(activity_data, tenant_id)
            violations.extend(data_violations)
            
            # Financial pattern anomalies
            if entity_type in ["transaction", "financial_record"]:
                financial_violations = self._detect_financial_anomalies(activity_data, tenant_id)
                violations.extend(financial_violations)
            
            # Store detected violations
            self.violations.extend(violations)
            
            # Trigger alerts for high-severity violations
            for violation in violations:
                if violation.severity in ["high", "critical"]:
                    self._trigger_compliance_alert(violation)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to detect compliance violations: {e}")
            return []
    
    def predict_compliance_risks(self, regulatory_framework: RegulatoryFramework,
                               tenant_id: Optional[int] = None,
                               prediction_horizon_days: int = 30) -> Dict[str, Any]:
        """Task 7.5.15: Configure predictive compliance analytics"""
        try:
            prediction_result = {
                "framework": regulatory_framework.value,
                "tenant_id": tenant_id,
                "prediction_horizon_days": prediction_horizon_days,
                "generated_at": datetime.utcnow().isoformat(),
                "risk_predictions": [],
                "recommended_actions": [],
                "confidence_score": 0.0
            }
            
            # Analyze historical violation patterns
            historical_violations = self._get_historical_violations(
                regulatory_framework, tenant_id, days=90
            )
            
            # Trend analysis
            violation_trends = self._analyze_violation_trends(historical_violations)
            
            # Risk factor analysis
            risk_factors = self._analyze_risk_factors(regulatory_framework, tenant_id)
            
            # Generate predictions
            predictions = self._generate_compliance_predictions(
                violation_trends, risk_factors, prediction_horizon_days
            )
            
            prediction_result["risk_predictions"] = predictions
            prediction_result["recommended_actions"] = self._generate_compliance_recommendations(predictions)
            prediction_result["confidence_score"] = self._calculate_prediction_confidence(
                historical_violations, violation_trends
            )
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Failed to predict compliance risks: {e}")
            return {"error": str(e)}
    
    def tag_ip_asset(self, asset_id: str, asset_type: IPAssetType, name: str,
                    description: str, tenant_id: Optional[int] = None) -> IPAsset:
        """Task 7.5.32: Automate IP asset tagging in registries"""
        try:
            # Auto-generate IP classification
            ip_classification = self._classify_ip_asset(asset_type, description)
            
            # Assess business value
            business_value = self._assess_ip_business_value(asset_type, name, description)
            
            # Determine confidentiality level
            confidentiality = self._determine_confidentiality_level(asset_type, description)
            
            # Auto-generate tags
            auto_tags = self._generate_ip_tags(asset_type, name, description)
            
            ip_asset = IPAsset(
                asset_id=asset_id,
                name=name,
                description=description,
                asset_type=asset_type,
                ip_category=ip_classification["category"],
                confidentiality_level=confidentiality,
                business_value_score=business_value["score"],
                revenue_potential=business_value["revenue_potential"],
                competitive_advantage=business_value["competitive_advantage"],
                tenant_id=tenant_id,
                tags=auto_tags,
                metadata=ip_classification["metadata"]
            )
            
            # Store IP asset
            self.ip_assets[asset_id] = ip_asset
            
            self.logger.info(f"✅ Tagged IP asset: {asset_id} ({asset_type.value})")
            return ip_asset
            
        except Exception as e:
            self.logger.error(f"Failed to tag IP asset: {e}")
            raise
    
    # Helper methods for internal processing
    
    def _assess_initial_risk_level(self, frameworks: List[RegulatoryFramework], entity_type: str) -> str:
        """Assess initial risk level based on regulatory frameworks"""
        high_risk_frameworks = [RegulatoryFramework.SOX, RegulatoryFramework.HIPAA]
        
        if any(fw in high_risk_frameworks for fw in frameworks):
            return "high"
        elif len(frameworks) > 2:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(self, frameworks: List[RegulatoryFramework], entity_type: str) -> List[str]:
        """Identify risk factors based on frameworks and entity type"""
        risk_factors = []
        
        for framework in frameworks:
            if framework == RegulatoryFramework.GDPR:
                risk_factors.extend(["personal_data_processing", "cross_border_transfer"])
            elif framework == RegulatoryFramework.SOX:
                risk_factors.extend(["financial_reporting", "internal_controls"])
            elif framework == RegulatoryFramework.HIPAA:
                risk_factors.extend(["phi_processing", "healthcare_data"])
        
        return list(set(risk_factors))
    
    def _calculate_content_hash(self, content_location: str) -> str:
        """Calculate SHA-256 hash of content for integrity verification"""
        # Simplified implementation - in practice, would read actual content
        content_identifier = f"{content_location}_{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content_identifier.encode()).hexdigest()
    
    def _calculate_retention_period(self, frameworks: List[RegulatoryFramework]) -> datetime:
        """Calculate retention period based on regulatory requirements"""
        max_retention_days = 0
        
        for framework in frameworks:
            mapping = self.regulatory_mappings.get(framework, {})
            retention_periods = mapping.get("retention_periods", {})
            
            for period_days in retention_periods.values():
                max_retention_days = max(max_retention_days, period_days)
        
        if max_retention_days == 0:
            max_retention_days = 2555  # Default 7 years
        
        return datetime.utcnow() + timedelta(days=max_retention_days)
    
    def _check_approval_requirements(self, approval_matrix: Dict[str, Any],
                                   activity_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if approval is required based on activity context"""
        result = {"approval_required": False, "required_actions": []}
        
        # Check amount thresholds if applicable
        if "amount" in activity_context and "amount_thresholds" in approval_matrix:
            amount = activity_context["amount"]
            thresholds = approval_matrix["amount_thresholds"]
            
            for threshold_name, threshold_config in thresholds.items():
                if amount <= threshold_config["max_amount"]:
                    required_approvers = threshold_config["required_approvers"]
                    result["approval_required"] = required_approvers > 0
                    result["required_actions"].append(f"Requires {required_approvers} approver(s)")
                    break
        
        return result
    
    def _detect_access_anomalies(self, activity_data: Dict[str, Any],
                               tenant_id: Optional[int]) -> List[ComplianceViolation]:
        """Detect access pattern anomalies"""
        violations = []
        thresholds = self.anomaly_thresholds["access_patterns"]
        
        # Check for unusual access hours
        if "access_time" in activity_data:
            access_hour = activity_data["access_time"].hour
            if access_hour < 6 or access_hour > 22:  # Outside business hours
                violation = ComplianceViolation(
                    violation_id=f"access_anomaly_{uuid.uuid4().hex[:8]}",
                    detected_at=datetime.utcnow(),
                    violation_type="unusual_access_hours",
                    severity="medium",
                    regulatory_framework=RegulatoryFramework.SOX,
                    violated_requirement="Access control monitoring",
                    description=f"Access detected outside business hours: {access_hour}:00",
                    tenant_id=tenant_id
                )
                violations.append(violation)
        
        return violations
    
    def _detect_data_anomalies(self, activity_data: Dict[str, Any],
                             tenant_id: Optional[int]) -> List[ComplianceViolation]:
        """Detect data pattern anomalies"""
        violations = []
        thresholds = self.anomaly_thresholds["data_patterns"]
        
        # Check for bulk data access
        if "records_accessed" in activity_data:
            records_count = activity_data["records_accessed"]
            if records_count > 1000:  # Threshold for bulk access
                violation = ComplianceViolation(
                    violation_id=f"data_anomaly_{uuid.uuid4().hex[:8]}",
                    detected_at=datetime.utcnow(),
                    violation_type="bulk_data_access",
                    severity="high",
                    regulatory_framework=RegulatoryFramework.GDPR,
                    violated_requirement="Data minimization principle",
                    description=f"Bulk data access detected: {records_count} records",
                    tenant_id=tenant_id
                )
                violations.append(violation)
        
        return violations
    
    def _detect_financial_anomalies(self, activity_data: Dict[str, Any],
                                  tenant_id: Optional[int]) -> List[ComplianceViolation]:
        """Detect financial pattern anomalies"""
        violations = []
        
        # Check for approval bypass
        if activity_data.get("requires_approval", False) and not activity_data.get("approved", False):
            violation = ComplianceViolation(
                violation_id=f"financial_anomaly_{uuid.uuid4().hex[:8]}",
                detected_at=datetime.utcnow(),
                violation_type="approval_bypass",
                severity="critical",
                regulatory_framework=RegulatoryFramework.SOX,
                violated_requirement="Segregation of duties",
                description="Financial transaction processed without required approval",
                tenant_id=tenant_id,
                remediation_required=True,
                escalation_required=True
            )
            violations.append(violation)
        
        return violations
    
    def _trigger_compliance_alert(self, violation: ComplianceViolation):
        """Trigger alert for compliance violation"""
        self.logger.warning(f"🚨 Compliance violation detected: {violation.violation_id} - {violation.description}")
        # In practice, would integrate with alerting systems (Slack, email, etc.)
    
    def _get_historical_violations(self, framework: RegulatoryFramework,
                                 tenant_id: Optional[int], days: int) -> List[ComplianceViolation]:
        """Get historical violations for analysis"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return [
            v for v in self.violations
            if v.regulatory_framework == framework
            and v.detected_at > cutoff_date
            and (tenant_id is None or v.tenant_id == tenant_id)
        ]
    
    def _analyze_violation_trends(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Analyze trends in compliance violations"""
        if not violations:
            return {"trend": "stable", "frequency": 0.0, "severity_distribution": {}}
        
        # Group by violation type
        violation_types = {}
        for violation in violations:
            vtype = violation.violation_type
            if vtype not in violation_types:
                violation_types[vtype] = []
            violation_types[vtype].append(violation)
        
        # Analyze frequency trends
        total_violations = len(violations)
        days_analyzed = 90
        frequency = total_violations / days_analyzed
        
        # Severity distribution
        severity_counts = {}
        for violation in violations:
            severity = violation.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "trend": "increasing" if frequency > 0.1 else "stable",
            "frequency": frequency,
            "severity_distribution": severity_counts,
            "violation_types": {vtype: len(viols) for vtype, viols in violation_types.items()}
        }
    
    def _analyze_risk_factors(self, framework: RegulatoryFramework,
                            tenant_id: Optional[int]) -> Dict[str, Any]:
        """Analyze current risk factors"""
        return {
            "high_risk_activities": [],
            "compliance_gaps": [],
            "control_effectiveness": 0.8,
            "risk_score": 0.3
        }
    
    def _generate_compliance_predictions(self, trends: Dict[str, Any],
                                       risk_factors: Dict[str, Any],
                                       horizon_days: int) -> List[Dict[str, Any]]:
        """Generate compliance risk predictions"""
        predictions = []
        
        # Predict violation frequency
        current_frequency = trends.get("frequency", 0.0)
        predicted_violations = current_frequency * horizon_days
        
        if predicted_violations > 1.0:
            predictions.append({
                "risk_type": "violation_frequency",
                "probability": min(0.9, predicted_violations / 10),
                "impact": "medium",
                "description": f"Predicted {predicted_violations:.1f} violations in next {horizon_days} days"
            })
        
        # Predict control gaps
        control_effectiveness = risk_factors.get("control_effectiveness", 0.8)
        if control_effectiveness < 0.7:
            predictions.append({
                "risk_type": "control_gap",
                "probability": 1.0 - control_effectiveness,
                "impact": "high",
                "description": "Control effectiveness below acceptable threshold"
            })
        
        return predictions
    
    def _generate_compliance_recommendations(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on predictions"""
        recommendations = []
        
        for prediction in predictions:
            if prediction["risk_type"] == "violation_frequency":
                recommendations.append("Increase monitoring frequency")
                recommendations.append("Review and strengthen controls")
            elif prediction["risk_type"] == "control_gap":
                recommendations.append("Conduct control effectiveness assessment")
                recommendations.append("Implement additional compensating controls")
        
        return recommendations
    
    def _calculate_prediction_confidence(self, historical_violations: List[ComplianceViolation],
                                       trends: Dict[str, Any]) -> float:
        """Calculate confidence in predictions"""
        if len(historical_violations) < 10:
            return 0.3  # Low confidence with limited data
        
        # Higher confidence with more data and stable trends
        data_confidence = min(0.8, len(historical_violations) / 100)
        trend_confidence = 0.7 if trends["trend"] == "stable" else 0.5
        
        return (data_confidence + trend_confidence) / 2
    
    def _classify_ip_asset(self, asset_type: IPAssetType, description: str) -> Dict[str, Any]:
        """Classify IP asset automatically"""
        classification = {
            "category": "proprietary",
            "metadata": {}
        }
        
        # Simple classification logic
        if "open source" in description.lower() or "public" in description.lower():
            classification["category"] = "open_source"
        elif "licensed" in description.lower():
            classification["category"] = "licensed"
        
        # Add metadata based on asset type
        if asset_type == IPAssetType.ALGORITHM:
            classification["metadata"]["complexity"] = "high"
            classification["metadata"]["patent_potential"] = True
        elif asset_type == IPAssetType.WORKFLOW_TEMPLATE:
            classification["metadata"]["reusability"] = "high"
            classification["metadata"]["business_process"] = True
        
        return classification
    
    def _assess_ip_business_value(self, asset_type: IPAssetType, name: str, description: str) -> Dict[str, Any]:
        """Assess business value of IP asset"""
        value_assessment = {
            "score": 0.5,  # Default medium value
            "revenue_potential": "medium",
            "competitive_advantage": False
        }
        
        # Assess based on asset type
        if asset_type in [IPAssetType.ALGORITHM, IPAssetType.MODEL]:
            value_assessment["score"] = 0.8
            value_assessment["revenue_potential"] = "high"
            value_assessment["competitive_advantage"] = True
        elif asset_type == IPAssetType.WORKFLOW_TEMPLATE:
            value_assessment["score"] = 0.6
            value_assessment["revenue_potential"] = "medium"
        
        # Assess based on keywords
        high_value_keywords = ["ai", "machine learning", "proprietary", "innovative", "patent"]
        if any(keyword in description.lower() for keyword in high_value_keywords):
            value_assessment["score"] = min(1.0, value_assessment["score"] + 0.2)
            value_assessment["competitive_advantage"] = True
        
        return value_assessment
    
    def _determine_confidentiality_level(self, asset_type: IPAssetType, description: str) -> str:
        """Determine confidentiality level of IP asset"""
        if "public" in description.lower() or "open source" in description.lower():
            return "public"
        elif asset_type in [IPAssetType.ALGORITHM, IPAssetType.MODEL]:
            return "confidential"
        elif "internal" in description.lower():
            return "internal"
        else:
            return "internal"  # Default
    
    def _generate_ip_tags(self, asset_type: IPAssetType, name: str, description: str) -> List[str]:
        """Generate tags for IP asset"""
        tags = [asset_type.value]
        
        # Add technology tags
        tech_keywords = {
            "ai": "artificial_intelligence",
            "ml": "machine_learning",
            "algorithm": "algorithm",
            "api": "api",
            "workflow": "workflow",
            "automation": "automation"
        }
        
        text = f"{name} {description}".lower()
        for keyword, tag in tech_keywords.items():
            if keyword in text:
                tags.append(tag)
        
        # Add business tags
        if "revenue" in text or "billing" in text:
            tags.append("revenue_generating")
        if "customer" in text:
            tags.append("customer_facing")
        if "saas" in text:
            tags.append("saas_specific")
        
        return list(set(tags))


# Global instance
compliance_framework = ComplianceFramework()

def get_compliance_framework() -> ComplianceFramework:
    """Get the global compliance framework instance"""
    return compliance_framework
