"""
Task 7.3-T54: Compliance Overlays (RBI/SOX/HIPAA fields)
Regulator-ready overlay schemas enforced at publish
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

import asyncpg


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "SOX"
    GDPR = "GDPR"
    RBI = "RBI"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"
    IRDAI = "IRDAI"
    BASEL_III = "BASEL_III"
    DPDP = "DPDP"
    CCPA = "CCPA"


class ComplianceLevel(Enum):
    """Compliance enforcement levels"""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    AUDIT_ONLY = "audit_only"


class FieldType(Enum):
    """Types of compliance fields"""
    METADATA = "metadata"
    VALIDATION_RULE = "validation_rule"
    AUDIT_FIELD = "audit_field"
    RETENTION_POLICY = "retention_policy"
    ACCESS_CONTROL = "access_control"
    ENCRYPTION_RULE = "encryption_rule"


@dataclass
class ComplianceField:
    """Individual compliance field requirement"""
    field_id: str
    field_name: str
    field_type: FieldType
    framework: ComplianceFramework
    compliance_level: ComplianceLevel
    
    # Field specification
    data_type: str = "string"
    required: bool = True
    default_value: Optional[Any] = None
    validation_rules: List[str] = field(default_factory=list)
    
    # Documentation
    description: str = ""
    regulation_reference: str = ""
    implementation_notes: str = ""
    
    # Lifecycle
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "field_type": self.field_type.value,
            "framework": self.framework.value,
            "compliance_level": self.compliance_level.value,
            "data_type": self.data_type,
            "required": self.required,
            "default_value": self.default_value,
            "validation_rules": self.validation_rules,
            "description": self.description,
            "regulation_reference": self.regulation_reference,
            "implementation_notes": self.implementation_notes,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None
        }


@dataclass
class ComplianceOverlay:
    """Complete compliance overlay for a framework"""
    overlay_id: str
    framework: ComplianceFramework
    version: str
    name: str
    description: str
    
    # Fields and rules
    fields: List[ComplianceField] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    audit_requirements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Applicability
    applicable_industries: List[str] = field(default_factory=list)
    applicable_regions: List[str] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    effective_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    review_date: Optional[datetime] = None
    
    @property
    def required_fields(self) -> List[ComplianceField]:
        return [f for f in self.fields if f.compliance_level == ComplianceLevel.REQUIRED]
    
    @property
    def is_active(self) -> bool:
        now = datetime.now(timezone.utc)
        return (self.effective_date <= now and 
                (self.review_date is None or self.review_date > now))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlay_id": self.overlay_id,
            "framework": self.framework.value,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
            "validation_rules": self.validation_rules,
            "audit_requirements": self.audit_requirements,
            "applicable_industries": self.applicable_industries,
            "applicable_regions": self.applicable_regions,
            "created_at": self.created_at.isoformat(),
            "effective_date": self.effective_date.isoformat(),
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "is_active": self.is_active
        }


class ComplianceOverlayManager:
    """Manager for compliance overlays"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.overlays: Dict[str, ComplianceOverlay] = {}
        self._initialize_standard_overlays()
    
    def _initialize_standard_overlays(self) -> None:
        """Initialize standard compliance overlays"""
        
        # SOX Compliance Overlay
        sox_overlay = self._create_sox_overlay()
        self.overlays[sox_overlay.overlay_id] = sox_overlay
        
        # GDPR Compliance Overlay
        gdpr_overlay = self._create_gdpr_overlay()
        self.overlays[gdpr_overlay.overlay_id] = gdpr_overlay
        
        # RBI Compliance Overlay
        rbi_overlay = self._create_rbi_overlay()
        self.overlays[rbi_overlay.overlay_id] = rbi_overlay
        
        # HIPAA Compliance Overlay
        hipaa_overlay = self._create_hipaa_overlay()
        self.overlays[hipaa_overlay.overlay_id] = hipaa_overlay
        
        # PCI DSS Compliance Overlay
        pci_overlay = self._create_pci_dss_overlay()
        self.overlays[pci_overlay.overlay_id] = pci_overlay
    
    def _create_sox_overlay(self) -> ComplianceOverlay:
        """Create SOX compliance overlay"""
        
        overlay = ComplianceOverlay(
            overlay_id="sox_2024_v1",
            framework=ComplianceFramework.SOX,
            version="2024.1",
            name="Sarbanes-Oxley Act Compliance",
            description="SOX compliance requirements for financial reporting and internal controls",
            applicable_industries=["Banking", "Insurance", "SaaS", "E-commerce"],
            applicable_regions=["US", "Global"]
        )
        
        # SOX-specific fields
        overlay.fields = [
            ComplianceField(
                field_id="sox_control_id",
                field_name="sox_control_identification",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.SOX,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="string",
                description="Unique identifier for SOX internal control",
                regulation_reference="SOX Section 404",
                validation_rules=["pattern:^SOX-[A-Z0-9]{6,12}$"]
            ),
            ComplianceField(
                field_id="sox_control_owner",
                field_name="control_owner",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.SOX,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="string",
                description="Person responsible for the control",
                regulation_reference="SOX Section 302"
            ),
            ComplianceField(
                field_id="sox_testing_frequency",
                field_name="testing_frequency",
                field_type=FieldType.VALIDATION_RULE,
                framework=ComplianceFramework.SOX,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="enum",
                validation_rules=["enum:quarterly,semi-annual,annual"],
                description="Frequency of control testing",
                regulation_reference="SOX Section 404"
            ),
            ComplianceField(
                field_id="sox_segregation_duties",
                field_name="segregation_of_duties_verified",
                field_type=FieldType.AUDIT_FIELD,
                framework=ComplianceFramework.SOX,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Verification that duties are properly segregated",
                regulation_reference="SOX Section 404"
            ),
            ComplianceField(
                field_id="sox_audit_trail",
                field_name="audit_trail_complete",
                field_type=FieldType.AUDIT_FIELD,
                framework=ComplianceFramework.SOX,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Complete audit trail maintained",
                regulation_reference="SOX Section 404"
            )
        ]
        
        # SOX validation rules
        overlay.validation_rules = [
            {
                "rule_id": "sox_dual_approval",
                "description": "Financial transactions require dual approval",
                "condition": "workflow_type == 'financial' AND amount > 10000",
                "requirement": "approval_count >= 2",
                "severity": "critical"
            },
            {
                "rule_id": "sox_change_control",
                "description": "All system changes must be documented and approved",
                "condition": "workflow_type == 'system_change'",
                "requirement": "change_approval_documented == true",
                "severity": "high"
            }
        ]
        
        # SOX audit requirements
        overlay.audit_requirements = [
            {
                "requirement_id": "sox_quarterly_review",
                "description": "Quarterly review of all financial controls",
                "frequency": "quarterly",
                "evidence_required": ["control_test_results", "exception_reports", "remediation_plans"]
            },
            {
                "requirement_id": "sox_annual_certification",
                "description": "Annual management certification of internal controls",
                "frequency": "annual",
                "evidence_required": ["management_assertion", "auditor_opinion", "deficiency_reports"]
            }
        ]
        
        return overlay
    
    def _create_gdpr_overlay(self) -> ComplianceOverlay:
        """Create GDPR compliance overlay"""
        
        overlay = ComplianceOverlay(
            overlay_id="gdpr_2024_v1",
            framework=ComplianceFramework.GDPR,
            version="2024.1",
            name="General Data Protection Regulation",
            description="GDPR compliance requirements for data protection and privacy",
            applicable_industries=["SaaS", "E-commerce", "Healthcare", "Banking"],
            applicable_regions=["EU", "EEA", "Global"]
        )
        
        overlay.fields = [
            ComplianceField(
                field_id="gdpr_data_category",
                field_name="personal_data_category",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.GDPR,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="enum",
                validation_rules=["enum:personal,sensitive,special_category,none"],
                description="Category of personal data processed",
                regulation_reference="GDPR Article 4"
            ),
            ComplianceField(
                field_id="gdpr_lawful_basis",
                field_name="lawful_basis_processing",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.GDPR,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="enum",
                validation_rules=["enum:consent,contract,legal_obligation,vital_interests,public_task,legitimate_interests"],
                description="Lawful basis for processing personal data",
                regulation_reference="GDPR Article 6"
            ),
            ComplianceField(
                field_id="gdpr_retention_period",
                field_name="data_retention_days",
                field_type=FieldType.RETENTION_POLICY,
                framework=ComplianceFramework.GDPR,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="integer",
                validation_rules=["min:1", "max:2555"],  # Max ~7 years
                description="Data retention period in days",
                regulation_reference="GDPR Article 5(1)(e)"
            ),
            ComplianceField(
                field_id="gdpr_dpo_contact",
                field_name="data_protection_officer",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.GDPR,
                compliance_level=ComplianceLevel.RECOMMENDED,
                data_type="string",
                description="Contact information for Data Protection Officer",
                regulation_reference="GDPR Article 37"
            )
        ]
        
        overlay.validation_rules = [
            {
                "rule_id": "gdpr_consent_required",
                "description": "Explicit consent required for sensitive data",
                "condition": "personal_data_category == 'sensitive'",
                "requirement": "lawful_basis_processing == 'consent' AND consent_explicit == true",
                "severity": "critical"
            },
            {
                "rule_id": "gdpr_data_minimization",
                "description": "Only necessary data should be collected",
                "condition": "personal_data_category != 'none'",
                "requirement": "data_necessity_justified == true",
                "severity": "high"
            }
        ]
        
        return overlay
    
    def _create_rbi_overlay(self) -> ComplianceOverlay:
        """Create RBI compliance overlay"""
        
        overlay = ComplianceOverlay(
            overlay_id="rbi_2024_v1",
            framework=ComplianceFramework.RBI,
            version="2024.1",
            name="Reserve Bank of India Regulations",
            description="RBI compliance requirements for banking and financial services",
            applicable_industries=["Banking", "FinTech", "Insurance"],
            applicable_regions=["India"]
        )
        
        overlay.fields = [
            ComplianceField(
                field_id="rbi_data_localization",
                field_name="data_stored_in_india",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.RBI,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Confirmation that payment data is stored in India",
                regulation_reference="RBI Circular on Storage of Payment System Data"
            ),
            ComplianceField(
                field_id="rbi_kyc_compliance",
                field_name="kyc_verification_complete",
                field_type=FieldType.AUDIT_FIELD,
                framework=ComplianceFramework.RBI,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="KYC verification completed as per RBI norms",
                regulation_reference="RBI Master Direction on KYC"
            ),
            ComplianceField(
                field_id="rbi_aml_check",
                field_name="aml_screening_performed",
                field_type=FieldType.AUDIT_FIELD,
                framework=ComplianceFramework.RBI,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="AML screening performed",
                regulation_reference="RBI Guidelines on AML/CFT"
            ),
            ComplianceField(
                field_id="rbi_transaction_limit",
                field_name="transaction_within_limits",
                field_type=FieldType.VALIDATION_RULE,
                framework=ComplianceFramework.RBI,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Transaction within prescribed limits",
                regulation_reference="RBI Directions on Payment Systems"
            )
        ]
        
        return overlay
    
    def _create_hipaa_overlay(self) -> ComplianceOverlay:
        """Create HIPAA compliance overlay"""
        
        overlay = ComplianceOverlay(
            overlay_id="hipaa_2024_v1",
            framework=ComplianceFramework.HIPAA,
            version="2024.1",
            name="Health Insurance Portability and Accountability Act",
            description="HIPAA compliance requirements for healthcare data protection",
            applicable_industries=["Healthcare", "Insurance"],
            applicable_regions=["US"]
        )
        
        overlay.fields = [
            ComplianceField(
                field_id="hipaa_phi_category",
                field_name="protected_health_info_category",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.HIPAA,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="enum",
                validation_rules=["enum:phi,non_phi,de_identified"],
                description="Category of health information",
                regulation_reference="HIPAA Privacy Rule 45 CFR 164.501"
            ),
            ComplianceField(
                field_id="hipaa_minimum_necessary",
                field_name="minimum_necessary_standard_met",
                field_type=FieldType.VALIDATION_RULE,
                framework=ComplianceFramework.HIPAA,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Minimum necessary standard applied",
                regulation_reference="HIPAA Privacy Rule 45 CFR 164.502(b)"
            ),
            ComplianceField(
                field_id="hipaa_encryption_required",
                field_name="data_encrypted_at_rest",
                field_type=FieldType.ENCRYPTION_RULE,
                framework=ComplianceFramework.HIPAA,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="PHI encrypted at rest",
                regulation_reference="HIPAA Security Rule 45 CFR 164.312(a)(2)(iv)"
            ),
            ComplianceField(
                field_id="hipaa_access_log",
                field_name="access_logging_enabled",
                field_type=FieldType.AUDIT_FIELD,
                framework=ComplianceFramework.HIPAA,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Access to PHI is logged",
                regulation_reference="HIPAA Security Rule 45 CFR 164.312(b)"
            )
        ]
        
        return overlay
    
    def _create_pci_dss_overlay(self) -> ComplianceOverlay:
        """Create PCI DSS compliance overlay"""
        
        overlay = ComplianceOverlay(
            overlay_id="pci_dss_2024_v1",
            framework=ComplianceFramework.PCI_DSS,
            version="4.0",
            name="Payment Card Industry Data Security Standard",
            description="PCI DSS requirements for payment card data protection",
            applicable_industries=["E-commerce", "Banking", "SaaS"],
            applicable_regions=["Global"]
        )
        
        overlay.fields = [
            ComplianceField(
                field_id="pci_cardholder_data",
                field_name="cardholder_data_present",
                field_type=FieldType.METADATA,
                framework=ComplianceFramework.PCI_DSS,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Workflow processes cardholder data",
                regulation_reference="PCI DSS Requirement 3"
            ),
            ComplianceField(
                field_id="pci_encryption_transit",
                field_name="data_encrypted_in_transit",
                field_type=FieldType.ENCRYPTION_RULE,
                framework=ComplianceFramework.PCI_DSS,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Cardholder data encrypted in transit",
                regulation_reference="PCI DSS Requirement 4"
            ),
            ComplianceField(
                field_id="pci_access_control",
                field_name="access_control_implemented",
                field_type=FieldType.ACCESS_CONTROL,
                framework=ComplianceFramework.PCI_DSS,
                compliance_level=ComplianceLevel.REQUIRED,
                data_type="boolean",
                description="Access control measures implemented",
                regulation_reference="PCI DSS Requirement 7"
            )
        ]
        
        return overlay
    
    async def get_applicable_overlays(
        self,
        industry: str,
        region: str,
        workflow_metadata: Dict[str, Any]
    ) -> List[ComplianceOverlay]:
        """Get applicable compliance overlays for a workflow"""
        
        applicable_overlays = []
        
        # Check in-memory overlays
        for overlay in self.overlays.values():
            if self._is_overlay_applicable(overlay, industry, region, workflow_metadata):
                applicable_overlays.append(overlay)
        
        # Check database overlays if available
        if self.db_pool:
            try:
                db_overlays = await self._get_db_overlays(industry, region)
                applicable_overlays.extend(db_overlays)
            except Exception:
                pass
        
        return applicable_overlays
    
    def _is_overlay_applicable(
        self,
        overlay: ComplianceOverlay,
        industry: str,
        region: str,
        workflow_metadata: Dict[str, Any]
    ) -> bool:
        """Check if overlay is applicable"""
        
        # Check if overlay is active
        if not overlay.is_active:
            return False
        
        # Check industry applicability
        if overlay.applicable_industries and industry not in overlay.applicable_industries:
            return False
        
        # Check region applicability
        if overlay.applicable_regions and region not in overlay.applicable_regions:
            # Check if "Global" is supported
            if "Global" not in overlay.applicable_regions:
                return False
        
        # Additional workflow-specific checks
        workflow_type = workflow_metadata.get("workflow_type", "")
        risk_level = workflow_metadata.get("risk_level", "medium")
        
        # High-risk workflows require more compliance
        if risk_level in ["high", "critical"]:
            return True
        
        # Financial workflows require SOX/RBI
        if "financial" in workflow_type.lower():
            return overlay.framework in [ComplianceFramework.SOX, ComplianceFramework.RBI]
        
        # Healthcare workflows require HIPAA
        if "health" in workflow_type.lower():
            return overlay.framework == ComplianceFramework.HIPAA
        
        # Payment workflows require PCI DSS
        if "payment" in workflow_type.lower():
            return overlay.framework == ComplianceFramework.PCI_DSS
        
        return True
    
    async def validate_workflow_compliance(
        self,
        workflow_data: Dict[str, Any],
        workflow_metadata: Dict[str, Any],
        industry: str,
        region: str
    ) -> Dict[str, Any]:
        """Validate workflow against applicable compliance overlays"""
        
        # Get applicable overlays
        overlays = await self.get_applicable_overlays(industry, region, workflow_metadata)
        
        validation_results = {
            "workflow_id": workflow_metadata.get("workflow_id"),
            "compliance_status": "compliant",
            "applicable_frameworks": [o.framework.value for o in overlays],
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Validate against each overlay
        for overlay in overlays:
            overlay_result = self._validate_against_overlay(
                workflow_data, workflow_metadata, overlay
            )
            
            # Merge results
            validation_results["violations"].extend(overlay_result["violations"])
            validation_results["warnings"].extend(overlay_result["warnings"])
            validation_results["recommendations"].extend(overlay_result["recommendations"])
        
        # Determine overall compliance status
        if validation_results["violations"]:
            critical_violations = [v for v in validation_results["violations"] if v["severity"] == "critical"]
            if critical_violations:
                validation_results["compliance_status"] = "non_compliant"
            else:
                validation_results["compliance_status"] = "partially_compliant"
        
        return validation_results
    
    def _validate_against_overlay(
        self,
        workflow_data: Dict[str, Any],
        workflow_metadata: Dict[str, Any],
        overlay: ComplianceOverlay
    ) -> Dict[str, Any]:
        """Validate workflow against a specific overlay"""
        
        result = {
            "framework": overlay.framework.value,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check required fields
        for field in overlay.required_fields:
            field_value = workflow_metadata.get(field.field_name)
            
            if field_value is None and field.required:
                result["violations"].append({
                    "field": field.field_name,
                    "framework": overlay.framework.value,
                    "severity": "critical",
                    "message": f"Required compliance field '{field.field_name}' is missing",
                    "regulation_reference": field.regulation_reference
                })
            elif field_value is not None:
                # Validate field value
                validation_errors = self._validate_field_value(field, field_value)
                for error in validation_errors:
                    result["violations"].append({
                        "field": field.field_name,
                        "framework": overlay.framework.value,
                        "severity": "high",
                        "message": error,
                        "regulation_reference": field.regulation_reference
                    })
        
        # Check validation rules
        for rule in overlay.validation_rules:
            rule_result = self._evaluate_validation_rule(
                rule, workflow_data, workflow_metadata
            )
            
            if not rule_result["passed"]:
                result["violations"].append({
                    "rule": rule["rule_id"],
                    "framework": overlay.framework.value,
                    "severity": rule.get("severity", "medium"),
                    "message": rule_result["message"],
                    "regulation_reference": rule.get("regulation_reference", "")
                })
        
        return result
    
    def _validate_field_value(self, field: ComplianceField, value: Any) -> List[str]:
        """Validate field value against validation rules"""
        
        errors = []
        
        for rule in field.validation_rules:
            if rule.startswith("pattern:"):
                import re
                pattern = rule[8:]  # Remove "pattern:" prefix
                if not re.match(pattern, str(value)):
                    errors.append(f"Value '{value}' does not match required pattern '{pattern}'")
            
            elif rule.startswith("enum:"):
                allowed_values = rule[5:].split(",")  # Remove "enum:" prefix
                if str(value) not in allowed_values:
                    errors.append(f"Value '{value}' not in allowed values: {allowed_values}")
            
            elif rule.startswith("min:"):
                min_val = int(rule[4:])  # Remove "min:" prefix
                if isinstance(value, (int, float)) and value < min_val:
                    errors.append(f"Value {value} is below minimum {min_val}")
            
            elif rule.startswith("max:"):
                max_val = int(rule[4:])  # Remove "max:" prefix
                if isinstance(value, (int, float)) and value > max_val:
                    errors.append(f"Value {value} exceeds maximum {max_val}")
        
        return errors
    
    def _evaluate_validation_rule(
        self,
        rule: Dict[str, Any],
        workflow_data: Dict[str, Any],
        workflow_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a validation rule"""
        
        # Simplified rule evaluation
        condition = rule.get("condition", "")
        requirement = rule.get("requirement", "")
        
        # This would normally use a proper expression evaluator
        # For now, return a simple check
        passed = True
        message = f"Rule '{rule['rule_id']}' evaluation not implemented"
        
        return {
            "passed": passed,
            "message": message
        }
    
    async def _get_db_overlays(self, industry: str, region: str) -> List[ComplianceOverlay]:
        """Get overlays from database"""
        
        if not self.db_pool:
            return []
        
        try:
            query = """
                SELECT * FROM compliance_overlays 
                WHERE is_active = true
                AND (applicable_industries IS NULL OR $1 = ANY(applicable_industries))
                AND (applicable_regions IS NULL OR $2 = ANY(applicable_regions) OR 'Global' = ANY(applicable_regions))
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, industry, region)
                
                overlays = []
                for row in rows:
                    # Deserialize overlay from database
                    overlay_data = json.loads(row["overlay_data"])
                    overlay = self._deserialize_overlay(overlay_data)
                    overlays.append(overlay)
                
                return overlays
        except Exception:
            return []
    
    def _deserialize_overlay(self, overlay_data: Dict[str, Any]) -> ComplianceOverlay:
        """Deserialize overlay from database"""
        
        fields = []
        for field_data in overlay_data.get("fields", []):
            field = ComplianceField(
                field_id=field_data["field_id"],
                field_name=field_data["field_name"],
                field_type=FieldType(field_data["field_type"]),
                framework=ComplianceFramework(field_data["framework"]),
                compliance_level=ComplianceLevel(field_data["compliance_level"]),
                data_type=field_data.get("data_type", "string"),
                required=field_data.get("required", True),
                default_value=field_data.get("default_value"),
                validation_rules=field_data.get("validation_rules", []),
                description=field_data.get("description", ""),
                regulation_reference=field_data.get("regulation_reference", ""),
                implementation_notes=field_data.get("implementation_notes", "")
            )
            fields.append(field)
        
        return ComplianceOverlay(
            overlay_id=overlay_data["overlay_id"],
            framework=ComplianceFramework(overlay_data["framework"]),
            version=overlay_data["version"],
            name=overlay_data["name"],
            description=overlay_data["description"],
            fields=fields,
            validation_rules=overlay_data.get("validation_rules", []),
            audit_requirements=overlay_data.get("audit_requirements", []),
            applicable_industries=overlay_data.get("applicable_industries", []),
            applicable_regions=overlay_data.get("applicable_regions", []),
            created_at=datetime.fromisoformat(overlay_data["created_at"]),
            effective_date=datetime.fromisoformat(overlay_data["effective_date"]),
            review_date=datetime.fromisoformat(overlay_data["review_date"]) if overlay_data.get("review_date") else None
        )
    
    async def store_overlay(self, overlay: ComplianceOverlay) -> None:
        """Store compliance overlay in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO compliance_overlays (
                    overlay_id, framework, version, name, description,
                    overlay_data, applicable_industries, applicable_regions,
                    is_active, effective_date, review_date
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (overlay_id) 
                DO UPDATE SET
                    overlay_data = EXCLUDED.overlay_data,
                    updated_at = NOW()
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    overlay.overlay_id,
                    overlay.framework.value,
                    overlay.version,
                    overlay.name,
                    overlay.description,
                    json.dumps(overlay.to_dict()),
                    overlay.applicable_industries,
                    overlay.applicable_regions,
                    overlay.is_active,
                    overlay.effective_date,
                    overlay.review_date
                )
        except Exception:
            # Log error but don't fail
            pass


# Database schema for compliance overlays
COMPLIANCE_OVERLAYS_SCHEMA_SQL = """
-- Compliance overlays
CREATE TABLE IF NOT EXISTS compliance_overlays (
    overlay_id VARCHAR(100) PRIMARY KEY,
    framework VARCHAR(20) NOT NULL,
    version VARCHAR(20) NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    overlay_data JSONB NOT NULL,
    applicable_industries TEXT[],
    applicable_regions TEXT[],
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    effective_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    review_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_framework CHECK (framework IN ('SOX', 'GDPR', 'RBI', 'HIPAA', 'PCI_DSS', 'IRDAI', 'BASEL_III', 'DPDP', 'CCPA'))
);

-- Compliance validation results
CREATE TABLE IF NOT EXISTS compliance_validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    version_id UUID,
    validation_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    compliance_status VARCHAR(20) NOT NULL,
    applicable_frameworks TEXT[],
    validation_results JSONB NOT NULL,
    
    CONSTRAINT chk_compliance_status CHECK (compliance_status IN ('compliant', 'partially_compliant', 'non_compliant'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_overlays_framework ON compliance_overlays (framework, is_active);
CREATE INDEX IF NOT EXISTS idx_overlays_effective ON compliance_overlays (effective_date, review_date);
CREATE INDEX IF NOT EXISTS idx_validation_results_workflow ON compliance_validation_results (workflow_id, validation_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_validation_results_status ON compliance_validation_results (compliance_status);
"""
