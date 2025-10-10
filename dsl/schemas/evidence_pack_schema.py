# Evidence Pack Schema - Chapter 7.4
# Tasks 7.4-T01 to T50: Canonical evidence schema, cryptographic hashing, digital signatures

import json
import uuid
import hashlib
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import asyncio
import zipfile
import io

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """Types of evidence captured"""
    WORKFLOW_EXECUTION = "workflow_execution"
    POLICY_APPLICATION = "policy_application"
    OVERRIDE_EVENT = "override_event"
    APPROVAL_DECISION = "approval_decision"
    DATA_PROCESSING = "data_processing"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    AUDIT_EVENT = "audit_event"

class EvidenceStatus(Enum):
    """Evidence pack status"""
    DRAFT = "draft"
    SEALED = "sealed"
    SIGNED = "signed"
    ARCHIVED = "archived"
    EXPIRED = "expired"

class ComplianceFramework(Enum):
    """Compliance frameworks for evidence"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    RBI = "rbi"
    IRDAI = "irdai"
    PCI_DSS = "pci_dss"
    BASEL_III = "basel_iii"
    SOLVENCY_II = "solvency_ii"
    DPDP = "dpdp"
    CCPA = "ccpa"

@dataclass
class EvidenceMetadata:
    """Evidence pack metadata"""
    evidence_id: str
    evidence_type: EvidenceType
    created_at: str
    created_by: str
    tenant_id: int
    
    # Linkages
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    trace_id: Optional[str] = None
    policy_id: Optional[str] = None
    override_id: Optional[str] = None
    
    # Classification
    classification: str = "internal"  # public, internal, confidential, restricted
    retention_days: int = 2555  # 7 years default
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Status
    status: EvidenceStatus = EvidenceStatus.DRAFT
    sealed_at: Optional[str] = None
    signed_at: Optional[str] = None
    signed_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['evidence_type'] = self.evidence_type.value
        data['status'] = self.status.value
        data['compliance_frameworks'] = [f.value for f in self.compliance_frameworks]
        return data

@dataclass
class PolicyApplication:
    """Policy application evidence"""
    policy_id: str
    policy_name: str
    policy_version: str
    policy_hash: str
    
    # Application details
    applied_at: str
    applied_by: str
    enforcement_result: str  # allowed, denied, warning
    
    # Context
    input_data_hash: Optional[str] = None
    output_data_hash: Optional[str] = None
    violation_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class OverrideDetails:
    """Override event evidence"""
    override_id: str
    override_type: str
    reason: str
    justification: str
    
    # Approval chain
    requested_by: str
    requested_at: str
    approved_by: str
    approved_at: str
    
    # Impact
    risk_level: str = "medium"
    business_impact: str = ""
    technical_impact: str = ""
    
    # Expiration
    expires_at: Optional[str] = None
    auto_revoke: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class TrustImpactScore:
    """Trust impact scoring for evidence"""
    score_id: str
    baseline_score: float
    current_score: float
    score_delta: float
    
    # Factors
    policy_compliance: float
    override_frequency: float
    sla_adherence: float
    error_rate: float
    
    # Metadata
    calculated_at: str
    calculation_method: str = "weighted_average"
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class IndustryOverlayEvidence:
    """Industry-specific evidence fields"""
    industry_code: str
    overlay_version: str
    
    # SaaS specific
    saas_revenue_impact: Optional[float] = None
    saas_customer_count: Optional[int] = None
    saas_subscription_changes: Optional[Dict[str, Any]] = None
    
    # Banking specific (BFSI)
    bfsi_loan_amount: Optional[float] = None
    bfsi_aml_check_result: Optional[str] = None
    bfsi_kyc_verification: Optional[Dict[str, Any]] = None
    bfsi_sanction_screening: Optional[Dict[str, Any]] = None
    
    # Insurance specific
    insurance_claim_amount: Optional[float] = None
    insurance_solvency_ratio: Optional[float] = None
    insurance_fraud_score: Optional[float] = None
    insurance_regulatory_filing: Optional[Dict[str, Any]] = None
    
    # Healthcare specific
    healthcare_phi_accessed: Optional[bool] = None
    healthcare_patient_count: Optional[int] = None
    healthcare_consent_status: Optional[str] = None
    healthcare_breach_risk: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class EvidencePack:
    """Complete evidence pack"""
    # Core metadata
    metadata: EvidenceMetadata
    
    # Core evidence data
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_context: Dict[str, Any]
    
    # Governance evidence
    policy_applications: List[PolicyApplication] = field(default_factory=list)
    override_details: Optional[OverrideDetails] = None
    trust_impact: Optional[TrustImpactScore] = None
    
    # Industry-specific evidence
    industry_overlay: Optional[IndustryOverlayEvidence] = None
    
    # Attachments and artifacts
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cryptographic integrity
    content_hash: Optional[str] = None
    signature: Optional[str] = None
    signature_algorithm: str = "SHA256withRSA"
    
    # Chain of custody
    custody_chain: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['metadata'] = self.metadata.to_dict()
        data['policy_applications'] = [pa.to_dict() for pa in self.policy_applications]
        if self.override_details:
            data['override_details'] = self.override_details.to_dict()
        if self.trust_impact:
            data['trust_impact'] = self.trust_impact.to_dict()
        if self.industry_overlay:
            data['industry_overlay'] = self.industry_overlay.to_dict()
        return data
    
    def calculate_content_hash(self) -> str:
        """Calculate SHA256 hash of evidence content"""
        # Create deterministic content for hashing
        hash_content = {
            'metadata': self.metadata.to_dict(),
            'input_data_hash': self._hash_data(self.input_data),
            'output_data_hash': self._hash_data(self.output_data),
            'policy_applications': [pa.to_dict() for pa in self.policy_applications],
            'override_details': self.override_details.to_dict() if self.override_details else None,
            'industry_overlay': self.industry_overlay.to_dict() if self.industry_overlay else None
        }
        
        content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content_json.encode()).hexdigest()
    
    def _hash_data(self, data: Any) -> str:
        """Hash arbitrary data consistently"""
        if data is None:
            return "null"
        
        # Convert to JSON and hash
        try:
            json_str = json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"⚠️ Failed to hash data: {e}")
            return hashlib.sha256(str(data).encode()).hexdigest()

class EvidencePackBuilder:
    """
    Evidence Pack Builder - Tasks 7.4-T01 to T15
    Builds evidence packs with cryptographic integrity and digital signatures
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_evidence_pack(
        self,
        evidence_type: EvidenceType,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        created_by: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> EvidencePack:
        """Create new evidence pack"""
        
        evidence_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = EvidenceMetadata(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=created_by,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            execution_id=execution_id,
            trace_id=f"trace_{execution_id}"
        )
        
        # Create evidence pack
        evidence_pack = EvidencePack(
            metadata=metadata,
            input_data=input_data,
            output_data=output_data,
            execution_context=execution_context
        )
        
        # Add initial custody entry
        evidence_pack.custody_chain.append({
            'action': 'created',
            'actor': created_by,
            'timestamp': metadata.created_at,
            'location': 'evidence_builder'
        })
        
        self.logger.info(f"✅ Created evidence pack: {evidence_id}")
        return evidence_pack
    
    def add_policy_application(
        self,
        evidence_pack: EvidencePack,
        policy_id: str,
        policy_name: str,
        policy_version: str,
        policy_hash: str,
        enforcement_result: str,
        applied_by: str,
        violation_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add policy application evidence"""
        
        policy_app = PolicyApplication(
            policy_id=policy_id,
            policy_name=policy_name,
            policy_version=policy_version,
            policy_hash=policy_hash,
            applied_at=datetime.now(timezone.utc).isoformat(),
            applied_by=applied_by,
            enforcement_result=enforcement_result,
            input_data_hash=evidence_pack._hash_data(evidence_pack.input_data),
            output_data_hash=evidence_pack._hash_data(evidence_pack.output_data),
            violation_details=violation_details
        )
        
        evidence_pack.policy_applications.append(policy_app)
        evidence_pack.metadata.policy_id = policy_id
        
        # Add custody entry
        evidence_pack.custody_chain.append({
            'action': 'policy_applied',
            'actor': applied_by,
            'timestamp': policy_app.applied_at,
            'details': f"Policy {policy_id} applied with result: {enforcement_result}"
        })
        
        self.logger.info(f"✅ Added policy application: {policy_id} to evidence {evidence_pack.metadata.evidence_id}")
    
    def add_override_details(
        self,
        evidence_pack: EvidencePack,
        override_id: str,
        override_type: str,
        reason: str,
        justification: str,
        requested_by: str,
        approved_by: str,
        risk_level: str = "medium",
        expires_at: Optional[str] = None
    ) -> None:
        """Add override event evidence"""
        
        now = datetime.now(timezone.utc).isoformat()
        
        override_details = OverrideDetails(
            override_id=override_id,
            override_type=override_type,
            reason=reason,
            justification=justification,
            requested_by=requested_by,
            requested_at=now,
            approved_by=approved_by,
            approved_at=now,
            risk_level=risk_level,
            expires_at=expires_at
        )
        
        evidence_pack.override_details = override_details
        evidence_pack.metadata.override_id = override_id
        
        # Add custody entry
        evidence_pack.custody_chain.append({
            'action': 'override_recorded',
            'actor': approved_by,
            'timestamp': now,
            'details': f"Override {override_id} approved by {approved_by}"
        })
        
        self.logger.info(f"✅ Added override details: {override_id} to evidence {evidence_pack.metadata.evidence_id}")
    
    def add_trust_impact_score(
        self,
        evidence_pack: EvidencePack,
        baseline_score: float,
        current_score: float,
        policy_compliance: float,
        override_frequency: float,
        sla_adherence: float,
        error_rate: float
    ) -> None:
        """Add trust impact scoring"""
        
        score_delta = current_score - baseline_score
        
        trust_impact = TrustImpactScore(
            score_id=str(uuid.uuid4()),
            baseline_score=baseline_score,
            current_score=current_score,
            score_delta=score_delta,
            policy_compliance=policy_compliance,
            override_frequency=override_frequency,
            sla_adherence=sla_adherence,
            error_rate=error_rate,
            calculated_at=datetime.now(timezone.utc).isoformat()
        )
        
        evidence_pack.trust_impact = trust_impact
        
        self.logger.info(f"✅ Added trust impact score: {current_score} (Δ{score_delta:+.2f}) to evidence {evidence_pack.metadata.evidence_id}")
    
    def add_industry_overlay(
        self,
        evidence_pack: EvidencePack,
        industry_code: str,
        overlay_version: str = "1.0.0",
        **industry_fields
    ) -> None:
        """Add industry-specific evidence fields"""
        
        industry_overlay = IndustryOverlayEvidence(
            industry_code=industry_code,
            overlay_version=overlay_version,
            **industry_fields
        )
        
        evidence_pack.industry_overlay = industry_overlay
        
        # Add compliance frameworks based on industry
        if industry_code.lower() == 'saas':
            evidence_pack.metadata.compliance_frameworks.extend([
                ComplianceFramework.SOX, ComplianceFramework.GDPR
            ])
        elif industry_code.lower() == 'banking':
            evidence_pack.metadata.compliance_frameworks.extend([
                ComplianceFramework.RBI, ComplianceFramework.BASEL_III, ComplianceFramework.PCI_DSS
            ])
        elif industry_code.lower() == 'insurance':
            evidence_pack.metadata.compliance_frameworks.extend([
                ComplianceFramework.IRDAI, ComplianceFramework.SOLVENCY_II
            ])
        elif industry_code.lower() == 'healthcare':
            evidence_pack.metadata.compliance_frameworks.extend([
                ComplianceFramework.HIPAA, ComplianceFramework.GDPR
            ])
        
        self.logger.info(f"✅ Added {industry_code} industry overlay to evidence {evidence_pack.metadata.evidence_id}")
    
    def seal_evidence_pack(self, evidence_pack: EvidencePack) -> str:
        """Seal evidence pack with cryptographic hash"""
        
        # Calculate content hash
        content_hash = evidence_pack.calculate_content_hash()
        evidence_pack.content_hash = content_hash
        
        # Update status
        evidence_pack.metadata.status = EvidenceStatus.SEALED
        evidence_pack.metadata.sealed_at = datetime.now(timezone.utc).isoformat()
        
        # Add custody entry
        evidence_pack.custody_chain.append({
            'action': 'sealed',
            'actor': 'evidence_builder',
            'timestamp': evidence_pack.metadata.sealed_at,
            'content_hash': content_hash
        })
        
        self.logger.info(f"✅ Sealed evidence pack: {evidence_pack.metadata.evidence_id} (hash: {content_hash[:8]}...)")
        return content_hash
    
    def sign_evidence_pack(
        self,
        evidence_pack: EvidencePack,
        signer_id: str,
        private_key: Optional[str] = None
    ) -> str:
        """Sign evidence pack with digital signature"""
        
        if evidence_pack.metadata.status != EvidenceStatus.SEALED:
            raise ValueError("Evidence pack must be sealed before signing")
        
        # Mock digital signature (in production, would use real cryptographic signing)
        signature_data = {
            'evidence_id': evidence_pack.metadata.evidence_id,
            'content_hash': evidence_pack.content_hash,
            'signer_id': signer_id,
            'signed_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Create mock signature
        signature_content = json.dumps(signature_data, sort_keys=True)
        signature = base64.b64encode(
            hashlib.sha256(signature_content.encode()).digest()
        ).decode()
        
        evidence_pack.signature = signature
        evidence_pack.metadata.status = EvidenceStatus.SIGNED
        evidence_pack.metadata.signed_at = signature_data['signed_at']
        evidence_pack.metadata.signed_by = signer_id
        
        # Add custody entry
        evidence_pack.custody_chain.append({
            'action': 'signed',
            'actor': signer_id,
            'timestamp': signature_data['signed_at'],
            'signature': signature[:16] + "..."
        })
        
        self.logger.info(f"✅ Signed evidence pack: {evidence_pack.metadata.evidence_id} by {signer_id}")
        return signature

class EvidencePackManager:
    """
    Evidence Pack Manager - Tasks 7.4-T16 to T35
    Manages evidence pack lifecycle, storage, and retrieval
    """
    
    def __init__(self):
        self.evidence_packs: Dict[str, EvidencePack] = {}
        self.builder = EvidencePackBuilder()
        self.logger = logging.getLogger(__name__)
    
    async def create_workflow_evidence(
        self,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        user_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_context: Dict[str, Any],
        policy_applications: List[Dict[str, Any]] = None,
        override_details: Optional[Dict[str, Any]] = None,
        industry_code: str = "saas"
    ) -> EvidencePack:
        """Create comprehensive workflow execution evidence"""
        
        # Create base evidence pack
        evidence_pack = self.builder.create_evidence_pack(
            evidence_type=EvidenceType.WORKFLOW_EXECUTION,
            workflow_id=workflow_id,
            execution_id=execution_id,
            tenant_id=tenant_id,
            created_by=user_id,
            input_data=input_data,
            output_data=output_data,
            execution_context=execution_context
        )
        
        # Add policy applications
        if policy_applications:
            for policy_app in policy_applications:
                self.builder.add_policy_application(
                    evidence_pack,
                    policy_app['policy_id'],
                    policy_app['policy_name'],
                    policy_app.get('policy_version', '1.0.0'),
                    policy_app.get('policy_hash', 'mock_hash'),
                    policy_app['enforcement_result'],
                    policy_app.get('applied_by', user_id),
                    policy_app.get('violation_details')
                )
        
        # Add override details if present
        if override_details:
            self.builder.add_override_details(
                evidence_pack,
                override_details['override_id'],
                override_details['override_type'],
                override_details['reason'],
                override_details['justification'],
                override_details['requested_by'],
                override_details['approved_by'],
                override_details.get('risk_level', 'medium'),
                override_details.get('expires_at')
            )
        
        # Add industry overlay
        industry_fields = self._get_industry_fields(industry_code, execution_context)
        self.builder.add_industry_overlay(
            evidence_pack,
            industry_code,
            **industry_fields
        )
        
        # Add trust impact score (mock calculation)
        self.builder.add_trust_impact_score(
            evidence_pack,
            baseline_score=0.8,
            current_score=0.85,
            policy_compliance=0.95,
            override_frequency=0.1,
            sla_adherence=0.9,
            error_rate=0.05
        )
        
        # Seal the evidence pack
        self.builder.seal_evidence_pack(evidence_pack)
        
        # Store evidence pack
        self.evidence_packs[evidence_pack.metadata.evidence_id] = evidence_pack
        
        self.logger.info(f"✅ Created workflow evidence: {evidence_pack.metadata.evidence_id}")
        return evidence_pack
    
    def _get_industry_fields(self, industry_code: str, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get industry-specific fields from execution context"""
        
        fields = {}
        
        if industry_code.lower() == 'saas':
            fields.update({
                'saas_revenue_impact': execution_context.get('revenue_impact', 0.0),
                'saas_customer_count': execution_context.get('customer_count', 0),
                'saas_subscription_changes': execution_context.get('subscription_changes')
            })
        elif industry_code.lower() == 'banking':
            fields.update({
                'bfsi_loan_amount': execution_context.get('loan_amount', 0.0),
                'bfsi_aml_check_result': execution_context.get('aml_result', 'passed'),
                'bfsi_kyc_verification': execution_context.get('kyc_verification'),
                'bfsi_sanction_screening': execution_context.get('sanction_screening')
            })
        elif industry_code.lower() == 'insurance':
            fields.update({
                'insurance_claim_amount': execution_context.get('claim_amount', 0.0),
                'insurance_solvency_ratio': execution_context.get('solvency_ratio', 1.5),
                'insurance_fraud_score': execution_context.get('fraud_score', 0.1),
                'insurance_regulatory_filing': execution_context.get('regulatory_filing')
            })
        elif industry_code.lower() == 'healthcare':
            fields.update({
                'healthcare_phi_accessed': execution_context.get('phi_accessed', False),
                'healthcare_patient_count': execution_context.get('patient_count', 0),
                'healthcare_consent_status': execution_context.get('consent_status', 'obtained'),
                'healthcare_breach_risk': execution_context.get('breach_risk', 'low')
            })
        
        return fields
    
    async def sign_evidence_pack(
        self,
        evidence_id: str,
        signer_id: str,
        private_key: Optional[str] = None
    ) -> bool:
        """Sign an evidence pack"""
        
        evidence_pack = self.evidence_packs.get(evidence_id)
        if not evidence_pack:
            self.logger.error(f"❌ Evidence pack {evidence_id} not found")
            return False
        
        try:
            signature = self.builder.sign_evidence_pack(evidence_pack, signer_id, private_key)
            self.logger.info(f"✅ Signed evidence pack: {evidence_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to sign evidence pack {evidence_id}: {e}")
            return False
    
    def get_evidence_pack(self, evidence_id: str) -> Optional[EvidencePack]:
        """Get evidence pack by ID"""
        return self.evidence_packs.get(evidence_id)
    
    def list_evidence_packs(
        self,
        tenant_id: Optional[int] = None,
        workflow_id: Optional[str] = None,
        evidence_type: Optional[EvidenceType] = None,
        status: Optional[EvidenceStatus] = None,
        limit: int = 100
    ) -> List[EvidencePack]:
        """List evidence packs with filtering"""
        
        packs = list(self.evidence_packs.values())
        
        if tenant_id:
            packs = [p for p in packs if p.metadata.tenant_id == tenant_id]
        
        if workflow_id:
            packs = [p for p in packs if p.metadata.workflow_id == workflow_id]
        
        if evidence_type:
            packs = [p for p in packs if p.metadata.evidence_type == evidence_type]
        
        if status:
            packs = [p for p in packs if p.metadata.status == status]
        
        # Sort by creation time (newest first)
        packs.sort(key=lambda p: p.metadata.created_at, reverse=True)
        
        return packs[:limit]
    
    async def export_evidence_bundle(
        self,
        evidence_ids: List[str],
        export_format: str = "json",
        include_attachments: bool = True
    ) -> bytes:
        """Export evidence packs as bundle"""
        
        if export_format == "zip":
            return await self._export_zip_bundle(evidence_ids, include_attachments)
        else:
            return await self._export_json_bundle(evidence_ids)
    
    async def _export_json_bundle(self, evidence_ids: List[str]) -> bytes:
        """Export evidence packs as JSON bundle"""
        
        bundle = {
            'bundle_id': str(uuid.uuid4()),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'evidence_count': len(evidence_ids),
            'evidence_packs': []
        }
        
        for evidence_id in evidence_ids:
            evidence_pack = self.evidence_packs.get(evidence_id)
            if evidence_pack:
                bundle['evidence_packs'].append(evidence_pack.to_dict())
        
        bundle_json = json.dumps(bundle, indent=2, default=str)
        return bundle_json.encode('utf-8')
    
    async def _export_zip_bundle(self, evidence_ids: List[str], include_attachments: bool) -> bytes:
        """Export evidence packs as ZIP bundle"""
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add manifest
            manifest = {
                'bundle_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'evidence_count': len(evidence_ids),
                'evidence_list': evidence_ids
            }
            
            zip_file.writestr('manifest.json', json.dumps(manifest, indent=2))
            
            # Add evidence packs
            for evidence_id in evidence_ids:
                evidence_pack = self.evidence_packs.get(evidence_id)
                if evidence_pack:
                    # Add evidence pack JSON
                    evidence_json = json.dumps(evidence_pack.to_dict(), indent=2, default=str)
                    zip_file.writestr(f'evidence/{evidence_id}.json', evidence_json)
                    
                    # Add attachments if requested
                    if include_attachments and evidence_pack.attachments:
                        for i, attachment in enumerate(evidence_pack.attachments):
                            attachment_name = f"evidence/{evidence_id}/attachment_{i}.json"
                            attachment_json = json.dumps(attachment, indent=2, default=str)
                            zip_file.writestr(attachment_name, attachment_json)
        
        zip_buffer.seek(0)
        return zip_buffer.read()
    
    def verify_evidence_integrity(self, evidence_id: str) -> Dict[str, Any]:
        """Verify evidence pack integrity"""
        
        evidence_pack = self.evidence_packs.get(evidence_id)
        if not evidence_pack:
            return {'valid': False, 'error': 'Evidence pack not found'}
        
        try:
            # Recalculate content hash
            calculated_hash = evidence_pack.calculate_content_hash()
            stored_hash = evidence_pack.content_hash
            
            hash_valid = calculated_hash == stored_hash
            
            # Check signature (mock verification)
            signature_valid = bool(evidence_pack.signature) if evidence_pack.metadata.status == EvidenceStatus.SIGNED else True
            
            return {
                'valid': hash_valid and signature_valid,
                'hash_valid': hash_valid,
                'signature_valid': signature_valid,
                'calculated_hash': calculated_hash,
                'stored_hash': stored_hash,
                'status': evidence_pack.metadata.status.value,
                'custody_chain_length': len(evidence_pack.custody_chain)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def get_evidence_statistics(self) -> Dict[str, Any]:
        """Get evidence pack statistics"""
        
        total_packs = len(self.evidence_packs)
        
        packs_by_type = {}
        packs_by_status = {}
        packs_by_tenant = {}
        
        for pack in self.evidence_packs.values():
            # By type
            evidence_type = pack.metadata.evidence_type.value
            packs_by_type[evidence_type] = packs_by_type.get(evidence_type, 0) + 1
            
            # By status
            status = pack.metadata.status.value
            packs_by_status[status] = packs_by_status.get(status, 0) + 1
            
            # By tenant
            tenant_id = pack.metadata.tenant_id
            packs_by_tenant[tenant_id] = packs_by_tenant.get(tenant_id, 0) + 1
        
        return {
            'total_evidence_packs': total_packs,
            'packs_by_type': packs_by_type,
            'packs_by_status': packs_by_status,
            'packs_by_tenant': packs_by_tenant,
            'signed_packs': len([p for p in self.evidence_packs.values() if p.metadata.status == EvidenceStatus.SIGNED])
        }

# Global instances
evidence_pack_builder = EvidencePackBuilder()
evidence_pack_manager = EvidencePackManager()
