"""
Task 4.4.19: Define regulatory acceptance metric (audits passed, evidence packs delivered)
- Compliance proof
- Audit reporting service
- Industry overlays
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Regulatory Acceptance Metrics")
logger = logging.getLogger(__name__)

class AuditStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"

class EvidencePackStatus(str, Enum):
    DELIVERED = "delivered"
    PENDING = "pending"
    REJECTED = "rejected"
    APPROVED = "approved"

class RegulatoryAcceptanceMetric(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Audit metrics
    audits_passed: int = 0
    audits_failed: int = 0
    audits_in_progress: int = 0
    audit_pass_rate: float = 0.0
    
    # Evidence pack metrics
    evidence_packs_delivered: int = 0
    evidence_packs_approved: int = 0
    evidence_packs_rejected: int = 0
    evidence_pack_approval_rate: float = 0.0
    
    # Regulatory compliance score
    overall_acceptance_score: float = 0.0
    
    # Time metrics
    avg_audit_duration_days: float = 0.0
    avg_evidence_pack_delivery_days: float = 0.0
    
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    measurement_period: str = "quarterly"

class AuditRecord(BaseModel):
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    audit_type: str
    regulator_name: str
    status: AuditStatus
    start_date: datetime
    completion_date: Optional[datetime] = None
    findings: List[str] = Field(default_factory=list)
    evidence_packs_requested: int = 0
    evidence_packs_provided: int = 0

class EvidencePackRecord(BaseModel):
    pack_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    audit_id: Optional[str] = None
    pack_type: str
    status: EvidencePackStatus
    requested_date: datetime
    delivered_date: Optional[datetime] = None
    approval_date: Optional[datetime] = None
    regulator_feedback: Optional[str] = None

# In-memory storage
acceptance_metrics_store: Dict[str, RegulatoryAcceptanceMetric] = {}
audit_records_store: Dict[str, AuditRecord] = {}
evidence_pack_records_store: Dict[str, EvidencePackRecord] = {}

class RegulatoryAcceptanceCalculator:
    def __init__(self):
        pass
    
    def calculate_acceptance_metrics(self, tenant_id: str) -> RegulatoryAcceptanceMetric:
        """Calculate regulatory acceptance metrics for a tenant"""
        
        # Get audit records for tenant
        tenant_audits = [
            audit for audit in audit_records_store.values()
            if audit.tenant_id == tenant_id
        ]
        
        # Get evidence pack records for tenant
        tenant_evidence_packs = [
            pack for pack in evidence_pack_records_store.values()
            if pack.tenant_id == tenant_id
        ]
        
        # Calculate audit metrics
        audits_passed = len([a for a in tenant_audits if a.status == AuditStatus.PASSED])
        audits_failed = len([a for a in tenant_audits if a.status == AuditStatus.FAILED])
        audits_in_progress = len([a for a in tenant_audits if a.status == AuditStatus.IN_PROGRESS])
        
        total_completed_audits = audits_passed + audits_failed
        audit_pass_rate = (audits_passed / total_completed_audits * 100) if total_completed_audits > 0 else 0.0
        
        # Calculate evidence pack metrics
        evidence_packs_delivered = len([p for p in tenant_evidence_packs if p.status == EvidencePackStatus.DELIVERED])
        evidence_packs_approved = len([p for p in tenant_evidence_packs if p.status == EvidencePackStatus.APPROVED])
        evidence_packs_rejected = len([p for p in tenant_evidence_packs if p.status == EvidencePackStatus.REJECTED])
        
        total_reviewed_packs = evidence_packs_approved + evidence_packs_rejected
        evidence_pack_approval_rate = (evidence_packs_approved / total_reviewed_packs * 100) if total_reviewed_packs > 0 else 0.0
        
        # Calculate overall acceptance score
        overall_acceptance_score = (audit_pass_rate * 0.6 + evidence_pack_approval_rate * 0.4)
        
        # Calculate time metrics
        completed_audits = [a for a in tenant_audits if a.completion_date]
        if completed_audits:
            audit_durations = [(a.completion_date - a.start_date).days for a in completed_audits]
            avg_audit_duration = sum(audit_durations) / len(audit_durations)
        else:
            avg_audit_duration = 0.0
        
        delivered_packs = [p for p in tenant_evidence_packs if p.delivered_date]
        if delivered_packs:
            delivery_durations = [(p.delivered_date - p.requested_date).days for p in delivered_packs]
            avg_delivery_duration = sum(delivery_durations) / len(delivery_durations)
        else:
            avg_delivery_duration = 0.0
        
        return RegulatoryAcceptanceMetric(
            tenant_id=tenant_id,
            audits_passed=audits_passed,
            audits_failed=audits_failed,
            audits_in_progress=audits_in_progress,
            audit_pass_rate=audit_pass_rate,
            evidence_packs_delivered=evidence_packs_delivered,
            evidence_packs_approved=evidence_packs_approved,
            evidence_packs_rejected=evidence_packs_rejected,
            evidence_pack_approval_rate=evidence_pack_approval_rate,
            overall_acceptance_score=overall_acceptance_score,
            avg_audit_duration_days=avg_audit_duration,
            avg_evidence_pack_delivery_days=avg_delivery_duration
        )

# Global calculator instance
calculator = RegulatoryAcceptanceCalculator()

@app.post("/regulatory/audits/record", response_model=AuditRecord)
async def record_audit(audit: AuditRecord):
    """Record an audit event"""
    
    audit_records_store[audit.audit_id] = audit
    
    logger.info(f"✅ Recorded audit {audit.audit_id} for tenant {audit.tenant_id} - Status: {audit.status}")
    return audit

@app.post("/regulatory/evidence-packs/record", response_model=EvidencePackRecord)
async def record_evidence_pack(pack: EvidencePackRecord):
    """Record an evidence pack event"""
    
    evidence_pack_records_store[pack.pack_id] = pack
    
    logger.info(f"✅ Recorded evidence pack {pack.pack_id} for tenant {pack.tenant_id} - Status: {pack.status}")
    return pack

@app.post("/regulatory/metrics/calculate", response_model=RegulatoryAcceptanceMetric)
async def calculate_regulatory_metrics(tenant_id: str):
    """Calculate regulatory acceptance metrics for a tenant"""
    
    metric = calculator.calculate_acceptance_metrics(tenant_id)
    
    # Store metric
    acceptance_metrics_store[metric.metric_id] = metric
    
    logger.info(f"✅ Calculated regulatory acceptance metrics for tenant {tenant_id} - Score: {metric.overall_acceptance_score:.1f}")
    return metric

@app.get("/regulatory/metrics/tenant/{tenant_id}")
async def get_tenant_regulatory_metrics(tenant_id: str):
    """Get regulatory acceptance metrics for a tenant"""
    
    tenant_metrics = [
        metric for metric in acceptance_metrics_store.values()
        if metric.tenant_id == tenant_id
    ]
    
    if not tenant_metrics:
        raise HTTPException(status_code=404, detail="No regulatory metrics found for tenant")
    
    # Return most recent metric
    latest_metric = max(tenant_metrics, key=lambda x: x.measurement_date)
    
    return latest_metric

@app.get("/regulatory/audits/tenant/{tenant_id}")
async def get_tenant_audits(tenant_id: str, status: Optional[AuditStatus] = None):
    """Get audit records for a tenant"""
    
    tenant_audits = [
        audit for audit in audit_records_store.values()
        if audit.tenant_id == tenant_id
    ]
    
    if status:
        tenant_audits = [a for a in tenant_audits if a.status == status]
    
    return {
        "tenant_id": tenant_id,
        "audit_count": len(tenant_audits),
        "audits": tenant_audits
    }

@app.get("/regulatory/evidence-packs/tenant/{tenant_id}")
async def get_tenant_evidence_packs(tenant_id: str, status: Optional[EvidencePackStatus] = None):
    """Get evidence pack records for a tenant"""
    
    tenant_packs = [
        pack for pack in evidence_pack_records_store.values()
        if pack.tenant_id == tenant_id
    ]
    
    if status:
        tenant_packs = [p for p in tenant_packs if p.status == status]
    
    return {
        "tenant_id": tenant_id,
        "evidence_pack_count": len(tenant_packs),
        "evidence_packs": tenant_packs
    }

@app.get("/regulatory/summary")
async def get_regulatory_summary():
    """Get overall regulatory acceptance summary"""
    
    total_metrics = len(acceptance_metrics_store)
    
    if total_metrics == 0:
        return {
            "total_tenants": 0,
            "average_acceptance_score": 0.0,
            "total_audits": 0,
            "total_evidence_packs": 0
        }
    
    # Calculate averages
    all_metrics = list(acceptance_metrics_store.values())
    avg_acceptance_score = sum([m.overall_acceptance_score for m in all_metrics]) / len(all_metrics)
    avg_audit_pass_rate = sum([m.audit_pass_rate for m in all_metrics]) / len(all_metrics)
    avg_evidence_approval_rate = sum([m.evidence_pack_approval_rate for m in all_metrics]) / len(all_metrics)
    
    return {
        "total_tenants": len(set(m.tenant_id for m in all_metrics)),
        "average_acceptance_score": round(avg_acceptance_score, 2),
        "average_audit_pass_rate": round(avg_audit_pass_rate, 2),
        "average_evidence_approval_rate": round(avg_evidence_approval_rate, 2),
        "total_audits": len(audit_records_store),
        "total_evidence_packs": len(evidence_pack_records_store)
    }
