"""
Task 3.2.25: Retention & purge policies for logs/evidence with WORM options
- Minimal, compliant storage
- Lifecycle jobs
- Audited purges
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import hashlib
import asyncio
from dataclasses import dataclass

app = FastAPI(title="RBIA Retention & Purge Policy Service")
logger = logging.getLogger(__name__)

class RetentionPolicyType(str, Enum):
    COMPLIANCE_BASED = "compliance_based"
    BUSINESS_BASED = "business_based"
    LEGAL_HOLD = "legal_hold"
    REGULATORY = "regulatory"

class DataCategory(str, Enum):
    EVIDENCE_PACKS = "evidence_packs"
    AUDIT_LOGS = "audit_logs"
    EXECUTION_TRACES = "execution_traces"
    OVERRIDE_LEDGER = "override_ledger"
    GOVERNANCE_EVENTS = "governance_events"
    MODEL_ARTIFACTS = "model_artifacts"
    TRAINING_DATA = "training_data"

class PurgeStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Blocked by legal hold or other constraint

class WORMLevel(str, Enum):
    NONE = "none"
    SOFTWARE = "software"  # Software-enforced WORM
    HARDWARE = "hardware"  # Hardware-enforced WORM
    REGULATORY = "regulatory"  # Regulatory compliance WORM

class RetentionPolicy(BaseModel):
    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    policy_name: str = Field(..., description="Retention policy name")
    policy_type: RetentionPolicyType = Field(..., description="Type of retention policy")
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Scope
    data_categories: List[DataCategory] = Field(..., description="Data categories covered")
    compliance_frameworks: List[str] = Field(default=[], description="Applicable compliance frameworks")
    
    # Retention settings
    retention_period_days: int = Field(..., ge=1, description="Retention period in days")
    grace_period_days: int = Field(default=30, description="Grace period before purge")
    
    # WORM settings
    worm_level: WORMLevel = Field(default=WORMLevel.NONE, description="WORM enforcement level")
    immutable_period_days: int = Field(default=0, description="Immutable period in days")
    
    # Conditions
    min_age_days: int = Field(default=0, description="Minimum age before policy applies")
    max_size_gb: Optional[float] = Field(None, description="Maximum size threshold for purge")
    exclude_active_legal_holds: bool = Field(default=True, description="Exclude data under legal hold")
    
    # Metadata
    created_by: str = Field(..., description="Policy creator")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    
    # Audit
    last_execution: Optional[datetime] = Field(None, description="Last policy execution")
    next_execution: Optional[datetime] = Field(None, description="Next scheduled execution")

class PurgeJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = Field(..., description="Retention policy ID")
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Job details
    data_category: DataCategory = Field(..., description="Data category to purge")
    cutoff_date: datetime = Field(..., description="Purge data older than this date")
    estimated_records: int = Field(default=0, description="Estimated records to purge")
    estimated_size_gb: float = Field(default=0.0, description="Estimated size to purge")
    
    # Execution
    status: PurgeStatus = Field(default=PurgeStatus.SCHEDULED)
    scheduled_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Results
    records_purged: int = Field(default=0)
    size_purged_gb: float = Field(default=0.0)
    records_skipped: int = Field(default=0, description="Records skipped due to constraints")
    
    # Audit
    purge_reason: str = Field(..., description="Reason for purge")
    executed_by: str = Field(default="system", description="Who executed the purge")
    audit_trail: List[Dict[str, Any]] = Field(default=[], description="Detailed audit trail")

class WORMRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    data_category: DataCategory = Field(..., description="Data category")
    original_record_id: str = Field(..., description="Original record identifier")
    
    # WORM settings
    worm_level: WORMLevel = Field(..., description="WORM enforcement level")
    immutable_until: datetime = Field(..., description="Immutable until this date")
    retention_until: datetime = Field(..., description="Retain until this date")
    
    # Integrity
    content_hash: str = Field(..., description="Hash of protected content")
    worm_signature: str = Field(..., description="WORM integrity signature")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified_at: Optional[datetime] = Field(None)
    verification_count: int = Field(default=0)
    
    # Status
    is_tampered: bool = Field(default=False)
    tamper_detected_at: Optional[datetime] = Field(None)

class RetentionReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    report_type: str = Field(..., description="Type of retention report")
    
    # Report data
    total_records: int = Field(default=0)
    total_size_gb: float = Field(default=0.0)
    records_by_category: Dict[str, int] = Field(default={})
    oldest_record_date: Optional[datetime] = Field(None)
    newest_record_date: Optional[datetime] = Field(None)
    
    # Compliance
    compliance_frameworks: List[str] = Field(default=[])
    retention_violations: List[Dict[str, Any]] = Field(default=[])
    
    # Generated
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = Field(..., description="Report generator")

# In-memory storage (replace with actual database)
retention_policies: Dict[str, RetentionPolicy] = {}
purge_jobs: Dict[str, PurgeJob] = {}
worm_records: Dict[str, WORMRecord] = {}
retention_reports: Dict[str, RetentionReport] = {}

# Default retention policies by compliance framework
DEFAULT_RETENTION_POLICIES = {
    "SOX": {
        "evidence_packs": 2555,  # 7 years
        "audit_logs": 2555,
        "override_ledger": 2555
    },
    "GDPR": {
        "evidence_packs": 1095,  # 3 years
        "audit_logs": 1095,
        "execution_traces": 365
    },
    "HIPAA": {
        "evidence_packs": 2190,  # 6 years
        "audit_logs": 2190,
        "governance_events": 2190
    }
}

@app.post("/retention-policies", response_model=RetentionPolicy)
async def create_retention_policy(policy: RetentionPolicy):
    """
    Create new retention policy
    """
    # Validate WORM settings
    if policy.worm_level != WORMLevel.NONE and policy.immutable_period_days == 0:
        raise HTTPException(
            status_code=400,
            detail="WORM policies must specify immutable period"
        )
    
    # Set next execution
    policy.next_execution = datetime.utcnow() + timedelta(days=1)  # Daily execution
    
    retention_policies[policy.policy_id] = policy
    
    logger.info(f"Retention policy created: {policy.policy_id} - {policy.policy_name}")
    
    return policy

@app.post("/retention-policies/{policy_id}/execute")
async def execute_retention_policy(policy_id: str, background_tasks: BackgroundTasks):
    """
    Execute retention policy (create purge jobs)
    """
    if policy_id not in retention_policies:
        raise HTTPException(status_code=404, detail="Retention policy not found")
    
    policy = retention_policies[policy_id]
    
    if not policy.is_active:
        raise HTTPException(status_code=400, detail="Policy is not active")
    
    # Create purge jobs for each data category
    purge_jobs_created = []
    
    for data_category in policy.data_categories:
        cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)
        
        # Estimate records to purge
        estimation = await estimate_purge_impact(policy.tenant_id, data_category, cutoff_date)
        
        purge_job = PurgeJob(
            policy_id=policy_id,
            tenant_id=policy.tenant_id,
            data_category=data_category,
            cutoff_date=cutoff_date,
            estimated_records=estimation["record_count"],
            estimated_size_gb=estimation["size_gb"],
            purge_reason=f"Retention policy execution: {policy.policy_name}"
        )
        
        purge_jobs[purge_job.job_id] = purge_job
        purge_jobs_created.append(purge_job.job_id)
        
        # Execute purge job in background
        background_tasks.add_task(execute_purge_job, purge_job.job_id)
    
    # Update policy execution time
    policy.last_execution = datetime.utcnow()
    policy.next_execution = datetime.utcnow() + timedelta(days=1)
    
    logger.info(f"Retention policy executed: {policy_id} - Created {len(purge_jobs_created)} purge jobs")
    
    return {
        "policy_id": policy_id,
        "execution_time": policy.last_execution.isoformat(),
        "purge_jobs_created": purge_jobs_created,
        "next_execution": policy.next_execution.isoformat()
    }

async def estimate_purge_impact(tenant_id: str, data_category: DataCategory, cutoff_date: datetime) -> Dict[str, Any]:
    """Estimate impact of purge operation"""
    # Simulate estimation based on data category
    base_count = {
        DataCategory.EVIDENCE_PACKS: 1000,
        DataCategory.AUDIT_LOGS: 50000,
        DataCategory.EXECUTION_TRACES: 25000,
        DataCategory.OVERRIDE_LEDGER: 500,
        DataCategory.GOVERNANCE_EVENTS: 10000,
        DataCategory.MODEL_ARTIFACTS: 100,
        DataCategory.TRAINING_DATA: 50
    }
    
    # Simulate age-based reduction
    days_old = (datetime.utcnow() - cutoff_date).days
    age_factor = min(days_old / 365, 1.0)  # Max 100% of base count
    
    estimated_count = int(base_count.get(data_category, 1000) * age_factor)
    estimated_size = estimated_count * 0.001  # 1MB per record average
    
    return {
        "record_count": estimated_count,
        "size_gb": estimated_size
    }

async def execute_purge_job(job_id: str):
    """Execute purge job in background"""
    if job_id not in purge_jobs:
        return
    
    job = purge_jobs[job_id]
    
    try:
        job.status = PurgeStatus.IN_PROGRESS
        job.started_at = datetime.utcnow()
        
        # Check for blocking conditions
        blocking_conditions = await check_purge_blockers(job)
        
        if blocking_conditions:
            job.status = PurgeStatus.BLOCKED
            job.audit_trail.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "purge_blocked",
                "details": blocking_conditions
            })
            logger.warning(f"Purge job blocked: {job_id} - {blocking_conditions}")
            return
        
        # Execute purge based on data category
        purge_result = await execute_category_purge(job)
        
        # Update job with results
        job.records_purged = purge_result["records_purged"]
        job.size_purged_gb = purge_result["size_purged_gb"]
        job.records_skipped = purge_result["records_skipped"]
        job.audit_trail.extend(purge_result["audit_events"])
        
        job.status = PurgeStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        
        logger.info(f"Purge job completed: {job_id} - Purged {job.records_purged} records")
        
    except Exception as e:
        job.status = PurgeStatus.FAILED
        job.audit_trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "purge_failed",
            "error": str(e)
        })
        logger.error(f"Purge job failed: {job_id} - {e}")

async def check_purge_blockers(job: PurgeJob) -> List[str]:
    """Check for conditions that would block purge"""
    blockers = []
    
    # Check for active legal holds
    if job.data_category in [DataCategory.EVIDENCE_PACKS, DataCategory.AUDIT_LOGS]:
        # Simulate legal hold check
        has_legal_hold = False  # Would check actual legal holds
        if has_legal_hold:
            blockers.append("Active legal hold prevents purge")
    
    # Check WORM protection
    worm_protected = [w for w in worm_records.values() 
                     if w.tenant_id == job.tenant_id and 
                     w.data_category == job.data_category and
                     w.immutable_until > datetime.utcnow()]
    
    if worm_protected:
        blockers.append(f"{len(worm_protected)} records protected by WORM")
    
    # Check compliance requirements
    policy = retention_policies.get(job.policy_id)
    if policy and policy.exclude_active_legal_holds:
        # Additional compliance checks would go here
        pass
    
    return blockers

async def execute_category_purge(job: PurgeJob) -> Dict[str, Any]:
    """Execute purge for specific data category"""
    audit_events = []
    
    # Simulate purge execution based on category
    if job.data_category == DataCategory.EVIDENCE_PACKS:
        result = await purge_evidence_packs(job)
    elif job.data_category == DataCategory.AUDIT_LOGS:
        result = await purge_audit_logs(job)
    elif job.data_category == DataCategory.EXECUTION_TRACES:
        result = await purge_execution_traces(job)
    else:
        # Generic purge
        result = await generic_purge(job)
    
    audit_events.append({
        "timestamp": datetime.utcnow().isoformat(),
        "event": f"purged_{job.data_category.value}",
        "records_purged": result["records_purged"],
        "size_purged_gb": result["size_purged_gb"]
    })
    
    result["audit_events"] = audit_events
    return result

async def purge_evidence_packs(job: PurgeJob) -> Dict[str, Any]:
    """Purge evidence packs"""
    # Simulate evidence pack purge
    records_to_purge = min(job.estimated_records, 800)  # Some records might be protected
    records_skipped = job.estimated_records - records_to_purge
    
    size_purged = records_to_purge * 0.001  # 1MB per record
    
    # In production, would execute actual database purge
    logger.info(f"Purged {records_to_purge} evidence packs for tenant {job.tenant_id}")
    
    return {
        "records_purged": records_to_purge,
        "size_purged_gb": size_purged,
        "records_skipped": records_skipped
    }

async def purge_audit_logs(job: PurgeJob) -> Dict[str, Any]:
    """Purge audit logs"""
    # Simulate audit log purge
    records_to_purge = min(job.estimated_records, int(job.estimated_records * 0.95))
    records_skipped = job.estimated_records - records_to_purge
    
    size_purged = records_to_purge * 0.0001  # 100KB per log record
    
    logger.info(f"Purged {records_to_purge} audit log records for tenant {job.tenant_id}")
    
    return {
        "records_purged": records_to_purge,
        "size_purged_gb": size_purged,
        "records_skipped": records_skipped
    }

async def purge_execution_traces(job: PurgeJob) -> Dict[str, Any]:
    """Purge execution traces"""
    # Simulate execution trace purge
    records_to_purge = job.estimated_records
    records_skipped = 0
    
    size_purged = records_to_purge * 0.0005  # 500KB per trace
    
    logger.info(f"Purged {records_to_purge} execution traces for tenant {job.tenant_id}")
    
    return {
        "records_purged": records_to_purge,
        "size_purged_gb": size_purged,
        "records_skipped": records_skipped
    }

async def generic_purge(job: PurgeJob) -> Dict[str, Any]:
    """Generic purge for other data categories"""
    records_to_purge = job.estimated_records
    records_skipped = 0
    size_purged = job.estimated_size_gb
    
    logger.info(f"Purged {records_to_purge} {job.data_category.value} records for tenant {job.tenant_id}")
    
    return {
        "records_purged": records_to_purge,
        "size_purged_gb": size_purged,
        "records_skipped": records_skipped
    }

@app.post("/worm/protect", response_model=WORMRecord)
async def create_worm_protection(
    tenant_id: str,
    data_category: DataCategory,
    original_record_id: str,
    worm_level: WORMLevel,
    immutable_days: int,
    retention_days: int,
    content_hash: str
):
    """
    Create WORM protection for a record
    """
    if worm_level == WORMLevel.NONE:
        raise HTTPException(status_code=400, detail="WORM level cannot be NONE for protection")
    
    immutable_until = datetime.utcnow() + timedelta(days=immutable_days)
    retention_until = datetime.utcnow() + timedelta(days=retention_days)
    
    # Generate WORM signature
    worm_data = f"{tenant_id}:{original_record_id}:{content_hash}:{immutable_until.isoformat()}"
    worm_signature = hashlib.sha256(worm_data.encode()).hexdigest()
    
    worm_record = WORMRecord(
        tenant_id=tenant_id,
        data_category=data_category,
        original_record_id=original_record_id,
        worm_level=worm_level,
        immutable_until=immutable_until,
        retention_until=retention_until,
        content_hash=content_hash,
        worm_signature=worm_signature
    )
    
    worm_records[worm_record.record_id] = worm_record
    
    logger.info(f"WORM protection created: {worm_record.record_id} for {original_record_id}")
    
    return worm_record

@app.post("/worm/verify/{record_id}")
async def verify_worm_protection(record_id: str):
    """
    Verify WORM protection integrity
    """
    if record_id not in worm_records:
        raise HTTPException(status_code=404, detail="WORM record not found")
    
    worm_record = worm_records[record_id]
    
    # Verify signature
    worm_data = f"{worm_record.tenant_id}:{worm_record.original_record_id}:{worm_record.content_hash}:{worm_record.immutable_until.isoformat()}"
    expected_signature = hashlib.sha256(worm_data.encode()).hexdigest()
    
    is_valid = worm_record.worm_signature == expected_signature
    
    # Update verification tracking
    worm_record.verification_count += 1
    worm_record.last_verified_at = datetime.utcnow()
    
    if not is_valid and not worm_record.is_tampered:
        worm_record.is_tampered = True
        worm_record.tamper_detected_at = datetime.utcnow()
        logger.critical(f"WORM tampering detected: {record_id}")
    
    return {
        "record_id": record_id,
        "is_valid": is_valid,
        "is_immutable": worm_record.immutable_until > datetime.utcnow(),
        "verification_count": worm_record.verification_count,
        "last_verified_at": worm_record.last_verified_at.isoformat(),
        "tamper_detected": worm_record.is_tampered
    }

@app.get("/retention-policies")
async def list_retention_policies(
    tenant_id: Optional[str] = None,
    policy_type: Optional[RetentionPolicyType] = None,
    is_active: Optional[bool] = None
):
    """List retention policies with filtering"""
    filtered_policies = list(retention_policies.values())
    
    if tenant_id:
        filtered_policies = [p for p in filtered_policies if p.tenant_id == tenant_id]
    
    if policy_type:
        filtered_policies = [p for p in filtered_policies if p.policy_type == policy_type]
    
    if is_active is not None:
        filtered_policies = [p for p in filtered_policies if p.is_active == is_active]
    
    return {
        "retention_policies": filtered_policies,
        "total_count": len(filtered_policies)
    }

@app.get("/purge-jobs")
async def list_purge_jobs(
    tenant_id: Optional[str] = None,
    status: Optional[PurgeStatus] = None,
    data_category: Optional[DataCategory] = None,
    limit: int = 100
):
    """List purge jobs with filtering"""
    filtered_jobs = list(purge_jobs.values())
    
    if tenant_id:
        filtered_jobs = [j for j in filtered_jobs if j.tenant_id == tenant_id]
    
    if status:
        filtered_jobs = [j for j in filtered_jobs if j.status == status]
    
    if data_category:
        filtered_jobs = [j for j in filtered_jobs if j.data_category == data_category]
    
    # Sort by scheduled time, most recent first
    filtered_jobs.sort(key=lambda x: x.scheduled_at, reverse=True)
    
    return {
        "purge_jobs": filtered_jobs[:limit],
        "total_count": len(filtered_jobs)
    }

@app.get("/retention/report/{tenant_id}")
async def generate_retention_report(tenant_id: str, background_tasks: BackgroundTasks):
    """Generate retention compliance report"""
    
    report = RetentionReport(
        tenant_id=tenant_id,
        report_type="retention_compliance",
        generated_by="system"
    )
    
    # Generate report in background
    background_tasks.add_task(build_retention_report, report.report_id, tenant_id)
    
    retention_reports[report.report_id] = report
    
    return report

async def build_retention_report(report_id: str, tenant_id: str):
    """Build retention report in background"""
    if report_id not in retention_reports:
        return
    
    report = retention_reports[report_id]
    
    # Simulate report building
    report.total_records = 75000
    report.total_size_gb = 150.5
    report.records_by_category = {
        "evidence_packs": 1000,
        "audit_logs": 50000,
        "execution_traces": 20000,
        "override_ledger": 500,
        "governance_events": 3500
    }
    report.oldest_record_date = datetime.utcnow() - timedelta(days=2555)  # 7 years
    report.newest_record_date = datetime.utcnow()
    report.compliance_frameworks = ["SOX", "GDPR", "HIPAA"]
    
    # Check for retention violations
    violations = []
    for category, count in report.records_by_category.items():
        if category == "evidence_packs" and count > 10000:  # Example violation
            violations.append({
                "category": category,
                "violation_type": "retention_exceeded",
                "description": f"Evidence packs exceed recommended retention: {count} records"
            })
    
    report.retention_violations = violations
    
    logger.info(f"Retention report completed: {report_id} for tenant {tenant_id}")

@app.post("/lifecycle/schedule")
async def schedule_lifecycle_jobs():
    """
    Schedule automatic lifecycle jobs for all active policies
    """
    scheduled_jobs = []
    
    for policy in retention_policies.values():
        if not policy.is_active:
            continue
        
        # Check if policy is due for execution
        if policy.next_execution and policy.next_execution <= datetime.utcnow():
            try:
                # Execute policy (this would normally be done by a scheduler)
                execution_result = await execute_retention_policy(policy.policy_id, BackgroundTasks())
                scheduled_jobs.append({
                    "policy_id": policy.policy_id,
                    "status": "scheduled",
                    "purge_jobs": execution_result.get("purge_jobs_created", [])
                })
            except Exception as e:
                scheduled_jobs.append({
                    "policy_id": policy.policy_id,
                    "status": "failed",
                    "error": str(e)
                })
    
    return {
        "scheduled_jobs": scheduled_jobs,
        "total_scheduled": len(scheduled_jobs)
    }

@app.get("/health")
async def health_check():
    active_policies = len([p for p in retention_policies.values() if p.is_active])
    completed_purges = len([j for j in purge_jobs.values() if j.status == PurgeStatus.COMPLETED])
    worm_protected = len(worm_records)
    
    return {
        "status": "healthy",
        "service": "RBIA Retention & Purge Policy Service",
        "task": "3.2.25",
        "active_policies": active_policies,
        "completed_purge_jobs": completed_purges,
        "worm_protected_records": worm_protected,
        "supported_data_categories": [cat.value for cat in DataCategory],
        "supported_worm_levels": [level.value for level in WORMLevel]
    }
