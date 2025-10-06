"""
Task 9.2.17: Configure Retention Manager
Comprehensive data retention management with compliance-aware policies

Features:
- Multi-tier retention policies (Bronze/Silver/Gold)
- Compliance framework-specific retention rules
- Automated data lifecycle management
- GDPR "Right to Erasure" support
- Audit trail for retention actions
- Cost-optimized storage tiering
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class RetentionTier(Enum):
    """Data retention tiers"""
    BRONZE = "bronze"    # 90 days
    SILVER = "silver"    # 400 days  
    GOLD = "gold"        # 2 years (730 days)
    PLATINUM = "platinum"  # 7 years (2555 days)
    REGULATORY = "regulatory"  # 10+ years based on framework

class DataCategory(Enum):
    """Categories of data for retention policies"""
    TRANSACTIONAL = "transactional"
    AUDIT_LOGS = "audit_logs"
    EVIDENCE_PACKS = "evidence_packs"
    OVERRIDE_LEDGER = "override_ledger"
    EXECUTION_TRACES = "execution_traces"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    USER_DATA = "user_data"
    COMPLIANCE_DATA = "compliance_data"
    TELEMETRY = "telemetry"
    BACKUPS = "backups"

class ComplianceFramework(Enum):
    """Compliance frameworks with specific retention requirements"""
    SOX = "SOX"          # 7 years
    GDPR = "GDPR"        # Varies, with erasure rights
    HIPAA = "HIPAA"      # 6 years
    RBI = "RBI"          # 10 years
    DPDP = "DPDP"        # Varies, with erasure rights
    PCI_DSS = "PCI_DSS"  # 1 year minimum
    CCPA = "CCPA"        # Varies, with deletion rights

class RetentionAction(Enum):
    """Actions to take on data"""
    RETAIN = "retain"
    ARCHIVE = "archive"
    DELETE = "delete"
    ANONYMIZE = "anonymize"
    ENCRYPT = "encrypt"

@dataclass
class RetentionPolicy:
    """Data retention policy definition"""
    policy_id: str
    policy_name: str
    data_category: DataCategory
    retention_tier: RetentionTier
    retention_days: int
    compliance_frameworks: List[ComplianceFramework]
    
    # Lifecycle actions
    archive_after_days: Optional[int] = None
    delete_after_days: Optional[int] = None
    anonymize_after_days: Optional[int] = None
    
    # Special handling
    supports_erasure: bool = False  # GDPR/DPDP right to erasure
    requires_approval: bool = False
    cost_optimization: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tenant_specific: bool = False

@dataclass
class RetentionJob:
    """Retention job execution record"""
    job_id: str
    tenant_id: int
    policy_id: str
    data_category: DataCategory
    action: RetentionAction
    
    # Execution details
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "scheduled"  # scheduled, running, completed, failed
    
    # Results
    records_processed: int = 0
    records_retained: int = 0
    records_archived: int = 0
    records_deleted: int = 0
    records_anonymized: int = 0
    
    # Metadata
    error_message: Optional[str] = None
    approval_required: bool = False
    approved_by: Optional[str] = None

class RetentionManager:
    """
    Task 9.2.17: Configure Retention Manager
    Comprehensive data retention management with compliance awareness
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Service configuration - DYNAMIC, no hardcoding
        self.config = {
            "service_name": "retention_manager",
            "version": "1.0.0",
            "default_retention_days": int(os.getenv("DEFAULT_RETENTION_DAYS", "2555")),  # 7 years
            "archive_enabled": os.getenv("RETENTION_ARCHIVE_ENABLED", "true").lower() == "true",
            "cost_optimization_enabled": os.getenv("RETENTION_COST_OPTIMIZATION", "true").lower() == "true",
            "approval_required_for_deletion": os.getenv("RETENTION_APPROVAL_REQUIRED", "true").lower() == "true",
            "job_execution_interval_hours": int(os.getenv("RETENTION_JOB_INTERVAL", "24")),
            "batch_size": int(os.getenv("RETENTION_BATCH_SIZE", "1000"))
        }
        
        # Initialize default retention policies
        self.retention_policies: Dict[str, RetentionPolicy] = {}
        self.tenant_policies: Dict[int, Dict[str, RetentionPolicy]] = {}
        self.retention_jobs: Dict[str, RetentionJob] = {}
        
        # Metrics and monitoring
        self.metrics = {
            "total_policies": 0,
            "active_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "data_retained_gb": 0.0,
            "data_archived_gb": 0.0,
            "data_deleted_gb": 0.0,
            "cost_savings_usd": 0.0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info("📁 Retention Manager initialized with compliance-aware policies")
    
    def _initialize_default_policies(self):
        """Initialize default retention policies for different data categories"""
        
        # Standard retention policies
        default_policies = [
            # Audit and compliance data - long retention
            RetentionPolicy(
                policy_id="audit_logs_regulatory",
                policy_name="Audit Logs - Regulatory Compliance",
                data_category=DataCategory.AUDIT_LOGS,
                retention_tier=RetentionTier.REGULATORY,
                retention_days=3650,  # 10 years
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.RBI],
                archive_after_days=365,  # Archive after 1 year
                supports_erasure=False,
                requires_approval=True
            ),
            
            # Evidence packs - compliance retention
            RetentionPolicy(
                policy_id="evidence_packs_compliance",
                policy_name="Evidence Packs - Compliance",
                data_category=DataCategory.EVIDENCE_PACKS,
                retention_tier=RetentionTier.PLATINUM,
                retention_days=2555,  # 7 years
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GDPR],
                archive_after_days=730,  # Archive after 2 years
                supports_erasure=True,  # GDPR compliance
                requires_approval=True
            ),
            
            # Override ledger - immutable retention
            RetentionPolicy(
                policy_id="override_ledger_immutable",
                policy_name="Override Ledger - Immutable",
                data_category=DataCategory.OVERRIDE_LEDGER,
                retention_tier=RetentionTier.REGULATORY,
                retention_days=3650,  # 10 years
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.RBI],
                supports_erasure=False,  # Immutable for compliance
                requires_approval=True
            ),
            
            # Execution traces - operational data
            RetentionPolicy(
                policy_id="execution_traces_operational",
                policy_name="Execution Traces - Operational",
                data_category=DataCategory.EXECUTION_TRACES,
                retention_tier=RetentionTier.GOLD,
                retention_days=730,  # 2 years
                compliance_frameworks=[ComplianceFramework.SOX],
                archive_after_days=90,  # Archive after 3 months
                delete_after_days=730,
                cost_optimization=True
            ),
            
            # User data - privacy-aware retention
            RetentionPolicy(
                policy_id="user_data_privacy",
                policy_name="User Data - Privacy Compliant",
                data_category=DataCategory.USER_DATA,
                retention_tier=RetentionTier.SILVER,
                retention_days=400,
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.DPDP],
                anonymize_after_days=365,  # Anonymize after 1 year
                supports_erasure=True,  # Right to erasure
                requires_approval=False
            ),
            
            # Telemetry data - short retention
            RetentionPolicy(
                policy_id="telemetry_operational",
                policy_name="Telemetry - Operational",
                data_category=DataCategory.TELEMETRY,
                retention_tier=RetentionTier.BRONZE,
                retention_days=90,
                compliance_frameworks=[],
                archive_after_days=30,
                delete_after_days=90,
                cost_optimization=True
            ),
            
            # Knowledge graph - long-term value
            RetentionPolicy(
                policy_id="knowledge_graph_strategic",
                policy_name="Knowledge Graph - Strategic Asset",
                data_category=DataCategory.KNOWLEDGE_GRAPH,
                retention_tier=RetentionTier.PLATINUM,
                retention_days=2555,  # 7 years
                compliance_frameworks=[ComplianceFramework.SOX],
                archive_after_days=365,
                supports_erasure=True,  # For GDPR compliance
                cost_optimization=True
            )
        ]
        
        # Register default policies
        for policy in default_policies:
            self.retention_policies[policy.policy_id] = policy
            self.metrics["total_policies"] += 1
        
        logger.info(f"✅ Initialized {len(default_policies)} default retention policies")
    
    async def create_retention_policy(self, policy: RetentionPolicy, 
                                    tenant_id: Optional[int] = None) -> bool:
        """Create a new retention policy"""
        try:
            # Validate policy
            if not self._validate_retention_policy(policy):
                logger.error(f"❌ Invalid retention policy: {policy.policy_id}")
                return False
            
            # Store policy
            if tenant_id:
                # Tenant-specific policy
                if tenant_id not in self.tenant_policies:
                    self.tenant_policies[tenant_id] = {}
                self.tenant_policies[tenant_id][policy.policy_id] = policy
                policy.tenant_specific = True
            else:
                # Global policy
                self.retention_policies[policy.policy_id] = policy
            
            self.metrics["total_policies"] += 1
            
            logger.info(f"✅ Created retention policy: {policy.policy_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create retention policy: {e}")
            return False
    
    async def schedule_retention_job(self, tenant_id: int, policy_id: str,
                                   action: RetentionAction,
                                   scheduled_at: Optional[datetime] = None) -> str:
        """Schedule a retention job for execution"""
        try:
            # Get policy
            policy = await self._get_retention_policy(tenant_id, policy_id)
            if not policy:
                logger.error(f"❌ Retention policy not found: {policy_id}")
                return ""
            
            # Create retention job
            job_id = str(uuid.uuid4())
            job = RetentionJob(
                job_id=job_id,
                tenant_id=tenant_id,
                policy_id=policy_id,
                data_category=policy.data_category,
                action=action,
                scheduled_at=scheduled_at or datetime.utcnow(),
                approval_required=policy.requires_approval
            )
            
            # Store job
            self.retention_jobs[job_id] = job
            self.metrics["active_jobs"] += 1
            
            logger.info(f"📅 Scheduled retention job: {job_id} for {policy.data_category.value}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to schedule retention job: {e}")
            return ""
    
    async def execute_retention_job(self, job_id: str) -> bool:
        """Execute a retention job"""
        try:
            job = self.retention_jobs.get(job_id)
            if not job:
                logger.error(f"❌ Retention job not found: {job_id}")
                return False
            
            # Check if approval is required and granted
            if job.approval_required and not job.approved_by:
                logger.warning(f"⚠️ Retention job requires approval: {job_id}")
                return False
            
            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            
            # Get policy
            policy = await self._get_retention_policy(job.tenant_id, job.policy_id)
            if not policy:
                job.status = "failed"
                job.error_message = "Policy not found"
                return False
            
            # Execute retention action
            success = await self._execute_retention_action(job, policy)
            
            # Update job status
            if success:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                self.metrics["completed_jobs"] += 1
                self.metrics["active_jobs"] -= 1
            else:
                job.status = "failed"
                self.metrics["failed_jobs"] += 1
                self.metrics["active_jobs"] -= 1
            
            logger.info(f"{'✅' if success else '❌'} Retention job {job_id}: {job.status}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to execute retention job {job_id}: {e}")
            return False
    
    async def handle_erasure_request(self, tenant_id: int, user_id: str,
                                   data_categories: List[DataCategory]) -> Dict[str, Any]:
        """
        Handle GDPR/DPDP "Right to Erasure" request
        """
        try:
            erasure_request_id = str(uuid.uuid4())
            
            # Find applicable policies that support erasure
            erasure_jobs = []
            
            for category in data_categories:
                # Find policies for this category
                policies = await self._get_policies_for_category(tenant_id, category)
                
                for policy in policies:
                    if policy.supports_erasure:
                        # Schedule erasure job
                        job_id = await self.schedule_retention_job(
                            tenant_id=tenant_id,
                            policy_id=policy.policy_id,
                            action=RetentionAction.DELETE
                        )
                        
                        if job_id:
                            erasure_jobs.append({
                                "job_id": job_id,
                                "data_category": category.value,
                                "policy_id": policy.policy_id
                            })
            
            erasure_response = {
                "erasure_request_id": erasure_request_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "requested_categories": [cat.value for cat in data_categories],
                "scheduled_jobs": erasure_jobs,
                "status": "scheduled",
                "estimated_completion": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"🗑️ Erasure request processed: {erasure_request_id} ({len(erasure_jobs)} jobs)")
            return erasure_response
            
        except Exception as e:
            logger.error(f"❌ Failed to handle erasure request: {e}")
            return {}
    
    async def get_retention_status(self, tenant_id: int) -> Dict[str, Any]:
        """Get retention status for tenant"""
        try:
            # Get tenant jobs
            tenant_jobs = [
                job for job in self.retention_jobs.values()
                if job.tenant_id == tenant_id
            ]
            
            # Calculate statistics
            status = {
                "tenant_id": tenant_id,
                "active_policies": len(await self._get_tenant_policies(tenant_id)),
                "jobs": {
                    "total": len(tenant_jobs),
                    "scheduled": len([j for j in tenant_jobs if j.status == "scheduled"]),
                    "running": len([j for j in tenant_jobs if j.status == "running"]),
                    "completed": len([j for j in tenant_jobs if j.status == "completed"]),
                    "failed": len([j for j in tenant_jobs if j.status == "failed"])
                },
                "data_lifecycle": {
                    "records_retained": sum(j.records_retained for j in tenant_jobs),
                    "records_archived": sum(j.records_archived for j in tenant_jobs),
                    "records_deleted": sum(j.records_deleted for j in tenant_jobs),
                    "records_anonymized": sum(j.records_anonymized for j in tenant_jobs)
                },
                "compliance_status": await self._get_compliance_status(tenant_id),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get retention status: {e}")
            return {}
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return {
            "service_config": self.config,
            "metrics": self.metrics,
            "active_policies": len(self.retention_policies),
            "tenant_specific_policies": sum(len(policies) for policies in self.tenant_policies.values()),
            "active_jobs": len([j for j in self.retention_jobs.values() if j.status in ["scheduled", "running"]]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _validate_retention_policy(self, policy: RetentionPolicy) -> bool:
        """Validate retention policy configuration"""
        if policy.retention_days <= 0:
            return False
        
        if policy.archive_after_days and policy.archive_after_days >= policy.retention_days:
            return False
        
        if policy.delete_after_days and policy.delete_after_days > policy.retention_days:
            return False
        
        return True
    
    async def _get_retention_policy(self, tenant_id: int, policy_id: str) -> Optional[RetentionPolicy]:
        """Get retention policy by ID (tenant-specific or global)"""
        # Check tenant-specific policies first
        if tenant_id in self.tenant_policies:
            tenant_policy = self.tenant_policies[tenant_id].get(policy_id)
            if tenant_policy:
                return tenant_policy
        
        # Check global policies
        return self.retention_policies.get(policy_id)
    
    async def _get_tenant_policies(self, tenant_id: int) -> List[RetentionPolicy]:
        """Get all applicable policies for tenant"""
        policies = []
        
        # Add global policies
        policies.extend(self.retention_policies.values())
        
        # Add tenant-specific policies
        if tenant_id in self.tenant_policies:
            policies.extend(self.tenant_policies[tenant_id].values())
        
        return policies
    
    async def _get_policies_for_category(self, tenant_id: int, 
                                       category: DataCategory) -> List[RetentionPolicy]:
        """Get policies applicable to specific data category"""
        all_policies = await self._get_tenant_policies(tenant_id)
        return [policy for policy in all_policies if policy.data_category == category]
    
    async def _execute_retention_action(self, job: RetentionJob, 
                                      policy: RetentionPolicy) -> bool:
        """Execute the actual retention action"""
        try:
            # Simulate retention action execution
            # In production, this would interact with actual data stores
            
            batch_size = self.config["batch_size"]
            
            if job.action == RetentionAction.RETAIN:
                # Keep data in current location
                job.records_retained = batch_size
                
            elif job.action == RetentionAction.ARCHIVE:
                # Move to cheaper storage tier
                job.records_archived = batch_size
                self.metrics["data_archived_gb"] += 10.0  # Simulated
                
            elif job.action == RetentionAction.DELETE:
                # Permanently delete data
                job.records_deleted = batch_size
                self.metrics["data_deleted_gb"] += 5.0  # Simulated
                
            elif job.action == RetentionAction.ANONYMIZE:
                # Remove PII while keeping aggregate data
                job.records_anonymized = batch_size
                
            job.records_processed = batch_size
            
            logger.info(f"📊 Retention action executed: {job.action.value} on {batch_size} records")
            return True
            
        except Exception as e:
            job.error_message = str(e)
            logger.error(f"❌ Retention action failed: {e}")
            return False
    
    async def _get_compliance_status(self, tenant_id: int) -> Dict[str, Any]:
        """Get compliance status for tenant"""
        policies = await self._get_tenant_policies(tenant_id)
        
        framework_status = {}
        for framework in ComplianceFramework:
            applicable_policies = [
                p for p in policies 
                if framework in p.compliance_frameworks
            ]
            
            framework_status[framework.value] = {
                "applicable_policies": len(applicable_policies),
                "supports_erasure": any(p.supports_erasure for p in applicable_policies),
                "longest_retention_days": max((p.retention_days for p in applicable_policies), default=0)
            }
        
        return framework_status

# Global retention manager instance
retention_manager = RetentionManager()
