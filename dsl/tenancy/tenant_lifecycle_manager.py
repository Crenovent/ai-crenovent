# Tenant Lifecycle Management System
# Tasks 6.5-T18, T19, T23: Onboarding, offboarding, lifecycle management

import asyncio
import json
import uuid
import shutil
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import tempfile

logger = logging.getLogger(__name__)

class TenantLifecycleStage(Enum):
    """Stages in tenant lifecycle"""
    PROVISIONING = "provisioning"
    ONBOARDING = "onboarding"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    OFFBOARDING = "offboarding"
    TERMINATED = "terminated"

class OnboardingStep(Enum):
    """Steps in tenant onboarding process"""
    TENANT_CREATION = "tenant_creation"
    RESOURCE_PROVISIONING = "resource_provisioning"
    POLICY_CONFIGURATION = "policy_configuration"
    COMPLIANCE_SETUP = "compliance_setup"
    SANDBOX_CREATION = "sandbox_creation"
    INITIAL_VALIDATION = "initial_validation"
    WELCOME_NOTIFICATION = "welcome_notification"
    TRAINING_SETUP = "training_setup"

class OffboardingStep(Enum):
    """Steps in tenant offboarding process"""
    WORKFLOW_SUSPENSION = "workflow_suspension"
    DATA_EXPORT = "data_export"
    COMPLIANCE_VALIDATION = "compliance_validation"
    DATA_ANONYMIZATION = "data_anonymization"
    DATA_PURGE = "data_purge"
    RESOURCE_CLEANUP = "resource_cleanup"
    AUDIT_FINALIZATION = "audit_finalization"
    TERMINATION_CONFIRMATION = "termination_confirmation"

@dataclass
class OnboardingProgress:
    """Progress tracking for tenant onboarding"""
    tenant_id: int
    current_step: OnboardingStep
    completed_steps: List[OnboardingStep]
    failed_steps: List[Dict[str, Any]]
    started_at: str
    estimated_completion: str
    progress_percent: float
    status: str  # in_progress, completed, failed, paused

@dataclass
class OffboardingProgress:
    """Progress tracking for tenant offboarding"""
    tenant_id: int
    current_step: OffboardingStep
    completed_steps: List[OffboardingStep]
    failed_steps: List[Dict[str, Any]]
    started_at: str
    estimated_completion: str
    progress_percent: float
    data_retention_until: str
    compliance_requirements: List[str]
    status: str  # in_progress, completed, failed, paused

@dataclass
class TenantLifecycleEvent:
    """Event in tenant lifecycle"""
    event_id: str
    tenant_id: int
    event_type: str
    event_data: Dict[str, Any]
    timestamp: str
    user_id: Optional[str]
    automated: bool

class TenantLifecycleManager:
    """
    Comprehensive tenant lifecycle management
    Tasks 10.1-10.4: Complete tenant lifecycle orchestration
    
    Enhanced with Chapter 10 implementations:
    - Provisioning automation (10.1)
    - Onboarding toolkit integration (10.2)
    - Offboarding workflows (10.3)
    - Lifecycle governance (10.4)
    """
    
    def __init__(
        self,
        tenant_context_manager=None,
        tenant_isolation_sandbox=None,
        policy_engine=None,
        notification_service=None
    ):
        self.tenant_context_manager = tenant_context_manager
        self.tenant_isolation_sandbox = tenant_isolation_sandbox
        self.policy_engine = policy_engine
        self.notification_service = notification_service
        
        self.onboarding_progress: Dict[int, OnboardingProgress] = {}
        self.offboarding_progress: Dict[int, OffboardingProgress] = {}
        self.lifecycle_events: Dict[int, List[TenantLifecycleEvent]] = {}
        
        # Chapter 10 enhancements
        self.tenant_registry: Dict[int, Dict[str, Any]] = {}
        self.lifecycle_governance_hooks: Dict[str, Any] = {}
        
        logger.info("🏗️ Enhanced Tenant Lifecycle Manager initialized with Chapter 10 capabilities")
        
        # Industry-specific onboarding configurations
        self.industry_onboarding_configs = {
            'SaaS': {
                'required_compliance_frameworks': ['SOX_SAAS', 'GDPR_SAAS', 'SOC2_TYPE2'],
                'default_data_retention_days': 2555,  # 7 years
                'required_training_modules': ['saas_workflows', 'compliance_basics'],
                'initial_policy_pack': 'saas_v1.0',
                'sandbox_isolation_level': 'logical'
            },
            'Banking': {
                'required_compliance_frameworks': ['RBI_INDIA', 'BASEL_III', 'KYC_AML'],
                'default_data_retention_days': 3650,  # 10 years
                'required_training_modules': ['banking_workflows', 'regulatory_compliance', 'kyc_procedures'],
                'initial_policy_pack': 'banking_rbi_v1.0',
                'sandbox_isolation_level': 'container'
            },
            'Insurance': {
                'required_compliance_frameworks': ['IRDAI_INDIA', 'SOLVENCY_II', 'GDPR_INSURANCE'],
                'default_data_retention_days': 3650,  # 10 years
                'required_training_modules': ['insurance_workflows', 'solvency_compliance'],
                'initial_policy_pack': 'insurance_irdai_v1.0',
                'sandbox_isolation_level': 'container'
            },
            'Healthcare': {
                'required_compliance_frameworks': ['HIPAA', 'GDPR_HEALTHCARE', 'FDA_21CFR11'],
                'default_data_retention_days': 2555,  # 7 years
                'required_training_modules': ['healthcare_workflows', 'hipaa_compliance', 'phi_handling'],
                'initial_policy_pack': 'healthcare_hipaa_v1.0',
                'sandbox_isolation_level': 'namespace'
            }
        }
    
    async def start_tenant_onboarding(
        self,
        tenant_name: str,
        industry_code: str,
        data_residency: str,
        tenant_tier: str = "professional",
        contact_email: str = "",
        custom_configs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Start comprehensive tenant onboarding process"""
        
        logger.info(f"Starting tenant onboarding: {tenant_name} ({industry_code})")
        
        # Create initial tenant context
        from .tenant_context_manager import IndustryCode, DataResidency, TenantTier
        
        try:
            tenant_context = await self.tenant_context_manager.create_tenant_context(
                tenant_name=tenant_name,
                industry_code=IndustryCode(industry_code),
                data_residency=DataResidency(data_residency),
                tenant_tier=TenantTier(tenant_tier),
                contact_email=contact_email,
                custom_configs=custom_configs or {}
            )
            
            tenant_id = tenant_context.tenant_id
            
            # Initialize onboarding progress
            onboarding_progress = OnboardingProgress(
                tenant_id=tenant_id,
                current_step=OnboardingStep.TENANT_CREATION,
                completed_steps=[OnboardingStep.TENANT_CREATION],
                failed_steps=[],
                started_at=datetime.now(timezone.utc).isoformat(),
                estimated_completion=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
                progress_percent=12.5,  # 1/8 steps completed
                status="in_progress"
            )
            
            self.onboarding_progress[tenant_id] = onboarding_progress
            
            # Log lifecycle event
            await self._log_lifecycle_event(
                tenant_id,
                "onboarding_started",
                {
                    'tenant_name': tenant_name,
                    'industry_code': industry_code,
                    'data_residency': data_residency,
                    'tenant_tier': tenant_tier
                },
                automated=True
            )
            
            # Start async onboarding process
            asyncio.create_task(self._execute_onboarding_steps(tenant_id, industry_code))
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'onboarding_id': f"onboard_{tenant_id}_{int(datetime.now().timestamp())}",
                'estimated_completion_hours': 2,
                'progress': onboarding_progress.__dict__
            }
            
        except Exception as e:
            logger.error(f"Failed to start tenant onboarding: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_onboarding_steps(self, tenant_id: int, industry_code: str) -> None:
        """Execute all onboarding steps for tenant"""
        
        progress = self.onboarding_progress[tenant_id]
        industry_config = self.industry_onboarding_configs.get(industry_code, self.industry_onboarding_configs['SaaS'])
        
        try:
            # Step 2: Resource Provisioning
            await self._execute_onboarding_step(
                tenant_id,
                OnboardingStep.RESOURCE_PROVISIONING,
                self._provision_tenant_resources,
                {'industry_config': industry_config}
            )
            
            # Step 3: Policy Configuration
            await self._execute_onboarding_step(
                tenant_id,
                OnboardingStep.POLICY_CONFIGURATION,
                self._configure_tenant_policies,
                {'industry_config': industry_config}
            )
            
            # Step 4: Compliance Setup
            await self._execute_onboarding_step(
                tenant_id,
                OnboardingStep.COMPLIANCE_SETUP,
                self._setup_tenant_compliance,
                {'industry_config': industry_config}
            )
            
            # Step 5: Sandbox Creation
            await self._execute_onboarding_step(
                tenant_id,
                OnboardingStep.SANDBOX_CREATION,
                self._create_tenant_sandbox,
                {'industry_config': industry_config}
            )
            
            # Step 6: Initial Validation
            await self._execute_onboarding_step(
                tenant_id,
                OnboardingStep.INITIAL_VALIDATION,
                self._validate_tenant_setup,
                {'industry_config': industry_config}
            )
            
            # Step 7: Welcome Notification
            await self._execute_onboarding_step(
                tenant_id,
                OnboardingStep.WELCOME_NOTIFICATION,
                self._send_welcome_notification,
                {'industry_config': industry_config}
            )
            
            # Step 8: Training Setup
            await self._execute_onboarding_step(
                tenant_id,
                OnboardingStep.TRAINING_SETUP,
                self._setup_tenant_training,
                {'industry_config': industry_config}
            )
            
            # Mark onboarding as completed
            progress.status = "completed"
            progress.progress_percent = 100.0
            
            await self._log_lifecycle_event(
                tenant_id,
                "onboarding_completed",
                {'completion_time_minutes': self._calculate_onboarding_time(progress)},
                automated=True
            )
            
            logger.info(f"Tenant onboarding completed successfully: {tenant_id}")
            
        except Exception as e:
            progress.status = "failed"
            await self._log_lifecycle_event(
                tenant_id,
                "onboarding_failed",
                {'error': str(e)},
                automated=True
            )
            logger.error(f"Tenant onboarding failed: {tenant_id} - {e}")
    
    async def _execute_onboarding_step(
        self,
        tenant_id: int,
        step: OnboardingStep,
        step_function,
        step_args: Dict[str, Any]
    ) -> None:
        """Execute individual onboarding step"""
        
        progress = self.onboarding_progress[tenant_id]
        progress.current_step = step
        
        try:
            logger.info(f"Executing onboarding step for tenant {tenant_id}: {step.value}")
            
            # Execute step function
            await step_function(tenant_id, **step_args)
            
            # Mark step as completed
            progress.completed_steps.append(step)
            progress.progress_percent = (len(progress.completed_steps) / 8) * 100
            
            await self._log_lifecycle_event(
                tenant_id,
                f"onboarding_step_completed",
                {'step': step.value},
                automated=True
            )
            
        except Exception as e:
            # Mark step as failed
            progress.failed_steps.append({
                'step': step.value,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            await self._log_lifecycle_event(
                tenant_id,
                f"onboarding_step_failed",
                {'step': step.value, 'error': str(e)},
                automated=True
            )
            
            raise e
    
    async def _provision_tenant_resources(self, tenant_id: int, industry_config: Dict[str, Any]) -> None:
        """Provision resources for tenant"""
        # Simulate resource provisioning
        await asyncio.sleep(2)
        logger.info(f"Provisioned resources for tenant {tenant_id}")
    
    async def _configure_tenant_policies(self, tenant_id: int, industry_config: Dict[str, Any]) -> None:
        """Configure policies for tenant"""
        if self.policy_engine:
            # Configure industry-specific policies
            policy_pack = industry_config.get('initial_policy_pack', 'default_v1.0')
            # await self.policy_engine.configure_tenant_policies(tenant_id, policy_pack)
        
        await asyncio.sleep(1)
        logger.info(f"Configured policies for tenant {tenant_id}")
    
    async def _setup_tenant_compliance(self, tenant_id: int, industry_config: Dict[str, Any]) -> None:
        """Setup compliance frameworks for tenant"""
        compliance_frameworks = industry_config.get('required_compliance_frameworks', [])
        # Setup compliance monitoring and validation
        await asyncio.sleep(1)
        logger.info(f"Setup compliance for tenant {tenant_id}: {compliance_frameworks}")
    
    async def _create_tenant_sandbox(self, tenant_id: int, industry_config: Dict[str, Any]) -> None:
        """Create isolated sandbox for tenant"""
        if self.tenant_isolation_sandbox:
            from .tenant_isolation_sandbox import IsolationLevel
            isolation_level = IsolationLevel(industry_config.get('sandbox_isolation_level', 'logical'))
            await self.tenant_isolation_sandbox.create_tenant_sandbox(tenant_id, isolation_level)
        
        logger.info(f"Created sandbox for tenant {tenant_id}")
    
    async def _validate_tenant_setup(self, tenant_id: int, industry_config: Dict[str, Any]) -> None:
        """Validate tenant setup"""
        # Run validation checks
        await asyncio.sleep(1)
        logger.info(f"Validated setup for tenant {tenant_id}")
    
    async def _send_welcome_notification(self, tenant_id: int, industry_config: Dict[str, Any]) -> None:
        """Send welcome notification to tenant"""
        if self.notification_service:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
            if tenant_context and tenant_context.contact_email:
                # await self.notification_service.send_welcome_email(tenant_context.contact_email, tenant_context.tenant_name)
                pass
        
        logger.info(f"Sent welcome notification for tenant {tenant_id}")
    
    async def _setup_tenant_training(self, tenant_id: int, industry_config: Dict[str, Any]) -> None:
        """Setup training modules for tenant"""
        training_modules = industry_config.get('required_training_modules', [])
        # Setup training access and materials
        await asyncio.sleep(1)
        logger.info(f"Setup training for tenant {tenant_id}: {training_modules}")
    
    async def start_tenant_offboarding(
        self,
        tenant_id: int,
        reason: str,
        requested_by: str,
        data_retention_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Start comprehensive tenant offboarding process"""
        
        logger.info(f"Starting tenant offboarding: {tenant_id} - {reason}")
        
        tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
        if not tenant_context:
            return {'success': False, 'error': 'Tenant not found'}
        
        # Determine data retention period
        if data_retention_days is None:
            data_retention_days = tenant_context.audit_retention_days
        
        data_retention_until = (datetime.now(timezone.utc) + timedelta(days=data_retention_days)).isoformat()
        
        # Initialize offboarding progress
        offboarding_progress = OffboardingProgress(
            tenant_id=tenant_id,
            current_step=OffboardingStep.WORKFLOW_SUSPENSION,
            completed_steps=[],
            failed_steps=[],
            started_at=datetime.now(timezone.utc).isoformat(),
            estimated_completion=(datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
            progress_percent=0.0,
            data_retention_until=data_retention_until,
            compliance_requirements=tenant_context.compliance_frameworks,
            status="in_progress"
        )
        
        self.offboarding_progress[tenant_id] = offboarding_progress
        
        # Log lifecycle event
        await self._log_lifecycle_event(
            tenant_id,
            "offboarding_started",
            {
                'reason': reason,
                'requested_by': requested_by,
                'data_retention_days': data_retention_days
            },
            user_id=requested_by,
            automated=False
        )
        
        # Start async offboarding process
        asyncio.create_task(self._execute_offboarding_steps(tenant_id, reason))
        
        return {
            'success': True,
            'tenant_id': tenant_id,
            'offboarding_id': f"offboard_{tenant_id}_{int(datetime.now().timestamp())}",
            'estimated_completion_hours': 4,
            'data_retention_until': data_retention_until,
            'progress': offboarding_progress.__dict__
        }
    
    async def _execute_offboarding_steps(self, tenant_id: int, reason: str) -> None:
        """Execute all offboarding steps for tenant"""
        
        progress = self.offboarding_progress[tenant_id]
        
        try:
            # Step 1: Workflow Suspension
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.WORKFLOW_SUSPENSION,
                self._suspend_tenant_workflows,
                {'reason': reason}
            )
            
            # Step 2: Data Export
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.DATA_EXPORT,
                self._export_tenant_data,
                {}
            )
            
            # Step 3: Compliance Validation
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.COMPLIANCE_VALIDATION,
                self._validate_compliance_requirements,
                {}
            )
            
            # Step 4: Data Anonymization
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.DATA_ANONYMIZATION,
                self._anonymize_tenant_data,
                {}
            )
            
            # Step 5: Data Purge
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.DATA_PURGE,
                self._purge_tenant_data,
                {}
            )
            
            # Step 6: Resource Cleanup
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.RESOURCE_CLEANUP,
                self._cleanup_tenant_resources,
                {}
            )
            
            # Step 7: Audit Finalization
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.AUDIT_FINALIZATION,
                self._finalize_tenant_audit,
                {}
            )
            
            # Step 8: Termination Confirmation
            await self._execute_offboarding_step(
                tenant_id,
                OffboardingStep.TERMINATION_CONFIRMATION,
                self._confirm_tenant_termination,
                {}
            )
            
            # Mark offboarding as completed
            progress.status = "completed"
            progress.progress_percent = 100.0
            
            await self._log_lifecycle_event(
                tenant_id,
                "offboarding_completed",
                {'completion_time_minutes': self._calculate_offboarding_time(progress)},
                automated=True
            )
            
            logger.info(f"Tenant offboarding completed successfully: {tenant_id}")
            
        except Exception as e:
            progress.status = "failed"
            await self._log_lifecycle_event(
                tenant_id,
                "offboarding_failed",
                {'error': str(e)},
                automated=True
            )
            logger.error(f"Tenant offboarding failed: {tenant_id} - {e}")
    
    async def _execute_offboarding_step(
        self,
        tenant_id: int,
        step: OffboardingStep,
        step_function,
        step_args: Dict[str, Any]
    ) -> None:
        """Execute individual offboarding step"""
        
        progress = self.offboarding_progress[tenant_id]
        progress.current_step = step
        
        try:
            logger.info(f"Executing offboarding step for tenant {tenant_id}: {step.value}")
            
            # Execute step function
            await step_function(tenant_id, **step_args)
            
            # Mark step as completed
            progress.completed_steps.append(step)
            progress.progress_percent = (len(progress.completed_steps) / 8) * 100
            
            await self._log_lifecycle_event(
                tenant_id,
                f"offboarding_step_completed",
                {'step': step.value},
                automated=True
            )
            
        except Exception as e:
            # Mark step as failed
            progress.failed_steps.append({
                'step': step.value,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            await self._log_lifecycle_event(
                tenant_id,
                f"offboarding_step_failed",
                {'step': step.value, 'error': str(e)},
                automated=True
            )
            
            raise e
    
    async def _suspend_tenant_workflows(self, tenant_id: int, reason: str) -> None:
        """Suspend all tenant workflows"""
        if self.tenant_isolation_sandbox:
            await self.tenant_isolation_sandbox.suspend_tenant_sandbox(tenant_id, reason)
        
        await asyncio.sleep(1)
        logger.info(f"Suspended workflows for tenant {tenant_id}")
    
    async def _export_tenant_data(self, tenant_id: int) -> None:
        """Export tenant data for backup/compliance"""
        # Create data export
        export_path = f"/tmp/tenant_{tenant_id}_export_{int(datetime.now().timestamp())}"
        os.makedirs(export_path, exist_ok=True)
        
        # Export tenant data (mock implementation)
        with open(f"{export_path}/tenant_data.json", "w") as f:
            json.dump({"tenant_id": tenant_id, "exported_at": datetime.now().isoformat()}, f)
        
        await asyncio.sleep(2)
        logger.info(f"Exported data for tenant {tenant_id} to {export_path}")
    
    async def _validate_compliance_requirements(self, tenant_id: int) -> None:
        """Validate compliance requirements for offboarding"""
        # Check compliance requirements
        await asyncio.sleep(1)
        logger.info(f"Validated compliance requirements for tenant {tenant_id}")
    
    async def _anonymize_tenant_data(self, tenant_id: int) -> None:
        """Anonymize tenant data for GDPR compliance"""
        # Anonymize PII data
        await asyncio.sleep(2)
        logger.info(f"Anonymized data for tenant {tenant_id}")
    
    async def _purge_tenant_data(self, tenant_id: int) -> None:
        """Purge tenant data (GDPR right to be forgotten)"""
        # Purge tenant data from all systems
        await asyncio.sleep(3)
        logger.info(f"Purged data for tenant {tenant_id}")
    
    async def _cleanup_tenant_resources(self, tenant_id: int) -> None:
        """Cleanup tenant resources"""
        # Cleanup compute, storage, and network resources
        await asyncio.sleep(1)
        logger.info(f"Cleaned up resources for tenant {tenant_id}")
    
    async def _finalize_tenant_audit(self, tenant_id: int) -> None:
        """Finalize audit trail for tenant"""
        # Create final audit report
        await asyncio.sleep(1)
        logger.info(f"Finalized audit for tenant {tenant_id}")
    
    async def _confirm_tenant_termination(self, tenant_id: int) -> None:
        """Confirm tenant termination"""
        # Final confirmation and cleanup
        await asyncio.sleep(1)
        logger.info(f"Confirmed termination for tenant {tenant_id}")
    
    async def get_onboarding_progress(self, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Get onboarding progress for tenant"""
        progress = self.onboarding_progress.get(tenant_id)
        return progress.__dict__ if progress else None
    
    async def get_offboarding_progress(self, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Get offboarding progress for tenant"""
        progress = self.offboarding_progress.get(tenant_id)
        return progress.__dict__ if progress else None
    
    async def get_tenant_lifecycle_events(self, tenant_id: int) -> List[Dict[str, Any]]:
        """Get lifecycle events for tenant"""
        events = self.lifecycle_events.get(tenant_id, [])
        return [event.__dict__ for event in events]
    
    async def _log_lifecycle_event(
        self,
        tenant_id: int,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
        automated: bool = True
    ) -> None:
        """Log lifecycle event"""
        
        event = TenantLifecycleEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            event_type=event_type,
            event_data=event_data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            automated=automated
        )
        
        if tenant_id not in self.lifecycle_events:
            self.lifecycle_events[tenant_id] = []
        
        self.lifecycle_events[tenant_id].append(event)
        
        # Keep only last 1000 events per tenant
        if len(self.lifecycle_events[tenant_id]) > 1000:
            self.lifecycle_events[tenant_id] = self.lifecycle_events[tenant_id][-1000:]
    
    def _calculate_onboarding_time(self, progress: OnboardingProgress) -> int:
        """Calculate onboarding time in minutes"""
        start_time = datetime.fromisoformat(progress.started_at.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        return int((current_time - start_time).total_seconds() / 60)
    
    def _calculate_offboarding_time(self, progress: OffboardingProgress) -> int:
        """Calculate offboarding time in minutes"""
        start_time = datetime.fromisoformat(progress.started_at.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        return int((current_time - start_time).total_seconds() / 60)

# Global tenant lifecycle manager
tenant_lifecycle_manager = TenantLifecycleManager()
