# Override Ledger Schema - Chapter 7.5
# Tasks 7.5-T01 to T50: Override ledger, maker-checker workflows, tamper-evidence

import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)

class OverrideType(Enum):
    """Types of overrides"""
    POLICY_BYPASS = "policy_bypass"
    MANUAL_CORRECTION = "manual_correction"
    ESCALATED_APPROVAL = "escalated_approval"
    EMERGENCY_OVERRIDE = "emergency_override"
    COMPLIANCE_EXCEPTION = "compliance_exception"
    BUSINESS_EXCEPTION = "business_exception"
    TECHNICAL_OVERRIDE = "technical_override"
    RISK_ACCEPTANCE = "risk_acceptance"

class OverrideStatus(Enum):
    """Override status"""
    REQUESTED = "requested"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DENIED = "denied"
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    COMPLETED = "completed"

class OverrideRiskLevel(Enum):
    """Override risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ApprovalRole(Enum):
    """Roles that can approve overrides"""
    COMPLIANCE_MANAGER = "compliance_manager"
    CFO = "cfo"
    CRO = "cro"
    RISK_MANAGER = "risk_manager"
    OPERATIONS_MANAGER = "operations_manager"
    SECURITY_OFFICER = "security_officer"
    LEGAL_COUNSEL = "legal_counsel"

@dataclass
class OverrideRequest:
    """Override request details"""
    request_id: str
    override_type: OverrideType
    requested_by: str
    requested_at: str
    
    # Context
    workflow_id: str
    execution_id: str
    tenant_id: int
    
    # Justification
    reason: str
    business_justification: str
    technical_details: str
    risk_assessment: str
    
    # Risk and impact
    risk_level: OverrideRiskLevel
    estimated_impact: str
    
    # Optional fields with defaults
    policy_id: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    affected_users: int = 0
    compliance_frameworks: List[str] = field(default_factory=list)
    regulatory_impact: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['override_type'] = self.override_type.value
        data['risk_level'] = self.risk_level.value
        return data

@dataclass
class ApprovalDecision:
    """Approval decision details"""
    decision_id: str
    approver_id: str
    approver_role: ApprovalRole
    decision: str  # approved, denied, escalated
    decision_at: str
    
    # Decision rationale
    approval_notes: str
    conditions: List[str] = field(default_factory=list)
    expiration_date: Optional[str] = None
    
    # Additional requirements
    monitoring_required: bool = False
    reporting_required: bool = False
    review_frequency_days: int = 30
    
    # Digital signature
    signature: Optional[str] = None
    signature_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['approver_role'] = self.approver_role.value
        return data

@dataclass
class OverrideExecution:
    """Override execution details"""
    execution_id: str
    executed_by: str
    executed_at: str
    
    # Execution context
    original_values: Dict[str, Any]
    override_values: Dict[str, Any]
    execution_method: str
    
    # Results
    success: bool
    error_message: Optional[str] = None
    rollback_available: bool = True
    rollback_procedure: Optional[str] = None
    
    # Evidence
    before_snapshot: Dict[str, Any] = field(default_factory=dict)
    after_snapshot: Dict[str, Any] = field(default_factory=dict)
    execution_logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class OverrideMonitoring:
    """Override monitoring and compliance tracking"""
    monitoring_id: str
    override_id: str
    
    # Monitoring schedule
    start_date: str
    end_date: Optional[str] = None
    frequency_days: int = 7
    
    # Monitoring results
    checks_performed: int = 0
    violations_detected: int = 0
    last_check_at: Optional[str] = None
    next_check_at: Optional[str] = None
    
    # Compliance status
    compliance_status: str = "compliant"
    risk_score: float = 0.0
    trend: str = "stable"  # improving, stable, degrading
    
    # Alerts
    alerts_generated: int = 0
    escalations_triggered: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class OverrideLedgerEntry:
    """Complete override ledger entry"""
    # Core identification
    override_id: str
    ledger_entry_id: str
    tenant_id: int
    
    # Request details
    request: OverrideRequest
    
    # Approval chain
    approval_decisions: List[ApprovalDecision] = field(default_factory=list)
    
    # Execution details
    execution: Optional[OverrideExecution] = None
    
    # Monitoring
    monitoring: Optional[OverrideMonitoring] = None
    
    # Status and lifecycle
    status: OverrideStatus = OverrideStatus.REQUESTED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Expiration and revocation
    expires_at: Optional[str] = None
    auto_revoke: bool = True
    revoked_at: Optional[str] = None
    revoked_by: Optional[str] = None
    revocation_reason: Optional[str] = None
    
    # Audit trail
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cryptographic integrity
    entry_hash: Optional[str] = None
    previous_hash: Optional[str] = None
    signature: Optional[str] = None
    
    # Links to other systems
    evidence_pack_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['request'] = self.request.to_dict()
        data['approval_decisions'] = [ad.to_dict() for ad in self.approval_decisions]
        data['status'] = self.status.value
        if self.execution:
            data['execution'] = self.execution.to_dict()
        if self.monitoring:
            data['monitoring'] = self.monitoring.to_dict()
        return data
    
    def calculate_entry_hash(self, previous_hash: Optional[str] = None) -> str:
        """Calculate hash for blockchain-style chaining"""
        
        hash_content = {
            'override_id': self.override_id,
            'tenant_id': self.tenant_id,
            'request': self.request.to_dict(),
            'approval_decisions': [ad.to_dict() for ad in self.approval_decisions],
            'execution': self.execution.to_dict() if self.execution else None,
            'status': self.status.value,
            'created_at': self.created_at,
            'previous_hash': previous_hash
        }
        
        content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content_json.encode()).hexdigest()
    
    def add_audit_entry(self, action: str, actor: str, details: str) -> None:
        """Add audit trail entry"""
        
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'action': action,
            'actor': actor,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details
        }
        
        self.audit_trail.append(audit_entry)
        self.updated_at = audit_entry['timestamp']

class OverrideLedgerManager:
    """
    Override Ledger Manager - Tasks 7.5-T01 to T25
    Manages override requests, approvals, and execution with full audit trails
    """
    
    def __init__(self):
        self.ledger_entries: Dict[str, OverrideLedgerEntry] = {}
        self.approval_workflows: Dict[OverrideType, List[ApprovalRole]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize approval workflows
        self._initialize_approval_workflows()
    
    def _initialize_approval_workflows(self) -> None:
        """Initialize approval workflows for different override types"""
        
        self.approval_workflows = {
            OverrideType.POLICY_BYPASS: [ApprovalRole.COMPLIANCE_MANAGER, ApprovalRole.CFO],
            OverrideType.MANUAL_CORRECTION: [ApprovalRole.OPERATIONS_MANAGER],
            OverrideType.ESCALATED_APPROVAL: [ApprovalRole.CRO, ApprovalRole.CFO],
            OverrideType.EMERGENCY_OVERRIDE: [ApprovalRole.OPERATIONS_MANAGER, ApprovalRole.COMPLIANCE_MANAGER],
            OverrideType.COMPLIANCE_EXCEPTION: [ApprovalRole.COMPLIANCE_MANAGER, ApprovalRole.LEGAL_COUNSEL],
            OverrideType.BUSINESS_EXCEPTION: [ApprovalRole.CRO],
            OverrideType.TECHNICAL_OVERRIDE: [ApprovalRole.OPERATIONS_MANAGER, ApprovalRole.SECURITY_OFFICER],
            OverrideType.RISK_ACCEPTANCE: [ApprovalRole.RISK_MANAGER, ApprovalRole.CFO]
        }
        
        self.logger.info(f"✅ Initialized approval workflows for {len(self.approval_workflows)} override types")
    
    async def create_override_request(
        self,
        override_type: OverrideType,
        workflow_id: str,
        execution_id: str,
        tenant_id: int,
        requested_by: str,
        reason: str,
        business_justification: str,
        technical_details: str,
        risk_assessment: str,
        risk_level: OverrideRiskLevel,
        estimated_impact: str,
        policy_id: Optional[str] = None,
        compliance_frameworks: List[str] = None,
        affected_systems: List[str] = None
    ) -> OverrideLedgerEntry:
        """Create new override request"""
        
        override_id = str(uuid.uuid4())
        ledger_entry_id = str(uuid.uuid4())
        
        # Create override request
        request = OverrideRequest(
            request_id=str(uuid.uuid4()),
            override_type=override_type,
            requested_by=requested_by,
            requested_at=datetime.now(timezone.utc).isoformat(),
            workflow_id=workflow_id,
            execution_id=execution_id,
            tenant_id=tenant_id,
            policy_id=policy_id,
            reason=reason,
            business_justification=business_justification,
            technical_details=technical_details,
            risk_assessment=risk_assessment,
            risk_level=risk_level,
            estimated_impact=estimated_impact,
            compliance_frameworks=compliance_frameworks or [],
            affected_systems=affected_systems or []
        )
        
        # Create ledger entry
        ledger_entry = OverrideLedgerEntry(
            override_id=override_id,
            ledger_entry_id=ledger_entry_id,
            tenant_id=tenant_id,
            request=request,
            status=OverrideStatus.REQUESTED
        )
        
        # Add initial audit entry
        ledger_entry.add_audit_entry(
            action="override_requested",
            actor=requested_by,
            details=f"Override request created: {override_type.value} for {workflow_id}"
        )
        
        # Calculate initial hash
        ledger_entry.entry_hash = ledger_entry.calculate_entry_hash()
        
        # Store in ledger
        self.ledger_entries[override_id] = ledger_entry
        
        # Automatically move to pending approval if approval required
        required_approvers = self.approval_workflows.get(override_type, [])
        if required_approvers:
            ledger_entry.status = OverrideStatus.PENDING_APPROVAL
            ledger_entry.add_audit_entry(
                action="approval_required",
                actor="system",
                details=f"Requires approval from: {[role.value for role in required_approvers]}"
            )
        
        self.logger.info(f"✅ Created override request: {override_id} ({override_type.value})")
        return ledger_entry
    
    async def approve_override(
        self,
        override_id: str,
        approver_id: str,
        approver_role: ApprovalRole,
        decision: str,
        approval_notes: str,
        conditions: List[str] = None,
        expiration_hours: Optional[int] = None,
        monitoring_required: bool = False
    ) -> bool:
        """Approve or deny override request"""
        
        ledger_entry = self.ledger_entries.get(override_id)
        if not ledger_entry:
            self.logger.error(f"❌ Override {override_id} not found")
            return False
        
        if ledger_entry.status != OverrideStatus.PENDING_APPROVAL:
            self.logger.error(f"❌ Override {override_id} not in pending approval status")
            return False
        
        # Create approval decision
        decision_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        expiration_date = None
        if expiration_hours:
            expiration_date = (now + timedelta(hours=expiration_hours)).isoformat()
        
        approval_decision = ApprovalDecision(
            decision_id=decision_id,
            approver_id=approver_id,
            approver_role=approver_role,
            decision=decision,
            decision_at=now.isoformat(),
            approval_notes=approval_notes,
            conditions=conditions or [],
            expiration_date=expiration_date,
            monitoring_required=monitoring_required
        )
        
        # Add digital signature (mock)
        signature_data = f"{decision_id}:{approver_id}:{decision}:{now.isoformat()}"
        approval_decision.signature = hashlib.sha256(signature_data.encode()).hexdigest()
        approval_decision.signature_timestamp = now.isoformat()
        
        # Add to ledger entry
        ledger_entry.approval_decisions.append(approval_decision)
        
        # Update status based on decision
        if decision.lower() == "approved":
            # Check if all required approvals are obtained
            required_approvers = self.approval_workflows.get(ledger_entry.request.override_type, [])
            approver_roles = [ad.approver_role for ad in ledger_entry.approval_decisions if ad.decision.lower() == "approved"]
            
            if all(role in approver_roles for role in required_approvers):
                ledger_entry.status = OverrideStatus.APPROVED
                ledger_entry.expires_at = expiration_date
                
                # Set up monitoring if required
                if monitoring_required:
                    await self._setup_monitoring(ledger_entry)
            
        elif decision.lower() == "denied":
            ledger_entry.status = OverrideStatus.DENIED
        
        # Add audit entry
        ledger_entry.add_audit_entry(
            action=f"override_{decision.lower()}",
            actor=approver_id,
            details=f"Override {decision.lower()} by {approver_role.value}: {approval_notes}"
        )
        
        # Update hash
        ledger_entry.entry_hash = ledger_entry.calculate_entry_hash(ledger_entry.entry_hash)
        
        self.logger.info(f"✅ Override {override_id} {decision.lower()} by {approver_id}")
        return True
    
    async def execute_override(
        self,
        override_id: str,
        executed_by: str,
        original_values: Dict[str, Any],
        override_values: Dict[str, Any],
        execution_method: str,
        before_snapshot: Dict[str, Any] = None,
        rollback_procedure: Optional[str] = None
    ) -> bool:
        """Execute approved override"""
        
        ledger_entry = self.ledger_entries.get(override_id)
        if not ledger_entry:
            self.logger.error(f"❌ Override {override_id} not found")
            return False
        
        if ledger_entry.status != OverrideStatus.APPROVED:
            self.logger.error(f"❌ Override {override_id} not approved for execution")
            return False
        
        # Check expiration
        if ledger_entry.expires_at:
            expiry = datetime.fromisoformat(ledger_entry.expires_at.replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expiry:
                ledger_entry.status = OverrideStatus.EXPIRED
                ledger_entry.add_audit_entry(
                    action="override_expired",
                    actor="system",
                    details="Override expired before execution"
                )
                return False
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        try:
            # Mock execution (in production, would perform actual override)
            success = True
            error_message = None
            
            # Create after snapshot (mock)
            after_snapshot = {**before_snapshot, **override_values} if before_snapshot else override_values
            
            execution = OverrideExecution(
                execution_id=execution_id,
                executed_by=executed_by,
                executed_at=now,
                original_values=original_values,
                override_values=override_values,
                execution_method=execution_method,
                success=success,
                error_message=error_message,
                rollback_available=bool(rollback_procedure),
                rollback_procedure=rollback_procedure,
                before_snapshot=before_snapshot or {},
                after_snapshot=after_snapshot,
                execution_logs=[f"Override executed at {now}"]
            )
            
            ledger_entry.execution = execution
            ledger_entry.status = OverrideStatus.ACTIVE if success else OverrideStatus.DENIED
            
            # Add audit entry
            ledger_entry.add_audit_entry(
                action="override_executed",
                actor=executed_by,
                details=f"Override executed successfully using {execution_method}"
            )
            
            # Update hash
            ledger_entry.entry_hash = ledger_entry.calculate_entry_hash(ledger_entry.entry_hash)
            
            self.logger.info(f"✅ Override {override_id} executed successfully")
            return True
            
        except Exception as e:
            # Record execution failure
            execution = OverrideExecution(
                execution_id=execution_id,
                executed_by=executed_by,
                executed_at=now,
                original_values=original_values,
                override_values=override_values,
                execution_method=execution_method,
                success=False,
                error_message=str(e),
                before_snapshot=before_snapshot or {},
                execution_logs=[f"Override execution failed at {now}: {str(e)}"]
            )
            
            ledger_entry.execution = execution
            ledger_entry.status = OverrideStatus.DENIED
            
            ledger_entry.add_audit_entry(
                action="override_execution_failed",
                actor=executed_by,
                details=f"Override execution failed: {str(e)}"
            )
            
            self.logger.error(f"❌ Override {override_id} execution failed: {e}")
            return False
    
    async def _setup_monitoring(self, ledger_entry: OverrideLedgerEntry) -> None:
        """Set up monitoring for approved override"""
        
        monitoring_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        # Determine monitoring period
        end_date = None
        if ledger_entry.expires_at:
            end_date = ledger_entry.expires_at
        
        monitoring = OverrideMonitoring(
            monitoring_id=monitoring_id,
            override_id=ledger_entry.override_id,
            start_date=now.isoformat(),
            end_date=end_date,
            frequency_days=7,  # Weekly monitoring
            next_check_at=(now + timedelta(days=7)).isoformat()
        )
        
        ledger_entry.monitoring = monitoring
        
        ledger_entry.add_audit_entry(
            action="monitoring_setup",
            actor="system",
            details=f"Monitoring setup with {monitoring.frequency_days}-day frequency"
        )
        
        self.logger.info(f"✅ Monitoring setup for override {ledger_entry.override_id}")
    
    async def revoke_override(
        self,
        override_id: str,
        revoked_by: str,
        revocation_reason: str
    ) -> bool:
        """Revoke active override"""
        
        ledger_entry = self.ledger_entries.get(override_id)
        if not ledger_entry:
            self.logger.error(f"❌ Override {override_id} not found")
            return False
        
        if ledger_entry.status not in [OverrideStatus.APPROVED, OverrideStatus.ACTIVE]:
            self.logger.error(f"❌ Override {override_id} cannot be revoked (status: {ledger_entry.status.value})")
            return False
        
        # Update status
        ledger_entry.status = OverrideStatus.REVOKED
        ledger_entry.revoked_at = datetime.now(timezone.utc).isoformat()
        ledger_entry.revoked_by = revoked_by
        ledger_entry.revocation_reason = revocation_reason
        
        # Add audit entry
        ledger_entry.add_audit_entry(
            action="override_revoked",
            actor=revoked_by,
            details=f"Override revoked: {revocation_reason}"
        )
        
        # Update hash
        ledger_entry.entry_hash = ledger_entry.calculate_entry_hash(ledger_entry.entry_hash)
        
        self.logger.info(f"✅ Override {override_id} revoked by {revoked_by}")
        return True
    
    def get_override_entry(self, override_id: str) -> Optional[OverrideLedgerEntry]:
        """Get override ledger entry"""
        return self.ledger_entries.get(override_id)
    
    def list_overrides(
        self,
        tenant_id: Optional[int] = None,
        status: Optional[OverrideStatus] = None,
        override_type: Optional[OverrideType] = None,
        requested_by: Optional[str] = None,
        limit: int = 100
    ) -> List[OverrideLedgerEntry]:
        """List override entries with filtering"""
        
        entries = list(self.ledger_entries.values())
        
        if tenant_id:
            entries = [e for e in entries if e.tenant_id == tenant_id]
        
        if status:
            entries = [e for e in entries if e.status == status]
        
        if override_type:
            entries = [e for e in entries if e.request.override_type == override_type]
        
        if requested_by:
            entries = [e for e in entries if e.request.requested_by == requested_by]
        
        # Sort by creation time (newest first)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        
        return entries[:limit]
    
    def get_pending_approvals(
        self,
        approver_role: Optional[ApprovalRole] = None,
        tenant_id: Optional[int] = None
    ) -> List[OverrideLedgerEntry]:
        """Get overrides pending approval"""
        
        pending_entries = [
            entry for entry in self.ledger_entries.values()
            if entry.status == OverrideStatus.PENDING_APPROVAL
        ]
        
        if tenant_id:
            pending_entries = [e for e in pending_entries if e.tenant_id == tenant_id]
        
        if approver_role:
            # Filter to overrides that need this specific role's approval
            filtered_entries = []
            for entry in pending_entries:
                required_roles = self.approval_workflows.get(entry.request.override_type, [])
                existing_approvals = [ad.approver_role for ad in entry.approval_decisions if ad.decision.lower() == "approved"]
                
                if approver_role in required_roles and approver_role not in existing_approvals:
                    filtered_entries.append(entry)
            
            pending_entries = filtered_entries
        
        return pending_entries
    
    def verify_ledger_integrity(self, override_id: str) -> Dict[str, Any]:
        """Verify override ledger entry integrity"""
        
        entry = self.ledger_entries.get(override_id)
        if not entry:
            return {'valid': False, 'error': 'Override entry not found'}
        
        try:
            # Recalculate hash
            calculated_hash = entry.calculate_entry_hash(entry.previous_hash)
            stored_hash = entry.entry_hash
            
            hash_valid = calculated_hash == stored_hash
            
            # Verify approval signatures
            signature_valid = True
            for approval in entry.approval_decisions:
                if approval.signature:
                    expected_sig_data = f"{approval.decision_id}:{approval.approver_id}:{approval.decision}:{approval.decision_at}"
                    expected_signature = hashlib.sha256(expected_sig_data.encode()).hexdigest()
                    if approval.signature != expected_signature:
                        signature_valid = False
                        break
            
            return {
                'valid': hash_valid and signature_valid,
                'hash_valid': hash_valid,
                'signature_valid': signature_valid,
                'calculated_hash': calculated_hash,
                'stored_hash': stored_hash,
                'audit_trail_length': len(entry.audit_trail),
                'approval_count': len(entry.approval_decisions)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def get_ledger_statistics(self) -> Dict[str, Any]:
        """Get override ledger statistics"""
        
        total_overrides = len(self.ledger_entries)
        
        overrides_by_type = {}
        overrides_by_status = {}
        overrides_by_tenant = {}
        overrides_by_risk = {}
        
        for entry in self.ledger_entries.values():
            # By type
            override_type = entry.request.override_type.value
            overrides_by_type[override_type] = overrides_by_type.get(override_type, 0) + 1
            
            # By status
            status = entry.status.value
            overrides_by_status[status] = overrides_by_status.get(status, 0) + 1
            
            # By tenant
            tenant_id = entry.tenant_id
            overrides_by_tenant[tenant_id] = overrides_by_tenant.get(tenant_id, 0) + 1
            
            # By risk level
            risk_level = entry.request.risk_level.value
            overrides_by_risk[risk_level] = overrides_by_risk.get(risk_level, 0) + 1
        
        return {
            'total_overrides': total_overrides,
            'overrides_by_type': overrides_by_type,
            'overrides_by_status': overrides_by_status,
            'overrides_by_tenant': overrides_by_tenant,
            'overrides_by_risk_level': overrides_by_risk,
            'pending_approvals': len([e for e in self.ledger_entries.values() if e.status == OverrideStatus.PENDING_APPROVAL]),
            'active_overrides': len([e for e in self.ledger_entries.values() if e.status == OverrideStatus.ACTIVE])
        }

# Global instance
override_ledger_manager = OverrideLedgerManager()
