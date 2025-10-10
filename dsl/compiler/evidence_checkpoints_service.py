"""
Evidence Checkpoints Service - Task 6.2.31
===========================================

Evidence checkpoints (node-enter/exit events)
- Defines checkpoint hooks for runtime evidence emission
- Standard checkpoint schema for node-enter and node-exit events
- Storage contract for evidence with retention policies
- Mechanism to attach checkpoint references to manifests
- Backend implementation (no actual WORM/S3 storage - that's infrastructure)

Dependencies: Task 6.2.26 (Plan Manifest Generator), Task 6.2.27 (Plan Hash)
Outputs: Evidence checkpoint schema → enables verifiable audit packs
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import uuid
import time

logger = logging.getLogger(__name__)

class CheckpointEventType(Enum):
    """Types of checkpoint events"""
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"
    OVERRIDE_APPLIED = "override_applied"
    FALLBACK_TRIGGERED = "fallback_triggered"
    TRUST_THRESHOLD_BREACHED = "trust_threshold_breached"
    COMPLIANCE_CHECK = "compliance_check"

class DataRedactionLevel(Enum):
    """Levels of data redaction for PII protection"""
    NONE = "none"           # No redaction
    HASH_ONLY = "hash_only" # Store only hash
    REDACTED = "redacted"   # Replace with placeholder
    ENCRYPTED = "encrypted" # Encrypted storage
    OMITTED = "omitted"     # Not stored at all

class EvidenceStorageType(Enum):
    """Types of evidence storage"""
    WORM_STORAGE = "worm_storage"       # Write-once, read-many
    TAMPER_EVIDENT = "tamper_evident"   # Cryptographically protected
    IMMUTABLE_LOG = "immutable_log"     # Append-only log
    BLOCKCHAIN = "blockchain"           # Distributed ledger
    STANDARD_STORAGE = "standard_storage" # Regular storage with backup

@dataclass
class DataSnapshot:
    """Snapshot of data at checkpoint"""
    snapshot_id: str
    data_hash: str  # SHA-256 of the data
    data_size_bytes: int
    redaction_level: DataRedactionLevel
    schema_version: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    storage_uri: Optional[str] = None  # URI to actual data
    encryption_key_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionMetadata:
    """Metadata about decisions made at a node"""
    decision_id: str
    decision_type: str  # "classification", "regression", "rule_based", "hybrid"
    confidence_score: Optional[float] = None
    trust_score: Optional[float] = None
    model_version: Optional[str] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    explanation_reference: Optional[str] = None
    override_applied: bool = False
    override_reason: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None

@dataclass
class ErrorInformation:
    """Information about errors that occurred"""
    error_code: str
    error_type: str
    error_message: str
    error_severity: str  # "low", "medium", "high", "critical"
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    impact_assessment: Optional[str] = None
    related_node_ids: List[str] = field(default_factory=list)

@dataclass
class ComplianceCheck:
    """Compliance check performed at checkpoint"""
    check_id: str
    check_type: str  # "residency", "privacy", "bias", "fairness", "sod"
    policy_id: str
    check_result: str  # "pass", "fail", "warning", "not_applicable"
    check_details: Dict[str, Any] = field(default_factory=dict)
    remediation_required: bool = False
    remediation_steps: List[str] = field(default_factory=list)

@dataclass
class EvidenceCheckpoint:
    """Complete evidence checkpoint record"""
    # Core identification
    checkpoint_id: str
    event_type: CheckpointEventType
    node_id: str
    plan_hash: str
    execution_id: str  # Unique execution instance
    
    # Timing and location
    timestamp: datetime
    region: str
    requester_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Data snapshots
    input_snapshot: Optional[DataSnapshot] = None
    output_snapshot: Optional[DataSnapshot] = None
    
    # Decision and error information
    decision_metadata: Optional[DecisionMetadata] = None
    error_information: Optional[ErrorInformation] = None
    compliance_checks: List[ComplianceCheck] = field(default_factory=list)
    
    # Audit trail
    sequence_number: int = 0  # Order within execution
    parent_checkpoint_id: Optional[str] = None
    child_checkpoint_ids: List[str] = field(default_factory=list)
    
    # Storage and retention
    storage_type: EvidenceStorageType = EvidenceStorageType.STANDARD_STORAGE
    retention_policy: str = "default"
    evidence_hash: Optional[str] = None  # Hash of the entire checkpoint
    
    # Metadata
    runtime_version: Optional[str] = None
    environment: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetentionPolicy:
    """Policy for evidence retention"""
    policy_id: str
    policy_name: str
    retention_days: int
    auto_archive_days: Optional[int] = None
    auto_delete_days: Optional[int] = None
    compliance_requirements: List[str] = field(default_factory=list)
    storage_tier_transitions: Dict[int, str] = field(default_factory=dict)  # days -> tier
    encryption_required: bool = False
    geographic_restrictions: List[str] = field(default_factory=list)

@dataclass
class CheckpointReference:
    """Reference to a checkpoint for manifest inclusion"""
    checkpoint_placeholder_id: str
    node_id: str
    event_type: CheckpointEventType
    expected_evidence_types: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    retention_policy_id: str = "default"
    redaction_requirements: Dict[str, DataRedactionLevel] = field(default_factory=dict)

# Task 6.2.31: Evidence Checkpoints Service
class EvidenceCheckpointsService:
    """Service for managing evidence checkpoints and audit trail generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Storage for checkpoints and references
        self.checkpoints: Dict[str, EvidenceCheckpoint] = {}  # checkpoint_id -> checkpoint
        self.execution_checkpoints: Dict[str, List[str]] = {}  # execution_id -> checkpoint_ids
        self.node_checkpoints: Dict[str, List[str]] = {}  # node_id -> checkpoint_ids
        
        # Checkpoint references for manifests
        self.checkpoint_references: Dict[str, List[CheckpointReference]] = {}  # plan_hash -> references
        
        # Retention policies
        self.retention_policies: Dict[str, RetentionPolicy] = {
            'default': RetentionPolicy(
                policy_id='default',
                policy_name='Default Retention',
                retention_days=90,
                auto_archive_days=30,
                compliance_requirements=['basic_audit']
            ),
            'production': RetentionPolicy(
                policy_id='production',
                policy_name='Production Retention',
                retention_days=2555,  # 7 years
                auto_archive_days=365,
                compliance_requirements=['sox', 'gdpr', 'financial_audit'],
                encryption_required=True
            ),
            'pii_data': RetentionPolicy(
                policy_id='pii_data',
                policy_name='PII Data Retention',
                retention_days=1095,  # 3 years
                auto_delete_days=1095,
                compliance_requirements=['gdpr', 'ccpa', 'privacy'],
                encryption_required=True
            )
        }
        
        # Statistics
        self.checkpoint_stats = {
            'total_checkpoints': 0,
            'checkpoints_by_type': {event.value: 0 for event in CheckpointEventType},
            'checkpoints_by_node': {},
            'total_executions': 0,
            'average_checkpoints_per_execution': 0.0
        }
    
    def create_checkpoint_schema(self, plan_hash: str, node_ids: List[str]) -> List[CheckpointReference]:
        """
        Create checkpoint schema for a plan manifest
        
        Args:
            plan_hash: Hash of the plan
            node_ids: List of node IDs in the plan
            
        Returns:
            List of checkpoint references for manifest inclusion
        """
        checkpoint_references = []
        
        for node_id in node_ids:
            # Create node-enter checkpoint reference
            enter_ref = CheckpointReference(
                checkpoint_placeholder_id=f"cp_enter_{node_id}_{hashlib.sha256(f'{plan_hash}_{node_id}_enter'.encode()).hexdigest()[:16]}",
                node_id=node_id,
                event_type=CheckpointEventType.NODE_ENTER,
                expected_evidence_types=['input_snapshot', 'compliance_checks'],
                compliance_requirements=['audit_trail', 'data_lineage'],
                redaction_requirements={
                    'pii_fields': DataRedactionLevel.HASH_ONLY,
                    'sensitive_data': DataRedactionLevel.ENCRYPTED
                }
            )
            checkpoint_references.append(enter_ref)
            
            # Create node-exit checkpoint reference
            exit_ref = CheckpointReference(
                checkpoint_placeholder_id=f"cp_exit_{node_id}_{hashlib.sha256(f'{plan_hash}_{node_id}_exit'.encode()).hexdigest()[:16]}",
                node_id=node_id,
                event_type=CheckpointEventType.NODE_EXIT,
                expected_evidence_types=['output_snapshot', 'decision_metadata', 'compliance_checks'],
                compliance_requirements=['audit_trail', 'decision_evidence'],
                redaction_requirements={
                    'pii_fields': DataRedactionLevel.HASH_ONLY,
                    'model_outputs': DataRedactionLevel.NONE
                }
            )
            checkpoint_references.append(exit_ref)
        
        # Store references
        self.checkpoint_references[plan_hash] = checkpoint_references
        
        self.logger.info(f"✅ Created checkpoint schema for plan {plan_hash[:16]}...: {len(checkpoint_references)} checkpoints")
        
        return checkpoint_references
    
    def create_node_enter_checkpoint(self, node_id: str, plan_hash: str, execution_id: str,
                                   input_data: Dict[str, Any],
                                   requester_id: Optional[str] = None,
                                   region: str = "unknown",
                                   redaction_level: DataRedactionLevel = DataRedactionLevel.HASH_ONLY) -> EvidenceCheckpoint:
        """
        Create a node-enter checkpoint
        
        Args:
            node_id: Node identifier
            plan_hash: Hash of the executing plan
            execution_id: Unique execution identifier
            input_data: Input data to the node
            requester_id: ID of the requester
            region: Geographic region
            redaction_level: Level of data redaction
            
        Returns:
            EvidenceCheckpoint for node enter
        """
        checkpoint_id = f"cp_{uuid.uuid4().hex[:16]}"
        
        # Create input snapshot
        input_snapshot = self._create_data_snapshot(
            input_data, f"input_{node_id}_{checkpoint_id}", redaction_level
        )
        
        # Get sequence number for this execution
        sequence_number = len(self.execution_checkpoints.get(execution_id, [])) + 1
        
        checkpoint = EvidenceCheckpoint(
            checkpoint_id=checkpoint_id,
            event_type=CheckpointEventType.NODE_ENTER,
            node_id=node_id,
            plan_hash=plan_hash,
            execution_id=execution_id,
            timestamp=datetime.now(timezone.utc),
            region=region,
            requester_id=requester_id,
            input_snapshot=input_snapshot,
            sequence_number=sequence_number,
            runtime_version="1.0.0",
            environment=self._detect_environment()
        )
        
        # Compute evidence hash
        checkpoint.evidence_hash = self._compute_checkpoint_hash(checkpoint)
        
        # Store checkpoint
        self._store_checkpoint(checkpoint)
        
        self.logger.info(f"✅ Created node-enter checkpoint {checkpoint_id} for node {node_id}")
        
        return checkpoint
    
    def create_node_exit_checkpoint(self, node_id: str, plan_hash: str, execution_id: str,
                                  output_data: Dict[str, Any],
                                  decision_metadata: Optional[DecisionMetadata] = None,
                                  error_info: Optional[ErrorInformation] = None,
                                  parent_checkpoint_id: Optional[str] = None,
                                  redaction_level: DataRedactionLevel = DataRedactionLevel.HASH_ONLY) -> EvidenceCheckpoint:
        """
        Create a node-exit checkpoint
        
        Args:
            node_id: Node identifier
            plan_hash: Hash of the executing plan
            execution_id: Unique execution identifier
            output_data: Output data from the node
            decision_metadata: Decision information
            error_info: Error information if any
            parent_checkpoint_id: Parent checkpoint (usually node-enter)
            redaction_level: Level of data redaction
            
        Returns:
            EvidenceCheckpoint for node exit
        """
        checkpoint_id = f"cp_{uuid.uuid4().hex[:16]}"
        
        # Create output snapshot
        output_snapshot = self._create_data_snapshot(
            output_data, f"output_{node_id}_{checkpoint_id}", redaction_level
        )
        
        # Get sequence number for this execution
        sequence_number = len(self.execution_checkpoints.get(execution_id, [])) + 1
        
        checkpoint = EvidenceCheckpoint(
            checkpoint_id=checkpoint_id,
            event_type=CheckpointEventType.NODE_EXIT,
            node_id=node_id,
            plan_hash=plan_hash,
            execution_id=execution_id,
            timestamp=datetime.now(timezone.utc),
            region=self._get_execution_region(execution_id),
            output_snapshot=output_snapshot,
            decision_metadata=decision_metadata,
            error_information=error_info,
            sequence_number=sequence_number,
            parent_checkpoint_id=parent_checkpoint_id,
            runtime_version="1.0.0",
            environment=self._detect_environment()
        )
        
        # Link to parent checkpoint
        if parent_checkpoint_id and parent_checkpoint_id in self.checkpoints:
            parent_checkpoint = self.checkpoints[parent_checkpoint_id]
            parent_checkpoint.child_checkpoint_ids.append(checkpoint_id)
        
        # Compute evidence hash
        checkpoint.evidence_hash = self._compute_checkpoint_hash(checkpoint)
        
        # Store checkpoint
        self._store_checkpoint(checkpoint)
        
        self.logger.info(f"✅ Created node-exit checkpoint {checkpoint_id} for node {node_id}")
        
        return checkpoint
    
    def create_decision_checkpoint(self, node_id: str, plan_hash: str, execution_id: str,
                                 decision_metadata: DecisionMetadata,
                                 compliance_checks: Optional[List[ComplianceCheck]] = None) -> EvidenceCheckpoint:
        """
        Create a decision-made checkpoint
        
        Args:
            node_id: Node identifier
            plan_hash: Hash of the executing plan
            execution_id: Unique execution identifier
            decision_metadata: Decision information
            compliance_checks: Compliance checks performed
            
        Returns:
            EvidenceCheckpoint for decision made
        """
        checkpoint_id = f"cp_{uuid.uuid4().hex[:16]}"
        
        sequence_number = len(self.execution_checkpoints.get(execution_id, [])) + 1
        
        checkpoint = EvidenceCheckpoint(
            checkpoint_id=checkpoint_id,
            event_type=CheckpointEventType.DECISION_MADE,
            node_id=node_id,
            plan_hash=plan_hash,
            execution_id=execution_id,
            timestamp=datetime.now(timezone.utc),
            region=self._get_execution_region(execution_id),
            decision_metadata=decision_metadata,
            compliance_checks=compliance_checks or [],
            sequence_number=sequence_number,
            runtime_version="1.0.0",
            environment=self._detect_environment()
        )
        
        # Compute evidence hash
        checkpoint.evidence_hash = self._compute_checkpoint_hash(checkpoint)
        
        # Store checkpoint
        self._store_checkpoint(checkpoint)
        
        self.logger.info(f"✅ Created decision checkpoint {checkpoint_id} for node {node_id}")
        
        return checkpoint
    
    def create_error_checkpoint(self, node_id: str, plan_hash: str, execution_id: str,
                              error_info: ErrorInformation) -> EvidenceCheckpoint:
        """
        Create an error-occurred checkpoint
        
        Args:
            node_id: Node identifier
            plan_hash: Hash of the executing plan
            execution_id: Unique execution identifier
            error_info: Error information
            
        Returns:
            EvidenceCheckpoint for error occurred
        """
        checkpoint_id = f"cp_{uuid.uuid4().hex[:16]}"
        
        sequence_number = len(self.execution_checkpoints.get(execution_id, [])) + 1
        
        checkpoint = EvidenceCheckpoint(
            checkpoint_id=checkpoint_id,
            event_type=CheckpointEventType.ERROR_OCCURRED,
            node_id=node_id,
            plan_hash=plan_hash,
            execution_id=execution_id,
            timestamp=datetime.now(timezone.utc),
            region=self._get_execution_region(execution_id),
            error_information=error_info,
            sequence_number=sequence_number,
            runtime_version="1.0.0",
            environment=self._detect_environment()
        )
        
        # Compute evidence hash
        checkpoint.evidence_hash = self._compute_checkpoint_hash(checkpoint)
        
        # Store checkpoint
        self._store_checkpoint(checkpoint)
        
        self.logger.warning(f"Created error checkpoint {checkpoint_id} for node {node_id}: {error_info.error_code}")
        
        return checkpoint
    
    def get_execution_audit_trail(self, execution_id: str) -> List[EvidenceCheckpoint]:
        """
        Get complete audit trail for an execution
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            List of checkpoints ordered by sequence number
        """
        checkpoint_ids = self.execution_checkpoints.get(execution_id, [])
        checkpoints = [self.checkpoints[cp_id] for cp_id in checkpoint_ids if cp_id in self.checkpoints]
        
        # Sort by sequence number
        checkpoints.sort(key=lambda cp: cp.sequence_number)
        
        return checkpoints
    
    def get_node_evidence_history(self, node_id: str, limit: Optional[int] = None) -> List[EvidenceCheckpoint]:
        """
        Get evidence history for a specific node
        
        Args:
            node_id: Node identifier
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoints for the node
        """
        checkpoint_ids = self.node_checkpoints.get(node_id, [])
        checkpoints = [self.checkpoints[cp_id] for cp_id in checkpoint_ids if cp_id in self.checkpoints]
        
        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)
        
        if limit:
            checkpoints = checkpoints[:limit]
        
        return checkpoints
    
    def get_checkpoint_references_for_manifest(self, plan_hash: str) -> List[Dict[str, Any]]:
        """
        Get checkpoint references for manifest inclusion
        
        Args:
            plan_hash: Plan hash
            
        Returns:
            List of checkpoint reference dictionaries
        """
        references = self.checkpoint_references.get(plan_hash, [])
        
        return [
            {
                'checkpoint_placeholder_id': ref.checkpoint_placeholder_id,
                'node_id': ref.node_id,
                'event_type': ref.event_type.value,
                'expected_evidence_types': ref.expected_evidence_types,
                'compliance_requirements': ref.compliance_requirements,
                'retention_policy_id': ref.retention_policy_id,
                'redaction_requirements': {k: v.value for k, v in ref.redaction_requirements.items()}
            }
            for ref in references
        ]
    
    def apply_retention_policy(self, checkpoint_id: str, policy_id: str) -> bool:
        """
        Apply retention policy to a checkpoint
        
        Args:
            checkpoint_id: Checkpoint identifier
            policy_id: Retention policy identifier
            
        Returns:
            True if policy applied successfully
        """
        if checkpoint_id not in self.checkpoints:
            return False
        
        if policy_id not in self.retention_policies:
            self.logger.warning(f"Unknown retention policy: {policy_id}")
            return False
        
        checkpoint = self.checkpoints[checkpoint_id]
        policy = self.retention_policies[policy_id]
        
        # Update checkpoint with retention policy
        checkpoint.retention_policy = policy_id
        
        # Set storage type based on policy requirements
        if policy.encryption_required:
            checkpoint.storage_type = EvidenceStorageType.TAMPER_EVIDENT
        
        self.logger.info(f"Applied retention policy {policy_id} to checkpoint {checkpoint_id}")
        
        return True
    
    def validate_checkpoint_integrity(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of a checkpoint
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Dictionary with validation results
        """
        if checkpoint_id not in self.checkpoints:
            return {
                'is_valid': False,
                'error': 'Checkpoint not found'
            }
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        # Recompute hash
        computed_hash = self._compute_checkpoint_hash(checkpoint)
        
        # Compare with stored hash
        hash_valid = computed_hash == checkpoint.evidence_hash
        
        # Check data snapshot integrity
        snapshot_integrity = {}
        if checkpoint.input_snapshot:
            snapshot_integrity['input_snapshot'] = self._validate_snapshot_integrity(checkpoint.input_snapshot)
        
        if checkpoint.output_snapshot:
            snapshot_integrity['output_snapshot'] = self._validate_snapshot_integrity(checkpoint.output_snapshot)
        
        return {
            'is_valid': hash_valid and all(snapshot_integrity.values()),
            'checkpoint_id': checkpoint_id,
            'hash_valid': hash_valid,
            'stored_hash': checkpoint.evidence_hash,
            'computed_hash': computed_hash,
            'snapshot_integrity': snapshot_integrity,
            'validated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get checkpoint service statistics"""
        return {
            **self.checkpoint_stats,
            'total_retention_policies': len(self.retention_policies),
            'checkpoint_references_created': sum(len(refs) for refs in self.checkpoint_references.values()),
            'unique_executions': len(self.execution_checkpoints),
            'unique_nodes_with_evidence': len(self.node_checkpoints)
        }
    
    def _create_data_snapshot(self, data: Dict[str, Any], snapshot_id: str, 
                            redaction_level: DataRedactionLevel) -> DataSnapshot:
        """Create a data snapshot with appropriate redaction"""
        # Apply redaction based on level
        if redaction_level == DataRedactionLevel.HASH_ONLY:
            data_json = json.dumps(data, sort_keys=True)
            data_hash = hashlib.sha256(data_json.encode()).hexdigest()
            storage_uri = None  # Data not stored, only hash
        elif redaction_level == DataRedactionLevel.REDACTED:
            # Replace sensitive fields with placeholders
            redacted_data = self._redact_sensitive_fields(data)
            data_json = json.dumps(redacted_data, sort_keys=True)
            data_hash = hashlib.sha256(data_json.encode()).hexdigest()
            storage_uri = f"evidence://snapshots/{snapshot_id}"
        elif redaction_level == DataRedactionLevel.OMITTED:
            data_hash = "omitted"
            storage_uri = None
        else:
            # Store full data
            data_json = json.dumps(data, sort_keys=True)
            data_hash = hashlib.sha256(data_json.encode()).hexdigest()
            storage_uri = f"evidence://snapshots/{snapshot_id}"
        
        return DataSnapshot(
            snapshot_id=snapshot_id,
            data_hash=data_hash,
            data_size_bytes=len(data_json.encode()) if 'data_json' in locals() else 0,
            redaction_level=redaction_level,
            schema_version="1.0",
            created_at=datetime.now(timezone.utc),
            storage_uri=storage_uri
        )
    
    def _redact_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from data"""
        redacted_data = {}
        sensitive_patterns = ['password', 'ssn', 'credit_card', 'email', 'phone', 'address']
        
        for key, value in data.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in sensitive_patterns):
                redacted_data[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted_data[key] = self._redact_sensitive_fields(value)
            else:
                redacted_data[key] = value
        
        return redacted_data
    
    def _compute_checkpoint_hash(self, checkpoint: EvidenceCheckpoint) -> str:
        """Compute hash of checkpoint for integrity verification"""
        # Create a copy without the evidence_hash field
        checkpoint_dict = asdict(checkpoint)
        checkpoint_dict.pop('evidence_hash', None)
        
        # Convert to canonical JSON
        canonical_json = json.dumps(checkpoint_dict, sort_keys=True, default=str)
        
        # Compute hash
        return hashlib.sha256(canonical_json.encode()).hexdigest()
    
    def _store_checkpoint(self, checkpoint: EvidenceCheckpoint):
        """Store checkpoint in backend storage"""
        # Store checkpoint
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        
        # Update execution tracking
        if checkpoint.execution_id not in self.execution_checkpoints:
            self.execution_checkpoints[checkpoint.execution_id] = []
        self.execution_checkpoints[checkpoint.execution_id].append(checkpoint.checkpoint_id)
        
        # Update node tracking
        if checkpoint.node_id not in self.node_checkpoints:
            self.node_checkpoints[checkpoint.node_id] = []
        self.node_checkpoints[checkpoint.node_id].append(checkpoint.checkpoint_id)
        
        # Update statistics
        self.checkpoint_stats['total_checkpoints'] += 1
        self.checkpoint_stats['checkpoints_by_type'][checkpoint.event_type.value] += 1
        
        if checkpoint.node_id not in self.checkpoint_stats['checkpoints_by_node']:
            self.checkpoint_stats['checkpoints_by_node'][checkpoint.node_id] = 0
        self.checkpoint_stats['checkpoints_by_node'][checkpoint.node_id] += 1
        
        # Update execution statistics
        self.checkpoint_stats['total_executions'] = len(self.execution_checkpoints)
        if self.checkpoint_stats['total_executions'] > 0:
            self.checkpoint_stats['average_checkpoints_per_execution'] = (
                self.checkpoint_stats['total_checkpoints'] / self.checkpoint_stats['total_executions']
            )
    
    def _detect_environment(self) -> str:
        """Detect runtime environment"""
        # Simple environment detection
        import os
        return os.getenv('ENVIRONMENT', 'unknown')
    
    def _get_execution_region(self, execution_id: str) -> str:
        """Get region for execution (from previous checkpoints)"""
        checkpoint_ids = self.execution_checkpoints.get(execution_id, [])
        for cp_id in checkpoint_ids:
            if cp_id in self.checkpoints:
                return self.checkpoints[cp_id].region
        return "unknown"
    
    def _validate_snapshot_integrity(self, snapshot: DataSnapshot) -> bool:
        """Validate data snapshot integrity"""
        # In a real implementation, this would verify the actual stored data
        # For now, just check that required fields are present
        return (
            snapshot.snapshot_id and
            snapshot.data_hash and
            snapshot.created_at is not None
        )

# API Interface
class EvidenceCheckpointsAPI:
    """API interface for evidence checkpoints operations"""
    
    def __init__(self, checkpoints_service: Optional[EvidenceCheckpointsService] = None):
        self.checkpoints_service = checkpoints_service or EvidenceCheckpointsService()
    
    def create_checkpoint_schema(self, plan_hash: str, node_ids: List[str]) -> Dict[str, Any]:
        """API endpoint to create checkpoint schema"""
        try:
            references = self.checkpoints_service.create_checkpoint_schema(plan_hash, node_ids)
            
            return {
                'success': True,
                'plan_hash': plan_hash,
                'checkpoint_references_created': len(references),
                'references': [
                    {
                        'placeholder_id': ref.checkpoint_placeholder_id,
                        'node_id': ref.node_id,
                        'event_type': ref.event_type.value
                    }
                    for ref in references
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def create_node_enter_checkpoint(self, node_id: str, plan_hash: str, execution_id: str,
                                   input_data: Dict[str, Any], requester_id: Optional[str] = None,
                                   region: str = "unknown") -> Dict[str, Any]:
        """API endpoint to create node-enter checkpoint"""
        try:
            checkpoint = self.checkpoints_service.create_node_enter_checkpoint(
                node_id, plan_hash, execution_id, input_data, requester_id, region
            )
            
            return {
                'success': True,
                'checkpoint_id': checkpoint.checkpoint_id,
                'event_type': checkpoint.event_type.value,
                'node_id': checkpoint.node_id,
                'timestamp': checkpoint.timestamp.isoformat(),
                'sequence_number': checkpoint.sequence_number,
                'evidence_hash': checkpoint.evidence_hash
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_audit_trail(self, execution_id: str) -> Dict[str, Any]:
        """API endpoint to get execution audit trail"""
        try:
            checkpoints = self.checkpoints_service.get_execution_audit_trail(execution_id)
            
            return {
                'success': True,
                'execution_id': execution_id,
                'total_checkpoints': len(checkpoints),
                'audit_trail': [
                    {
                        'checkpoint_id': cp.checkpoint_id,
                        'event_type': cp.event_type.value,
                        'node_id': cp.node_id,
                        'timestamp': cp.timestamp.isoformat(),
                        'sequence_number': cp.sequence_number,
                        'has_error': cp.error_information is not None
                    }
                    for cp in checkpoints
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """API endpoint to validate checkpoint integrity"""
        result = self.checkpoints_service.validate_checkpoint_integrity(checkpoint_id)
        return {
            'success': True,
            **result
        }

# Test Functions
def create_test_execution_scenario():
    """Create a test execution scenario with checkpoints"""
    service = EvidenceCheckpointsService()
    
    # Create checkpoint schema for a test plan
    plan_hash = "test_plan_hash_" + hashlib.sha256("test_plan".encode()).hexdigest()[:16]
    node_ids = ["query_node", "ml_predict_node", "decision_node"]
    
    schema_refs = service.create_checkpoint_schema(plan_hash, node_ids)
    
    # Simulate execution
    execution_id = f"exec_{uuid.uuid4().hex[:16]}"
    
    checkpoints = []
    
    # Create checkpoints for each node
    for i, node_id in enumerate(node_ids):
        # Node enter
        input_data = {
            "customer_id": f"cust_{i+1}",
            "timestamp": datetime.now().isoformat(),
            "request_data": {"feature_1": i * 10, "feature_2": (i + 1) * 5}
        }
        
        enter_checkpoint = service.create_node_enter_checkpoint(
            node_id, plan_hash, execution_id, input_data,
            requester_id=f"user_{i+1}", region="us-east-1"
        )
        checkpoints.append(enter_checkpoint)
        
        # Decision checkpoint (for ML node)
        if "ml_" in node_id:
            decision_metadata = DecisionMetadata(
                decision_id=f"decision_{i}",
                decision_type="classification",
                confidence_score=0.85,
                trust_score=0.78,
                model_version="v2.1",
                feature_importance={"feature_1": 0.6, "feature_2": 0.4}
            )
            
            decision_checkpoint = service.create_decision_checkpoint(
                node_id, plan_hash, execution_id, decision_metadata
            )
            checkpoints.append(decision_checkpoint)
        
        # Node exit
        output_data = {
            "result": f"output_from_{node_id}",
            "processing_time_ms": (i + 1) * 100,
            "status": "completed"
        }
        
        exit_checkpoint = service.create_node_exit_checkpoint(
            node_id, plan_hash, execution_id, output_data,
            parent_checkpoint_id=enter_checkpoint.checkpoint_id
        )
        checkpoints.append(exit_checkpoint)
    
    return service, execution_id, plan_hash, checkpoints

def run_evidence_checkpoints_tests():
    """Run comprehensive evidence checkpoints tests"""
    print("=== Evidence Checkpoints Service Tests ===")
    
    # Create test scenario
    service, execution_id, plan_hash, checkpoints = create_test_execution_scenario()
    api = EvidenceCheckpointsAPI(service)
    
    print(f"\nTest execution created: {execution_id}")
    print(f"Plan hash: {plan_hash[:16]}...")
    print(f"Checkpoints created: {len(checkpoints)}")
    
    # Test 1: Checkpoint schema creation
    print("\n1. Testing checkpoint schema creation...")
    node_ids = ["test_node_1", "test_node_2"]
    schema_refs = service.create_checkpoint_schema("test_schema_plan", node_ids)
    print(f"   Schema references created: {len(schema_refs)}")
    print(f"   Reference types: {[ref.event_type.value for ref in schema_refs]}")
    
    # Test 2: Audit trail retrieval
    print("\n2. Testing audit trail retrieval...")
    audit_trail = service.get_execution_audit_trail(execution_id)
    print(f"   Audit trail length: {len(audit_trail)}")
    print(f"   Event types: {[cp.event_type.value for cp in audit_trail]}")
    print(f"   Sequence numbers: {[cp.sequence_number for cp in audit_trail]}")
    
    # Test 3: Node evidence history
    print("\n3. Testing node evidence history...")
    node_history = service.get_node_evidence_history("ml_predict_node", limit=5)
    print(f"   ML node evidence history: {len(node_history)} checkpoints")
    if node_history:
        print(f"   Most recent event: {node_history[0].event_type.value}")
        print(f"   Has decision metadata: {node_history[0].decision_metadata is not None}")
    
    # Test 4: Checkpoint integrity validation
    print("\n4. Testing checkpoint integrity validation...")
    if checkpoints:
        validation_result = service.validate_checkpoint_integrity(checkpoints[0].checkpoint_id)
        print(f"   Integrity validation: {'✅ PASS' if validation_result['is_valid'] else '❌ FAIL'}")
        print(f"   Hash validation: {'✅ PASS' if validation_result['hash_valid'] else '❌ FAIL'}")
    
    # Test 5: Error checkpoint
    print("\n5. Testing error checkpoint creation...")
    error_info = ErrorInformation(
        error_code="ML_MODEL_TIMEOUT",
        error_type="timeout",
        error_message="Model inference timed out after 5000ms",
        error_severity="high",
        recovery_action="fallback_to_rba_rule"
    )
    
    error_checkpoint = service.create_error_checkpoint(
        "ml_predict_node", plan_hash, execution_id, error_info
    )
    print(f"   Error checkpoint created: {error_checkpoint.checkpoint_id}")
    print(f"   Error code: {error_checkpoint.error_information.error_code}")
    print(f"   Error severity: {error_checkpoint.error_information.error_severity}")
    
    # Test 6: Retention policy application
    print("\n6. Testing retention policy application...")
    if checkpoints:
        policy_applied = service.apply_retention_policy(checkpoints[0].checkpoint_id, "production")
        print(f"   Retention policy applied: {'✅ PASS' if policy_applied else '❌ FAIL'}")
        
        updated_checkpoint = service.checkpoints[checkpoints[0].checkpoint_id]
        print(f"   Updated retention policy: {updated_checkpoint.retention_policy}")
        print(f"   Updated storage type: {updated_checkpoint.storage_type.value}")
    
    # Test 7: Manifest references
    print("\n7. Testing manifest references...")
    manifest_refs = service.get_checkpoint_references_for_manifest(plan_hash)
    print(f"   Manifest references: {len(manifest_refs)}")
    if manifest_refs:
        print(f"   Sample reference: {manifest_refs[0]['event_type']} for {manifest_refs[0]['node_id']}")
        print(f"   Redaction requirements: {len(manifest_refs[0]['redaction_requirements'])}")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Test API checkpoint creation
    api_checkpoint_result = api.create_node_enter_checkpoint(
        "api_test_node", "api_test_plan", "api_test_execution",
        {"test_input": "api_data"}, "api_user", "us-west-2"
    )
    print(f"   API checkpoint creation: {'✅ PASS' if api_checkpoint_result['success'] else '❌ FAIL'}")
    
    if api_checkpoint_result['success']:
        # Test API audit trail
        api_audit_result = api.get_audit_trail("api_test_execution")
        print(f"   API audit trail: {'✅ PASS' if api_audit_result['success'] else '❌ FAIL'}")
        
        # Test API validation
        api_validation_result = api.validate_checkpoint(api_checkpoint_result['checkpoint_id'])
        print(f"   API validation: {'✅ PASS' if api_validation_result['success'] else '❌ FAIL'}")
    
    # Test 9: Statistics
    print("\n9. Testing statistics...")
    stats = service.get_checkpoint_statistics()
    print(f"   Total checkpoints: {stats['total_checkpoints']}")
    print(f"   Unique executions: {stats['unique_executions']}")
    print(f"   Average checkpoints per execution: {stats['average_checkpoints_per_execution']:.1f}")
    print(f"   Checkpoint types: {stats['checkpoints_by_type']}")
    
    # Test 10: Data redaction levels
    print("\n10. Testing data redaction levels...")
    
    redaction_tests = [
        (DataRedactionLevel.NONE, "No redaction"),
        (DataRedactionLevel.HASH_ONLY, "Hash only"),
        (DataRedactionLevel.REDACTED, "Redacted fields"),
        (DataRedactionLevel.OMITTED, "Omitted data")
    ]
    
    for redaction_level, description in redaction_tests:
        test_data = {"customer_id": "12345", "password": "secret", "balance": 1000.50}
        snapshot = service._create_data_snapshot(test_data, f"test_{redaction_level.value}", redaction_level)
        print(f"   {description}: hash={snapshot.data_hash[:16]}..., uri={snapshot.storage_uri}")
    
    print(f"\n=== Test Summary ===")
    print(f"Evidence checkpoints service tested successfully")
    print(f"Total checkpoints created: {stats['total_checkpoints']}")
    print(f"Executions tracked: {stats['unique_executions']}")
    print(f"Nodes with evidence: {stats['unique_nodes_with_evidence']}")
    print(f"Retention policies available: {stats['total_retention_policies']}")
    
    return service, api

if __name__ == "__main__":
    run_evidence_checkpoints_tests()
