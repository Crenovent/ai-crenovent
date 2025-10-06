"""
Chapter 24.2: Rollback Service
Tasks 24.2.1-24.2.9: Implement safe rollback capabilities with automated triggers and governance

This service provides safe, governed rollback paths to the last stable version when newly promoted versions
introduce regressions or compliance risks.
"""

import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass

from ai_crenovent.database.database_manager import DatabaseManager  # Assuming this exists
from ai_crenovent.dsl.governance.evidence_service import EvidenceService  # Assuming this exists
from ai_crenovent.dsl.knowledge.version_lifecycle_service import VersionLifecycleService, LifecycleState

logger = logging.getLogger(__name__)

class RollbackTrigger(Enum):
    """Types of rollback triggers"""
    AUTOMATED_SLA = "automated_sla"
    AUTOMATED_DRIFT = "automated_drift"
    AUTOMATED_BIAS = "automated_bias"
    AUTOMATED_SECURITY = "automated_security"
    MANUAL_EMERGENCY = "manual_emergency"
    MANUAL_PLANNED = "manual_planned"

class RollbackStatus(Enum):
    """Rollback execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class RollbackRequest:
    """Request for version rollback with justification and targeting"""
    current_version_id: str
    rollback_target_id: str
    rollback_reason: str
    rollback_trigger: RollbackTrigger
    tenant_id: int
    executed_by_user_id: Optional[int] = None
    evidence_pack_id: Optional[str] = None
    impact_assessment: Dict[str, Any] = None

@dataclass
class RollbackManifest:
    """Signed rollback manifest with cryptographic integrity"""
    rollback_id: str
    current_version_id: str
    rollback_target_id: str
    rollback_reason: str
    rollback_trigger: RollbackTrigger
    tenant_id: int
    created_at: datetime
    executed_by_user_id: Optional[int]
    digital_signature: Optional[str] = None
    manifest_hash: Optional[str] = None

class RollbackService:
    """
    Rollback Service for safe version rollback with governance and automation.
    Tasks 24.2.1-24.2.9: Complete rollback management with automated triggers.
    """
    
    def __init__(self, db_manager: DatabaseManager, evidence_service: EvidenceService,
                 version_lifecycle_service: VersionLifecycleService):
        self.db_manager = db_manager
        self.evidence_service = evidence_service
        self.version_lifecycle_service = version_lifecycle_service
        
        # Task 24.2.4-24.2.6: Define automated trigger thresholds
        self.trigger_thresholds = {
            RollbackTrigger.AUTOMATED_SLA: {
                "latency_p95_ms": 5000,  # 5 second p95 latency
                "error_rate_percent": 5.0,  # 5% error rate
                "availability_percent": 99.0  # 99% availability
            },
            RollbackTrigger.AUTOMATED_DRIFT: {
                "accuracy_drop_percent": 10.0,  # 10% accuracy drop
                "drift_score_threshold": 0.3  # Drift score > 0.3
            },
            RollbackTrigger.AUTOMATED_BIAS: {
                "fairness_score_threshold": 0.8,  # Fairness score < 0.8
                "bias_violation_count": 3  # 3 or more bias violations
            },
            RollbackTrigger.AUTOMATED_SECURITY: {
                "critical_vulnerabilities": 1,  # Any critical vulnerability
                "high_vulnerabilities": 3  # 3 or more high vulnerabilities
            }
        }

    async def create_rollback_request(self, rollback_request: RollbackRequest) -> str:
        """
        Task 24.2.1: Create rollback request with target tracking.
        Task 24.2.7: Generate rollback manifest with reason.
        """
        rollback_id = str(uuid.uuid4())
        
        # Validate rollback target exists and is stable
        target_metadata = await self.version_lifecycle_service.get_version_metadata(
            rollback_request.rollback_target_id, rollback_request.tenant_id
        )
        
        if not target_metadata:
            raise ValueError(f"Rollback target {rollback_request.rollback_target_id} not found")
        
        if target_metadata.lifecycle_state not in [LifecycleState.PROMOTED, LifecycleState.PUBLISHED]:
            raise ValueError(f"Rollback target must be in promoted or published state, got {target_metadata.lifecycle_state}")
        
        # Create rollback entry
        await self.db_manager.execute(
            """
            INSERT INTO cap_version_rollback 
            (rollback_id, current_version_id, rollback_target_id, rollback_reason, 
             rollback_trigger, rollback_status, executed_by_user_id, tenant_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (rollback_id, rollback_request.current_version_id, rollback_request.rollback_target_id,
             rollback_request.rollback_reason, rollback_request.rollback_trigger.value,
             RollbackStatus.PENDING.value, rollback_request.executed_by_user_id, rollback_request.tenant_id)
        )
        
        # Task 24.2.7: Generate rollback manifest
        manifest = await self._generate_rollback_manifest(rollback_id, rollback_request)
        
        # Task 24.2.8: Log evidence pack for rollback request
        await self.evidence_service.log_evidence(
            event_type="rollback_requested",
            event_data={
                "rollback_id": rollback_id,
                "current_version_id": rollback_request.current_version_id,
                "rollback_target_id": rollback_request.rollback_target_id,
                "rollback_reason": rollback_request.rollback_reason,
                "rollback_trigger": rollback_request.rollback_trigger.value,
                "impact_assessment": rollback_request.impact_assessment,
                "manifest_hash": manifest.manifest_hash
            },
            tenant_id=rollback_request.tenant_id,
            user_id=rollback_request.executed_by_user_id
        )
        
        logger.info(f"Created rollback request {rollback_id} for version {rollback_request.current_version_id}")
        return rollback_id

    async def execute_rollback(self, rollback_id: str, tenant_id: int) -> bool:
        """
        Task 24.2.2: Implement rollback protocol with orchestrator resolution.
        Task 24.2.3: Support tenant-pinned rollbacks.
        """
        # Get rollback details
        rollback_data = await self.db_manager.fetch_one(
            """
            SELECT rollback_id, current_version_id, rollback_target_id, rollback_reason,
                   rollback_trigger, executed_by_user_id, tenant_id
            FROM cap_version_rollback 
            WHERE rollback_id = ? AND tenant_id = ? AND rollback_status = ?
            """,
            (rollback_id, tenant_id, RollbackStatus.PENDING.value)
        )
        
        if not rollback_data:
            logger.error(f"Rollback {rollback_id} not found or not in pending state")
            return False
        
        try:
            # Update status to in_progress
            await self.db_manager.execute(
                "UPDATE cap_version_rollback SET rollback_status = ? WHERE rollback_id = ?",
                (RollbackStatus.IN_PROGRESS.value, rollback_id)
            )
            
            # Task 24.2.3: Check if tenant has version pinning
            await self._handle_tenant_pinned_rollback(rollback_data, tenant_id)
            
            # Task 24.2.2: Execute orchestrator-level rollback
            # This would integrate with the orchestrator to switch active versions
            await self._execute_orchestrator_rollback(rollback_data)
            
            # Update status to completed
            await self.db_manager.execute(
                """
                UPDATE cap_version_rollback 
                SET rollback_status = ?, executed_at = ? 
                WHERE rollback_id = ?
                """,
                (RollbackStatus.COMPLETED.value, datetime.now(timezone.utc), rollback_id)
            )
            
            # Task 24.2.8: Log evidence pack for rollback execution
            await self.evidence_service.log_evidence(
                event_type="rollback_executed",
                event_data={
                    "rollback_id": rollback_id,
                    "current_version_id": rollback_data['current_version_id'],
                    "rollback_target_id": rollback_data['rollback_target_id'],
                    "rollback_trigger": rollback_data['rollback_trigger'],
                    "executed_at": datetime.now(timezone.utc).isoformat()
                },
                tenant_id=tenant_id,
                user_id=rollback_data['executed_by_user_id']
            )
            
            logger.info(f"Successfully executed rollback {rollback_id}")
            return True
            
        except Exception as e:
            # Update status to failed
            await self.db_manager.execute(
                "UPDATE cap_version_rollback SET rollback_status = ? WHERE rollback_id = ?",
                (RollbackStatus.FAILED.value, rollback_id)
            )
            
            # Log failure evidence
            await self.evidence_service.log_evidence(
                event_type="rollback_failed",
                event_data={
                    "rollback_id": rollback_id,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                tenant_id=tenant_id,
                user_id=rollback_data['executed_by_user_id']
            )
            
            logger.error(f"Failed to execute rollback {rollback_id}: {str(e)}")
            return False

    async def check_sla_breach_trigger(self, version_id: str, tenant_id: int, 
                                     metrics: Dict[str, float]) -> Optional[str]:
        """
        Task 24.2.4: SLA breach trigger for automated rollback.
        """
        thresholds = self.trigger_thresholds[RollbackTrigger.AUTOMATED_SLA]
        violations = []
        
        # Check latency threshold
        if metrics.get('latency_p95_ms', 0) > thresholds['latency_p95_ms']:
            violations.append(f"Latency P95 {metrics['latency_p95_ms']}ms > {thresholds['latency_p95_ms']}ms")
        
        # Check error rate threshold
        if metrics.get('error_rate_percent', 0) > thresholds['error_rate_percent']:
            violations.append(f"Error rate {metrics['error_rate_percent']}% > {thresholds['error_rate_percent']}%")
        
        # Check availability threshold
        if metrics.get('availability_percent', 100) < thresholds['availability_percent']:
            violations.append(f"Availability {metrics['availability_percent']}% < {thresholds['availability_percent']}%")
        
        if violations:
            # Find last stable version for rollback target
            rollback_target = await self._find_last_stable_version(version_id, tenant_id)
            if rollback_target:
                rollback_request = RollbackRequest(
                    current_version_id=version_id,
                    rollback_target_id=rollback_target,
                    rollback_reason=f"SLA breach: {'; '.join(violations)}",
                    rollback_trigger=RollbackTrigger.AUTOMATED_SLA,
                    tenant_id=tenant_id,
                    impact_assessment={"sla_violations": violations, "metrics": metrics}
                )
                
                return await self.create_rollback_request(rollback_request)
        
        return None

    async def check_drift_bias_trigger(self, version_id: str, tenant_id: int,
                                     drift_metrics: Dict[str, float],
                                     bias_metrics: Dict[str, float]) -> Optional[str]:
        """
        Task 24.2.5: Drift/bias trigger for automated rollback.
        """
        violations = []
        trigger_type = None
        
        # Check drift thresholds
        drift_thresholds = self.trigger_thresholds[RollbackTrigger.AUTOMATED_DRIFT]
        if drift_metrics.get('accuracy_drop_percent', 0) > drift_thresholds['accuracy_drop_percent']:
            violations.append(f"Accuracy drop {drift_metrics['accuracy_drop_percent']}% > {drift_thresholds['accuracy_drop_percent']}%")
            trigger_type = RollbackTrigger.AUTOMATED_DRIFT
        
        if drift_metrics.get('drift_score', 0) > drift_thresholds['drift_score_threshold']:
            violations.append(f"Drift score {drift_metrics['drift_score']} > {drift_thresholds['drift_score_threshold']}")
            trigger_type = RollbackTrigger.AUTOMATED_DRIFT
        
        # Check bias thresholds
        bias_thresholds = self.trigger_thresholds[RollbackTrigger.AUTOMATED_BIAS]
        if bias_metrics.get('fairness_score', 1.0) < bias_thresholds['fairness_score_threshold']:
            violations.append(f"Fairness score {bias_metrics['fairness_score']} < {bias_thresholds['fairness_score_threshold']}")
            trigger_type = RollbackTrigger.AUTOMATED_BIAS
        
        if bias_metrics.get('bias_violation_count', 0) >= bias_thresholds['bias_violation_count']:
            violations.append(f"Bias violations {bias_metrics['bias_violation_count']} >= {bias_thresholds['bias_violation_count']}")
            trigger_type = RollbackTrigger.AUTOMATED_BIAS
        
        if violations and trigger_type:
            rollback_target = await self._find_last_stable_version(version_id, tenant_id)
            if rollback_target:
                rollback_request = RollbackRequest(
                    current_version_id=version_id,
                    rollback_target_id=rollback_target,
                    rollback_reason=f"Model quality degradation: {'; '.join(violations)}",
                    rollback_trigger=trigger_type,
                    tenant_id=tenant_id,
                    impact_assessment={
                        "violations": violations,
                        "drift_metrics": drift_metrics,
                        "bias_metrics": bias_metrics
                    }
                )
                
                return await self.create_rollback_request(rollback_request)
        
        return None

    async def check_security_compliance_trigger(self, version_id: str, tenant_id: int,
                                              security_results: Dict[str, Any]) -> Optional[str]:
        """
        Task 24.2.6: Security/compliance trigger for automated rollback.
        """
        thresholds = self.trigger_thresholds[RollbackTrigger.AUTOMATED_SECURITY]
        violations = []
        
        # Check critical vulnerabilities
        critical_vulns = security_results.get('critical_vulnerabilities', 0)
        if critical_vulns >= thresholds['critical_vulnerabilities']:
            violations.append(f"Critical vulnerabilities: {critical_vulns}")
        
        # Check high vulnerabilities
        high_vulns = security_results.get('high_vulnerabilities', 0)
        if high_vulns >= thresholds['high_vulnerabilities']:
            violations.append(f"High vulnerabilities: {high_vulns}")
        
        # Check compliance failures
        compliance_failures = security_results.get('compliance_failures', [])
        if compliance_failures:
            violations.append(f"Compliance failures: {', '.join(compliance_failures)}")
        
        if violations:
            rollback_target = await self._find_last_stable_version(version_id, tenant_id)
            if rollback_target:
                rollback_request = RollbackRequest(
                    current_version_id=version_id,
                    rollback_target_id=rollback_target,
                    rollback_reason=f"Security/compliance violations: {'; '.join(violations)}",
                    rollback_trigger=RollbackTrigger.AUTOMATED_SECURITY,
                    tenant_id=tenant_id,
                    impact_assessment={
                        "security_violations": violations,
                        "security_results": security_results
                    }
                )
                
                return await self.create_rollback_request(rollback_request)
        
        return None

    async def get_rollback_history(self, tenant_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Task 24.2.9: Get rollback history for dashboards.
        """
        results = await self.db_manager.fetch_all(
            """
            SELECT rollback_id, current_version_id, rollback_target_id, rollback_reason,
                   rollback_trigger, rollback_status, created_at, executed_at, executed_by_user_id
            FROM cap_version_rollback 
            WHERE tenant_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (tenant_id, limit)
        )
        
        return [dict(row) for row in results]

    async def _generate_rollback_manifest(self, rollback_id: str, 
                                        rollback_request: RollbackRequest) -> RollbackManifest:
        """
        Task 24.2.7: Generate signed rollback manifest.
        """
        manifest = RollbackManifest(
            rollback_id=rollback_id,
            current_version_id=rollback_request.current_version_id,
            rollback_target_id=rollback_request.rollback_target_id,
            rollback_reason=rollback_request.rollback_reason,
            rollback_trigger=rollback_request.rollback_trigger,
            tenant_id=rollback_request.tenant_id,
            created_at=datetime.now(timezone.utc),
            executed_by_user_id=rollback_request.executed_by_user_id
        )
        
        # Generate manifest hash for integrity
        manifest_content = {
            "rollback_id": manifest.rollback_id,
            "current_version_id": manifest.current_version_id,
            "rollback_target_id": manifest.rollback_target_id,
            "rollback_reason": manifest.rollback_reason,
            "rollback_trigger": manifest.rollback_trigger.value,
            "tenant_id": manifest.tenant_id,
            "created_at": manifest.created_at.isoformat()
        }
        
        import hashlib
        manifest.manifest_hash = hashlib.sha256(
            json.dumps(manifest_content, sort_keys=True).encode('utf-8')
        ).hexdigest()
        
        # In a real system, this would be cryptographically signed
        manifest.digital_signature = f"SIMULATED_SIGNATURE_{manifest.manifest_hash[:16]}"
        
        return manifest

    async def _handle_tenant_pinned_rollback(self, rollback_data: Dict[str, Any], tenant_id: int):
        """
        Task 24.2.3: Handle tenant-specific version pinning during rollback.
        """
        # Check if tenant has version pinning for this capability
        # This would integrate with the capability registry to find the capability_id
        # For now, we'll create a pin entry for the rollback target
        
        pin_id = str(uuid.uuid4())
        await self.db_manager.execute(
            """
            INSERT INTO cap_version_tenant_pins 
            (pin_id, tenant_id, capability_id, pinned_version_id, pin_reason, created_by_user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pin_id, tenant_id, rollback_data['current_version_id'],  # Using current as capability reference
             rollback_data['rollback_target_id'], 
             f"Rollback pin: {rollback_data['rollback_reason']}", 
             rollback_data['executed_by_user_id'])
        )
        
        logger.info(f"Created tenant pin {pin_id} for rollback target {rollback_data['rollback_target_id']}")

    async def _execute_orchestrator_rollback(self, rollback_data: Dict[str, Any]):
        """
        Task 24.2.2: Execute orchestrator-level rollback resolution.
        This would integrate with the routing orchestrator to switch active versions.
        """
        # Simulate orchestrator integration
        logger.info(f"Executing orchestrator rollback from {rollback_data['current_version_id']} to {rollback_data['rollback_target_id']}")
        
        # In a real system, this would:
        # 1. Update the orchestrator's active version registry
        # 2. Drain existing requests using the current version
        # 3. Route new requests to the rollback target version
        # 4. Verify the rollback was successful
        
        # For now, we'll just log the action
        await self.evidence_service.log_evidence(
            event_type="orchestrator_rollback_executed",
            event_data={
                "from_version": rollback_data['current_version_id'],
                "to_version": rollback_data['rollback_target_id'],
                "rollback_trigger": rollback_data['rollback_trigger']
            },
            tenant_id=rollback_data['tenant_id'],
            user_id=rollback_data['executed_by_user_id']
        )

    async def _find_last_stable_version(self, current_version_id: str, tenant_id: int) -> Optional[str]:
        """Find the last stable (promoted) version to use as rollback target."""
        # Get version history and find the last promoted version
        history = await self.version_lifecycle_service.get_version_history(current_version_id, tenant_id)
        
        for version_metadata in history:
            if (version_metadata.lifecycle_state == LifecycleState.PROMOTED and 
                version_metadata.version_id != current_version_id):
                return version_metadata.version_id
        
        # If no promoted version found, look for published versions
        for version_metadata in history:
            if (version_metadata.lifecycle_state == LifecycleState.PUBLISHED and 
                version_metadata.version_id != current_version_id):
                return version_metadata.version_id
        
        return None

    async def get_rollback_statistics(self, tenant_id: int) -> Dict[str, Any]:
        """Get rollback statistics for monitoring and dashboards."""
        stats = {}
        
        # Count rollbacks by trigger type
        for trigger in RollbackTrigger:
            count = await self.db_manager.fetch_one(
                "SELECT COUNT(*) as count FROM cap_version_rollback WHERE rollback_trigger = ? AND tenant_id = ?",
                (trigger.value, tenant_id)
            )
            stats[f"rollbacks_{trigger.value}"] = count['count'] if count else 0
        
        # Count rollbacks by status
        for status in RollbackStatus:
            count = await self.db_manager.fetch_one(
                "SELECT COUNT(*) as count FROM cap_version_rollback WHERE rollback_status = ? AND tenant_id = ?",
                (status.value, tenant_id)
            )
            stats[f"rollbacks_{status.value}"] = count['count'] if count else 0
        
        # Get success rate
        total_rollbacks = sum(stats[f"rollbacks_{status.value}"] for status in RollbackStatus)
        completed_rollbacks = stats.get('rollbacks_completed', 0)
        stats['success_rate_percent'] = (completed_rollbacks / total_rollbacks * 100) if total_rollbacks > 0 else 100
        
        return stats
