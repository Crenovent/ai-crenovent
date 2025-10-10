"""
Chapter 24.1: Version Lifecycle Service
Tasks 24.1.1-24.1.9: Implement versioning with lifecycle states, cryptographic hashing, and promotion workflows

This service manages the complete lifecycle of knowledge assets (templates, workflows, models, prompts, data contracts)
with immutable versioning, cryptographic integrity, and governed promotion workflows.
"""

import uuid
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass

from ai_crenovent.database.database_manager import DatabaseManager  # Assuming this exists
from ai_crenovent.dsl.governance.evidence_service import EvidenceService  # Assuming this exists
from ai_crenovent.dsl.tenancy.tenant_context_manager import TenantContextManager, TenantContext

logger = logging.getLogger(__name__)

class LifecycleState(Enum):
    """Task 24.1.2: Define lifecycle states for version management"""
    DRAFT = "draft"
    PUBLISHED = "published"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class PromotionResult(Enum):
    """Promotion workflow results"""
    SUCCESS = "success"
    FAILED_POLICY = "failed_policy"
    FAILED_EVIDENCE = "failed_evidence"
    FAILED_TESTS = "failed_tests"
    BLOCKED = "blocked"

@dataclass
class VersionMetadata:
    """Version metadata with lifecycle and hash information"""
    version_id: str
    lifecycle_state: LifecycleState
    content_hash: str
    parent_version_id: Optional[str] = None
    promotion_evidence_id: Optional[str] = None
    created_at: datetime = None
    created_by_user_id: int = None
    tenant_id: int = None

@dataclass
class PromotionRequest:
    """Request for version promotion with evidence and policy requirements"""
    version_id: str
    target_state: LifecycleState
    promotion_reason: str
    evidence_pack_id: Optional[str] = None
    policy_test_results: Dict[str, Any] = None
    requested_by_user_id: int = None
    tenant_id: int = None

class VersionLifecycleService:
    """
    Version Lifecycle Service for managing knowledge asset versions.
    Tasks 24.1.1-24.1.9: Complete version lifecycle management with governance.
    """
    
    def __init__(self, db_manager: DatabaseManager, evidence_service: EvidenceService, 
                 tenant_context_manager: TenantContextManager):
        self.db_manager = db_manager
        self.evidence_service = evidence_service
        self.tenant_context_manager = tenant_context_manager
        
        # Task 24.1.2: Define valid lifecycle transitions
        self.valid_transitions = {
            LifecycleState.DRAFT: [LifecycleState.PUBLISHED, LifecycleState.RETIRED],
            LifecycleState.PUBLISHED: [LifecycleState.PROMOTED, LifecycleState.DEPRECATED, LifecycleState.RETIRED],
            LifecycleState.PROMOTED: [LifecycleState.DEPRECATED, LifecycleState.RETIRED],
            LifecycleState.DEPRECATED: [LifecycleState.RETIRED, LifecycleState.PROMOTED],  # Can be un-deprecated
            LifecycleState.RETIRED: []  # Terminal state
        }

    async def create_version(self, version_id: str, content: Dict[str, Any], 
                           tenant_id: int, created_by_user_id: int,
                           parent_version_id: Optional[str] = None) -> VersionMetadata:
        """
        Task 24.1.1: Create a new version with lifecycle tracking.
        Task 24.1.3: Add cryptographic hashing for immutability.
        """
        # Generate content hash for immutability
        content_hash = self._generate_content_hash(content)
        
        # Create lifecycle entry
        lifecycle_id = str(uuid.uuid4())
        await self.db_manager.execute(
            """
            INSERT INTO cap_version_lifecycle 
            (lifecycle_id, version_id, lifecycle_state, parent_version_id, created_by_user_id, tenant_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (lifecycle_id, version_id, LifecycleState.DRAFT.value, parent_version_id, created_by_user_id, tenant_id)
        )
        
        # Create hash entry
        hash_id = str(uuid.uuid4())
        hash_metadata = {
            "algorithm": "SHA256",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_size": len(json.dumps(content)),
            "parent_hash": await self._get_parent_hash(parent_version_id) if parent_version_id else None
        }
        
        await self.db_manager.execute(
            """
            INSERT INTO cap_version_hash 
            (hash_id, version_id, content_hash, hash_algorithm, hash_metadata, tenant_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (hash_id, version_id, content_hash, "SHA256", json.dumps(hash_metadata), tenant_id)
        )
        
        # Log evidence pack for version creation
        await self.evidence_service.log_evidence(
            event_type="version_created",
            event_data={
                "version_id": version_id,
                "lifecycle_state": LifecycleState.DRAFT.value,
                "content_hash": content_hash,
                "parent_version_id": parent_version_id,
                "created_by_user_id": created_by_user_id
            },
            tenant_id=tenant_id,
            user_id=created_by_user_id
        )
        
        logger.info(f"Created version {version_id} with hash {content_hash} for tenant {tenant_id}")
        
        return VersionMetadata(
            version_id=version_id,
            lifecycle_state=LifecycleState.DRAFT,
            content_hash=content_hash,
            parent_version_id=parent_version_id,
            created_at=datetime.now(timezone.utc),
            created_by_user_id=created_by_user_id,
            tenant_id=tenant_id
        )

    async def promote_version(self, promotion_request: PromotionRequest) -> Tuple[PromotionResult, str]:
        """
        Task 24.1.4: Implement promotion workflows with evidence and policy tests.
        Task 24.1.7: Log version evidence packs for promotion decisions.
        """
        version_id = promotion_request.version_id
        target_state = promotion_request.target_state
        tenant_id = promotion_request.tenant_id
        
        # Get current version metadata
        current_metadata = await self.get_version_metadata(version_id, tenant_id)
        if not current_metadata:
            return PromotionResult.FAILED_POLICY, f"Version {version_id} not found"
        
        # Validate transition
        if target_state not in self.valid_transitions[current_metadata.lifecycle_state]:
            return PromotionResult.FAILED_POLICY, f"Invalid transition from {current_metadata.lifecycle_state.value} to {target_state.value}"
        
        # Task 24.1.4: Validate evidence and policy requirements
        validation_result = await self._validate_promotion_requirements(promotion_request)
        if validation_result != PromotionResult.SUCCESS:
            return validation_result, "Promotion validation failed"
        
        # Execute promotion
        try:
            # Update lifecycle state
            await self.db_manager.execute(
                """
                UPDATE cap_version_lifecycle 
                SET lifecycle_state = ?, updated_at = ?, promotion_evidence_id = ?
                WHERE version_id = ? AND tenant_id = ?
                """,
                (target_state.value, datetime.now(timezone.utc), 
                 promotion_request.evidence_pack_id, version_id, tenant_id)
            )
            
            # Task 24.1.7: Log evidence pack for promotion
            await self.evidence_service.log_evidence(
                event_type="version_promoted",
                event_data={
                    "version_id": version_id,
                    "from_state": current_metadata.lifecycle_state.value,
                    "to_state": target_state.value,
                    "promotion_reason": promotion_request.promotion_reason,
                    "evidence_pack_id": promotion_request.evidence_pack_id,
                    "policy_test_results": promotion_request.policy_test_results,
                    "requested_by_user_id": promotion_request.requested_by_user_id
                },
                tenant_id=tenant_id,
                user_id=promotion_request.requested_by_user_id
            )
            
            logger.info(f"Successfully promoted version {version_id} from {current_metadata.lifecycle_state.value} to {target_state.value}")
            return PromotionResult.SUCCESS, f"Version promoted to {target_state.value}"
            
        except Exception as e:
            logger.error(f"Failed to promote version {version_id}: {str(e)}")
            return PromotionResult.FAILED_POLICY, f"Promotion execution failed: {str(e)}"

    async def get_version_metadata(self, version_id: str, tenant_id: int) -> Optional[VersionMetadata]:
        """Get complete version metadata including lifecycle and hash information."""
        result = await self.db_manager.fetch_one(
            """
            SELECT l.version_id, l.lifecycle_state, l.parent_version_id, l.promotion_evidence_id,
                   l.created_at, l.created_by_user_id, l.tenant_id,
                   h.content_hash
            FROM cap_version_lifecycle l
            JOIN cap_version_hash h ON l.version_id = h.version_id
            WHERE l.version_id = ? AND l.tenant_id = ?
            """,
            (version_id, tenant_id)
        )
        
        if not result:
            return None
        
        return VersionMetadata(
            version_id=result['version_id'],
            lifecycle_state=LifecycleState(result['lifecycle_state']),
            content_hash=result['content_hash'],
            parent_version_id=result['parent_version_id'],
            promotion_evidence_id=result['promotion_evidence_id'],
            created_at=result['created_at'],
            created_by_user_id=result['created_by_user_id'],
            tenant_id=result['tenant_id']
        )

    async def get_version_history(self, version_id: str, tenant_id: int) -> List[VersionMetadata]:
        """
        Task 24.1.5: Track version history across tenants.
        Get complete version lineage starting from the given version.
        """
        versions = []
        current_version_id = version_id
        
        while current_version_id:
            metadata = await self.get_version_metadata(current_version_id, tenant_id)
            if not metadata:
                break
            
            versions.append(metadata)
            current_version_id = metadata.parent_version_id
        
        return versions

    async def get_active_versions_by_state(self, tenant_id: int, 
                                         lifecycle_state: LifecycleState) -> List[VersionMetadata]:
        """Get all versions in a specific lifecycle state for a tenant."""
        results = await self.db_manager.fetch_all(
            """
            SELECT l.version_id, l.lifecycle_state, l.parent_version_id, l.promotion_evidence_id,
                   l.created_at, l.created_by_user_id, l.tenant_id,
                   h.content_hash
            FROM cap_version_lifecycle l
            JOIN cap_version_hash h ON l.version_id = h.version_id
            WHERE l.lifecycle_state = ? AND l.tenant_id = ?
            ORDER BY l.created_at DESC
            """,
            (lifecycle_state.value, tenant_id)
        )
        
        return [
            VersionMetadata(
                version_id=row['version_id'],
                lifecycle_state=LifecycleState(row['lifecycle_state']),
                content_hash=row['content_hash'],
                parent_version_id=row['parent_version_id'],
                promotion_evidence_id=row['promotion_evidence_id'],
                created_at=row['created_at'],
                created_by_user_id=row['created_by_user_id'],
                tenant_id=row['tenant_id']
            )
            for row in results
        ]

    async def deprecate_version(self, version_id: str, tenant_id: int, 
                              deprecation_reason: str, user_id: int) -> bool:
        """
        Task 24.1.6: Enable deprecation with alerts.
        Deprecate a version and trigger alerts for tenants using it.
        """
        # Promote to deprecated state
        promotion_request = PromotionRequest(
            version_id=version_id,
            target_state=LifecycleState.DEPRECATED,
            promotion_reason=deprecation_reason,
            requested_by_user_id=user_id,
            tenant_id=tenant_id
        )
        
        result, message = await self.promote_version(promotion_request)
        if result != PromotionResult.SUCCESS:
            return False
        
        # Task 24.1.6: Trigger deprecation alerts (simulated)
        await self._trigger_deprecation_alerts(version_id, tenant_id, deprecation_reason)
        
        return True

    async def verify_version_integrity(self, version_id: str, content: Dict[str, Any], 
                                     tenant_id: int) -> bool:
        """
        Task 24.1.3: Verify version integrity using cryptographic hash.
        """
        stored_hash = await self.db_manager.fetch_one(
            "SELECT content_hash FROM cap_version_hash WHERE version_id = ? AND tenant_id = ?",
            (version_id, tenant_id)
        )
        
        if not stored_hash:
            return False
        
        computed_hash = self._generate_content_hash(content)
        is_valid = stored_hash['content_hash'] == computed_hash
        
        # Update verification status
        await self.db_manager.execute(
            "UPDATE cap_version_hash SET is_verified = ? WHERE version_id = ? AND tenant_id = ?",
            (is_valid, version_id, tenant_id)
        )
        
        # Log verification result
        await self.evidence_service.log_evidence(
            event_type="version_integrity_check",
            event_data={
                "version_id": version_id,
                "stored_hash": stored_hash['content_hash'],
                "computed_hash": computed_hash,
                "is_valid": is_valid
            },
            tenant_id=tenant_id
        )
        
        return is_valid

    def _generate_content_hash(self, content: Dict[str, Any]) -> str:
        """
        Task 24.1.3: Generate SHA256 hash of version content for immutability.
        """
        # Normalize content for consistent hashing
        normalized_content = json.dumps(content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()

    async def _get_parent_hash(self, parent_version_id: str) -> Optional[str]:
        """Get the content hash of a parent version."""
        if not parent_version_id:
            return None
        
        result = await self.db_manager.fetch_one(
            "SELECT content_hash FROM cap_version_hash WHERE version_id = ?",
            (parent_version_id,)
        )
        
        return result['content_hash'] if result else None

    async def _validate_promotion_requirements(self, promotion_request: PromotionRequest) -> PromotionResult:
        """
        Task 24.1.4: Validate evidence and policy requirements for promotion.
        """
        # Check if evidence pack is provided for non-draft promotions
        if (promotion_request.target_state != LifecycleState.DRAFT and 
            not promotion_request.evidence_pack_id):
            logger.warning(f"No evidence pack provided for promotion to {promotion_request.target_state.value}")
            return PromotionResult.FAILED_EVIDENCE
        
        # Validate policy test results for promoted state
        if (promotion_request.target_state == LifecycleState.PROMOTED and 
            not promotion_request.policy_test_results):
            logger.warning("No policy test results provided for promotion to promoted state")
            return PromotionResult.FAILED_TESTS
        
        # Check policy test results if provided
        if promotion_request.policy_test_results:
            failed_tests = [
                test_name for test_name, result in promotion_request.policy_test_results.items()
                if not result.get('passed', False)
            ]
            if failed_tests:
                logger.warning(f"Policy tests failed: {failed_tests}")
                return PromotionResult.FAILED_TESTS
        
        return PromotionResult.SUCCESS

    async def _trigger_deprecation_alerts(self, version_id: str, tenant_id: int, reason: str):
        """
        Task 24.1.6: Trigger alerts for version deprecation.
        In a real system, this would integrate with Prometheus/alerting systems.
        """
        # Log deprecation alert as evidence
        await self.evidence_service.log_evidence(
            event_type="version_deprecation_alert",
            event_data={
                "version_id": version_id,
                "deprecation_reason": reason,
                "alert_type": "version_deprecated",
                "severity": "warning"
            },
            tenant_id=tenant_id
        )
        
        logger.info(f"Triggered deprecation alert for version {version_id}: {reason}")

    async def get_version_statistics(self, tenant_id: int) -> Dict[str, Any]:
        """
        Task 24.1.9: Get version statistics for dashboards.
        """
        stats = {}
        
        for state in LifecycleState:
            count = await self.db_manager.fetch_one(
                "SELECT COUNT(*) as count FROM cap_version_lifecycle WHERE lifecycle_state = ? AND tenant_id = ?",
                (state.value, tenant_id)
            )
            stats[state.value] = count['count'] if count else 0
        
        # Get total versions
        total = await self.db_manager.fetch_one(
            "SELECT COUNT(*) as count FROM cap_version_lifecycle WHERE tenant_id = ?",
            (tenant_id,)
        )
        stats['total'] = total['count'] if total else 0
        
        # Get integrity verification stats
        verified = await self.db_manager.fetch_one(
            "SELECT COUNT(*) as count FROM cap_version_hash WHERE is_verified = TRUE AND tenant_id = ?",
            (tenant_id,)
        )
        stats['verified_count'] = verified['count'] if verified else 0
        
        return stats
