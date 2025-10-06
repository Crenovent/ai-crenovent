"""
Chapter 24: Knowledge & IP Lifecycle API
Tasks 24.1.7-24.1.9, 24.2.8-24.2.9, 24.3.8-24.3.9: Comprehensive API for version lifecycle, rollback, and anonymization

This API provides endpoints for managing the complete lifecycle of knowledge assets with versioning,
rollback capabilities, and privacy-preserving anonymization.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel, Field

from ai_crenovent.database.database_manager import DatabaseManager
from ai_crenovent.dsl.governance.evidence_service import EvidenceService
from ai_crenovent.dsl.tenancy.tenant_context_manager import TenantContextManager
from ai_crenovent.dsl.knowledge.version_lifecycle_service import (
    VersionLifecycleService, LifecycleState, PromotionRequest, PromotionResult
)
from ai_crenovent.dsl.knowledge.rollback_service import (
    RollbackService, RollbackRequest, RollbackTrigger, RollbackStatus
)
from ai_crenovent.dsl.knowledge.anonymization_service import (
    AnonymizationService, ArtifactType, AnonymizationStandard
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class CreateVersionRequest(BaseModel):
    version_id: str
    content: Dict[str, Any]
    parent_version_id: Optional[str] = None

class PromoteVersionRequest(BaseModel):
    version_id: str
    target_state: str
    promotion_reason: str
    evidence_pack_id: Optional[str] = None
    policy_test_results: Optional[Dict[str, Any]] = None

class CreateRollbackRequest(BaseModel):
    current_version_id: str
    rollback_target_id: str
    rollback_reason: str
    rollback_trigger: str
    impact_assessment: Optional[Dict[str, Any]] = None

class AnonymizeDatasetRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    industry_code: str
    source_tenant_count: int = 1

class ScrubModelOutputRequest(BaseModel):
    model_output: str
    industry_code: str

class AnonymizeTraceRequest(BaseModel):
    trace_data: Dict[str, Any]
    industry_code: str

# Dependency injection
def get_db_manager() -> DatabaseManager:
    return DatabaseManager()

def get_evidence_service(db: DatabaseManager = Depends(get_db_manager)) -> EvidenceService:
    return EvidenceService(db)

def get_tenant_context_manager(db: DatabaseManager = Depends(get_db_manager)) -> TenantContextManager:
    return TenantContextManager(db)

def get_version_lifecycle_service(
    db: DatabaseManager = Depends(get_db_manager),
    evidence: EvidenceService = Depends(get_evidence_service),
    tenant_context: TenantContextManager = Depends(get_tenant_context_manager)
) -> VersionLifecycleService:
    return VersionLifecycleService(db, evidence, tenant_context)

def get_rollback_service(
    db: DatabaseManager = Depends(get_db_manager),
    evidence: EvidenceService = Depends(get_evidence_service),
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
) -> RollbackService:
    return RollbackService(db, evidence, version_service)

def get_anonymization_service(
    db: DatabaseManager = Depends(get_db_manager),
    evidence: EvidenceService = Depends(get_evidence_service),
    tenant_context: TenantContextManager = Depends(get_tenant_context_manager)
) -> AnonymizationService:
    return AnonymizationService(db, evidence, tenant_context)

# ===============================================================================
# Section 24.1: Version Lifecycle APIs
# Tasks 24.1.7-24.1.9: Version management with evidence packs and dashboards
# ===============================================================================

@router.post("/versions", summary="Create a new version", status_code=status.HTTP_201_CREATED)
async def create_version(
    request: CreateVersionRequest,
    tenant_id: int,
    user_id: int,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
):
    """Task 24.1.1: Create version with lifecycle tracking and cryptographic hashing."""
    try:
        metadata = await version_service.create_version(
            version_id=request.version_id,
            content=request.content,
            tenant_id=tenant_id,
            created_by_user_id=user_id,
            parent_version_id=request.parent_version_id
        )
        
        return {
            "message": f"Version {request.version_id} created successfully",
            "version_metadata": {
                "version_id": metadata.version_id,
                "lifecycle_state": metadata.lifecycle_state.value,
                "content_hash": metadata.content_hash,
                "parent_version_id": metadata.parent_version_id,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/versions/{version_id}/promote", summary="Promote version to next lifecycle state")
async def promote_version(
    version_id: str,
    request: PromoteVersionRequest,
    tenant_id: int,
    user_id: int,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
):
    """Task 24.1.4: Implement promotion workflows with evidence and policy validation."""
    try:
        # Validate target state
        try:
            target_state = LifecycleState(request.target_state)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid lifecycle state: {request.target_state}"
            )
        
        promotion_request = PromotionRequest(
            version_id=version_id,
            target_state=target_state,
            promotion_reason=request.promotion_reason,
            evidence_pack_id=request.evidence_pack_id,
            policy_test_results=request.policy_test_results,
            requested_by_user_id=user_id,
            tenant_id=tenant_id
        )
        
        result, message = await version_service.promote_version(promotion_request)
        
        if result == PromotionResult.SUCCESS:
            return {"message": message, "promotion_result": result.value}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Promotion failed: {message}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/versions/{version_id}", summary="Get version metadata")
async def get_version_metadata(
    version_id: str,
    tenant_id: int,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
):
    """Get complete version metadata including lifecycle and hash information."""
    metadata = await version_service.get_version_metadata(version_id, tenant_id)
    
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version_id} not found"
        )
    
    return {
        "version_id": metadata.version_id,
        "lifecycle_state": metadata.lifecycle_state.value,
        "content_hash": metadata.content_hash,
        "parent_version_id": metadata.parent_version_id,
        "promotion_evidence_id": metadata.promotion_evidence_id,
        "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
        "created_by_user_id": metadata.created_by_user_id,
        "tenant_id": metadata.tenant_id
    }

@router.get("/versions/{version_id}/history", summary="Get version history")
async def get_version_history(
    version_id: str,
    tenant_id: int,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
):
    """Task 24.1.5: Track version history across tenants."""
    history = await version_service.get_version_history(version_id, tenant_id)
    
    return {
        "version_id": version_id,
        "history": [
            {
                "version_id": v.version_id,
                "lifecycle_state": v.lifecycle_state.value,
                "content_hash": v.content_hash,
                "parent_version_id": v.parent_version_id,
                "created_at": v.created_at.isoformat() if v.created_at else None
            }
            for v in history
        ]
    }

@router.post("/versions/{version_id}/deprecate", summary="Deprecate a version")
async def deprecate_version(
    version_id: str,
    deprecation_reason: str,
    tenant_id: int,
    user_id: int,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
):
    """Task 24.1.6: Enable deprecation with alerts."""
    success = await version_service.deprecate_version(
        version_id, tenant_id, deprecation_reason, user_id
    )
    
    if success:
        return {"message": f"Version {version_id} deprecated successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to deprecate version"
        )

@router.post("/versions/{version_id}/verify", summary="Verify version integrity")
async def verify_version_integrity(
    version_id: str,
    content: Dict[str, Any],
    tenant_id: int,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
):
    """Task 24.1.3: Verify version integrity using cryptographic hash."""
    is_valid = await version_service.verify_version_integrity(version_id, content, tenant_id)
    
    return {
        "version_id": version_id,
        "integrity_valid": is_valid,
        "message": "Version integrity verified" if is_valid else "Version integrity check failed"
    }

@router.get("/versions/statistics", summary="Get version statistics")
async def get_version_statistics(
    tenant_id: int,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service)
):
    """Task 24.1.9: Get version statistics for dashboards."""
    stats = await version_service.get_version_statistics(tenant_id)
    return {"tenant_id": tenant_id, "statistics": stats}

# ===============================================================================
# Section 24.2: Rollback APIs
# Tasks 24.2.8-24.2.9: Rollback management with evidence packs and dashboards
# ===============================================================================

@router.post("/rollbacks", summary="Create rollback request", status_code=status.HTTP_201_CREATED)
async def create_rollback_request(
    request: CreateRollbackRequest,
    tenant_id: int,
    user_id: int,
    rollback_service: RollbackService = Depends(get_rollback_service)
):
    """Task 24.2.1: Create rollback request with target tracking."""
    try:
        # Validate rollback trigger
        try:
            trigger = RollbackTrigger(request.rollback_trigger)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid rollback trigger: {request.rollback_trigger}"
            )
        
        rollback_request = RollbackRequest(
            current_version_id=request.current_version_id,
            rollback_target_id=request.rollback_target_id,
            rollback_reason=request.rollback_reason,
            rollback_trigger=trigger,
            tenant_id=tenant_id,
            executed_by_user_id=user_id,
            impact_assessment=request.impact_assessment
        )
        
        rollback_id = await rollback_service.create_rollback_request(rollback_request)
        
        return {
            "message": "Rollback request created successfully",
            "rollback_id": rollback_id
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/rollbacks/{rollback_id}/execute", summary="Execute rollback")
async def execute_rollback(
    rollback_id: str,
    tenant_id: int,
    background_tasks: BackgroundTasks,
    rollback_service: RollbackService = Depends(get_rollback_service)
):
    """Task 24.2.2: Execute rollback protocol with orchestrator resolution."""
    try:
        # Execute rollback in background for better responsiveness
        background_tasks.add_task(rollback_service.execute_rollback, rollback_id, tenant_id)
        
        return {
            "message": f"Rollback {rollback_id} execution started",
            "rollback_id": rollback_id,
            "status": "in_progress"
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/rollbacks/check-sla-trigger", summary="Check SLA breach trigger")
async def check_sla_breach_trigger(
    version_id: str,
    tenant_id: int,
    metrics: Dict[str, float],
    rollback_service: RollbackService = Depends(get_rollback_service)
):
    """Task 24.2.4: SLA breach trigger for automated rollback."""
    rollback_id = await rollback_service.check_sla_breach_trigger(version_id, tenant_id, metrics)
    
    if rollback_id:
        return {
            "message": "SLA breach detected, rollback request created",
            "rollback_id": rollback_id,
            "trigger": "automated_sla"
        }
    else:
        return {
            "message": "No SLA breach detected",
            "rollback_id": None
        }

@router.post("/rollbacks/check-drift-bias-trigger", summary="Check drift/bias trigger")
async def check_drift_bias_trigger(
    version_id: str,
    tenant_id: int,
    drift_metrics: Dict[str, float],
    bias_metrics: Dict[str, float],
    rollback_service: RollbackService = Depends(get_rollback_service)
):
    """Task 24.2.5: Drift/bias trigger for automated rollback."""
    rollback_id = await rollback_service.check_drift_bias_trigger(
        version_id, tenant_id, drift_metrics, bias_metrics
    )
    
    if rollback_id:
        return {
            "message": "Model quality degradation detected, rollback request created",
            "rollback_id": rollback_id,
            "trigger": "automated_drift_or_bias"
        }
    else:
        return {
            "message": "No model quality issues detected",
            "rollback_id": None
        }

@router.post("/rollbacks/check-security-trigger", summary="Check security/compliance trigger")
async def check_security_compliance_trigger(
    version_id: str,
    tenant_id: int,
    security_results: Dict[str, Any],
    rollback_service: RollbackService = Depends(get_rollback_service)
):
    """Task 24.2.6: Security/compliance trigger for automated rollback."""
    rollback_id = await rollback_service.check_security_compliance_trigger(
        version_id, tenant_id, security_results
    )
    
    if rollback_id:
        return {
            "message": "Security/compliance violations detected, rollback request created",
            "rollback_id": rollback_id,
            "trigger": "automated_security"
        }
    else:
        return {
            "message": "No security/compliance issues detected",
            "rollback_id": None
        }

@router.get("/rollbacks/history", summary="Get rollback history")
async def get_rollback_history(
    tenant_id: int,
    limit: int = 50,
    rollback_service: RollbackService = Depends(get_rollback_service)
):
    """Task 24.2.9: Get rollback history for dashboards."""
    history = await rollback_service.get_rollback_history(tenant_id, limit)
    return {"tenant_id": tenant_id, "rollback_history": history}

@router.get("/rollbacks/statistics", summary="Get rollback statistics")
async def get_rollback_statistics(
    tenant_id: int,
    rollback_service: RollbackService = Depends(get_rollback_service)
):
    """Get rollback statistics for monitoring and dashboards."""
    stats = await rollback_service.get_rollback_statistics(tenant_id)
    return {"tenant_id": tenant_id, "statistics": stats}

# ===============================================================================
# Section 24.3: Anonymization APIs
# Tasks 24.3.8-24.3.9: Anonymization with evidence packs and dashboards
# ===============================================================================

@router.post("/anonymization/dataset", summary="Anonymize dataset", status_code=status.HTTP_201_CREATED)
async def anonymize_dataset(
    request: AnonymizeDatasetRequest,
    anonymization_service: AnonymizationService = Depends(get_anonymization_service)
):
    """Task 24.3.4: Implement data transformation with privacy metrics."""
    try:
        result = await anonymization_service.anonymize_dataset(
            dataset=request.dataset,
            industry_code=request.industry_code,
            source_tenant_count=request.source_tenant_count
        )
        
        return {
            "message": "Dataset anonymized successfully",
            "artifact_id": result.artifact_id,
            "privacy_metrics": result.privacy_metrics,
            "is_cross_tenant_safe": result.is_cross_tenant_safe,
            "anonymization_policy_id": result.anonymization_policy_id
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/anonymization/trace", summary="Anonymize OpenTelemetry trace")
async def anonymize_trace(
    request: AnonymizeTraceRequest,
    anonymization_service: AnonymizationService = Depends(get_anonymization_service)
):
    """Task 24.3.5: Create trace anonymizer for OpenTelemetry traces."""
    try:
        anonymized_trace_id = await anonymization_service.anonymize_trace(
            trace_data=request.trace_data,
            industry_code=request.industry_code
        )
        
        return {
            "message": "Trace anonymized successfully",
            "original_trace_id": request.trace_data.get('trace_id'),
            "anonymized_trace_id": anonymized_trace_id
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/anonymization/model-output", summary="Scrub model output")
async def scrub_model_output(
    request: ScrubModelOutputRequest,
    anonymization_service: AnonymizationService = Depends(get_anonymization_service)
):
    """Task 24.3.6: Add model output scrubber for AI outputs."""
    try:
        scrubbed_output = await anonymization_service.scrub_model_output(
            model_output=request.model_output,
            industry_code=request.industry_code
        )
        
        return {
            "message": "Model output scrubbed successfully",
            "original_length": len(request.model_output),
            "scrubbed_length": len(scrubbed_output),
            "scrubbed_output": scrubbed_output
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/anonymization/policies/{industry_code}", summary="Get anonymization policy")
async def get_anonymization_policy(
    industry_code: str,
    anonymization_service: AnonymizationService = Depends(get_anonymization_service)
):
    """Task 24.3.1: Get anonymization policy for industry."""
    policy = await anonymization_service.get_anonymization_policy(industry_code)
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No anonymization policy found for industry: {industry_code}"
        )
    
    return {
        "policy_id": policy.policy_id,
        "policy_name": policy.policy_name,
        "industry_code": policy.industry_code,
        "anonymization_standard": policy.anonymization_standard.value,
        "k_anonymity_threshold": policy.k_anonymity_threshold,
        "l_diversity_threshold": policy.l_diversity_threshold,
        "t_closeness_threshold": policy.t_closeness_threshold,
        "sensitive_attributes": policy.sensitive_attributes,
        "quasi_identifiers": policy.quasi_identifiers
    }

@router.get("/anonymization/field-mappings/{policy_id}", summary="Get field mappings")
async def get_field_mappings(
    policy_id: str,
    anonymization_service: AnonymizationService = Depends(get_anonymization_service)
):
    """Task 24.3.2: Get field mappings for sensitive data transformation."""
    mappings = await anonymization_service.get_field_mappings(policy_id)
    
    return {
        "policy_id": policy_id,
        "field_mappings": [
            {
                "field_name": fm.field_name,
                "field_type": fm.field_type.value,
                "sensitivity_level": fm.sensitivity_level,
                "transformation_method": fm.transformation_method.value,
                "transformation_params": fm.transformation_params,
                "compliance_frameworks": fm.compliance_frameworks
            }
            for fm in mappings
        ]
    }

@router.get("/anonymization/statistics", summary="Get anonymization statistics")
async def get_anonymization_statistics(
    industry_code: Optional[str] = None,
    anonymization_service: AnonymizationService = Depends(get_anonymization_service)
):
    """Task 24.3.9: Get anonymization statistics for dashboards."""
    stats = await anonymization_service.get_anonymization_statistics(industry_code)
    return {
        "industry_code": industry_code,
        "statistics": stats
    }

# ===============================================================================
# Combined Dashboard APIs
# Tasks 24.1.9, 24.2.9, 24.3.9: Comprehensive dashboard data
# ===============================================================================

@router.get("/dashboard/overview", summary="Get comprehensive lifecycle overview")
async def get_lifecycle_overview(
    tenant_id: int,
    industry_code: Optional[str] = None,
    version_service: VersionLifecycleService = Depends(get_version_lifecycle_service),
    rollback_service: RollbackService = Depends(get_rollback_service),
    anonymization_service: AnonymizationService = Depends(get_anonymization_service)
):
    """Get comprehensive overview for Knowledge & IP Lifecycle dashboards."""
    try:
        # Get all statistics in parallel
        version_stats = await version_service.get_version_statistics(tenant_id)
        rollback_stats = await rollback_service.get_rollback_statistics(tenant_id)
        anonymization_stats = await anonymization_service.get_anonymization_statistics(industry_code)
        
        return {
            "tenant_id": tenant_id,
            "industry_code": industry_code,
            "overview": {
                "versioning": version_stats,
                "rollbacks": rollback_stats,
                "anonymization": anonymization_stats
            },
            "summary": {
                "total_versions": version_stats.get('total', 0),
                "active_versions": version_stats.get('promoted', 0) + version_stats.get('published', 0),
                "rollback_success_rate": rollback_stats.get('success_rate_percent', 100),
                "cross_tenant_safe_artifacts": anonymization_stats.get('cross_tenant_safe_count', 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
