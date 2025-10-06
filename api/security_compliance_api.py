# Security & Compliance API
# Integrates RBAC/ABAC, PII Handling, Residency Enforcement, and Penetration Testing

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
import uuid
import logging

from ..dsl.security.rbac_abac_middleware import (
    SecurityMiddlewareManager, AccessResult, AccessContext, require_access_control
)
from ..dsl.security.pii_handling_service import (
    PIIHandlingManager, PIIClassification, MaskingRule, AnonymizationConfig
)
from ..dsl.security.residency_enforcement_service import (
    ResidencyComplianceManager, ResidencyRequest, ResidencyResult, Region, DataClassification
)
from ..dsl.security.penetration_testing_service import (
    PenTestOrchestrator, PenTestConfig, TestScope, TestType, AttackVector
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/security", tags=["Security & Compliance"])
security = HTTPBearer()

# Initialize services
security_middleware = SecurityMiddlewareManager()
pii_manager = PIIHandlingManager()
residency_manager = ResidencyComplianceManager()
pentest_orchestrator = PenTestOrchestrator()

# ================================
# RBAC/ABAC Endpoints
# ================================

@router.get("/access/validate")
async def validate_access(
    request: Request,
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Validate access control for current request
    Task 18.1.8-18.1.9: RBAC/ABAC enforcement
    """
    return {
        "access_granted": access_result.decision.value == "allow",
        "decision": access_result.decision.value,
        "reason": access_result.reason,
        "user_id": access_result.context.user_id,
        "tenant_id": access_result.context.tenant_id,
        "permissions": list(access_result.context.permissions),
        "applicable_policies": access_result.applicable_policies,
        "evidence_id": access_result.evidence_id
    }

@router.post("/access/evaluate")
async def evaluate_access_policy(
    request_data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Evaluate access policy for specific resource
    Task 18.1.15: Bind policies to orchestration
    """
    try:
        # Extract resource information
        resource_id = request_data.get("resource_id")
        resource_type = request_data.get("resource_type")
        operation = request_data.get("operation")
        
        # Create enhanced context
        enhanced_context = AccessContext(
            user_id=access_result.context.user_id,
            tenant_id=access_result.context.tenant_id,
            role_ids=access_result.context.role_ids,
            permissions=access_result.context.permissions,
            attributes=access_result.context.attributes,
            request_path=f"/resource/{resource_type}/{resource_id}",
            request_method=operation.upper() if operation else "GET",
            resource_id=resource_id,
            industry_overlay=access_result.context.industry_overlay,
            region=access_result.context.region,
            sla_tier=access_result.context.sla_tier
        )
        
        # Evaluate ABAC policies
        abac_result = await security_middleware.abac_middleware.evaluate_abac_policies(enhanced_context)
        
        return {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "operation": operation,
            "access_decision": abac_result.decision.value,
            "reason": abac_result.reason,
            "applicable_policies": abac_result.applicable_policies,
            "violations": abac_result.violations,
            "override_required": abac_result.override_required
        }
        
    except Exception as e:
        logger.error(f"❌ Access policy evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy evaluation error: {str(e)}")

@router.get("/roles/{tenant_id}")
async def get_tenant_roles(
    tenant_id: int,
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Get roles for tenant
    Task 18.1.3-18.1.4: Define base roles and industry overlays
    """
    if access_result.context.tenant_id != tenant_id and "admin" not in access_result.context.role_ids:
        raise HTTPException(status_code=403, detail="Access denied to tenant roles")
    
    # Mock role data - in production, query from authz_policy_role table
    roles = [
        {
            "role_id": "cro",
            "role_name": "Chief Revenue Officer",
            "role_level": "executive",
            "permissions": ["workflow_read", "workflow_approve", "audit_read"],
            "industry_overlay": access_result.context.industry_overlay
        },
        {
            "role_id": "revops_manager",
            "role_name": "RevOps Manager",
            "role_level": "manager",
            "permissions": ["workflow_read", "workflow_create", "dashboard_access"],
            "industry_overlay": access_result.context.industry_overlay
        }
    ]
    
    return {
        "tenant_id": tenant_id,
        "roles": roles,
        "industry_overlay": access_result.context.industry_overlay
    }

# ================================
# PII Handling Endpoints
# ================================

@router.post("/pii/classify")
async def classify_pii_data(
    data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Classify PII in data
    Task 18.2.5-18.2.7: Industry-specific PII classification
    """
    try:
        classifications = await pii_manager.detector.classify_pii_fields(
            data, access_result.context.industry_overlay
        )
        
        return {
            "tenant_id": access_result.context.tenant_id,
            "industry_overlay": access_result.context.industry_overlay,
            "pii_fields_detected": len(classifications),
            "classifications": [
                {
                    "field_name": c.field_name,
                    "category": c.category.value,
                    "sensitivity_level": c.sensitivity_level.value,
                    "confidence_score": c.confidence_score,
                    "regulatory_frameworks": c.regulatory_frameworks,
                    "masking_required": c.masking_required
                }
                for c in classifications
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ PII classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"PII classification error: {str(e)}")

@router.post("/pii/mask")
async def mask_pii_data(
    request_data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Apply PII masking to data
    Task 18.2.8: Field-level masking rules
    """
    try:
        data = request_data.get("data", {})
        
        # Process data with PII protection
        masked_data = await pii_manager.process_data_with_pii_protection(
            data,
            access_result.context.tenant_id,
            access_result.context.industry_overlay,
            access_result.context.sla_tier
        )
        
        return {
            "tenant_id": access_result.context.tenant_id,
            "industry_overlay": access_result.context.industry_overlay,
            "masking_applied": masked_data.get("_pii_masking_applied", False),
            "masked_data": masked_data,
            "masking_log": masked_data.get("_pii_masking_log", [])
        }
        
    except Exception as e:
        logger.error(f"❌ PII masking failed: {e}")
        raise HTTPException(status_code=500, detail=f"PII masking error: {str(e)}")

@router.post("/pii/anonymize")
async def anonymize_dataset(
    request_data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Anonymize dataset
    Task 18.2.11: Anonymization rules (k-anonymity, l-diversity, t-closeness)
    """
    try:
        dataset = request_data.get("dataset", [])
        config_name = request_data.get("config", "k_anonymity_basic")
        
        anonymized_dataset = await pii_manager.anonymization_service.anonymize_dataset(
            dataset, config_name
        )
        
        return {
            "tenant_id": access_result.context.tenant_id,
            "original_records": len(dataset),
            "anonymized_records": len(anonymized_dataset),
            "anonymization_config": config_name,
            "anonymized_dataset": anonymized_dataset
        }
        
    except Exception as e:
        logger.error(f"❌ Dataset anonymization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anonymization error: {str(e)}")

@router.post("/pii/purge/schedule")
async def schedule_pii_purge(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Schedule PII data purge
    Task 18.2.12: Build purge service for expired data
    """
    try:
        sla_tier = request_data.get("sla_tier", access_result.context.sla_tier)
        
        purge_job = await pii_manager.purge_service.schedule_purge_jobs(
            access_result.context.tenant_id, sla_tier
        )
        
        return {
            "tenant_id": access_result.context.tenant_id,
            "sla_tier": sla_tier,
            "purge_job": purge_job
        }
        
    except Exception as e:
        logger.error(f"❌ PII purge scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Purge scheduling error: {str(e)}")

# ================================
# Residency Enforcement Endpoints
# ================================

@router.post("/residency/setup")
async def setup_tenant_residency(
    request_data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Setup tenant residency compliance
    Task 18.3.4-18.3.6: Industry-specific residency overlays
    """
    try:
        if "admin" not in access_result.context.role_ids:
            raise HTTPException(status_code=403, detail="Admin role required for residency setup")
        
        tenant_id = request_data.get("tenant_id", access_result.context.tenant_id)
        industry_overlay = request_data.get("industry_overlay", access_result.context.industry_overlay)
        primary_region = Region(request_data.get("primary_region", "US"))
        compliance_frameworks = request_data.get("compliance_frameworks", ["GDPR", "CCPA"])
        
        setup_result = await residency_manager.setup_tenant_residency(
            tenant_id, industry_overlay, primary_region, compliance_frameworks
        )
        
        return setup_result
        
    except Exception as e:
        logger.error(f"❌ Residency setup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Residency setup error: {str(e)}")

@router.post("/residency/validate")
async def validate_cross_border_request(
    request_data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Validate cross-border data request
    Task 18.3.13: Configure policy engine for residency
    """
    try:
        source_region = request_data.get("source_region", access_result.context.region)
        target_region = request_data.get("target_region")
        data_classification = request_data.get("data_classification", "internal")
        operation_type = request_data.get("operation_type", "data_transfer")
        
        if not target_region:
            raise HTTPException(status_code=400, detail="Target region is required")
        
        validation_result = await residency_manager.validate_cross_border_request(
            access_result.context.tenant_id,
            access_result.context.user_id,
            source_region,
            target_region,
            data_classification,
            operation_type
        )
        
        return validation_result
        
    except Exception as e:
        logger.error(f"❌ Cross-border validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Residency validation error: {str(e)}")

@router.post("/residency/infrastructure/bind")
async def bind_infrastructure_to_region(
    request_data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Bind tenant infrastructure to region
    Task 18.3.8-18.3.12: Infrastructure region binding
    """
    try:
        if "admin" not in access_result.context.role_ids:
            raise HTTPException(status_code=403, detail="Admin role required for infrastructure binding")
        
        tenant_id = request_data.get("tenant_id", access_result.context.tenant_id)
        region = Region(request_data.get("region", access_result.context.region))
        services = request_data.get("services", ["storage", "database", "compute", "keyvault"])
        
        binding_result = await residency_manager.infra_manager.bind_tenant_to_region(
            tenant_id, region, services
        )
        
        return binding_result
        
    except Exception as e:
        logger.error(f"❌ Infrastructure binding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Infrastructure binding error: {str(e)}")

# ================================
# Penetration Testing Endpoints
# ================================

@router.post("/pentest/config")
async def create_pentest_config(
    request_data: Dict[str, Any],
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Create penetration test configuration
    Task 18.4.1: Pen-test configuration and scope
    """
    try:
        if "admin" not in access_result.context.role_ids and "security_officer" not in access_result.context.role_ids:
            raise HTTPException(status_code=403, detail="Admin or Security Officer role required")
        
        config = PenTestConfig(
            config_id=str(uuid.uuid4()),
            config_name=request_data.get("config_name", "Default Pen Test"),
            test_scope=TestScope(request_data.get("test_scope", "application")),
            industry_overlay=request_data.get("industry_overlay", access_result.context.industry_overlay),
            test_types=[TestType(t) for t in request_data.get("test_types", ["api_security"])],
            target_systems=request_data.get("target_systems", []),
            attack_vectors=[AttackVector(v) for v in request_data.get("attack_vectors", ["external"])],
            test_schedule=request_data.get("test_schedule", "on_demand"),
            automation_enabled=request_data.get("automation_enabled", True),
            ci_cd_integration=request_data.get("ci_cd_integration", False),
            compliance_frameworks=request_data.get("compliance_frameworks", ["SOX", "GDPR"]),
            severity_thresholds=request_data.get("severity_thresholds", {"critical": 0, "high": 5}),
            notification_rules=request_data.get("notification_rules", {}),
            tenant_id=access_result.context.tenant_id
        )
        
        return {
            "config_id": config.config_id,
            "config_name": config.config_name,
            "test_scope": config.test_scope.value,
            "industry_overlay": config.industry_overlay,
            "test_types": [t.value for t in config.test_types],
            "target_systems": config.target_systems,
            "automation_enabled": config.automation_enabled,
            "ci_cd_integration": config.ci_cd_integration
        }
        
    except Exception as e:
        logger.error(f"❌ Pen test config creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Config creation error: {str(e)}")

@router.post("/pentest/execute")
async def execute_penetration_test(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Execute penetration test
    Task 18.4.6-18.4.20: Automated pen-test suite execution
    """
    try:
        if "admin" not in access_result.context.role_ids and "security_officer" not in access_result.context.role_ids:
            raise HTTPException(status_code=403, detail="Admin or Security Officer role required")
        
        config_id = request_data.get("config_id")
        if not config_id:
            raise HTTPException(status_code=400, detail="Config ID is required")
        
        # Create test configuration
        config = PenTestConfig(
            config_id=config_id,
            config_name=request_data.get("config_name", "Ad-hoc Pen Test"),
            test_scope=TestScope(request_data.get("test_scope", "application")),
            industry_overlay=access_result.context.industry_overlay,
            test_types=[TestType(t) for t in request_data.get("test_types", ["api_security", "sql_injection"])],
            target_systems=request_data.get("target_systems", ["localhost:8000"]),
            attack_vectors=[AttackVector(v) for v in request_data.get("attack_vectors", ["external"])],
            test_schedule="on_demand",
            automation_enabled=True,
            ci_cd_integration=False,
            compliance_frameworks=request_data.get("compliance_frameworks", ["SOX", "GDPR"]),
            severity_thresholds={"critical": 0, "high": 5, "medium": 10},
            notification_rules={},
            tenant_id=access_result.context.tenant_id
        )
        
        # Execute pen test in background
        test_run_id = str(uuid.uuid4())
        background_tasks.add_task(
            execute_pentest_background,
            config,
            test_run_id
        )
        
        return {
            "test_run_id": test_run_id,
            "config_id": config.config_id,
            "status": "started",
            "tenant_id": access_result.context.tenant_id,
            "industry_overlay": config.industry_overlay,
            "test_types": [t.value for t in config.test_types],
            "target_systems": config.target_systems,
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Pen test execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pen test execution error: {str(e)}")

@router.get("/pentest/results/{test_run_id}")
async def get_pentest_results(
    test_run_id: str,
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Get penetration test results
    Task 18.4.2: Pen-test results and evidence
    """
    try:
        # Mock results - in production, query from pentest_policy_result table
        mock_results = {
            "test_run_id": test_run_id,
            "tenant_id": access_result.context.tenant_id,
            "status": "completed",
            "summary": {
                "total_tests": 12,
                "severity_breakdown": {
                    "critical": 0,
                    "high": 1,
                    "medium": 3,
                    "low": 5,
                    "info": 3
                },
                "status_breakdown": {
                    "success": 2,
                    "blocked": 8,
                    "failed": 2
                },
                "risk_score": 15,
                "security_posture": "strong",
                "tests_passed": 10,
                "tests_failed": 2,
                "completion_rate": 100.0
            },
            "detailed_results": [
                {
                    "result_id": str(uuid.uuid4()),
                    "test_type": "api_security",
                    "attack_vector": "external",
                    "target_system": "localhost:8000",
                    "result_status": "blocked",
                    "severity": "low",
                    "vulnerability_details": {"test_type": "api_security_validation"},
                    "remediation_status": "passed"
                }
            ],
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        return mock_results
        
    except Exception as e:
        logger.error(f"❌ Failed to get pen test results: {e}")
        raise HTTPException(status_code=500, detail=f"Results retrieval error: {str(e)}")

# ================================
# Security Dashboard Endpoints
# ================================

@router.get("/dashboard/overview")
async def get_security_overview(
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Get security overview dashboard
    Task 18.1.23-18.1.25: Build audit, industry, SLA dashboards
    """
    try:
        # Mock security overview - in production, aggregate from various sources
        overview = {
            "tenant_id": access_result.context.tenant_id,
            "industry_overlay": access_result.context.industry_overlay,
            "security_posture": {
                "overall_score": 85,
                "rbac_compliance": 92,
                "pii_protection": 88,
                "residency_compliance": 95,
                "pen_test_score": 78
            },
            "recent_activities": [
                {
                    "activity_type": "access_granted",
                    "user_id": access_result.context.user_id,
                    "resource": "/api/workflows",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "compliance_status": {
                "gdpr": "compliant",
                "sox": "compliant",
                "hipaa": "not_applicable" if access_result.context.industry_overlay != "insurance" else "compliant",
                "rbi": "not_applicable" if access_result.context.industry_overlay != "banking" else "compliant"
            },
            "alerts": [
                {
                    "alert_type": "info",
                    "message": "All security controls are functioning normally",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"❌ Failed to get security overview: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@router.get("/compliance/status")
async def get_compliance_status(
    access_result: AccessResult = Depends(require_access_control)
):
    """
    Get compliance status
    Task 18.1.20: Attach compliance packs
    """
    try:
        # Determine applicable compliance frameworks based on industry
        frameworks = []
        if access_result.context.industry_overlay == "saas":
            frameworks = ["GDPR", "CCPA", "SOX"]
        elif access_result.context.industry_overlay == "banking":
            frameworks = ["RBI", "DPDP", "AML", "KYC"]
        elif access_result.context.industry_overlay == "insurance":
            frameworks = ["HIPAA", "NAIC", "SOX"]
        else:
            frameworks = ["GDPR", "SOX"]
        
        compliance_status = {
            "tenant_id": access_result.context.tenant_id,
            "industry_overlay": access_result.context.industry_overlay,
            "applicable_frameworks": frameworks,
            "compliance_details": {
                framework: {
                    "status": "compliant",
                    "last_audit": datetime.now(timezone.utc).isoformat(),
                    "next_review": (datetime.now(timezone.utc)).isoformat(),
                    "violations": 0
                }
                for framework in frameworks
            },
            "overall_compliance_score": 95
        }
        
        return compliance_status
        
    except Exception as e:
        logger.error(f"❌ Failed to get compliance status: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance status error: {str(e)}")

# Background task for pen test execution
async def execute_pentest_background(config: PenTestConfig, test_run_id: str):
    """Background task to execute penetration test"""
    try:
        logger.info(f"🔍 Starting pen test execution: {test_run_id}")
        
        # Execute comprehensive pen test
        results = await pentest_orchestrator.execute_comprehensive_pentest(config)
        
        # In production, store results in database
        logger.info(f"✅ Pen test completed: {test_run_id}")
        
    except Exception as e:
        logger.error(f"❌ Background pen test failed: {e}")

# Health check endpoint
@router.get("/health")
async def security_health_check():
    """Security services health check"""
    return {
        "status": "healthy",
        "services": {
            "rbac_abac": "operational",
            "pii_handling": "operational",
            "residency_enforcement": "operational",
            "penetration_testing": "operational"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
