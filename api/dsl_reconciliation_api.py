"""
DSL Reconciliation API
=====================

Task 7.1.16: Configure DSL reconciliation checks
API endpoints for DSL workflow reconciliation and validation
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

# Import reconciliation engine
try:
    from ..dsl.validation.dsl_reconciliation_engine import (
        get_reconciliation_engine,
        ReconciliationResult,
        ReconciliationStatus,
        ReconciliationSeverity
    )
except ImportError:
    # Fallback if reconciliation engine is not available
    def get_reconciliation_engine():
        return None

# Import connection pool
try:
    from ..src.services.connection_pool_manager import get_pool_manager
except ImportError:
    def get_pool_manager():
        return None

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class ReconciliationRequest(BaseModel):
    """Request model for DSL reconciliation"""
    workflow_id: str = Field(..., description="Workflow identifier")
    execution_id: str = Field(..., description="Execution identifier")
    tenant_id: int = Field(..., description="Tenant identifier")
    deep_validation: bool = Field(False, description="Enable comprehensive validation")

class ReconciliationIssueResponse(BaseModel):
    """Response model for reconciliation issues"""
    issue_id: str
    severity: str
    category: str
    message: str
    dsl_reference: Optional[str] = None
    trace_reference: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    recommendation: Optional[str] = None

class ReconciliationResponse(BaseModel):
    """Response model for reconciliation results"""
    workflow_id: str
    execution_id: str
    status: str
    issues: List[ReconciliationIssueResponse]
    checks_performed: int
    checks_passed: int
    checks_failed: int
    checks_skipped: int
    reconciliation_timestamp: datetime
    summary: Dict[str, Any]

class BulkReconciliationRequest(BaseModel):
    """Request model for bulk reconciliation"""
    tenant_id: int = Field(..., description="Tenant identifier")
    workflow_ids: Optional[List[str]] = Field(None, description="Specific workflow IDs to reconcile")
    time_range_hours: int = Field(24, description="Time range in hours for recent executions")
    max_executions: int = Field(100, description="Maximum number of executions to reconcile")

class ReconciliationSummaryResponse(BaseModel):
    """Response model for reconciliation summary"""
    total_executions: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    reconciliation_rate: float
    most_common_issues: List[Dict[str, Any]]

@router.post("/reconcile", response_model=ReconciliationResponse)
async def reconcile_workflow_execution(
    request: ReconciliationRequest,
    pool_manager=Depends(get_pool_manager)
):
    """
    Reconcile a specific workflow execution against its DSL definition
    
    Validates consistency between DSL workflow definition and execution trace
    """
    try:
        logger.info(f"🔍 Starting DSL reconciliation: {request.workflow_id} (execution: {request.execution_id})")
        
        # Get reconciliation engine
        reconciliation_engine = get_reconciliation_engine()
        if not reconciliation_engine:
            raise HTTPException(status_code=500, detail="Reconciliation engine not available")
        
        # Set pool manager
        reconciliation_engine.pool_manager = pool_manager
        
        # Perform reconciliation
        result = await reconciliation_engine.reconcile_workflow_execution(
            workflow_id=request.workflow_id,
            execution_id=request.execution_id,
            tenant_id=request.tenant_id,
            deep_validation=request.deep_validation
        )
        
        # Convert issues to response format
        issues_response = []
        for issue in result.issues:
            issues_response.append(ReconciliationIssueResponse(
                issue_id=issue.issue_id,
                severity=issue.severity.value,
                category=issue.category,
                message=issue.message,
                dsl_reference=issue.dsl_reference,
                trace_reference=issue.trace_reference,
                expected_value=issue.expected_value,
                actual_value=issue.actual_value,
                recommendation=issue.recommendation
            ))
        
        # Create summary
        summary = {
            "overall_status": result.status.value,
            "success_rate": (result.checks_passed / max(result.checks_performed, 1)) * 100,
            "issue_breakdown": {
                "critical": len([i for i in result.issues if i.severity == ReconciliationSeverity.CRITICAL]),
                "high": len([i for i in result.issues if i.severity == ReconciliationSeverity.HIGH]),
                "medium": len([i for i in result.issues if i.severity == ReconciliationSeverity.MEDIUM]),
                "low": len([i for i in result.issues if i.severity == ReconciliationSeverity.LOW]),
                "info": len([i for i in result.issues if i.severity == ReconciliationSeverity.INFO])
            },
            "validation_categories": list(set([i.category for i in result.issues])),
            "recommendations": [i.recommendation for i in result.issues if i.recommendation]
        }
        
        logger.info(f"✅ DSL reconciliation completed: {result.status.value} ({len(result.issues)} issues)")
        
        return ReconciliationResponse(
            workflow_id=result.workflow_id,
            execution_id=result.execution_id,
            status=result.status.value,
            issues=issues_response,
            checks_performed=result.checks_performed,
            checks_passed=result.checks_passed,
            checks_failed=result.checks_failed,
            checks_skipped=result.checks_skipped,
            reconciliation_timestamp=result.reconciliation_timestamp,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"❌ DSL reconciliation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {str(e)}")

@router.post("/reconcile-bulk", response_model=ReconciliationSummaryResponse)
async def reconcile_bulk_executions(
    request: BulkReconciliationRequest,
    pool_manager=Depends(get_pool_manager)
):
    """
    Perform bulk reconciliation of recent workflow executions
    
    Reconciles multiple executions and provides summary statistics
    """
    try:
        logger.info(f"🔍 Starting bulk DSL reconciliation for tenant {request.tenant_id}")
        
        # Get reconciliation engine
        reconciliation_engine = get_reconciliation_engine()
        if not reconciliation_engine:
            raise HTTPException(status_code=500, detail="Reconciliation engine not available")
        
        # Set pool manager
        reconciliation_engine.pool_manager = pool_manager
        
        # Get recent executions to reconcile
        executions = await _get_recent_executions(
            pool_manager=pool_manager,
            tenant_id=request.tenant_id,
            workflow_ids=request.workflow_ids,
            time_range_hours=request.time_range_hours,
            max_executions=request.max_executions
        )
        
        if not executions:
            return ReconciliationSummaryResponse(
                total_executions=0,
                passed=0,
                failed=0,
                warnings=0,
                skipped=0,
                critical_issues=0,
                high_issues=0,
                medium_issues=0,
                low_issues=0,
                reconciliation_rate=0.0,
                most_common_issues=[]
            )
        
        # Perform reconciliation for each execution
        results = []
        for execution in executions:
            try:
                result = await reconciliation_engine.reconcile_workflow_execution(
                    workflow_id=execution['workflow_id'],
                    execution_id=execution['execution_id'],
                    tenant_id=request.tenant_id,
                    deep_validation=False  # Skip deep validation for bulk operations
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to reconcile execution {execution['execution_id']}: {e}")
                continue
        
        # Calculate summary statistics
        total_executions = len(results)
        passed = len([r for r in results if r.status == ReconciliationStatus.PASSED])
        failed = len([r for r in results if r.status == ReconciliationStatus.FAILED])
        warnings = len([r for r in results if r.status == ReconciliationStatus.WARNING])
        skipped = len([r for r in results if r.status == ReconciliationStatus.SKIPPED])
        
        # Count issues by severity
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        critical_issues = len([i for i in all_issues if i.severity == ReconciliationSeverity.CRITICAL])
        high_issues = len([i for i in all_issues if i.severity == ReconciliationSeverity.HIGH])
        medium_issues = len([i for i in all_issues if i.severity == ReconciliationSeverity.MEDIUM])
        low_issues = len([i for i in all_issues if i.severity == ReconciliationSeverity.LOW])
        
        # Calculate reconciliation rate
        reconciliation_rate = (passed / max(total_executions, 1)) * 100
        
        # Find most common issues
        issue_counts = {}
        for issue in all_issues:
            key = f"{issue.category}:{issue.issue_id}"
            issue_counts[key] = issue_counts.get(key, 0) + 1
        
        most_common_issues = [
            {"issue_type": key, "count": count, "category": key.split(':')[0]}
            for key, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        logger.info(f"✅ Bulk DSL reconciliation completed: {total_executions} executions, "
                   f"{passed} passed, {failed} failed, {warnings} warnings")
        
        return ReconciliationSummaryResponse(
            total_executions=total_executions,
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            reconciliation_rate=reconciliation_rate,
            most_common_issues=most_common_issues
        )
        
    except Exception as e:
        logger.error(f"❌ Bulk DSL reconciliation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk reconciliation failed: {str(e)}")

@router.get("/reconciliation-rules")
async def get_reconciliation_rules():
    """
    Get current reconciliation rules configuration
    """
    try:
        reconciliation_engine = get_reconciliation_engine()
        if not reconciliation_engine:
            raise HTTPException(status_code=500, detail="Reconciliation engine not available")
        
        return {
            "rules": reconciliation_engine.reconciliation_rules,
            "description": "DSL reconciliation rules configuration",
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get reconciliation rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")

@router.post("/reconciliation-rules")
async def update_reconciliation_rules(
    rules: Dict[str, Any] = Body(..., description="Updated reconciliation rules")
):
    """
    Update reconciliation rules configuration
    """
    try:
        reconciliation_engine = get_reconciliation_engine()
        if not reconciliation_engine:
            raise HTTPException(status_code=500, detail="Reconciliation engine not available")
        
        # Validate rules structure
        required_categories = ['step_consistency', 'execution_flow', 'data_consistency', 
                             'governance_compliance', 'performance_validation', 'error_handling']
        
        for category in required_categories:
            if category not in rules:
                raise HTTPException(status_code=400, detail=f"Missing required rule category: {category}")
        
        # Update rules
        reconciliation_engine.reconciliation_rules = rules
        
        logger.info("✅ Reconciliation rules updated successfully")
        
        return {
            "status": "success",
            "message": "Reconciliation rules updated successfully",
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to update reconciliation rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update rules: {str(e)}")

@router.get("/health")
async def reconciliation_health_check():
    """
    Health check for DSL reconciliation service
    """
    try:
        reconciliation_engine = get_reconciliation_engine()
        
        return {
            "status": "healthy" if reconciliation_engine else "unhealthy",
            "service": "DSL Reconciliation API",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "single_reconciliation": reconciliation_engine is not None,
                "bulk_reconciliation": reconciliation_engine is not None,
                "rule_configuration": reconciliation_engine is not None
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def _get_recent_executions(
    pool_manager,
    tenant_id: int,
    workflow_ids: Optional[List[str]] = None,
    time_range_hours: int = 24,
    max_executions: int = 100
) -> List[Dict[str, Any]]:
    """Get recent workflow executions for reconciliation"""
    
    if not pool_manager:
        return []
    
    try:
        async with pool_manager.get_connection() as conn:
            # Build query
            time_threshold = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            if workflow_ids:
                workflow_filter = "AND workflow_id = ANY($3)"
                params = [tenant_id, time_threshold, workflow_ids, max_executions]
            else:
                workflow_filter = ""
                params = [tenant_id, time_threshold, max_executions]
            
            query = f"""
                SELECT execution_id, workflow_id, execution_status, created_at
                FROM dsl_execution_traces 
                WHERE tenant_id = $1 
                AND created_at >= $2
                {workflow_filter}
                ORDER BY created_at DESC 
                LIMIT ${len(params)}
            """
            
            rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to get recent executions: {e}")
        return []
