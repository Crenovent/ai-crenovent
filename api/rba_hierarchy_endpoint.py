"""
RBA Hierarchy Processing API Endpoint
====================================
This endpoint should be called by the Node.js backend instead of manual hierarchy processing.

The Node.js backend should ONLY:
1. Handle CSV upload
2. Call this RBA endpoint 
3. Store the results

NO manual hierarchy processing logic should remain in Node.js or frontend.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import tempfile
import os
from datetime import datetime

# Initialize logger first
logger = logging.getLogger(__name__)

# Import RBA workflow executor with proper path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.rba.hierarchy_workflow_executor import (
        HierarchyProcessingWorkflowExecutor,
        process_csv_hierarchy_via_rba
    )
    RBA_EXECUTOR_AVAILABLE = True
    logger.info("✅ RBA workflow executor loaded successfully")
except ImportError as e:
    logger.error(f"❌ RBA workflow executor not available: {e}")
    RBA_EXECUTOR_AVAILABLE = False
    
    # Create mock classes for fallback
    class HierarchyProcessingWorkflowExecutor:
        def __init__(self):
            pass
    
    async def process_csv_hierarchy_via_rba(*args, **kwargs):
        raise HTTPException(status_code=503, detail="RBA workflow executor not available")

# Create router
router = APIRouter(prefix="/api/rba/hierarchy", tags=["RBA Hierarchy Processing"])

# Request/Response Models
class HierarchyProcessingRequest(BaseModel):
    """Request model for hierarchy processing"""
    tenant_id: int = Field(..., description="Tenant ID for multi-tenant isolation")
    uploaded_by_user_id: int = Field(..., description="User ID who uploaded the CSV")
    processing_options: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional processing configuration"
    )

class HierarchyProcessingResponse(BaseModel):
    """Response model for hierarchy processing"""
    success: bool
    execution_id: str
    status: str
    processing_time_ms: int
    hierarchy_metrics: Dict[str, Any]
    validation_results: Dict[str, Any]
    audit_trail_id: Optional[str] = None
    recommendations: List[str] = []
    errors: List[str] = []

class HierarchyProcessingStatus(BaseModel):
    """Status model for checking processing progress"""
    execution_id: str
    status: str
    progress_percentage: float
    current_step: str
    estimated_completion: Optional[datetime] = None

# Global executor instance
hierarchy_executor = HierarchyProcessingWorkflowExecutor()

@router.post("/process-csv", response_model=HierarchyProcessingResponse)
async def process_hierarchy_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file to process"),
    tenant_id: int = Form(..., description="Tenant ID"),
    uploaded_by_user_id: int = Form(..., description="User ID who uploaded the file"),
    processing_options: Optional[str] = Form(None, description="JSON string of processing options")
):
    """
    Process CSV hierarchy using RBA DSL workflow automation.
    
    This endpoint replaces ALL manual hierarchy processing logic.
    The Node.js backend should call this instead of doing manual processing.
    
    Args:
        file: Uploaded CSV file
        tenant_id: Tenant ID for isolation
        uploaded_by_user_id: User who uploaded the file
        processing_options: Optional JSON configuration
        
    Returns:
        HierarchyProcessingResponse: Complete processing results
    """
    logger.info(f"🚀 [RBA API] Processing hierarchy CSV for tenant {tenant_id}")
    
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Save uploaded file temporarily
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"📁 Saved uploaded file to: {temp_file_path}")
        
        # Parse processing options
        options = None
        if processing_options:
            import json
            try:
                options = json.loads(processing_options)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in processing_options, using defaults")
        
        # Execute RBA workflow
        start_time = datetime.utcnow()
        
        result = await process_csv_hierarchy_via_rba(
            csv_file_path=temp_file_path,
            tenant_id=tenant_id,
            uploaded_by_user_id=uploaded_by_user_id,
            processing_options=options
        )
        
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Extract results
        hierarchy_metrics = result.final_output.get("hierarchy_metrics", {})
        validation_results = result.final_output.get("validation_details", {})
        
        logger.info(f"✅ [RBA API] Processing completed in {processing_time_ms}ms")
        
        # Clean up temp file in background
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return HierarchyProcessingResponse(
            success=result.status == "completed",
            execution_id=result.execution_id,
            status=result.status,
            processing_time_ms=processing_time_ms,
            hierarchy_metrics=hierarchy_metrics,
            validation_results=validation_results,
            audit_trail_id=result.audit_trail_id,
            recommendations=validation_results.get("recommendations", []),
            errors=[result.error] if result.error else []
        )
        
    except Exception as e:
        logger.error(f"❌ [RBA API] Hierarchy processing failed: {e}")
        
        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Hierarchy processing failed: {str(e)}")

@router.get("/status/{execution_id}", response_model=HierarchyProcessingStatus)
async def get_processing_status(execution_id: str):
    """
    Get the status of a hierarchy processing job.
    
    Args:
        execution_id: Execution ID returned from process-csv endpoint
        
    Returns:
        HierarchyProcessingStatus: Current processing status
    """
    # TODO: Implement actual status tracking
    # This would query the workflow engine for execution status
    
    return HierarchyProcessingStatus(
        execution_id=execution_id,
        status="completed",
        progress_percentage=100.0,
        current_step="completed"
    )

@router.post("/validate-csv", response_model=Dict[str, Any])
async def validate_hierarchy_csv(
    file: UploadFile = File(..., description="CSV file to validate"),
    tenant_id: int = Form(..., description="Tenant ID")
):
    """
    Validate CSV file structure without processing.
    
    Args:
        file: CSV file to validate
        tenant_id: Tenant ID
        
    Returns:
        Dict: Validation results
    """
    logger.info(f"🔍 [RBA API] Validating CSV for tenant {tenant_id}")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV for validation
        content = await file.read()
        
        # Save temporarily for validation
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Execute CSV ingestion step only
        executor = HierarchyProcessingWorkflowExecutor()
        validation_result = await executor.execute_csv_ingestion_step({
            "file_path": temp_file_path,
            "validation_rules": {
                "required_columns": ["Name", "Email"],
                "email_validation": True,
                "duplicate_detection": True
            }
        })
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "validation_passed": validation_result["validation_summary"]["validation_passed"],
            "validation_summary": validation_result["validation_summary"],
            "recommendations": []
        }
        
    except Exception as e:
        logger.error(f"❌ [RBA API] CSV validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"CSV validation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for RBA hierarchy processing service"""
    return {
        "status": "healthy" if RBA_EXECUTOR_AVAILABLE else "degraded",
        "service": "RBA Hierarchy Processing",
        "version": "1.0.0",
        "rba_executor_available": RBA_EXECUTOR_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }

def cleanup_temp_file(file_path: str):
    """Background task to clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"🧹 Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")

# Export router
__all__ = ["router"]
