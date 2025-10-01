#!/usr/bin/env python3
"""
Dynamic Capabilities API
========================

FastAPI endpoints for the Dynamic SaaS Capability System.
Integrates with existing Node.js backend for seamless capability management.

Endpoints:
- POST /api/capabilities/discover - Discover patterns and generate templates
- GET /api/capabilities/recommendations - Get intelligent recommendations
- POST /api/capabilities/adapt - Trigger template adaptation
- GET /api/capabilities/status - Get system status and metrics
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import asyncio

from src.services.connection_pool_manager import pool_manager
from src.services.jwt_auth import get_current_user
from dsl.capability_registry.dynamic_capability_orchestrator import DynamicCapabilityOrchestrator

logger = logging.getLogger(__name__)

# Initialize the orchestrator (singleton pattern)
_orchestrator_instance = None

async def get_orchestrator():
    """Get or create the orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = DynamicCapabilityOrchestrator(pool_manager)
        await _orchestrator_instance.initialize()
    return _orchestrator_instance

# Pydantic models for API
class CapabilityDiscoveryRequest(BaseModel):
    """Request model for capability discovery"""
    tenant_id: int = Field(..., description="Tenant identifier")
    force_refresh: bool = Field(False, description="Force rediscovery of patterns")
    include_templates: bool = Field(True, description="Generate templates from patterns")
    
class RecommendationRequest(BaseModel):
    """Request model for getting recommendations"""
    tenant_id: int = Field(..., description="Tenant identifier")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context (role, preferences, etc.)")
    max_recommendations: int = Field(10, description="Maximum number of recommendations")
    
class AdaptationRequest(BaseModel):
    """Request model for template adaptation"""
    tenant_id: int = Field(..., description="Tenant identifier")
    template_ids: Optional[List[str]] = Field(None, description="Specific templates to adapt (optional)")
    
class CapabilityStatusResponse(BaseModel):
    """Response model for system status"""
    status: str
    tenant_metrics: Dict[str, Any]
    system_health: Dict[str, Any]
    last_updated: datetime

# Create router
router = APIRouter(prefix="/api/capabilities", tags=["Dynamic Capabilities"])

@router.post("/discover")
async def discover_capabilities(
    request: CapabilityDiscoveryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    orchestrator: DynamicCapabilityOrchestrator = Depends(get_orchestrator)
):
    """
    Discover SaaS business patterns and generate dynamic capability templates
    
    This endpoint:
    1. Analyzes tenant data to discover business patterns
    2. Generates adaptive automation templates
    3. Returns comprehensive discovery report
    4. Triggers background learning processes
    """
    try:
        logger.info(f"Capability discovery requested for tenant {request.tenant_id} by user {current_user.get('user_id')}")
        
        # Validate tenant access
        if not await _validate_tenant_access(current_user, request.tenant_id):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")
        
        # Run discovery and generation
        report = await orchestrator.discover_and_generate_capabilities(
            tenant_id=request.tenant_id,
            force_refresh=request.force_refresh
        )
        
        if report['status'] == 'completed':
            # Trigger background learning processes
            background_tasks.add_task(
                _trigger_background_learning,
                orchestrator,
                request.tenant_id
            )
            
            # Log successful discovery
            logger.info(f"Capability discovery completed for tenant {request.tenant_id}: "
                       f"{report['summary']['templates_generated']} templates generated")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Capability discovery completed successfully",
                    "data": report,
                    "metadata": {
                        "processing_time": _calculate_processing_time(report),
                        "user_id": current_user.get('user_id'),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
        else:
            logger.error(f"Capability discovery failed for tenant {request.tenant_id}: {report.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Capability discovery failed: {report.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in capability discovery: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/recommendations")
async def get_recommendations(
    tenant_id: int,
    user_role: Optional[str] = None,
    focus_area: Optional[str] = None,
    max_results: int = 10,
    current_user: dict = Depends(get_current_user),
    orchestrator: DynamicCapabilityOrchestrator = Depends(get_orchestrator)
):
    """
    Get intelligent capability recommendations based on tenant patterns and user context
    
    Returns personalized recommendations for:
    - Quick start automation workflows
    - Role-specific capabilities
    - Business model optimizations
    - Performance improvements
    """
    try:
        logger.info(f"Recommendations requested for tenant {tenant_id} by user {current_user.get('user_id')}")
        
        # Validate tenant access
        if not await _validate_tenant_access(current_user, tenant_id):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")
        
        # Build user context
        user_context = {
            "user_role": user_role or current_user.get("role", "User"),
            "focus_area": focus_area,
            "user_id": current_user.get("user_id"),
            "tenant_id": tenant_id
        }
        
        # Get recommendations
        recommendations = await orchestrator.get_intelligent_recommendations(
            tenant_id=tenant_id,
            user_context=user_context
        )
        
        # Limit results
        limited_recommendations = recommendations[:max_results]
        
        logger.info(f"Generated {len(limited_recommendations)} recommendations for tenant {tenant_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "recommendations": limited_recommendations,
                    "total_available": len(recommendations),
                    "user_context": user_context,
                    "generated_at": datetime.now().isoformat()
                },
                "metadata": {
                    "tenant_id": tenant_id,
                    "user_id": current_user.get('user_id'),
                    "recommendation_count": len(limited_recommendations)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/adapt")
async def adapt_templates(
    request: AdaptationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    orchestrator: DynamicCapabilityOrchestrator = Depends(get_orchestrator)
):
    """
    Trigger template adaptation based on usage patterns and performance data
    
    This endpoint:
    1. Analyzes template usage and performance
    2. Adapts templates for better performance
    3. Creates new templates for emerging patterns
    4. Deprecates unused templates
    """
    try:
        logger.info(f"Template adaptation requested for tenant {request.tenant_id} by user {current_user.get('user_id')}")
        
        # Validate tenant access
        if not await _validate_tenant_access(current_user, request.tenant_id):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")
        
        # Run adaptation in background for better performance
        background_tasks.add_task(
            _run_template_adaptation,
            orchestrator,
            request.tenant_id,
            request.template_ids,
            current_user.get('user_id')
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "success": True,
                "message": "Template adaptation started successfully",
                "data": {
                    "tenant_id": request.tenant_id,
                    "status": "processing",
                    "started_at": datetime.now().isoformat()
                },
                "metadata": {
                    "user_id": current_user.get('user_id'),
                    "processing_mode": "background"
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting template adaptation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status")
async def get_system_status(
    tenant_id: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
    orchestrator: DynamicCapabilityOrchestrator = Depends(get_orchestrator)
):
    """
    Get system status and metrics for dynamic capabilities
    
    Returns:
    - Overall system health
    - Tenant-specific metrics (if tenant_id provided)
    - Template generation statistics
    - Learning system status
    """
    try:
        logger.info(f"System status requested by user {current_user.get('user_id')}")
        
        # Base system status
        system_status = {
            "status": "operational",
            "components": {
                "pattern_engine": "active",
                "template_generators": "active", 
                "learning_engine": "active",
                "orchestrator": "active"
            },
            "last_health_check": datetime.now().isoformat()
        }
        
        # Tenant-specific metrics if requested
        tenant_metrics = {}
        if tenant_id:
            if not await _validate_tenant_access(current_user, tenant_id):
                raise HTTPException(status_code=403, detail="Access denied to tenant data")
            
            # Get tenant metrics from orchestrator
            tenant_metrics = {
                "tenant_id": tenant_id,
                "patterns_discovered": len(orchestrator.tenant_patterns.get(tenant_id, [])),
                "templates_generated": len(orchestrator.generated_templates.get(tenant_id, [])),
                "recommendations_available": len(orchestrator.recommendation_cache.get(tenant_id, [])),
                "last_discovery": "2024-01-15T10:30:00Z",  # Placeholder
                "adaptation_cycles": 0  # Placeholder
            }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "system_status": system_status,
                    "tenant_metrics": tenant_metrics,
                    "global_metrics": {
                        "total_tenants_active": len(orchestrator.tenant_patterns),
                        "total_templates_generated": sum(len(templates) for templates in orchestrator.generated_templates.values()),
                        "system_uptime": "99.9%",  # Placeholder
                        "avg_discovery_time": "2.3s"  # Placeholder
                    }
                },
                "metadata": {
                    "user_id": current_user.get('user_id'),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/templates")
async def get_templates(
    tenant_id: int,
    template_type: Optional[str] = None,
    category: Optional[str] = None,
    business_model: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    orchestrator: DynamicCapabilityOrchestrator = Depends(get_orchestrator)
):
    """
    Get available templates for a tenant with optional filtering
    
    Filters:
    - template_type: RBA_TEMPLATE, RBIA_MODEL, AALA_AGENT
    - category: revenue_tracking, pipeline_management, etc.
    - business_model: subscription_recurring, usage_based, etc.
    """
    try:
        logger.info(f"Templates requested for tenant {tenant_id} by user {current_user.get('user_id')}")
        
        # Validate tenant access
        if not await _validate_tenant_access(current_user, tenant_id):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")
        
        # Get templates for tenant
        all_templates = orchestrator.generated_templates.get(tenant_id, [])
        
        # Apply filters
        filtered_templates = all_templates
        if template_type:
            filtered_templates = [t for t in filtered_templates if t.capability_type == template_type]
        if category:
            filtered_templates = [t for t in filtered_templates if t.category == category]
        if business_model:
            filtered_templates = [t for t in filtered_templates if t.business_model.value == business_model]
        
        # Convert to dict format for JSON response
        template_data = []
        for template in filtered_templates:
            template_data.append({
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "capability_type": template.capability_type,
                "category": template.category,
                "business_model": template.business_model.value,
                "confidence_score": template.confidence_score,
                "created_at": template.created_at.isoformat(),
                "input_schema": template.input_schema,
                "output_schema": template.output_schema
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "templates": template_data,
                    "total_count": len(template_data),
                    "filters_applied": {
                        "template_type": template_type,
                        "category": category,
                        "business_model": business_model
                    }
                },
                "metadata": {
                    "tenant_id": tenant_id,
                    "user_id": current_user.get('user_id'),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Helper functions
async def _validate_tenant_access(current_user: dict, tenant_id: int) -> bool:
    """Validate that user has access to tenant data"""
    user_tenant_id = current_user.get("tenant_id")
    return user_tenant_id == tenant_id

async def _trigger_background_learning(orchestrator: DynamicCapabilityOrchestrator, tenant_id: int):
    """Trigger background learning processes"""
    try:
        logger.info(f"Starting background learning for tenant {tenant_id}")
        # This would trigger various learning processes
        await asyncio.sleep(1)  # Simulate background processing
        logger.info(f"Background learning completed for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Background learning failed for tenant {tenant_id}: {e}")

async def _run_template_adaptation(orchestrator: DynamicCapabilityOrchestrator, tenant_id: int, template_ids: Optional[List[str]], user_id: int):
    """Run template adaptation in background"""
    try:
        logger.info(f"Starting template adaptation for tenant {tenant_id}")
        adaptation_report = await orchestrator.adapt_templates_from_usage(tenant_id)
        logger.info(f"Template adaptation completed for tenant {tenant_id}: {adaptation_report['status']}")
    except Exception as e:
        logger.error(f"Template adaptation failed for tenant {tenant_id}: {e}")

def _calculate_processing_time(report: Dict[str, Any]) -> float:
    """Calculate processing time from report"""
    try:
        start = datetime.fromisoformat(report['started_at'])
        end = datetime.fromisoformat(report['completed_at'])
        return (end - start).total_seconds()
    except:
        return 0.0
