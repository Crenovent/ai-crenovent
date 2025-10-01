"""
Hub Router - FastAPI endpoints for Hub Architecture functionality
Provides centralized workflow discovery, execution, and analytics
"""

import asyncio
import uuid
import json
import hashlib
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from .execution_hub import ExecutionHub
from .workflow_registry import WorkflowRegistry
from .agent_orchestrator import AgentOrchestrator, ExecutionPriority
from .knowledge_connector import KnowledgeConnector

# Pydantic models for Hub API
class WorkflowDiscoveryRequest(BaseModel):
    module: Optional[str] = Field(None, description="Filter by module (Forecast, Pipeline, Planning)")
    automation_type: Optional[str] = Field(None, description="Filter by automation type (RBA, RBIA, AALA)")
    user_context: Dict[str, Any] = Field(..., description="User context for personalized discovery")

class WorkflowExecutionRequest(BaseModel):
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    user_context: Dict[str, Any] = Field(..., description="User context with user_id (int), tenant_id (int), session_id")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the workflow")
    priority: str = Field("medium", description="Execution priority (low, medium, high, critical)")

class RBAAgentRegistrationRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the RBA agent")
    name: str = Field(..., description="Display name of the agent")
    module: str = Field(..., description="Module the agent belongs to (Forecast, Pipeline, Planning)")
    workflow_file: str = Field(..., description="YAML workflow file name")
    business_value: str = Field(..., description="Business value description")
    customer_impact: str = Field(..., description="Customer impact description")
    industry_tags: List[str] = Field(default_factory=list, description="Industry tags")
    persona_tags: List[str] = Field(default_factory=list, description="Target persona tags")
    estimated_time_minutes: int = Field(5, description="Estimated execution time in minutes")

class WorkflowAnalyticsRequest(BaseModel):
    workflow_id: Optional[str] = Field(None, description="Specific workflow ID for analytics")
    module: Optional[str] = Field(None, description="Module for analytics")
    time_range: str = Field("24h", description="Time range (1h, 24h, 7d, 30d)")
    tenant_id: Optional[int] = Field(None, description="Tenant ID for scoped analytics")

# Hub Router
hub_router = APIRouter(prefix="/api/hub", tags=["Hub Architecture"])

# Global hub instance (will be initialized by main app)
_execution_hub: Optional[ExecutionHub] = None

def get_execution_hub() -> ExecutionHub:
    """Dependency to get the execution hub instance"""
    if _execution_hub is None:
        raise HTTPException(status_code=500, detail="Execution Hub not initialized")
    return _execution_hub

def set_execution_hub(hub: ExecutionHub):
    """Set the execution hub instance"""
    global _execution_hub
    _execution_hub = hub

# ============================================================================
# WORKFLOW DISCOVERY ENDPOINTS
# ============================================================================

@hub_router.get("/workflows/discover")
async def discover_workflows_get(
    module: Optional[str] = None,
    automation_type: Optional[str] = None,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Discover available workflows (GET endpoint for easy testing)
    """
    try:
        # Direct database query approach - bypass complex registry logic
        tenant_id = int(os.getenv('MAIN_TENANT_ID', '1300'))
        
        # Get pool manager from hub
        pool_manager = hub.pool_manager
        if not pool_manager or not pool_manager.postgres_pool:
            raise Exception("Database connection not available")
        
        # Build query with filters
        where_conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1
        
        if module:
            param_count += 1
            where_conditions.append(f"module = ${param_count}")
            params.append(module)
            
        if automation_type:
            param_count += 1
            where_conditions.append(f"automation_type = ${param_count}")
            params.append(automation_type)
        
        where_clause = " AND ".join(where_conditions)
        
        # Execute direct database query
        async with pool_manager.postgres_pool.acquire() as conn:
            query = f"""
                SELECT 
                    workflow_id, name, description, module, automation_type,
                    version, status, industry, sla_tier, tags,
                    execution_count, success_rate, avg_execution_time_ms,
                    created_at, updated_at
                FROM dsl_workflows
                WHERE {where_clause}
                ORDER BY updated_at DESC
            """
            
            rows = await conn.fetch(query, *params)
            
            # Convert to workflow format
            workflows = []
            for row in rows:
                # Parse tags if they're stored as JSON string
                tags = row['tags'] if row['tags'] else []
                if isinstance(tags, str):
                    try:
                        import json
                        tags = json.loads(tags) if tags else []
                    except:
                        tags = []
                
                workflow = {
                    "workflow_id": row['workflow_id'],
                    "name": row['name'],
                    "module": row['module'],
                    "automation_type": row['automation_type'],
                    "version": row['version'],
                    "business_value": "Automates business processes efficiently",
                    "customer_impact": "Improves operational efficiency and accuracy",
                    "tags": tags,
                    "success_rate": float(row['success_rate']) if row['success_rate'] else 0.0,
                    "avg_execution_time_ms": int(row['avg_execution_time_ms']) if row['avg_execution_time_ms'] else 0,
                    "usage_count": int(row['execution_count']) if row['execution_count'] else 0,
                    "last_executed": None,
                    "status": row['status'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                }
                workflows.append(workflow)
        
        result = {
            "workflows": workflows,
            "suggestions": [],
            "total_count": len(workflows)
        }
        
        return {
            "success": True,
            "workflows": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Workflow discovery error: {e}")
        return {
            "success": False,
            "error": str(e),
            "workflows": {"workflows": [], "suggestions": [], "total_count": 0},
            "timestamp": datetime.utcnow().isoformat()
        }

@hub_router.post("/workflows/discover")
async def discover_workflows(
    request: WorkflowDiscoveryRequest,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Discover available workflows based on module, automation type, and user context
    """
    try:
        workflows = await hub.discover_workflows(
            module=request.module,
            automation_type=request.automation_type,
            user_context=request.user_context
        )
        
        return {
            "success": True,
            "data": workflows,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error discovering workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/workflows/{workflow_id}/details")
async def get_workflow_details(
    workflow_id: str,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get detailed information about a specific workflow
    """
    try:
        details = await hub.get_workflow_details(workflow_id)
        
        if not details:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "success": True,
            "data": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/workflows/recommendations")
async def get_workflow_recommendations(
    user_id: int,
    tenant_id: str,
    role: str = "user",
    limit: int = 5,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get personalized workflow recommendations for a user
    """
    try:
        user_context = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "role": role
        }
        
        # Use knowledge connector for recommendations
        if hasattr(hub, 'knowledge_connector'):
            recommendations = await hub.knowledge_connector.get_workflow_recommendations(user_context)
        else:
            recommendations = []
        
        return {
            "success": True,
            "data": recommendations[:limit],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WORKFLOW EXECUTION ENDPOINTS
# ============================================================================

@hub_router.post("/workflows/execute")
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Execute a workflow through the Hub Architecture
    """
    try:
        # Validate user context
        if not request.user_context.get('user_id') or not request.user_context.get('tenant_id'):
            raise HTTPException(status_code=400, detail="user_id and tenant_id required in user_context")
        
        # Execute workflow
        result = await hub.execute_workflow(
            workflow_id=request.workflow_id,
            user_context=request.user_context,
            input_data=request.input_data,
            priority=request.priority
        )
        
        # Capture knowledge in background
        if result.get('success'):
            background_tasks.add_task(hub.capture_execution_knowledge, result)
        
        # Store analytics data in background
        background_tasks.add_task(_store_execution_analytics, request, result, hub)
        
        return {
            "success": result.get('success', False),
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/workflows/execution/{request_id}/status")
async def get_execution_status(
    request_id: str,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get status of a workflow execution
    """
    try:
        status = await hub.get_execution_status(request_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Execution {request_id} not found")
        
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RBA AGENT MANAGEMENT ENDPOINTS
# ============================================================================

@hub_router.post("/agents/register")
async def register_rba_agent(
    request: RBAAgentRegistrationRequest,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Register a new RBA agent in the hub
    """
    try:
        agent_config = {
            "name": request.name,
            "module": request.module,
            "workflow_file": request.workflow_file,
            "business_value": request.business_value,
            "customer_impact": request.customer_impact,
            "industry_tags": request.industry_tags,
            "persona_tags": request.persona_tags,
            "estimated_time_minutes": request.estimated_time_minutes
        }
        
        success = await hub.register_rba_agent(request.agent_id, agent_config)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register RBA agent")
        
        return {
            "success": True,
            "message": f"RBA Agent {request.agent_id} registered successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error registering RBA agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/agents")
async def get_rba_agents(
    module: Optional[str] = None,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get list of registered RBA agents
    """
    try:
        agents = await hub.get_rba_agents(module=module)
        
        return {
            "success": True,
            "data": agents,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting RBA agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.post("/agents/{agent_id}/execute")
async def execute_rba_agent(
    agent_id: str,
    user_context: Dict[str, Any],
    input_data: Dict[str, Any] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Execute a specific RBA agent
    """
    try:
        result = await hub.execute_rba_agent(
            agent_id=agent_id,
            user_context=user_context,
            input_data=input_data or {}
        )
        
        # Capture knowledge in background
        if result.get('success'):
            background_tasks.add_task(hub.capture_execution_knowledge, result)
        
        return {
            "success": result.get('success', False),
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing RBA agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ANALYTICS AND MONITORING ENDPOINTS
# ============================================================================

@hub_router.get("/analytics/dashboard")
async def get_hub_analytics(
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get comprehensive hub analytics for dashboard
    """
    try:
        analytics = await hub.get_hub_analytics()
        
        return {
            "success": True,
            "data": analytics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting hub analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/analytics/module/{module}")
async def get_module_performance(
    module: str,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get performance analytics for a specific module
    """
    try:
        if module not in ['Forecast', 'Pipeline', 'Planning']:
            raise HTTPException(status_code=400, detail="Invalid module. Must be Forecast, Pipeline, or Planning")
        
        performance = await hub.get_module_performance(module)
        
        return {
            "success": True,
            "data": performance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting module performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/analytics/knowledge")
async def get_knowledge_analytics(
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get knowledge graph and analytics data
    """
    try:
        # Get real Knowledge Graph statistics
        tenant_id = int(os.getenv('MAIN_TENANT_ID', '1300'))
        stats = await hub.kg_store.get_knowledge_stats(tenant_id)
        
        knowledge_data = {
            "total_entities": stats.get('total_entities', 0),
            "total_relationships": stats.get('total_relationships', 0),
            "entity_counts": stats.get('entity_counts', {}),
            "relationship_counts": stats.get('relationship_counts', {}),
            "recent_activity": stats.get('recent_activity', {}),
            "knowledge_graph_status": "PostgreSQL + pgvector - Fully operational",
            "knowledge_ingestion_rate": "Real-time"
        }
        
        return {
            "success": True,
            "data": knowledge_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/analytics/workflow/{workflow_id}")
async def get_workflow_analytics(
    workflow_id: str,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get detailed analytics for a specific workflow
    """
    try:
        analytics = await hub.registry.get_workflow_analytics(workflow_id)
        
        return {
            "success": True,
            "data": analytics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/orchestrator/status")
async def get_orchestrator_status(
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get real-time orchestrator status
    """
    try:
        status = await hub.orchestrator.get_queue_status()
        metrics = await hub.orchestrator.get_performance_metrics()
        
        return {
            "success": True,
            "data": {
                "status": status,
                "metrics": metrics
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# KNOWLEDGE AND INTELLIGENCE ENDPOINTS
# ============================================================================

@hub_router.get("/knowledge/patterns")
async def get_execution_patterns(
    module: Optional[str] = None,
    pattern_type: Optional[str] = None,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get discovered execution patterns
    """
    try:
        # This will be implemented when Knowledge Connector is fully integrated
        patterns = []  # await hub.knowledge_connector.discover_patterns(module=module)
        
        return {
            "success": True,
            "data": patterns,
            "message": "Pattern discovery will be enhanced with Knowledge Graph implementation",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting execution patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/knowledge/insights/workflow/{workflow_id}")
async def get_workflow_insights(
    workflow_id: str,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get AI-powered insights for a workflow
    """
    try:
        # This will be implemented when Knowledge Connector is fully integrated
        insights = {"message": "Workflow insights will be available with Knowledge Graph implementation"}
        
        return {
            "success": True,
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hub_router.get("/knowledge/insights/user/{user_id}")
async def get_user_behavior_insights(
    user_id: int,
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Get behavior insights for a user
    """
    try:
        # This will be implemented when Knowledge Connector is fully integrated
        insights = {"message": "User behavior insights will be available with Knowledge Graph implementation"}
        
        return {
            "success": True,
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting user insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SYSTEM HEALTH AND MANAGEMENT
# ============================================================================

@hub_router.get("/health")
async def hub_health_check(
    hub: ExecutionHub = Depends(get_execution_hub)
):
    """
    Comprehensive health check of the Hub Architecture
    """
    try:
        health = await hub.health_check()
        
        return {
            "success": True,
            "data": health,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting hub health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _store_execution_analytics(
    request: WorkflowExecutionRequest, 
    result: Dict[str, Any], 
    hub: ExecutionHub
):
    """
    Store execution analytics in the database (background task)
    """
    try:
        if not hub.pool_manager or not hub.pool_manager.postgres_pool:
            return
        
        # Calculate data hashes for pattern matching
        input_hash = hashlib.sha256(json.dumps(request.input_data, sort_keys=True).encode()).hexdigest()
        output_hash = hashlib.sha256(json.dumps(result.get('result', {}), sort_keys=True).encode()).hexdigest()
        
        async with hub.pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflow_execution_analytics (
                    workflow_id, request_id, user_id, tenant_id, execution_status,
                    execution_time_ms, priority_level, input_data_hash, output_data_hash,
                    user_context, execution_context, error_details, evidence_pack_id,
                    requested_at, started_at, completed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                request.workflow_id,
                result.get('request_id'),
                request.user_context.get('user_id'),
                request.user_context.get('tenant_id'),
                'completed' if result.get('success') else 'failed',
                result.get('execution_time_ms', 0),
                request.priority,
                input_hash,
                output_hash,
                json.dumps(request.user_context),
                json.dumps({"priority": request.priority}),
                json.dumps({"error": result.get('error_message')}) if result.get('error_message') else None,
                result.get('evidence_pack_id'),
                datetime.fromisoformat(result.get('started_at').replace('Z', '+00:00')) if result.get('started_at') else datetime.utcnow(),
                datetime.fromisoformat(result.get('started_at').replace('Z', '+00:00')) if result.get('started_at') else datetime.utcnow(),
                datetime.fromisoformat(result.get('completed_at').replace('Z', '+00:00')) if result.get('completed_at') else datetime.utcnow()
            )
        
        logger.info(f"Stored execution analytics for workflow {request.workflow_id}")
        
    except Exception as e:
        logger.error(f"Error storing execution analytics: {e}")

# ============================================================================
# HUB INITIALIZATION
# ============================================================================

async def initialize_hub(pool_manager) -> ExecutionHub:
    """
    Initialize the Execution Hub with all components
    """
    try:
        hub = ExecutionHub(pool_manager=pool_manager)
        await hub.initialize()
        
        # Set the global hub instance
        set_execution_hub(hub)
        logger.info("🚀 Hub Architecture initialized successfully")
        return hub
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Hub Architecture: {e}")
        raise
