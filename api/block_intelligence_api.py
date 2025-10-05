"""
Block Intelligence API
=====================

Task 7.3.1: Build block intelligence engine
API endpoints for AI-powered block recommendations and workflow optimization
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Import block intelligence engine
try:
    from ..dsl.intelligence.block_intelligence_engine import (
        get_block_intelligence_engine,
        WorkflowIntelligenceReport,
        BlockRecommendation,
        RecommendationType,
        RecommendationPriority
    )
except ImportError:
    # Fallback if block intelligence engine is not available
    def get_block_intelligence_engine():
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
class WorkflowAnalysisRequest(BaseModel):
    """Request model for workflow intelligence analysis"""
    workflow_definition: Dict[str, Any] = Field(..., description="DSL workflow definition")
    tenant_id: int = Field(..., description="Tenant identifier")
    include_execution_history: bool = Field(True, description="Include historical execution data")
    industry_context: Optional[str] = Field(None, description="Industry context (SaaS, Banking, etc.)")
    analysis_depth: str = Field("standard", description="Analysis depth: basic, standard, comprehensive")

class BlockRecommendationResponse(BaseModel):
    """Response model for block recommendations"""
    recommendation_id: str
    recommendation_type: str
    priority: str
    title: str
    description: str
    target_block_id: Optional[str] = None
    suggested_block_type: Optional[str] = None
    suggested_configuration: Optional[Dict[str, Any]] = None
    estimated_impact: str
    effort_required: str
    confidence_score: float
    reasoning: List[str]
    implementation_steps: List[str]
    related_blocks: List[str]
    compliance_impact: List[str]
    performance_impact: Dict[str, Any]
    created_at: datetime

class WorkflowIntelligenceResponse(BaseModel):
    """Response model for workflow intelligence analysis"""
    workflow_id: str
    overall_score: float
    recommendations: List[BlockRecommendationResponse]
    workflow_patterns: Dict[str, Any]
    optimization_summary: Dict[str, Any]
    compliance_summary: Dict[str, Any]
    performance_summary: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    generated_at: datetime

class RecommendationFilterRequest(BaseModel):
    """Request model for filtering recommendations"""
    tenant_id: int = Field(..., description="Tenant identifier")
    workflow_ids: Optional[List[str]] = Field(None, description="Filter by workflow IDs")
    recommendation_types: Optional[List[str]] = Field(None, description="Filter by recommendation types")
    priorities: Optional[List[str]] = Field(None, description="Filter by priorities")
    min_confidence_score: float = Field(0.0, description="Minimum confidence score")
    limit: int = Field(50, description="Maximum number of recommendations")

class OptimizationInsightsResponse(BaseModel):
    """Response model for optimization insights"""
    total_workflows_analyzed: int
    avg_workflow_score: float
    top_optimization_opportunities: List[Dict[str, Any]]
    common_anti_patterns: List[Dict[str, Any]]
    industry_benchmarks: Dict[str, Any]
    recommendations_by_category: Dict[str, int]
    estimated_improvement_potential: float

@router.post("/analyze-workflow", response_model=WorkflowIntelligenceResponse)
async def analyze_workflow_intelligence(
    request: WorkflowAnalysisRequest,
    pool_manager=Depends(get_pool_manager)
):
    """
    Perform comprehensive intelligence analysis of a DSL workflow
    
    Analyzes workflow patterns, performance, compliance, and generates
    AI-powered recommendations for optimization
    """
    try:
        logger.info(f"🧠 Starting workflow intelligence analysis for tenant {request.tenant_id}")
        
        # Get block intelligence engine
        intelligence_engine = get_block_intelligence_engine()
        if not intelligence_engine:
            raise HTTPException(status_code=500, detail="Block intelligence engine not available")
        
        # Set pool manager
        intelligence_engine.pool_manager = pool_manager
        
        # Get execution history if requested
        execution_history = None
        if request.include_execution_history:
            execution_history = await _get_workflow_execution_history(
                request.workflow_definition.get('workflow_id'),
                request.tenant_id,
                pool_manager
            )
        
        # Perform intelligence analysis
        report = await intelligence_engine.analyze_workflow_intelligence(
            workflow_definition=request.workflow_definition,
            tenant_id=request.tenant_id,
            execution_history=execution_history,
            industry_context=request.industry_context
        )
        
        # Convert recommendations to response format
        recommendations_response = []
        for rec in report.recommendations:
            recommendations_response.append(BlockRecommendationResponse(
                recommendation_id=rec.recommendation_id,
                recommendation_type=rec.recommendation_type.value,
                priority=rec.priority.value,
                title=rec.title,
                description=rec.description,
                target_block_id=rec.target_block_id,
                suggested_block_type=rec.suggested_block_type,
                suggested_configuration=rec.suggested_configuration,
                estimated_impact=rec.estimated_impact,
                effort_required=rec.effort_required,
                confidence_score=rec.confidence_score,
                reasoning=rec.reasoning,
                implementation_steps=rec.implementation_steps,
                related_blocks=rec.related_blocks,
                compliance_impact=rec.compliance_impact,
                performance_impact=rec.performance_impact,
                created_at=rec.created_at
            ))
        
        # Create analysis metadata
        analysis_metadata = {
            "analysis_depth": request.analysis_depth,
            "industry_context": request.industry_context,
            "execution_history_included": request.include_execution_history,
            "total_blocks_analyzed": len(report.block_analyses),
            "analysis_duration_ms": (datetime.utcnow() - report.generated_at).total_seconds() * 1000,
            "engine_version": "1.0.0"
        }
        
        logger.info(f"✅ Workflow intelligence analysis completed: {report.overall_score:.2f} score, "
                   f"{len(report.recommendations)} recommendations")
        
        return WorkflowIntelligenceResponse(
            workflow_id=report.workflow_id,
            overall_score=report.overall_score,
            recommendations=recommendations_response,
            workflow_patterns=report.workflow_patterns,
            optimization_summary=report.optimization_summary,
            compliance_summary=report.compliance_summary,
            performance_summary=report.performance_summary,
            analysis_metadata=analysis_metadata,
            generated_at=report.generated_at
        )
        
    except Exception as e:
        logger.error(f"❌ Workflow intelligence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/recommendations/filter", response_model=List[BlockRecommendationResponse])
async def filter_recommendations(
    request: RecommendationFilterRequest,
    pool_manager=Depends(get_pool_manager)
):
    """
    Filter and retrieve recommendations based on criteria
    
    Allows filtering recommendations by workflow, type, priority, and confidence
    """
    try:
        logger.info(f"🔍 Filtering recommendations for tenant {request.tenant_id}")
        
        # Get stored recommendations from database
        recommendations = await _get_stored_recommendations(
            tenant_id=request.tenant_id,
            workflow_ids=request.workflow_ids,
            recommendation_types=request.recommendation_types,
            priorities=request.priorities,
            min_confidence_score=request.min_confidence_score,
            limit=request.limit,
            pool_manager=pool_manager
        )
        
        logger.info(f"✅ Retrieved {len(recommendations)} filtered recommendations")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"❌ Failed to filter recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Filter failed: {str(e)}")

@router.get("/optimization-insights/{tenant_id}", response_model=OptimizationInsightsResponse)
async def get_optimization_insights(
    tenant_id: int,
    time_range_days: int = Query(30, description="Time range for analysis in days"),
    include_benchmarks: bool = Query(True, description="Include industry benchmarks"),
    pool_manager=Depends(get_pool_manager)
):
    """
    Get optimization insights and trends for a tenant
    
    Provides high-level insights about workflow optimization opportunities
    """
    try:
        logger.info(f"📊 Getting optimization insights for tenant {tenant_id}")
        
        # Get intelligence engine
        intelligence_engine = get_block_intelligence_engine()
        if not intelligence_engine:
            raise HTTPException(status_code=500, detail="Block intelligence engine not available")
        
        # Get tenant workflows and analyses
        insights = await _generate_optimization_insights(
            tenant_id=tenant_id,
            time_range_days=time_range_days,
            include_benchmarks=include_benchmarks,
            intelligence_engine=intelligence_engine,
            pool_manager=pool_manager
        )
        
        logger.info(f"✅ Generated optimization insights for {insights['total_workflows_analyzed']} workflows")
        
        return OptimizationInsightsResponse(**insights)
        
    except Exception as e:
        logger.error(f"❌ Failed to get optimization insights: {e}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")

@router.post("/recommendations/{recommendation_id}/implement")
async def implement_recommendation(
    recommendation_id: str,
    tenant_id: int = Body(..., description="Tenant identifier"),
    auto_apply: bool = Body(False, description="Automatically apply the recommendation"),
    pool_manager=Depends(get_pool_manager)
):
    """
    Implement a specific recommendation
    
    Provides implementation guidance and optionally applies the recommendation
    """
    try:
        logger.info(f"🔧 Implementing recommendation {recommendation_id} for tenant {tenant_id}")
        
        # Get recommendation details
        recommendation = await _get_recommendation_by_id(
            recommendation_id, tenant_id, pool_manager
        )
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Generate implementation plan
        implementation_plan = await _generate_implementation_plan(recommendation)
        
        result = {
            "recommendation_id": recommendation_id,
            "implementation_plan": implementation_plan,
            "auto_applied": False,
            "next_steps": recommendation.get('implementation_steps', []),
            "estimated_effort": recommendation.get('effort_required', 'medium'),
            "estimated_impact": recommendation.get('estimated_impact', 'medium')
        }
        
        # Auto-apply if requested and safe
        if auto_apply and recommendation.get('auto_applicable', False):
            apply_result = await _auto_apply_recommendation(recommendation, tenant_id, pool_manager)
            result.update({
                "auto_applied": True,
                "apply_result": apply_result
            })
        
        logger.info(f"✅ Implementation plan generated for recommendation {recommendation_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to implement recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Implementation failed: {str(e)}")

@router.get("/block-patterns")
async def get_block_patterns():
    """
    Get available block patterns and anti-patterns
    
    Returns the library of optimal patterns and anti-patterns used for analysis
    """
    try:
        intelligence_engine = get_block_intelligence_engine()
        if not intelligence_engine:
            raise HTTPException(status_code=500, detail="Block intelligence engine not available")
        
        return {
            "optimal_patterns": intelligence_engine.block_patterns.get('optimal_patterns', {}),
            "anti_patterns": intelligence_engine.block_patterns.get('anti_patterns', {}),
            "pattern_count": len(intelligence_engine.block_patterns.get('optimal_patterns', {})),
            "anti_pattern_count": len(intelligence_engine.block_patterns.get('anti_patterns', {})),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get block patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get patterns: {str(e)}")

@router.get("/performance-benchmarks")
async def get_performance_benchmarks():
    """
    Get performance benchmarks for different block types
    
    Returns the performance benchmarks used for analysis
    """
    try:
        intelligence_engine = get_block_intelligence_engine()
        if not intelligence_engine:
            raise HTTPException(status_code=500, detail="Block intelligence engine not available")
        
        return {
            "benchmarks": intelligence_engine.performance_benchmarks,
            "benchmark_count": len(intelligence_engine.performance_benchmarks),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get benchmarks: {str(e)}")

@router.get("/health")
async def block_intelligence_health_check():
    """
    Health check for block intelligence service
    """
    try:
        intelligence_engine = get_block_intelligence_engine()
        
        return {
            "status": "healthy" if intelligence_engine else "unhealthy",
            "service": "Block Intelligence API",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "workflow_analysis": intelligence_engine is not None,
                "recommendation_filtering": intelligence_engine is not None,
                "optimization_insights": intelligence_engine is not None,
                "pattern_matching": intelligence_engine is not None,
                "performance_benchmarking": intelligence_engine is not None
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Helper functions

async def _get_workflow_execution_history(
    workflow_id: Optional[str],
    tenant_id: int,
    pool_manager
) -> Optional[List[Dict[str, Any]]]:
    """Get execution history for a workflow"""
    
    if not pool_manager or not workflow_id:
        return None
    
    try:
        async with pool_manager.get_connection() as conn:
            query = """
                SELECT execution_id, workflow_id, trace_data, execution_status,
                       execution_time_ms, trust_score, created_at
                FROM dsl_execution_traces 
                WHERE workflow_id = $1 AND tenant_id = $2
                ORDER BY created_at DESC 
                LIMIT 100
            """
            
            rows = await conn.fetch(query, workflow_id, tenant_id)
            
            history = []
            for row in rows:
                trace_data = row['trace_data'] if isinstance(row['trace_data'], dict) else {}
                
                # Extract block-level execution data
                for step in trace_data.get('steps', []):
                    history.append({
                        'execution_id': row['execution_id'],
                        'workflow_id': row['workflow_id'],
                        'block_id': step.get('step_id'),
                        'execution_time_ms': step.get('execution_time_ms', 0),
                        'memory_usage_mb': step.get('memory_usage_mb', 0),
                        'status': step.get('status', 'unknown'),
                        'error_message': step.get('error_message'),
                        'timestamp': row['created_at'].isoformat(),
                        'parameters': step.get('inputs', {})
                    })
            
            return history
            
    except Exception as e:
        logger.error(f"Failed to get execution history: {e}")
        return None

async def _get_stored_recommendations(
    tenant_id: int,
    workflow_ids: Optional[List[str]] = None,
    recommendation_types: Optional[List[str]] = None,
    priorities: Optional[List[str]] = None,
    min_confidence_score: float = 0.0,
    limit: int = 50,
    pool_manager=None
) -> List[BlockRecommendationResponse]:
    """Get stored recommendations from database"""
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Query the database for stored recommendations
    # 2. Apply filters
    # 3. Return formatted results
    
    # For now, return empty list
    return []

async def _generate_optimization_insights(
    tenant_id: int,
    time_range_days: int,
    include_benchmarks: bool,
    intelligence_engine,
    pool_manager
) -> Dict[str, Any]:
    """Generate optimization insights for a tenant"""
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Analyze all tenant workflows
    # 2. Aggregate recommendations
    # 3. Calculate trends and insights
    # 4. Compare against benchmarks
    
    return {
        "total_workflows_analyzed": 0,
        "avg_workflow_score": 0.8,
        "top_optimization_opportunities": [],
        "common_anti_patterns": [],
        "industry_benchmarks": {},
        "recommendations_by_category": {},
        "estimated_improvement_potential": 0.2
    }

async def _get_recommendation_by_id(
    recommendation_id: str,
    tenant_id: int,
    pool_manager
) -> Optional[Dict[str, Any]]:
    """Get a specific recommendation by ID"""
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Query the database for the recommendation
    # 2. Validate tenant access
    # 3. Return recommendation details
    
    return None

async def _generate_implementation_plan(recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """Generate implementation plan for a recommendation"""
    
    return {
        "steps": recommendation.get('implementation_steps', []),
        "prerequisites": [],
        "estimated_duration": "1-2 hours",
        "risk_level": "low",
        "rollback_plan": "Revert configuration changes",
        "validation_steps": [
            "Test workflow execution",
            "Verify performance improvement",
            "Check compliance status"
        ]
    }

async def _auto_apply_recommendation(
    recommendation: Dict[str, Any],
    tenant_id: int,
    pool_manager
) -> Dict[str, Any]:
    """Auto-apply a recommendation if safe"""
    
    return {
        "applied": False,
        "reason": "Auto-application not implemented yet",
        "manual_steps_required": True
    }
