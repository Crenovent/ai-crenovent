"""
Pipeline Agents API
FastAPI endpoints for dynamic pipeline agent execution
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .models import PipelineAgentRequest, clean_for_json_serialization
from .csv_data_loader import crm_data_loader
from dsl.compiler.runtime import WorkflowRuntime
from dsl.orchestration.dynamic_rba_orchestrator import dynamic_rba_orchestrator
from dsl.observability.metrics_collector import get_metrics_collector
from dsl.registry.enhanced_capability_registry import EnhancedCapabilityRegistry
from dsl.hub.intent_parser_v2 import CleanIntentParser
from src.services.connection_pool_manager import ConnectionPoolManager
import numpy as np
from src.services.connection_pool_manager import pool_manager

logger = logging.getLogger(__name__)

# Create pipeline agents router
router = APIRouter(prefix="/api/pipeline", tags=["Pipeline Agents"])


@router.get("/agents")
async def list_available_agents():
    """
    List all available RBA agents dynamically from the registry
    
    Returns:
    - RBA Agents: All available Rule-Based Automation agents
    - Analysis Types: All supported analysis types
    - Agent Metadata: Complete agent information
    """
    try:
        # Initialize the RBA registry
        registry = EnhancedCapabilityRegistry()
        await registry.initialize()
        
        # Get all RBA agents
        rba_agents = []
        all_agents = await registry.get_all_agents()
        
        for agent_name, agent_info in all_agents.items():
            agent_data = {
                "name": agent_name,
                "description": agent_info.agent_description,
                "type": agent_info.agent_type,
                "supported_analysis_types": agent_info.supported_analysis_types,
                "class_name": agent_info.class_name,
                "module_path": agent_info.module_path,
                "priority": agent_info.priority
            }
            rba_agents.append(agent_data)
        
        # Get supported analysis types
        supported_analysis_types = await registry.get_supported_analysis_types()
        
        # Get registry stats
        registry_stats = await registry.get_registry_stats()
        
        return {
            "success": True,
            "total_agents": registry_stats['total_agents'],
            "total_analysis_types": registry_stats['total_analysis_types'],
            "agents": {
                "rba_agents": rba_agents
            },
            "supported_analysis_types": supported_analysis_types,
            "registry_stats": registry_stats,
            "message": f"Found {registry_stats['total_agents']} RBA agents supporting {registry_stats['total_analysis_types']} analysis types"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.post("/test")
async def test_pipeline_agents():
    """
    Test endpoint to verify the dynamic pipeline agent system
    
    Runs a series of test prompts to demonstrate:
    1. Data quality audit
    2. Pipeline summary 
    3. Risk analysis
    4. Coaching insights
    """
    try:
        test_prompts = [
            "Audit pipeline data quality - check for missing close dates and amounts",
            "Get my pipeline summary with total value and deal count",
            "Identify deals at risk and calculate risk scores", 
            "Generate coaching insights for stalled deals"
        ]
        
        results = []
        
        for prompt in test_prompts:
            try:
                # Execute each test prompt
                request = PipelineAgentRequest(
                    user_input=prompt,
                    context={"test_mode": True},
                    tenant_id="1300",
                    user_id="1319"
                )
                
                result = await execute_pipeline_agent(request)
                results.append({
                    "prompt": prompt,
                    "success": result.get("success", False),
                    "result": result
                })
                
            except Exception as prompt_error:
                results.append({
                    "prompt": prompt,
                    "success": False,
                    "error": str(prompt_error)
                })
        
        successful_tests = len([r for r in results if r.get("success", False)])
        
        return {
            "success": successful_tests > 0,
            "message": f"Completed {len(test_prompts)} test prompts, {successful_tests} successful",
            "test_results": results,
            "summary": {
                "total_tests": len(test_prompts),
                "successful": successful_tests,
                "failed": len(test_prompts) - successful_tests
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline agent testing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Testing failed: {str(e)}")


@router.post("/execute")
async def execute_pipeline_agent(request: PipelineAgentRequest, fastapi_request: Request = None):
    """
    🤖 Execute Dynamic RBA Analysis using Modular Agent Architecture
    
    Features:
    - 🎯 Dynamic agent selection based on analysis type
    - 📊 Real-time opportunity data processing
    - 🔍 29 different analysis types supported
    - 🚀 Modular, extensible architecture
    """
    try:
        logger.info(f"🤖 RBA Pipeline execution request: {request.user_input}")
        logger.info(f"📋 Request context received: {request.context}")
        
        # Parse the user input to determine analysis type and extract config
        analysis_type = await parse_analysis_type_from_input(request.user_input)
        prompt_config = extract_config_from_prompt(request.user_input)
        
        # Try AI-powered configuration extraction as fallback (only if regex failed)
        if not prompt_config:
            ai_config = await extract_config_with_ai(request.user_input, analysis_type)
            prompt_config.update(ai_config)
        
        # Apply smart defaults and contextual hints based on analysis type
        prompt_config = apply_smart_defaults(prompt_config, analysis_type, request.user_input)
        
        # Debug: Show what we extracted
        logger.info(f"🔍 Final prompt config after all extractions: {prompt_config}")
        
        logger.info(f"🔍 Extracted config from prompt: {prompt_config}")
        
        if not analysis_type:
            return {
                "success": False,
                "error": "Could not determine analysis type from input",
                "available_analysis_types": dynamic_rba_orchestrator.get_available_analysis_types(),
                "suggestion": "Try specifying one of the supported analysis types"
            }
        
        # Get opportunity data from CSV file
        opportunities = await get_csv_opportunity_data(request.tenant_id, request.user_id)
        
        # Execute through dynamic RBA orchestrator
        start_time = datetime.now()
        
        # Extract workflow_config from context (frontend sends it nested)
        ui_config = (request.context or {}).get('workflow_config', {})
        
        # Merge UI config with prompt-extracted config (prompt takes precedence)
        user_config = {**ui_config, **prompt_config}
        
        logger.info(f"📋 Raw request.context: {request.context}")
        logger.info(f"📋 UI config: {ui_config}")
        logger.info(f"📋 Prompt config: {prompt_config}")
        logger.info(f"📋 Final user config: {user_config}")
        logger.info(f"🔍 Looking for high_value_threshold in user_config: {user_config.get('high_value_threshold', 'NOT_FOUND')}")
        
        # Merge user-provided config with defaults from business rules
        merged_config = await _merge_user_config_with_defaults(analysis_type, user_config)
        
        logger.info(f"🔧 Using config for {analysis_type}: {merged_config}")
        
        # Get KG store from app state if available for execution tracing
        kg_store = getattr(fastapi_request.app.state, 'kg_store', None) if fastapi_request else None
        
        result = await dynamic_rba_orchestrator.execute_rba_analysis(
            analysis_type=analysis_type,
            opportunities=opportunities,
            config=merged_config,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            kg_store=kg_store  # Enable Knowledge Graph tracing
        )
        
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ RBA analysis completed in {execution_time_ms:.2f}ms")
        
        # Return structured response compatible with frontend modal expectations
        return {
            "success": result.get('success', False),
            "analysis_type": analysis_type,
            "execution_time_ms": execution_time_ms,
            "agent_used": result.get('orchestration_metadata', {}).get('agent_selected', 'Unknown'),
            "total_opportunities": result.get('total_opportunities', 0),
            "flagged_opportunities": result.get('flagged_opportunities', 0),
            "results": clean_for_json_serialization(result),
            "orchestration_metadata": result.get('orchestration_metadata', {}),
            "user_input": request.user_input,
            "timestamp": datetime.now().isoformat(),
            
            # Frontend compatibility: Add routing_result structure for modal display
            "routing_result": {
                "workflow_category": analysis_type,
                "workflow_name": f"{analysis_type.replace('_', ' ').title()} Analysis",
                "execution_result": clean_for_json_serialization(result),
                "execution_time": execution_time_ms,
                "success": result.get('success', False)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ RBA Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.post("/execute-dsl-workflow")
async def execute_dsl_workflow_endpoint(
    workflow_dsl: Dict[str, Any],
    input_data: Dict[str, Any] = {},
    tenant_id: str = "1300",
    user_id: str = "1323",
    fastapi_request: Request = None
):
    """
    Execute DSL workflow with enhanced governance and idempotency
    
    This endpoint integrates the new RBA Build Plan components:
    - DSL Parser integration
    - Runtime execution engine  
    - Idempotency framework
    - Governance-first design
    
    Args:
        workflow_dsl: DSL workflow definition (YAML or dict)
        input_data: Input data for workflow execution
        tenant_id: Tenant identifier for isolation
        user_id: User identifier
        
    Returns:
        Enhanced workflow execution results with governance metadata
    """
    try:
        logger.info(f"🚀 [Enhanced RBA] Executing DSL workflow for tenant {tenant_id}")
        
        # Initialize enhanced orchestrator if needed
        if not hasattr(dynamic_rba_orchestrator, 'workflow_runtime') or not dynamic_rba_orchestrator.workflow_runtime:
            await dynamic_rba_orchestrator.initialize_enhanced_components(pool_manager)
        
        # Create user context
        user_context = {
            'tenant_id': tenant_id,
            'user_id': user_id,
            'execution_timestamp': datetime.now().isoformat(),
            'source': 'api_endpoint',
            'governance_required': True,
            'evidence_capture': True
        }
        
        # Execute DSL workflow with enhanced capabilities
        result = await dynamic_rba_orchestrator.execute_dsl_workflow(
            workflow_dsl=workflow_dsl,
            input_data=input_data,
            user_context=user_context,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        logger.info(f"✅ [Enhanced RBA] DSL workflow execution completed: {result.get('execution_id')}")
        
        return {
            "success": result.get('success', True),
            "execution_id": result.get('execution_id'),
            "workflow_id": result.get('workflow_id'),
            "result_data": result.get('result_data', {}),
            "execution_time_ms": result.get('execution_time_ms', 0),
            "evidence_pack_id": result.get('evidence_pack_id'),
            "orchestrator_type": result.get('orchestrator_type', 'dynamic_rba_enhanced'),
            "governance_metadata": {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "policy_compliance": True,
                "evidence_captured": bool(result.get('evidence_pack_id')),
                "audit_trail_available": True
            },
            "error_message": result.get('error_message'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ [Enhanced RBA] DSL workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced DSL workflow execution failed: {str(e)}")

@router.get("/metrics")
async def get_observability_metrics(
    tenant_id: Optional[int] = None,
    time_window_minutes: int = 60,
    format: str = "json"  # json or prometheus
):
    """
    Get observability metrics for monitoring dashboards (Task 20.1-T16 to 20.1-T20)
    
    Args:
        tenant_id: Optional tenant filter
        time_window_minutes: Time window for metrics aggregation
        format: Output format (json or prometheus)
    
    Returns:
        Aggregated metrics in requested format
    """
    try:
        metrics_collector = get_metrics_collector()
        
        if format == "prometheus":
            # Return Prometheus format for scraping
            prometheus_metrics = metrics_collector.export_prometheus_metrics()
            return Response(
                content=prometheus_metrics,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        else:
            # Return JSON format for dashboards
            aggregated_metrics = metrics_collector.get_aggregated_metrics(
                tenant_id=tenant_id,
                time_window_minutes=time_window_minutes
            )
            
            # Add tenant-specific summary if tenant_id provided
            tenant_summary = None
            if tenant_id:
                tenant_summary = metrics_collector.get_tenant_metrics_summary(tenant_id)
            
            return {
                "success": True,
                "metrics": aggregated_metrics,
                "tenant_summary": tenant_summary,
                "time_window_minutes": time_window_minutes,
                "timestamp": datetime.now().isoformat(),
                "total_metrics_collected": len(metrics_collector.metrics_buffer)
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to get observability metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/metrics/tenant/{tenant_id}")
async def get_tenant_metrics(tenant_id: int):
    """
    Get tenant-specific metrics summary for executive dashboards (Task 6.7-T09)
    """
    try:
        metrics_collector = get_metrics_collector()
        tenant_summary = metrics_collector.get_tenant_metrics_summary(tenant_id)
        
        return {
            "success": True,
            "tenant_metrics": tenant_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get tenant metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tenant metrics: {str(e)}")

@router.post("/metrics/business-impact")
async def record_business_impact(
    kpi_type: str,
    impact_value: float,
    impact_category: str,
    tenant_id: int,
    workflow_id: Optional[str] = None,
    industry_code: str = "SaaS",
    region: str = "US"
):
    """
    Record business KPI impact metrics (Task 20.1-T03)
    """
    try:
        metrics_collector = get_metrics_collector()
        
        metrics_collector.record_business_impact(
            kpi_type=kpi_type,
            impact_value=impact_value,
            impact_category=impact_category,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            industry_code=industry_code,
            region=region
        )
        
        return {
            "success": True,
            "message": "Business impact metrics recorded",
            "kpi_type": kpi_type,
            "impact_value": impact_value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to record business impact: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record business impact: {str(e)}")


async def parse_analysis_type_from_input(user_input: str) -> str:
    """Parse analysis type from user input using intelligent semantic analysis"""
    try:
        # Use the existing CleanIntentParser for intelligent analysis
        intent_parser = CleanIntentParser(
            pool_manager=pool_manager
        )
        
        # Parse the intent using LLM-powered semantic understanding
        parsed_intent = await intent_parser.parse_intent(user_input)
        
        if parsed_intent and parsed_intent.workflow_category:
            analysis_type = parsed_intent.workflow_category
            
            # Validate against available agents dynamically
            registry = EnhancedCapabilityRegistry()
            await registry.initialize()
            available_types = await registry.get_supported_analysis_types()
            
            if analysis_type in available_types:
                logger.info(f"🎯 Semantic Intent Analysis: '{user_input}' → {analysis_type}")
                logger.info(f"   🧠 Confidence: {parsed_intent.confidence:.2f}")
                logger.info(f"   💭 Reasoning: {parsed_intent.llm_reasoning}")
                return analysis_type
            else:
                # Try to find the closest match
                closest_match = _find_closest_analysis_type(analysis_type, available_types)
                if closest_match:
                    logger.info(f"🎯 LLM suggested '{analysis_type}' → mapped to '{closest_match}'")
                    return closest_match
                else:
                    logger.warning(f"⚠️ LLM suggested '{analysis_type}' but no close match found, using fallback")
        
        # Fallback to intelligent pattern matching if LLM parsing fails
        return await _intelligent_pattern_matching(user_input)
        
    except Exception as e:
        logger.error(f"❌ Failed to parse analysis type: {e}")
        # Use fallback pattern matching as last resort
        return await _intelligent_pattern_matching(user_input)


def _find_closest_analysis_type(requested_type: str, available_types: list[str]) -> Optional[str]:
    """Find the closest matching analysis type from available options"""
    # Exact match first
    if requested_type in available_types:
        return requested_type
    
    # Fuzzy matching based on keywords
    requested_words = set(requested_type.lower().replace('_', ' ').split())
    
    best_match = None
    best_score = 0
    
    for available_type in available_types:
        available_words = set(available_type.lower().replace('_', ' ').split())
        
        # Calculate overlap score
        common_words = requested_words.intersection(available_words)
        score = len(common_words) / max(len(requested_words), len(available_words))
        
        if score > best_score and score > 0.3:  # Minimum 30% overlap
            best_score = score
            best_match = available_type
    
    return best_match


async def _intelligent_pattern_matching(user_input: str) -> str:
    """Intelligent pattern matching that maps frontend requests to available agents"""
    
    # Get available analysis types from registry
    registry = EnhancedCapabilityRegistry()
    await registry.initialize()
    available_types = await registry.get_supported_analysis_types()
    
    # Dynamic mapping based on keywords in user input
    input_lower = user_input.lower()
    
    # Frontend-to-Agent mapping patterns (order matters - most specific first)
    mapping_patterns = {
        # Exact matches first (highest priority)
        ('sandbagging detection',): ['sandbagging_detection'],
        ('pipeline summary',): ['pipeline_summary', 'pipeline_overview'],
        ('activity gap analysis',): ['activity_tracking', 'missing_activities'],
        ('duplicate deal detection',): ['duplicate_detection', 'duplicate_deals'],
        ('ownerless deals detection',): ['ownerless_deals', 'unassigned_deals', 'ownerless_deals_detection'],
        ('deal risk scoring',): ['deal_risk_scoring', 'risk_assessment'],
        ('stage velocity analysis',): ['stage_velocity', 'progression_analysis'],
        ('coverage ratio enforcement',): ['coverage_analysis', 'pipeline_coverage'],
        ('company-wide health check',): ['health_overview', 'pipeline_health'],
        ('forecast comparison',): ['forecast_alignment', 'forecast_variance'],
        ('pipeline data quality audit',): ['data_quality', 'missing_fields', 'data_quality_audit'],
        ('at-risk deals analysis',): ['deals_at_risk', 'slipping_deals'],
        ('pipeline hygiene check',): ['stale_deals', 'pipeline_hygiene_stale_deals'],
        ('close date standardization',): ['quarter_end_dumping', 'close_date_patterns'],
        ('probability alignment check',): ['probability_alignment'],
        ('conversion rate analysis',): ['conversion_rate'],
        
        # Keyword-based patterns (lower priority)
        ('data quality', 'audit', 'missing'): ['data_quality', 'missing_fields', 'data_quality_audit'],
        ('duplicate', 'duplicate deals'): ['duplicate_detection', 'duplicate_deals'],
        ('ownerless', 'unassigned'): ['ownerless_deals', 'unassigned_deals', 'ownerless_deals_detection'],
        ('activity gap', 'activity', 'missing activities'): ['activity_tracking', 'missing_activities'],
        ('sandbagging', 'inflated deals'): ['sandbagging_detection'],
        ('risk scoring', 'deal risk'): ['deal_risk_scoring', 'risk_assessment'],
        ('at risk', 'slipping'): ['deals_at_risk', 'slipping_deals'],
        ('hygiene', 'stuck', 'stale'): ['stale_deals', 'pipeline_hygiene_stale_deals'],
        ('velocity', 'stage velocity'): ['stage_velocity', 'progression_analysis'],
        ('probability alignment'): ['probability_alignment'],
        ('close date', 'standardization'): ['quarter_end_dumping', 'close_date_patterns'],
        ('conversion rate'): ['conversion_rate'],
        ('pipeline summary', 'summary'): ['pipeline_summary', 'pipeline_overview'],
        ('coverage', 'coverage ratio'): ['coverage_analysis', 'pipeline_coverage'],
        ('health check', 'company-wide'): ['health_overview', 'pipeline_health'],
        ('forecast comparison', 'forecast'): ['forecast_alignment', 'forecast_variance']
    }
    
    # Find best matching pattern (check exact matches first)
    for keywords, agent_types in mapping_patterns.items():
        # Check if ALL keywords match (for exact matches with single keyword)
        if len(keywords) == 1 and keywords[0] == input_lower:
            # Exact match - highest priority
            for agent_type in agent_types:
                if agent_type in available_types:
                    logger.info(f"🎯 Exact match: '{user_input}' → '{agent_type}'")
                    return agent_type
        # Check if ANY keyword matches (for partial matches)
        elif any(keyword in input_lower for keyword in keywords):
            # Find the first available agent type
            for agent_type in agent_types:
                if agent_type in available_types:
                    logger.info(f"🎯 Pattern match: '{user_input}' → '{agent_type}' (keywords: {keywords})")
                    return agent_type
    
    # Fallback: Use the first available type as default
    if available_types:
        default_type = available_types[0]
        logger.warning(f"⚠️ No specific match found, using default: {default_type}")
        return default_type
    
    # Last resort fallback
    logger.error("❌ No agents available in registry")
    return "general_analysis"


async def _fallback_pattern_matching(user_input: str) -> str:
    """Fallback pattern matching when semantic analysis fails"""
    try:
        user_input_lower = user_input.lower()
        available_types = dynamic_rba_orchestrator.get_available_analysis_types()
        
        # Smart semantic matching without hardcoded priorities
        semantic_patterns = {}
        
        # Dynamically generate patterns from analysis type names
        for analysis_type in available_types:
            # Convert analysis_type to human-readable patterns
            readable_name = analysis_type.replace('_', ' ')
            words = readable_name.split()
            
            patterns = [
                readable_name,  # "sandbagging detection"
                analysis_type,  # "sandbagging_detection"
            ]
            
            # Add variations
            if 'detection' in words:
                patterns.append(readable_name.replace(' detection', ''))
            if 'analysis' in words:
                patterns.append(readable_name.replace(' analysis', ''))
            
            # Add specific domain knowledge patterns
            if analysis_type == 'sandbagging_detection':
                patterns.extend([
                    'sandbagging', 'sandbag', 'inflated deals', 
                    'low probability but high value', 'probability manipulation'
                ])
            elif analysis_type == 'stale_deals':
                patterns.extend(['stuck deals', 'old deals', 'stagnant deals'])
            elif analysis_type == 'missing_fields':
                patterns.extend(['incomplete data', 'empty fields', 'missing data'])
            
            semantic_patterns[analysis_type] = patterns
        
        # Find the best semantic match
        best_matches = []
        
        for analysis_type, patterns in semantic_patterns.items():
            match_score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern in user_input_lower:
                    # Weight longer phrases higher for better accuracy
                    weight = len(pattern.split()) * 2 if ' ' in pattern else 1
                    match_score += weight
                    matched_patterns.append(pattern)
            
            if match_score > 0:
                best_matches.append({
                    'type': analysis_type,
                    'score': match_score,
                    'patterns': matched_patterns
                })
        
        # Return the highest scoring match
        if best_matches:
            best_matches.sort(key=lambda x: x['score'], reverse=True)
            best_match = best_matches[0]
            
            logger.info(f"🎯 Pattern Match: '{user_input}' → {best_match['type']}")
            logger.info(f"   📊 Score: {best_match['score']}, Patterns: {best_match['patterns']}")
            
            return best_match['type']
        
        # Final word-level fallback
        for analysis_type in available_types:
            keywords = analysis_type.replace('_', ' ').split()
            if any(keyword in user_input_lower for keyword in keywords):
                logger.info(f"🎯 Word-level fallback: {analysis_type}")
                return analysis_type
        
        logger.warning(f"⚠️ No match found for: '{user_input}'")
        return None
        
    except Exception as e:
        logger.error(f"❌ Fallback pattern matching failed: {e}")
        return None


async def get_csv_opportunity_data(tenant_id: str, user_id: str) -> list:
    """Get opportunity data from CSV file"""
    try:
        logger.info(f"📊 Loading opportunity data from CSV for tenant {tenant_id}, user {user_id}")
        
        # Get ALL opportunities from CSV loader (no limit for comprehensive analysis)
        opportunities = crm_data_loader.get_opportunities(
            tenant_id=tenant_id, 
            user_id=user_id, 
            limit=5000  # Analyze all available opportunities for comprehensive insights
        )
        
        if opportunities:
            logger.info(f"✅ Loaded {len(opportunities)} opportunities from CSV")
            
            # Log sample data for debugging
            sample_opp = opportunities[0] if opportunities else {}
            logger.info(f"📋 Sample opportunity: {sample_opp.get('Name', 'Unknown')} - ${sample_opp.get('Amount', 0):,.0f} ({sample_opp.get('Probability', 0)}%)")
            
            return opportunities
        else:
            logger.warning("⚠️ No opportunities found in CSV, using fallback test data")
            return await get_test_opportunity_data_with_sandbagging_cases(tenant_id)
            
    except Exception as e:
        logger.error(f"❌ Failed to load CSV data: {e}")
        logger.info("🔄 Falling back to enhanced test data")
        return await get_test_opportunity_data_with_sandbagging_cases(tenant_id)


async def get_real_opportunity_data(tenant_id: str, user_id: str) -> list:
    """Get real opportunity data from Salesforce via Fabric"""
    try:
        logger.info(f"🔍 Fetching real opportunity data for tenant {tenant_id}, user {user_id}")
        
        # Use the existing Fabric service to get opportunities
        fabric_service = pool_manager.fabric_service
        
        # Query for opportunities with all necessary fields for RBA analysis
        query = """
        SELECT 
            Id,
            Name,
            Amount,
            Probability,
            StageName,
            CloseDate,
            CreatedDate,
            LastModifiedDate,
            LastActivityDate,
            Account_Name__c as AccountName,
            Owner_Name__c as OwnerName,
            Account_Industry__c as AccountIndustry,
            Days_Since_Last_Activity__c as DaysSinceActivity,
            Stage_Duration_Days__c as StageDuration,
            Expected_Revenue__c as ExpectedRevenue,
            Forecast_Category__c as ForecastCategory,
            Type as OpportunityType,
            LeadSource
        FROM Opportunity_View
        WHERE IsDeleted = 0 
        AND IsClosed = 0
        AND Amount > 0
        ORDER BY Amount DESC, LastModifiedDate DESC
        OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY
        """
        
        # Execute query
        result = await fabric_service.execute_query(query)
        
        if result and 'data' in result:
            opportunities = result['data']
            logger.info(f"✅ Retrieved {len(opportunities)} real opportunities from Salesforce")
            
            # Transform data to match expected format
            formatted_opportunities = []
            for opp in opportunities:
                formatted_opp = {
                    'Id': opp.get('Id'),
                    'Name': opp.get('Name'),
                    'Amount': float(opp.get('Amount', 0)) if opp.get('Amount') else 0,
                    'Probability': float(opp.get('Probability', 0)) if opp.get('Probability') else 0,
                    'StageName': opp.get('StageName', ''),
                    'CloseDate': opp.get('CloseDate'),
                    'CreatedDate': opp.get('CreatedDate'),
                    'LastModifiedDate': opp.get('LastModifiedDate'),
                    'ActivityDate': opp.get('LastActivityDate'),
                    'Account': {
                        'Name': opp.get('AccountName', ''),
                        'Industry': opp.get('AccountIndustry', '')
                    },
                    'Owner': {
                        'Name': opp.get('OwnerName', '')
                    },
                    'DaysSinceActivity': int(opp.get('DaysSinceActivity', 0)) if opp.get('DaysSinceActivity') else None,
                    'StageDuration': int(opp.get('StageDuration', 0)) if opp.get('StageDuration') else None,
                    'ExpectedRevenue': float(opp.get('ExpectedRevenue', 0)) if opp.get('ExpectedRevenue') else None,
                    'ForecastCategory': opp.get('ForecastCategory', ''),
                    'OpportunityType': opp.get('OpportunityType', ''),
                    'LeadSource': opp.get('LeadSource', '')
                }
                formatted_opportunities.append(formatted_opp)
            
            return formatted_opportunities
        else:
            logger.warning("⚠️ No opportunity data returned from Fabric, using fallback test data")
            return await get_test_opportunity_data_with_sandbagging_cases(tenant_id)
            
    except Exception as e:
        logger.error(f"❌ Failed to get real opportunity data: {e}")
        logger.info("🔄 Falling back to enhanced test data with sandbagging cases")
        return await get_test_opportunity_data_with_sandbagging_cases(tenant_id)


async def get_test_opportunity_data_with_sandbagging_cases(tenant_id: str) -> list:
    """Enhanced test data that includes actual sandbagging cases for testing"""
    try:
        logger.info("📊 Using enhanced test data with sandbagging cases")
        return [
            # SANDBAGGING CASE 1: High value, low probability
            {
                'Id': 'OPP001_SANDBAG',
                'Name': 'Enterprise Software Deal - SANDBAGGING SUSPECT',
                'Amount': 500000,  # HIGH VALUE
                'Probability': 25,  # LOW PROBABILITY - Classic sandbagging
                'StageName': 'Proposal/Price Quote',  # Advanced stage
                'CloseDate': '2024-12-31',
                'Account': {'Name': 'TechCorp Inc', 'Industry': 'Technology'},
                'Owner': {'Name': 'John Smith'},
                'ActivityDate': '2024-09-10',
                'CreatedDate': '2024-08-15'
            },
            # SANDBAGGING CASE 2: Very high value, very low probability
            {
                'Id': 'OPP002_SANDBAG',
                'Name': 'Cloud Migration Project - POTENTIAL SANDBAGGING',
                'Amount': 750000,  # VERY HIGH VALUE
                'Probability': 15,  # VERY LOW PROBABILITY - Clear sandbagging
                'StageName': 'Contract Negotiation',  # Very advanced stage
                'CloseDate': '2024-11-15',
                'Account': {'Name': 'StartupXYZ', 'Industry': 'Software'},
                'Owner': {'Name': 'Jane Doe'},
                'ActivityDate': '2024-08-20',
                'CreatedDate': '2024-07-10'
            },
            # NORMAL CASE 1: High value, high probability (not sandbagging)
            {
                'Id': 'OPP003_NORMAL',
                'Name': 'Security Upgrade',
                'Amount': 250000,
                'Probability': 85,  # High probability - normal
                'StageName': 'Contract Negotiation',
                'CloseDate': '2024-10-30',
                'Account': {'Name': 'SecureBank', 'Industry': 'Financial Services'},
                'Owner': {'Name': 'Mike Johnson'},
                'ActivityDate': '2024-09-15',
                'CreatedDate': '2024-06-01'
            },
            # SANDBAGGING CASE 3: Medium-high value, low probability
            {
                'Id': 'OPP004_SANDBAG',
                'Name': 'Data Analytics Platform - SANDBAGGING LIKELY',
                'Amount': 300000,  # Medium-high value
                'Probability': 20,  # Low probability - sandbagging
                'StageName': 'Proposal/Price Quote',
                'CloseDate': '2025-01-31',
                'Account': {'Name': 'DataCorp', 'Industry': 'Analytics'},
                'Owner': {'Name': 'Sarah Wilson'},
                'ActivityDate': '2024-07-01',
                'CreatedDate': '2024-05-15'
            },
            # NORMAL CASE 2: Low value, low probability (not sandbagging)
            {
                'Id': 'OPP005_NORMAL',
                'Name': 'Mobile App Development',
                'Amount': 50000,  # Low value
                'Probability': 30,  # Low probability - but not sandbagging due to low value
                'StageName': 'Value Proposition',
                'CloseDate': '2024-12-15',
                'Account': {'Name': 'MobileFirst', 'Industry': 'Mobile'},
                'Owner': {'Name': 'Tom Brown'},
                'ActivityDate': '2024-09-12',
                'CreatedDate': '2024-08-01'
            }
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get enhanced test data: {e}")
        return []


async def get_test_opportunity_data(tenant_id: str) -> list:
    """Get test opportunity data for demonstration"""
    try:
        # In a real application, this would fetch from Salesforce/database
        # For now, return test data similar to our test architecture
        return [
            {
                'Id': 'OPP001',
                'Name': 'Enterprise Software Deal',
                'Amount': 150000,
                'Probability': 75,
                'StageName': 'Proposal/Price Quote',
                'CloseDate': '2024-12-31',
                'Account': {'Name': 'TechCorp Inc'},
                'Owner': {'Name': 'John Smith'},
                'ActivityDate': '2024-09-10',
                'CreatedDate': '2024-08-15'
            },
            {
                'Id': 'OPP002',
                'Name': 'Cloud Migration Project',
                'Amount': 75000,
                'Probability': 60,
                'StageName': 'Needs Analysis',
                'CloseDate': '2024-11-15',
                'Account': {'Name': 'StartupXYZ'},
                'Owner': {'Name': 'Jane Doe'},
                'ActivityDate': '2024-08-20',
                'CreatedDate': '2024-07-10'
            },
            {
                'Id': 'OPP003',
                'Name': 'Security Upgrade',
                'Amount': 25000,
                'Probability': 90,
                'StageName': 'Contract Negotiation',
                'CloseDate': '2024-10-30',
                'Account': {'Name': 'SecureBank'},
                'Owner': {'Name': 'Mike Johnson'},
                'ActivityDate': '2024-09-15',
                'CreatedDate': '2024-06-01'
            },
            {
                'Id': 'OPP004',
                'Name': 'Data Analytics Platform',
                'Amount': 200000,
                'Probability': 45,
                'StageName': 'Qualification',
                'CloseDate': '2025-01-31',
                'Account': {'Name': 'DataCorp'},
                'Owner': {'Name': 'Sarah Wilson'},
                'ActivityDate': '2024-07-01',
                'CreatedDate': '2024-05-15'
            },
            {
                'Id': 'OPP005',
                'Name': 'Mobile App Development',
                'Amount': 80000,
                'Probability': 65,
                'StageName': 'Value Proposition',
                'CloseDate': '2024-12-15',
                'Account': {'Name': 'MobileFirst'},
                'Owner': {'Name': 'Tom Brown'},
                'ActivityDate': '2024-09-12',
                'CreatedDate': '2024-08-01'
            }
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get test data: {e}")
        return []


@router.get("/analysis-types")
async def get_supported_analysis_types():
    """Get all supported analysis types from the dynamic RBA orchestrator"""
    try:
        # Get available analysis types
        analysis_types = dynamic_rba_orchestrator.get_available_analysis_types()
        
        # Group by category for better frontend UX
        categorized_types = {
            "data_quality": [t for t in analysis_types if any(x in t for x in ['missing', 'data_quality', 'field_validation'])],
            "risk_analysis": [t for t in analysis_types if any(x in t for x in ['risk', 'sandbagging', 'deals_at_risk'])],
            "velocity_analysis": [t for t in analysis_types if any(x in t for x in ['velocity', 'stage', 'progression'])],
            "pipeline_analysis": [t for t in analysis_types if any(x in t for x in ['pipeline', 'coverage', 'summary'])],
            "activity_analysis": [t for t in analysis_types if any(x in t for x in ['activity', 'tracking'])],
            "forecast_analysis": [t for t in analysis_types if any(x in t for x in ['forecast', 'alignment'])],
            "other": [t for t in analysis_types if not any(any(x in t for x in category_keywords) for category_keywords in [
                ['missing', 'data_quality', 'field_validation'],
                ['risk', 'sandbagging', 'deals_at_risk'],
                ['velocity', 'stage', 'progression'],
                ['pipeline', 'coverage', 'summary'],
                ['activity', 'tracking'],
                ['forecast', 'alignment']
            ])]
        }
        
        return {
            "success": True,
            "total_analysis_types": len(analysis_types),
            "analysis_types": analysis_types,
            "categorized_types": categorized_types,
            "message": f"Found {len(analysis_types)} supported analysis types"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get analysis types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis types: {str(e)}")


@router.get("/sample-prompts")
async def get_sample_pipeline_prompts():
    """
    Get sample pipeline prompts to test the policy-aware agent system
    These are the exact prompts you mentioned that need to work
    """
    return {
        "success": True,
        "sample_prompts": {
            "data_quality": [
                "Audit pipeline data quality – identify deals missing close dates, amounts, or owners",
                "Check duplicate deals across Salesforce/CRM and flag for review",
                "Detect ownerless or unassigned deals",
                "Highlight deals missing activities/logs (no calls/emails logged in last 14 days)"
            ],
            "risk_analysis": [
                "Apply risk scoring to deals and categorize into high/medium/low",
                "Run pipeline hygiene check – deals stuck >60 days in stage",
                "Identify sandbagging or inflated deals (low probability but high value)",
                "Check my deals at risk (slipping, no activity, customer disengagement)"
            ],
            "velocity_analysis": [
                "Validate probability alignment (system probability vs rep forecast)",
                "Standardize close dates (flag last-day-of-quarter dumping)",
                "Review pipeline velocity by stage (days per stage)",
                "Analyze stage conversion rates by team/rep"
            ],
            "performance_analysis": [
                "Get my pipeline summary (total open, weighted, committed)",
                "Enforce pipeline coverage ratio (e.g., 3x quota rule)",
                "View company-wide pipeline health (coverage, risk, stage distribution)",
                "Compare rep forecast vs system forecast vs AI prediction"
            ],
            "coaching_insights": [
                "Coach reps on stalled deals (pipeline stuck >30 days)",
                "Get my next best actions for deals in negotiation stage",
                "Identify coaching opportunities (rep-specific risk flags)",
                "Spot under-forecasting reps (low commit vs system forecast)"
            ]
        },
        "test_instructions": "Use any of these prompts with the /execute endpoint to test policy-aware pipeline agents",
        "policy_integration_note": "All prompts will automatically fetch and apply relevant pipeline policies configured by revenue managers"
    }


async def _merge_user_config_with_defaults(analysis_type: str, user_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user-provided configuration with defaults from business rules"""
    try:
        from dsl.rules.business_rules_engine import BusinessRulesEngine
        
        # Load business rules to get defaults
        rules_engine = BusinessRulesEngine()
        
        if analysis_type not in rules_engine.rules:
            logger.warning(f"⚠️ No rules found for {analysis_type}, using user config only")
            return user_config
        
        rule_set = rules_engine.rules[analysis_type]
        defaults = rule_set.get("defaults", {})
        
        # Start with defaults, then override with user-provided values
        merged_config = defaults.copy()
        merged_config.update(user_config)
        
        logger.info(f"🔧 Config merge for {analysis_type}:")
        logger.info(f"   📋 Defaults: {defaults}")
        logger.info(f"   👤 User overrides: {user_config}")
        logger.info(f"   🔍 high_value_threshold in defaults: {defaults.get('high_value_threshold', 'NOT_FOUND')}")
        logger.info(f"   🔍 high_value_threshold in user_config: {user_config.get('high_value_threshold', 'NOT_FOUND')}")
        logger.info(f"   ⚙️ Final config: {merged_config}")
        logger.info(f"   🔍 high_value_threshold in final config: {merged_config.get('high_value_threshold', 'NOT_FOUND')}")
        
        return merged_config
        
    except Exception as e:
        logger.error(f"❌ Failed to merge config for {analysis_type}: {e}")
        # Fallback to user config only
        return user_config


def extract_config_from_prompt(user_input: str) -> Dict[str, Any]:
    """
    Enhanced Dynamic Configuration Extraction from Natural Language
    
    Supports multiple formats, example given below:
    - "high value threshold 999000"
    - "deals over $500K" 
    - "sandbagging threshold of 30%"
    - "probability below 25%"
    - "confidence level 80"
    - "stage velocity over 45 days"
    - "with threshold 999K"
    """
    import re
    
    config = {}
    input_lower = user_input.lower()
    
    # Helper function to parse numeric values with units
    def parse_numeric_value(text: str) -> int:
        """Parse numeric values with K, M suffixes"""
        text = text.replace(',', '').replace('$', '').strip()
        if text.endswith('k') or text.endswith('K'):
            return int(float(text[:-1]) * 1000)
        elif text.endswith('m') or text.endswith('M'):
            return int(float(text[:-1]) * 1000000)
        elif text.endswith('%'):
            return int(float(text[:-1]))
        else:
            return int(float(text))
    
    # ENHANCED HIGH VALUE THRESHOLD PATTERNS
    high_value_patterns = [
        # Direct threshold mentions (number after threshold)
        r'high.*value.*threshold[:\s]*(?:of|is|=)?\s*\$?([0-9,]+[km]?)',
        r'high.*value[:\s]*(?:of|is|=)?\s*\$?([0-9,]+[km]?)',
        r'value.*threshold[:\s]*(?:of|is|=)?\s*\$?([0-9,]+[km]?)',
        r'threshold[:\s]*(?:of|is|=)?\s*\$?([0-9,]+[km]?)',
        
        # Number BEFORE threshold patterns (key fix!)
        r'with[:\s]+\$?([0-9,]+[km]?)[:\s]+high.*value.*threshold',
        r'with[:\s]+\$?([0-9,]+[km]?)[:\s]+value.*threshold',  
        r'with[:\s]+\$?([0-9,]+[km]?)[:\s]+threshold',
        r'\$?([0-9,]+[km]?)[:\s]+high.*value.*threshold',
        r'\$?([0-9,]+[km]?)[:\s]+value.*threshold',
        
        # Deal value mentions  
        r'deals?\s+(?:over|above|greater than)\s*\$?([0-9,]+[km]?)',
        r'deals?\s+(?:worth|valued at)\s*\$?([0-9,]+[km]?)',
        r'(?:over|above)\s*\$?([0-9,]+[km]?)\s+deals?',
        
        # Amount/value mentions
        r'amount[:\s]*(?:over|above|>=)\s*\$?([0-9,]+[km]?)',
        r'value[:\s]*(?:over|above|>=)\s*\$?([0-9,]+[km]?)',
        
        # With threshold patterns (number after threshold)
        r'with.*threshold[:\s]*\$?([0-9,]+[km]?)',
        r'using.*threshold[:\s]*\$?([0-9,]+[km]?)'
    ]
    
    for pattern in high_value_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                config['high_value_threshold'] = parse_numeric_value(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # ENHANCED SANDBAGGING THRESHOLD PATTERNS
    # NOTE: Only extract explicit sandbagging thresholds, not generic "threshold" mentions
    sandbagging_patterns = [
        r'sandbagging.*threshold[:\s]*(?:of|is|=)?\s*([0-9]+)%?',
        r'sandbagging.*(?:score|level)[:\s]*(?:of|is|=)?\s*([0-9]+)%?',
        r'flag.*deals.*(?:score|threshold)[:\s]*([0-9]+)%?',
        r'detect.*(?:score|threshold)[:\s]*([0-9]+)%?',
        # Removed the generic "threshold.*sandbagging" pattern that was causing issues
    ]
    
    for pattern in sandbagging_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                threshold_value = int(match.group(1))
                # Validate: sandbagging thresholds should be 0-100, not financial values
                if 0 <= threshold_value <= 100:
                    config['sandbagging_threshold'] = threshold_value
                    break
            except (ValueError, IndexError):
                continue
    
    # ENHANCED PROBABILITY THRESHOLD PATTERNS
    prob_patterns = [
        r'probability.*threshold[:\s]*(?:of|is|=)?\s*([0-9]+)%?',
        r'prob.*threshold[:\s]*(?:of|is|=)?\s*([0-9]+)%?',
        r'low.*probability[:\s]*(?:of|is|=)?\s*([0-9]+)%?',
        r'probability.*(?:below|under|less than)[:\s]*([0-9]+)%?',
        r'prob.*(?:below|under|less than)[:\s]*([0-9]+)%?',
        r'(?:below|under|less than)[:\s]*([0-9]+)%?\s*probability'
    ]
    
    for pattern in prob_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                config['low_probability_threshold'] = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # ADVANCED STAGE PROBABILITY PATTERNS
    stage_prob_patterns = [
        r'(?:advanced|late).*stage.*probability[:\s]*([0-9]+)%?',
        r'stage.*probability.*threshold[:\s]*([0-9]+)%?',
        r'(?:negotiation|proposal|closing).*probability[:\s]*([0-9]+)%?'
    ]
    
    for pattern in stage_prob_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                config['advanced_stage_probability_threshold'] = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # CONFIDENCE THRESHOLD PATTERNS
    confidence_patterns = [
        r'confidence.*(?:level|threshold)[:\s]*([0-9]+)%?',
        r'confidence[:\s]*(?:of|is|=)?\s*([0-9]+)%?',
        r'(?:min|minimum).*confidence[:\s]*([0-9]+)%?'
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                config['confidence_threshold'] = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # STAGE VELOCITY / DAYS PATTERNS
    velocity_patterns = [
        r'(?:stage|velocity).*(?:over|above|more than)[:\s]*([0-9]+)\s*days?',
        r'(?:stuck|stale).*(?:for|over)[:\s]*([0-9]+)\s*days?',
        r'days.*(?:in|per)\s*stage[:\s]*([0-9]+)',
        r'velocity.*threshold[:\s]*([0-9]+)\s*days?'
    ]
    
    for pattern in velocity_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                config['stale_threshold_days'] = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # MEGA DEAL PATTERNS
    mega_patterns = [
        r'(?:mega|million|large).*deal.*threshold[:\s]*\$?([0-9,]+[km]?)',
        r'(?:mega|million|large).*deals?[:\s]*(?:over|above)\s*\$?([0-9,]+[km]?)',
        r'million.*deal.*probability[:\s]*([0-9]+)%?'
    ]
    
    for pattern in mega_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                value = parse_numeric_value(match.group(1))
                if value >= 1000000:  # Only if it's actually a mega deal threshold
                    config['mega_deal_threshold'] = value
                break
            except (ValueError, IndexError):
                continue
    
    # INDUSTRY ADJUSTMENTS
    if any(word in input_lower for word in ['industry', 'sector', 'vertical', 'adjust', 'calibrate']):
        if any(word in input_lower for word in ['enable', 'on', 'yes', 'true', 'with']):
            config['enable_industry_adjustments'] = True
        elif any(word in input_lower for word in ['disable', 'off', 'no', 'false', 'without']):
            config['enable_industry_adjustments'] = False
    
    # ACTIVITY TRACKING PATTERNS
    activity_patterns = [
        r'activity.*(?:within|last)[:\s]*([0-9]+)\s*days?',
        r'(?:no|without).*activity.*(?:for|over)[:\s]*([0-9]+)\s*days?',
        r'inactive.*(?:for|over)[:\s]*([0-9]+)\s*days?'
    ]
    
    for pattern in activity_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                config['minimum_activity_days'] = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # Debug logging for extracted configuration
    if config:
        logger.info(f"🎯 Natural Language Config Extraction:")
        for key, value in config.items():
            logger.info(f"   📋 {key}: {value}")
    
    return config


async def extract_config_with_ai(user_input: str, analysis_type: str) -> Dict[str, Any]:
    """
    AI-Powered Configuration Extraction using LLM
    
    Uses the existing intent parser to intelligently extract configuration
    parameters from natural language using semantic understanding.
    """
    try:
        from dsl.hub.intent_parser_v2 import CleanIntentParser
        
        # Use the existing CleanIntentParser for AI-powered extraction
        intent_parser = CleanIntentParser(pool_manager=pool_manager)
        
        # Create a specialized prompt for configuration extraction
        config_extraction_prompt = f"""
        Extract configuration parameters from this user request for {analysis_type} analysis:
        
        User Input: "{user_input}"
        
        Look for these parameter types and return ONLY a JSON object:
        - high_value_threshold: dollar amounts (convert K/M to numbers)
        - sandbagging_threshold: percentage scores (0-100)  
        - low_probability_threshold: probability percentages (0-100)
        - confidence_threshold: confidence levels (0-100)
        - stale_threshold_days: number of days
        - advanced_stage_probability_threshold: probability percentages
        - mega_deal_threshold: large dollar amounts
        - enable_industry_adjustments: boolean
        
        Examples:
        "deals over $500K" → {{"high_value_threshold": 500000}}
        "sandbagging threshold 30%" → {{"sandbagging_threshold": 30}}
        "probability below 25%" → {{"low_probability_threshold": 25}}
        "confidence level 80" → {{"confidence_threshold": 80}}
        
        Return empty JSON {{}} if no parameters found.
        """
        
        # Use the intent parser to extract configuration
        result = await intent_parser.parse_intent(config_extraction_prompt)
        
        # Try to parse the JSON response
        if result and hasattr(result, 'llm_response'):
            import json
            try:
                config = json.loads(result.llm_response)
                if isinstance(config, dict):
                    return config
            except json.JSONDecodeError:
                # If not valid JSON, try to extract from text response
                return parse_config_from_ai_response(result.llm_response)
        
        return {}
        
    except Exception as e:
        logger.warning(f"⚠️ AI config extraction failed: {e}")
        return {}


def parse_config_from_ai_response(ai_response: str) -> Dict[str, Any]:
    """Parse configuration from AI response text if JSON parsing fails"""
    import re
    import json
    
    config = {}
    
    # Try to find JSON blocks in the response
    json_pattern = r'\{[^}]*\}'
    json_matches = re.findall(json_pattern, ai_response)
    
    for match in json_matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                config.update(parsed)
        except json.JSONDecodeError:
            continue
    
    # If no JSON found, try to extract key-value pairs from text
    if not config:
        kv_patterns = [
            r'high_value_threshold[:\s]*([0-9,]+)',
            r'sandbagging_threshold[:\s]*([0-9]+)',
            r'low_probability_threshold[:\s]*([0-9]+)',
            r'confidence_threshold[:\s]*([0-9]+)'
        ]
        
        for pattern in kv_patterns:
            match = re.search(pattern, ai_response.lower())
            if match:
                key = pattern.split('[')[0]  # Extract key name
                try:
                    config[key] = int(match.group(1).replace(',', ''))
                except (ValueError, IndexError):
                    continue
    
    return config


def apply_smart_defaults(config: Dict[str, Any], analysis_type: str, user_input: str) -> Dict[str, Any]:
    """
    Apply Smart Contextual Defaults based on Analysis Type and User Intent
    
    Intelligently infers configuration based on:
    - Analysis type (sandbagging vs stale deals vs data quality)
    - User language patterns (aggressive, conservative, strict, lenient)
    - Industry context (enterprise, SMB, startup)
    - Urgency indicators (immediate, critical, routine)
    """
    
    enhanced_config = config.copy()
    input_lower = user_input.lower()
    
    logger.info(f"🎯 Smart Defaults - Input config: {config}")
    logger.info(f"🎯 Smart Defaults - Analysis type: {analysis_type}")
    logger.info(f"🎯 Smart Defaults - User input: '{user_input}'")
    
    # SANDBAGGING DETECTION SMART DEFAULTS
    if analysis_type == 'sandbagging_detection':
        
        # Detect user intent for sensitivity
        if any(word in input_lower for word in ['aggressive', 'strict', 'tight', 'conservative', 'sensitive']):
            # More sensitive detection
            if 'sandbagging_threshold' not in enhanced_config:
                enhanced_config['sandbagging_threshold'] = 45  # Lower threshold = more sensitive
            if 'low_probability_threshold' not in enhanced_config:
                enhanced_config['low_probability_threshold'] = 40  # Higher threshold = more deals caught
        
        elif any(word in input_lower for word in ['lenient', 'relaxed', 'loose', 'broad', 'less sensitive']):
            # Less sensitive detection
            if 'sandbagging_threshold' not in enhanced_config:
                enhanced_config['sandbagging_threshold'] = 75  # Higher threshold = less sensitive
            if 'low_probability_threshold' not in enhanced_config:
                enhanced_config['low_probability_threshold'] = 25  # Lower threshold = fewer deals caught
        
        # Detect enterprise vs SMB context
        if any(word in input_lower for word in ['enterprise', 'large', 'big', 'major', 'fortune']):
            if 'high_value_threshold' not in enhanced_config:
                enhanced_config['high_value_threshold'] = 500000  # $500K for enterprise
        elif any(word in input_lower for word in ['smb', 'small', 'startup', 'mid-market']):
            if 'high_value_threshold' not in enhanced_config:
                enhanced_config['high_value_threshold'] = 50000   # $50K for SMB
        
        # Detect urgency
        if any(word in input_lower for word in ['urgent', 'immediate', 'critical', 'asap', 'now']):
            enhanced_config['confidence_threshold'] = enhanced_config.get('confidence_threshold', 60)  # Lower confidence for urgent
        
        # Detect quarter-end context
        if any(word in input_lower for word in ['quarter', 'month-end', 'closing', 'end of']):
            enhanced_config['advanced_stage_probability_threshold'] = 50  # Higher scrutiny for quarter-end
    
    # STALE DEALS SMART DEFAULTS
    elif analysis_type in ['stale_deals', 'pipeline_hygiene_stale_deals']:
        
        # Detect velocity expectations
        if any(word in input_lower for word in ['fast', 'quick', 'rapid', 'accelerated']):
            enhanced_config['stale_threshold_days'] = enhanced_config.get('stale_threshold_days', 21)  # 3 weeks
        elif any(word in input_lower for word in ['slow', 'patient', 'long', 'extended']):
            enhanced_config['stale_threshold_days'] = enhanced_config.get('stale_threshold_days', 60)  # 2 months
        
        # Default based on deal complexity indicators
        if any(word in input_lower for word in ['complex', 'enterprise', 'large', 'strategic']):
            enhanced_config['stale_threshold_days'] = enhanced_config.get('stale_threshold_days', 45)  # Longer for complex deals
        elif any(word in input_lower for word in ['simple', 'transactional', 'quick', 'small']):
            enhanced_config['stale_threshold_days'] = enhanced_config.get('stale_threshold_days', 21)  # Shorter for simple deals
    
    logger.info(f"🎯 Smart Defaults - Final enhanced config: {enhanced_config}")
    return enhanced_config
