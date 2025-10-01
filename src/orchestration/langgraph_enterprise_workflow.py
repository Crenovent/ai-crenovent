"""
LangGraph Enterprise Workflow for Strategic Planning
Revolutionary multi-agent orchestration with business intelligence
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, asdict

# LangGraph imports
try:
    from langgraph import StateGraph, END
    from langgraph.graph import Graph
    import operator
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnterprisePlanningState(TypedDict):
    """State for enterprise planning workflow"""
    # Input
    user_message: str
    user_context: Dict[str, Any]
    
    # User Intelligence
    user_intelligence: Dict[str, Any]
    
    # Processing stages
    nlp_extracted_data: Dict[str, Any]
    database_enriched_data: Dict[str, Any]
    market_intelligence: Dict[str, Any]
    analytics_results: Dict[str, Any]
    
    # Output
    final_form_data: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float
    
    # Workflow control
    current_step: str
    error_state: Optional[str]
    processing_time: float

@dataclass
class WorkflowNode:
    """Represents a workflow node with its configuration"""
    name: str
    description: str
    required_inputs: List[str]
    outputs: List[str]
    timeout: int = 30
    retries: int = 2

class EnterpriseStrategicPlanningWorkflow:
    """
    Revolutionary LangGraph workflow for enterprise strategic planning
    
    Workflow Steps:
    1. User Intelligence Loading
    2. Advanced NLP Extraction  
    3. Database Enhancement
    4. Market Intelligence Gathering
    5. Analytics Processing
    6. Recommendation Generation
    7. Final Response Assembly
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.user_intelligence_service = None
        self.revolutionary_form_filler = None
        self.enterprise_toolkit = None
        self.revolutionary_crud_agent = None
        
        self._initialize_services()
        
        # Workflow configuration
        self.workflow = None
        self._build_workflow()
    
    def _initialize_services(self):
        """Initialize all enterprise services"""
        try:
            if self.pool_manager:
                # User Intelligence Service
                from ..services.user_intelligence_service import get_user_intelligence_service
                self.user_intelligence_service = get_user_intelligence_service(self.pool_manager)
                
                # Revolutionary Form Filler
                from ..tools.revolutionary_form_filler import get_revolutionary_form_filler
                self.revolutionary_form_filler = get_revolutionary_form_filler(self.pool_manager)
                
                # Enterprise Toolkit
                from ..tools.enterprise_langchain_toolkit import get_enterprise_toolkit
                self.enterprise_toolkit = get_enterprise_toolkit(self.pool_manager)
                
                # Revolutionary CRUD Agent
                from ..agents.revolutionary_crud_agent import get_revolutionary_crud_agent
                self.revolutionary_crud_agent = get_revolutionary_crud_agent(self.pool_manager)
                
                self.logger.info("✅ Enterprise workflow services initialized")
            else:
                self.logger.warning("⚠️ Pool manager not available")
                
        except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {e}")
    
    def _build_workflow(self):
        """Build the enterprise LangGraph workflow"""
        try:
            if not LANGGRAPH_AVAILABLE:
                self.logger.warning("⚠️ LangGraph not available, using sequential processing")
                return
            
            # Create workflow graph
            workflow = StateGraph(EnterprisePlanningState)
            
            # Add workflow nodes
            workflow.add_node("load_user_intelligence", self._load_user_intelligence)
            workflow.add_node("extract_with_nlp", self._extract_with_nlp)
            workflow.add_node("enrich_from_database", self._enrich_from_database)
            workflow.add_node("gather_market_intelligence", self._gather_market_intelligence)
            workflow.add_node("perform_analytics", self._perform_analytics)
            workflow.add_node("generate_recommendations", self._generate_recommendations)
            workflow.add_node("assemble_final_response", self._assemble_final_response)
            workflow.add_node("handle_error", self._handle_error)
            
            # Define workflow edges
            workflow.set_entry_point("load_user_intelligence")
            
            # Sequential flow
            workflow.add_edge("load_user_intelligence", "extract_with_nlp")
            workflow.add_edge("extract_with_nlp", "enrich_from_database")
            workflow.add_edge("enrich_from_database", "gather_market_intelligence")
            workflow.add_edge("gather_market_intelligence", "perform_analytics")
            workflow.add_edge("perform_analytics", "generate_recommendations")
            workflow.add_edge("generate_recommendations", "assemble_final_response")
            workflow.add_edge("assemble_final_response", END)
            
            # Error handling
            workflow.add_edge("handle_error", END)
            
            # Conditional routing for complex scenarios
            workflow.add_conditional_edges(
                "extract_with_nlp",
                self._should_skip_market_research,
                {
                    "skip_market": "perform_analytics",
                    "continue": "enrich_from_database"
                }
            )
            
            workflow.add_conditional_edges(
                "enrich_from_database", 
                self._should_perform_crud_operation,
                {
                    "crud_operation": "handle_crud_request",
                    "continue": "gather_market_intelligence"
                }
            )
            
            # Compile workflow
            self.workflow = workflow.compile()
            
            self.logger.info("✅ Enterprise LangGraph workflow built successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Workflow build failed: {e}")
            self.workflow = None
    
    async def process_strategic_planning_request(self, message: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process strategic planning request through enterprise workflow
        
        Args:
            message: User's natural language input
            user_context: User context (user_id, tenant_id, etc.)
            
        Returns:
            Complete form data with intelligence and recommendations
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Starting enterprise workflow for: '{message[:100]}...'")
            
            # Initialize workflow state
            initial_state = EnterprisePlanningState(
                user_message=message,
                user_context=user_context,
                user_intelligence={},
                nlp_extracted_data={},
                database_enriched_data={},
                market_intelligence={},
                analytics_results={},
                final_form_data={},
                recommendations=[],
                confidence_score=0.0,
                current_step="initializing",
                error_state=None,
                processing_time=0.0
            )
            
            # Execute workflow
            if self.workflow:
                # Use LangGraph workflow
                final_state = await self._execute_langgraph_workflow(initial_state)
            else:
                # Fallback to sequential processing
                final_state = await self._execute_sequential_workflow(initial_state)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            final_state["processing_time"] = processing_time
            
            # Generate final response
            response = self._format_final_response(final_state)
            
            self.logger.info(f"✅ Enterprise workflow completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Enterprise workflow failed: {e}")
            return self._generate_error_response(str(e), message, user_context)
    
    async def _execute_langgraph_workflow(self, initial_state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Execute using LangGraph workflow"""
        try:
            # Convert to dict for LangGraph compatibility
            state_dict = dict(initial_state)
            
            # Execute workflow
            result = await self.workflow.ainvoke(state_dict)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ LangGraph execution failed: {e}")
            # Fallback to sequential
            return await self._execute_sequential_workflow(initial_state)
    
    async def _execute_sequential_workflow(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Fallback sequential workflow execution"""
        try:
            # Execute workflow steps sequentially
            state = await self._load_user_intelligence(state)
            state = await self._extract_with_nlp(state)
            state = await self._enrich_from_database(state)
            state = await self._gather_market_intelligence(state)
            state = await self._perform_analytics(state)
            state = await self._generate_recommendations(state)
            state = await self._assemble_final_response(state)
            
            return state
            
        except Exception as e:
            state["error_state"] = str(e)
            return await self._handle_error(state)
    
    # ========================================
    # Workflow Node Implementations
    # ========================================
    
    async def _load_user_intelligence(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Load comprehensive user intelligence"""
        try:
            state["current_step"] = "loading_user_intelligence"
            self.logger.info("🧠 Loading user intelligence...")
            
            if self.user_intelligence_service:
                user_id = int(state["user_context"].get("user_id", 1319))
                intelligence = await self.user_intelligence_service.get_comprehensive_user_intelligence(user_id)
                
                state["user_intelligence"] = {
                    "strategic_patterns": intelligence.strategic_patterns,
                    "success_predictors": intelligence.success_predictors,
                    "stakeholder_network": intelligence.stakeholder_network,
                    "industry_expertise": intelligence.industry_expertise,
                    "performance_trajectory": intelligence.performance_trajectory,
                    "communication_preferences": intelligence.communication_preferences,
                    "preferred_strategies": intelligence.preferred_strategies
                }
                
                self.logger.info(f"✅ User intelligence loaded: {len(state['user_intelligence'])} categories")
            else:
                state["user_intelligence"] = {"status": "service_unavailable"}
                self.logger.warning("⚠️ User intelligence service not available")
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ User intelligence loading failed: {e}")
            state["user_intelligence"] = {"error": str(e)}
            return state
    
    async def _extract_with_nlp(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Advanced NLP extraction with user context"""
        try:
            state["current_step"] = "nlp_extraction"
            self.logger.info("🧠 Performing advanced NLP extraction...")
            
            if self.revolutionary_form_filler:
                extracted_data = await self.revolutionary_form_filler.fill_complete_form(
                    state["user_message"],
                    state["user_context"]
                )
                
                state["nlp_extracted_data"] = extracted_data
                field_count = len([v for v in extracted_data.values() if v and not str(v).startswith('_')])
                
                self.logger.info(f"✅ NLP extraction completed: {field_count} fields extracted")
            else:
                state["nlp_extracted_data"] = {"error": "form_filler_unavailable"}
                self.logger.warning("⚠️ Revolutionary form filler not available")
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ NLP extraction failed: {e}")
            state["nlp_extracted_data"] = {"error": str(e)}
            return state
    
    async def _enrich_from_database(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Enrich data from database using SQL toolkit"""
        try:
            state["current_step"] = "database_enrichment"
            self.logger.info("🔍 Enriching data from database...")
            
            # Start with NLP extracted data
            enriched_data = state["nlp_extracted_data"].copy()
            
            if self.enterprise_toolkit and self.enterprise_toolkit.is_available():
                # Use SQL toolkit for database queries
                tools = self.enterprise_toolkit.get_tools()
                
                # Find SQL query tool
                sql_tool = None
                for tool in tools:
                    if "sql_query" in tool.name.lower():
                        sql_tool = tool
                        break
                
                if sql_tool and "account_id" in enriched_data:
                    # Query for similar accounts
                    account_query = f"Find similar accounts to {enriched_data['account_id']} for benchmarking"
                    similar_accounts = await sql_tool.func(account_query)
                    enriched_data["_similar_accounts"] = similar_accounts
                
                # Query for user's historical patterns
                if state["user_context"].get("user_id"):
                    user_id = state["user_context"]["user_id"]
                    pattern_query = f"Get planning patterns for user {user_id}"
                    if sql_tool:
                        user_patterns = await sql_tool.func(pattern_query)
                        enriched_data["_user_patterns"] = user_patterns
                
                self.logger.info("✅ Database enrichment completed")
            else:
                self.logger.warning("⚠️ Enterprise toolkit not available for database enrichment")
            
            state["database_enriched_data"] = enriched_data
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Database enrichment failed: {e}")
            state["database_enriched_data"] = state["nlp_extracted_data"].copy()
            return state
    
    async def _gather_market_intelligence(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Gather market intelligence using external tools"""
        try:
            state["current_step"] = "market_intelligence"
            self.logger.info("🌐 Gathering market intelligence...")
            
            market_data = {}
            
            if self.enterprise_toolkit and self.enterprise_toolkit.is_available():
                tools = self.enterprise_toolkit.get_tools()
                
                # Find market research tool
                market_tool = None
                for tool in tools:
                    if "market_intelligence" in tool.name.lower():
                        market_tool = tool
                        break
                
                if market_tool:
                    account_id = state["database_enriched_data"].get("account_id", "")
                    industry = state["database_enriched_data"].get("industry", "")
                    
                    if account_id:
                        # Research specific account
                        research_query = f"Research {account_id} market position and recent developments"
                        account_intelligence = await market_tool.func(research_query)
                        market_data["account_research"] = account_intelligence
                    
                    if industry:
                        # Research industry trends
                        industry_query = f"Analyze {industry} industry trends and opportunities 2024"
                        industry_intelligence = await market_tool.func(industry_query)
                        market_data["industry_trends"] = industry_intelligence
                    
                    self.logger.info(f"✅ Market intelligence gathered: {len(market_data)} categories")
                else:
                    self.logger.warning("⚠️ Market intelligence tool not available")
            
            state["market_intelligence"] = market_data
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Market intelligence gathering failed: {e}")
            state["market_intelligence"] = {"error": str(e)}
            return state
    
    async def _perform_analytics(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Perform advanced analytics and calculations"""
        try:
            state["current_step"] = "analytics"
            self.logger.info("📊 Performing advanced analytics...")
            
            analytics_results = {}
            
            if self.enterprise_toolkit and self.enterprise_toolkit.is_available():
                tools = self.enterprise_toolkit.get_tools()
                
                # Find analytics tool
                analytics_tool = None
                for tool in tools:
                    if "business_analytics" in tool.name.lower():
                        analytics_tool = tool
                        break
                
                if analytics_tool:
                    # Revenue forecasting
                    revenue = state["database_enriched_data"].get("annual_revenue")
                    growth_target = state["database_enriched_data"].get("revenue_growth_target")
                    
                    if revenue and growth_target:
                        forecast_request = f"Calculate revenue projections for base revenue {revenue} with {growth_target}% growth including seasonality"
                        revenue_forecast = await analytics_tool.func(forecast_request)
                        analytics_results["revenue_forecast"] = revenue_forecast
                    
                    # Risk assessment
                    known_risks = state["database_enriched_data"].get("known_risks", "")
                    if known_risks:
                        risk_request = f"Perform Monte Carlo risk assessment for strategic plan"
                        risk_analysis = await analytics_tool.func(risk_request)
                        analytics_results["risk_analysis"] = risk_analysis
                    
                    # Stakeholder analysis
                    stakeholders = state["database_enriched_data"].get("stakeholders", [])
                    if stakeholders:
                        stakeholder_request = f"Analyze stakeholder influence scores and relationships"
                        stakeholder_analysis = await analytics_tool.func(stakeholder_request)
                        analytics_results["stakeholder_analysis"] = stakeholder_analysis
                    
                    self.logger.info(f"✅ Analytics completed: {len(analytics_results)} analyses")
                else:
                    self.logger.warning("⚠️ Analytics tool not available")
            
            state["analytics_results"] = analytics_results
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Analytics failed: {e}")
            state["analytics_results"] = {"error": str(e)}
            return state
    
    async def _generate_recommendations(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Generate intelligent recommendations"""
        try:
            state["current_step"] = "generating_recommendations"
            self.logger.info("💡 Generating intelligent recommendations...")
            
            recommendations = []
            
            # Strategic recommendations based on user intelligence
            user_intelligence = state["user_intelligence"]
            if user_intelligence.get("preferred_strategies"):
                strategies = user_intelligence["preferred_strategies"]
                recommendations.append(f"Leverage your expertise in {', '.join(strategies[:2])} for maximum impact")
            
            # Market-based recommendations
            market_intelligence = state["market_intelligence"]
            if market_intelligence.get("industry_trends"):
                recommendations.append("Consider current industry trends for strategic positioning")
            
            # Analytics-based recommendations
            analytics = state["analytics_results"]
            if analytics.get("revenue_forecast"):
                recommendations.append("Revenue projections suggest focusing on high-growth opportunities")
            
            if analytics.get("risk_analysis"):
                recommendations.append("Implement comprehensive risk mitigation strategies based on analysis")
            
            # Data quality recommendations
            form_data = state["database_enriched_data"]
            missing_fields = [field for field in ["stakeholders", "planned_activities", "risk_mitigation_strategies"] 
                            if not form_data.get(field)]
            
            if missing_fields:
                recommendations.append(f"Consider adding details for: {', '.join(missing_fields)}")
            
            # User pattern recommendations
            if user_intelligence.get("strategic_patterns"):
                patterns = user_intelligence["strategic_patterns"]
                if patterns.get("typical_account_tiers"):
                    most_common_tier = max(patterns["typical_account_tiers"].items(), key=lambda x: x[1])[0]
                    recommendations.append(f"Plan aligns with your {most_common_tier} account expertise")
            
            state["recommendations"] = recommendations
            self.logger.info(f"✅ Generated {len(recommendations)} recommendations")
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation failed: {e}")
            state["recommendations"] = ["Review and validate all plan components"]
            return state
    
    async def _assemble_final_response(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Assemble the final response with all intelligence"""
        try:
            state["current_step"] = "assembling_response"
            self.logger.info("🎯 Assembling final response...")
            
            # Start with enriched form data
            final_form_data = state["database_enriched_data"].copy()
            
            # Calculate confidence score
            confidence_factors = []
            
            # NLP extraction quality
            nlp_data = state["nlp_extracted_data"]
            field_count = len([v for v in nlp_data.values() if v and not str(v).startswith('_')])
            confidence_factors.append(min(0.4, field_count / 20 * 0.4))  # Max 40% from field completion
            
            # User intelligence availability
            if state["user_intelligence"] and not state["user_intelligence"].get("error"):
                confidence_factors.append(0.2)  # 20% for user intelligence
            
            # Database enrichment
            if state["database_enriched_data"] and not state["database_enriched_data"].get("error"):
                confidence_factors.append(0.15)  # 15% for database enrichment
            
            # Market intelligence
            if state["market_intelligence"] and state["market_intelligence"] and not state["market_intelligence"].get("error"):
                confidence_factors.append(0.15)  # 15% for market intelligence
            
            # Analytics
            if state["analytics_results"] and not state["analytics_results"].get("error"):
                confidence_factors.append(0.1)  # 10% for analytics
            
            confidence_score = sum(confidence_factors)
            
            # Add intelligence metadata
            final_form_data["_intelligence_metadata"] = {
                "user_intelligence": bool(state["user_intelligence"] and not state["user_intelligence"].get("error")),
                "market_intelligence": bool(state["market_intelligence"] and not state["market_intelligence"].get("error")),
                "analytics_performed": bool(state["analytics_results"] and not state["analytics_results"].get("error")),
                "confidence_score": confidence_score,
                "processing_timestamp": datetime.now().isoformat(),
                "workflow_version": "enterprise_v1.0"
            }
            
            state["final_form_data"] = final_form_data
            state["confidence_score"] = confidence_score
            
            self.logger.info(f"✅ Final response assembled with {confidence_score:.2f} confidence")
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Final response assembly failed: {e}")
            state["final_form_data"] = state["database_enriched_data"].copy()
            state["confidence_score"] = 0.5
            return state
    
    async def _handle_error(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Handle workflow errors gracefully"""
        try:
            self.logger.error(f"🚨 Handling workflow error: {state.get('error_state', 'Unknown')}")
            
            # Ensure we have some form data
            if not state.get("final_form_data"):
                state["final_form_data"] = state.get("nlp_extracted_data", {})
            
            # Add error information
            state["final_form_data"]["_error_info"] = {
                "error": state.get("error_state", "Unknown error"),
                "failed_step": state.get("current_step", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add basic recommendations
            if not state.get("recommendations"):
                state["recommendations"] = [
                    "Workflow encountered errors - please review and validate all data",
                    "Consider retrying the operation",
                    "Contact support if issues persist"
                ]
            
            # Set low confidence
            state["confidence_score"] = 0.3
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Error handling failed: {e}")
            return state
    
    # ========================================
    # Conditional Routing Functions
    # ========================================
    
    def _should_skip_market_research(self, state: EnterprisePlanningState) -> str:
        """Decide whether to skip market research"""
        try:
            # Skip if account_id is generic or missing
            account_id = state.get("nlp_extracted_data", {}).get("account_id", "")
            
            if not account_id or len(account_id) < 3:
                return "skip_market"
            
            # Skip for internal/test accounts
            skip_keywords = ["test", "example", "demo", "internal"]
            if any(keyword in account_id.lower() for keyword in skip_keywords):
                return "skip_market"
            
            return "continue"
            
        except Exception:
            return "continue"
    
    def _should_perform_crud_operation(self, state: EnterprisePlanningState) -> str:
        """Decide whether to perform CRUD operation"""
        try:
            message = state.get("user_message", "").lower()
            
            # Check for CRUD keywords
            crud_keywords = ["delete", "update", "create", "modify", "remove"]
            if any(keyword in message for keyword in crud_keywords):
                return "crud_operation"
            
            return "continue"
            
        except Exception:
            return "continue"
    
    async def handle_crud_request(self, state: EnterprisePlanningState) -> EnterprisePlanningState:
        """Handle CRUD operation request"""
        try:
            if self.revolutionary_crud_agent:
                crud_result = await self.revolutionary_crud_agent.handle_message(
                    state["user_message"],
                    state["user_context"]
                )
                
                state["crud_result"] = crud_result
                
                # If successful, merge results
                if crud_result.get("success"):
                    state["database_enriched_data"].update({
                        "_crud_operation": crud_result
                    })
            
            return state
            
        except Exception as e:
            state["error_state"] = f"CRUD operation failed: {e}"
            return state
    
    def _format_final_response(self, state: EnterprisePlanningState) -> Dict[str, Any]:
        """Format the final response for the client"""
        try:
            response = {
                "form_prefill": state["final_form_data"],
                "recommendations": state["recommendations"],
                "confidence_score": state["confidence_score"],
                "intelligence_insights": {
                    "user_patterns": state["user_intelligence"],
                    "market_data": state["market_intelligence"],
                    "analytics": state["analytics_results"]
                },
                "workflow_metadata": {
                    "processing_time": state["processing_time"],
                    "workflow_version": "enterprise_v1.0",
                    "steps_completed": state["current_step"],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Add error information if present
            if state.get("error_state"):
                response["error_info"] = {
                    "error": state["error_state"],
                    "partial_results": True
                }
            
            return response
            
        except Exception as e:
            return self._generate_error_response(str(e), "", {})
    
    def _generate_error_response(self, error: str, message: str, user_context: Dict) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "form_prefill": {"error": error},
            "recommendations": ["Please try again or contact support"],
            "confidence_score": 0.1,
            "error": error,
            "workflow_metadata": {
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
        }

# Global instance
enterprise_workflow = None

def get_enterprise_workflow(pool_manager=None):
    """Get or create enterprise workflow instance"""
    global enterprise_workflow
    if enterprise_workflow is None:
        enterprise_workflow = EnterpriseStrategicPlanningWorkflow(pool_manager)
    return enterprise_workflow

