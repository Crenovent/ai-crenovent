"""
Enterprise Planning AI Orchestrator
Coordinates AI agents, RPA workflows, and automation tools for maximum credibility
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import uuid

from src.agents.strategic_account_planner_enhanced import StrategicAccountPlannerWithAutomation
from src.agents.form_monitor_agent import FormMonitorAgent
from src.agents.crud_agent import CRUDAgent
from src.agents.conversation_agent import ConversationAgent
from src.automation.rpa_workflows import RPAOrchestrator, RPAResult
from src.services.embedding_service import EmbeddingService
from src.services.fabric_service import FabricService
from src.services.database_service import DatabaseService as PostgresService
from src.services.validation_service import ValidationService
from src.services.notification_service import NotificationService

@dataclass
class PlanningRequest:
    user_id: str
    tenant_id: str
    session_id: str
    message: str
    account_id: Optional[str] = None
    action_type: str = "general"
    mode: str = "hybrid"
    context: Dict[str, Any] = None

@dataclass
class PlanningResponse:
    success: bool
    response: str
    suggested_actions: List[Dict]
    form_prefill: Dict[str, Any]
    confidence_score: float
    automation_summary: Dict[str, Any]
    validation_summary: Dict[str, Any]
    compliance_status: Dict[str, Any]
    data_sources: List[str]
    processing_time_ms: float
    session_id: str
    recommendations: List[str]
    error_message: Optional[str] = None

class PlanningAIOrchestrator:
    """
    Master orchestrator for enterprise planning AI with automation and RPA
    
    Capabilities:
    - AI-powered conversation and plan generation
    - Automated data collection and validation
    - RPA-driven quality assurance
    - Compliance checking and approval workflows
    - Real-time stakeholder notifications
    - Continuous learning through embeddings
    """
    
    def __init__(self):
        self.setup_services()
        self.setup_monitoring()
        self.active_sessions = {}
        self.execution_metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def setup_services(self):
        """Initialize all enterprise services"""
        self.rpa_orchestrator = RPAOrchestrator()
        self.embedding_service = EmbeddingService()
        self.fabric_service = FabricService()
        self.postgres_service = PostgresService()
        self.validation_service = ValidationService()
        self.notification_service = NotificationService()
        
        # Initialize multi-agent system
        self.form_monitor = FormMonitorAgent()
        self.crud_agent = CRUDAgent(backend_url=os.getenv('BACKEND_BASE_URL', 'http://localhost:3001'))
        self.conversation_agent = ConversationAgent(azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'))
        
        # AI agents pool for concurrent processing
        self.ai_agent_pool = {}
        self.max_concurrent_sessions = 50
        
        # Agent registry for intelligent routing
        self.specialized_agents = {
            'form_monitor': self.form_monitor,
            'crud_agent': self.crud_agent,
            'conversation_agent': self.conversation_agent
        }
    
    def setup_monitoring(self):
        """Setup monitoring and metrics collection"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0,
            "automation_success_rate": 0,
            "confidence_scores": [],
            "error_rates": {}
        }
    
    async def process_planning_request(self, request: PlanningRequest) -> PlanningResponse:
        """
        Main entry point for processing planning requests with full automation pipeline
        """
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Processing planning request {request_id} for user {request.user_id}")
            self.metrics["total_requests"] += 1
            
            # Step 1: Initialize or retrieve AI agent for user session
            ai_agent = await self.get_or_create_ai_agent(request.user_id, request.tenant_id)
            
            # Step 2: Analyze user intent and determine automation requirements
            intent_analysis = await self.analyze_user_intent(request.message, request.context)
            
            # Step 3: Execute RPA and automation pipeline based on intent
            automation_results = await self.execute_automation_pipeline(request, intent_analysis)
            
            # Step 4: Validate all collected data
            validation_results = await self.execute_validation_pipeline(automation_results)
            
            # Step 5: Check compliance requirements
            compliance_results = await self.execute_compliance_pipeline(request, automation_results)
            
            # Step 6: Generate AI response with validated data
            ai_response = await self.generate_ai_response(
                ai_agent, request, intent_analysis, automation_results, validation_results
            )
            
            # Step 7: Execute post-processing actions
            await self.execute_post_processing_actions(
                request, ai_response, automation_results, validation_results
            )
            
            # Step 8: Calculate metrics and prepare response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            overall_confidence = self.calculate_overall_confidence(
                ai_response, automation_results, validation_results
            )
            
            response = PlanningResponse(
                success=True,
                response=ai_response["response"],
                suggested_actions=ai_response["suggested_actions"],
                form_prefill=ai_response["form_prefill"],
                confidence_score=overall_confidence,
                automation_summary=self.summarize_automation_results(automation_results),
                validation_summary=self.summarize_validation_results(validation_results),
                compliance_status=compliance_results,
                data_sources=self.extract_data_sources(automation_results),
                processing_time_ms=processing_time,
                session_id=request.session_id,
                recommendations=self.generate_recommendations(
                    automation_results, validation_results, compliance_results
                )
            )
            
            # Update metrics
            self.update_success_metrics(response)
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"Planning request {request_id} failed: {str(e)}")
            
            return PlanningResponse(
                success=False,
                response="I encountered an issue processing your request. Please try again.",
                suggested_actions=[],
                form_prefill={},
                confidence_score=0.0,
                automation_summary={},
                validation_summary={},
                compliance_status={},
                data_sources=[],
                processing_time_ms=processing_time,
                session_id=request.session_id,
                recommendations=[],
                error_message=str(e)
            )
    
    async def get_or_create_ai_agent(self, user_id: str, tenant_id: str) -> StrategicAccountPlannerWithAutomation:
        """Get existing AI agent or create new one for user session"""
        agent_key = f"{user_id}_{tenant_id}"
        
        if agent_key not in self.ai_agent_pool:
            # Create new AI agent with user context
            agent = StrategicAccountPlannerWithAutomation(user_id, tenant_id)
            
            # Load user context and historical data
            await self.load_user_context(agent, user_id, tenant_id)
            
            self.ai_agent_pool[agent_key] = agent
        
        return self.ai_agent_pool[agent_key]
    
    async def analyze_user_intent(self, message: str, context: Dict) -> Dict[str, Any]:
        """Analyze user intent to determine required automation workflows"""
        try:
            # Use LLM to analyze intent
            intent_prompt = f"""
            Analyze the user's message and determine their intent for strategic account planning.
            
            Message: "{message}"
            Context: {json.dumps(context or {}, indent=2)}
            
            Determine:
            1. Primary action (create_plan, update_plan, analyze_account, get_recommendations, etc.)
            2. Required data sources (salesforce, market_intelligence, stakeholder_data, etc.)
            3. Automation requirements (rpa_validation, compliance_check, data_enrichment, etc.)
            4. Urgency level (high, medium, low)
            5. Expected response format (conversation, form_prefill, analysis_report, etc.)
            
            Return JSON format only.
            """
            
            # This would use your LLM service
            intent_analysis = await self.analyze_with_llm(intent_prompt)
            
            return intent_analysis
            
        except Exception as e:
            # Fallback intent analysis
            return {
                "primary_action": "general_assistance",
                "required_data_sources": ["salesforce"],
                "automation_requirements": ["rpa_validation"],
                "urgency_level": "medium",
                "expected_format": "conversation"
            }
    
    async def execute_automation_pipeline(self, request: PlanningRequest, intent_analysis: Dict) -> Dict[str, RPAResult]:
        """Execute automation and RPA workflows based on user intent"""
        try:
            # Prepare automation request data
            automation_request = {
                "user_id": request.user_id,
                "tenant_id": request.tenant_id,
                "account_id": request.account_id,
                "action": intent_analysis.get("primary_action"),
                "message": request.message,
                "context": request.context or {},
                "user_credentials": await self.get_user_credentials(request.user_id),
                "company_name": await self.extract_company_name(request),
                "industry": await self.extract_industry(request),
                "plan_data": request.context.get("plan_data", {}) if request.context else {},
                "account_data": await self.get_account_data(request.account_id) if request.account_id else {},
                "target_data": request.context.get("target_data", {}) if request.context else {}
            }
            
            # Execute RPA pipeline
            automation_results = await self.rpa_orchestrator.execute_planning_rpa_pipeline(
                automation_request
            )
            
            return automation_results.get("rpa_results", {})
            
        except Exception as e:
            self.logger.error(f"Automation pipeline error: {str(e)}")
            return {}
    
    async def execute_validation_pipeline(self, automation_results: Dict[str, RPAResult]) -> Dict[str, Any]:
        """Execute comprehensive data validation"""
        try:
            validation_tasks = []
            
            # Data accuracy validation
            if "salesforce_validation" in automation_results:
                validation_tasks.append(
                    self.validation_service.validate_data_accuracy(
                        automation_results["salesforce_validation"].data
                    )
                )
            
            # Market intelligence validation
            if "market_intelligence" in automation_results:
                validation_tasks.append(
                    self.validation_service.validate_market_data(
                        automation_results["market_intelligence"].data
                    )
                )
            
            # Stakeholder data validation
            if "stakeholder_identification" in automation_results:
                validation_tasks.append(
                    self.validation_service.validate_stakeholder_data(
                        automation_results["stakeholder_identification"].data
                    )
                )
            
            # Execute all validations in parallel
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Aggregate validation results
            aggregated_validation = {
                "overall_passed": all(
                    result.get("passed", False) for result in validation_results 
                    if not isinstance(result, Exception)
                ),
                "validation_scores": [
                    result.get("score", 0.0) for result in validation_results 
                    if not isinstance(result, Exception)
                ],
                "failed_validations": [
                    str(result) for result in validation_results 
                    if isinstance(result, Exception)
                ],
                "detailed_results": [
                    result for result in validation_results 
                    if not isinstance(result, Exception)
                ]
            }
            
            return aggregated_validation
            
        except Exception as e:
            self.logger.error(f"Validation pipeline error: {str(e)}")
            return {"overall_passed": False, "error": str(e)}
    
    async def execute_compliance_pipeline(self, request: PlanningRequest, automation_results: Dict) -> Dict[str, Any]:
        """Execute compliance checking pipeline"""
        try:
            compliance_checks = []
            
            # GDPR compliance
            compliance_checks.append(
                self.check_gdpr_compliance(request, automation_results)
            )
            
            # SOX compliance
            compliance_checks.append(
                self.check_sox_compliance(request, automation_results)
            )
            
            # Industry-specific compliance
            compliance_checks.append(
                self.check_industry_compliance(request, automation_results)
            )
            
            # Data retention compliance
            compliance_checks.append(
                self.check_data_retention_compliance(request, automation_results)
            )
            
            compliance_results = await asyncio.gather(*compliance_checks, return_exceptions=True)
            
            return {
                "overall_compliant": all(
                    result.get("compliant", False) for result in compliance_results 
                    if not isinstance(result, Exception)
                ),
                "compliance_score": sum(
                    result.get("score", 0.0) for result in compliance_results 
                    if not isinstance(result, Exception)
                ) / len(compliance_results) if compliance_results else 0.0,
                "detailed_checks": [
                    result for result in compliance_results 
                    if not isinstance(result, Exception)
                ],
                "compliance_issues": [
                    str(result) for result in compliance_results 
                    if isinstance(result, Exception)
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Compliance pipeline error: {str(e)}")
            return {"overall_compliant": False, "error": str(e)}
    
    async def generate_ai_response(self, 
                                 ai_agent: StrategicAccountPlannerWithAutomation,
                                 request: PlanningRequest,
                                 intent_analysis: Dict,
                                 automation_results: Dict,
                                 validation_results: Dict) -> Dict[str, Any]:
        """Generate AI response enhanced with automation data"""
        try:
            # Prepare enhanced context for AI
            enhanced_context = {
                "user_message": request.message,
                "intent": intent_analysis,
                "automation_data": {
                    name: result.data for name, result in automation_results.items()
                    if result.success
                },
                "validation_status": validation_results,
                "data_confidence": {
                    name: result.confidence_score for name, result in automation_results.items()
                }
            }
            
            # Generate AI response with enhanced context
            ai_response = await ai_agent.process_enhanced_request(
                message=request.message,
                session_id=request.session_id,
                enhanced_context=enhanced_context
            )
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"AI response generation error: {str(e)}")
            return {
                "response": "I encountered an issue generating a response. Please try again.",
                "suggested_actions": [],
                "form_prefill": {},
                "confidence": 0.0
            }
    
    async def execute_post_processing_actions(self,
                                            request: PlanningRequest,
                                            ai_response: Dict,
                                            automation_results: Dict,
                                            validation_results: Dict):
        """Execute automated post-processing actions"""
        try:
            post_processing_tasks = []
            
            # Create embeddings for new data
            if any(result.success for result in automation_results.values()):
                post_processing_tasks.append(
                    self.create_embeddings_for_new_data(
                        request, ai_response, automation_results
                    )
                )
            
            # Send notifications if required
            if ai_response.get("suggested_actions"):
                post_processing_tasks.append(
                    self.send_automated_notifications(
                        request, ai_response, automation_results
                    )
                )
            
            # Update user context and learning
            post_processing_tasks.append(
                self.update_user_learning_context(
                    request, ai_response, automation_results
                )
            )
            
            # Log analytics and metrics
            post_processing_tasks.append(
                self.log_analytics_data(
                    request, ai_response, automation_results, validation_results
                )
            )
            
            # Execute all post-processing tasks
            await asyncio.gather(*post_processing_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Post-processing error: {str(e)}")
    
    async def create_embeddings_for_new_data(self, request: PlanningRequest, ai_response: Dict, automation_results: Dict):
        """Create embeddings for new data to enhance future AI responses"""
        try:
            embedding_tasks = []
            
            for workflow_name, result in automation_results.items():
                if result.success and result.data:
                    embedding_tasks.append(
                        self.embedding_service.create_workflow_embeddings(
                            workflow_name=workflow_name,
                            data=result.data,
                            user_id=request.user_id,
                            tenant_id=request.tenant_id,
                            confidence_score=result.confidence_score
                        )
                    )
            
            # Create conversation embedding
            if ai_response.get("response"):
                embedding_tasks.append(
                    self.embedding_service.create_conversation_embedding(
                        user_message=request.message,
                        ai_response=ai_response["response"],
                        context=automation_results,
                        user_id=request.user_id,
                        tenant_id=request.tenant_id
                    )
                )
            
            await asyncio.gather(*embedding_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Embedding creation error: {str(e)}")
    
    def calculate_overall_confidence(self, ai_response: Dict, automation_results: Dict, validation_results: Dict) -> float:
        """Calculate overall confidence score for the response"""
        try:
            confidence_scores = []
            
            # AI confidence
            if ai_response.get("confidence"):
                confidence_scores.append(ai_response["confidence"])
            
            # Automation confidence scores
            for result in automation_results.values():
                if result.success:
                    confidence_scores.append(result.confidence_score)
            
            # Validation confidence
            if validation_results.get("validation_scores"):
                confidence_scores.extend(validation_results["validation_scores"])
            
            if confidence_scores:
                # Weighted average with higher weight for AI and validation
                weights = [0.4] + [0.3 / len(automation_results)] * len(automation_results) + [0.3]
                weights = weights[:len(confidence_scores)]
                
                weighted_confidence = sum(
                    score * weight for score, weight in zip(confidence_scores, weights)
                ) / sum(weights)
                
                return min(1.0, max(0.0, weighted_confidence))
            
            return 0.5  # Default moderate confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {str(e)}")
            return 0.0
    
    def generate_recommendations(self, automation_results: Dict, validation_results: Dict, compliance_results: Dict) -> List[str]:
        """Generate actionable recommendations based on automation and validation results"""
        recommendations = []
        
        try:
            # Data quality recommendations
            if not validation_results.get("overall_passed", True):
                recommendations.append("Consider updating data sources for improved accuracy")
            
            # Compliance recommendations
            if not compliance_results.get("overall_compliant", True):
                recommendations.append("Review compliance requirements before finalizing plans")
            
            # Automation-specific recommendations
            for workflow_name, result in automation_results.items():
                if result.success and result.confidence_score < 0.8:
                    recommendations.append(f"Verify {workflow_name} data manually for higher confidence")
            
            # Performance recommendations
            if automation_results and all(result.execution_time_ms > 5000 for result in automation_results.values()):
                recommendations.append("Consider optimizing data sources for faster response times")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {str(e)}")
            return ["Review all data before making final decisions"]
    
    async def route_to_specialized_agent(self, 
                                       request: PlanningRequest) -> Dict[str, Any]:
        """Route requests to specialized agents based on message content"""
        
        try:
            user_context = {
                'user_id': request.user_id,
                'tenant_id': request.tenant_id,
                'session_id': request.session_id,
                'mode': request.mode,
                'context': request.context or {}
            }
            
            # Always update conversation context first
            conversation_response = await self.conversation_agent.handle_message(
                request.message, user_context
            )
            
            # Check which specialized agents should handle this
            agent_responses = []
            
            # Check form monitoring
            if await self.form_monitor.should_handle_message(request.message, user_context):
                form_response = await self.form_monitor.handle_message(request.message, user_context)
                agent_responses.append(form_response)
            
            # Check CRUD operations
            if await self.crud_agent.should_handle_message(request.message, user_context):
                crud_response = await self.crud_agent.handle_message(request.message, user_context)
                agent_responses.append(crud_response)
            
            # If no specialized agents handled it, use the main strategic planner
            if not agent_responses:
                # Use existing strategic planner logic
                planning_response = await self.process_planning_request(request)
                
                # Convert PlanningResponse to dict format
                if hasattr(planning_response, '__dict__'):
                    return {
                        "success": planning_response.success,
                        "response": planning_response.response,
                        "suggested_actions": planning_response.suggested_actions,
                        "form_prefill": planning_response.form_prefill,
                        "confidence_score": planning_response.confidence_score,
                        "automation_summary": planning_response.automation_summary,
                        "validation_summary": planning_response.validation_summary,
                        "compliance_status": planning_response.compliance_status,
                        "data_sources": planning_response.data_sources,
                        "processing_time_ms": planning_response.processing_time_ms,
                        "recommendations": planning_response.recommendations,
                        "error_message": planning_response.error_message
                    }
                else:
                    return planning_response
            
            # Combine responses from specialized agents
            combined_response = {
                "success": True,
                "message": "Multi-agent processing completed",
                "agent_responses": agent_responses,
                "conversation_context": conversation_response,
                "processing_time_ms": 0  # Will be calculated by caller
            }
            
            # Add conversation response to memory
            for response in agent_responses:
                if response.get('agent'):
                    await self.conversation_agent.add_agent_response(
                        request.user_id,
                        request.session_id,
                        response['agent'],
                        response.get('response', '')
                    )
            
            return combined_response
            
        except Exception as e:
            self.logger.error(f"❌ Multi-agent routing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Error in multi-agent processing"
            }
    
    async def handle_form_monitoring(self, 
                                   user_id: str,
                                   session_id: str,
                                   field_name: str,
                                   field_value: Any,
                                   form_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle real-time form field monitoring"""
        
        try:
            suggestions = await self.form_monitor.monitor_field_change(
                user_id, field_name, field_value, form_context
            )
            
            return [
                {
                    "field": suggestion.field_name,
                    "suggested_value": suggestion.suggested_value,
                    "confidence": suggestion.confidence,
                    "reasoning": suggestion.reasoning,
                    "source": suggestion.source
                }
                for suggestion in suggestions
            ]
            
        except Exception as e:
            self.logger.error(f"❌ Form monitoring error: {e}")
            return []




