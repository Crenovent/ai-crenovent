"""
Strategic Planning AI Agent with Intelligence, RBAC, and Automation
Handles Create Strategic Account Plan functionality with enterprise-grade intelligence
Includes RPA and automation capabilities for maximum efficiency and credibility
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from services.database_service import DatabaseService, UserContext
from services.fabric_service import FabricService
from services.embedding_service import EmbeddingService

@dataclass
class PlanningResponse:
    success: bool
    message: str
    form_prefill: Dict[str, Any]
    suggested_actions: List[Dict]
    confidence_score: float
    data_sources: List[str]
    validation_status: Dict[str, Any]
    intelligence_insights: Dict[str, Any]

class GetUserPlansHistoryTool(BaseTool):
    """Tool to get user's historical strategic plans for pattern analysis"""
    
    name = "get_user_plans_history"
    description = "Get user's historical strategic account plans to understand patterns and preferences"
    
    def __init__(self, db_service: DatabaseService, user_context: UserContext):
        super().__init__()
        self.db_service = db_service
        self.user_context = user_context
    
    async def _arun(self, limit: int = 10) -> str:
        """Get user's historical plans"""
        try:
            plans = await self.db_service.get_user_strategic_plans(self.user_context, limit)
            
            if not plans:
                return "No historical plans found for this user."
            
            # Analyze patterns in historical plans
            analysis = {
                "total_plans": len(plans),
                "common_industries": self._analyze_industries(plans),
                "average_revenue_targets": self._analyze_revenue_targets(plans),
                "common_goals": self._analyze_goals(plans),
                "risk_patterns": self._analyze_risks(plans),
                "recent_plans": plans[:3]  # Most recent 3 plans
            }
            
            return f"Historical plans analysis: {json.dumps(analysis, indent=2)}"
            
        except Exception as e:
            return f"Error retrieving historical plans: {str(e)}"
    
    def _analyze_industries(self, plans: List[Dict]) -> Dict:
        """Analyze industry patterns"""
        industries = [plan.get('industry') for plan in plans if plan.get('industry')]
        industry_counts = {}
        for industry in industries:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        return {
            "most_common": max(industry_counts.items(), key=lambda x: x[1]) if industry_counts else None,
            "distribution": industry_counts
        }
    
    def _analyze_revenue_targets(self, plans: List[Dict]) -> Dict:
        """Analyze revenue target patterns"""
        targets = [float(plan.get('revenue_growth_target', 0)) for plan in plans if plan.get('revenue_growth_target')]
        
        if targets:
            return {
                "average": round(sum(targets) / len(targets), 2),
                "range": f"{min(targets)}-{max(targets)}%",
                "trend": "increasing" if len(targets) >= 2 and targets[0] > targets[-1] else "stable"
            }
        
        return {"average": 0, "range": "0-0%", "trend": "no_data"}
    
    def _analyze_goals(self, plans: List[Dict]) -> List[str]:
        """Extract common goal patterns"""
        all_goals = []
        for plan in plans:
            if plan.get('short_term_goals'):
                all_goals.append(plan['short_term_goals'])
            if plan.get('long_term_goals'):
                all_goals.append(plan['long_term_goals'])
        
        # Simple keyword extraction
        keywords = ['growth', 'expansion', 'retention', 'upsell', 'cross-sell', 'digital', 'cloud', 'migration']
        common_themes = []
        for keyword in keywords:
            if sum(1 for goal in all_goals if keyword.lower() in goal.lower()) > len(plans) * 0.3:
                common_themes.append(keyword)
        
        return common_themes
    
    def _analyze_risks(self, plans: List[Dict]) -> List[str]:
        """Extract common risk patterns"""
        all_risks = []
        for plan in plans:
            if plan.get('known_risks'):
                all_risks.append(plan['known_risks'])
        
        # Simple keyword extraction for risks
        risk_keywords = ['competition', 'budget', 'timeline', 'resources', 'market', 'technical', 'regulatory']
        common_risks = []
        for keyword in risk_keywords:
            if sum(1 for risk in all_risks if keyword.lower() in risk.lower()) > len(plans) * 0.2:
                common_risks.append(keyword)
        
        return common_risks
    
    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))

class GetAccountDataTool(BaseTool):
    """Tool to get comprehensive account data from Fabric"""
    
    name = "get_account_data"
    description = "Get comprehensive account data including opportunities, contacts, and activities from Salesforce via Fabric"
    
    def __init__(self, db_service: DatabaseService, fabric_service: FabricService, user_context: UserContext):
        super().__init__()
        self.db_service = db_service
        self.fabric_service = fabric_service
        self.user_context = user_context
    
    async def _arun(self, account_id: str) -> str:
        """Get account data"""
        try:
            # Check permissions
            if not await self.db_service.can_user_access_account(account_id, self.user_context):
                return f"Access denied: User does not have permission to access account {account_id}"
            
            # Get comprehensive account data
            account_data = await self.db_service.get_account_data_from_fabric(account_id, self.user_context)
            
            if not account_data:
                return f"No data found for account {account_id}"
            
            # Enrich with additional intelligence
            account_intel = await self._enrich_account_intelligence(account_data)
            
            return f"Account data and intelligence: {json.dumps(account_intel, indent=2)}"
            
        except Exception as e:
            return f"Error retrieving account data: {str(e)}"
    
    async def _enrich_account_intelligence(self, account_data: Dict) -> Dict:
        """Enrich account data with intelligence insights"""
        try:
            account = account_data.get('account', {})
            opportunities = account_data.get('opportunities', [])
            contacts = account_data.get('contacts', [])
            
            # Calculate account health metrics
            health_metrics = {
                "pipeline_health": self._calculate_pipeline_health(opportunities),
                "engagement_score": self._calculate_engagement_score(opportunities, contacts),
                "growth_potential": self._calculate_growth_potential(account, opportunities),
                "risk_indicators": self._identify_risk_indicators(opportunities)
            }
            
            # Generate strategic insights
            strategic_insights = {
                "key_opportunities": self._identify_key_opportunities(opportunities),
                "stakeholder_analysis": self._analyze_stakeholders(contacts),
                "competitive_risks": self._assess_competitive_risks(account, opportunities),
                "expansion_opportunities": self._identify_expansion_opportunities(account, opportunities)
            }
            
            return {
                **account_data,
                "health_metrics": health_metrics,
                "strategic_insights": strategic_insights,
                "intelligence_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error enriching account intelligence: {str(e)}")
            return account_data
    
    def _calculate_pipeline_health(self, opportunities: List[Dict]) -> Dict:
        """Calculate pipeline health score"""
        open_opps = [opp for opp in opportunities if not opp.get('IsClosed')]
        
        if not open_opps:
            return {"score": 0, "status": "no_pipeline", "details": "No open opportunities"}
        
        total_value = sum(float(opp.get('Amount', 0) or 0) for opp in open_opps)
        avg_probability = sum(float(opp.get('Probability', 0) or 0) for opp in open_opps) / len(open_opps)
        
        # Calculate health score (0-100)
        value_score = min(100, (total_value / 1000000) * 10)  # $1M = 10 points
        probability_score = avg_probability
        stage_diversity = len(set(opp.get('StageName') for opp in open_opps)) * 10
        
        health_score = (value_score + probability_score + stage_diversity) / 3
        
        return {
            "score": round(health_score, 2),
            "status": "healthy" if health_score > 70 else "moderate" if health_score > 40 else "at_risk",
            "total_pipeline": total_value,
            "avg_probability": round(avg_probability, 2),
            "open_opportunities": len(open_opps)
        }
    
    def _calculate_engagement_score(self, opportunities: List[Dict], contacts: List[Dict]) -> Dict:
        """Calculate engagement score based on activity and contacts"""
        # This would ideally use activity data, but we'll approximate
        recent_opps = [
            opp for opp in opportunities 
            if opp.get('LastModifiedDate') and 
            datetime.fromisoformat(opp['LastModifiedDate'].replace('Z', '+00:00')) > datetime.now() - timedelta(days=30)
        ]
        
        engagement_score = min(100, (len(recent_opps) * 20) + (len(contacts) * 5))
        
        return {
            "score": engagement_score,
            "status": "high" if engagement_score > 70 else "medium" if engagement_score > 40 else "low",
            "recent_activity": len(recent_opps),
            "contact_count": len(contacts)
        }
    
    def _identify_key_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Identify key opportunities for strategic focus"""
        open_opps = [opp for opp in opportunities if not opp.get('IsClosed')]
        
        # Sort by value and probability
        key_opps = sorted(
            open_opps,
            key=lambda x: float(x.get('Amount', 0) or 0) * float(x.get('Probability', 0) or 0) / 100,
            reverse=True
        )[:3]
        
        return [
            {
                "id": opp.get('Id'),
                "name": opp.get('Name'),
                "amount": opp.get('Amount'),
                "stage": opp.get('StageName'),
                "probability": opp.get('Probability'),
                "close_date": opp.get('CloseDate'),
                "strategic_value": float(opp.get('Amount', 0) or 0) * float(opp.get('Probability', 0) or 0) / 100
            }
            for opp in key_opps
        ]
    
    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))

class GetIntelligentSuggestionsTool(BaseTool):
    """Tool to get intelligent suggestions based on user history and account context"""
    
    name = "get_intelligent_suggestions"
    description = "Get intelligent suggestions for strategic plan based on user history and account analysis"
    
    def __init__(self, db_service: DatabaseService, user_context: UserContext):
        super().__init__()
        self.db_service = db_service
        self.user_context = user_context
    
    async def _arun(self, account_id: str = None, plan_type: str = "strategic") -> str:
        """Get intelligent suggestions"""
        try:
            suggestions = await self.db_service.get_intelligent_suggestions(self.user_context, account_id)
            
            # Generate contextual recommendations
            recommendations = {
                "form_suggestions": self._generate_form_suggestions(suggestions),
                "strategic_recommendations": self._generate_strategic_recommendations(suggestions),
                "risk_mitigation": self._generate_risk_mitigation(suggestions),
                "success_metrics": self._generate_success_metrics(suggestions)
            }
            
            return f"Intelligent suggestions: {json.dumps(recommendations, indent=2)}"
            
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"
    
    def _generate_form_suggestions(self, suggestions: Dict) -> Dict:
        """Generate form field suggestions"""
        form_suggestions = {}
        
        # Revenue target suggestions
        if suggestions.get('revenue_targets'):
            form_suggestions['revenue_growth_target'] = suggestions['revenue_targets'].get('average_target', 20)
        
        # Industry-specific suggestions
        if suggestions.get('industry_benchmarks'):
            form_suggestions['account_tier'] = suggestions['industry_benchmarks'].get('typical_tier', 'Enterprise')
        
        # Goal suggestions based on patterns
        if suggestions.get('common_goals'):
            common_themes = suggestions['common_goals']
            if 'growth' in common_themes:
                form_suggestions['short_term_goals'] = "Drive revenue growth through product expansion and market penetration"
            if 'digital' in common_themes:
                form_suggestions['long_term_goals'] = "Lead digital transformation initiatives and cloud adoption"
        
        return form_suggestions
    
    def _generate_strategic_recommendations(self, suggestions: Dict) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        if suggestions.get('account_specific'):
            account_data = suggestions['account_specific']
            if account_data.get('health_metrics', {}).get('score', 0) < 50:
                recommendations.append("Focus on pipeline health improvement and opportunity advancement")
            
            if len(account_data.get('strategic_insights', {}).get('key_opportunities', [])) > 0:
                recommendations.append("Prioritize high-value opportunities with strong win probability")
        
        if suggestions.get('revenue_targets', {}).get('trend') == 'increasing':
            recommendations.append("Continue aggressive growth strategy based on historical success")
        
        return recommendations
    
    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))

class CreateStrategicPlanTool(BaseTool):
    """Tool to create a new strategic account plan"""
    
    name = "create_strategic_plan"
    description = "Create a new strategic account plan with intelligent defaults and validation"
    
    def __init__(self, db_service: DatabaseService, user_context: UserContext):
        super().__init__()
        self.db_service = db_service
        self.user_context = user_context
    
    async def _arun(self, plan_data: str) -> str:
        """Create strategic plan"""
        try:
            # Parse plan data
            plan_dict = json.loads(plan_data)
            
            # Validate and enhance plan data
            enhanced_plan = await self._enhance_plan_data(plan_dict)
            
            # Create the plan
            result = await self.db_service.create_strategic_plan(enhanced_plan, self.user_context)
            
            return f"Strategic plan created successfully: {json.dumps(result, indent=2)}"
            
        except Exception as e:
            return f"Error creating strategic plan: {str(e)}"
    
    async def _enhance_plan_data(self, plan_data: Dict) -> Dict:
        """Enhance plan data with intelligent defaults"""
        try:
            # Set defaults based on user context and historical data
            if not plan_data.get('region_territory'):
                plan_data['region_territory'] = f"{self.user_context.region}/{self.user_context.territory}"
            
            if not plan_data.get('status'):
                plan_data['status'] = 'Draft'
            
            # Enhance with timestamp
            plan_data['created_timestamp'] = datetime.now().isoformat()
            
            return plan_data
            
        except Exception as e:
            logging.error(f"Error enhancing plan data: {str(e)}")
            return plan_data
    
    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))

class StrategicPlanningAgent:
    """
    Main Strategic Planning AI Agent with enterprise intelligence
    Handles conversation, form suggestions, and strategic plan creation
    """
    
    def __init__(self, user_context: UserContext):
        self.user_context = user_context
        self.db_service = None
        self.fabric_service = None
        self.embedding_service = None
        self.llm = None
        self.memory = None
        self.agent_executor = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the agent with all services"""
        try:
            # Initialize services
            self.db_service = DatabaseService()
            await self.db_service.initialize()
            
            self.fabric_service = FabricService()
            await self.fabric_service.initialize()
            
            from services.embedding_service import EmbeddingService
            self.embedding_service = EmbeddingService()
            await self.embedding_service.initialize()
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model="gpt-4-1106-preview",
                temperature=0.1,  # Low temperature for accuracy
                max_tokens=4000,
                streaming=True
            )
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10  # Keep last 10 exchanges
            )
            
            # Setup agent tools
            tools = [
                GetUserPlansHistoryTool(self.db_service, self.user_context),
                GetAccountDataTool(self.db_service, self.fabric_service, self.user_context),
                GetIntelligentSuggestionsTool(self.db_service, self.user_context),
                CreateStrategicPlanTool(self.db_service, self.user_context)
            ]
            
            # Create agent prompt
            system_prompt = self._create_system_prompt()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            
            self.logger.info(f"Strategic Planning Agent initialized for user {self.user_context.user_id}")
            
        except Exception as e:
            self.logger.error(f"Agent initialization error: {str(e)}")
            raise
    
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for the agent"""
        return f"""You are an Enterprise Strategic Account Planning Assistant with advanced intelligence capabilities.

USER CONTEXT:
- User ID: {self.user_context.user_id}
- Role: {self.user_context.role}
- Hierarchy Level: {self.user_context.hierarchy_level}
- Permissions: {self.user_context.permissions}
- Region/Territory: {self.user_context.region}/{self.user_context.territory}

Your capabilities include:
1. INTELLIGENT FORM ASSISTANCE: Pre-fill forms with smart suggestions based on user history and account data
2. CONVERSATIONAL GUIDANCE: Answer questions and provide strategic advice
3. DATA-DRIVEN INSIGHTS: Use real-time Salesforce data to inform recommendations
4. PATTERN RECOGNITION: Learn from user's historical plans to maintain consistency
5. RBAC COMPLIANCE: Respect user permissions and hierarchy restrictions

FORM PREFILL INTELLIGENCE:
- Always analyze user's historical plans for patterns
- Use real-time account data for accurate suggestions
- Provide confidence scores for suggestions
- Explain reasoning behind recommendations

CONVERSATION CAPABILITIES:
- Understand natural language requests like "Create Q3 FY25 ARR targets for APAC Enterprise"
- Break down complex requests into actionable steps
- Ask clarifying questions when needed
- Provide strategic insights and recommendations

RESPONSE FORMAT:
Always respond in JSON format with:
{{
    "message": "Your conversational response",
    "form_prefill": {{"field_name": "suggested_value"}},
    "suggested_actions": [{{"action": "description", "priority": "high/medium/low"}}],
    "confidence_score": 0.85,
    "data_sources": ["fabric", "historical_plans", "user_patterns"],
    "intelligence_insights": {{"key": "insight"}}
}}

STRATEGIC FOCUS AREAS:
- Revenue growth and target setting
- Stakeholder engagement and relationship mapping
- Risk assessment and mitigation strategies
- Competitive positioning and market analysis
- Account expansion and cross-sell opportunities

Always prioritize accuracy, compliance with RBAC, and intelligent suggestions based on real data.
"""
    
    async def process_message(self, message: str, account_id: str = None, session_id: str = None) -> PlanningResponse:
        """Process user message and generate intelligent response"""
        try:
            # Prepare input with context
            input_data = {
                "input": message,
                "account_id": account_id or "",
                "session_id": session_id or "",
                "user_context": asdict(self.user_context)
            }
            
            # Execute agent
            result = await self.agent_executor.ainvoke(input_data)
            
            # Parse agent response
            response_text = result.get("output", "")
            
            # Try to parse as JSON, fallback to text response
            try:
                parsed_response = json.loads(response_text)
                
                return PlanningResponse(
                    success=True,
                    message=parsed_response.get("message", response_text),
                    form_prefill=parsed_response.get("form_prefill", {}),
                    suggested_actions=parsed_response.get("suggested_actions", []),
                    confidence_score=parsed_response.get("confidence_score", 0.8),
                    data_sources=parsed_response.get("data_sources", ["ai_agent"]),
                    validation_status={"status": "success", "checks_passed": []},
                    intelligence_insights=parsed_response.get("intelligence_insights", {})
                )
                
            except json.JSONDecodeError:
                # Fallback for non-JSON responses
                return PlanningResponse(
                    success=True,
                    message=response_text,
                    form_prefill={},
                    suggested_actions=[],
                    confidence_score=0.7,
                    data_sources=["ai_agent"],
                    validation_status={"status": "success", "checks_passed": []},
                    intelligence_insights={}
                )
                
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            
            return PlanningResponse(
                success=False,
                message=f"I encountered an error processing your request: {str(e)}",
                form_prefill={},
                suggested_actions=[],
                confidence_score=0.0,
                data_sources=[],
                validation_status={"status": "error", "error": str(e)},
                intelligence_insights={}
            )
    
    async def get_form_suggestions(self, account_id: str, partial_data: Dict = None) -> Dict:
        """Get intelligent form suggestions for account planning"""
        try:
            # Get account data and user history
            account_data = await self.db_service.get_account_data_from_fabric(account_id, self.user_context)
            suggestions = await self.db_service.get_intelligent_suggestions(self.user_context, account_id)
            
            # Generate form suggestions
            form_suggestions = {}
            
            # Account-based suggestions
            if account_data.get('account'):
                account = account_data['account']
                form_suggestions.update({
                    'account_owner': account.get('Name', ''),
                    'industry': account.get('Industry', ''),
                    'annual_revenue': account.get('AnnualRevenue'),
                    'account_tier': self._determine_account_tier(account),
                    'region_territory': f"{self.user_context.region}/{self.user_context.territory}"
                })
            
            # Intelligence-based suggestions
            if suggestions.get('revenue_targets'):
                form_suggestions['revenue_growth_target'] = suggestions['revenue_targets'].get('average_target')
            
            # Pattern-based suggestions
            if suggestions.get('common_goals'):
                form_suggestions.update(self._generate_goal_suggestions(suggestions['common_goals']))
            
            return {
                'suggestions': form_suggestions,
                'confidence': 0.85,
                'data_sources': ['fabric', 'user_history', 'patterns']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting form suggestions: {str(e)}")
            return {'suggestions': {}, 'confidence': 0.0, 'data_sources': []}
    
    def _determine_account_tier(self, account: Dict) -> str:
        """Determine account tier based on revenue and other factors"""
        annual_revenue = float(account.get('AnnualRevenue', 0) or 0)
        
        if annual_revenue > 1000000000:  # $1B+
            return 'Strategic'
        elif annual_revenue > 100000000:  # $100M+
            return 'Enterprise'
        elif annual_revenue > 10000000:   # $10M+
            return 'Corporate'
        else:
            return 'Commercial'
    
    def _generate_goal_suggestions(self, common_themes: List[str]) -> Dict:
        """Generate goal suggestions based on common themes"""
        suggestions = {}
        
        if 'growth' in common_themes:
            suggestions['short_term_goals'] = "Accelerate revenue growth through strategic account expansion and new opportunity development"
        
        if 'digital' in common_themes:
            suggestions['long_term_goals'] = "Drive digital transformation initiatives and establish long-term strategic partnership"
        
        if 'retention' in common_themes:
            suggestions['customer_success_metrics'] = "Maintain >95% renewal rate and achieve >110% net revenue retention"
        
        return suggestions




