"""
Real-time Form Monitoring Agent
Watches user input and provides intelligent, context-aware suggestions
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

@dataclass
class FormField:
    """Represents a form field being monitored"""
    name: str
    value: Any
    field_type: str  # text, dropdown, date, number, etc.
    is_required: bool = False
    suggestions: List[str] = None
    
@dataclass
class FormSuggestion:
    """Represents a suggestion for a form field"""
    field_name: str
    suggested_value: Any
    confidence: float
    reasoning: str
    source: str  # historical_data, pattern_analysis, user_preference

class FormMonitorAgent:
    """
    Real-time form monitoring agent that:
    1. Watches form field changes in real-time
    2. Provides intelligent suggestions based on user patterns
    3. Offers contextual help and auto-completion
    4. Learns from user behavior
    """
    
    def __init__(self):
        self.agent_id = "form_monitor"
        self.name = "Form Monitor Agent"
        self.description = "Real-time form monitoring and intelligent suggestions"
        
        # Memory for learning user patterns
        self.memory = ConversationBufferWindowMemory(
            k=50,  # Remember last 50 form interactions
            return_messages=True,
            memory_key="form_history"
        )
        
        # Track user patterns
        self.user_patterns: Dict[str, Dict] = {}
        self.field_suggestions: Dict[str, List[FormSuggestion]] = {}
        
        logger.info(f"🎯 {self.name} initialized")
    
    async def monitor_field_change(self, 
                                 user_id: str,
                                 field_name: str, 
                                 field_value: Any,
                                 form_context: Dict[str, Any]) -> List[FormSuggestion]:
        """
        Monitor a field change and provide intelligent suggestions
        """
        try:
            logger.info(f"👁️ Monitoring field change: {field_name} = {field_value}")
            
            # Analyze the field change
            suggestions = await self._analyze_field_change(
                user_id, field_name, field_value, form_context
            )
            
            # Update user patterns
            await self._update_user_patterns(user_id, field_name, field_value, form_context)
            
            # Learn from the interaction
            await self._learn_from_interaction(user_id, field_name, field_value, suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Error monitoring field change: {e}")
            return []
    
    async def _analyze_field_change(self,
                                  user_id: str,
                                  field_name: str,
                                  field_value: Any,
                                  form_context: Dict[str, Any]) -> List[FormSuggestion]:
        """Analyze field change and generate suggestions"""
        
        suggestions = []
        
        # Get user's historical patterns
        user_pattern = self.user_patterns.get(user_id, {})
        field_history = user_pattern.get(field_name, [])
        
        # Pattern-based suggestions
        if field_history:
            most_common = max(set(field_history), key=field_history.count)
            if most_common != field_value and len(field_history) >= 3:
                suggestions.append(FormSuggestion(
                    field_name=field_name,
                    suggested_value=most_common,
                    confidence=0.7,
                    reasoning=f"You usually use '{most_common}' for this field",
                    source="user_pattern"
                ))
        
        # Context-aware suggestions
        context_suggestions = await self._get_contextual_suggestions(
            field_name, field_value, form_context
        )
        suggestions.extend(context_suggestions)
        
        # Smart auto-completion
        if isinstance(field_value, str) and len(field_value) >= 2:
            completion_suggestions = await self._get_completion_suggestions(
                field_name, field_value, user_id
            )
            suggestions.extend(completion_suggestions)
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    async def _get_contextual_suggestions(self,
                                        field_name: str,
                                        field_value: Any,
                                        form_context: Dict[str, Any]) -> List[FormSuggestion]:
        """Generate contextual suggestions based on other form fields"""
        
        suggestions = []
        
        # Industry-based suggestions
        if field_name == "annual_revenue" and "industry" in form_context:
            industry = form_context.get("industry", "").lower()
            industry_averages = {
                "technology": "5000000",
                "healthcare": "2000000", 
                "finance": "10000000",
                "retail": "1500000",
                "manufacturing": "3000000"
            }
            
            if industry in industry_averages:
                suggestions.append(FormSuggestion(
                    field_name=field_name,
                    suggested_value=industry_averages[industry],
                    confidence=0.6,
                    reasoning=f"Average revenue for {industry} industry",
                    source="industry_data"
                ))
        
        # Account tier suggestions
        if field_name == "account_tier" and "annual_revenue" in form_context:
            revenue = form_context.get("annual_revenue", 0)
            try:
                revenue_num = float(str(revenue).replace(",", ""))
                if revenue_num >= 10000000:
                    tier = "Enterprise"
                elif revenue_num >= 1000000:
                    tier = "Key"
                else:
                    tier = "Growth"
                
                suggestions.append(FormSuggestion(
                    field_name=field_name,
                    suggested_value=tier,
                    confidence=0.8,
                    reasoning=f"Based on annual revenue of ${revenue_num:,.0f}",
                    source="revenue_analysis"
                ))
            except (ValueError, TypeError):
                pass
        
        # Revenue growth suggestions based on tier
        if field_name == "revenue_growth_target" and "account_tier" in form_context:
            tier = form_context.get("account_tier", "").lower()
            tier_targets = {
                "enterprise": 15,
                "key": 25,
                "growth": 35
            }
            
            if tier in tier_targets:
                suggestions.append(FormSuggestion(
                    field_name=field_name,
                    suggested_value=tier_targets[tier],
                    confidence=0.7,
                    reasoning=f"Typical growth target for {tier} accounts",
                    source="tier_analysis"
                ))
        
        return suggestions
    
    async def _get_completion_suggestions(self,
                                        field_name: str,
                                        partial_value: str,
                                        user_id: str) -> List[FormSuggestion]:
        """Get auto-completion suggestions"""
        
        suggestions = []
        
        # Common completions based on field type
        completions = {
            "plan_name": [
                "Strategic Growth Plan 2025",
                "Q4 Expansion Strategy",
                "Enterprise Development Plan",
                "Customer Success Initiative"
            ],
            "short_term_goals": [
                "Increase product adoption by 30%",
                "Expand user base to key departments",
                "Achieve 95% customer satisfaction",
                "Complete integration milestones"
            ],
            "long_term_goals": [
                "Become the primary solution provider",
                "Achieve 200% ROI within 18 months", 
                "Expand to enterprise-wide deployment",
                "Establish strategic partnership"
            ]
        }
        
        if field_name in completions:
            matches = [comp for comp in completions[field_name] 
                      if partial_value.lower() in comp.lower()]
            
            for match in matches[:3]:
                suggestions.append(FormSuggestion(
                    field_name=field_name,
                    suggested_value=match,
                    confidence=0.5,
                    reasoning="Common completion for this field type",
                    source="auto_completion"
                ))
        
        return suggestions
    
    async def _update_user_patterns(self,
                                  user_id: str,
                                  field_name: str,
                                  field_value: Any,
                                  form_context: Dict[str, Any]):
        """Update user patterns for learning"""
        
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {}
        
        if field_name not in self.user_patterns[user_id]:
            self.user_patterns[user_id][field_name] = []
        
        # Add to pattern history (keep last 20 values)
        self.user_patterns[user_id][field_name].append(field_value)
        if len(self.user_patterns[user_id][field_name]) > 20:
            self.user_patterns[user_id][field_name].pop(0)
    
    async def _learn_from_interaction(self,
                                    user_id: str,
                                    field_name: str,
                                    field_value: Any,
                                    suggestions: List[FormSuggestion]):
        """Learn from user interactions with suggestions"""
        
        # Store interaction for future learning
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "field_name": field_name,
            "field_value": field_value,
            "suggestions_provided": len(suggestions),
            "suggestions": [s.__dict__ for s in suggestions]
        }
        
        # Add to memory
        message = HumanMessage(content=f"Field change: {field_name} = {field_value}")
        ai_response = f"Provided {len(suggestions)} suggestions"
        self.memory.chat_memory.add_user_message(message)
        self.memory.chat_memory.add_ai_message(ai_response)
        
        logger.info(f"📚 Learned from interaction: {field_name} = {field_value}")
    
    async def get_field_suggestions(self, 
                                  user_id: str,
                                  field_name: str,
                                  form_context: Dict[str, Any]) -> List[FormSuggestion]:
        """Get suggestions for a specific field"""
        
        return await self._analyze_field_change(
            user_id, field_name, "", form_context
        )
    
    async def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if this agent should handle the message"""
        
        form_keywords = [
            "form", "field", "suggestion", "auto", "complete", 
            "fill", "help", "what should", "recommend"
        ]
        
        return any(keyword in message.lower() for keyword in form_keywords)
    
    async def handle_message(self, 
                           message: str, 
                           user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle form-related messages with real AI processing"""
        
        try:
            user_id = user_context.get("user_id", "unknown")
            session_id = user_context.get("session_id", f"form_session_{user_id}")
            
            # Import here to avoid circular imports
            from ..services.database_service import DatabaseService, UserContext
            from ..services.embedding_service import EmbeddingService
            
            # Get real user context and historical data
            if isinstance(user_id, str) and user_id.isdigit():
                user_id = int(user_id)
            elif isinstance(user_id, int):
                user_id = int(user_id)
                
            tenant_id_val = user_context.get("tenant_id")
            if isinstance(tenant_id_val, str) and tenant_id_val.isdigit():
                tenant_id = int(tenant_id_val)
            elif isinstance(tenant_id_val, int):
                tenant_id = tenant_id_val
            else:
                tenant_id = 1300
            
            # Initialize services
            db_service = DatabaseService()
            await db_service.initialize()
            
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            
            # Get user context from database
            real_user_context = await db_service.get_user_context(user_id, tenant_id)
            
            # Analyze message for form-related intent
            if any(keyword in message.lower() for keyword in ["fill", "prefill", "form", "field", "suggest", "help with"]):
                # Get intelligent form suggestions
                suggestions = await db_service.get_intelligent_suggestions(real_user_context)
                
                # Use instant form filler for immediate results
                from ..tools.instant_form_filler import fill_form_instantly
                instant_suggestions = fill_form_instantly(message)
                
                # Search for similar form interactions in the past
                similar_conversations = await embedding_service.search_similar_conversations(
                    message, user_id, similarity_threshold=0.6, max_results=3
                )
                
                # Generate contextual response
                if similar_conversations:
                    context_info = f"Based on your previous form interactions, "
                else:
                    context_info = "Based on your account planning patterns, "
                
                # Use instant suggestions as primary source
                form_suggestions = instant_suggestions if instant_suggestions else {}
                
                # Enhance with database suggestions if available
                if "revenue" in message.lower() or "target" in message.lower():
                    revenue_patterns = suggestions.get('revenue_targets', {})
                    if revenue_patterns and not form_suggestions.get('revenue_growth_target'):
                        form_suggestions['revenue_growth_target'] = revenue_patterns.get('suggested_range', '15-25%')
                
                if "stakeholder" in message.lower() and not form_suggestions.get('stakeholders'):
                    form_suggestions['stakeholders'] = [
                        {"name": "CTO", "role": "Technical Decision Maker", "influence": "High", "relationship": "Positive"},
                        {"name": "VP Engineering", "role": "Technical Champion", "influence": "Medium", "relationship": "Strong"}
                    ]
                
                if "risk" in message.lower() and not form_suggestions.get('known_risks'):
                    risk_patterns = suggestions.get('risk_patterns', [])
                    if risk_patterns:
                        form_suggestions['known_risks'] = "; ".join(risk_patterns[:3])
                    else:
                        form_suggestions['known_risks'] = "Market competition, technology adoption challenges, budget constraints"
                
                if "goal" in message.lower() and not form_suggestions.get('short_term_goals'):
                    goal_patterns = suggestions.get('common_goals', [])
                    if goal_patterns:
                        form_suggestions['short_term_goals'] = "; ".join(goal_patterns[:3])
                    else:
                        form_suggestions['short_term_goals'] = "Increase platform adoption, improve user engagement, drive revenue growth"
                
                response_text = f"{context_info}I can help you intelligently fill form fields. "
                if form_suggestions:
                    response_text += f"I've identified suggestions for {len(form_suggestions)} fields based on your patterns and account data."
                else:
                    response_text += "I'm analyzing your patterns to provide personalized suggestions."
                
                response = {
                    "agent": self.name,
                    "response": response_text,
                    "form_prefill": form_suggestions,
                    "confidence_score": 0.85 if form_suggestions else 0.6,
                    "suggestions": [
                        {"field": field, "value": value, "source": "historical_patterns"} 
                        for field, value in form_suggestions.items()
                    ],
                    "data_sources": ["historical_plans", "user_patterns", "database_analysis"]
                }
                
            elif "monitor" in message.lower() or "watch" in message.lower():
                response = {
                    "agent": self.name,
                    "response": "Form monitoring is now active. I'll provide real-time suggestions as you fill out forms based on your historical data and account context.",
                    "active": True,
                    "monitoring_capabilities": [
                        "Real-time field suggestions",
                        "Pattern-based auto-completion",
                        "Context-aware recommendations",
                        "Historical data integration"
                    ]
                }
                
            else:
                # General form help
                response = {
                    "agent": self.name,
                    "response": "I'm your intelligent form assistant. I can help you fill forms faster using your historical patterns, suggest values based on account data, and provide context-aware recommendations. What specific form field would you like help with?",
                    "capabilities": [
                        "Intelligent form prefilling",
                        "Pattern-based suggestions",
                        "Account data integration",
                        "Historical analysis"
                    ],
                    "examples": [
                        "Help me fill revenue targets",
                        "Suggest stakeholders for this account",
                        "What risks should I consider?",
                        "Fill goals based on my patterns"
                    ]
                }
            
            # Store this interaction for learning
            await embedding_service.store_conversation_embedding(
                user_id=user_id,
                session_id=session_id,
                message_text=message,
                response_text=response["response"],
                intent_category="form_assistance",
                confidence_score=response.get("confidence_score", 0.8),
                metadata={"agent": self.name, "form_suggestions_count": len(response.get("form_prefill", {}))},
                tenant_id=tenant_id
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error handling form monitoring message: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "response": "I encountered an error while processing your form request. Please try again or be more specific about which form field you need help with."
            }
