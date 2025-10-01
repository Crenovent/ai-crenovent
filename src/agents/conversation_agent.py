"""
Conversation Memory Agent
Manages conversational context and history using LangChain
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Represents conversation context for a user session"""
    user_id: str
    session_id: str
    started_at: datetime
    last_activity: datetime
    message_count: int
    topics: List[str]
    entities_mentioned: List[str]
    current_task: Optional[str] = None

class ConversationAgent:
    """
    Conversation Memory Agent that:
    1. Maintains conversation history across sessions
    2. Provides context to other agents
    3. Summarizes long conversations
    4. Tracks topics and entities mentioned
    5. Helps maintain coherent multi-turn interactions
    """
    
    def __init__(self, azure_openai_api_key: Optional[str] = None):
        self.agent_id = "conversation_agent"
        self.name = "Conversation Memory Agent"
        self.description = "Conversation context and memory management"
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI for conversation summarization
        if azure_openai_api_key:
            self.llm = AzureChatOpenAI(
                api_key=azure_openai_api_key,
                api_version="2024-02-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                temperature=0.3
            )
        else:
            self.llm = None
            logger.warning("⚠️ Azure OpenAI API key not provided - summarization disabled")
        
        # Session memories - using different memory types for different needs
        self.session_memories: Dict[str, ConversationBufferWindowMemory] = {}
        self.session_contexts: Dict[str, ConversationContext] = {}
        
        # Long-term memory for important context (simplified to avoid deprecation)
        self.long_term_memory: Dict[str, ChatMessageHistory] = {}
        
        logger.info(f"💭 {self.name} initialized")
    
    async def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """This agent always processes messages to maintain context"""
        return True  # Always handle to maintain conversation context
    
    async def handle_message(self, 
                           message: str, 
                           user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conversational messages with intelligent responses and data integration"""
        
        try:
            user_id = user_context.get("user_id", "unknown")
            session_id = user_context.get("session_id", "default")
            tenant_id = user_context.get("tenant_id", 1300)
            session_key = f"{user_id}_{session_id}"
            
            # Import only what we need for simplified mode
            # from ..services.database_service import DatabaseService, UserContext
            # from ..services.embedding_service import EmbeddingService
            # from ..tools.form_prefill_tool import form_prefill_tool
            
            # Convert user_id to int for database operations
            if isinstance(user_id, str) and user_id.isdigit():
                user_id_int = int(user_id)
            elif isinstance(user_id, int):
                user_id_int = user_id
            else:
                user_id_int = 1319  # fallback
                
            if isinstance(tenant_id, str) and tenant_id.isdigit():
                tenant_id = int(tenant_id)
            elif isinstance(tenant_id, int):
                tenant_id = int(tenant_id)
            
            # Enable database mode for smart form filling
            real_user_context = None
            self.logger.info("🚀 Using smart database mode for enhanced form filling")
            
            # Update conversation context
            await self._update_conversation_context(session_key, message, user_id, session_id)
            
            # Add to memory
            await self._add_to_memory(session_key, message, user_context)
            
            # Extract entities and topics
            entities = await self._extract_entities(message)
            topics = await self._extract_topics(message)
            
            # Update session context
            if session_key in self.session_contexts:
                context_obj = self.session_contexts[session_key]
                context_obj.entities_mentioned.extend(entities)
                context_obj.topics.extend(topics)
                context_obj.last_activity = datetime.now()
                context_obj.message_count += 1
                
                # Keep only unique items
                context_obj.entities_mentioned = list(set(context_obj.entities_mentioned))
                context_obj.topics = list(set(context_obj.topics))
            
            # Generate form prefill data using revolutionary form filler (enterprise-grade AI)
            from ..tools.revolutionary_form_filler import get_revolutionary_form_filler
            from ..services.connection_pool_manager import pool_manager
            
            revolutionary_filler = get_revolutionary_form_filler(pool_manager)
            form_prefill_data = await revolutionary_filler.fill_complete_form(message, {
                'user_id': user_id_int,
                'tenant_id': tenant_id,
                'session_id': session_id
            })
            
            # Generate intelligent, data-driven response
            response_text, data_sources, confidence = await self._generate_smart_response(
                message, form_prefill_data, user_id_int, tenant_id
            )
            
            response_data = {
                "response": response_text,
                "form_prefill": form_prefill_data,
                "confidence_score": confidence,
                "data_sources": data_sources,
                "suggested_actions": self._generate_smart_actions(form_prefill_data, message),
                "intelligence_insights": await self._generate_insights(form_prefill_data, user_id_int, tenant_id)
            }
            
            self.logger.info(f"🎯 Generated response with {len(form_prefill_data)} form fields: {list(form_prefill_data.keys()) if form_prefill_data else []}")
            
            # Check if conversation should be summarized
            if self.llm and self._should_summarize(session_key):
                await self._summarize_conversation(session_key)
            
            # Skip embedding storage in simplified mode
            self.logger.info("📝 Skipped conversation embedding storage (simplified mode)")
            
            return {
                "agent": self.name,
                "response": response_data["response"],
                "form_prefill": response_data.get("form_prefill", {}),
                "suggested_actions": response_data.get("suggested_actions", []),
                "confidence_score": response_data.get("confidence", 0.8),
                "data_sources": response_data.get("data_sources", ["conversation_memory"]),
                "context_summary": await self._get_context_summary(session_key),
                "entities_mentioned": entities,
                "topics_discussed": topics,
                "memory_active": True,
                "intelligence_insights": response_data.get("insights", {})
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in conversation memory handling: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "response": "I encountered an error while processing your message. Please try rephrasing your question."
            }
    
    async def _generate_smart_response(self, message: str, form_data: Dict, user_id: int, tenant_id: int) -> tuple:
        """Generate intelligent, context-aware response based on data"""
        try:
            from ..services.connection_pool_manager import pool_manager
            
            account_name = form_data.get('account_id', 'the account')
            arr_value = form_data.get('annual_revenue', '')
            field_count = len(form_data)
            
            # Check if we have historical data for this user
            has_historical = False
            user_plan_count = 0
            
            if pool_manager and pool_manager.postgres_pool:
                try:
                    async with pool_manager.postgres_pool.acquire() as conn:
                        user_plan_count = await conn.fetchval(
                            "SELECT COUNT(*) FROM strategic_account_plans WHERE created_by_user_id = $1 AND tenant_id = $2",
                            user_id, tenant_id
                        )
                        has_historical = user_plan_count > 0
                except:
                    pass
            
            # Generate contextual response
            if has_historical and field_count > 5:
                if arr_value:
                    arr_formatted = f"${int(arr_value):,}" if arr_value.isdigit() else arr_value
                    response = f"Perfect! I've created a strategic plan for {account_name} with an ARR target of {arr_formatted}. Based on your {user_plan_count} previous plans, I've pre-filled {field_count} fields with intelligent suggestions including goals, stakeholders, and risk strategies from your successful planning patterns."
                else:
                    response = f"I've prepared a comprehensive strategic plan for {account_name}. Drawing from your {user_plan_count} previous plans, I've intelligently pre-filled {field_count} fields including your typical goals, stakeholders, and strategies."
                
                data_sources = ["user_historical_data", "smart_database_analysis"]
                confidence = 0.95
                
            elif field_count > 3:
                if arr_value:
                    arr_formatted = f"${int(arr_value):,}" if arr_value.isdigit() else arr_value
                    response = f"Great! I've created a strategic plan for {account_name} with an ARR target of {arr_formatted}. I've intelligently pre-filled {field_count} key fields with industry best practices and smart defaults based on your request."
                else:
                    response = f"I've prepared a strategic plan for {account_name} with {field_count} intelligently pre-filled fields based on your requirements and industry standards."
                
                data_sources = ["smart_pattern_matching", "industry_defaults"]
                confidence = 0.85
                
            else:
                response = f"I'll help you create a strategic plan for {account_name}. I've started with some basic information - please provide more details about revenue targets, goals, and stakeholders so I can generate more comprehensive suggestions."
                data_sources = ["basic_extraction"]
                confidence = 0.70
            
            # Add user-specific insights if available
            if has_historical:
                if user_plan_count == 1:
                    response += " I notice this builds on your previous planning experience."
                elif user_plan_count > 5:
                    response += f" Your extensive planning history ({user_plan_count} plans) allows me to provide highly personalized suggestions."
                else:
                    response += f" I've leveraged insights from your {user_plan_count} previous plans for better accuracy."
            
            return response, data_sources, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error generating smart response: {e}")
            return "I'll help you create a strategic account plan.", ["fallback"], 0.6
    
    def _generate_smart_actions(self, form_data: Dict, message: str) -> List[Dict]:
        """Generate contextual suggested actions"""
        actions = []
        
        if len(form_data) > 5:
            actions.append({"action": "Review the auto-filled form fields", "priority": "high"})
            actions.append({"action": "Modify any fields that need adjustment", "priority": "medium"})
            actions.append({"action": "Submit the comprehensive plan", "priority": "high"})
        elif len(form_data) > 0:
            actions.append({"action": "Review the pre-filled fields", "priority": "high"})
            actions.append({"action": "Add more specific goals and stakeholders", "priority": "medium"})
            actions.append({"action": "Complete remaining fields before submitting", "priority": "high"})
        else:
            actions.append({"action": "Provide account name and revenue target", "priority": "high"})
            actions.append({"action": "Specify planning goals and objectives", "priority": "medium"})
        
        # Add context-specific actions
        if 'cloud' in message.lower():
            actions.append({"action": "Consider cloud migration timeline", "priority": "medium"})
        if 'renewal' in message.lower():
            actions.append({"action": "Plan renewal strategy and timeline", "priority": "high"})
        
        return actions
    
    async def _generate_insights(self, form_data: Dict, user_id: int, tenant_id: int) -> Dict:
        """Generate intelligent insights about the plan"""
        insights = {}
        
        try:
            # Revenue insights
            if form_data.get('annual_revenue'):
                revenue = int(form_data['annual_revenue']) if form_data['annual_revenue'].isdigit() else 0
                if revenue > 5000000:
                    insights['revenue_category'] = 'enterprise_major'
                    insights['complexity'] = 'high'
                elif revenue > 1000000:
                    insights['revenue_category'] = 'enterprise'
                    insights['complexity'] = 'medium'
                else:
                    insights['revenue_category'] = 'growth'
                    insights['complexity'] = 'standard'
            
            # Tier insights
            tier = form_data.get('account_tier', '').lower()
            if tier == 'enterprise':
                insights['stakeholder_count_recommendation'] = '5-8 key stakeholders'
                insights['review_frequency'] = 'Monthly'
            elif tier in ['key', 'strategic']:
                insights['stakeholder_count_recommendation'] = '3-5 key stakeholders'
                insights['review_frequency'] = 'Quarterly'
            else:
                insights['stakeholder_count_recommendation'] = '2-4 key stakeholders'
                insights['review_frequency'] = 'Bi-annual'
            
            # Field completion analysis
            total_possible_fields = 15  # Estimate of total form fields
            completion_rate = (len(form_data) / total_possible_fields) * 100
            insights['completion_percentage'] = min(100, int(completion_rate))
            
            if completion_rate > 80:
                insights['readiness'] = 'ready_to_submit'
            elif completion_rate > 50:
                insights['readiness'] = 'needs_minor_completion'
            else:
                insights['readiness'] = 'needs_significant_input'
            
        except Exception as e:
            self.logger.error(f"❌ Error generating insights: {e}")
            insights['error'] = 'Unable to generate insights'
        
        return insights
    
    async def _update_conversation_context(self, 
                                         session_key: str,
                                         message: str,
                                         user_id: str,
                                         session_id: str):
        """Update conversation context for session"""
        
        if session_key not in self.session_contexts:
            self.session_contexts[session_key] = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                started_at=datetime.now(),
                last_activity=datetime.now(),
                message_count=0,
                topics=[],
                entities_mentioned=[]
            )
        
        # Detect current task from message
        task_keywords = {
            "plan": ["create plan", "strategic plan", "account plan"],
            "stakeholder": ["stakeholder", "contact", "champion"],
            "activity": ["activity", "task", "meeting", "call"],
            "analysis": ["analyze", "analysis", "report", "summary"],
            "crud": ["update", "change", "delete", "modify", "create"]
        }
        
        current_task = None
        for task, keywords in task_keywords.items():
            if any(keyword in message.lower() for keyword in keywords):
                current_task = task
                break
        
        if current_task:
            self.session_contexts[session_key].current_task = current_task
    
    async def _add_to_memory(self, 
                           session_key: str,
                           message: str,
                           user_context: Dict[str, Any]):
        """Add message to conversation memory"""
        
        # Initialize memory if not exists
        if session_key not in self.session_memories:
            self.session_memories[session_key] = ConversationBufferWindowMemory(
                k=20,  # Keep last 20 messages
                return_messages=True,
                memory_key="chat_history"
            )
        
        # Add human message
        human_message = HumanMessage(content=message)
        self.session_memories[session_key].chat_memory.add_user_message(message)
        
        # Also add to long-term memory if available (simplified)
        if session_key not in self.long_term_memory:
            self.long_term_memory[session_key] = ChatMessageHistory()
        
        if session_key in self.long_term_memory:
            self.long_term_memory[session_key].add_user_message(message)
    
    async def _extract_entities(self, message: str) -> List[str]:
        """Extract entities (account names, plan names, etc.) from message"""
        
        entities = []
        
        # Extract account names
        import re
        account_patterns = [
            r'\baccount\s+([A-Za-z0-9\-_]+)',
            r'\bplan\s+for\s+([A-Za-z0-9\-_]+)',
            r'\b([A-Z][A-Za-z0-9\-_]*)\s+account\b'
        ]
        
        for pattern in account_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            entities.extend(matches)
        
        # Extract plan names
        plan_patterns = [
            r'\bplan\s+([A-Za-z0-9\-_\s]+?)(?:\s|$)',
            r'\bstrategic\s+([A-Za-z0-9\-_\s]+?)\s+plan'
        ]
        
        for pattern in plan_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            entities.extend([m.strip() for m in matches])
        
        # Extract stakeholder names
        stakeholder_patterns = [
            r'\bstakeholder\s+([A-Za-z\s]+?)(?:\s|$|,)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'  # Names like "John Smith"
        ]
        
        for pattern in stakeholder_patterns:
            matches = re.findall(pattern, message)
            entities.extend([m.strip() for m in matches])
        
        return list(set([e for e in entities if len(e) > 1]))
    
    async def _extract_topics(self, message: str) -> List[str]:
        """Extract topics from message"""
        
        topics = []
        
        # Define topic keywords
        topic_keywords = {
            "account_planning": ["account plan", "strategic plan", "planning"],
            "revenue": ["revenue", "arr", "growth target", "sales"],
            "stakeholders": ["stakeholder", "champion", "decision maker"],
            "activities": ["activity", "meeting", "call", "task"],
            "goals": ["goal", "objective", "target", "milestone"],
            "risks": ["risk", "threat", "challenge", "issue"],
            "opportunities": ["opportunity", "expansion", "upsell", "cross-sell"],
            "analysis": ["analysis", "report", "summary", "data"],
            "automation": ["automate", "rpa", "agent", "ai"],
            "database": ["database", "update", "create", "delete", "modify"]
        }
        
        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _should_summarize(self, session_key: str) -> bool:
        """Determine if conversation should be summarized"""
        
        if session_key not in self.session_memories:
            return False
        
        memory = self.session_memories[session_key]
        
        # Summarize if we have more than 15 messages
        return len(memory.chat_memory.messages) > 15
    
    async def _summarize_conversation(self, session_key: str):
        """Summarize conversation to save memory"""
        
        if not self.llm or session_key not in self.session_memories:
            return
        
        try:
            memory = self.session_memories[session_key]
            
            # Get messages to summarize (keep last 5, summarize the rest)
            messages = memory.chat_memory.messages
            if len(messages) <= 10:
                return
            
            messages_to_summarize = messages[:-5]
            recent_messages = messages[-5:]
            
            # Create summary prompt
            summary_prompt = "Please summarize the following conversation, focusing on:\n"
            summary_prompt += "- Key decisions made\n"
            summary_prompt += "- Important entities mentioned (accounts, plans, stakeholders)\n"
            summary_prompt += "- Current tasks or objectives\n"
            summary_prompt += "- Any CRUD operations performed\n\n"
            summary_prompt += "Conversation:\n"
            
            for msg in messages_to_summarize:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                summary_prompt += f"{role}: {msg.content}\n"
            
            # Get summary from LLM
            summary_response = await self.llm.ainvoke(summary_prompt)
            summary = summary_response.content
            
            # Replace old messages with summary
            memory.chat_memory.clear()
            memory.chat_memory.add_ai_message(f"[CONVERSATION SUMMARY]: {summary}")
            
            # Add back recent messages
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    memory.chat_memory.add_user_message(msg.content)
                else:
                    memory.chat_memory.add_ai_message(msg.content)
            
            logger.info(f"📝 Summarized conversation for session {session_key}")
            
        except Exception as e:
            logger.error(f"❌ Error summarizing conversation: {e}")
    
    async def _get_context_summary(self, session_key: str) -> Dict[str, Any]:
        """Get summary of current conversation context"""
        
        if session_key not in self.session_contexts:
            return {}
        
        context = self.session_contexts[session_key]
        memory = self.session_memories.get(session_key)
        
        return {
            "session_duration_minutes": int((datetime.now() - context.started_at).total_seconds() / 60),
            "message_count": context.message_count,
            "current_task": context.current_task,
            "topics_discussed": context.topics[:5],  # Top 5 topics
            "entities_mentioned": context.entities_mentioned[:10],  # Top 10 entities
            "memory_length": len(memory.chat_memory.messages) if memory else 0,
            "last_activity": context.last_activity.isoformat()
        }
    
    async def get_conversation_context(self, 
                                     user_id: str, 
                                     session_id: str) -> Dict[str, Any]:
        """Get conversation context for other agents"""
        
        session_key = f"{user_id}_{session_id}"
        
        if session_key not in self.session_memories:
            return {"context": "No conversation history"}
        
        memory = self.session_memories[session_key]
        context = self.session_contexts.get(session_key)
        
        # Get recent messages
        recent_messages = []
        for msg in memory.chat_memory.messages[-5:]:  # Last 5 messages
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            recent_messages.append({
                "role": role,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()  # Approximate
            })
        
        return {
            "recent_messages": recent_messages,
            "context_summary": await self._get_context_summary(session_key) if context else {},
            "memory_available": True
        }
    
    async def add_agent_response(self, 
                               user_id: str,
                               session_id: str,
                               agent_name: str,
                               response: str):
        """Add agent response to conversation memory"""
        
        session_key = f"{user_id}_{session_id}"
        
        if session_key not in self.session_memories:
            return
        
        # Add AI response to memory
        ai_message = f"[{agent_name}]: {response}"
        self.session_memories[session_key].chat_memory.add_ai_message(ai_message)
        
        # Also add to long-term memory
        if session_key in self.long_term_memory:
            self.long_term_memory[session_key].add_ai_message(ai_message)
    
    async def cleanup_old_sessions(self, hours: int = 24):
        """Clean up old conversation sessions"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        sessions_to_remove = []
        
        for session_key, context in self.session_contexts.items():
            if context.last_activity < cutoff_time:
                sessions_to_remove.append(session_key)
        
        for session_key in sessions_to_remove:
            # Remove from all memory stores
            self.session_memories.pop(session_key, None)
            self.session_contexts.pop(session_key, None)
            self.long_term_memory.pop(session_key, None)
            
            logger.info(f"🗑️ Cleaned up old session: {session_key}")
        
        return len(sessions_to_remove)
    
    async def _generate_intelligent_response(self,
                                           message: str,
                                           session_key: str,
                                           user_context: Optional[Any],
                                           db_service: Any,
                                           embedding_service: Any) -> Dict[str, Any]:
        """Generate intelligent response based on message content and user context"""
        
        try:
            message_lower = message.lower()
            
            # Check if this is a data query
            if any(keyword in message_lower for keyword in [
                "how many", "count", "show me", "list", "what", "who", "when", "where"
            ]):
                if user_context:
                    # Use intelligent query system
                    query_result = await db_service.intelligent_query(message, user_context)
                    
                    if query_result and not query_result.get("error"):
                        return {
                            "response": query_result["answer"],
                            "data_sources": ["postgresql", "fabric", "intelligent_query"],
                            "confidence": query_result.get("confidence", 0.9),
                            "intent": "data_query",
                            "insights": {
                                "query_type": query_result.get("query_type", "unknown"),
                                "data_count": len(query_result.get("data", []))
                            }
                        }
            
            # Check if this is a greeting or general conversation
            if any(keyword in message_lower for keyword in [
                "hello", "hi", "hey", "good morning", "good afternoon", "thanks", "thank you"
            ]):
                greeting_response = self._generate_contextual_greeting(message_lower, session_key, user_context)
                return {
                    "response": greeting_response,
                    "data_sources": ["conversation_context"],
                    "confidence": 0.9,
                    "intent": "greeting"
                }
            
            # Check if this is a help request
            if any(keyword in message_lower for keyword in [
                "help", "what can you do", "how does this work", "capabilities"
            ]):
                help_response = self._generate_help_response(user_context)
                return {
                    "response": help_response,
                    "data_sources": ["system_capabilities"],
                    "confidence": 0.95,
                    "intent": "help_request",
                    "suggested_actions": [
                        {"action": "Try asking about your strategic plans", "priority": "high"},
                        {"action": "Ask for form assistance", "priority": "medium"},
                        {"action": "Request data analysis", "priority": "medium"}
                    ]
                }
            
            # Check for strategic planning related queries
            if any(keyword in message_lower for keyword in [
                "plan", "strategic", "account", "revenue", "goal", "stakeholder", "risk"
            ]):
                if user_context:
                    # Get relevant historical data
                    try:
                        suggestions = await db_service.get_intelligent_suggestions(user_context)
                        similar_plans = await embedding_service.search_similar_plans(
                            message, user_context.tenant_id, similarity_threshold=0.7, max_results=3
                        )
                        
                        planning_response = self._generate_planning_response(
                            message, suggestions, similar_plans, user_context
                        )
                        
                        return {
                            "response": planning_response["response"],
                            "form_prefill": planning_response.get("form_prefill", {}),
                            "data_sources": ["historical_plans", "intelligent_suggestions", "embeddings"],
                            "confidence": 0.85,
                            "intent": "strategic_planning",
                            "insights": {
                                "similar_plans_found": len(similar_plans),
                                "suggestions_available": len(suggestions)
                            }
                        }
                    except Exception as e:
                        logger.warning(f"Could not get planning suggestions: {e}")
            
            # Use LLM for general conversation if available
            if self.llm:
                try:
                    # Get conversation context
                    context_summary = ""
                    if session_key in self.session_contexts:
                        ctx = self.session_contexts[session_key]
                        context_summary = f"Session context: {ctx.message_count} messages exchanged. "
                        if ctx.topics:
                            context_summary += f"Recent topics: {', '.join(ctx.topics[-3:])}. "
                        if ctx.current_task:
                            context_summary += f"Current task: {ctx.current_task}. "
                    
                    # Create conversation prompt
                    system_prompt = """You are an intelligent AI assistant for strategic account planning. 
                    You help users with planning activities, data analysis, and form assistance.
                    Be helpful, concise, and professional. If you don't have specific data, acknowledge that 
                    and suggest how the user can get the information they need."""
                    
                    full_prompt = f"{system_prompt}\n\nContext: {context_summary}\n\nUser: {message}\n\nAssistant:"
                    
                    llm_response = await self.llm.ainvoke(full_prompt)
                    
                    return {
                        "response": llm_response.content,
                        "data_sources": ["azure_openai", "conversation_context"],
                        "confidence": 0.8,
                        "intent": "general_conversation"
                    }
                    
                except Exception as e:
                    logger.warning(f"LLM response generation failed: {e}")
            
            # Fallback response
            return {
                "response": "I understand you're asking about strategic planning. I can help you with creating plans, analyzing data, and filling forms. Could you be more specific about what you'd like to do?",
                "data_sources": ["fallback"],
                "confidence": 0.6,
                "intent": "general",
                "suggested_actions": [
                    {"action": "Ask 'How many strategic plans do I have?'", "priority": "high"},
                    {"action": "Say 'Help me create a strategic plan'", "priority": "high"},
                    {"action": "Request 'Show me my opportunities'", "priority": "medium"}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating intelligent response: {e}")
            return {
                "response": "I encountered an error while processing your request. Please try rephrasing your question.",
                "data_sources": ["error_handler"],
                "confidence": 0.3,
                "intent": "error"
            }
    
    def _generate_contextual_greeting(self, message: str, session_key: str, user_context: Any) -> str:
        """Generate contextual greeting based on conversation history"""
        
        # Check if this is a returning user
        if session_key in self.session_contexts:
            ctx = self.session_contexts[session_key]
            if ctx.message_count > 1:
                if ctx.current_task:
                    return f"Hello! Welcome back. I see we were working on {ctx.current_task}. How can I help you continue?"
                else:
                    return f"Hello again! We've exchanged {ctx.message_count} messages so far. What would you like to work on today?"
        
        # New session greeting
        greetings = {
            "good morning": "Good morning! Ready to tackle some strategic planning today?",
            "good afternoon": "Good afternoon! How can I assist with your account planning needs?",
            "thanks": "You're welcome! I'm here to help with strategic planning whenever you need it.",
            "thank you": "My pleasure! Feel free to ask me about strategic plans, data analysis, or form assistance."
        }
        
        for keyword, response in greetings.items():
            if keyword in message:
                return response
        
        return "Hello! I'm your AI assistant for strategic account planning. I can help you create plans, analyze data, fill forms, and answer questions about your accounts and opportunities. What would you like to work on?"
    
    def _generate_help_response(self, user_context: Any) -> str:
        """Generate help response based on user permissions and context"""
        
        capabilities = [
            "📊 **Data Analysis**: Ask questions like 'How many strategic plans do I have?' or 'Show me opportunities closing this quarter'",
            "📋 **Form Assistance**: Get intelligent suggestions for form fields based on your historical patterns",
            "🔍 **Information Retrieval**: Search through your plans, accounts, and opportunities using natural language",
            "🤖 **Conversation**: I maintain context across our conversation and learn from your patterns"
        ]
        
        if user_context and hasattr(user_context, 'permissions'):
            if user_context.permissions.get('create'):
                capabilities.append("➕ **Plan Creation**: Help you create new strategic account plans with intelligent suggestions")
            if user_context.permissions.get('edit'):
                capabilities.append("✏️ **Data Updates**: Perform database operations like updating plan details")
        
        help_text = "I'm your intelligent AI assistant for strategic account planning. Here's what I can do:\n\n"
        help_text += "\n".join(capabilities)
        help_text += "\n\n**Try asking**: 'Show me my recent plans', 'Help me fill revenue targets', or 'What opportunities are closing this quarter?'"
        
        return help_text
    
    def _generate_planning_response(self, message: str, suggestions: Dict, similar_plans: List, user_context: Any) -> Dict[str, Any]:
        """Generate response for strategic planning queries"""
        
        response_parts = []
        form_prefill = {}
        
        # Use suggestions if available
        if suggestions:
            if "revenue" in message.lower() and suggestions.get('revenue_targets'):
                revenue_info = suggestions['revenue_targets']
                response_parts.append(f"Based on your historical patterns, your average revenue growth target is {revenue_info.get('average_target', 'N/A')}%.")
                form_prefill['revenue_growth_target'] = revenue_info.get('suggested_range', '15-25%')
            
            if "goal" in message.lower() and suggestions.get('common_goals'):
                goals = suggestions['common_goals'][:3]
                if goals:
                    response_parts.append(f"Your most common planning goals include: {', '.join(goals)}.")
                    form_prefill['short_term_goals'] = '; '.join(goals)
            
            if "risk" in message.lower() and suggestions.get('risk_patterns'):
                risks = suggestions['risk_patterns'][:3]
                if risks:
                    response_parts.append(f"Common risks in your plans: {', '.join(risks)}.")
                    form_prefill['known_risks'] = '; '.join(risks)
        
        # Use similar plans if found
        if similar_plans:
            response_parts.append(f"I found {len(similar_plans)} similar plans that might be relevant.")
            
            # Extract patterns from similar plans
            for plan in similar_plans[:2]:  # Use top 2 similar plans
                metadata = plan.metadata
                if metadata.get('plan_id'):
                    response_parts.append(f"Similar plan (confidence: {plan.similarity_score:.0%}): {plan.content[:100]}...")
        
        # Generate main response
        if response_parts:
            main_response = "I can help you with strategic planning. " + " ".join(response_parts)
        else:
            main_response = "I'm ready to help with strategic planning. I can provide suggestions based on your historical data and account information."
        
        if form_prefill:
            main_response += f" I've prepared some intelligent suggestions for {len(form_prefill)} form fields based on your patterns."
        
        return {
            "response": main_response,
            "form_prefill": form_prefill
        }
