"""
Agent Pool Service for Efficient Agent Management
Manages agent instances to prevent connection exhaustion
"""

import os
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentStats:
    requests_handled: int = 0
    last_used: datetime = None
    created_at: datetime = None
    active_connections: int = 0

class AgentPoolService:
    """
    Manages a pool of AI agents to prevent connection exhaustion
    - Singleton agent instances per agent type
    - Shared connection pools across all agents
    - Request queuing and throttling
    - Health monitoring and cleanup
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentPoolService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, pool_manager=None):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            
            # Single instances of each agent type
            self.form_monitor = None
            self.crud_agent = None
            self.conversation_agent = None
            
            # Agent statistics
            self.agent_stats = {
                'form_monitor': AgentStats(),
                'crud_agent': AgentStats(),
                'conversation_agent': AgentStats()
            }
            
            # Request queuing
            self.request_queues = {
                'form_monitor': asyncio.Queue(maxsize=100),
                'crud_agent': asyncio.Queue(maxsize=50),
                'conversation_agent': asyncio.Queue(maxsize=200)
            }
            
            # Connection pool manager
            self.pool_manager = pool_manager
            
            # Configuration from your environment variables
            self.config = {
                'max_concurrent_requests': int(os.getenv('MAX_CONCURRENT_REQUESTS', '10')),
                'agent_timeout': int(os.getenv('AGENT_RESPONSE_TIMEOUT', '30')),
                'cleanup_interval': int(os.getenv('CLEANUP_INTERVAL_MINUTES', '15')),
                'database_timeout': int(os.getenv('DATABASE_QUERY_TIMEOUT', '10')),
                'rpa_timeout': int(os.getenv('RPA_TIMEOUT', '60')),
                'enable_rpa': os.getenv('ENABLE_RPA_AUTOMATION', 'true').lower() == 'true',
                'enable_form_monitoring': os.getenv('ENABLE_FORM_MONITORING', 'true').lower() == 'true',
                'enable_crud': os.getenv('ENABLE_CRUD_OPERATIONS', 'true').lower() == 'true',
                'enable_conversation_memory': os.getenv('ENABLE_CONVERSATION_MEMORY', 'true').lower() == 'true'
            }
            
            self._initialized = True
    
    def set_pool_manager(self, pool_manager):
        """Set or update the connection pool manager"""
        self.pool_manager = pool_manager
    
    async def initialize(self):
        """Initialize the agent pool service"""
        try:
            # Initialize connection pool manager first
            from .connection_pool_manager import pool_manager
            self.pool_manager = pool_manager
            
            if not self.pool_manager._initialized:
                await self.pool_manager.initialize()
            
            # Initialize singleton agents
            await self._initialize_agents()
            
            # Start background tasks
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._health_monitor_task())
            
            self.logger.info("✅ Agent Pool Service initialized")
            self.logger.info(f"📊 Configuration: {self.config}")
            
        except Exception as e:
            self.logger.error(f"❌ Agent Pool Service initialization failed: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize singleton agent instances"""
        try:
            # Import agents
            from ..agents.form_monitor_agent import FormMonitorAgent
            from ..agents.crud_agent import CRUDAgent
            from ..agents.conversation_agent import ConversationAgent
            
            # Initialize form monitor (lightweight)
            if not self.form_monitor:
                self.form_monitor = FormMonitorAgent()
                self.agent_stats['form_monitor'].created_at = datetime.now()
                self.logger.info("✅ Form Monitor Agent initialized")
            
            # Initialize CRUD agent if enabled
            if not self.crud_agent and self.config['enable_crud']:
                backend_url = os.getenv('BACKEND_BASE_URL', 'http://localhost:3001')
                self.crud_agent = CRUDAgent(backend_url=backend_url)
                self.agent_stats['crud_agent'].created_at = datetime.now()
                self.logger.info("✅ CRUD Agent initialized")
            elif not self.config['enable_crud']:
                self.logger.info("⚠️ CRUD Agent disabled by configuration")
            
            # Initialize conversation agent
            if not self.conversation_agent:
                azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
                self.conversation_agent = ConversationAgent(azure_openai_api_key=azure_openai_key)
                self.agent_stats['conversation_agent'].created_at = datetime.now()
                self.logger.info("✅ Conversation Agent initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Agent initialization failed: {e}")
            raise
    
    async def get_form_monitor(self):
        """Get singleton form monitor agent"""
        if not self.form_monitor:
            await self._initialize_agents()
        
        # Update stats
        self.agent_stats['form_monitor'].last_used = datetime.now()
        self.agent_stats['form_monitor'].requests_handled += 1
        
        return self.form_monitor
    
    async def get_crud_agent(self):
        """Get singleton CRUD agent"""
        if not self.crud_agent:
            await self._initialize_agents()
        
        # Update stats
        self.agent_stats['crud_agent'].last_used = datetime.now()
        self.agent_stats['crud_agent'].requests_handled += 1
        
        return self.crud_agent
    
    async def get_conversation_agent(self):
        """Get singleton conversation agent"""
        if not self.conversation_agent:
            await self._initialize_agents()
        
        # Update stats
        self.agent_stats['conversation_agent'].last_used = datetime.now()
        self.agent_stats['conversation_agent'].requests_handled += 1
        
        return self.conversation_agent
    
    async def handle_agent_request(self, 
                                  agent_type: str,
                                  message: str,
                                  user_context: Dict[str, Any],
                                  timeout: Optional[int] = None) -> Dict[str, Any]:
        """Handle agent request with queuing and throttling"""
        request_timeout = timeout or self.config['agent_timeout']
        
        try:
            # Get appropriate agent
            if agent_type == 'form_monitor':
                agent = await self.get_form_monitor()
                should_handle = await agent.should_handle_message(message, user_context)
                if should_handle:
                    return await asyncio.wait_for(
                        agent.handle_message(message, user_context),
                        timeout=request_timeout
                    )
            
            elif agent_type == 'crud_agent':
                agent = await self.get_crud_agent()
                should_handle = await agent.should_handle_message(message, user_context)
                if should_handle:
                    return await asyncio.wait_for(
                        agent.handle_message(message, user_context),
                        timeout=request_timeout
                    )
            
            elif agent_type == 'conversation_agent':
                agent = await self.get_conversation_agent()
                # Conversation agent always handles messages
                return await asyncio.wait_for(
                    agent.handle_message(message, user_context),
                    timeout=request_timeout
                )
            
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            return {"error": "Agent declined to handle request"}
            
        except asyncio.TimeoutError:
            self.logger.error(f"❌ Agent {agent_type} request timed out after {request_timeout}s")
            return {
                "error": f"Request timed out after {request_timeout} seconds",
                "agent": agent_type,
                "timeout": True
            }
        
        except Exception as e:
            self.logger.error(f"❌ Agent {agent_type} request failed: {e}")
            return {
                "error": str(e),
                "agent": agent_type,
                "success": False
            }
    
    async def route_to_best_agent(self, 
                                 message: str,
                                 user_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route message to the most appropriate agent"""
        try:
            message_lower = message.lower()
            
            # Check CRUD operations first (highest priority) - be specific to avoid false matches
            crud_keywords = ['update plan', 'change plan', 'modify plan', 'delete plan', 'remove plan', 'edit plan']
            if any(keyword in message_lower for keyword in crud_keywords):
                result = await self.handle_agent_request('crud_agent', message, user_context)
                if result and not result.get('error'):
                    return result
            
            # Check form monitoring
            form_keywords = ['fill', 'prefill', 'form', 'field', 'suggest', 'help with']
            if any(keyword in message_lower for keyword in form_keywords):
                result = await self.handle_agent_request('form_monitor', message, user_context)
                if result and not result.get('error'):
                    return result
            
            # Default to conversation agent
            return await self.handle_agent_request('conversation_agent', message, user_context)
            
        except Exception as e:
            self.logger.error(f"❌ Agent routing failed: {e}")
            return {
                "error": "Agent routing failed",
                "details": str(e),
                "success": False
            }
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get current agent pool statistics"""
        pool_stats = await self.pool_manager.get_pool_stats() if self.pool_manager else {}
        
        agent_summary = {}
        for agent_type, stats in self.agent_stats.items():
            agent_summary[agent_type] = {
                'requests_handled': stats.requests_handled,
                'last_used': stats.last_used.isoformat() if stats.last_used else None,
                'created_at': stats.created_at.isoformat() if stats.created_at else None,
                'uptime_minutes': (
                    (datetime.now() - stats.created_at).total_seconds() / 60
                    if stats.created_at else 0
                )
            }
        
        return {
            'agent_stats': agent_summary,
            'connection_pool_stats': pool_stats,
            'configuration': self.config,
            'request_queue_sizes': {
                name: queue.qsize() for name, queue in self.request_queues.items()
            }
        }
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.config['cleanup_interval'] * 60)
                
                # Clean up old conversation sessions
                if self.conversation_agent:
                    cleaned = await self.conversation_agent.cleanup_old_sessions(hours=2)
                    if cleaned > 0:
                        self.logger.info(f"🗑️ Cleaned up {cleaned} old conversation sessions")
                
                # Log statistics
                stats = await self.get_agent_stats()
                self.logger.info(f"📊 Agent Pool Stats: {stats['agent_stats']}")
                
            except Exception as e:
                self.logger.error(f"❌ Cleanup task error: {e}")
    
    async def _health_monitor_task(self):
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check agent health
                for agent_type in ['form_monitor', 'crud_agent', 'conversation_agent']:
                    try:
                        # Simple health check with timeout
                        result = await asyncio.wait_for(
                            self.handle_agent_request(agent_type, "health check", {"user_id": "health"}),
                            timeout=10
                        )
                        
                        if result and result.get('error'):
                            self.logger.warning(f"⚠️ Agent {agent_type} health check failed: {result.get('error')}")
                    
                    except asyncio.TimeoutError:
                        self.logger.warning(f"⚠️ Agent {agent_type} health check timed out")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Agent {agent_type} health check error: {e}")
                
            except Exception as e:
                self.logger.error(f"❌ Health monitor task error: {e}")
    
    async def close(self):
        """Close all agents and cleanup"""
        try:
            # Close agents if they have cleanup methods
            if self.conversation_agent and hasattr(self.conversation_agent, 'cleanup_old_sessions'):
                await self.conversation_agent.cleanup_old_sessions(hours=0)
            
            # Clear references
            self.form_monitor = None
            self.crud_agent = None
            self.conversation_agent = None
            
            self.logger.info("✅ Agent Pool Service closed")
            
        except Exception as e:
            self.logger.error(f"❌ Error closing Agent Pool Service: {e}")

# Global instance
agent_pool = AgentPoolService()
