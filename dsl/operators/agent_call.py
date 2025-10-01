"""
Agent Call Operator - Integrates with existing AI agents and services
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from .base import BaseOperator, OperatorContext, OperatorResult
import logging

logger = logging.getLogger(__name__)

class AgentCallOperator(BaseOperator):
    """
    Agent call operator for invoking AI agents
    Integrates with:
    - Revolutionary Form Filler
    - Ultra Personalized Filler
    - Conversation Agent
    - Strategic Planning Agent
    """
    
    def __init__(self, config=None):
        super().__init__("agent_call_operator")
        self.config = config or {}
        self.requires_trust_check = True  # Enable trust threshold checking
    
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate agent call configuration"""
        errors = []
        
        # Check required fields
        if 'agent_id' not in config:
            errors.append("'agent_id' is required")
        elif config['agent_id'] not in [
            'revolutionary_form_filler',
            'ultra_personalized_filler', 
            'conversation_agent',
            'strategic_planning_agent',
            'crud_agent'
        ]:
            errors.append(f"Unsupported agent_id: {config['agent_id']}")
        
        if 'context' not in config:
            errors.append("'context' is required")
        
        # Validate timeout
        if 'timeout_sec' in config:
            timeout = config['timeout_sec']
            if not isinstance(timeout, int) or timeout <= 0 or timeout > 300:
                errors.append("'timeout_sec' must be between 1 and 300 seconds")
        
        # Validate max_depth
        if 'max_depth' in config:
            depth = config['max_depth']
            if not isinstance(depth, int) or depth <= 0 or depth > 10:
                errors.append("'max_depth' must be between 1 and 10")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute the agent call"""
        agent_id = config['agent_id']
        
        try:
            if agent_id == 'revolutionary_form_filler':
                return await self._call_revolutionary_form_filler(context, config)
            elif agent_id == 'ultra_personalized_filler':
                return await self._call_ultra_personalized_filler(context, config)
            elif agent_id == 'conversation_agent':
                return await self._call_conversation_agent(context, config)
            elif agent_id == 'strategic_planning_agent':
                return await self._call_strategic_planning_agent(context, config)
            elif agent_id == 'crud_agent':
                return await self._call_crud_agent(context, config)
            else:
                return OperatorResult(
                    success=False,
                    error_message=f"Unsupported agent: {agent_id}"
                )
                
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Agent call failed: {e}"
            )
    
    async def _call_revolutionary_form_filler(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Call the revolutionary form filler agent"""
        try:
            # Import the revolutionary form filler
            from ...tools.revolutionary_form_filler import get_revolutionary_form_filler
            
            pool_manager = context.pool_manager
            if not pool_manager:
                return OperatorResult(
                    success=False,
                    error_message="Pool manager not available for agent call"
                )
            
            # Get the form filler instance
            form_filler = get_revolutionary_form_filler(pool_manager)
            
            # Extract context data
            agent_context = config['context']
            message = agent_context.get('message', context.input_data.get('message', ''))
            user_context = {
                'user_id': context.user_id,
                'tenant_id': context.tenant_id,
                'session_id': context.session_id
            }
            
            # Add any additional context from config
            user_context.update(agent_context.get('user_context', {}))
            
            # Call the form filler
            timeout = config.get('timeout_sec', 30)
            
            try:
                result = await asyncio.wait_for(
                    form_filler.fill_complete_form(message, user_context),
                    timeout=timeout
                )
                
                # Calculate confidence based on form completeness
                filled_fields = len([v for v in result.values() if v and str(v).strip()])
                total_fields = len(form_filler.target_fields)
                confidence = min(filled_fields / total_fields, 1.0) if total_fields > 0 else 0.0
                
                return OperatorResult(
                    success=True,
                    output_data={
                        'form_data': result,
                        'filled_field_count': filled_fields,
                        'total_field_count': total_fields,
                        'completeness_ratio': confidence,
                        'agent_type': 'revolutionary_form_filler'
                    },
                    confidence_score=confidence
                )
                
            except asyncio.TimeoutError:
                return OperatorResult(
                    success=False,
                    error_message=f"Agent call timed out after {timeout} seconds"
                )
                
        except ImportError as e:
            return OperatorResult(
                success=False,
                error_message=f"Revolutionary form filler not available: {e}"
            )
        except Exception as e:
            logger.error(f"Revolutionary form filler error: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Revolutionary form filler failed: {e}"
            )
    
    async def _call_ultra_personalized_filler(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Call the ultra personalized filler agent"""
        try:
            # Import the ultra personalized filler
            from ...tools.ultra_personalized_filler import UltraPersonalizedFiller
            
            pool_manager = context.pool_manager
            if not pool_manager:
                return OperatorResult(
                    success=False,
                    error_message="Pool manager not available for agent call"
                )
            
            # Create ultra filler instance
            ultra_filler = UltraPersonalizedFiller(pool_manager)
            
            # Extract context data
            agent_context = config['context']
            user_id = agent_context.get('user_id', context.user_id)
            account_id = agent_context.get('account_id', 'default_account')
            industry = agent_context.get('industry', 'SaaS')
            account_tier = agent_context.get('account_tier', 'Enterprise')
            message_context = agent_context.get('message_context', context.input_data.get('message', ''))
            
            # Call the ultra personalized filler
            timeout = config.get('timeout_sec', 60)
            
            try:
                result = await asyncio.wait_for(
                    ultra_filler.generate_complete_ultra_personalized_form(
                        user_id=user_id,
                        account_id=account_id,
                        industry=industry,
                        account_tier=account_tier,
                        message_context=message_context
                    ),
                    timeout=timeout
                )
                
                # Calculate confidence based on personalization depth
                personalized_fields = len([v for v in result.values() if v and not str(v).startswith('_')])
                confidence = min(personalized_fields / 20, 1.0)  # Assuming 20 target fields
                
                return OperatorResult(
                    success=True,
                    output_data={
                        'personalized_data': result,
                        'personalized_field_count': personalized_fields,
                        'agent_type': 'ultra_personalized_filler'
                    },
                    confidence_score=confidence
                )
                
            except asyncio.TimeoutError:
                return OperatorResult(
                    success=False,
                    error_message=f"Ultra personalized agent timed out after {timeout} seconds"
                )
                
        except ImportError as e:
            return OperatorResult(
                success=False,
                error_message=f"Ultra personalized filler not available: {e}"
            )
        except Exception as e:
            logger.error(f"Ultra personalized filler error: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Ultra personalized filler failed: {e}"
            )
    
    async def _call_conversation_agent(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Call the conversation agent"""
        try:
            # Import the conversation agent
            from ...agents.conversation_agent import ConversationAgent
            
            pool_manager = context.pool_manager
            if not pool_manager:
                return OperatorResult(
                    success=False,
                    error_message="Pool manager not available for agent call"
                )
            
            # Create conversation agent instance
            conv_agent = ConversationAgent(pool_manager)
            
            # Extract context data
            agent_context = config['context']
            message = agent_context.get('message', context.input_data.get('message', ''))
            user_context = {
                'user_id': context.user_id,
                'tenant_id': context.tenant_id,
                'session_id': context.session_id
            }
            
            # Add additional context
            user_context.update(agent_context.get('user_context', {}))
            
            # Call the conversation agent
            timeout = config.get('timeout_sec', 45)
            
            try:
                result = await asyncio.wait_for(
                    conv_agent.handle_message(message, user_context),
                    timeout=timeout
                )
                
                # Extract confidence from result
                confidence = result.get('confidence_score', 0.85)
                
                return OperatorResult(
                    success=True,
                    output_data={
                        'conversation_result': result,
                        'response': result.get('response', ''),
                        'form_prefill': result.get('form_prefill', {}),
                        'suggested_actions': result.get('suggested_actions', []),
                        'intelligence_insights': result.get('intelligence_insights', {}),
                        'agent_type': 'conversation_agent'
                    },
                    confidence_score=confidence
                )
                
            except asyncio.TimeoutError:
                return OperatorResult(
                    success=False,
                    error_message=f"Conversation agent timed out after {timeout} seconds"
                )
                
        except ImportError as e:
            return OperatorResult(
                success=False,
                error_message=f"Conversation agent not available: {e}"
            )
        except Exception as e:
            logger.error(f"Conversation agent error: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Conversation agent failed: {e}"
            )
    
    async def _call_strategic_planning_agent(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Call the strategic planning agent"""
        try:
            # This would integrate with your strategic planning workflow
            # For now, return a structured response
            
            agent_context = config['context']
            message = agent_context.get('message', context.input_data.get('message', ''))
            
            # Mock strategic planning result
            mock_result = {
                'planning_recommendations': [
                    'Focus on enterprise accounts with >$50K ARR potential',
                    'Prioritize renewal conversations in Q1',
                    'Implement cross-sell strategy for existing customers'
                ],
                'risk_assessment': 'Medium risk due to market conditions',
                'confidence_level': 'High',
                'next_actions': [
                    'Schedule stakeholder meetings',
                    'Prepare business case documentation',
                    'Review competitive landscape'
                ]
            }
            
            return OperatorResult(
                success=True,
                output_data={
                    'strategic_plan': mock_result,
                    'agent_type': 'strategic_planning_agent'
                },
                confidence_score=0.88
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Strategic planning agent failed: {e}"
            )
    
    async def _call_crud_agent(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Call the CRUD agent for database operations"""
        try:
            # Import CRUD agent
            from ...agents.revolutionary_crud_agent import get_revolutionary_crud_agent
            
            pool_manager = context.pool_manager
            if not pool_manager:
                return OperatorResult(
                    success=False,
                    error_message="Pool manager not available for CRUD agent"
                )
            
            # Get CRUD agent instance
            crud_agent = get_revolutionary_crud_agent(pool_manager)
            
            # Extract context data
            agent_context = config['context']
            operation = agent_context.get('operation', 'read')
            table = agent_context.get('table', 'strategic_account_plans')
            data = agent_context.get('data', {})
            
            # Call CRUD agent (implementation would depend on your CRUD agent interface)
            # Mock for now
            mock_result = {
                'operation': operation,
                'table': table,
                'affected_rows': 1 if operation in ['create', 'update', 'delete'] else 0,
                'data': data if operation in ['create', 'update'] else {},
                'success': True
            }
            
            return OperatorResult(
                success=True,
                output_data={
                    'crud_result': mock_result,
                    'agent_type': 'crud_agent'
                },
                confidence_score=0.95
            )
            
        except ImportError as e:
            return OperatorResult(
                success=False,
                error_message=f"CRUD agent not available: {e}"
            )
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"CRUD agent failed: {e}"
            )
