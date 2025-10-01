"""
Base RBA Agent
Base class for all Rule-Based Automation agents

Provides common functionality and implements abstract methods from BaseOperator
"""

import logging
from typing import Dict, List, Any
from abc import ABC, abstractmethod

from ..base import BaseOperator, OperatorContext, OperatorResult

logger = logging.getLogger(__name__)

class BaseRBAAgent(BaseOperator):
    """
    Base class for all RBA agents
    
    Implements abstract methods from BaseOperator and provides
    common RBA-specific functionality
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.default_config = {}
        # Configuration management attributes
        self.config_schema = self._get_config_schema()
        self.result_schema = self._get_result_schema()
    
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Default config validation for RBA agents
        Override in specific agents for custom validation
        """
        errors = []
        
        # Basic validation - check if config is a dict
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return errors
        
        # Call agent-specific validation if implemented
        try:
            agent_errors = await self._validate_agent_config(config)
            if agent_errors:
                errors.extend(agent_errors)
        except NotImplementedError:
            # Agent doesn't have custom validation - that's fine
            pass
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """
        Execute RBA agent with async context
        Converts context to input_data format and calls execute
        """
        try:
            # Convert OperatorContext to input_data format
            input_data = {
                'opportunities': context.get('opportunities', []),
                'config': config
            }
            
            # Execute the agent
            result = await self.execute(input_data)
            
            # Convert result to OperatorResult format
            return OperatorResult(
                success=result.get('success', True),
                data=result,
                metadata={
                    'agent_name': result.get('agent_name'),
                    'analysis_type': result.get('analysis_type'),
                    'timestamp': result.get('analysis_timestamp')
                }
            )
            
        except Exception as e:
            logger.error(f"❌ RBA Agent execution failed: {e}")
            return OperatorResult(
                success=False,
                data={'error': str(e)},
                metadata={'agent_name': getattr(self, 'AGENT_NAME', 'unknown')}
            )
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the RBA agent analysis
        Must be implemented by each specific RBA agent
        """
        pass
    
    async def _validate_agent_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Agent-specific config validation
        Override in specific agents for custom validation
        """
        raise NotImplementedError("Agent-specific validation not implemented")
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """
        Safely convert value to float, handling None and invalid values
        """
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """
        Safely convert value to int, handling None and invalid values
        """
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_str(self, value: Any, default: str = '') -> str:
        """
        Safely convert value to string, handling None values
        """
        if value is None:
            return default
        return str(value)
    
    @classmethod
    def get_agent_metadata(cls) -> Dict[str, Any]:
        """Get agent metadata for registry - must be implemented by each agent"""
        return {
            'agent_type': getattr(cls, 'AGENT_TYPE', 'RBA'),
            'agent_name': getattr(cls, 'AGENT_NAME', 'unknown'),
            'agent_description': getattr(cls, 'AGENT_DESCRIPTION', 'RBA Agent'),
            'supported_analysis_types': getattr(cls, 'SUPPORTED_ANALYSIS_TYPES', []),
            'class_name': cls.__name__,
            'module_path': cls.__module__
        }
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for this agent.
        Override in specific agents to define parameters.
        """
        return {}
    
    def _get_result_schema(self) -> Dict[str, Any]:
        """
        Get result schema for this agent.
        Override in specific agents to define visualization types.
        """
        return {
            'primary_visualizations': ['table'],
            'drill_down_fields': ['opportunity_id'],
            'export_formats': ['csv']
        }
    
    def get_config_schema_for_ui(self) -> Dict[str, Any]:
        """
        Get configuration schema formatted for UI generation.
        """
        agent_name = getattr(self, 'AGENT_NAME', 'unknown')
        agent_description = getattr(self, 'AGENT_DESCRIPTION', 'No description available')
        
        return {
            'agent_name': agent_name,
            'description': agent_description,
            'config_schema': self.config_schema,
            'result_schema': self.result_schema,
            'default_config': self.default_config
        }
    
    def merge_config_with_defaults(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user configuration with agent defaults.
        """
        merged_config = self.default_config.copy()
        merged_config.update(user_config)
        return merged_config
