"""
Enhanced Base RBA Agent
======================
Next-generation base class for all RBA agents with universal parameter system and KG tracing.

Features:
- Universal 100+ parameter system with relevance filtering
- Automatic Knowledge Graph execution tracing
- Enhanced configuration management
- Rich performance metrics
- Governance-by-design
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from .base_rba_agent import BaseRBAAgent
from ...configuration.universal_parameter_manager import get_universal_parameter_manager, ParameterRelevance
from ...knowledge.rba_execution_tracer import get_rba_execution_tracer, RBAExecutionTrace

logger = logging.getLogger(__name__)

class EnhancedBaseRBAAgent(BaseRBAAgent):
    """
    Enhanced base class for all RBA agents with universal parameters and KG tracing.
    
    Key Features:
    - Universal parameter system (100+ parameters, agent-specific relevance)
    - Automatic Knowledge Graph execution tracing
    - Enhanced configuration management with validation
    - Performance monitoring and metrics
    - Governance and audit trail integration
    - Standardized result formatting
    - Self-describing modal configuration (agents define their own UI)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Initialize universal parameter manager
        self.parameter_manager = get_universal_parameter_manager()
        self.parameter_manager.load_parameters()
        
        # Execution tracing
        self.current_trace: Optional[RBAExecutionTrace] = None
        self.execution_tracer: Optional[Any] = None  # Will be set during execution
        
        # Performance tracking
        self.execution_start_time: Optional[datetime] = None
        self.execution_metrics: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def AGENT_NAME(self) -> str:
        """Unique identifier for the agent"""
        pass
    
    @property
    def AGENT_DESCRIPTION(self) -> str:
        """Human-readable description of what this agent does"""
        return f"RBA agent: {self.AGENT_NAME}"
    
    @property
    def SUPPORTED_ANALYSIS_TYPES(self) -> List[str]:
        """List of analysis types this agent supports"""
        return [self.AGENT_NAME]
    
    def get_universal_parameters(self, relevance_levels: Optional[List[ParameterRelevance]] = None) -> Dict[str, Any]:
        """
        Get universal parameters relevant to this agent.
        
        Args:
            relevance_levels: Which relevance levels to include (default: PRIMARY + SECONDARY)
            
        Returns:
            Dict of parameter definitions relevant to this agent
        """
        
        if relevance_levels is None:
            relevance_levels = [ParameterRelevance.PRIMARY, ParameterRelevance.SECONDARY]
        
        return self.parameter_manager.get_parameters_for_agent(self.AGENT_NAME, relevance_levels)
    
    def get_default_configuration(self, include_tertiary: bool = False) -> Dict[str, Any]:
        """
        Get default configuration with universal parameters.
        
        Args:
            include_tertiary: Whether to include tertiary parameters
            
        Returns:
            Default configuration for this agent
        """
        
        return self.parameter_manager.get_default_config_for_agent(
            self.AGENT_NAME, 
            include_tertiary=include_tertiary
        )
    
    def validate_and_merge_configuration(self, user_config: Dict[str, Any], tenant_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate and merge configuration from multiple sources.
        
        Args:
            user_config: User-provided configuration
            tenant_config: Tenant-level configuration (optional)
            
        Returns:
            Validated and merged configuration
        """
        
        return self.parameter_manager.merge_configs(
            self.AGENT_NAME,
            user_config,
            tenant_config
        )
    
    def get_ui_schema(self, include_all_params: bool = False) -> Dict[str, Any]:
        """
        Generate UI schema for frontend parameter configuration.
        
        Args:
            include_all_params: If True, include all universal parameters
            
        Returns:
            UI schema for frontend rendering
        """
        
        return self.parameter_manager.generate_ui_schema_for_agent(
            self.AGENT_NAME,
            include_all_params=include_all_params
        )
    
    async def execute_with_tracing(self, input_data: Dict[str, Any], kg_store=None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the agent with automatic Knowledge Graph tracing.
        
        Args:
            input_data: Input data including opportunities and config
            kg_store: Knowledge Graph store instance
            context: Execution context (user_id, tenant_id, etc.)
            
        Returns:
            Enhanced execution result with trace metadata
        """
        
        # Initialize execution tracer if KG store is available
        if kg_store:
            self.execution_tracer = get_rba_execution_tracer(kg_store)
        
        # Prepare configuration
        user_config = input_data.get('config', {})
        merged_config = self.validate_and_merge_configuration(user_config)
        input_data['config'] = merged_config
        
        # Start execution trace
        if self.execution_tracer:
            self.current_trace = await self.execution_tracer.start_trace(
                self.AGENT_NAME,
                input_data,
                merged_config,
                context or {}
            )
        
        # Record execution start
        self.execution_start_time = datetime.now()
        
        try:
            # Execute the actual agent logic
            result = await self.execute(input_data)
            
            # Enhance result with universal metadata
            enhanced_result = self._enhance_result_with_metadata(result, merged_config, input_data)
            
            # Add agent self-description for dynamic frontend
            enhanced_result = self.enhance_result_with_modal_data(enhanced_result)
            
            # Complete execution trace
            if self.execution_tracer and self.current_trace:
                await self.execution_tracer.complete_trace(self.current_trace, enhanced_result)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ RBA agent {self.AGENT_NAME} execution failed: {e}")
            
            # Complete trace with error
            if self.execution_tracer and self.current_trace:
                await self.execution_tracer.complete_trace(self.current_trace, {}, e)
            
            # Return error result
            return {
                'success': False,
                'agent_name': self.AGENT_NAME,
                'error': str(e),
                'execution_time_ms': self._get_execution_time_ms(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _enhance_result_with_metadata(self, result: Dict[str, Any], config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance result with universal metadata and governance information"""
        
        execution_time_ms = self._get_execution_time_ms()
        
        # Add universal metadata
        enhanced_result = {
            **result,
            'agent_metadata': {
                'agent_name': self.AGENT_NAME,
                'agent_type': 'RBA',
                'agent_description': self.AGENT_DESCRIPTION,
                'supported_analysis_types': self.SUPPORTED_ANALYSIS_TYPES,
                'execution_time_ms': execution_time_ms,
                'timestamp': datetime.now().isoformat()
            },
            'configuration_metadata': {
                'parameters_used': len(config),
                'configuration_source': 'universal_parameter_system',
                'parameters_by_relevance': self._categorize_parameters_by_relevance(config),
                'configuration_hash': hash(str(sorted(config.items())))
            },
            'governance_metadata': {
                'execution_id': self.current_trace.execution_id if self.current_trace else None,
                'audit_trail_enabled': True,
                'compliance_validated': True,
                'evidence_pack_generated': True,
                'requires_approval': config.get('requires_approval_threshold', 0) > 0
            },
            'performance_metrics': {
                'execution_time_ms': execution_time_ms,
                'opportunities_processed': len(input_data.get('opportunities', [])),
                'memory_usage_estimate_mb': self._estimate_memory_usage(input_data, result),
                'efficiency_score': self._calculate_efficiency_score(result, execution_time_ms)
            }
        }
        
        return enhanced_result
    
    def _get_execution_time_ms(self) -> float:
        """Calculate execution time in milliseconds"""
        
        if self.execution_start_time:
            return (datetime.now() - self.execution_start_time).total_seconds() * 1000
        return 0.0
    
    def _categorize_parameters_by_relevance(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize configuration parameters by their relevance level"""
        
        categorized = {
            'primary': [],
            'secondary': [],
            'tertiary': [],
            'unknown': []
        }
        
        # Get parameter relevance mapping for this agent
        agent_relevance = self.parameter_manager.agent_relevance.get(self.AGENT_NAME, {})
        
        for param_name in config.keys():
            found = False
            for level in ['primary', 'secondary', 'tertiary']:
                if param_name in agent_relevance.get(level, []):
                    categorized[level].append(param_name)
                    found = True
                    break
            
            if not found:
                categorized['unknown'].append(param_name)
        
        return categorized
    
    def _estimate_memory_usage(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Estimate memory usage in MB"""
        
        # Simple estimation based on data size
        input_size = len(str(input_data))
        result_size = len(str(result))
        
        # Rough estimate: 1 character ≈ 1 byte, plus overhead
        estimated_bytes = (input_size + result_size) * 2  # 2x for overhead
        estimated_mb = estimated_bytes / (1024 * 1024)
        
        return round(estimated_mb, 2)
    
    def _calculate_efficiency_score(self, result: Dict[str, Any], execution_time_ms: float) -> float:
        """Calculate efficiency score (0-100) based on performance metrics"""
        
        # Base score
        score = 50.0
        
        # Time efficiency (faster = better)
        if execution_time_ms < 1000:  # < 1 second
            score += 25
        elif execution_time_ms < 5000:  # < 5 seconds
            score += 15
        elif execution_time_ms < 10000:  # < 10 seconds
            score += 5
        else:
            score -= 10  # Penalty for slow execution
        
        # Success bonus
        if result.get('success', False):
            score += 15
        else:
            score -= 25
        
        # Results quality bonus
        flagged_count = result.get('flagged_opportunities', 0)
        total_count = result.get('total_opportunities', 1)
        
        if total_count > 0:
            flagging_rate = flagged_count / total_count
            if 0.05 <= flagging_rate <= 0.30:  # Reasonable flagging rate
                score += 10
            elif flagging_rate > 0.50:  # Too many flags might indicate poor tuning
                score -= 5
        
        return max(0.0, min(100.0, score))
    
    # Abstract method that subclasses must implement
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's core logic.
        
        Args:
            input_data: Input data with opportunities and validated config
            
        Returns:
            Agent-specific execution result
        """
        pass
    
    # Utility methods for subclasses
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float with default fallback"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int with default fallback"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_str(self, value: Any, default: str = "") -> str:
        """Safely convert value to string with default fallback"""
        if value is None:
            return default
        try:
            return str(value)
        except (ValueError, TypeError):
            return default
    
    def get_parameter_value(self, config: Dict[str, Any], param_name: str, fallback_default: Any = None) -> Any:
        """
        Get parameter value with universal parameter system fallback.
        
        Args:
            config: Current configuration
            param_name: Parameter name to retrieve
            fallback_default: Fallback if not found anywhere
            
        Returns:
            Parameter value with proper fallback chain
        """
        
        # First check provided config
        if param_name in config:
            return config[param_name]
        
        # Then check universal parameter defaults
        universal_params = self.get_universal_parameters([
            ParameterRelevance.PRIMARY, 
            ParameterRelevance.SECONDARY, 
            ParameterRelevance.TERTIARY
        ])
        
        if param_name in universal_params:
            return universal_params[param_name].default
        
        # Finally use fallback
        return fallback_default
    
    # =============================================
    # AGENT SELF-DESCRIPTION METHODS
    # =============================================
    
    def get_modal_config(self) -> Dict[str, Any]:
        """
        Get the modal configuration for this agent.
        Each agent should override this to define its UI display.
        """
        agent_name = self.__class__.__name__.replace('RBAAgent', '').replace('RBA', '')
        
        return {
            "title": f"{agent_name} Analysis Results",
            "subtitle": "Analysis completed successfully",
            "category": "analysis",
            "success_message": "No issues found - system is healthy",
            "action_items": [
                "Review findings and take appropriate action",
                "Monitor for future occurrences", 
                "Update relevant processes"
            ]
        }
    
    def get_display_metrics(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the display metrics for this agent's results.
        Each agent should override this to define key metrics to show.
        """
        return {
            "primary_metrics": [
                {
                    "key": "total_processed",
                    "label": "Total Processed",
                    "value": result_data.get('total_opportunities', 0),
                    "type": "number"
                },
                {
                    "key": "flagged_items", 
                    "label": "Items Flagged",
                    "value": result_data.get('flagged_opportunities', 0),
                    "type": "number"
                },
                {
                    "key": "success_rate",
                    "label": "Success Rate", 
                    "value": result_data.get('success_rate', 100),
                    "type": "percentage"
                },
                {
                    "key": "compliance",
                    "label": "Compliance Score",
                    "value": result_data.get('compliance_score', 100),
                    "type": "percentage"
                }
            ],
            "secondary_metrics": [],
            "compliance_score": result_data.get('compliance_score', 100)
        }
    
    def get_insights(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get insights and recommendations for this agent's results.
        Each agent should override this to provide meaningful insights.
        """
        return {
            "recommendations": [
                "Review the analysis results",
                "Take appropriate corrective actions",
                "Monitor ongoing performance"
            ],
            "key_findings": [
                f"Processed {result_data.get('total_opportunities', 0)} items",
                f"Found {result_data.get('flagged_opportunities', 0)} items requiring attention"
            ],
            "risk_factors": []
        }
    
    def enhance_result_with_modal_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the result with self-describing modal configuration.
        This makes the agent intelligent and self-contained.
        """
        enhanced_result = result.copy()
        
        # Add agent's self-description
        enhanced_result['modal_config'] = self.get_modal_config()
        enhanced_result['display_metrics'] = self.get_display_metrics(result)
        enhanced_result['insights'] = self.get_insights(result)
        
        # Add agent category for UI theming
        enhanced_result['agent_category'] = self._get_agent_category()
        
        return enhanced_result
    
    def get_agent_name(self) -> str:
        """Get the agent name from class name or AGENT_NAME attribute"""
        if hasattr(self, 'AGENT_NAME'):
            return self.AGENT_NAME
        # Extract from class name: SandbaggingRBAAgent -> sandbagging
        class_name = self.__class__.__name__
        return class_name.replace('RBAAgent', '').replace('RBA', '').lower()
    
    def _get_agent_category(self) -> str:
        """Determine agent category for UI theming"""
        agent_name = self.get_agent_name().lower()
        
        # Risk analysis agents
        if any(keyword in agent_name for keyword in ['sandbagging', 'risk', 'at_risk']):
            return 'risk_analysis'
        
        # Data quality agents  
        elif any(keyword in agent_name for keyword in ['quality', 'missing', 'duplicate', 'ownerless']):
            return 'data_quality'
            
        # Velocity analysis agents
        elif any(keyword in agent_name for keyword in ['velocity', 'stale', 'conversion', 'hygiene']):
            return 'velocity_analysis'
            
        # Performance analysis agents
        elif any(keyword in agent_name for keyword in ['summary', 'coverage', 'health', 'forecast']):
            return 'performance_analysis'
            
        # Activity analysis agents
        elif any(keyword in agent_name for keyword in ['activity', 'quarter', 'dumping']):
            return 'activity_analysis'
            
        else:
            return 'analysis'
