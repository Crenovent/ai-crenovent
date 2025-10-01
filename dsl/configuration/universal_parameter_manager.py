"""
Universal RBA Parameter Manager
==============================
Centralized parameter management system for all RBA agents.
Handles parameter loading, relevance filtering, validation, and UI schema generation.
"""

import yaml
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ParameterRelevance(Enum):
    """Parameter relevance levels for agents"""
    PRIMARY = "primary"      # Core parameters essential for agent function
    SECONDARY = "secondary"  # Important parameters that enhance functionality  
    TERTIARY = "tertiary"    # Optional parameters for fine-tuning
    UNUSED = "unused"        # Parameters not relevant to this agent

@dataclass
class ParameterDefinition:
    """Definition of a universal parameter"""
    name: str
    type: str  # currency, percentage, int, float, bool, str, enum
    default: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    ui_component: str = "input"  # input, slider, dropdown, currency_input, checkbox
    relevant_agents: List[str] = None
    ui_group: str = "General"
    
    def __post_init__(self):
        if self.relevant_agents is None:
            self.relevant_agents = []

class UniversalParameterManager:
    """
    Centralized manager for universal RBA parameters.
    
    Features:
    - Load 100+ universal parameters from YAML
    - Filter parameters by agent relevance
    - Generate UI schemas for frontend
    - Validate parameter values
    - Merge user configs with defaults
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterDefinition] = {}
        self.agent_relevance: Dict[str, Dict[str, List[str]]] = {}
        self.ui_groups: Dict[str, List[str]] = {}
        self.loaded = False
        
    def load_parameters(self, config_path: Optional[str] = None) -> None:
        """Load universal parameters from YAML configuration"""
        
        if self.loaded:
            return
            
        try:
            if config_path is None:
                config_path = Path(__file__).parent / "universal_rba_parameters.yaml"
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Load parameter definitions from all sections
            universal_params = config.get('universal_rba_parameters', {})
            
            for section_name, section_params in universal_params.items():
                if section_name in ['version', 'description']:
                    continue
                    
                if isinstance(section_params, dict):
                    for param_name, param_def in section_params.items():
                        if isinstance(param_def, dict) and 'type' in param_def:
                            self._load_parameter_definition(param_name, param_def, section_name)
            
            # Load agent relevance mapping
            self.agent_relevance = config.get('agent_parameter_relevance', {})
            
            # Load UI groupings
            self.ui_groups = config.get('ui_parameter_groups', {})
            
            self.loaded = True
            logger.info(f"✅ Loaded {len(self.parameters)} universal parameters for RBA agents")
            
        except Exception as e:
            logger.error(f"❌ Failed to load universal parameters: {e}")
            raise
    
    def _load_parameter_definition(self, name: str, definition: Dict[str, Any], section_name: str = None) -> None:
        """Load a single parameter definition"""
        
        param = ParameterDefinition(
            name=name,
            type=definition.get('type', 'str'),
            default=definition.get('default'),
            min_value=definition.get('min'),
            max_value=definition.get('max'),
            description=definition.get('description', ''),
            ui_component=definition.get('ui_component', 'input'),
            relevant_agents=definition.get('relevant_agents', [])
        )
        
        # Use section name as UI group with proper formatting
        if section_name:
            # Convert section names like "deal_value_parameters" to "Deal Value & Financial Metrics"
            ui_group_name = self._format_section_name_to_ui_group(section_name)
            param.ui_group = ui_group_name
        else:
            # Fallback to UI groups mapping or default
            for group_name, group_params in self.ui_groups.items():
                if name in group_params:
                    param.ui_group = group_name
                    break
        
        self.parameters[name] = param
    
    def _format_section_name_to_ui_group(self, section_name: str) -> str:
        """Convert YAML section names to user-friendly UI group names"""
        
        section_to_group_map = {
            'deal_value_parameters': 'Deal Value & Financial Metrics',
            'probability_parameters': 'Probability & Confidence',
            'time_parameters': 'Time & Velocity Settings',
            'data_quality_parameters': 'Data Quality & Compliance',
            'activity_parameters': 'Activity & Engagement',
            'risk_parameters': 'Risk Assessment',
            'performance_parameters': 'Performance & Benchmarks',
            'scoring_parameters': 'Scoring & Weighting',
            'threshold_parameters': 'Thresholds & Limits',
            'behavioral_parameters': 'Behavioral Analysis',
            'pipeline_parameters': 'Pipeline Management',
            'forecast_parameters': 'Forecast & Planning',
            'notification_parameters': 'Notifications & Alerts'
        }
        
        return section_to_group_map.get(section_name, section_name.replace('_', ' ').title())
    
    def get_parameters_for_agent(self, agent_name: str, relevance_levels: Optional[List[ParameterRelevance]] = None) -> Dict[str, ParameterDefinition]:
        """
        Get parameters relevant to a specific agent.
        
        Args:
            agent_name: Name of the RBA agent
            relevance_levels: Which relevance levels to include (default: PRIMARY + SECONDARY)
            
        Returns:
            Dict of parameter name -> ParameterDefinition for relevant parameters
        """
        
        if not self.loaded:
            self.load_parameters()
        
        if relevance_levels is None:
            relevance_levels = [ParameterRelevance.PRIMARY, ParameterRelevance.SECONDARY]
        
        relevant_params = {}
        agent_relevance = self.agent_relevance.get(agent_name, {})
        
        # Include parameters based on relevance level
        for level in relevance_levels:
            level_params = agent_relevance.get(level.value, [])
            for param_name in level_params:
                if param_name in self.parameters:
                    relevant_params[param_name] = self.parameters[param_name]
        
        # Also include parameters that explicitly list this agent as relevant
        for param_name, param_def in self.parameters.items():
            if agent_name in param_def.relevant_agents:
                relevant_params[param_name] = param_def
        
        return relevant_params
    
    def get_all_parameters(self) -> Dict[str, ParameterDefinition]:
        """Get all universal parameters"""
        
        if not self.loaded:
            self.load_parameters()
        
        return self.parameters.copy()
    
    def get_default_config_for_agent(self, agent_name: str, include_tertiary: bool = False) -> Dict[str, Any]:
        """
        Get default configuration for an agent with only relevant parameters.
        
        Args:
            agent_name: Name of the RBA agent
            include_tertiary: Whether to include tertiary parameters
            
        Returns:
            Dict of parameter name -> default value
        """
        
        relevance_levels = [ParameterRelevance.PRIMARY, ParameterRelevance.SECONDARY]
        if include_tertiary:
            relevance_levels.append(ParameterRelevance.TERTIARY)
        
        relevant_params = self.get_parameters_for_agent(agent_name, relevance_levels)
        
        return {
            param_name: param_def.default
            for param_name, param_def in relevant_params.items()
            if param_def.default is not None
        }
    
    def validate_config_for_agent(self, agent_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and type-convert configuration for an agent.
        
        Args:
            agent_name: Name of the RBA agent
            config: Configuration to validate
            
        Returns:
            Validated configuration with type conversion and range checking
        """
        
        relevant_params = self.get_parameters_for_agent(
            agent_name, 
            [ParameterRelevance.PRIMARY, ParameterRelevance.SECONDARY, ParameterRelevance.TERTIARY]
        )
        
        validated_config = {}
        
        for param_name, value in config.items():
            if param_name not in relevant_params:
                # Parameter not relevant to this agent - skip it
                continue
            
            param_def = relevant_params[param_name]
            validated_value = self._validate_parameter_value(param_def, value)
            
            if validated_value is not None:
                validated_config[param_name] = validated_value
        
        return validated_config
    
    def _validate_parameter_value(self, param_def: ParameterDefinition, value: Any) -> Any:
        """Validate and convert a single parameter value"""
        
        if value is None:
            return param_def.default
        
        try:
            # Type conversion
            if param_def.type in ['int']:
                converted_value = int(value)
            elif param_def.type in ['float', 'currency', 'percentage']:
                converted_value = float(value)
            elif param_def.type == 'bool':
                if isinstance(value, bool):
                    converted_value = value
                else:
                    converted_value = str(value).lower() in ['true', '1', 'yes', 'on']
            else:
                converted_value = str(value)
            
            # Range validation
            if param_def.min_value is not None and converted_value < param_def.min_value:
                logger.warning(f"⚠️ {param_def.name} value {converted_value} below minimum {param_def.min_value}, using minimum")
                converted_value = param_def.min_value
            
            # Maximum value validation disabled per user request - no limits on parameters
            # if param_def.max_value is not None and converted_value > param_def.max_value:
            #     logger.warning(f"⚠️ {param_def.name} value {converted_value} above maximum {param_def.max_value}, using maximum")
            #     converted_value = param_def.max_value
            
            return converted_value
            
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ Invalid value for {param_def.name}: {value}, using default {param_def.default}")
            return param_def.default
    
    def generate_ui_schema_for_agent(self, agent_name: str, include_all_params: bool = False) -> Dict[str, Any]:
        """
        Generate UI schema for frontend parameter configuration.
        
        Args:
            agent_name: Name of the RBA agent
            include_all_params: If True, include all parameters; if False, only relevant ones
            
        Returns:
            UI schema suitable for frontend rendering
        """
        
        if include_all_params:
            # Include all parameters for universal UI
            relevant_params = self.get_all_parameters()
        else:
            # Include only relevant parameters for agent-specific UI
            relevant_params = self.get_parameters_for_agent(
                agent_name,
                [ParameterRelevance.PRIMARY, ParameterRelevance.SECONDARY, ParameterRelevance.TERTIARY]
            )
        
        # Group parameters by UI groups
        grouped_params = {}
        for param_name, param_def in relevant_params.items():
            group = param_def.ui_group
            if group not in grouped_params:
                grouped_params[group] = []
            
            grouped_params[group].append({
                'name': param_name,
                'type': param_def.type,
                'default': param_def.default,
                'min': param_def.min_value,
                'max': param_def.max_value,
                'description': param_def.description,
                'ui_component': param_def.ui_component,
                'relevance': self._get_parameter_relevance_for_agent(agent_name, param_name)
            })
        
        return {
            'agent_name': agent_name,
            'parameter_groups': grouped_params,
            'total_parameters': len(relevant_params),
            'schema_version': '1.0.0'
        }
    
    def _get_parameter_relevance_for_agent(self, agent_name: str, param_name: str) -> str:
        """Determine parameter relevance level for an agent"""
        
        agent_relevance = self.agent_relevance.get(agent_name, {})
        
        for level in [ParameterRelevance.PRIMARY, ParameterRelevance.SECONDARY, ParameterRelevance.TERTIARY]:
            if param_name in agent_relevance.get(level.value, []):
                return level.value
        
        # Check if explicitly listed as relevant
        param_def = self.parameters.get(param_name)
        if param_def and agent_name in param_def.relevant_agents:
            return ParameterRelevance.SECONDARY.value
        
        return ParameterRelevance.UNUSED.value
    
    def merge_configs(self, agent_name: str, user_config: Dict[str, Any], tenant_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge configuration from multiple sources with validation.
        
        Priority order: user_config > tenant_config > defaults
        
        Args:
            agent_name: Name of the RBA agent
            user_config: User-provided configuration
            tenant_config: Tenant-level configuration (optional)
            
        Returns:
            Merged and validated configuration
        """
        
        # Start with defaults
        final_config = self.get_default_config_for_agent(agent_name, include_tertiary=True)
        
        # Apply tenant config
        if tenant_config:
            validated_tenant = self.validate_config_for_agent(agent_name, tenant_config)
            final_config.update(validated_tenant)
        
        # Apply user config (highest priority)
        if user_config:
            validated_user = self.validate_config_for_agent(agent_name, user_config)
            final_config.update(validated_user)
        
        return final_config
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get statistics about the parameter system"""
        
        if not self.loaded:
            self.load_parameters()
        
        # Count parameters by type
        type_counts = {}
        for param in self.parameters.values():
            type_counts[param.type] = type_counts.get(param.type, 0) + 1
        
        # Count parameters by UI component
        ui_component_counts = {}
        for param in self.parameters.values():
            ui_component_counts[param.ui_component] = ui_component_counts.get(param.ui_component, 0) + 1
        
        # Count parameters by relevance
        agent_param_counts = {}
        for agent_name, relevance_map in self.agent_relevance.items():
            total_relevant = sum(len(params) for params in relevance_map.values())
            agent_param_counts[agent_name] = total_relevant
        
        return {
            'total_parameters': len(self.parameters),
            'parameter_types': type_counts,
            'ui_components': ui_component_counts,
            'parameters_per_agent': agent_param_counts,
            'ui_groups': {group: len(params) for group, params in self.ui_groups.items()},
            'agents_supported': len(self.agent_relevance)
        }

# Global instance
universal_parameter_manager = UniversalParameterManager()

def get_universal_parameter_manager() -> UniversalParameterManager:
    """Get the global universal parameter manager instance"""
    return universal_parameter_manager
