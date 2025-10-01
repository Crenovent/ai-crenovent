"""
Enhanced Configuration Manager for RBA Agents
==============================================
Manages configurations for all RBA agents with multi-level overrides,
validation, UI schema generation, and real-time preview capabilities.
"""

import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .rba_agent_discovery import get_discovery_instance

logger = logging.getLogger(__name__)

class ConfigParameterType(Enum):
    """Types of configuration parameters"""
    STRING = "string"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    ENUM = "enum"
    OBJECT = "object"
    ARRAY = "array"

class UIComponent(Enum):
    """UI component types for parameter rendering"""
    INPUT = "input"
    SLIDER = "slider"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    CURRENCY_INPUT = "currency_input"
    PERCENTAGE_SLIDER = "percentage_slider"
    STAGE_MATRIX = "stage_matrix"
    STAGE_PROBABILITY_MATRIX = "stage_probability_matrix"
    MULTI_SELECT = "multi_select"

@dataclass
class ConfigParameter:
    """Definition of a configuration parameter"""
    name: str
    type: ConfigParameterType
    default: Any
    label: str
    description: str
    ui_component: UIComponent
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    options: Optional[List[Any]] = None
    required: bool = True
    category: Optional[str] = None

@dataclass
class ConfigTemplate:
    """Configuration template for quick setup"""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str

class EnhancedConfigManager:
    """
    Enhanced configuration manager for RBA agents.
    
    Features:
    - Multi-level configuration merging (tenant → user → prompt)
    - Dynamic parameter validation and type conversion
    - UI schema generation for frontend components
    - Configuration templates (Conservative, Aggressive, etc.)
    - Real-time configuration preview
    - YAML-based configuration storage
    """
    
    def __init__(self, config_schemas_dir: str = "dsl/rules/config_schemas"):
        self.config_schemas_dir = Path(config_schemas_dir)
        self.discovery = get_discovery_instance()
        self.agent_configs: Dict[str, Dict[str, ConfigParameter]] = {}
        self.config_templates: Dict[str, List[ConfigTemplate]] = {}
        self._load_config_schemas()
    
    def _load_config_schemas(self) -> None:
        """Load configuration schemas from YAML files"""
        logger.info(f"📋 Loading config schemas from {self.config_schemas_dir}")
        
        if not self.config_schemas_dir.exists():
            logger.warning(f"⚠️ Config schemas directory not found: {self.config_schemas_dir}")
            self._generate_default_schemas()
            return
        
        # Load YAML config files
        for yaml_file in self.config_schemas_dir.glob("*.yaml"):
            try:
                self._load_yaml_config_file(yaml_file)
            except Exception as e:
                logger.error(f"❌ Failed to load config file {yaml_file}: {e}")
    
    def _load_yaml_config_file(self, yaml_file: Path) -> None:
        """Load configuration from a YAML file"""
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Process each agent configuration
        for agent_name, agent_config in config_data.items():
            if agent_name.startswith('_'):  # Skip metadata entries
                continue
            
            parameters = {}
            param_data = agent_config.get('parameters', {})
            
            for param_name, param_def in param_data.items():
                try:
                    parameter = ConfigParameter(
                        name=param_name,
                        type=ConfigParameterType(param_def.get('type', 'string')),
                        default=param_def.get('default'),
                        label=param_def.get('label', param_name.replace('_', ' ').title()),
                        description=param_def.get('description', ''),
                        ui_component=UIComponent(param_def.get('ui_component', 'input')),
                        min_value=param_def.get('min'),
                        max_value=param_def.get('max'),
                        options=param_def.get('options'),
                        required=param_def.get('required', True),
                        category=param_def.get('category')
                    )
                    parameters[param_name] = parameter
                except Exception as e:
                    logger.error(f"❌ Failed to parse parameter {param_name} for {agent_name}: {e}")
            
            self.agent_configs[agent_name] = parameters
            logger.info(f"📋 Loaded {len(parameters)} parameters for {agent_name}")
    
    def _generate_default_schemas(self) -> None:
        """Generate default configuration schemas for discovered agents"""
        logger.info("🔧 Generating default config schemas for discovered agents")
        
        agents = self.discovery.discover_agents()
        
        for agent_name, metadata in agents.items():
            # Create basic schema based on agent metadata
            parameters = self._infer_parameters_from_agent(agent_name, metadata)
            self.agent_configs[agent_name] = parameters
    
    def _infer_parameters_from_agent(self, agent_name: str, metadata) -> Dict[str, ConfigParameter]:
        """Infer configuration parameters from agent metadata"""
        parameters = {}
        
        # Common parameters for all agents
        common_params = {
            'confidence_threshold': ConfigParameter(
                name='confidence_threshold',
                type=ConfigParameterType.PERCENTAGE,
                default=70,
                label='Confidence Threshold',
                description='Minimum confidence level for flagging',
                ui_component=UIComponent.SLIDER,
                min_value=50,
                max_value=100
            )
        }
        
        # Category-specific parameters
        if metadata.category == 'risk_analysis':
            risk_params = {
                'high_value_threshold': ConfigParameter(
                    name='high_value_threshold',
                    type=ConfigParameterType.CURRENCY,
                    default=250000,
                    label='High Value Threshold',
                    description='Minimum deal amount to be considered high-value',
                    ui_component=UIComponent.CURRENCY_INPUT,
                    min_value=1000,
                    max_value=50000000
                ),
                'risk_threshold': ConfigParameter(
                    name='risk_threshold',
                    type=ConfigParameterType.PERCENTAGE,
                    default=65,
                    label='Risk Score Threshold',
                    description='Minimum score to flag as risky',
                    ui_component=UIComponent.SLIDER,
                    min_value=0,
                    max_value=100
                )
            }
            parameters.update(risk_params)
        
        elif metadata.category == 'data_quality':
            quality_params = {
                'minimum_completion_rate': ConfigParameter(
                    name='minimum_completion_rate',
                    type=ConfigParameterType.PERCENTAGE,
                    default=80,
                    label='Minimum Completion Rate',
                    description='Minimum data completion rate required',
                    ui_component=UIComponent.SLIDER,
                    min_value=0,
                    max_value=100
                )
            }
            parameters.update(quality_params)
        
        elif metadata.category == 'velocity_analysis':
            velocity_params = {
                'max_days_in_stage': ConfigParameter(
                    name='max_days_in_stage',
                    type=ConfigParameterType.INTEGER,
                    default=30,
                    label='Max Days in Stage',
                    description='Maximum days allowed in any stage',
                    ui_component=UIComponent.INPUT,
                    min_value=1,
                    max_value=365
                )
            }
            parameters.update(velocity_params)
        
        parameters.update(common_params)
        return parameters
    
    def get_config_schema_for_agent(self, agent_name: str) -> Dict[str, ConfigParameter]:
        """Get configuration schema for a specific agent"""
        return self.agent_configs.get(agent_name, {})
    
    def get_ui_schema_for_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Generate UI-friendly configuration schema for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            UI schema dictionary for frontend consumption
        """
        agent_metadata = self.discovery.discovered_agents.get(agent_name)
        if not agent_metadata:
            return {}
        
        parameters = self.get_config_schema_for_agent(agent_name)
        
        ui_schema = {
            'agent_name': agent_name,
            'category': agent_metadata.category,
            'description': agent_metadata.description,
            'parameter_groups': self._group_parameters_for_ui(parameters),
            'templates': self._get_templates_for_agent(agent_name)
        }
        
        return ui_schema
    
    def _group_parameters_for_ui(self, parameters: Dict[str, ConfigParameter]) -> Dict[str, List[Dict[str, Any]]]:
        """Group parameters by category for better UI organization"""
        groups = {
            'Core Settings': [],
            'Thresholds': [],
            'Advanced': [],
            'Other': []
        }
        
        for param in parameters.values():
            param_dict = {
                'name': param.name,
                'type': param.type.value,
                'label': param.label,
                'description': param.description,
                'default': param.default,
                'ui_component': param.ui_component.value,
                'required': param.required
            }
            
            # Add validation constraints
            if param.min_value is not None:
                param_dict['min'] = param.min_value
            if param.max_value is not None:
                param_dict['max'] = param.max_value
            if param.options is not None:
                param_dict['options'] = param.options
            
            # Assign to group
            if 'threshold' in param.name.lower():
                groups['Thresholds'].append(param_dict)
            elif param.category == 'advanced' or 'weight' in param.name.lower():
                groups['Advanced'].append(param_dict)
            elif param.name in ['high_value_threshold', 'confidence_threshold']:
                groups['Core Settings'].append(param_dict)
            else:
                groups['Other'].append(param_dict)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _get_templates_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get configuration templates for an agent"""
        agent_metadata = self.discovery.discovered_agents.get(agent_name)
        if not agent_metadata:
            return []
        
        templates = []
        
        # Conservative template
        conservative_config = self._generate_conservative_config(agent_name)
        templates.append({
            'name': 'Conservative',
            'description': 'Lower thresholds, higher confidence requirements',
            'config': conservative_config
        })
        
        # Aggressive template
        aggressive_config = self._generate_aggressive_config(agent_name)
        templates.append({
            'name': 'Aggressive',
            'description': 'Higher thresholds, more sensitive detection',
            'config': aggressive_config
        })
        
        # Balanced template (default)
        default_config = self._get_default_config(agent_name)
        templates.append({
            'name': 'Balanced',
            'description': 'Recommended settings for most use cases',
            'config': default_config
        })
        
        return templates
    
    def _generate_conservative_config(self, agent_name: str) -> Dict[str, Any]:
        """Generate conservative configuration template"""
        default_config = self._get_default_config(agent_name)
        
        # Apply conservative adjustments
        conservative_adjustments = {
            'confidence_threshold': 85,  # Higher confidence required
            'high_value_threshold': 500000,  # Higher value threshold
            'risk_threshold': 80,  # Higher risk threshold
            'sandbagging_threshold': 80  # Higher sandbagging threshold
        }
        
        for key, value in conservative_adjustments.items():
            if key in default_config:
                default_config[key] = value
        
        return default_config
    
    def _generate_aggressive_config(self, agent_name: str) -> Dict[str, Any]:
        """Generate aggressive configuration template"""
        default_config = self._get_default_config(agent_name)
        
        # Apply aggressive adjustments
        aggressive_adjustments = {
            'confidence_threshold': 60,  # Lower confidence required
            'high_value_threshold': 100000,  # Lower value threshold
            'risk_threshold': 50,  # Lower risk threshold
            'sandbagging_threshold': 50  # Lower sandbagging threshold
        }
        
        for key, value in aggressive_adjustments.items():
            if key in default_config:
                default_config[key] = value
        
        return default_config
    
    def _get_default_config(self, agent_name: str) -> Dict[str, Any]:
        """Get default configuration for an agent"""
        parameters = self.get_config_schema_for_agent(agent_name)
        return {param.name: param.default for param in parameters.values()}
    
    def merge_configurations(self, agent_name: str, tenant_config: Dict[str, Any] = None, 
                           user_config: Dict[str, Any] = None, prompt_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Merge configurations from multiple sources with proper precedence.
        
        Precedence (highest to lowest):
        1. Prompt-level overrides
        2. User-level defaults
        3. Tenant-level defaults
        4. Agent-level defaults
        """
        # Start with agent defaults
        config = self._get_default_config(agent_name)
        
        # Apply tenant overrides
        if tenant_config:
            config.update({k: v for k, v in tenant_config.items() if k in config})
        
        # Apply user overrides
        if user_config:
            config.update({k: v for k, v in user_config.items() if k in config})
        
        # Apply prompt overrides
        if prompt_config:
            config.update({k: v for k, v in prompt_config.items() if k in config})
        
        return config
    
    def validate_configuration(self, agent_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and type-convert configuration parameters.
        
        Args:
            agent_name: Name of the agent
            config: Configuration to validate
            
        Returns:
            Dictionary with validated config and any errors
        """
        parameters = self.get_config_schema_for_agent(agent_name)
        validated_config = {}
        errors = []
        
        for param_name, param_def in parameters.items():
            value = config.get(param_name, param_def.default)
            
            try:
                # Type conversion
                if param_def.type == ConfigParameterType.INTEGER:
                    validated_value = int(float(value)) if value is not None else param_def.default
                elif param_def.type == ConfigParameterType.FLOAT:
                    validated_value = float(value) if value is not None else param_def.default
                elif param_def.type in [ConfigParameterType.CURRENCY, ConfigParameterType.PERCENTAGE]:
                    validated_value = float(value) if value is not None else param_def.default
                elif param_def.type == ConfigParameterType.BOOLEAN:
                    validated_value = bool(value) if isinstance(value, bool) else str(value).lower() in ['true', '1', 'yes']
                else:
                    validated_value = str(value) if value is not None else param_def.default
                
                # Range validation
                if param_def.min_value is not None and validated_value < param_def.min_value:
                    errors.append(f"{param_name}: Value {validated_value} below minimum {param_def.min_value}")
                    validated_value = param_def.min_value
                
                if param_def.max_value is not None and validated_value > param_def.max_value:
                    errors.append(f"{param_name}: Value {validated_value} above maximum {param_def.max_value}")
                    validated_value = param_def.max_value
                
                validated_config[param_name] = validated_value
                
            except (ValueError, TypeError) as e:
                errors.append(f"{param_name}: Invalid value '{value}' - {str(e)}")
                validated_config[param_name] = param_def.default
        
        return {
            'config': validated_config,
            'errors': errors,
            'is_valid': len(errors) == 0
        }
    
    def get_all_ui_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get UI schemas for all discovered agents"""
        agents = self.discovery.discover_agents()
        ui_schemas = {}
        
        for agent_name in agents.keys():
            ui_schemas[agent_name] = self.get_ui_schema_for_agent(agent_name)
        
        return ui_schemas
    
    def preview_configuration_impact(self, agent_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preview the impact of a configuration change.
        
        Args:
            agent_name: Name of the agent
            config: Configuration to preview
            
        Returns:
            Preview information including expected behavior changes
        """
        parameters = self.get_config_schema_for_agent(agent_name)
        default_config = self._get_default_config(agent_name)
        
        changes = []
        impact_summary = {
            'sensitivity': 'medium',  # low, medium, high
            'expected_results': 'similar',  # fewer, similar, more
            'confidence_level': 'medium'  # low, medium, high
        }
        
        for param_name, new_value in config.items():
            if param_name in parameters and param_name in default_config:
                default_value = default_config[param_name]
                param_def = parameters[param_name]
                
                if new_value != default_value:
                    change_impact = self._analyze_parameter_impact(param_name, param_def, default_value, new_value)
                    changes.append(change_impact)
        
        # Aggregate impact
        if changes:
            impact_summary = self._aggregate_impact(changes)
        
        return {
            'changes': changes,
            'impact_summary': impact_summary,
            'recommendations': self._generate_recommendations(agent_name, changes)
        }
    
    def _analyze_parameter_impact(self, param_name: str, param_def: ConfigParameter, 
                                 old_value: Any, new_value: Any) -> Dict[str, Any]:
        """Analyze the impact of changing a single parameter"""
        change_direction = 'increase' if new_value > old_value else 'decrease'
        change_magnitude = abs((new_value - old_value) / old_value) if old_value != 0 else 1
        
        impact = {
            'parameter': param_name,
            'old_value': old_value,
            'new_value': new_value,
            'change_direction': change_direction,
            'change_magnitude': change_magnitude,
            'expected_effect': self._predict_parameter_effect(param_name, change_direction)
        }
        
        return impact
    
    def _predict_parameter_effect(self, param_name: str, change_direction: str) -> str:
        """Predict the effect of changing a parameter"""
        threshold_effects = {
            'increase': 'fewer results, higher precision',
            'decrease': 'more results, lower precision'
        }
        
        if 'threshold' in param_name.lower():
            return threshold_effects[change_direction]
        elif 'confidence' in param_name.lower():
            return threshold_effects[change_direction]
        else:
            return 'moderate impact on results'
    
    def _aggregate_impact(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate the impact of multiple parameter changes"""
        total_magnitude = sum(change['change_magnitude'] for change in changes)
        
        if total_magnitude < 0.2:
            sensitivity = 'low'
            expected_results = 'similar'
        elif total_magnitude < 0.5:
            sensitivity = 'medium'
            expected_results = 'moderately different'
        else:
            sensitivity = 'high'
            expected_results = 'significantly different'
        
        return {
            'sensitivity': sensitivity,
            'expected_results': expected_results,
            'confidence_level': 'medium'
        }
    
    def _generate_recommendations(self, agent_name: str, changes: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on configuration changes"""
        recommendations = []
        
        high_magnitude_changes = [c for c in changes if c['change_magnitude'] > 0.5]
        if high_magnitude_changes:
            recommendations.append("Consider testing with a small dataset first due to significant parameter changes")
        
        threshold_decreases = [c for c in changes if 'threshold' in c['parameter'] and c['change_direction'] == 'decrease']
        if len(threshold_decreases) > 2:
            recommendations.append("Multiple threshold decreases may result in many false positives")
        
        if not recommendations:
            recommendations.append("Configuration changes look reasonable for production use")
        
        return recommendations

# Global config manager instance
_config_manager_instance = None

def get_config_manager() -> EnhancedConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = EnhancedConfigManager()
    return _config_manager_instance
