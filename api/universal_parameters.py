"""
Universal RBA Parameters API
===========================
API endpoints for managing universal RBA parameters across all agents.
Provides parameter discovery, configuration, and UI schema generation.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from dsl.configuration.universal_parameter_manager import get_universal_parameter_manager, ParameterRelevance

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/universal-parameters", tags=["Universal Parameters"])

# Pydantic models for request/response
class ParameterConfigRequest(BaseModel):
    agent_name: str = Field(..., description="Name of the RBA agent")
    configuration: Dict[str, Any] = Field(..., description="Parameter configuration to validate")
    tenant_config: Optional[Dict[str, Any]] = Field(None, description="Tenant-level configuration")

class ParameterConfigResponse(BaseModel):
    success: bool
    agent_name: str
    validated_config: Dict[str, Any]
    parameter_count: int
    validation_errors: List[str] = []

class UISchemaRequest(BaseModel):
    agent_name: str = Field(..., description="Name of the RBA agent")
    include_all_params: bool = Field(False, description="Include all universal parameters")

class UISchemaResponse(BaseModel):
    success: bool
    agent_name: str
    ui_schema: Dict[str, Any]  # Changed from 'schema' to avoid BaseModel conflict
    total_parameters: int

@router.get("/", summary="Get all universal parameters")
async def get_all_universal_parameters() -> Dict[str, Any]:
    """
    Get all universal parameters defined in the system.
    
    Returns:
        Complete list of universal parameters with definitions
    """
    
    try:
        param_manager = get_universal_parameter_manager()
        all_params = param_manager.get_all_parameters()
        
        # Convert to API response format
        parameters = {}
        for name, param_def in all_params.items():
            parameters[name] = {
                'name': name,
                'type': param_def.type,
                'default': param_def.default,
                'min_value': param_def.min_value,
                'max_value': param_def.max_value,
                'description': param_def.description,
                'ui_component': param_def.ui_component,
                'ui_group': param_def.ui_group,
                'relevant_agents': param_def.relevant_agents
            }
        
        stats = param_manager.get_parameter_statistics()
        
        return {
            'success': True,
            'parameters': parameters,
            'statistics': stats,
            'total_parameters': len(parameters)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get universal parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get universal parameters: {str(e)}")

@router.get("/agent/{agent_name}", summary="Get parameters for specific agent")
async def get_parameters_for_agent(
    agent_name: str,
    relevance_levels: Optional[List[str]] = Query(
        default=["primary", "secondary"], 
        description="Relevance levels to include"
    )
) -> Dict[str, Any]:
    """
    Get parameters relevant to a specific RBA agent.
    
    Args:
        agent_name: Name of the RBA agent
        relevance_levels: Which relevance levels to include
        
    Returns:
        Parameters relevant to the specified agent
    """
    
    try:
        param_manager = get_universal_parameter_manager()
        
        # Convert string relevance levels to enum
        relevance_enums = []
        for level in relevance_levels:
            try:
                relevance_enums.append(ParameterRelevance(level))
            except ValueError:
                logger.warning(f"⚠️ Invalid relevance level: {level}")
        
        # Get relevant parameters
        relevant_params = param_manager.get_parameters_for_agent(agent_name, relevance_enums)
        
        # Get default configuration
        default_config = param_manager.get_default_config_for_agent(agent_name, include_tertiary=True)
        
        # Convert to API response format
        parameters = {}
        for name, param_def in relevant_params.items():
            parameters[name] = {
                'name': name,
                'type': param_def.type,
                'default': param_def.default,
                'current_value': default_config.get(name, param_def.default),
                'min_value': param_def.min_value,
                'max_value': param_def.max_value,
                'description': param_def.description,
                'ui_component': param_def.ui_component,
                'ui_group': param_def.ui_group,
                'relevance': param_manager._get_parameter_relevance_for_agent(agent_name, name)
            }
        
        return {
            'success': True,
            'agent_name': agent_name,
            'parameters': parameters,
            'default_configuration': default_config,
            'total_parameters': len(parameters),
            'relevance_levels_included': relevance_levels
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get parameters for agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get parameters for agent: {str(e)}")

@router.post("/validate", summary="Validate parameter configuration")
async def validate_parameter_configuration(request: ParameterConfigRequest) -> ParameterConfigResponse:
    """
    Validate and merge parameter configuration for an agent.
    
    Args:
        request: Configuration validation request
        
    Returns:
        Validated configuration with any errors
    """
    
    try:
        param_manager = get_universal_parameter_manager()
        
        # Validate and merge configuration
        validated_config = param_manager.merge_configs(
            request.agent_name,
            request.configuration,
            request.tenant_config
        )
        
        return ParameterConfigResponse(
            success=True,
            agent_name=request.agent_name,
            validated_config=validated_config,
            parameter_count=len(validated_config),
            validation_errors=[]
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to validate configuration for {request.agent_name}: {e}")
        return ParameterConfigResponse(
            success=False,
            agent_name=request.agent_name,
            validated_config={},
            parameter_count=0,
            validation_errors=[str(e)]
        )

@router.post("/ui-schema", summary="Generate UI schema for parameter configuration")
async def generate_ui_schema(request: UISchemaRequest) -> UISchemaResponse:
    """
    Generate UI schema for frontend parameter configuration.
    
    Args:
        request: UI schema generation request
        
    Returns:
        UI schema suitable for frontend rendering
    """
    
    try:
        param_manager = get_universal_parameter_manager()
        
        # Generate UI schema
        ui_schema = param_manager.generate_ui_schema_for_agent(
            request.agent_name,
            include_all_params=request.include_all_params
        )
        
        return UISchemaResponse(
            success=True,
            agent_name=request.agent_name,
            ui_schema=ui_schema,
            total_parameters=ui_schema.get('total_parameters', 0)
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate UI schema for {request.agent_name}: {e}")
        return UISchemaResponse(
            success=False,
            agent_name=request.agent_name,
            ui_schema={},
            total_parameters=0
        )

@router.get("/statistics", summary="Get parameter system statistics")
async def get_parameter_statistics() -> Dict[str, Any]:
    """
    Get comprehensive statistics about the universal parameter system.
    
    Returns:
        Statistics about parameters, agents, and usage
    """
    
    try:
        param_manager = get_universal_parameter_manager()
        stats = param_manager.get_parameter_statistics()
        
        return {
            'success': True,
            'statistics': stats,
            'timestamp': '2025-09-19T12:30:00Z'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get parameter statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get parameter statistics: {str(e)}")

@router.get("/agents", summary="Get list of supported agents")
async def get_supported_agents() -> Dict[str, Any]:
    """
    Get list of all RBA agents that support universal parameters.
    
    Returns:
        List of supported agents with their parameter counts
    """
    
    try:
        param_manager = get_universal_parameter_manager()
        stats = param_manager.get_parameter_statistics()
        
        agents = []
        for agent_name, param_count in stats.get('parameters_per_agent', {}).items():
            # Get primary parameters for this agent
            primary_params = param_manager.get_parameters_for_agent(
                agent_name, 
                [ParameterRelevance.PRIMARY]
            )
            
            agents.append({
                'agent_name': agent_name,
                'total_parameters': param_count,
                'primary_parameters': len(primary_params),
                'description': f"RBA agent: {agent_name}",
                'supported_analysis_types': [agent_name]
            })
        
        return {
            'success': True,
            'agents': agents,
            'total_agents': len(agents)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get supported agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported agents: {str(e)}")

@router.get("/defaults/{agent_name}", summary="Get default configuration for agent")
async def get_default_configuration(
    agent_name: str,
    include_tertiary: bool = Query(False, description="Include tertiary parameters")
) -> Dict[str, Any]:
    """
    Get default configuration for a specific agent.
    
    Args:
        agent_name: Name of the RBA agent
        include_tertiary: Whether to include tertiary parameters
        
    Returns:
        Default configuration for the agent
    """
    
    try:
        param_manager = get_universal_parameter_manager()
        
        default_config = param_manager.get_default_config_for_agent(
            agent_name,
            include_tertiary=include_tertiary
        )
        
        return {
            'success': True,
            'agent_name': agent_name,
            'default_configuration': default_config,
            'parameter_count': len(default_config),
            'includes_tertiary': include_tertiary
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get default configuration for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get default configuration: {str(e)}")
