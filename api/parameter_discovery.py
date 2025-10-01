"""
Parameter Discovery API
Provides dynamic parameter schemas for frontend configuration
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from dsl.parameters.dynamic_parameter_discovery import parameter_discovery

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/parameters", tags=["Parameter Discovery"])

@router.get("/discover/{workflow_type}")
async def discover_workflow_parameters(
    workflow_type: str,
    expertise_level: str = Query(default="expert", description="User expertise level: basic, advanced, expert"),
    include_context: bool = Query(default=True, description="Include business context and help text")
) -> Dict[str, Any]:
    """
    Discover dynamic parameters for a workflow type
    
    Args:
        workflow_type: The workflow type (sandbagging_detection, stage_velocity_analysis, etc.)
        expertise_level: User expertise level (basic, advanced, expert)
        include_context: Whether to include business context and help text
    
    Returns:
        Dynamic parameter schema for frontend configuration
    """
    try:
        logger.info(f"🔍 Parameter discovery request: {workflow_type} at {expertise_level} level")
        
        # Get dynamic parameter schema
        schema = parameter_discovery.get_parameter_schema_for_frontend(
            workflow_type=workflow_type,
            expertise_level=expertise_level
        )
        
        # Filter out context if not requested
        if not include_context:
            for param in schema['parameters']:
                param.pop('help_text', None)
                param.pop('business_context', None)
        
        logger.info(f"✅ Generated schema with {schema['parameter_count']} parameters")
        
        return {
            "success": True,
            "message": f"Parameter schema generated for {workflow_type}",
            "data": schema,
            "metadata": {
                "available_workflows": ["sandbagging_detection", "stage_velocity_analysis", "pipeline_health_overview"],
                "expertise_levels": ["basic", "advanced", "expert"],
                "total_parameters": schema['parameter_count'],
                "expert_only_count": len([p for p in schema['parameters'] if p.get('expert_only', False)])
            }
        }
        
    except ValueError as e:
        logger.error(f"❌ Invalid workflow type: {workflow_type}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Parameter discovery failed: {e}")
        raise HTTPException(status_code=500, detail="Parameter discovery failed")

@router.get("/available-workflows")
async def get_available_workflows() -> Dict[str, Any]:
    """
    Get list of available workflows and their contexts
    """
    try:
        workflows = parameter_discovery.workflow_contexts
        
        return {
            "success": True,
            "message": "Available workflows retrieved",
            "data": {
                "workflows": list(workflows.keys()),
                "contexts": workflows,
                "total_count": len(workflows)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get workflows: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve workflows")

@router.get("/field-capabilities")
async def get_field_capabilities() -> Dict[str, Any]:
    """
    Get available CSV fields and their parameter generation capabilities
    """
    try:
        fields = parameter_discovery.available_fields
        
        return {
            "success": True,
            "message": "Field capabilities retrieved", 
            "data": {
                "available_fields": list(fields.keys()),
                "field_details": fields,
                "total_fields": len(fields)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get field capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve field capabilities")

@router.post("/validate-configuration")
async def validate_parameter_configuration(
    workflow_type: str,
    configuration: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate a parameter configuration against the workflow schema
    """
    try:
        logger.info(f"🔍 Validating configuration for {workflow_type}")
        
        # Get the schema for validation
        schema = parameter_discovery.get_parameter_schema_for_frontend(workflow_type, "expert")
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "applied_defaults": {}
        }
        
        # Validate each parameter
        for param in schema['parameters']:
            key = param['key']
            value = configuration.get(key)
            
            if value is None:
                # Apply default
                validation_results['applied_defaults'][key] = param['default']
                continue
            
            # Type validation
            param_type = param['type']
            if param_type in ['threshold', 'percentage'] and (value < 0 or value > 100):
                validation_results['errors'].append(f"{key}: Percentage must be between 0-100")
                validation_results['valid'] = False
            
            if param_type == 'amount' and value < 0:
                validation_results['errors'].append(f"{key}: Amount cannot be negative")
                validation_results['valid'] = False
            
            # Range validation
            if 'min' in param and value < param['min']:
                validation_results['errors'].append(f"{key}: Value {value} below minimum {param['min']}")
                validation_results['valid'] = False
            
            if 'max' in param and value > param['max']:
                validation_results['errors'].append(f"{key}: Value {value} above maximum {param['max']}")
                validation_results['valid'] = False
        
        logger.info(f"✅ Configuration validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
        
        return {
            "success": True,
            "message": "Configuration validated",
            "data": validation_results
        }
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        raise HTTPException(status_code=500, detail="Validation failed")
