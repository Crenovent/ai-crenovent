"""
Enterprise Workflow Builder API - Refactored
FastAPI endpoints organized into focused modules
"""

from fastapi import FastAPI
import logging

# Import all the focused API modules
from .workflow_builder import router as workflow_builder_router
from .pipeline_agents import router as pipeline_agents_router
from .rba_config import router as rba_config_router
from .universal_parameters import router as universal_parameters_router
from .templates import router as templates_router
from .csv_integration import router as csv_integration_router
from .governance import router as governance_router
from .parameter_discovery import router as parameter_discovery_router

logger = logging.getLogger(__name__)


def include_workflow_builder_routes(app: FastAPI):
    """Include all workflow builder routes in FastAPI app"""
    
    # Core workflow builder
    app.include_router(workflow_builder_router, tags=["Workflow Builder"])
    
    # Pipeline agents (the main feature)
    app.include_router(pipeline_agents_router, tags=["Pipeline Agents"])
    
    # RBA Configuration API
    app.include_router(rba_config_router, tags=["RBA Configuration"])
    app.include_router(universal_parameters_router, prefix="/api", tags=["Universal Parameters"])
    
    # Templates and agents
    app.include_router(templates_router, tags=["Templates"])
    
    # CSV integration
    app.include_router(csv_integration_router, tags=["CSV Integration"])
    
    # Governance and audit
    app.include_router(governance_router, tags=["Governance"])
    
    # Parameter discovery
    app.include_router(parameter_discovery_router, tags=["Parameter Discovery"])
    
    logger.info("✅ All Workflow Builder API routes registered")
    logger.info("📦 Modules: workflow_builder, pipeline_agents, templates, csv_integration, governance, parameter_discovery")


# Backward compatibility exports
from .pipeline_agents import router as pipeline_router
from .workflow_builder import router as router

# For any code that imports specific functions
from .workflow_builder import get_scenario_workflow, convert_visual_to_dsl
