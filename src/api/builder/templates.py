"""
Builder API - Template Management Endpoints
==========================================

Implements Chapter 12.1 Template APIs:
- SaaS template library management
- Template discovery and metadata
- Template application to workflows

Provides ready-to-use SaaS templates for frontend testing.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from dsl.registry.enhanced_capability_registry import EnhancedCapabilityRegistry, CapabilityType

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class TemplateResponse(BaseModel):
    template_id: str
    name: str
    description: str
    industry: str
    automation_type: str
    category: str
    trust_score: float
    complexity_score: float
    estimated_duration_ms: int
    usage_patterns: List[str]
    common_inputs: List[str]
    expected_outputs: List[str]

class TemplateDetailResponse(BaseModel):
    template_id: str
    name: str
    description: str
    workflow_definition: str
    metadata: Dict[str, Any]

# Dependency to get app state
async def get_app_state(request: Request):
    return request.app.state

@router.get("/templates", response_model=List[TemplateResponse])
async def list_templates(
    industry: Optional[str] = None,
    category: Optional[str] = None,
    automation_type: Optional[str] = None,
    app_state = Depends(get_app_state)
):
    """
    List available workflow templates
    Task 12.1-T07: SaaS block library (pipeline hygiene, comp plan adjustments)
    """
    try:
        logger.info(f"📋 Listing templates - industry: {industry}, category: {category}")
        
        # Get templates from Smart Capability Registry
        registry = app_state.registry
        
        # Filter templates
        templates = []
        for capability_id, capability in registry.capabilities.items():
            # Filter by type (only RBA templates)
            if capability.capability_type != CapabilityType.RBA_TEMPLATE:
                continue
                
            # Apply filters
            if industry and industry.lower() not in [tag.lower() for tag in capability.industry_tags]:
                continue
            if category and category.lower() not in capability.category.lower():
                continue
            if automation_type and automation_type.upper() != "RBA":
                continue
            
            templates.append(TemplateResponse(
                template_id=capability.id,
                name=capability.name,
                description=capability.description,
                industry=", ".join(capability.industry_tags),
                automation_type="RBA",
                category=capability.category,
                trust_score=capability.trust_score,
                complexity_score=capability.complexity_score,
                estimated_duration_ms=int(capability.avg_execution_time_ms),
                usage_patterns=capability.usage_patterns,
                common_inputs=capability.common_inputs,
                expected_outputs=capability.expected_outputs
            ))
        
        logger.info(f"✅ Found {len(templates)} templates")
        return templates
        
    except Exception as e:
        logger.error(f"❌ Failed to list templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")

@router.get("/templates/{template_id}", response_model=TemplateDetailResponse)
async def get_template(
    template_id: str,
    app_state = Depends(get_app_state)
):
    """
    Get specific template with workflow definition
    Task 12.1-T08: Template retrieval with DSL
    """
    try:
        logger.info(f"🔍 Getting template: {template_id}")
        
        # Get template from registry
        registry = app_state.registry
        capability = registry.capabilities.get(template_id)
        
        if not capability:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Get workflow definition based on template
        workflow_definition = await _get_template_workflow_definition(template_id)
        
        return TemplateDetailResponse(
            template_id=capability.id,
            name=capability.name,
            description=capability.description,
            workflow_definition=workflow_definition,
            metadata={
                "industry_tags": capability.industry_tags,
                "persona_tags": capability.persona_tags,
                "category": capability.category,
                "trust_score": capability.trust_score,
                "complexity_score": capability.complexity_score,
                "usage_patterns": capability.usage_patterns,
                "common_inputs": capability.common_inputs,
                "expected_outputs": capability.expected_outputs,
                "readiness_state": capability.readiness_state.value
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")

@router.get("/templates/categories")
async def list_template_categories(
    app_state = Depends(get_app_state)
):
    """
    List available template categories
    """
    try:
        registry = app_state.registry
        
        categories = set()
        industries = set()
        personas = set()
        
        for capability in registry.capabilities.values():
            if capability.capability_type == CapabilityType.RBA_TEMPLATE:
                categories.add(capability.category)
                industries.update(capability.industry_tags)
                personas.update(capability.persona_tags)
        
        return {
            "categories": sorted(list(categories)),
            "industries": sorted(list(industries)),
            "personas": sorted(list(personas))
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")

async def _get_template_workflow_definition(template_id: str) -> str:
    """Get workflow definition for template"""
    
    # Import here to avoid circular imports
    from dsl.compiler.parser import SAMPLE_PIPELINE_HYGIENE_WORKFLOW, SAMPLE_FORECAST_APPROVAL_WORKFLOW
    
    # Map template IDs to workflow definitions
    template_workflows = {
        # Pipeline Hygiene template
        "rba_saas_pipeline_hygiene": SAMPLE_PIPELINE_HYGIENE_WORKFLOW,
        
        # Forecast Approval template  
        "rba_saas_forecast_approval": SAMPLE_FORECAST_APPROVAL_WORKFLOW,
    }
    
    # Check if we have a predefined workflow for this template
    for key, workflow in template_workflows.items():
        if key in template_id:
            return workflow
    
    # Default template workflow
    return f"""
workflow_id: "{template_id}_workflow"
name: "Template Workflow"
description: "Generated from template {template_id}"
version: "1.0.0"

governance:
  policy_packs: ["saas_compliance"]
  evidence_required: true
  override_allowed: false

steps:
  - id: "template_step"
    name: "Template Step"
    type: "action"
    params:
      template_id: "{template_id}"
      action: "execute_template"
    next_steps: ["complete"]
    
  - id: "complete"
    name: "Complete Workflow"
    type: "governance"
    params:
      action: "create_evidence_pack"
      evidence_type: "template_execution"
"""
