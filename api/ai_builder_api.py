#!/usr/bin/env python3
"""
AI Builder API - Natural Language to DSL Conversion
==================================================
REST API endpoints for AI-assisted workflow builder

Endpoints:
- POST /ai-builder/convert-nl-to-dsl - Convert natural language to DSL
- POST /ai-builder/validate-dsl - Validate generated DSL
- GET /ai-builder/templates/{industry} - Get industry templates
- POST /ai-builder/enhance-workflow - Enhance workflow with AI suggestions
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

# DSL Components
from dsl.ai_builder.nl_to_dsl_converter import (
    NLToDSLConverter, 
    NLToDSLRequest, 
    DSLGenerationResult,
    get_nl_to_dsl_converter
)
from src.services.connection_pool_manager import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-builder", tags=["AI Builder"])

# Request/Response Models
class NLToDSLConversionRequest(BaseModel):
    natural_language: str = Field(..., description="Natural language workflow description")
    tenant_id: int = Field(..., description="Tenant ID")
    user_id: int = Field(..., description="User ID")
    industry: str = Field(default="SaaS", description="Industry context")
    persona: str = Field(default="RevOps Manager", description="User persona")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    compliance_frameworks: Optional[List[str]] = Field(default=None, description="Compliance frameworks")

class DSLValidationRequest(BaseModel):
    dsl_workflow: Dict[str, Any] = Field(..., description="DSL workflow to validate")
    tenant_id: int = Field(..., description="Tenant ID")
    industry: str = Field(default="SaaS", description="Industry context")

class WorkflowEnhancementRequest(BaseModel):
    dsl_workflow: Dict[str, Any] = Field(..., description="DSL workflow to enhance")
    tenant_id: int = Field(..., description="Tenant ID")
    enhancement_type: str = Field(default="suggestions", description="Type of enhancement")

class NLToDSLResponse(BaseModel):
    success: bool
    dsl_workflow: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    reasoning: str = ""
    validation_errors: List[str] = []
    governance_warnings: List[str] = []
    suggested_improvements: List[str] = []
    processing_time_ms: int = 0

class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

class IndustryTemplatesResponse(BaseModel):
    industry: str
    templates: Dict[str, Any]
    available_blocks: List[Dict[str, Any]]

@router.post("/convert-nl-to-dsl", response_model=NLToDSLResponse)
async def convert_natural_language_to_dsl(
    request: NLToDSLConversionRequest,
    pool_manager=Depends(get_pool_manager)
):
    """Convert natural language description to DSL workflow"""
    try:
        start_time = datetime.now()
        
        logger.info(f"🤖 Converting NL to DSL for tenant {request.tenant_id}: '{request.natural_language[:100]}...'")
        
        # Get NL → DSL converter
        converter = await get_nl_to_dsl_converter(pool_manager)
        
        # Create conversion request
        nl_request = NLToDSLRequest(
            natural_language=request.natural_language,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            industry=request.industry,
            persona=request.persona,
            context=request.context or {},
            compliance_frameworks=request.compliance_frameworks or ["SOX", "GDPR"]
        )
        
        # Perform conversion
        result = await converter.convert_nl_to_dsl(nl_request)
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        logger.info(f"✅ NL → DSL conversion completed in {processing_time}ms with confidence {result.confidence_score:.2f}")
        
        return NLToDSLResponse(
            success=result.success,
            dsl_workflow=result.dsl_workflow,
            confidence_score=result.confidence_score,
            reasoning=result.reasoning,
            validation_errors=result.validation_errors,
            governance_warnings=result.governance_warnings,
            suggested_improvements=result.suggested_improvements,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ NL → DSL conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@router.post("/validate-dsl", response_model=ValidationResponse)
async def validate_dsl_workflow(
    request: DSLValidationRequest,
    pool_manager=Depends(get_pool_manager)
):
    """Validate a DSL workflow"""
    try:
        logger.info(f"🔍 Validating DSL workflow for tenant {request.tenant_id}")
        
        # Get NL → DSL converter for validation
        converter = await get_nl_to_dsl_converter(pool_manager)
        
        # Create dummy request for validation
        nl_request = NLToDSLRequest(
            natural_language="validation request",
            tenant_id=request.tenant_id,
            user_id=1,
            industry=request.industry
        )
        
        # Validate DSL
        validation_result = await converter._validate_dsl(request.dsl_workflow, nl_request)
        
        # Generate suggestions
        suggestions = await converter._generate_suggestions(
            request.dsl_workflow, nl_request, validation_result
        )
        
        logger.info(f"✅ DSL validation completed: {'valid' if validation_result['valid'] else 'invalid'}")
        
        return ValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result.get("errors", []),
            warnings=validation_result.get("warnings", []),
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"❌ DSL validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/templates/{industry}", response_model=IndustryTemplatesResponse)
async def get_industry_templates(
    industry: str,
    pool_manager=Depends(get_pool_manager)
):
    """Get industry-specific workflow templates"""
    try:
        logger.info(f"📋 Getting templates for industry: {industry}")
        
        # Get NL → DSL converter
        converter = await get_nl_to_dsl_converter(pool_manager)
        
        # Get industry templates
        templates = converter.industry_templates.get(industry, {})
        
        # Get available blocks for the industry
        available_blocks = await _get_industry_blocks(industry)
        
        logger.info(f"✅ Retrieved {len(templates)} templates for {industry}")
        
        return IndustryTemplatesResponse(
            industry=industry,
            templates=templates,
            available_blocks=available_blocks
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get industry templates: {e}")
        raise HTTPException(status_code=500, detail=f"Template retrieval failed: {str(e)}")

@router.post("/enhance-workflow")
async def enhance_workflow_with_ai(
    request: WorkflowEnhancementRequest,
    pool_manager=Depends(get_pool_manager)
):
    """Enhance workflow with AI suggestions"""
    try:
        logger.info(f"✨ Enhancing workflow for tenant {request.tenant_id}")
        
        # Get NL → DSL converter
        converter = await get_nl_to_dsl_converter(pool_manager)
        
        # Create dummy request for enhancement
        nl_request = NLToDSLRequest(
            natural_language="enhancement request",
            tenant_id=request.tenant_id,
            user_id=1
        )
        
        # Validate current workflow
        validation_result = await converter._validate_dsl(request.dsl_workflow, nl_request)
        
        # Generate suggestions
        suggestions = await converter._generate_suggestions(
            request.dsl_workflow, nl_request, validation_result
        )
        
        # Apply enhancements based on type
        enhanced_workflow = request.dsl_workflow.copy()
        
        if request.enhancement_type == "governance":
            enhanced_workflow = await converter._embed_governance(enhanced_workflow, nl_request)
        elif request.enhancement_type == "industry":
            enhanced_workflow = await converter._apply_industry_overlays(enhanced_workflow, nl_request)
        
        logger.info(f"✅ Workflow enhancement completed with {len(suggestions)} suggestions")
        
        return {
            "success": True,
            "enhanced_workflow": enhanced_workflow,
            "suggestions": suggestions,
            "enhancement_type": request.enhancement_type
        }
        
    except Exception as e:
        logger.error(f"❌ Workflow enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@router.get("/capabilities")
async def get_ai_builder_capabilities():
    """Get AI Builder capabilities and supported features"""
    try:
        capabilities = {
            "supported_industries": [
                "SaaS", "Banking", "Insurance", "E-commerce", 
                "FinancialServices", "ITServices"
            ],
            "supported_personas": [
                "RevOps Manager", "Sales Manager", "CRO", "CFO", 
                "Compliance Officer", "Operations Manager"
            ],
            "compliance_frameworks": [
                "SOX", "GDPR", "HIPAA", "RBI", "IRDAI", "PCI_DSS", "SOC2"
            ],
            "workflow_types": [
                "pipeline_hygiene", "approval_workflow", "data_sync", 
                "notification", "compliance_check", "calculation"
            ],
            "features": [
                "natural_language_conversion",
                "industry_overlays",
                "governance_embedding",
                "validation",
                "suggestions",
                "confidence_scoring"
            ]
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"❌ Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Capabilities retrieval failed: {str(e)}")

async def _get_industry_blocks(industry: str) -> List[Dict[str, Any]]:
    """Get available blocks for an industry"""
    
    industry_blocks = {
        "SaaS": [
            {
                "id": "pipeline_query",
                "name": "Pipeline Query",
                "description": "Query pipeline data from Salesforce",
                "type": "query",
                "category": "data"
            },
            {
                "id": "opportunity_decision",
                "name": "Opportunity Decision",
                "description": "Make decisions based on opportunity data",
                "type": "decision",
                "category": "logic"
            },
            {
                "id": "slack_notification",
                "name": "Slack Notification",
                "description": "Send notifications to Slack",
                "type": "notify",
                "category": "communication"
            }
        ],
        "Banking": [
            {
                "id": "kyc_check",
                "name": "KYC Check",
                "description": "Perform KYC validation",
                "type": "governance",
                "category": "compliance"
            },
            {
                "id": "credit_score_query",
                "name": "Credit Score Query",
                "description": "Query credit score data",
                "type": "query",
                "category": "data"
            },
            {
                "id": "loan_approval_decision",
                "name": "Loan Approval Decision",
                "description": "Make loan approval decisions",
                "type": "decision",
                "category": "logic"
            }
        ],
        "Insurance": [
            {
                "id": "solvency_check",
                "name": "Solvency Check",
                "description": "Check solvency ratios",
                "type": "governance",
                "category": "compliance"
            },
            {
                "id": "claims_query",
                "name": "Claims Query",
                "description": "Query claims data",
                "type": "query",
                "category": "data"
            },
            {
                "id": "payout_decision",
                "name": "Payout Decision",
                "description": "Make claims payout decisions",
                "type": "decision",
                "category": "logic"
            }
        ]
    }
    
    return industry_blocks.get(industry, [])
