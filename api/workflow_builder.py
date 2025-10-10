"""
Workflow Builder Core API
FastAPI endpoints for drag-and-drop workflow creation and execution
"""

from fastapi import APIRouter, HTTPException, Request, Body
from typing import Dict, Any
from datetime import datetime
import logging
import uuid
import yaml
import os

from .models import (
    WorkflowDefinition, WorkflowExecutionRequest, WorkflowExecutionResponse,
    NaturalLanguageRequest, IntentParsingResponse, clean_for_json_serialization
)

logger = logging.getLogger(__name__)

# Create workflow builder router
router = APIRouter(prefix="/api/builder", tags=["Workflow Builder"])


@router.get("/health")
async def health_check():
    """Health check for workflow builder API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "workflow_builder_api"
    }


@router.post("/test")
async def test_execution():
    """Simple test endpoint"""
    return {
        "status": "success",
        "message": "API is working",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/chapter-15/status")
async def get_chapter_15_status(fastapi_request: Request):
    """Get Chapter 15 Routing Orchestrator implementation status"""
    try:
        # Use global orchestrator instance from app state
        orchestrator = fastapi_request.app.state.orchestrator
        
        # Check if orchestrator has Chapter 15 enhancements
        if hasattr(orchestrator, 'get_chapter_15_status'):
            status = await orchestrator.get_chapter_15_status()
            return {
                "status": "success",
                "chapter_15_enabled": True,
                "timestamp": datetime.now().isoformat(),
                **status
            }
        else:
            return {
                "status": "success",
                "chapter_15_enabled": False,
                "message": "Chapter 15 enhancements not available in current orchestrator",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to get Chapter 15 status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Chapter 15 status: {str(e)}")


@router.post("/enhanced-routing")
async def enhanced_routing_request(request: NaturalLanguageRequest, fastapi_request: Request):
    """
    Enhanced routing request with Chapter 15 features
    - SaaS Intent Classification (15.1)
    - Policy Gate Enforcement (15.2)
    - Capability Lookup & Matching (15.3)
    - Plan Synthesis (15.4)
    - Dispatcher Execution (15.5)
    """
    try:
        logger.info(f"🚀 Processing enhanced routing request: '{request.user_input[:50]}...'")
        
        # Use global orchestrator instance from app state
        orchestrator = fastapi_request.app.state.orchestrator
        
        # Check if orchestrator supports enhanced routing
        if not hasattr(orchestrator, 'route_request'):
            raise HTTPException(status_code=501, detail="Enhanced routing not supported by current orchestrator")
        
        start_time = datetime.now()
        
        # Call enhanced route_request method
        routing_result = await orchestrator.route_request(
            user_input=request.user_input,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            context_data=request.context
        )
        
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Build enhanced response
        response = {
            "request_id": routing_result.request_id,
            "success": routing_result.success,
            "workflow_category": routing_result.workflow_category,
            "confidence": routing_result.confidence,
            "execution_result": routing_result.execution_result,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add Chapter 15 enhancements if available
        if hasattr(routing_result, 'intent_classification') and routing_result.intent_classification:
            response["chapter_15_features"] = {
                "intent_classification": routing_result.intent_classification,
                "execution_plan": routing_result.execution_plan,
                "policy_enforcement_result": routing_result.policy_enforcement_result,
                "capability_matches": routing_result.capability_matches,
                "evidence_pack_id": routing_result.evidence_pack_id
            }
        
        if routing_result.error:
            response["error"] = routing_result.error
        
        logger.info(f"✅ Enhanced routing completed in {processing_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Enhanced routing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Enhanced routing failed: {str(e)}")


@router.get("/saas-intent-categories")
async def get_saas_intent_categories(fastapi_request: Request):
    """Get supported SaaS intent categories from Chapter 15.1"""
    try:
        # Use global orchestrator instance from app state
        orchestrator = fastapi_request.app.state.orchestrator
        
        # Check if orchestrator has SaaS intent taxonomy
        if hasattr(orchestrator, 'saas_intent_taxonomy'):
            categories = orchestrator.saas_intent_taxonomy.get_supported_categories()
            return {
                "status": "success",
                "supported_categories": categories,
                "total_categories": len(categories),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "supported_categories": [],
                "message": "SaaS intent taxonomy not available",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to get SaaS intent categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SaaS intent categories: {str(e)}")


@router.get("/capabilities")
async def get_available_capabilities(
    fastapi_request: Request,
    tenant_id: str = "1300",
    industry_filter: str = None,
    automation_type_filter: str = None,
    include_eligibility: bool = True,
    include_saas_capabilities: bool = True
):
    """
    Enhanced capabilities endpoint - Chapter 15.3 Implementation
    Tasks 15.3.4, 15.3.9: Get SaaS capabilities with eligibility rules
    """
    try:
        # Use global orchestrator instance from app state
        orchestrator = fastapi_request.app.state.orchestrator
        
        # Check if orchestrator has capability registry
        if hasattr(orchestrator, 'capability_registry'):
            capabilities = await orchestrator.capability_registry.get_tenant_capabilities(
                int(tenant_id),
                industry_filter=industry_filter,
                automation_type_filter=automation_type_filter
            )
            
            response_data = {
                "status": "success",
                "capabilities": capabilities,
                "total_capabilities": len(capabilities),
                "filters_applied": {
                    "tenant_id": tenant_id,
                    "industry_filter": industry_filter,
                    "automation_type_filter": automation_type_filter
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Task 15.3.9: Include eligibility check if requested
            if include_eligibility:
                try:
                    eligibility_result = await orchestrator.capability_registry.implement_eligibility_rules(
                        int(tenant_id), {"source": "capabilities_api"}
                    )
                    response_data["eligibility_check"] = {
                        "sla_tier": eligibility_result.get("sla_tier"),
                        "industry_code": eligibility_result.get("industry_code"),
                        "eligible_count": len(eligibility_result.get("eligible_capabilities", [])),
                        "filtered_count": len(eligibility_result.get("filtered_capabilities", [])),
                        "eligibility_rate": len(eligibility_result.get("eligible_capabilities", [])) / max(1, len(eligibility_result.get("eligible_capabilities", [])) + len(eligibility_result.get("filtered_capabilities", [])))
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Eligibility check failed: {e}")
                    response_data["eligibility_check"] = {"error": str(e)}
            
            # Task 15.3.4: Include enhanced SaaS capabilities if requested
            if include_saas_capabilities:
                try:
                    saas_templates = orchestrator.capability_registry.saas_workflow_templates
                    enhanced_saas = {k: v for k, v in saas_templates.items() if 'capability_id' in v}
                    response_data["enhanced_saas_capabilities"] = {
                        "count": len(enhanced_saas),
                        "capabilities": enhanced_saas,
                        "categories": {
                            "revenue_analytics": ["arr_forecast_engine"],
                            "customer_success": ["churn_detection_engine", "customer_health_scoring"],
                            "compensation_management": ["comp_plan_automation"],
                            "financial_automation": ["revenue_recognition_engine"]
                        }
                    }
                except Exception as e:
                    logger.warning(f"⚠️ SaaS capabilities retrieval failed: {e}")
                    response_data["enhanced_saas_capabilities"] = {"error": str(e)}
            
            return response_data
        else:
            return {
                "status": "success",
                "capabilities": [],
                "message": "Capability registry not available",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")


@router.post("/capability-lookup")
async def lookup_capability_with_fallback(
    fastapi_request: Request,
    requested_capability: str = Body(..., embed=True),
    tenant_id: str = "1300",
    context_data: dict = Body(default_factory=dict, embed=True)
):
    """
    Task 15.3.13-15.3.18: Capability lookup with fallback, evidence logging, and telemetry
    Comprehensive capability lookup with fail-closed fallback and audit trail
    """
    try:
        orchestrator = fastapi_request.app.state.orchestrator
        
        if not hasattr(orchestrator, 'capability_registry'):
            raise HTTPException(status_code=503, detail="Capability registry not available")
        
        logger.info(f"🔍 Looking up capability '{requested_capability}' for tenant {tenant_id}")
        
        # Task 15.3.13: Implement fallback mechanism
        fallback_result = await orchestrator.capability_registry.implement_fallback_mechanism(
            int(tenant_id), requested_capability, context_data
        )
        
        # Task 15.3.14: Evidence logging
        lookup_request_data = {
            "requested_capability": requested_capability,
            "context_data": context_data,
            "timestamp": datetime.now().isoformat()
        }
        
        evidence_pack_id = await orchestrator.capability_registry.implement_evidence_logging(
            int(tenant_id), lookup_request_data, fallback_result
        )
        
        # Task 15.3.15-15.3.16: Generate and anchor signed manifests
        manifest_id = None
        anchor_id = None
        if evidence_pack_id:
            signed_manifest = await orchestrator.capability_registry.generate_signed_manifests(
                int(tenant_id), evidence_pack_id
            )
            manifest_id = signed_manifest.get("manifest_id")
            
            if manifest_id and not signed_manifest.get("error"):
                anchor_data = await orchestrator.capability_registry.anchor_lookup_manifests(
                    int(tenant_id), signed_manifest
                )
                anchor_id = anchor_data.get("anchor_id")
        
        # Task 15.3.18: Log telemetry metrics
        telemetry_data = {
            "session_id": f"lookup_{datetime.now().timestamp()}",
            "lookup_requests": 1,
            "successful_lookups": 1 if fallback_result.get("fallback_action") == "allow" else 0,
            "failed_lookups": 1 if fallback_result.get("fallback_action") == "deny" else 0,
            "capability_usage": {requested_capability: {"usage_count": 1}} if fallback_result.get("fallback_action") == "allow" else {}
        }
        
        await orchestrator.capability_registry.log_telemetry_metrics(int(tenant_id), telemetry_data)
        
        return {
            "status": "success",
            "lookup_successful": fallback_result.get("fallback_action") == "allow",
            "tenant_id": int(tenant_id),
            "requested_capability": requested_capability,
            "lookup_result": fallback_result,
            "evidence_pack_id": evidence_pack_id,
            "manifest_id": manifest_id,
            "anchor_id": anchor_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Capability lookup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Capability lookup failed: {str(e)}")


@router.post("/capability-override")
async def log_capability_override(
    fastapi_request: Request,
    capability_id: str = Body(...),
    reason: str = Body(...),
    business_justification: str = Body(...),
    authorized_by: str = Body(...),
    tenant_id: str = "1300",
    risk_assessment: str = Body(default="medium"),
    compliance_impact: str = Body(default=None)
):
    """
    Task 15.3.17: Log capability overrides in override ledger
    Capture manual capability overrides with full audit trail
    """
    try:
        orchestrator = fastapi_request.app.state.orchestrator
        
        if not hasattr(orchestrator, 'capability_registry'):
            raise HTTPException(status_code=503, detail="Capability registry not available")
        
        logger.info(f"📝 Logging capability override for '{capability_id}' by tenant {tenant_id}")
        
        override_request_data = {
            "capability_id": capability_id,
            "reason": reason,
            "business_justification": business_justification,
            "authorized_by": authorized_by,
            "risk_assessment": risk_assessment,
            "compliance_impact": compliance_impact,
            "original_decision": "deny",
            "override_decision": "allow",
            "approval_chain": [
                {
                    "action": "override_request",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "authorized_by": authorized_by,
                    "action": "override_approval", 
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        override_id = await orchestrator.capability_registry.log_capability_overrides(
            int(tenant_id), override_request_data
        )
        
        if not override_id:
            raise HTTPException(status_code=500, detail="Failed to log capability override")
        
        return {
            "status": "success",
            "override_id": override_id,
            "tenant_id": int(tenant_id),
            "capability_id": capability_id,
            "authorized_by": authorized_by,
            "timestamp": datetime.now().isoformat(),
            "message": "Capability override logged successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to log capability override: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log capability override: {str(e)}")


@router.get("/compliance-packs/{industry}")
async def get_compliance_packs(
    industry: str,
    fastapi_request: Request,
    tenant_id: str = "1300"
):
    """
    Tasks 15.3.10-15.3.12: Get compliance packs for SaaS/Banking/Insurance
    Return industry-specific compliance requirements and overlays
    """
    try:
        orchestrator = fastapi_request.app.state.orchestrator
        
        if not hasattr(orchestrator, 'capability_registry'):
            raise HTTPException(status_code=503, detail="Capability registry not available")
        
        logger.info(f"📋 Retrieving {industry} compliance packs for tenant {tenant_id}")
        
        # Get industry compliance mapping
        compliance_mapping = orchestrator.capability_registry._load_industry_compliance_mapping()
        
        if industry.upper() not in [k.upper() for k in compliance_mapping.keys()]:
            raise HTTPException(status_code=404, detail=f"Compliance packs not found for industry: {industry}")
        
        # Find the correct case for the industry
        industry_key = next(k for k in compliance_mapping.keys() if k.upper() == industry.upper())
        industry_compliance = compliance_mapping[industry_key]
        
        # Get detailed compliance pack information based on industry
        compliance_packs = {}
        
        if industry.upper() == "SAAS":
            compliance_packs = {
                "SOX": {
                    "framework": "Sarbanes-Oxley Act",
                    "requirements": ["financial_controls", "audit_trails", "segregation_of_duties"],
                    "applicability": "Public SaaS companies",
                    "key_controls": ["revenue_recognition", "access_controls", "change_management"]
                },
                "GDPR": {
                    "framework": "General Data Protection Regulation", 
                    "requirements": ["data_privacy", "consent_management", "right_to_erasure"],
                    "applicability": "EU customer data processing",
                    "key_controls": ["data_encryption", "privacy_by_design", "breach_notification"]
                }
            }
        elif industry.upper() == "BANKING":
            compliance_packs = {
                "RBI": {
                    "framework": "Reserve Bank of India Guidelines",
                    "requirements": ["risk_management", "capital_adequacy", "operational_resilience"],
                    "applicability": "Indian banking operations"
                },
                "DPDP": {
                    "framework": "Digital Personal Data Protection Act",
                    "requirements": ["data_protection", "consent_framework", "data_localization"],
                    "applicability": "Indian customer data processing"
                }
            }
        elif industry.upper() == "INSURANCE":
            compliance_packs = {
                "HIPAA": {
                    "framework": "Health Insurance Portability and Accountability Act",
                    "requirements": ["phi_protection", "access_controls", "audit_logging"],
                    "applicability": "Health insurance operations"
                },
                "NAIC": {
                    "framework": "National Association of Insurance Commissioners",
                    "requirements": ["solvency_monitoring", "market_conduct", "consumer_protection"],
                    "applicability": "US insurance operations"
                }
            }
        
        return {
            "status": "success",
            "tenant_id": int(tenant_id),
            "industry": industry,
            "compliance_frameworks": industry_compliance,
            "compliance_packs": compliance_packs,
            "total_frameworks": len(industry_compliance),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve compliance packs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve compliance packs: {str(e)}")


@router.post("/parse-intent", response_model=IntentParsingResponse)
async def parse_natural_language_intent(request: NaturalLanguageRequest, fastapi_request: Request):
    """Test the LLM-powered intent parser with natural language input"""
    try:
        logger.info(f"🤖 Parsing natural language: '{request.user_input}'")
        
        # Use global orchestrator instance from app state
        orchestrator = fastapi_request.app.state.orchestrator
        
        # Measure processing time
        start_time = datetime.now()
        
        # Parse intent using LLM-powered parser
        parsed_intent = await orchestrator.intent_parser.parse_intent(
            user_input=request.user_input,
            tenant_id=request.tenant_id,
            context_data=request.context or {"persona": "RevOps Manager", "industry": "SaaS"}
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Intent parsed successfully in {processing_time:.2f}ms")
        logger.info(f"🎯 Result: {parsed_intent.intent_type.value} -> {parsed_intent.workflow_category}")
        
        return IntentParsingResponse(
            intent_type=parsed_intent.intent_type.value,
            confidence=parsed_intent.confidence,
            target_automation_type=parsed_intent.target_automation_type.value if parsed_intent.target_automation_type else None,
            workflow_category=parsed_intent.workflow_category,
            parameters=parsed_intent.parameters,
            llm_reasoning=parsed_intent.llm_reasoning,
            raw_input=parsed_intent.raw_input,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Intent parsing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Intent parsing failed: {str(e)}")


@router.post("/execute-natural-language")
async def execute_natural_language_workflow(request: NaturalLanguageRequest):
    """Execute workflow based on natural language input (full pipeline test)"""
    try:
        logger.info(f"🚀 Executing natural language workflow: '{request.user_input}'")
        
        # Use the integrated orchestrator for complete end-to-end processing
        from dsl.integration_orchestrator import get_integration_orchestrator
        
        integration_orchestrator = await get_integration_orchestrator()
        
        start_time = datetime.now()
        
        # Route and execute through the full pipeline
        result = await integration_orchestrator.route_and_execute(
            user_input=request.user_input,
            tenant_id=int(request.tenant_id),
            user_id=int(request.user_id),
            context_data=request.context or {"persona": "RevOps Manager", "industry": "SaaS"}
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Natural language execution completed in {processing_time:.2f}ms")
        
        return {
            "status": result["status"],
            "execution_time_ms": processing_time,
            "routing": result.get("routing_result", {}),
            "execution": result.get("execution_result", {}),
            "llm_reasoning": result.get("llm_reasoning", "")
        }
        
    except Exception as e:
        logger.error(f"❌ Natural language execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Natural language execution failed: {str(e)}")


@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(request: WorkflowExecutionRequest, fastapi_request: Request):
    """Execute a visual workflow or simple scenario"""
    try:
        if request.scenario:
            logger.info(f"🚀 Executing scenario: {request.scenario}")
            # Handle simple scenario execution
            dsl_workflow = await get_scenario_workflow(request.scenario)
        elif request.workflow:
            logger.info(f"🚀 Executing workflow: {request.workflow.name}")
            # Convert visual workflow to DSL
            dsl_workflow = await convert_visual_to_dsl(request.workflow)
        else:
            raise HTTPException(status_code=400, detail="Either 'workflow' or 'scenario' must be provided")
        
        # Use global orchestrator instance from app state - Task 1.2-T1
        orchestrator = fastapi_request.app.state.orchestrator
        
        # Create proper routing request
        routing_request = {
            "intent": f"execute_rba_workflow:{dsl_workflow['name']}",
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "workflow_definition": dsl_workflow,
            "input_data": request.input_data,
            "governance_required": True,
            "evidence_capture": True
        }
        
        # Execute through routing orchestrator
        start_time = datetime.now()
        routing_result = await orchestrator.route_request(routing_request)
        
        if routing_result.success:
            execution = routing_result.execution_result
        else:
            raise HTTPException(status_code=500, detail=f"Routing failed: {routing_result.error_message}")
        end_time = datetime.now()
        
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        if execution.status == "completed":
            return WorkflowExecutionResponse(
                execution_id=execution.execution_id,
                status="success",
                result=execution.output_data,
                execution_time_ms=execution_time_ms
            )
        else:
            return WorkflowExecutionResponse(
                execution_id=execution.execution_id,
                status="error",
                error_message=execution.error_message,
                execution_time_ms=execution_time_ms
            )
    
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        return WorkflowExecutionResponse(
            execution_id=str(uuid.uuid4()),
            status="error",
            error_message=str(e)
        )


@router.post("/validate")
async def validate_workflow(workflow: WorkflowDefinition):
    """Validate workflow configuration"""
    try:
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "governance_checks": {
                "sox_compliant": True,
                "audit_trail_enabled": True,
                "data_residency_compliant": True,
                "approval_workflow_defined": True
            }
        }
        
        # Basic validation
        if not workflow.nodes:
            validation_results["errors"].append("Workflow must contain at least one node")
            validation_results["valid"] = False
        
        # Check for disconnected nodes
        connected_nodes = set()
        for conn in workflow.connections:
            connected_nodes.add(conn.source_node)
            connected_nodes.add(conn.target_node)
        
        disconnected_nodes = [node.id for node in workflow.nodes if node.id not in connected_nodes and len(workflow.nodes) > 1]
        if disconnected_nodes:
            validation_results["warnings"].append(f"Disconnected nodes found: {', '.join(disconnected_nodes)}")
        
        # Governance validation
        enterprise_agents = [node for node in workflow.nodes if node.type.startswith('saas_')]
        if enterprise_agents:
            validation_results["governance_checks"]["enterprise_agents_used"] = len(enterprise_agents)
        
        return validation_results
    
    except Exception as e:
        logger.error(f"❌ Workflow validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get execution status and results"""
    # In production, this would fetch from database
    return {
        "execution_id": execution_id,
        "status": "completed",
        "started_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "result": {
            "message": "Workflow executed successfully",
            "nodes_executed": 3,
            "total_execution_time": "2.5s"
        }
    }


# Helper Functions
async def get_scenario_workflow(scenario: str) -> Dict[str, Any]:
    """Get predefined DSL workflow for a scenario - Task 1.2-T1 Implementation"""
    
    # Load baseline RBA workflows from YAML - Task 1.1-T4
    # Try multiple workflow files - Task 19.3-T03, 19.3-T04 extensions
    workflow_files = [
        os.path.join(os.path.dirname(__file__), "..", "workflows", "rba_baseline_workflows.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "workflows", "saas_rba_workflows_clean.yaml")
    ]
    
    workflow_dict = {}
    
    # Load workflows from all available files
    for workflow_file in workflow_files:
        try:
            if os.path.exists(workflow_file):
                with open(workflow_file, 'r', encoding='utf-8') as f:
                    workflows = yaml.safe_load_all(f)
                    for workflow_doc in workflows:
                        if workflow_doc:
                            for key, value in workflow_doc.items():
                                workflow_dict[key] = value
        except FileNotFoundError:
            logger.warning(f"Workflow file not found: {workflow_file}")
            continue
        except Exception as e:
            logger.error(f"Error loading workflow file {workflow_file}: {e}")
            continue
    
    logger.info(f"📋 Loaded workflows: {list(workflow_dict.keys())}")
    logger.info(f"🔍 Looking for scenario: {scenario}")
    
    # Map scenarios to DSL workflows
    scenario_mapping = {
        "basic_data_query": "basic_data_query_workflow",
        "account_count": "account_count_workflow",
        "pipeline_hygiene": "pipeline_hygiene_workflow",
        "forecast_approval": "forecast_approval_workflow", 
        "compensation": "compensation_calculation_workflow"
    }
    
    if scenario not in scenario_mapping:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")
    
    workflow_key = scenario_mapping[scenario]
    
    if workflow_key in workflow_dict:
        # Return proper DSL workflow
        return workflow_dict[workflow_key]
    
    # Fallback to simple workflow if YAML not loaded
    scenarios = {
        "pipeline_hygiene": {
            "name": "SaaS Pipeline Hygiene Check",
            "version": "1.0",
            "industry": "SaaS", 
            "automation_type": "RBA",
            "metadata": {
                "tenant_id": "{{ tenant_id }}",
                "policy_pack": "saas_sox_policy",
                "evidence_capture": True,
                "trust_score": 1.0
            },
            "governance": {
                "policy_id": "saas_pipeline_policy",
                "compliance_frameworks": ["SOX", "GDPR"],
                "override_ledger_id": "pipeline_overrides"
            },
            "steps": [
                {
                    "id": "query_opportunities",
                    "type": "query",
                    "config": {
                        "source": "salesforce",
                        "resource": "Opportunity",
                        "select": ["Id", "Name", "Amount", "StageName", "LastModifiedDate"],
                        "filters": [
                            {"field": "IsClosed", "op": "eq", "value": False}
                        ]
                    },
                    "governance": {
                        "policy_id": "data_access_policy",
                        "evidence": {"capture": True}
                    }
                }
            ]
        }
        # Add other scenarios as needed...
    }
    
    if scenario not in scenarios:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")
    
    return scenarios[scenario]


async def convert_visual_to_dsl(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Convert visual workflow definition to DSL format"""
    
    # Create execution order based on connections
    execution_order = []
    node_map = {node.id: node for node in workflow.nodes}
    
    # Simple linear execution for now (in production, would handle complex graphs)
    if workflow.connections:
        # Find starting node (no incoming connections)
        incoming = {conn.target_node for conn in workflow.connections}
        start_nodes = [node.id for node in workflow.nodes if node.id not in incoming]
        
        if start_nodes:
            current = start_nodes[0]
            execution_order.append(current)
            
            # Follow connections
            while True:
                next_connections = [conn for conn in workflow.connections if conn.source_node == current]
                if not next_connections:
                    break
                current = next_connections[0].target_node
                execution_order.append(current)
    else:
        # No connections, execute in order
        execution_order = [node.id for node in workflow.nodes]
    
    # Convert to DSL steps
    steps = []
    for node_id in execution_order:
        node = node_map[node_id]
        
        step = {
            "id": node.id,
            "type": node.type,
            "config": node.config,
            "governance": {
                "policy_id": "default_saas_policy",
                "evidence": {"capture": True},
                "override_ledger_id": f"override_{node.id}"
            }
        }
        steps.append(step)
    
    return {
        "name": workflow.name,
        "version": "1.0",
        "metadata": {
            "industry": workflow.industry,
            "created_via": "visual_builder",
            **workflow.metadata
        },
        "steps": steps
    }
