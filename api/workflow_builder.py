"""
Workflow Builder Core API
FastAPI endpoints for drag-and-drop workflow creation and execution
"""

from fastapi import APIRouter, HTTPException, Request
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
