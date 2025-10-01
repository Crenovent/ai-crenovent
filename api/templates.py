"""
Workflow Templates API
FastAPI endpoints for managing workflow templates and agents
"""

from fastapi import APIRouter
from typing import Dict, List, Any
from datetime import datetime

from .models import WorkflowTemplate, WorkflowNode, WorkflowConnection

# Create templates router
router = APIRouter(prefix="/api/builder", tags=["Templates"])


@router.get("/agents")
async def get_available_agents():
    """Get list of available workflow agents"""
    agents = {
        "saas_agents": [
            {
                "id": "saas_pipeline_hygiene",
                "name": "Pipeline Hygiene Agent",
                "description": "Automated pipeline health monitoring and stale deal detection",
                "icon": "fas fa-chart-line",
                "color": "#3b82f6",
                "category": "Analytics",
                "governance_level": "ENTERPRISE",
                "config_schema": {
                    "stale_threshold_days": {"type": "integer", "default": 15, "min": 1, "max": 90},
                    "minimum_hygiene_score": {"type": "integer", "default": 70, "min": 0, "max": 100},
                    "critical_threshold_days": {"type": "integer", "default": 30, "min": 1, "max": 180}
                },
                "outputs": ["hygiene_score", "stale_opportunities", "recommendations"]
            },
            {
                "id": "saas_forecast_approval",
                "name": "Forecast Approval Agent",
                "description": "Automated forecast validation and approval routing",
                "icon": "fas fa-chart-bar",
                "color": "#10b981",
                "category": "Approval",
                "governance_level": "ENTERPRISE",
                "config_schema": {
                    "variance_threshold": {"type": "number", "default": 15.0, "min": 0, "max": 100},
                    "confidence_threshold": {"type": "number", "default": 75.0, "min": 0, "max": 100},
                    "auto_approve_threshold": {"type": "number", "default": 5.0, "min": 0, "max": 25}
                },
                "outputs": ["approval_status", "variance_analysis", "evidence_pack"]
            },
            {
                "id": "saas_compensation",
                "name": "Compensation Calculator Agent",
                "description": "Automated commission and quota calculations",
                "icon": "fas fa-dollar-sign",
                "color": "#f59e0b",
                "category": "Finance",
                "governance_level": "ENTERPRISE",
                "config_schema": {
                    "calculation_type": {"type": "string", "enum": ["individual", "team", "bulk"], "default": "individual"},
                    "period": {"type": "string", "pattern": "\\d{4}-\\d{2}", "default": datetime.now().strftime("%Y-%m")},
                    "include_spiffs": {"type": "boolean", "default": True}
                },
                "outputs": ["total_compensation", "breakdown", "compliance_report"]
            }
        ],
        "core_operators": [
            {
                "id": "query",
                "name": "Query Operator",
                "description": "Fetch data from databases, APIs, or files",
                "icon": "fas fa-database",
                "color": "#8b5cf6",
                "category": "Data",
                "config_schema": {
                    "source": {"type": "string", "enum": ["salesforce", "postgres", "api"], "default": "salesforce"},
                    "resource": {"type": "string", "default": "Opportunity"},
                    "filters": {"type": "array", "default": []}
                },
                "outputs": ["records", "count"]
            },
            {
                "id": "decision",
                "name": "Decision Operator",
                "description": "Conditional logic and branching",
                "icon": "fas fa-code-branch",
                "color": "#f97316",
                "category": "Logic",
                "config_schema": {
                    "condition": {"type": "string", "default": ""},
                    "true_path": {"type": "string", "default": ""},
                    "false_path": {"type": "string", "default": ""}
                },
                "outputs": ["result", "path_taken"]
            },
            {
                "id": "notify",
                "name": "Notification Operator",
                "description": "Send notifications via email, Slack, or webhook",
                "icon": "fas fa-bell",
                "color": "#ef4444",
                "category": "Communication",
                "config_schema": {
                    "channel": {"type": "string", "enum": ["email", "slack", "webhook"], "default": "email"},
                    "recipients": {"type": "array", "default": []},
                    "message": {"type": "string", "default": ""}
                },
                "outputs": ["sent", "delivery_status"]
            }
        ]
    }
    return agents


@router.get("/templates")
async def get_workflow_templates():
    """Get pre-built workflow templates"""
    templates = [
        WorkflowTemplate(
            template_id="saas_pipeline_management",
            name="SaaS Pipeline Management",
            description="Complete pipeline health monitoring and management workflow",
            industry="SaaS",
            category="Sales Operations",
            nodes=[
                WorkflowNode(
                    id="hygiene_check",
                    type="saas_pipeline_hygiene",
                    x=100,
                    y=100,
                    title="Pipeline Hygiene Check",
                    description="Monitor pipeline health",
                    config={"stale_threshold_days": 15, "minimum_hygiene_score": 70}
                ),
                WorkflowNode(
                    id="forecast_validation",
                    type="saas_forecast_approval",
                    x=400,
                    y=100,
                    title="Forecast Validation",
                    description="Validate forecast submissions",
                    config={"variance_threshold": 15.0}
                ),
                WorkflowNode(
                    id="comp_calculation",
                    type="saas_compensation",
                    x=700,
                    y=100,
                    title="Commission Calculation",
                    description="Calculate sales commissions",
                    config={"calculation_type": "team"}
                )
            ],
            connections=[
                WorkflowConnection(
                    id="conn_1",
                    source_node="hygiene_check",
                    target_node="forecast_validation"
                ),
                WorkflowConnection(
                    id="conn_2",
                    source_node="forecast_validation",
                    target_node="comp_calculation"
                )
            ]
        ),
        WorkflowTemplate(
            template_id="saas_monthly_close",
            name="SaaS Monthly Close Process",
            description="Automated monthly sales close and reporting workflow",
            industry="SaaS",
            category="Finance",
            nodes=[
                WorkflowNode(
                    id="data_sync",
                    type="query",
                    x=100,
                    y=150,
                    title="Sync Sales Data",
                    description="Fetch latest sales data",
                    config={"source": "salesforce", "resource": "Opportunity"}
                ),
                WorkflowNode(
                    id="compensation_calc",
                    type="saas_compensation",
                    x=400,
                    y=150,
                    title="Calculate Commissions",
                    description="Monthly commission calculation",
                    config={"calculation_type": "bulk"}
                ),
                WorkflowNode(
                    id="close_notification",
                    type="notify",
                    x=700,
                    y=150,
                    title="Close Notification",
                    description="Notify stakeholders",
                    config={"channel": "email", "recipients": ["finance@company.com"]}
                )
            ],
            connections=[
                WorkflowConnection(id="conn_1", source_node="data_sync", target_node="compensation_calc"),
                WorkflowConnection(id="conn_2", source_node="compensation_calc", target_node="close_notification")
            ]
        )
    ]
    
    return {"templates": [template.dict() for template in templates]}
