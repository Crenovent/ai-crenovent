"""
Shared Pydantic Models for Workflow Builder APIs
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import numpy as np


def clean_for_json_serialization(obj):
    """Clean object for JSON serialization, handling NaN and inf values"""
    if isinstance(obj, dict):
        return {k: clean_for_json_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json_serialization(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item() if not (np.isnan(obj) or np.isinf(obj)) else None
    else:
        return obj


# Workflow Builder Models
class WorkflowNode(BaseModel):
    id: str
    type: str
    x: float
    y: float
    title: str
    description: str
    config: Dict[str, Any] = {}


class WorkflowConnection(BaseModel):
    id: str
    source_node: str
    target_node: str
    source_port: str = "output"
    target_port: str = "input"


class WorkflowDefinition(BaseModel):
    name: str
    description: Optional[str] = ""
    industry: str = "SaaS"
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection] = []
    metadata: Dict[str, Any] = {}


class WorkflowExecutionRequest(BaseModel):
    workflow: Optional[WorkflowDefinition] = None
    scenario: Optional[str] = None  # For simple scenario-based execution
    tenant_id: str  # Required - no default
    user_id: str    # Required - no default
    execution_mode: str = "sync"  # sync, async
    input_data: Dict[str, Any] = {}


class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class WorkflowTemplate(BaseModel):
    template_id: str
    name: str
    description: str
    industry: str
    category: str
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    governance_level: str = "ENTERPRISE"


# Pipeline Agent Models
class PipelineAgentRequest(BaseModel):
    """Request model for pipeline agent execution"""
    user_input: str
    context: Dict[str, Any] = {}
    tenant_id: str  # Required - no default
    user_id: str    # Required - no default


# Natural Language Processing Models
class NaturalLanguageRequest(BaseModel):
    user_input: str
    tenant_id: str  # Required - no default
    user_id: str    # Required - no default
    context: Optional[Dict[str, Any]] = None


class IntentParsingResponse(BaseModel):
    intent_type: str
    confidence: float
    target_automation_type: Optional[str]
    workflow_category: Optional[str]
    parameters: Dict[str, Any]
    llm_reasoning: Optional[str]
    raw_input: str
    processing_time_ms: float
