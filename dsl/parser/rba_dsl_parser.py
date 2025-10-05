"""
RBA DSL Parser - Task 6.2-T02
Translates DSL workflows into Abstract Syntax Tree (AST) for compilation and execution.
Implements governance-first parsing with mandatory policy fields.
"""

import json
import yaml
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timezone

class NodeType(Enum):
    """DSL Node Types"""
    WORKFLOW = "workflow"
    QUERY = "query"
    DECISION = "decision"
    ML_DECISION = "ml_decision"
    AGENT_CALL = "agent_call"
    NOTIFY = "notify"
    GOVERNANCE = "governance"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"

class PolicyLevel(Enum):
    """Policy enforcement levels"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"

@dataclass
class GovernanceMetadata:
    """Mandatory governance metadata for all DSL nodes"""
    policy_id: str
    evidence_capture: bool = True
    override_ledger_id: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)
    tenant_id: str = ""
    region: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class DSLNode:
    """Base DSL Node with governance-first design"""
    id: str
    type: NodeType
    name: str
    governance: GovernanceMetadata
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    next_nodes: List[str] = field(default_factory=list)
    error_handlers: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate governance requirements"""
        if not self.governance.policy_id:
            raise ValueError(f"Node {self.id}: policy_id is mandatory for governance")
        if not self.governance.tenant_id:
            raise ValueError(f"Node {self.id}: tenant_id is mandatory for multi-tenancy")

@dataclass
class WorkflowAST:
    """Complete workflow Abstract Syntax Tree"""
    id: str
    name: str
    version: str
    description: str
    governance: GovernanceMetadata
    nodes: Dict[str, DSLNode] = field(default_factory=dict)
    start_node: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    plan_hash: str = ""
    
    def __post_init__(self):
        """Generate immutable plan hash for audit trail"""
        self.plan_hash = self._generate_plan_hash()
    
    def _generate_plan_hash(self) -> str:
        """Generate SHA256 hash of workflow for audit trail"""
        workflow_data = {
            "id": self.id,
            "version": self.version,
            "nodes": {node_id: {
                "type": node.type.value,
                "params": node.params,
                "next_nodes": node.next_nodes
            } for node_id, node in self.nodes.items()},
            "start_node": self.start_node
        }
        workflow_json = json.dumps(workflow_data, sort_keys=True)
        return hashlib.sha256(workflow_json.encode()).hexdigest()

class RBADSLParser:
    """
    RBA DSL Parser implementing Task 6.2-T02
    Converts YAML/JSON DSL definitions into structured AST
    """
    
    def __init__(self):
        self.supported_node_types = {
            "query": self._parse_query_node,
            "decision": self._parse_decision_node,
            "ml_decision": self._parse_ml_decision_node,
            "agent_call": self._parse_agent_call_node,
            "notify": self._parse_notify_node,
            "governance": self._parse_governance_node
        }
        
    def parse_workflow(self, dsl_content: Union[str, Dict]) -> WorkflowAST:
        """
        Parse DSL workflow into AST
        
        Args:
            dsl_content: YAML string or parsed dictionary
            
        Returns:
            WorkflowAST: Structured workflow representation
            
        Raises:
            ValueError: If DSL is invalid or missing required fields
        """
        try:
            # Parse YAML/JSON if string
            if isinstance(dsl_content, str):
                try:
                    workflow_data = yaml.safe_load(dsl_content)
                except yaml.YAMLError as e:
                    try:
                        workflow_data = json.loads(dsl_content)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid DSL format: {e}")
            else:
                workflow_data = dsl_content
            
            # Validate required workflow fields
            self._validate_workflow_structure(workflow_data)
            
            # Parse governance metadata
            governance = self._parse_governance_metadata(workflow_data.get("governance", {}))
            
            # Create workflow AST
            workflow_ast = WorkflowAST(
                id=workflow_data["id"],
                name=workflow_data["name"],
                version=workflow_data.get("version", "1.0.0"),
                description=workflow_data.get("description", ""),
                governance=governance,
                start_node=workflow_data.get("start_node", ""),
                metadata=workflow_data.get("metadata", {})
            )
            
            # Parse all nodes
            for node_data in workflow_data.get("nodes", []):
                node = self._parse_node(node_data, governance.tenant_id)
                workflow_ast.nodes[node.id] = node
            
            # Validate workflow integrity
            self._validate_workflow_integrity(workflow_ast)
            
            return workflow_ast
            
        except Exception as e:
            raise ValueError(f"DSL parsing failed: {str(e)}")
    
    def _validate_workflow_structure(self, workflow_data: Dict) -> None:
        """Validate basic workflow structure"""
        required_fields = ["id", "name", "nodes"]
        for field in required_fields:
            if field not in workflow_data:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(workflow_data["nodes"], list):
            raise ValueError("Nodes must be a list")
        
        if len(workflow_data["nodes"]) == 0:
            raise ValueError("Workflow must contain at least one node")
    
    def _parse_governance_metadata(self, governance_data: Dict) -> GovernanceMetadata:
        """Parse governance metadata with validation"""
        if not governance_data.get("policy_id"):
            raise ValueError("Governance policy_id is mandatory")
        
        if not governance_data.get("tenant_id"):
            raise ValueError("Governance tenant_id is mandatory")
        
        return GovernanceMetadata(
            policy_id=governance_data["policy_id"],
            evidence_capture=governance_data.get("evidence_capture", True),
            override_ledger_id=governance_data.get("override_ledger_id"),
            compliance_tags=governance_data.get("compliance_tags", []),
            tenant_id=governance_data["tenant_id"],
            region=governance_data.get("region", ""),
            created_by=governance_data.get("created_by", "system"),
            created_at=datetime.now(timezone.utc)
        )
    
    def _parse_node(self, node_data: Dict, tenant_id: str) -> DSLNode:
        """Parse individual DSL node"""
        # Validate required node fields
        required_fields = ["id", "type"]
        for field in required_fields:
            if field not in node_data:
                raise ValueError(f"Node missing required field: {field}")
        
        node_type = node_data["type"]
        if node_type not in self.supported_node_types:
            raise ValueError(f"Unsupported node type: {node_type}")
        
        # Parse node governance (inherits from workflow if not specified)
        node_governance_data = node_data.get("governance", {})
        if not node_governance_data.get("tenant_id"):
            node_governance_data["tenant_id"] = tenant_id
        if not node_governance_data.get("policy_id"):
            raise ValueError(f"Node {node_data['id']}: policy_id is mandatory")
        
        node_governance = self._parse_governance_metadata(node_governance_data)
        
        # Create base node
        node = DSLNode(
            id=node_data["id"],
            type=NodeType(node_type),
            name=node_data.get("name", node_data["id"]),
            governance=node_governance,
            params=node_data.get("params", {}),
            inputs=node_data.get("inputs", []),
            outputs=node_data.get("outputs", []),
            next_nodes=node_data.get("next_nodes", []),
            error_handlers=node_data.get("error_handlers", [])
        )
        
        # Type-specific validation
        self.supported_node_types[node_type](node, node_data)
        
        return node
    
    def _parse_query_node(self, node: DSLNode, node_data: Dict) -> None:
        """Validate query node parameters"""
        required_params = ["source", "query"]
        for param in required_params:
            if param not in node.params:
                raise ValueError(f"Query node {node.id}: missing required param '{param}'")
        
        # Validate query structure
        if not isinstance(node.params.get("filters"), (dict, type(None))):
            raise ValueError(f"Query node {node.id}: filters must be a dictionary")
    
    def _parse_decision_node(self, node: DSLNode, node_data: Dict) -> None:
        """Validate decision node parameters"""
        required_params = ["conditions"]
        for param in required_params:
            if param not in node.params:
                raise ValueError(f"Decision node {node.id}: missing required param '{param}'")
        
        # Validate conditions structure
        conditions = node.params["conditions"]
        if not isinstance(conditions, list):
            raise ValueError(f"Decision node {node.id}: conditions must be a list")
        
        for i, condition in enumerate(conditions):
            if not all(key in condition for key in ["condition", "next_node"]):
                raise ValueError(f"Decision node {node.id}: condition {i} missing required fields")
    
    def _parse_ml_decision_node(self, node: DSLNode, node_data: Dict) -> None:
        """Validate ML decision node parameters"""
        required_params = ["model_id", "confidence_threshold"]
        for param in required_params:
            if param not in node.params:
                raise ValueError(f"ML decision node {node.id}: missing required param '{param}'")
        
        # Validate confidence threshold
        threshold = node.params["confidence_threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"ML decision node {node.id}: confidence_threshold must be between 0 and 1")
    
    def _parse_agent_call_node(self, node: DSLNode, node_data: Dict) -> None:
        """Validate agent call node parameters"""
        required_params = ["agent_id", "task"]
        for param in required_params:
            if param not in node.params:
                raise ValueError(f"Agent call node {node.id}: missing required param '{param}'")
    
    def _parse_notify_node(self, node: DSLNode, node_data: Dict) -> None:
        """Validate notify node parameters"""
        required_params = ["channel", "message"]
        for param in required_params:
            if param not in node.params:
                raise ValueError(f"Notify node {node.id}: missing required param '{param}'")
        
        # Validate channel
        valid_channels = ["email", "slack", "webhook", "sms"]
        if node.params["channel"] not in valid_channels:
            raise ValueError(f"Notify node {node.id}: invalid channel '{node.params['channel']}'")
    
    def _parse_governance_node(self, node: DSLNode, node_data: Dict) -> None:
        """Validate governance node parameters"""
        required_params = ["action"]
        for param in required_params:
            if param not in node.params:
                raise ValueError(f"Governance node {node.id}: missing required param '{param}'")
        
        # Validate governance action
        valid_actions = ["evidence_capture", "policy_check", "override_request", "audit_log"]
        if node.params["action"] not in valid_actions:
            raise ValueError(f"Governance node {node.id}: invalid action '{node.params['action']}'")
    
    def _validate_workflow_integrity(self, workflow_ast: WorkflowAST) -> None:
        """Validate workflow integrity and connectivity"""
        if not workflow_ast.start_node:
            # Auto-detect start node if not specified
            potential_starts = []
            for node_id, node in workflow_ast.nodes.items():
                is_referenced = any(node_id in other_node.next_nodes 
                                 for other_node in workflow_ast.nodes.values())
                if not is_referenced:
                    potential_starts.append(node_id)
            
            if len(potential_starts) == 1:
                workflow_ast.start_node = potential_starts[0]
            else:
                raise ValueError("Cannot determine unique start node")
        
        # Validate start node exists
        if workflow_ast.start_node not in workflow_ast.nodes:
            raise ValueError(f"Start node '{workflow_ast.start_node}' not found in workflow")
        
        # Validate all referenced nodes exist
        for node_id, node in workflow_ast.nodes.items():
            for next_node in node.next_nodes:
                if next_node not in workflow_ast.nodes:
                    raise ValueError(f"Node {node_id} references non-existent node: {next_node}")
            
            for error_handler in node.error_handlers:
                if error_handler not in workflow_ast.nodes:
                    raise ValueError(f"Node {node_id} references non-existent error handler: {error_handler}")
        
        # Check for unreachable nodes
        reachable = set()
        self._mark_reachable_nodes(workflow_ast.start_node, workflow_ast.nodes, reachable)
        
        unreachable = set(workflow_ast.nodes.keys()) - reachable
        if unreachable:
            raise ValueError(f"Unreachable nodes detected: {unreachable}")
    
    def _mark_reachable_nodes(self, node_id: str, nodes: Dict[str, DSLNode], reachable: set) -> None:
        """Recursively mark reachable nodes"""
        if node_id in reachable:
            return
        
        reachable.add(node_id)
        node = nodes[node_id]
        
        for next_node in node.next_nodes:
            self._mark_reachable_nodes(next_node, nodes, reachable)
        
        for error_handler in node.error_handlers:
            self._mark_reachable_nodes(error_handler, nodes, reachable)

def create_sample_workflow() -> str:
    """Create a sample RBA workflow for testing"""
    sample_workflow = {
        "id": "pipeline_hygiene_workflow",
        "name": "Pipeline Hygiene Automation",
        "version": "1.0.0",
        "description": "Automated pipeline hygiene workflow for SaaS companies",
        "governance": {
            "policy_id": "saas_pipeline_policy_v1",
            "tenant_id": "tenant_1300",
            "region": "us-east-1",
            "compliance_tags": ["SOX", "GDPR"],
            "created_by": "system"
        },
        "start_node": "fetch_stale_opportunities",
        "nodes": [
            {
                "id": "fetch_stale_opportunities",
                "type": "query",
                "name": "Fetch Stale Opportunities",
                "governance": {
                    "policy_id": "data_access_policy_v1",
                    "tenant_id": "tenant_1300"
                },
                "params": {
                    "source": "salesforce",
                    "query": "SELECT Id, Name, StageName, LastModifiedDate FROM Opportunity WHERE LastModifiedDate < LAST_N_DAYS:15 AND IsClosed = false",
                    "filters": {
                        "stage": ["Prospecting", "Qualification", "Proposal"]
                    }
                },
                "outputs": ["stale_opportunities"],
                "next_nodes": ["check_opportunity_count"]
            },
            {
                "id": "check_opportunity_count",
                "type": "decision",
                "name": "Check Opportunity Count",
                "governance": {
                    "policy_id": "business_logic_policy_v1",
                    "tenant_id": "tenant_1300"
                },
                "params": {
                    "conditions": [
                        {
                            "condition": "len(stale_opportunities) > 0",
                            "next_node": "notify_sales_manager"
                        },
                        {
                            "condition": "len(stale_opportunities) == 0",
                            "next_node": "log_no_action_needed"
                        }
                    ]
                },
                "inputs": ["stale_opportunities"],
                "next_nodes": ["notify_sales_manager", "log_no_action_needed"]
            },
            {
                "id": "notify_sales_manager",
                "type": "notify",
                "name": "Notify Sales Manager",
                "governance": {
                    "policy_id": "notification_policy_v1",
                    "tenant_id": "tenant_1300"
                },
                "params": {
                    "channel": "slack",
                    "message": "Found {count} stale opportunities requiring attention",
                    "recipients": ["sales_manager"],
                    "template_vars": {
                        "count": "len(stale_opportunities)"
                    }
                },
                "inputs": ["stale_opportunities"],
                "next_nodes": ["generate_evidence"]
            },
            {
                "id": "log_no_action_needed",
                "type": "governance",
                "name": "Log No Action Needed",
                "governance": {
                    "policy_id": "audit_policy_v1",
                    "tenant_id": "tenant_1300"
                },
                "params": {
                    "action": "audit_log",
                    "message": "Pipeline hygiene check completed - no stale opportunities found"
                },
                "next_nodes": ["generate_evidence"]
            },
            {
                "id": "generate_evidence",
                "type": "governance",
                "name": "Generate Evidence Pack",
                "governance": {
                    "policy_id": "evidence_policy_v1",
                    "tenant_id": "tenant_1300"
                },
                "params": {
                    "action": "evidence_capture",
                    "evidence_type": "workflow_execution",
                    "include_data": True
                }
            }
        ]
    }
    
    return yaml.dump(sample_workflow, default_flow_style=False)

# Example usage and testing
if __name__ == "__main__":
    # Test the parser
    parser = RBADSLParser()
    
    # Create and parse sample workflow
    sample_dsl = create_sample_workflow()
    print("Sample DSL Workflow:")
    print(sample_dsl)
    print("\n" + "="*50 + "\n")
    
    try:
        # Parse the workflow
        workflow_ast = parser.parse_workflow(sample_dsl)
        
        print("Parsed Workflow AST:")
        print(f"ID: {workflow_ast.id}")
        print(f"Name: {workflow_ast.name}")
        print(f"Version: {workflow_ast.version}")
        print(f"Plan Hash: {workflow_ast.plan_hash}")
        print(f"Start Node: {workflow_ast.start_node}")
        print(f"Node Count: {len(workflow_ast.nodes)}")
        print(f"Tenant ID: {workflow_ast.governance.tenant_id}")
        print(f"Policy ID: {workflow_ast.governance.policy_id}")
        
        print("\nNodes:")
        for node_id, node in workflow_ast.nodes.items():
            print(f"  {node_id}: {node.type.value} -> {node.next_nodes}")
        
        print("\nParsing completed successfully!")
        
    except Exception as e:
        print(f"Parsing failed: {e}")
