"""
Intermediate Representation (IR) for Hybrid Graphs
==================================================

Task 6.2.4: Create Intermediate Representation (IR) for hybrid graphs
- Language-neutral format for both deterministic rules and ML nodes
- Each node has: id, type, inputs, outputs, policies, trust budget, fallbacks
- JSON/protobuf serialization for other services to consume

Dependencies: Task 6.2.2 (Parser AST)
Outputs: IR → input for analyzers & optimizer (6.2.13+)
"""

import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import uuid

# Task 6.2.4: IR Node Types
class IRNodeType(Enum):
    """IR node types for hybrid graphs"""
    # Traditional RBA nodes
    QUERY = "query"
    DECISION = "decision"
    ACTION = "action"
    NOTIFY = "notify"
    GOVERNANCE = "governance"
    AGENT_CALL = "agent_call"
    
    # ML nodes
    ML_NODE = "ml_node"
    ML_PREDICT = "ml_predict"
    ML_SCORE = "ml_score"
    ML_CLASSIFY = "ml_classify"
    ML_EXPLAIN = "ml_explain"

class TrustLevel(Enum):
    """Trust levels for execution gating"""
    HIGH = "high"        # 0.9-1.0
    MEDIUM = "medium"    # 0.7-0.89
    LOW = "low"          # 0.5-0.69
    UNTRUSTED = "untrusted"  # 0.0-0.49

class ExecutionMode(Enum):
    """Execution modes based on trust budget"""
    AUTO_EXECUTE = "auto_execute"      # High trust, automatic execution
    ASSISTED = "assisted"              # Medium trust, human assistance
    MANUAL_REVIEW = "manual_review"    # Low trust, manual review required
    BLOCKED = "blocked"                # Untrusted, execution blocked

# Task 6.2.4: IR Data Structures
@dataclass
class IRInput:
    """IR input specification"""
    name: str
    type: str  # scalar, enum, money, percent, date, vector
    required: bool = True
    source: Optional[str] = None  # Source node id or external
    validation: Optional[Dict[str, Any]] = None
    
    # Task 6.2.10: Residency tags at input level
    region_id: Optional[str] = None
    data_residency_required: bool = False

@dataclass
class IROutput:
    """IR output specification"""
    name: str
    type: str
    target: Optional[str] = None  # Target node id or external
    schema: Optional[Dict[str, Any]] = None
    
    # Task 6.2.10: Residency tags at output level
    region_id: Optional[str] = None
    data_residency_required: bool = False

@dataclass
class IRPolicy:
    """IR policy specification"""
    policy_id: str
    policy_type: str  # residency, threshold, sod, bias, drift
    parameters: Dict[str, Any] = field(default_factory=dict)
    enforcement_action: str = "fail_closed"  # fail_closed, warn, log

@dataclass
class IRTrustBudget:
    """IR trust budget specification - Task 6.2.9 enhanced"""
    min_trust: float  # Minimum trust score to proceed
    auto_execute_above: float  # Trust level for automatic execution
    assisted_mode_below: float  # Trust level requiring assistance
    trust_level: TrustLevel = TrustLevel.MEDIUM
    execution_mode: ExecutionMode = ExecutionMode.ASSISTED
    
    # Task 6.2.9: Enhanced trust budget fields
    trust_min_auto: float = 0.9     # Minimum trust for auto-execution
    trust_min_assisted: float = 0.7  # Minimum trust for assisted mode
    trust_max_block: float = 0.5     # Maximum trust before blocking
    
    # Trust computation metadata
    trust_signals: List[str] = field(default_factory=lambda: ["model_confidence", "data_quality", "override_rate"])
    signal_weights: Dict[str, float] = field(default_factory=lambda: {"model_confidence": 0.6, "data_quality": 0.3, "override_rate": 0.1})
    normalization_method: str = "weighted_average"
    
    def compute_trust_score(self, signals: Dict[str, float]) -> float:
        """Compute trust score from input signals"""
        if not signals or not self.signal_weights:
            return self.min_trust
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for signal_name, weight in self.signal_weights.items():
            if signal_name in signals:
                weighted_sum += signals[signal_name] * weight
                total_weight += weight
        
        if total_weight == 0:
            return self.min_trust
        
        return weighted_sum / total_weight
    
    def get_execution_mode_for_trust(self, trust_score: float) -> ExecutionMode:
        """Get execution mode based on trust score"""
        if trust_score >= self.trust_min_auto:
            return ExecutionMode.AUTO_EXECUTE
        elif trust_score >= self.trust_min_assisted:
            return ExecutionMode.ASSISTED
        elif trust_score >= self.trust_max_block:
            return ExecutionMode.MANUAL_REVIEW
        else:
            return ExecutionMode.BLOCKED

@dataclass
class IRFallbackItem:
    """IR fallback item specification"""
    fallback_id: str
    type: str  # rba_rule, default_action, human_escalation, previous_step
    target_node_id: str
    condition: Optional[str] = None
    priority: int = 1  # Lower number = higher priority

@dataclass
class IRFallback:
    """IR fallback configuration"""
    enabled: bool = True
    fallback_items: List[IRFallbackItem] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    max_fallback_depth: int = 3

@dataclass
class IRNode:
    """
    IR Node - Uniform representation for both deterministic rules and ML nodes
    
    Each node contains:
    - id: unique identifier
    - type: node type (traditional or ML)
    - inputs: input specifications
    - outputs: output specifications  
    - policies: governance policies
    - trust_budget: trust and execution gating
    - fallbacks: fallback strategies
    """
    id: str
    type: IRNodeType
    inputs: List[IRInput] = field(default_factory=list)
    outputs: List[IROutput] = field(default_factory=list)
    policies: List[IRPolicy] = field(default_factory=list)
    trust_budget: Optional[IRTrustBudget] = None
    fallbacks: Optional[IRFallback] = None
    
    # Node-specific configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Graph connectivity
    dependencies: List[str] = field(default_factory=list)  # Input node ids
    dependents: List[str] = field(default_factory=list)    # Output node ids
    
    # Execution context
    region_id: Optional[str] = None
    sla_budget_ms: Optional[int] = None
    cost_class: Optional[str] = None  # T0, T1, T2
    
    # Task 6.2.10: Enhanced residency context
    data_residency_constraints: Dict[str, str] = field(default_factory=dict)  # data_type -> required_region
    external_call_regions: Dict[str, str] = field(default_factory=dict)  # call_id -> region
    
    def __post_init__(self):
        """Validate IR node after creation"""
        # Ensure trust budget for ML nodes
        if self.type in [IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                        IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN]:
            if not self.trust_budget:
                self.trust_budget = IRTrustBudget(
                    min_trust=0.7,
                    auto_execute_above=0.85,
                    assisted_mode_below=0.6
                )
            
            # Ensure fallback for ML nodes
            if not self.fallbacks:
                self.fallbacks = IRFallback(enabled=True)

@dataclass
class IRGraph:
    """IR Graph - Complete hybrid workflow representation"""
    workflow_id: str
    name: str
    version: str
    automation_type: str  # RBA, RBIA, AALA
    
    # Graph structure
    nodes: List[IRNode] = field(default_factory=list)
    edges: List[Dict[str, str]] = field(default_factory=list)  # from_node_id -> to_node_id
    
    # Task 6.2.10: Edge-level residency tracking
    edge_residency: Dict[str, Dict[str, str]] = field(default_factory=dict)  # edge_key -> {from_region, to_region, data_types}
    
    # Workflow-level configuration
    policy_pack: Optional[str] = None
    governance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Compilation metadata
    compiled_at: datetime = field(default_factory=datetime.utcnow)
    compiler_version: str = "6.2.4"
    ir_version: str = "1.0.0"
    
    def get_node(self, node_id: str) -> Optional[IRNode]:
        """Get node by ID"""
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def add_edge(self, from_node_id: str, to_node_id: str):
        """Add edge between nodes"""
        edge = {"from": from_node_id, "to": to_node_id}
        if edge not in self.edges:
            self.edges.append(edge)
            
            # Update node dependencies
            from_node = self.get_node(from_node_id)
            to_node = self.get_node(to_node_id)
            
            if from_node and to_node_id not in from_node.dependents:
                from_node.dependents.append(to_node_id)
                
            if to_node and from_node_id not in to_node.dependencies:
                to_node.dependencies.append(from_node_id)
    
    def get_ml_nodes(self) -> List[IRNode]:
        """Get all ML nodes in the graph"""
        ml_types = {IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                   IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN}
        return [node for node in self.nodes if node.type in ml_types]
    
    def get_traditional_nodes(self) -> List[IRNode]:
        """Get all traditional RBA nodes in the graph"""
        traditional_types = {IRNodeType.QUERY, IRNodeType.DECISION, IRNodeType.ACTION,
                           IRNodeType.NOTIFY, IRNodeType.GOVERNANCE, IRNodeType.AGENT_CALL}
        return [node for node in self.nodes if node.type in traditional_types]
    
    def validate(self) -> List[str]:
        """Validate IR graph structure"""
        errors = []
        
        # Check for duplicate node IDs
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs found")
        
        # Check edge references
        for edge in self.edges:
            if not self.get_node(edge["from"]):
                errors.append(f"Edge references non-existent from node: {edge['from']}")
            if not self.get_node(edge["to"]):
                errors.append(f"Edge references non-existent to node: {edge['to']}")
        
        # Check ML node requirements
        for node in self.get_ml_nodes():
            if not node.trust_budget:
                errors.append(f"ML node {node.id} missing trust budget")
            if not node.fallbacks or not node.fallbacks.enabled:
                errors.append(f"ML node {node.id} missing fallback configuration")
        
        return errors

# Task 6.2.4: JSON Serialization
class IRSerializer:
    """Serializer for IR graphs to JSON/protobuf"""
    
    @staticmethod
    def to_json(ir_graph: IRGraph) -> str:
        """Serialize IR graph to JSON"""
        # Convert dataclasses to dict with enum handling
        def convert_value(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return asdict(obj)
            return obj
        
        # Convert IR graph to serializable dict
        graph_dict = asdict(ir_graph)
        
        # Handle enums in nodes
        for node_dict in graph_dict.get('nodes', []):
            if 'type' in node_dict:
                node_dict['type'] = node_dict['type']  # Already converted by asdict
            
            # Handle trust budget enums
            if 'trust_budget' in node_dict and node_dict['trust_budget']:
                tb = node_dict['trust_budget']
                if 'trust_level' in tb:
                    tb['trust_level'] = tb['trust_level']
                if 'execution_mode' in tb:
                    tb['execution_mode'] = tb['execution_mode']
        
        return json.dumps(graph_dict, indent=2, default=convert_value)
    
    @staticmethod
    def from_json(json_str: str) -> IRGraph:
        """Deserialize IR graph from JSON"""
        data = json.loads(json_str)
        
        # Convert back to dataclasses
        nodes = []
        for node_data in data.get('nodes', []):
            # Convert node type
            node_data['type'] = IRNodeType(node_data['type'])
            
            # Convert inputs
            inputs = []
            for input_data in node_data.get('inputs', []):
                inputs.append(IRInput(**input_data))
            node_data['inputs'] = inputs
            
            # Convert outputs
            outputs = []
            for output_data in node_data.get('outputs', []):
                outputs.append(IROutput(**output_data))
            node_data['outputs'] = outputs
            
            # Convert policies
            policies = []
            for policy_data in node_data.get('policies', []):
                policies.append(IRPolicy(**policy_data))
            node_data['policies'] = policies
            
            # Convert trust budget
            if node_data.get('trust_budget'):
                tb_data = node_data['trust_budget']
                if 'trust_level' in tb_data:
                    tb_data['trust_level'] = TrustLevel(tb_data['trust_level'])
                if 'execution_mode' in tb_data:
                    tb_data['execution_mode'] = ExecutionMode(tb_data['execution_mode'])
                node_data['trust_budget'] = IRTrustBudget(**tb_data)
            
            # Convert fallbacks
            if node_data.get('fallbacks'):
                fb_data = node_data['fallbacks']
                fallback_items = []
                for item_data in fb_data.get('fallback_items', []):
                    fallback_items.append(IRFallbackItem(**item_data))
                fb_data['fallback_items'] = fallback_items
                node_data['fallbacks'] = IRFallback(**fb_data)
            
            nodes.append(IRNode(**node_data))
        
        data['nodes'] = nodes
        
        # Convert datetime
        if 'compiled_at' in data:
            data['compiled_at'] = datetime.fromisoformat(data['compiled_at'])
        
        return IRGraph(**data)
    
    @staticmethod
    def to_protobuf_schema() -> str:
        """Generate protobuf schema for IR graph"""
        return '''
syntax = "proto3";

package rbia.ir;

// IR Node Types
enum IRNodeType {
  QUERY = 0;
  DECISION = 1;
  ACTION = 2;
  NOTIFY = 3;
  GOVERNANCE = 4;
  AGENT_CALL = 5;
  ML_NODE = 6;
  ML_PREDICT = 7;
  ML_SCORE = 8;
  ML_CLASSIFY = 9;
  ML_EXPLAIN = 10;
}

// Trust and Execution
enum TrustLevel {
  HIGH = 0;
  MEDIUM = 1;
  LOW = 2;
  UNTRUSTED = 3;
}

enum ExecutionMode {
  AUTO_EXECUTE = 0;
  ASSISTED = 1;
  MANUAL_REVIEW = 2;
  BLOCKED = 3;
}

// IR Messages
message IRInput {
  string name = 1;
  string type = 2;
  bool required = 3;
  string source = 4;
  map<string, string> validation = 5;
}

message IROutput {
  string name = 1;
  string type = 2;
  string target = 3;
  map<string, string> schema = 4;
}

message IRPolicy {
  string policy_id = 1;
  string policy_type = 2;
  map<string, string> parameters = 3;
  string enforcement_action = 4;
}

message IRTrustBudget {
  float min_trust = 1;
  float auto_execute_above = 2;
  float assisted_mode_below = 3;
  TrustLevel trust_level = 4;
  ExecutionMode execution_mode = 5;
}

message IRFallbackItem {
  string fallback_id = 1;
  string type = 2;
  string target_node_id = 3;
  string condition = 4;
  int32 priority = 5;
}

message IRFallback {
  bool enabled = 1;
  repeated IRFallbackItem fallback_items = 2;
  repeated string trigger_conditions = 3;
  int32 max_fallback_depth = 4;
}

message IRNode {
  string id = 1;
  IRNodeType type = 2;
  repeated IRInput inputs = 3;
  repeated IROutput outputs = 4;
  repeated IRPolicy policies = 5;
  IRTrustBudget trust_budget = 6;
  IRFallback fallbacks = 7;
  map<string, string> parameters = 8;
  map<string, string> metadata = 9;
  repeated string dependencies = 10;
  repeated string dependents = 11;
  string region_id = 12;
  int32 sla_budget_ms = 13;
  string cost_class = 14;
}

message IREdge {
  string from = 1;
  string to = 2;
}

message IRGraph {
  string workflow_id = 1;
  string name = 2;
  string version = 3;
  string automation_type = 4;
  repeated IRNode nodes = 5;
  repeated IREdge edges = 6;
  string policy_pack = 7;
  map<string, string> governance = 8;
  map<string, string> metadata = 9;
  string compiled_at = 10;
  string compiler_version = 11;
  string ir_version = 12;
}
'''

# Task 6.2.4: AST to IR Converter
class ASTToIRConverter:
    """Convert AST from parser to IR for hybrid graphs"""
    
    def __init__(self):
        self.node_counter = 0
    
    def convert(self, workflow_ast) -> IRGraph:
        """Convert WorkflowAST to IRGraph"""
        from .parser import WorkflowAST, StepAST  # Import here to avoid circular imports
        
        ir_graph = IRGraph(
            workflow_id=workflow_ast.workflow_id,
            name=workflow_ast.name,
            version=getattr(workflow_ast, 'version', '1.0.0'),
            automation_type=workflow_ast.automation_type,
            policy_pack=workflow_ast.policy_pack,
            governance=workflow_ast.governance or {},
            metadata=workflow_ast.metadata
        )
        
        # Convert steps to IR nodes
        for step_ast in workflow_ast.steps:
            ir_node = self._convert_step_to_ir_node(step_ast)
            ir_graph.nodes.append(ir_node)
        
        # Build edges from next_steps relationships
        for step_ast in workflow_ast.steps:
            for next_step_id in getattr(step_ast, 'next_steps', []):
                ir_graph.add_edge(step_ast.id, next_step_id)
        
        return ir_graph
    
    def _convert_step_to_ir_node(self, step_ast) -> IRNode:
        """Convert StepAST to IRNode"""
        # Map step type
        ir_node_type = IRNodeType(step_ast.type.value)
        
        # Create IR node
        ir_node = IRNode(
            id=step_ast.id,
            type=ir_node_type,
            parameters=step_ast.params,
            metadata={"line": step_ast.line, "column": step_ast.column}
        )
        
        # Convert inputs from parameters
        ir_node.inputs = self._extract_inputs_from_params(step_ast.params)
        
        # Convert outputs
        ir_node.outputs = self._extract_outputs_from_step(step_ast)
        
        # Convert policies from governance
        ir_node.policies = self._extract_policies_from_governance(step_ast.governance)
        
        # Convert trust budget for ML nodes
        if ir_node_type in [IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                           IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN]:
            ir_node.trust_budget = self._extract_trust_budget_from_params(step_ast.params)
        
        # Convert fallbacks
        if hasattr(step_ast, 'fallback') and step_ast.fallback:
            ir_node.fallbacks = self._convert_fallback_ast_to_ir(step_ast.fallback)
        
        return ir_node
    
    def _extract_inputs_from_params(self, params: Dict[str, Any]) -> List[IRInput]:
        """Extract IR inputs from step parameters"""
        inputs = []
        
        # For ML nodes, extract model inputs
        if 'input_data' in params:
            inputs.append(IRInput(
                name="input_data",
                type="object",
                required=True,
                source="previous_step"
            ))
        
        # Extract other parameter-based inputs
        for key, value in params.items():
            if key.endswith('_input') or key in ['data', 'query', 'condition']:
                inputs.append(IRInput(
                    name=key,
                    type=self._infer_type(value),
                    required=True
                ))
        
        return inputs
    
    def _extract_outputs_from_step(self, step_ast) -> List[IROutput]:
        """Extract IR outputs from step"""
        outputs = []
        
        # Extract from outputs field
        if hasattr(step_ast, 'outputs'):
            for key, value in step_ast.outputs.items():
                outputs.append(IROutput(
                    name=key,
                    type="object",
                    target=value
                ))
        
        # Default outputs based on step type
        if step_ast.type.value in ['ml_predict', 'ml_score', 'ml_classify']:
            outputs.append(IROutput(name="prediction", type="object"))
            outputs.append(IROutput(name="confidence", type="float"))
        elif step_ast.type.value == 'ml_explain':
            outputs.append(IROutput(name="explanation", type="object"))
        
        return outputs
    
    def _extract_policies_from_governance(self, governance: Dict[str, Any]) -> List[IRPolicy]:
        """Extract IR policies from governance configuration"""
        policies = []
        
        if 'policy_id' in governance:
            policies.append(IRPolicy(
                policy_id=governance['policy_id'],
                policy_type="governance",
                parameters=governance
            ))
        
        # Extract specific policy types
        for policy_type in ['residency', 'threshold', 'sod', 'bias', 'drift']:
            if policy_type in governance:
                policies.append(IRPolicy(
                    policy_id=f"{policy_type}_policy",
                    policy_type=policy_type,
                    parameters=governance[policy_type]
                ))
        
        return policies
    
    def _extract_trust_budget_from_params(self, params: Dict[str, Any]) -> IRTrustBudget:
        """Extract trust budget from ML node parameters"""
        confidence_config = params.get('confidence', {})
        
        return IRTrustBudget(
            min_trust=confidence_config.get('min_confidence', 0.7),
            auto_execute_above=confidence_config.get('auto_execute_above', 0.85),
            assisted_mode_below=confidence_config.get('assisted_mode_below', 0.6),
            trust_level=TrustLevel.MEDIUM,
            execution_mode=ExecutionMode.ASSISTED
        )
    
    def _convert_fallback_ast_to_ir(self, fallback_ast) -> IRFallback:
        """Convert FallbackAST to IRFallback"""
        fallback_items = []
        
        for item_ast in fallback_ast.fallback_list:
            fallback_items.append(IRFallbackItem(
                fallback_id=f"fallback_{len(fallback_items)}",
                type=item_ast.type,
                target_node_id=item_ast.target,
                condition=item_ast.condition,
                priority=len(fallback_items) + 1
            ))
        
        return IRFallback(
            enabled=fallback_ast.enabled,
            fallback_items=fallback_items,
            trigger_conditions=fallback_ast.trigger_conditions
        )
    
    def _infer_type(self, value: Any) -> str:
        """Infer IR type from value"""
        if isinstance(value, str):
            return "string"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"
