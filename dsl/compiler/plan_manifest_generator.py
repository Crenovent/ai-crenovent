"""
Plan Manifest Generator - Task 6.2.26
======================================

Plan manifest generator (JSON) with canonical sort order
- Produces deterministic, canonical JSON manifest for every compiled plan
- Includes: plan graph, IR node metadata, trust budgets, fallback DAG, explainability hooks, provenance fields
- Implements canonical serialization strategy (strict key ordering, normalized number formatting, stable UUID ordering)
- Emits both human-friendly pretty-printed JSON and canonical compact JSON for hashing
- Unit tests asserting deterministic output for identical IR inputs

Dependencies: Task 6.2.4 (IR)
Outputs: Deterministic plan manifest → enables stable hashing, signing, and diffs
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import uuid
import logging
from datetime import datetime
from decimal import Decimal
import hashlib
import re

logger = logging.getLogger(__name__)

@dataclass
class PlanProvenance:
    """Provenance information for plan manifest"""
    source_refs: List[str] = field(default_factory=list)
    commit_shas: List[str] = field(default_factory=list)
    author: Optional[Dict[str, Any]] = None
    approver: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    compiler_version: str = "1.0.0"
    ir_version: str = "1.0.0"
    build_environment: Optional[Dict[str, str]] = None

@dataclass
class EvidenceCheckpoint:
    """Evidence checkpoint pointer"""
    checkpoint_id: str
    event_type: str  # "node_enter", "node_exit", "decision_made", "override_applied"
    node_id: str
    timestamp: datetime
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlanManifest:
    """Complete plan manifest with all metadata"""
    manifest_version: str = "1.0.0"
    plan_id: str = ""
    workflow_id: str = ""
    name: str = ""
    version: str = ""
    automation_type: str = ""
    
    # Core plan structure
    plan_graph: Dict[str, Any] = field(default_factory=dict)
    ir_metadata: Dict[str, Any] = field(default_factory=dict)
    trust_budgets: Dict[str, Any] = field(default_factory=dict)
    fallback_dag: Dict[str, Any] = field(default_factory=dict)
    explainability_hooks: Dict[str, Any] = field(default_factory=dict)
    
    # Governance and compliance
    policy_pack: Optional[str] = None
    governance: Dict[str, Any] = field(default_factory=dict)
    compliance_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance and audit
    provenance: Optional[PlanProvenance] = None
    evidence_checkpoints: List[EvidenceCheckpoint] = field(default_factory=list)
    
    # Manifest metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    canonical_hash: Optional[str] = None
    digital_signature: Optional[str] = None

class ManifestFormat(Enum):
    """Manifest output formats"""
    CANONICAL_COMPACT = "canonical_compact"
    HUMAN_READABLE = "human_readable"
    BOTH = "both"

# Task 6.2.26: Plan Manifest Generator
class PlanManifestGenerator:
    """Generates deterministic, canonical JSON manifests for compiled plans"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Canonical ordering configuration
        self.canonical_config = {
            'sort_keys': True,
            'separators': (',', ':'),
            'ensure_ascii': True,
            'indent': None,
            'number_precision': 6,  # For float normalization
            'datetime_format': '%Y-%m-%dT%H:%M:%S.%fZ'
        }
        
        # Human-readable formatting
        self.pretty_config = {
            'sort_keys': True,
            'separators': (', ', ': '),
            'ensure_ascii': False,
            'indent': 2
        }
    
    def generate_manifest(self, ir_graph, 
                         provenance: Optional[PlanProvenance] = None,
                         evidence_checkpoints: Optional[List[EvidenceCheckpoint]] = None,
                         format_type: ManifestFormat = ManifestFormat.BOTH) -> Union[str, Tuple[str, str]]:
        """Generate plan manifest from IR graph"""
        from .ir import IRGraph
        
        # Create manifest structure
        manifest = self._create_manifest_from_ir(ir_graph, provenance, evidence_checkpoints)
        
        # Generate canonical manifest
        canonical_manifest = self._generate_canonical_manifest(manifest)
        
        if format_type == ManifestFormat.CANONICAL_COMPACT:
            return canonical_manifest
        
        # Generate human-readable manifest
        pretty_manifest = self._generate_pretty_manifest(manifest)
        
        if format_type == ManifestFormat.HUMAN_READABLE:
            return pretty_manifest
        
        # Return both formats
        return canonical_manifest, pretty_manifest
    
    def _create_manifest_from_ir(self, ir_graph, 
                                provenance: Optional[PlanProvenance],
                                evidence_checkpoints: Optional[List[EvidenceCheckpoint]]) -> PlanManifest:
        """Create manifest structure from IR graph"""
        # Generate unique plan ID if not present
        plan_id = ir_graph.metadata.get('plan_id', str(uuid.uuid4()))
        
        # Extract plan graph structure
        plan_graph = self._extract_plan_graph(ir_graph)
        
        # Extract IR metadata
        ir_metadata = self._extract_ir_metadata(ir_graph)
        
        # Extract trust budgets
        trust_budgets = self._extract_trust_budgets(ir_graph)
        
        # Extract fallback DAG
        fallback_dag = self._extract_fallback_dag(ir_graph)
        
        # Extract explainability hooks
        explainability_hooks = self._extract_explainability_hooks(ir_graph)
        
        # Extract compliance metadata
        compliance_metadata = self._extract_compliance_metadata(ir_graph)
        
        # Create manifest
        manifest = PlanManifest(
            plan_id=plan_id,
            workflow_id=ir_graph.workflow_id,
            name=ir_graph.name,
            version=ir_graph.version,
            automation_type=ir_graph.automation_type,
            plan_graph=plan_graph,
            ir_metadata=ir_metadata,
            trust_budgets=trust_budgets,
            fallback_dag=fallback_dag,
            explainability_hooks=explainability_hooks,
            policy_pack=ir_graph.policy_pack,
            governance=ir_graph.governance or {},
            compliance_metadata=compliance_metadata,
            provenance=provenance,
            evidence_checkpoints=evidence_checkpoints or []
        )
        
        return manifest
    
    def _extract_plan_graph(self, ir_graph) -> Dict[str, Any]:
        """Extract plan graph structure (nodes and edges)"""
        nodes = []
        
        for node in ir_graph.nodes:
            node_data = {
                'id': node.id,
                'type': node.type.value if hasattr(node.type, 'value') else str(node.type),
                'inputs': [self._serialize_input(inp) for inp in node.inputs],
                'outputs': [self._serialize_output(out) for out in node.outputs],
                'dependencies': sorted(node.dependencies),  # Ensure stable ordering
                'dependents': sorted(node.dependents),
                'region_id': node.region_id,
                'sla_budget_ms': node.sla_budget_ms,
                'cost_class': node.cost_class,
                'parameters': self._canonicalize_dict(node.parameters),
                'metadata': self._canonicalize_dict(node.metadata)
            }
            
            # Add residency constraints if present
            if hasattr(node, 'data_residency_constraints') and node.data_residency_constraints:
                node_data['data_residency_constraints'] = self._canonicalize_dict(node.data_residency_constraints)
            
            if hasattr(node, 'external_call_regions') and node.external_call_regions:
                node_data['external_call_regions'] = self._canonicalize_dict(node.external_call_regions)
            
            nodes.append(node_data)
        
        # Sort nodes by ID for deterministic ordering
        nodes.sort(key=lambda x: x['id'])
        
        # Extract edges with stable ordering
        edges = []
        for edge in ir_graph.edges:
            edges.append({
                'from': edge.get('from', ''),
                'to': edge.get('to', '')
            })
        
        # Sort edges for deterministic ordering
        edges.sort(key=lambda x: (x['from'], x['to']))
        
        # Add edge residency if present
        edge_residency = {}
        if hasattr(ir_graph, 'edge_residency') and ir_graph.edge_residency:
            edge_residency = self._canonicalize_dict(ir_graph.edge_residency)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'edge_residency': edge_residency
        }
    
    def _extract_ir_metadata(self, ir_graph) -> Dict[str, Any]:
        """Extract IR-level metadata"""
        return {
            'compiled_at': ir_graph.compiled_at.isoformat() if ir_graph.compiled_at else None,
            'compiler_version': ir_graph.compiler_version,
            'ir_version': ir_graph.ir_version,
            'metadata': self._canonicalize_dict(ir_graph.metadata)
        }
    
    def _extract_trust_budgets(self, ir_graph) -> Dict[str, Any]:
        """Extract trust budget information from all nodes"""
        trust_budgets = {}
        
        for node in ir_graph.nodes:
            if node.trust_budget:
                trust_budgets[node.id] = {
                    'min_trust': self._normalize_number(node.trust_budget.min_trust),
                    'auto_execute_above': self._normalize_number(node.trust_budget.auto_execute_above),
                    'assisted_mode_below': self._normalize_number(node.trust_budget.assisted_mode_below),
                    'trust_level': node.trust_budget.trust_level.value if hasattr(node.trust_budget.trust_level, 'value') else str(node.trust_budget.trust_level),
                    'execution_mode': node.trust_budget.execution_mode.value if hasattr(node.trust_budget.execution_mode, 'value') else str(node.trust_budget.execution_mode)
                }
                
                # Add enhanced trust budget fields if present
                if hasattr(node.trust_budget, 'trust_min_auto'):
                    trust_budgets[node.id].update({
                        'trust_min_auto': self._normalize_number(node.trust_budget.trust_min_auto),
                        'trust_min_assisted': self._normalize_number(node.trust_budget.trust_min_assisted),
                        'trust_max_block': self._normalize_number(node.trust_budget.trust_max_block),
                        'trust_signals': sorted(node.trust_budget.trust_signals),
                        'signal_weights': self._canonicalize_dict(node.trust_budget.signal_weights),
                        'normalization_method': node.trust_budget.normalization_method
                    })
        
        return trust_budgets
    
    def _extract_fallback_dag(self, ir_graph) -> Dict[str, Any]:
        """Extract fallback DAG structure"""
        fallback_dag = {
            'nodes': {},
            'edges': [],
            'coverage_metrics': {}
        }
        
        for node in ir_graph.nodes:
            if node.fallbacks and node.fallbacks.enabled:
                fallback_items = []
                
                for item in node.fallbacks.fallback_items:
                    fallback_items.append({
                        'fallback_id': item.fallback_id,
                        'type': item.type,
                        'target_node_id': item.target_node_id,
                        'condition': item.condition,
                        'priority': item.priority
                    })
                
                # Sort fallback items by priority for deterministic ordering
                fallback_items.sort(key=lambda x: (x['priority'], x['fallback_id']))
                
                fallback_dag['nodes'][node.id] = {
                    'enabled': node.fallbacks.enabled,
                    'fallback_items': fallback_items,
                    'trigger_conditions': sorted(node.fallbacks.trigger_conditions),
                    'max_fallback_depth': node.fallbacks.max_fallback_depth
                }
                
                # Create fallback edges
                for item in node.fallbacks.fallback_items:
                    if item.target_node_id:
                        fallback_dag['edges'].append({
                            'from': node.id,
                            'to': item.target_node_id,
                            'type': 'fallback',
                            'condition': item.condition,
                            'priority': item.priority
                        })
        
        # Sort fallback edges for deterministic ordering
        fallback_dag['edges'].sort(key=lambda x: (x['from'], x['to'], x['priority']))
        
        return fallback_dag
    
    def _extract_explainability_hooks(self, ir_graph) -> Dict[str, Any]:
        """Extract explainability hooks from ML nodes"""
        explainability_hooks = {}
        
        for node in ir_graph.nodes:
            # Check if node is ML type and has explainability configuration
            if self._is_ml_node(node):
                explainability_config = node.parameters.get('explainability', {})
                if explainability_config or node.metadata.get('explainability_enabled', False):
                    explainability_hooks[node.id] = {
                        'enabled': explainability_config.get('enabled', True),
                        'method': explainability_config.get('method', 'shap'),
                        'params': self._canonicalize_dict(explainability_config.get('params', {})),
                        'output_format': explainability_config.get('output_format', 'json'),
                        'feature_importance': explainability_config.get('feature_importance', True),
                        'audit_integration': explainability_config.get('audit_integration', True)
                    }
        
        return explainability_hooks
    
    def _extract_compliance_metadata(self, ir_graph) -> Dict[str, Any]:
        """Extract compliance-related metadata"""
        compliance_metadata = {
            'policy_violations': [],
            'residency_compliance': {},
            'trust_compliance': {},
            'fallback_compliance': {},
            'audit_requirements': []
        }
        
        # Extract policy violations from validation results
        if 'validation_results' in ir_graph.metadata:
            validation_results = ir_graph.metadata['validation_results']
            if isinstance(validation_results, dict):
                compliance_metadata['policy_violations'] = validation_results.get('violations', [])
        
        # Extract residency compliance info
        regions_used = set()
        for node in ir_graph.nodes:
            if node.region_id:
                regions_used.add(node.region_id)
        
        compliance_metadata['residency_compliance'] = {
            'regions_used': sorted(list(regions_used)),
            'cross_region_calls': len([node for node in ir_graph.nodes if node.external_call_regions]),
            'data_residency_required': any(
                hasattr(node, 'data_residency_constraints') and node.data_residency_constraints 
                for node in ir_graph.nodes
            )
        }
        
        # Extract trust compliance info
        ml_nodes_count = len([node for node in ir_graph.nodes if self._is_ml_node(node)])
        nodes_with_trust_budget = len([node for node in ir_graph.nodes if node.trust_budget])
        
        compliance_metadata['trust_compliance'] = {
            'ml_nodes_total': ml_nodes_count,
            'nodes_with_trust_budget': nodes_with_trust_budget,
            'trust_coverage_percentage': (nodes_with_trust_budget / ml_nodes_count * 100) if ml_nodes_count > 0 else 100
        }
        
        # Extract fallback compliance info
        nodes_with_fallbacks = len([node for node in ir_graph.nodes if node.fallbacks and node.fallbacks.enabled])
        
        compliance_metadata['fallback_compliance'] = {
            'ml_nodes_total': ml_nodes_count,
            'nodes_with_fallbacks': nodes_with_fallbacks,
            'fallback_coverage_percentage': (nodes_with_fallbacks / ml_nodes_count * 100) if ml_nodes_count > 0 else 100
        }
        
        return compliance_metadata
    
    def _serialize_input(self, input_spec) -> Dict[str, Any]:
        """Serialize IR input specification"""
        result = {
            'name': input_spec.name,
            'type': input_spec.type,
            'required': input_spec.required,
            'source': input_spec.source
        }
        
        if hasattr(input_spec, 'validation') and input_spec.validation:
            result['validation'] = self._canonicalize_dict(input_spec.validation)
        
        # Add residency fields if present
        if hasattr(input_spec, 'region_id') and input_spec.region_id:
            result['region_id'] = input_spec.region_id
        
        if hasattr(input_spec, 'data_residency_required'):
            result['data_residency_required'] = input_spec.data_residency_required
        
        return result
    
    def _serialize_output(self, output_spec) -> Dict[str, Any]:
        """Serialize IR output specification"""
        result = {
            'name': output_spec.name,
            'type': output_spec.type,
            'target': output_spec.target
        }
        
        if hasattr(output_spec, 'schema') and output_spec.schema:
            result['schema'] = self._canonicalize_dict(output_spec.schema)
        
        # Add residency fields if present
        if hasattr(output_spec, 'region_id') and output_spec.region_id:
            result['region_id'] = output_spec.region_id
        
        if hasattr(output_spec, 'data_residency_required'):
            result['data_residency_required'] = output_spec.data_residency_required
        
        return result
    
    def _is_ml_node(self, node) -> bool:
        """Check if node is an ML node"""
        from .ir import IRNodeType
        if hasattr(IRNodeType, 'ML_NODE'):
            ml_types = {IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                       IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN}
            return node.type in ml_types
        return False
    
    def _canonicalize_dict(self, obj: Any) -> Any:
        """Canonicalize dictionary or other object for deterministic serialization"""
        if obj is None:
            return None
        elif isinstance(obj, dict):
            # Sort keys and recursively canonicalize values
            return {k: self._canonicalize_dict(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            # Canonicalize list elements
            return [self._canonicalize_dict(item) for item in obj]
        elif isinstance(obj, tuple):
            # Convert tuple to list for JSON serialization
            return [self._canonicalize_dict(item) for item in obj]
        elif isinstance(obj, set):
            # Convert set to sorted list
            return sorted([self._canonicalize_dict(item) for item in obj])
        elif isinstance(obj, (int, float)):
            return self._normalize_number(obj)
        elif isinstance(obj, datetime):
            return obj.strftime(self.canonical_config['datetime_format'])
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            # Handle dataclass or other objects with __dict__
            return self._canonicalize_dict(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__)
        else:
            return obj
    
    def _normalize_number(self, num: Union[int, float]) -> Union[int, float]:
        """Normalize number for consistent representation"""
        if isinstance(num, int):
            return num
        elif isinstance(num, float):
            # Round to specified precision to avoid floating point inconsistencies
            precision = self.canonical_config['number_precision']
            return round(num, precision)
        else:
            return num
    
    def _generate_canonical_manifest(self, manifest: PlanManifest) -> str:
        """Generate canonical compact JSON manifest"""
        # Convert manifest to dict
        manifest_dict = self._canonicalize_dict(manifest)
        
        # Generate canonical JSON
        canonical_json = json.dumps(
            manifest_dict,
            sort_keys=self.canonical_config['sort_keys'],
            separators=self.canonical_config['separators'],
            ensure_ascii=self.canonical_config['ensure_ascii'],
            indent=self.canonical_config['indent']
        )
        
        return canonical_json
    
    def _generate_pretty_manifest(self, manifest: PlanManifest) -> str:
        """Generate human-readable JSON manifest"""
        # Convert manifest to dict
        manifest_dict = self._canonicalize_dict(manifest)
        
        # Generate pretty JSON
        pretty_json = json.dumps(
            manifest_dict,
            sort_keys=self.pretty_config['sort_keys'],
            separators=self.pretty_config['separators'],
            ensure_ascii=self.pretty_config['ensure_ascii'],
            indent=self.pretty_config['indent']
        )
        
        return pretty_json
    
    def compute_manifest_hash(self, canonical_manifest: str) -> str:
        """Compute SHA-256 hash of canonical manifest"""
        return hashlib.sha256(canonical_manifest.encode('utf-8')).hexdigest()
    
    def verify_deterministic_generation(self, ir_graph, iterations: int = 3) -> bool:
        """Verify that manifest generation is deterministic"""
        manifests = []
        
        for _ in range(iterations):
            canonical_manifest, _ = self.generate_manifest(ir_graph, format_type=ManifestFormat.BOTH)
            manifests.append(canonical_manifest)
        
        # All manifests should be identical
        return all(manifest == manifests[0] for manifest in manifests)
    
    def create_manifest_diff(self, manifest1: str, manifest2: str) -> Dict[str, Any]:
        """Create a diff between two manifests"""
        try:
            dict1 = json.loads(manifest1)
            dict2 = json.loads(manifest2)
            
            diff = {
                'added': {},
                'removed': {},
                'modified': {},
                'unchanged_keys': []
            }
            
            # Simple diff implementation
            all_keys = set(dict1.keys()) | set(dict2.keys())
            
            for key in all_keys:
                if key in dict1 and key in dict2:
                    if dict1[key] != dict2[key]:
                        diff['modified'][key] = {
                            'old': dict1[key],
                            'new': dict2[key]
                        }
                    else:
                        diff['unchanged_keys'].append(key)
                elif key in dict1:
                    diff['removed'][key] = dict1[key]
                else:
                    diff['added'][key] = dict2[key]
            
            return diff
            
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {str(e)}'}

# Test helper functions
def create_test_ir_graph_for_manifest():
    """Create test IR graph for manifest generation testing"""
    from .ir import IRGraph, IRNode, IRNodeType, IRInput, IROutput, IRTrustBudget, IRFallback, IRFallbackItem, TrustLevel, ExecutionMode
    
    # Create test nodes
    nodes = [
        IRNode(
            id="query_node",
            type=IRNodeType.QUERY,
            inputs=[IRInput(name="customer_id", type="string", required=True)],
            outputs=[IROutput(name="customer_data", type="object")],
            parameters={"query": "SELECT * FROM customers WHERE id = ?"},
            metadata={"description": "Fetch customer data"},
            region_id="us-east-1",
            sla_budget_ms=1000,
            cost_class="T1"
        ),
        IRNode(
            id="ml_predict_node",
            type=IRNodeType.ML_PREDICT,
            inputs=[IRInput(name="customer_data", type="object", required=True)],
            outputs=[IROutput(name="prediction", type="float"), IROutput(name="confidence", type="float")],
            parameters={
                "model_id": "churn_predictor_v2",
                "explainability": {
                    "enabled": True,
                    "method": "shap",
                    "params": {"n_samples": 100}
                }
            },
            trust_budget=IRTrustBudget(
                min_trust=0.7,
                auto_execute_above=0.85,
                assisted_mode_below=0.6,
                trust_level=TrustLevel.MEDIUM,
                execution_mode=ExecutionMode.ASSISTED
            ),
            fallbacks=IRFallback(
                enabled=True,
                fallback_items=[
                    IRFallbackItem(
                        fallback_id="rba_fallback",
                        type="rba_rule",
                        target_node_id="decision_node",
                        condition="confidence < 0.7",
                        priority=1
                    )
                ],
                trigger_conditions=["low_confidence", "model_error"],
                max_fallback_depth=2
            ),
            region_id="us-east-1"
        ),
        IRNode(
            id="decision_node",
            type=IRNodeType.DECISION,
            inputs=[IRInput(name="prediction", type="float", required=True)],
            outputs=[IROutput(name="action", type="string")],
            parameters={"rules": [{"condition": "prediction > 0.8", "action": "retain"}]},
            region_id="us-east-1"
        )
    ]
    
    # Create IR graph
    ir_graph = IRGraph(
        workflow_id="test_manifest_workflow",
        name="Test Manifest Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=nodes,
        edges=[
            {"from": "query_node", "to": "ml_predict_node"},
            {"from": "ml_predict_node", "to": "decision_node"}
        ],
        policy_pack="saas_ml_governance_v1",
        governance={"compliance_level": "strict", "audit_required": True},
        metadata={"created_by": "test_user", "environment": "test"}
    )
    
    return ir_graph

def create_test_provenance() -> PlanProvenance:
    """Create test provenance information"""
    return PlanProvenance(
        source_refs=["workflows/churn_prediction.yaml", "models/churn_predictor_v2.pkl"],
        commit_shas=["abc123def456", "789ghi012jkl"],
        author={
            "user_id": "john.doe",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "role": "ml_engineer"
        },
        approver={
            "user_id": "jane.smith",
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "role": "engineering_director"
        },
        build_environment={
            "ci_system": "github_actions",
            "build_id": "run_123456",
            "environment": "production"
        }
    )

def create_test_evidence_checkpoints() -> List[EvidenceCheckpoint]:
    """Create test evidence checkpoints"""
    return [
        EvidenceCheckpoint(
            checkpoint_id="cp_001",
            event_type="node_enter",
            node_id="ml_predict_node",
            timestamp=datetime.utcnow(),
            evidence_refs=["evidence://audit/node_enter_001"],
            metadata={"input_hash": "sha256:abc123"}
        ),
        EvidenceCheckpoint(
            checkpoint_id="cp_002",
            event_type="decision_made",
            node_id="ml_predict_node",
            timestamp=datetime.utcnow(),
            evidence_refs=["evidence://audit/decision_002", "evidence://explainability/shap_002"],
            metadata={"confidence": 0.85, "prediction": 0.75}
        )
    ]

def run_manifest_generation_tests():
    """Run comprehensive manifest generation tests"""
    generator = PlanManifestGenerator()
    
    # Create test data
    ir_graph = create_test_ir_graph_for_manifest()
    provenance = create_test_provenance()
    evidence_checkpoints = create_test_evidence_checkpoints()
    
    print("=== Plan Manifest Generation Tests ===")
    
    # Test 1: Basic manifest generation
    print("\n1. Testing basic manifest generation...")
    canonical_manifest, pretty_manifest = generator.generate_manifest(
        ir_graph, provenance, evidence_checkpoints, ManifestFormat.BOTH
    )
    
    print(f"   Canonical manifest size: {len(canonical_manifest)} bytes")
    print(f"   Pretty manifest size: {len(pretty_manifest)} bytes")
    
    # Test 2: Deterministic generation
    print("\n2. Testing deterministic generation...")
    is_deterministic = generator.verify_deterministic_generation(ir_graph, iterations=5)
    print(f"   Deterministic generation: {'✅ PASS' if is_deterministic else '❌ FAIL'}")
    
    # Test 3: Manifest hash computation
    print("\n3. Testing manifest hash computation...")
    manifest_hash = generator.compute_manifest_hash(canonical_manifest)
    print(f"   Manifest hash: {manifest_hash[:16]}...")
    
    # Test 4: Hash consistency
    print("\n4. Testing hash consistency...")
    canonical_manifest2, _ = generator.generate_manifest(ir_graph, provenance, evidence_checkpoints, ManifestFormat.BOTH)
    manifest_hash2 = generator.compute_manifest_hash(canonical_manifest2)
    hash_consistent = manifest_hash == manifest_hash2
    print(f"   Hash consistency: {'✅ PASS' if hash_consistent else '❌ FAIL'}")
    
    # Test 5: Manifest structure validation
    print("\n5. Testing manifest structure...")
    try:
        manifest_data = json.loads(canonical_manifest)
        required_sections = ['plan_graph', 'ir_metadata', 'trust_budgets', 'fallback_dag', 'explainability_hooks']
        all_sections_present = all(section in manifest_data for section in required_sections)
        print(f"   Structure validation: {'✅ PASS' if all_sections_present else '❌ FAIL'}")
        
        # Print manifest sections summary
        print(f"   Plan graph nodes: {len(manifest_data['plan_graph']['nodes'])}")
        print(f"   Trust budgets: {len(manifest_data['trust_budgets'])}")
        print(f"   Fallback DAG nodes: {len(manifest_data['fallback_dag']['nodes'])}")
        print(f"   Explainability hooks: {len(manifest_data['explainability_hooks'])}")
        print(f"   Evidence checkpoints: {len(manifest_data['evidence_checkpoints'])}")
        
    except json.JSONDecodeError as e:
        print(f"   Structure validation: ❌ FAIL - Invalid JSON: {e}")
    
    # Test 6: Canonical ordering verification
    print("\n6. Testing canonical ordering...")
    # Parse and re-serialize to check if ordering is preserved
    try:
        parsed_manifest = json.loads(canonical_manifest)
        re_serialized = json.dumps(parsed_manifest, sort_keys=True, separators=(',', ':'))
        ordering_preserved = canonical_manifest == re_serialized
        print(f"   Canonical ordering: {'✅ PASS' if ordering_preserved else '❌ FAIL'}")
    except Exception as e:
        print(f"   Canonical ordering: ❌ FAIL - {e}")
    
    print(f"\n=== Test Summary ===")
    print(f"Generated canonical manifest: {len(canonical_manifest)} bytes")
    print(f"Generated pretty manifest: {len(pretty_manifest)} bytes")
    print(f"Manifest hash: {manifest_hash}")
    
    return canonical_manifest, pretty_manifest, manifest_hash

if __name__ == "__main__":
    run_manifest_generation_tests()
