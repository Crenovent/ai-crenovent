"""
Residency Tags Propagation - Task 6.2.10
==========================================

Implement residency tags (region_id) propagation
- Extend IR with residency attributes at node/edge/external call level
- Residency propagation logic through the workflow graph
- Validation of residency constraints at compile time

Dependencies: Task 6.2.4 (IR - enhanced with residency fields)
Outputs: Residency-validated IR → ensures data sovereignty compliance
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Task 6.2.10: Residency Propagation Engine
class ResidencyPropagationEngine:
    """Propagates residency tags through IR graph and validates constraints"""
    
    def __init__(self):
        self.propagation_errors: List[str] = []
        self.residency_violations: List[Dict[str, Any]] = []
    
    def propagate_residency_tags(self, ir_graph) -> bool:
        """Propagate residency tags through entire IR graph"""
        self.propagation_errors.clear()
        self.residency_violations.clear()
        
        # Step 1: Propagate node-level residency
        if not self._propagate_node_residency(ir_graph):
            return False
        
        # Step 2: Propagate input/output residency
        if not self._propagate_io_residency(ir_graph):
            return False
        
        # Step 3: Propagate edge-level residency
        if not self._propagate_edge_residency(ir_graph):
            return False
        
        # Step 4: Validate external call residency
        if not self._validate_external_call_residency(ir_graph):
            return False
        
        # Step 5: Check for residency violations
        if not self._validate_residency_constraints(ir_graph):
            return False
        
        return len(self.propagation_errors) == 0
    
    def _propagate_node_residency(self, ir_graph) -> bool:
        """Propagate residency tags at node level"""
        # Get workflow-level default region
        workflow_region = ir_graph.governance.get('region_id')
        
        for node in ir_graph.nodes:
            # If node doesn't have region, inherit from workflow
            if not node.region_id and workflow_region:
                node.region_id = workflow_region
                logger.debug(f"Node {node.id}: inherited region {workflow_region} from workflow")
            
            # Apply policy-based residency constraints
            self._apply_policy_residency_constraints(node)
            
            # Validate ML node model residency
            if self._is_ml_node(node):
                if not self._validate_ml_node_residency(node):
                    return False
        
        return True
    
    def _propagate_io_residency(self, ir_graph) -> bool:
        """Propagate residency tags to inputs and outputs"""
        for node in ir_graph.nodes:
            node_region = node.region_id
            
            # Propagate to inputs
            for input_spec in node.inputs:
                if not input_spec.region_id and node_region:
                    input_spec.region_id = node_region
                
                # Check data residency requirements
                if input_spec.data_residency_required and not input_spec.region_id:
                    self.propagation_errors.append(
                        f"Node {node.id}: input '{input_spec.name}' requires data residency but no region specified"
                    )
                    return False
            
            # Propagate to outputs
            for output_spec in node.outputs:
                if not output_spec.region_id and node_region:
                    output_spec.region_id = node_region
                
                # Check data residency requirements
                if output_spec.data_residency_required and not output_spec.region_id:
                    self.propagation_errors.append(
                        f"Node {node.id}: output '{output_spec.name}' requires data residency but no region specified"
                    )
                    return False
        
        return True
    
    def _propagate_edge_residency(self, ir_graph) -> bool:
        """Propagate residency tags to edges"""
        for edge in ir_graph.edges:
            from_node = ir_graph.get_node(edge["from"])
            to_node = ir_graph.get_node(edge["to"])
            
            if not from_node or not to_node:
                continue
            
            edge_key = f"{edge['from']}->{edge['to']}"
            
            # Track residency information for this edge
            edge_residency = {
                "from_region": from_node.region_id,
                "to_region": to_node.region_id,
                "data_types": []
            }
            
            # Determine data types flowing through edge
            for output in from_node.outputs:
                for input_spec in to_node.inputs:
                    if input_spec.source == from_node.id:
                        edge_residency["data_types"].append({
                            "field_name": output.name,
                            "data_type": output.type,
                            "from_region": output.region_id,
                            "to_region": input_spec.region_id
                        })
            
            ir_graph.edge_residency[edge_key] = edge_residency
        
        return True
    
    def _validate_external_call_residency(self, ir_graph) -> bool:
        """Validate external call residency constraints"""
        for node in ir_graph.nodes:
            # Check external model calls for ML nodes
            if self._is_ml_node(node):
                model_id = node.parameters.get('model_id')
                if model_id:
                    # Get model deployment region (would query model registry)
                    model_region = self._get_model_deployment_region(model_id)
                    if model_region:
                        node.external_call_regions[f"model_{model_id}"] = model_region
                        
                        # Validate node region matches model region
                        if node.region_id and node.region_id != model_region:
                            self.propagation_errors.append(
                                f"Node {node.id}: region mismatch with model {model_id} "
                                f"(node: {node.region_id}, model: {model_region})"
                            )
                            return False
            
            # Check external data source calls
            if node.type.value == "query":
                data_source = node.parameters.get('data_source')
                if data_source:
                    # Get data source region (would query data catalog)
                    source_region = self._get_data_source_region(data_source)
                    if source_region:
                        node.external_call_regions[f"datasource_{data_source}"] = source_region
                        
                        # Validate node region matches data source region
                        if node.region_id and node.region_id != source_region:
                            self.propagation_errors.append(
                                f"Node {node.id}: region mismatch with data source {data_source} "
                                f"(node: {node.region_id}, source: {source_region})"
                            )
                            return False
        
        return True
    
    def _validate_residency_constraints(self, ir_graph) -> bool:
        """Validate all residency constraints"""
        # Check cross-region data flows
        for edge_key, edge_residency in ir_graph.edge_residency.items():
            from_region = edge_residency["from_region"]
            to_region = edge_residency["to_region"]
            
            if from_region and to_region and from_region != to_region:
                # Check if cross-region flow is allowed
                if not self._is_cross_region_flow_allowed(ir_graph, edge_key, edge_residency):
                    self.residency_violations.append({
                        "type": "cross_region_data_flow",
                        "edge": edge_key,
                        "from_region": from_region,
                        "to_region": to_region,
                        "data_types": edge_residency["data_types"]
                    })
        
        # Check data sovereignty constraints
        for node in ir_graph.nodes:
            for data_type, required_region in node.data_residency_constraints.items():
                if node.region_id != required_region:
                    self.residency_violations.append({
                        "type": "data_sovereignty_violation",
                        "node_id": node.id,
                        "data_type": data_type,
                        "required_region": required_region,
                        "actual_region": node.region_id
                    })
        
        # Log violations but don't fail (warnings)
        if self.residency_violations:
            for violation in self.residency_violations:
                logger.warning(f"Residency violation: {violation}")
        
        return True
    
    def _apply_policy_residency_constraints(self, node):
        """Apply policy-based residency constraints to node"""
        for policy in node.policies:
            if policy.policy_type == "residency":
                params = policy.parameters
                
                # Apply allowed regions constraint
                allowed_regions = params.get("allowed_regions", [])
                if allowed_regions and node.region_id not in allowed_regions:
                    # Use first allowed region as fallback
                    fallback_region = params.get("fallback_region", allowed_regions[0] if allowed_regions else None)
                    if fallback_region:
                        logger.info(f"Node {node.id}: using fallback region {fallback_region}")
                        node.region_id = fallback_region
                
                # Apply data sovereignty constraints
                data_sovereignty = params.get("data_sovereignty", {})
                for data_type, required_region in data_sovereignty.items():
                    node.data_residency_constraints[data_type] = required_region
    
    def _validate_ml_node_residency(self, node) -> bool:
        """Validate ML node residency constraints"""
        model_id = node.parameters.get('model_id')
        if not model_id:
            return True
        
        # Check if model is deployed in node's region
        model_region = self._get_model_deployment_region(model_id)
        if model_region and node.region_id != model_region:
            # Check if cross-region ML calls are allowed
            cross_region_allowed = False
            for policy in node.policies:
                if policy.policy_type == "residency":
                    if policy.parameters.get("cross_region_allowed", False):
                        cross_region_allowed = True
                        break
            
            if not cross_region_allowed:
                self.propagation_errors.append(
                    f"ML node {node.id}: model {model_id} not available in region {node.region_id} "
                    f"(model region: {model_region})"
                )
                return False
        
        return True
    
    def _is_cross_region_flow_allowed(self, ir_graph, edge_key: str, edge_residency: Dict[str, Any]) -> bool:
        """Check if cross-region data flow is allowed"""
        # Check workflow-level policy
        workflow_policies = ir_graph.governance.get('policies', [])
        for policy in workflow_policies:
            if policy.get('type') == 'residency':
                if not policy.get('cross_region_allowed', False):
                    return False
        
        # Check node-level policies
        from_node_id = edge_key.split('->')[0]
        from_node = ir_graph.get_node(from_node_id)
        
        if from_node:
            for policy in from_node.policies:
                if policy.policy_type == "residency":
                    if not policy.parameters.get("cross_region_allowed", False):
                        return False
        
        return True
    
    def _get_model_deployment_region(self, model_id: str) -> Optional[str]:
        """Get model deployment region from model registry"""
        # Mock implementation - would query actual model registry
        model_regions = {
            "saas_churn_predictor_v2": "us-east-1",
            "fraud_detection_model": "us-west-2",
            "opportunity_score_model": "us-east-1"
        }
        return model_regions.get(model_id)
    
    def _get_data_source_region(self, data_source: str) -> Optional[str]:
        """Get data source region from data catalog"""
        # Mock implementation - would query actual data catalog
        source_regions = {
            "customer_db": "us-east-1",
            "opportunity_db": "us-east-1",
            "analytics_warehouse": "us-west-2"
        }
        return source_regions.get(data_source)
    
    def _is_ml_node(self, node) -> bool:
        """Check if node is an ML node"""
        from .ir import IRNodeType
        ml_types = {IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                   IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN}
        return node.type in ml_types
    
    def get_residency_report(self, ir_graph) -> Dict[str, Any]:
        """Get comprehensive residency report"""
        # Count nodes by region
        region_distribution = {}
        for node in ir_graph.nodes:
            region = node.region_id or "unspecified"
            region_distribution[region] = region_distribution.get(region, 0) + 1
        
        # Count cross-region edges
        cross_region_edges = 0
        for edge_residency in ir_graph.edge_residency.values():
            if (edge_residency["from_region"] and edge_residency["to_region"] and 
                edge_residency["from_region"] != edge_residency["to_region"]):
                cross_region_edges += 1
        
        # Count external calls by region
        external_call_regions = {}
        for node in ir_graph.nodes:
            for call_id, region in node.external_call_regions.items():
                external_call_regions[region] = external_call_regions.get(region, 0) + 1
        
        return {
            "workflow_id": ir_graph.workflow_id,
            "residency_summary": {
                "total_nodes": len(ir_graph.nodes),
                "region_distribution": region_distribution,
                "cross_region_edges": cross_region_edges,
                "total_edges": len(ir_graph.edges),
                "external_call_regions": external_call_regions
            },
            "violations": self.residency_violations,
            "compliance_status": "compliant" if not self.residency_violations else "violations_detected",
            "recommendations": self._generate_residency_recommendations(ir_graph)
        }
    
    def _generate_residency_recommendations(self, ir_graph) -> List[str]:
        """Generate recommendations for residency compliance"""
        recommendations = []
        
        # Check for nodes without regions
        nodes_without_region = [node.id for node in ir_graph.nodes if not node.region_id]
        if nodes_without_region:
            recommendations.append(f"Specify regions for nodes: {', '.join(nodes_without_region)}")
        
        # Check for cross-region violations
        if self.residency_violations:
            cross_region_violations = [v for v in self.residency_violations if v["type"] == "cross_region_data_flow"]
            if cross_region_violations:
                recommendations.append(f"Review {len(cross_region_violations)} cross-region data flows")
            
            sovereignty_violations = [v for v in self.residency_violations if v["type"] == "data_sovereignty_violation"]
            if sovereignty_violations:
                recommendations.append(f"Address {len(sovereignty_violations)} data sovereignty violations")
        
        # Check for region consolidation opportunities
        region_count = len(set(node.region_id for node in ir_graph.nodes if node.region_id))
        if region_count > 2:
            recommendations.append(f"Consider consolidating {region_count} regions to reduce complexity")
        
        return recommendations

# Task 6.2.10: Integration Function
def propagate_and_validate_residency(ir_graph) -> Tuple[bool, Dict[str, Any]]:
    """Propagate residency tags and validate constraints"""
    engine = ResidencyPropagationEngine()
    is_valid = engine.propagate_residency_tags(ir_graph)
    
    result = {
        "is_valid": is_valid,
        "errors": engine.propagation_errors,
        "violations": engine.residency_violations,
        "residency_report": engine.get_residency_report(ir_graph)
    }
    
    return is_valid, result
