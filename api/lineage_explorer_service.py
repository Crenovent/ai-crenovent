"""
Task 4.3.38: Lineage Explorer for Governance
End-to-end lineage tracking: dataset → model → output → decision → override
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging
import json

# Import existing services
from dsl.operators.override_ledger import OverrideLedger
from dsl.operators.explainability_service import ExplainabilityService
from dsl.intelligence.evidence_pack_generator import EvidencePackGenerator

app = FastAPI(title="RBIA Lineage Explorer API")
logger = logging.getLogger(__name__)

class LineageNodeType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"
    WORKFLOW = "workflow"
    STEP = "step"
    DECISION = "decision"
    OVERRIDE = "override"
    EVIDENCE = "evidence"
    EXPLANATION = "explanation"

class LineageNode(BaseModel):
    node_id: str
    node_type: LineageNodeType
    name: str
    description: str
    
    # Core attributes
    created_at: datetime
    created_by: Optional[str] = None
    tenant_id: str
    
    # Type-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Relationships
    inputs: List[str] = Field(default_factory=list)  # Node IDs that feed into this node
    outputs: List[str] = Field(default_factory=list)  # Node IDs that this node feeds into
    
    # Governance data
    governance_status: str = "compliant"  # compliant, flagged, overridden, quarantined
    risk_level: str = "low"  # low, medium, high, critical
    tags: List[str] = Field(default_factory=list)

class LineageEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str  # feeds_into, triggers, overrides, explains, evidences
    
    # Edge metadata
    created_at: datetime
    tenant_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LineageGraph(BaseModel):
    graph_id: str
    tenant_id: str
    nodes: List[LineageNode]
    edges: List[LineageEdge]
    
    # Graph metadata
    root_node_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    query_metadata: Dict[str, Any] = Field(default_factory=dict)

class LineageQuery(BaseModel):
    tenant_id: str
    
    # Query by specific node
    node_id: Optional[str] = None
    node_type: Optional[LineageNodeType] = None
    
    # Query by model/workflow
    model_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Query by time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Traversal options
    max_depth: int = Field(default=5, le=10)
    include_upstream: bool = True
    include_downstream: bool = True
    
    # Filters
    governance_status: Optional[List[str]] = None
    risk_levels: Optional[List[str]] = None

class LineageExplorer:
    """Service for exploring end-to-end data lineage in governance context"""
    
    def __init__(self):
        self.override_ledger = OverrideLedger()
        self.explainability_service = ExplainabilityService()
        self.evidence_generator = EvidencePackGenerator()
        self.logger = logging.getLogger(__name__)
    
    async def build_lineage_graph(self, query: LineageQuery) -> LineageGraph:
        """Build comprehensive lineage graph"""
        try:
            nodes = []
            edges = []
            
            # Start with root node(s)
            if query.node_id:
                root_nodes = await self._get_node_by_id(query.node_id, query.tenant_id)
            elif query.model_id:
                root_nodes = await self._get_nodes_by_model(query.model_id, query.tenant_id)
            elif query.workflow_id:
                root_nodes = await self._get_nodes_by_workflow(query.workflow_id, query.tenant_id)
            else:
                # Get recent decision nodes as starting points
                root_nodes = await self._get_recent_decision_nodes(query.tenant_id)
            
            if not root_nodes:
                return LineageGraph(
                    graph_id=str(uuid.uuid4()),
                    tenant_id=query.tenant_id,
                    nodes=[],
                    edges=[],
                    query_metadata={"message": "No matching nodes found"}
                )
            
            # Build graph through traversal
            visited_nodes = set()
            node_queue = root_nodes.copy()
            
            while node_queue and len(nodes) < 100:  # Limit graph size
                current_node = node_queue.pop(0)
                
                if current_node.node_id in visited_nodes:
                    continue
                
                visited_nodes.add(current_node.node_id)
                nodes.append(current_node)
                
                # Get related nodes and edges
                if query.include_upstream:
                    upstream_nodes, upstream_edges = await self._get_upstream_lineage(current_node, query)
                    node_queue.extend([n for n in upstream_nodes if n.node_id not in visited_nodes])
                    edges.extend(upstream_edges)
                
                if query.include_downstream:
                    downstream_nodes, downstream_edges = await self._get_downstream_lineage(current_node, query)
                    node_queue.extend([n for n in downstream_nodes if n.node_id not in visited_nodes])
                    edges.extend(downstream_edges)
            
            # Apply filters
            if query.governance_status:
                nodes = [n for n in nodes if n.governance_status in query.governance_status]
            
            if query.risk_levels:
                nodes = [n for n in nodes if n.risk_level in query.risk_levels]
            
            # Filter edges to only include those between remaining nodes
            node_ids = {n.node_id for n in nodes}
            edges = [e for e in edges if e.source_node_id in node_ids and e.target_node_id in node_ids]
            
            return LineageGraph(
                graph_id=str(uuid.uuid4()),
                tenant_id=query.tenant_id,
                nodes=nodes,
                edges=edges,
                root_node_id=root_nodes[0].node_id if root_nodes else None,
                query_metadata={
                    "query_time": datetime.utcnow().isoformat(),
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "max_depth": query.max_depth
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build lineage graph: {e}")
            raise
    
    async def _get_node_by_id(self, node_id: str, tenant_id: str) -> List[LineageNode]:
        """Get specific node by ID"""
        # This would query actual data sources
        # For now, return simulated node
        return [LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.MODEL,
            name=f"Model {node_id}",
            description="ML model node",
            created_at=datetime.utcnow(),
            tenant_id=tenant_id,
            metadata={"version": "v1.0", "accuracy": 0.85}
        )]
    
    async def _get_nodes_by_model(self, model_id: str, tenant_id: str) -> List[LineageNode]:
        """Get nodes related to a specific model"""
        nodes = []
        
        # Model node
        model_node = LineageNode(
            node_id=f"model_{model_id}",
            node_type=LineageNodeType.MODEL,
            name=f"Model {model_id}",
            description=f"ML model: {model_id}",
            created_at=datetime.utcnow(),
            tenant_id=tenant_id,
            metadata={
                "model_id": model_id,
                "version": "v1.0",
                "accuracy": 0.85,
                "training_date": datetime.utcnow().isoformat()
            }
        )
        nodes.append(model_node)
        
        # Get decisions made by this model
        decision_nodes = await self._get_decision_nodes_by_model(model_id, tenant_id)
        nodes.extend(decision_nodes)
        
        return nodes
    
    async def _get_nodes_by_workflow(self, workflow_id: str, tenant_id: str) -> List[LineageNode]:
        """Get nodes related to a specific workflow"""
        nodes = []
        
        # Workflow node
        workflow_node = LineageNode(
            node_id=f"workflow_{workflow_id}",
            node_type=LineageNodeType.WORKFLOW,
            name=f"Workflow {workflow_id}",
            description=f"DSL workflow: {workflow_id}",
            created_at=datetime.utcnow(),
            tenant_id=tenant_id,
            metadata={
                "workflow_id": workflow_id,
                "status": "active"
            }
        )
        nodes.append(workflow_node)
        
        return nodes
    
    async def _get_recent_decision_nodes(self, tenant_id: str) -> List[LineageNode]:
        """Get recent decision nodes as starting points"""
        # Get recent overrides as decision points
        try:
            override_entries = await self.override_ledger.get_entries_by_tenant(tenant_id)
            nodes = []
            
            for entry in override_entries[:10]:  # Limit to recent entries
                decision_node = LineageNode(
                    node_id=f"decision_{entry.entry_id}",
                    node_type=LineageNodeType.DECISION,
                    name=f"Decision: {entry.override_type}",
                    description=f"Decision point with override",
                    created_at=entry.created_at,
                    tenant_id=tenant_id,
                    metadata={
                        "original_prediction": entry.original_prediction,
                        "override_prediction": entry.override_prediction,
                        "model_id": entry.model_id,
                        "workflow_id": entry.workflow_id
                    },
                    governance_status="overridden",
                    risk_level=self._assess_override_risk(entry.override_type)
                )
                nodes.append(decision_node)
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get recent decision nodes: {e}")
            return []
    
    async def _get_decision_nodes_by_model(self, model_id: str, tenant_id: str) -> List[LineageNode]:
        """Get decision nodes for a specific model"""
        try:
            # Get explainability logs for decisions
            explanations = await self.explainability_service.get_explanations_by_model(model_id, tenant_id)
            nodes = []
            
            for explanation in explanations[:5]:  # Limit results
                decision_node = LineageNode(
                    node_id=f"decision_{explanation.log_id}",
                    node_type=LineageNodeType.DECISION,
                    name=f"Decision: {explanation.prediction}",
                    description=f"ML decision with explanation",
                    created_at=explanation.created_at,
                    tenant_id=tenant_id,
                    metadata={
                        "prediction": explanation.prediction,
                        "confidence": explanation.confidence,
                        "explanation_type": explanation.explanation_type,
                        "model_id": model_id
                    },
                    governance_status="explained"
                )
                nodes.append(decision_node)
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get decision nodes by model: {e}")
            return []
    
    async def _get_upstream_lineage(self, node: LineageNode, query: LineageQuery) -> tuple[List[LineageNode], List[LineageEdge]]:
        """Get upstream nodes and edges"""
        upstream_nodes = []
        upstream_edges = []
        
        if node.node_type == LineageNodeType.DECISION:
            # Decision comes from model
            model_id = node.metadata.get("model_id")
            if model_id:
                model_node = LineageNode(
                    node_id=f"model_{model_id}",
                    node_type=LineageNodeType.MODEL,
                    name=f"Model {model_id}",
                    description=f"ML model: {model_id}",
                    created_at=node.created_at,
                    tenant_id=query.tenant_id,
                    metadata={"model_id": model_id}
                )
                upstream_nodes.append(model_node)
                
                edge = LineageEdge(
                    edge_id=str(uuid.uuid4()),
                    source_node_id=model_node.node_id,
                    target_node_id=node.node_id,
                    relationship_type="generates",
                    created_at=node.created_at,
                    tenant_id=query.tenant_id
                )
                upstream_edges.append(edge)
                
                # Model comes from dataset
                dataset_node = LineageNode(
                    node_id=f"dataset_{model_id}_training",
                    node_type=LineageNodeType.DATASET,
                    name=f"Training Dataset",
                    description=f"Training data for {model_id}",
                    created_at=node.created_at,
                    tenant_id=query.tenant_id,
                    metadata={"model_id": model_id, "data_type": "training"}
                )
                upstream_nodes.append(dataset_node)
                
                dataset_edge = LineageEdge(
                    edge_id=str(uuid.uuid4()),
                    source_node_id=dataset_node.node_id,
                    target_node_id=model_node.node_id,
                    relationship_type="trains",
                    created_at=node.created_at,
                    tenant_id=query.tenant_id
                )
                upstream_edges.append(dataset_edge)
        
        return upstream_nodes, upstream_edges
    
    async def _get_downstream_lineage(self, node: LineageNode, query: LineageQuery) -> tuple[List[LineageNode], List[LineageEdge]]:
        """Get downstream nodes and edges"""
        downstream_nodes = []
        downstream_edges = []
        
        if node.node_type == LineageNodeType.DECISION:
            # Check for overrides
            try:
                override_entries = await self.override_ledger.get_entries_by_tenant(query.tenant_id)
                
                for entry in override_entries:
                    if entry.model_id == node.metadata.get("model_id"):
                        override_node = LineageNode(
                            node_id=f"override_{entry.entry_id}",
                            node_type=LineageNodeType.OVERRIDE,
                            name=f"Override: {entry.override_type}",
                            description=f"Manual override of decision",
                            created_at=entry.created_at,
                            tenant_id=query.tenant_id,
                            metadata={
                                "override_type": entry.override_type,
                                "justification": entry.justification,
                                "original_prediction": entry.original_prediction,
                                "override_prediction": entry.override_prediction
                            },
                            governance_status="overridden",
                            risk_level=self._assess_override_risk(entry.override_type)
                        )
                        downstream_nodes.append(override_node)
                        
                        edge = LineageEdge(
                            edge_id=str(uuid.uuid4()),
                            source_node_id=node.node_id,
                            target_node_id=override_node.node_id,
                            relationship_type="overridden_by",
                            created_at=entry.created_at,
                            tenant_id=query.tenant_id,
                            metadata={"justification": entry.justification}
                        )
                        downstream_edges.append(edge)
                        
                        # Evidence pack for override
                        evidence_node = LineageNode(
                            node_id=f"evidence_{entry.entry_id}",
                            node_type=LineageNodeType.EVIDENCE,
                            name=f"Override Evidence",
                            description=f"Evidence pack for override",
                            created_at=entry.created_at,
                            tenant_id=query.tenant_id,
                            metadata={
                                "evidence_type": "override_justification",
                                "override_id": entry.entry_id
                            }
                        )
                        downstream_nodes.append(evidence_node)
                        
                        evidence_edge = LineageEdge(
                            edge_id=str(uuid.uuid4()),
                            source_node_id=override_node.node_id,
                            target_node_id=evidence_node.node_id,
                            relationship_type="evidenced_by",
                            created_at=entry.created_at,
                            tenant_id=query.tenant_id
                        )
                        downstream_edges.append(evidence_edge)
                
            except Exception as e:
                self.logger.error(f"Failed to get override lineage: {e}")
        
        return downstream_nodes, downstream_edges
    
    def _assess_override_risk(self, override_type: str) -> str:
        """Assess risk level of override"""
        risk_map = {
            "emergency": "critical",
            "policy": "high", 
            "compliance": "high",
            "manual": "medium",
            "business_rule": "low"
        }
        return risk_map.get(override_type, "medium")

# Global service instance
lineage_explorer = LineageExplorer()

@app.post("/lineage/explore", response_model=LineageGraph)
async def explore_lineage(query: LineageQuery):
    """Explore data lineage for governance"""
    return await lineage_explorer.build_lineage_graph(query)

@app.get("/lineage/model/{model_id}")
async def get_model_lineage(
    model_id: str,
    tenant_id: str = Query(..., description="Tenant ID"),
    depth: int = Query(default=3, description="Maximum traversal depth")
):
    """Get lineage for a specific model"""
    query = LineageQuery(
        tenant_id=tenant_id,
        model_id=model_id,
        max_depth=depth
    )
    return await lineage_explorer.build_lineage_graph(query)

@app.get("/lineage/workflow/{workflow_id}")
async def get_workflow_lineage(
    workflow_id: str,
    tenant_id: str = Query(..., description="Tenant ID"),
    depth: int = Query(default=3, description="Maximum traversal depth")
):
    """Get lineage for a specific workflow"""
    query = LineageQuery(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        max_depth=depth
    )
    return await lineage_explorer.build_lineage_graph(query)

@app.get("/lineage/decisions/{tenant_id}")
async def get_recent_decisions_lineage(
    tenant_id: str,
    hours: int = Query(default=24, description="Hours to look back"),
    include_overrides: bool = Query(default=True, description="Include override lineage")
):
    """Get lineage for recent decisions"""
    from datetime import timedelta
    
    query = LineageQuery(
        tenant_id=tenant_id,
        start_date=datetime.utcnow() - timedelta(hours=hours),
        max_depth=4,
        include_downstream=include_overrides
    )
    return await lineage_explorer.build_lineage_graph(query)

@app.get("/lineage/governance-events/{tenant_id}")
async def get_governance_lineage(
    tenant_id: str,
    event_types: str = Query(default="override,quarantine", description="Comma-separated event types"),
    days: int = Query(default=7, description="Days to look back")
):
    """Get lineage for governance events"""
    from datetime import timedelta
    
    # Filter by governance status
    governance_statuses = []
    if "override" in event_types:
        governance_statuses.append("overridden")
    if "quarantine" in event_types:
        governance_statuses.append("quarantined")
    if "flagged" in event_types:
        governance_statuses.append("flagged")
    
    query = LineageQuery(
        tenant_id=tenant_id,
        start_date=datetime.utcnow() - timedelta(days=days),
        governance_status=governance_statuses,
        max_depth=5
    )
    return await lineage_explorer.build_lineage_graph(query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
