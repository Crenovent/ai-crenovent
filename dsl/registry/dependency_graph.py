"""
Task 7.3-T43: Build Dep Graph (template depends-on connectors, policies)
Graph store for blast radius calculation and impact analysis
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

import asyncpg


class DependencyType(Enum):
    """Types of dependencies"""
    CONNECTOR = "connector"
    POLICY = "policy"
    WORKFLOW = "workflow"
    DATA_SOURCE = "data_source"
    EXTERNAL_API = "external_api"
    LIBRARY = "library"
    CONFIGURATION = "configuration"


class DependencyScope(Enum):
    """Scope of dependency"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEVELOPMENT = "development"
    RUNTIME = "runtime"


@dataclass
class DependencyNode:
    """Node in dependency graph"""
    node_id: str
    node_type: DependencyType
    name: str
    version: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    provider: Optional[str] = None
    category: Optional[str] = None
    
    # Status
    is_active: bool = True
    health_status: str = "healthy"
    last_verified: Optional[datetime] = None
    
    # Configuration
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "provider": self.provider,
            "category": self.category,
            "is_active": self.is_active,
            "health_status": self.health_status,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "configuration": self.configuration
        }


@dataclass
class DependencyEdge:
    """Edge in dependency graph"""
    edge_id: str
    source_id: str
    target_id: str
    dependency_scope: DependencyScope
    
    # Relationship metadata
    relationship_type: str = "depends_on"
    version_constraint: Optional[str] = None
    
    # Impact analysis
    criticality: str = "medium"  # low, medium, high, critical
    failure_impact: str = "partial"  # none, partial, complete
    
    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "dependency_scope": self.dependency_scope.value,
            "relationship_type": self.relationship_type,
            "version_constraint": self.version_constraint,
            "criticality": self.criticality,
            "failure_impact": self.failure_impact,
            "created_at": self.created_at.isoformat()
        }


class DependencyGraphManager:
    """Manager for workflow dependency graphs"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.graph_cache: Dict[str, Dict[str, Any]] = {}
    
    async def build_workflow_dependency_graph(
        self,
        workflow_id: str,
        version_id: Optional[str] = None,
        include_transitive: bool = True,
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """Build complete dependency graph for a workflow"""
        
        try:
            # Get workflow definition
            workflow_data = await self._get_workflow_data(workflow_id, version_id)
            if not workflow_data:
                return {"error": "Workflow not found"}
            
            # Initialize graph
            graph = {
                "workflow_id": workflow_id,
                "version_id": version_id or workflow_data.get("version_id"),
                "nodes": {},
                "edges": {},
                "metadata": {
                    "build_timestamp": datetime.now(timezone.utc).isoformat(),
                    "include_transitive": include_transitive,
                    "max_depth": max_depth,
                    "total_nodes": 0,
                    "total_edges": 0
                }
            }
            
            # Add root workflow node
            root_node = DependencyNode(
                node_id=workflow_id,
                node_type=DependencyType.WORKFLOW,
                name=workflow_data.get("workflow_name", "Unknown"),
                version=workflow_data.get("version_number"),
                description=workflow_data.get("description"),
                category="root_workflow"
            )
            graph["nodes"][workflow_id] = root_node.to_dict()
            
            # Extract direct dependencies
            direct_deps = await self._extract_workflow_dependencies(workflow_data)
            
            # Add direct dependency nodes and edges
            for dep in direct_deps:
                # Add dependency node
                dep_node = DependencyNode(
                    node_id=dep["node_id"],
                    node_type=DependencyType(dep["type"]),
                    name=dep["name"],
                    version=dep.get("version"),
                    description=dep.get("description"),
                    provider=dep.get("provider"),
                    category=dep.get("category")
                )
                graph["nodes"][dep["node_id"]] = dep_node.to_dict()
                
                # Add dependency edge
                edge = DependencyEdge(
                    edge_id=f"{workflow_id}->{dep['node_id']}",
                    source_id=workflow_id,
                    target_id=dep["node_id"],
                    dependency_scope=DependencyScope(dep.get("scope", "required")),
                    relationship_type="depends_on",
                    version_constraint=dep.get("version_constraint"),
                    criticality=dep.get("criticality", "medium"),
                    failure_impact=dep.get("failure_impact", "partial")
                )
                graph["edges"][edge.edge_id] = edge.to_dict()
            
            # Build transitive dependencies if requested
            if include_transitive:
                await self._build_transitive_dependencies(
                    graph, direct_deps, max_depth - 1
                )
            
            # Calculate graph metrics
            graph["metadata"]["total_nodes"] = len(graph["nodes"])
            graph["metadata"]["total_edges"] = len(graph["edges"])
            graph["metadata"]["dependency_depth"] = self._calculate_max_depth(graph)
            graph["metadata"]["critical_path"] = self._find_critical_path(graph)
            
            # Store in cache
            cache_key = f"{workflow_id}:{version_id or 'latest'}"
            self.graph_cache[cache_key] = graph
            
            # Store in database
            if self.db_pool:
                await self._store_dependency_graph(graph)
            
            return graph
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_workflow_data(
        self,
        workflow_id: str,
        version_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get workflow data from registry"""
        
        if not self.db_pool:
            # Return mock data for testing
            return {
                "workflow_id": workflow_id,
                "workflow_name": "Mock Workflow",
                "version_id": version_id or str(uuid.uuid4()),
                "version_number": "1.0.0",
                "description": "Mock workflow for testing",
                "steps": [
                    {
                        "id": "step1",
                        "type": "query",
                        "params": {
                            "data_source": "salesforce",
                            "query": "SELECT * FROM opportunities"
                        }
                    },
                    {
                        "id": "step2",
                        "type": "notify",
                        "params": {
                            "channel": "slack",
                            "webhook_url": "https://hooks.slack.com/..."
                        }
                    }
                ]
            }
        
        try:
            # Query workflow and version data
            query = """
                SELECT 
                    w.workflow_id,
                    w.workflow_name,
                    w.description,
                    v.version_id,
                    v.version_number,
                    a.content_data
                FROM registry_workflows w
                JOIN workflow_versions v ON w.workflow_id = v.workflow_id
                LEFT JOIN workflow_artifacts a ON v.version_id = a.version_id 
                    AND a.artifact_type = 'dsl'
                WHERE w.workflow_id = $1
                AND ($2 IS NULL OR v.version_id = $2)
                AND v.status = 'published'
                ORDER BY v.created_at DESC
                LIMIT 1
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, workflow_id, version_id)
                
                if not row:
                    return None
                
                workflow_data = dict(row)
                
                # Parse DSL content if available
                if row["content_data"]:
                    try:
                        if isinstance(row["content_data"], str):
                            if row["content_data"].strip().startswith('{'):
                                dsl_data = json.loads(row["content_data"])
                            else:
                                import yaml
                                dsl_data = yaml.safe_load(row["content_data"])
                        else:
                            dsl_data = row["content_data"]
                        
                        workflow_data.update(dsl_data)
                    except Exception:
                        # Use basic workflow data if DSL parsing fails
                        pass
                
                return workflow_data
                
        except Exception:
            return None
    
    async def _extract_workflow_dependencies(
        self,
        workflow_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract dependencies from workflow definition"""
        
        dependencies = []
        steps = workflow_data.get("steps", [])
        
        for step in steps:
            step_type = step.get("type")
            params = step.get("params", {})
            
            # Extract connector dependencies
            if step_type == "query":
                data_source = params.get("data_source")
                if data_source:
                    dependencies.append({
                        "node_id": f"connector_{data_source}",
                        "type": "connector",
                        "name": data_source,
                        "category": "data_connector",
                        "scope": "required",
                        "criticality": "high",
                        "failure_impact": "complete"
                    })
            
            # Extract API dependencies
            elif step_type == "agent_call":
                endpoint = params.get("endpoint")
                if endpoint:
                    api_id = f"api_{endpoint.replace('https://', '').replace('http://', '').split('/')[0]}"
                    dependencies.append({
                        "node_id": api_id,
                        "type": "external_api",
                        "name": endpoint,
                        "category": "external_service",
                        "scope": "required",
                        "criticality": "medium",
                        "failure_impact": "partial"
                    })
            
            # Extract notification dependencies
            elif step_type == "notify":
                channel = params.get("channel")
                if channel:
                    dependencies.append({
                        "node_id": f"notification_{channel}",
                        "type": "external_api",
                        "name": f"{channel}_service",
                        "category": "notification_service",
                        "scope": "optional",
                        "criticality": "low",
                        "failure_impact": "none"
                    })
            
            # Extract policy dependencies
            governance = step.get("governance", {})
            policy_id = governance.get("policy_id")
            if policy_id:
                dependencies.append({
                    "node_id": policy_id,
                    "type": "policy",
                    "name": policy_id,
                    "category": "governance_policy",
                    "scope": "required",
                    "criticality": "high",
                    "failure_impact": "complete"
                })
        
        # Extract workflow-level dependencies
        workflow_deps = workflow_data.get("dependencies", [])
        for dep in workflow_deps:
            dependencies.append({
                "node_id": dep.get("id", dep.get("name")),
                "type": dep.get("type", "library"),
                "name": dep.get("name"),
                "version": dep.get("version"),
                "category": dep.get("category", "external_library"),
                "scope": "required" if dep.get("required", True) else "optional",
                "criticality": "medium",
                "failure_impact": "partial"
            })
        
        return dependencies
    
    async def _build_transitive_dependencies(
        self,
        graph: Dict[str, Any],
        direct_deps: List[Dict[str, Any]],
        remaining_depth: int
    ) -> None:
        """Build transitive dependencies recursively"""
        
        if remaining_depth <= 0:
            return
        
        for dep in direct_deps:
            dep_id = dep["node_id"]
            dep_type = dep["type"]
            
            # Get transitive dependencies based on type
            transitive_deps = []
            
            if dep_type == "connector":
                transitive_deps = await self._get_connector_dependencies(dep["name"])
            elif dep_type == "policy":
                transitive_deps = await self._get_policy_dependencies(dep["name"])
            elif dep_type == "workflow":
                # Recursive workflow dependencies
                sub_workflow_data = await self._get_workflow_data(dep_id, None)
                if sub_workflow_data:
                    transitive_deps = await self._extract_workflow_dependencies(sub_workflow_data)
            
            # Add transitive nodes and edges
            for trans_dep in transitive_deps:
                trans_node_id = trans_dep["node_id"]
                
                # Add node if not already present
                if trans_node_id not in graph["nodes"]:
                    trans_node = DependencyNode(
                        node_id=trans_node_id,
                        node_type=DependencyType(trans_dep["type"]),
                        name=trans_dep["name"],
                        version=trans_dep.get("version"),
                        description=trans_dep.get("description"),
                        category=trans_dep.get("category")
                    )
                    graph["nodes"][trans_node_id] = trans_node.to_dict()
                
                # Add edge if not already present
                edge_id = f"{dep_id}->{trans_node_id}"
                if edge_id not in graph["edges"]:
                    edge = DependencyEdge(
                        edge_id=edge_id,
                        source_id=dep_id,
                        target_id=trans_node_id,
                        dependency_scope=DependencyScope(trans_dep.get("scope", "required")),
                        relationship_type="transitive_depends_on",
                        criticality=trans_dep.get("criticality", "low"),
                        failure_impact=trans_dep.get("failure_impact", "none")
                    )
                    graph["edges"][edge_id] = edge.to_dict()
            
            # Recurse for next level
            if transitive_deps:
                await self._build_transitive_dependencies(
                    graph, transitive_deps, remaining_depth - 1
                )
    
    async def _get_connector_dependencies(self, connector_name: str) -> List[Dict[str, Any]]:
        """Get dependencies for a connector"""
        
        # Mock connector dependencies
        connector_deps = {
            "salesforce": [
                {
                    "node_id": "salesforce_api_v52",
                    "type": "external_api",
                    "name": "Salesforce REST API v52.0",
                    "category": "api_service",
                    "scope": "required",
                    "criticality": "critical"
                }
            ],
            "slack": [
                {
                    "node_id": "slack_webhook_api",
                    "type": "external_api", 
                    "name": "Slack Webhook API",
                    "category": "notification_api",
                    "scope": "required",
                    "criticality": "low"
                }
            ]
        }
        
        return connector_deps.get(connector_name, [])
    
    async def _get_policy_dependencies(self, policy_id: str) -> List[Dict[str, Any]]:
        """Get dependencies for a policy"""
        
        if not self.db_pool:
            return []
        
        try:
            # Query policy dependencies
            query = """
                SELECT 
                    policy_name,
                    policy_type,
                    industry_overlay,
                    policy_rules
                FROM dsl_policy_packs 
                WHERE policy_pack_id = $1
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, policy_id)
                
                if not row:
                    return []
                
                dependencies = []
                
                # Extract dependencies from policy rules
                policy_rules = row["policy_rules"]
                if isinstance(policy_rules, str):
                    try:
                        policy_rules = json.loads(policy_rules)
                    except:
                        policy_rules = {}
                
                # Check for compliance framework dependencies
                compliance_frameworks = policy_rules.get("compliance_frameworks", [])
                for framework in compliance_frameworks:
                    dependencies.append({
                        "node_id": f"compliance_{framework}",
                        "type": "configuration",
                        "name": f"{framework} Compliance Framework",
                        "category": "compliance_framework",
                        "scope": "required",
                        "criticality": "high"
                    })
                
                return dependencies
                
        except Exception:
            return []
    
    def _calculate_max_depth(self, graph: Dict[str, Any]) -> int:
        """Calculate maximum dependency depth"""
        
        # Simple BFS to find maximum depth
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        # Build adjacency list
        adj_list = {}
        for edge_data in edges.values():
            source = edge_data["source_id"]
            target = edge_data["target_id"]
            
            if source not in adj_list:
                adj_list[source] = []
            adj_list[source].append(target)
        
        # Find root nodes (nodes with no incoming edges)
        incoming = set()
        for edge_data in edges.values():
            incoming.add(edge_data["target_id"])
        
        root_nodes = [node_id for node_id in nodes.keys() if node_id not in incoming]
        
        # BFS from root nodes
        max_depth = 0
        for root in root_nodes:
            depth = self._bfs_max_depth(root, adj_list)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _bfs_max_depth(self, start_node: str, adj_list: Dict[str, List[str]]) -> int:
        """BFS to find maximum depth from a node"""
        
        from collections import deque
        
        queue = deque([(start_node, 0)])
        visited = set()
        max_depth = 0
        
        while queue:
            node, depth = queue.popleft()
            
            if node in visited:
                continue
            
            visited.add(node)
            max_depth = max(max_depth, depth)
            
            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return max_depth
    
    def _find_critical_path(self, graph: Dict[str, Any]) -> List[str]:
        """Find critical path in dependency graph"""
        
        # Simplified critical path - find longest path
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        # Find nodes with highest criticality
        critical_nodes = []
        for node_id, node_data in nodes.items():
            # Check if node has critical dependencies
            has_critical_edge = False
            for edge_data in edges.values():
                if (edge_data["target_id"] == node_id and 
                    edge_data["criticality"] in ["high", "critical"]):
                    has_critical_edge = True
                    break
            
            if has_critical_edge:
                critical_nodes.append(node_id)
        
        return critical_nodes[:5]  # Return top 5 critical nodes
    
    async def analyze_blast_radius(
        self,
        node_id: str,
        failure_type: str = "complete"
    ) -> Dict[str, Any]:
        """Analyze blast radius of a node failure"""
        
        # Get dependency graph for the node
        affected_workflows = []
        
        if self.db_pool:
            try:
                # Find all workflows that depend on this node
                query = """
                    SELECT DISTINCT workflow_id, impact_score
                    FROM dependency_graph_edges dge
                    JOIN dependency_graph_nodes dgn ON dge.source_id = dgn.node_id
                    WHERE dge.target_id = $1
                    AND dgn.node_type = 'workflow'
                """
                
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(query, node_id)
                    
                    for row in rows:
                        affected_workflows.append({
                            "workflow_id": row["workflow_id"],
                            "impact_score": row["impact_score"] or 0.5
                        })
            except Exception:
                pass
        
        # Calculate blast radius metrics
        total_workflows = len(affected_workflows)
        high_impact_workflows = len([w for w in affected_workflows if w["impact_score"] > 0.7])
        
        blast_radius = {
            "node_id": node_id,
            "failure_type": failure_type,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "affected_workflows": affected_workflows,
            "metrics": {
                "total_affected_workflows": total_workflows,
                "high_impact_workflows": high_impact_workflows,
                "blast_radius_score": min(1.0, total_workflows / 10.0),  # Normalize to 0-1
                "severity": "critical" if high_impact_workflows > 5 else 
                          "high" if total_workflows > 10 else
                          "medium" if total_workflows > 3 else "low"
            },
            "recommendations": self._generate_blast_radius_recommendations(
                total_workflows, high_impact_workflows
            )
        }
        
        return blast_radius
    
    def _generate_blast_radius_recommendations(
        self,
        total_workflows: int,
        high_impact_workflows: int
    ) -> List[str]:
        """Generate recommendations based on blast radius analysis"""
        
        recommendations = []
        
        if high_impact_workflows > 5:
            recommendations.append("Consider implementing circuit breakers for this dependency")
            recommendations.append("Add redundancy or fallback mechanisms")
        
        if total_workflows > 10:
            recommendations.append("This dependency has high fan-out - consider splitting")
            recommendations.append("Implement graceful degradation strategies")
        
        if total_workflows > 0:
            recommendations.append("Monitor this dependency closely")
            recommendations.append("Set up proactive alerting")
        
        if not recommendations:
            recommendations.append("Dependency has low blast radius - standard monitoring sufficient")
        
        return recommendations
    
    async def _store_dependency_graph(self, graph: Dict[str, Any]) -> None:
        """Store dependency graph in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Store nodes
                    for node_id, node_data in graph["nodes"].items():
                        await conn.execute("""
                            INSERT INTO dependency_graph_nodes (
                                node_id, workflow_id, node_type, name, version,
                                description, provider, category, is_active,
                                health_status, configuration, created_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            ON CONFLICT (node_id, workflow_id) 
                            DO UPDATE SET
                                name = EXCLUDED.name,
                                version = EXCLUDED.version,
                                description = EXCLUDED.description,
                                updated_at = NOW()
                        """,
                        node_id,
                        graph["workflow_id"],
                        node_data["node_type"],
                        node_data["name"],
                        node_data.get("version"),
                        node_data.get("description"),
                        node_data.get("provider"),
                        node_data.get("category"),
                        node_data.get("is_active", True),
                        node_data.get("health_status", "healthy"),
                        json.dumps(node_data.get("configuration", {})),
                        datetime.now(timezone.utc)
                        )
                    
                    # Store edges
                    for edge_id, edge_data in graph["edges"].items():
                        await conn.execute("""
                            INSERT INTO dependency_graph_edges (
                                edge_id, workflow_id, source_id, target_id,
                                dependency_scope, relationship_type, version_constraint,
                                criticality, failure_impact, created_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT (edge_id, workflow_id)
                            DO UPDATE SET
                                criticality = EXCLUDED.criticality,
                                failure_impact = EXCLUDED.failure_impact,
                                updated_at = NOW()
                        """,
                        edge_id,
                        graph["workflow_id"],
                        edge_data["source_id"],
                        edge_data["target_id"],
                        edge_data["dependency_scope"],
                        edge_data["relationship_type"],
                        edge_data.get("version_constraint"),
                        edge_data["criticality"],
                        edge_data["failure_impact"],
                        datetime.now(timezone.utc)
                        )
        except Exception:
            # Log error but don't fail
            pass


# Database schema for dependency graphs
DEPENDENCY_GRAPH_SCHEMA_SQL = """
-- Dependency graph nodes
CREATE TABLE IF NOT EXISTS dependency_graph_nodes (
    node_id VARCHAR(255) NOT NULL,
    workflow_id UUID NOT NULL,
    node_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    description TEXT,
    provider VARCHAR(100),
    category VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    health_status VARCHAR(20) NOT NULL DEFAULT 'healthy',
    configuration JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (node_id, workflow_id),
    CONSTRAINT chk_node_type CHECK (node_type IN ('connector', 'policy', 'workflow', 'data_source', 'external_api', 'library', 'configuration')),
    CONSTRAINT chk_health_status CHECK (health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown'))
);

-- Dependency graph edges
CREATE TABLE IF NOT EXISTS dependency_graph_edges (
    edge_id VARCHAR(255) NOT NULL,
    workflow_id UUID NOT NULL,
    source_id VARCHAR(255) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    dependency_scope VARCHAR(20) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL DEFAULT 'depends_on',
    version_constraint VARCHAR(100),
    criticality VARCHAR(20) NOT NULL DEFAULT 'medium',
    failure_impact VARCHAR(20) NOT NULL DEFAULT 'partial',
    impact_score DECIMAL(3,2) DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (edge_id, workflow_id),
    CONSTRAINT chk_dependency_scope CHECK (dependency_scope IN ('required', 'optional', 'development', 'runtime')),
    CONSTRAINT chk_criticality CHECK (criticality IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_failure_impact CHECK (failure_impact IN ('none', 'partial', 'complete')),
    CONSTRAINT chk_impact_score CHECK (impact_score >= 0 AND impact_score <= 1)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_dep_nodes_workflow ON dependency_graph_nodes (workflow_id, node_type);
CREATE INDEX IF NOT EXISTS idx_dep_nodes_type ON dependency_graph_nodes (node_type, is_active);
CREATE INDEX IF NOT EXISTS idx_dep_edges_workflow ON dependency_graph_edges (workflow_id);
CREATE INDEX IF NOT EXISTS idx_dep_edges_source ON dependency_graph_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_dep_edges_target ON dependency_graph_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_dep_edges_criticality ON dependency_graph_edges (criticality, failure_impact);
"""
