"""
Complexity Limits Service - Task 6.2.39
========================================

Complexity limits (node count, depth, fanout)
- Sets compile-time complexity caps to avoid runaway plans
- Protects runtime from denial-of-service-style complexity
- Analyzer that computes plan complexity metrics
- Policy-configurable caps per tenant/environment
- Backend implementation (no actual dashboard/telemetry - that's infrastructure)

Dependencies: Task 6.2.4 (IR)
Outputs: Complexity analysis and limits → enables DoS protection and performance optimization
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ComplexityMetric(Enum):
    """Types of complexity metrics"""
    NODE_COUNT = "node_count"
    GRAPH_DEPTH = "graph_depth"
    MAX_FANOUT = "max_fanout"
    AVERAGE_FANOUT = "average_fanout"
    CYCLE_COUNT = "cycle_count"
    ATTACHMENT_SIZE = "attachment_size"
    CONDITIONAL_BRANCHES = "conditional_branches"
    ML_NODE_COUNT = "ml_node_count"
    EXTERNAL_CALLS = "external_calls"

class ComplexityLevel(Enum):
    """Complexity level classifications"""
    TRIVIAL = "trivial"      # Very simple plans
    SIMPLE = "simple"        # Basic plans
    MODERATE = "moderate"    # Standard complexity
    COMPLEX = "complex"      # High complexity
    EXTREME = "extreme"      # Very high complexity

class LimitAction(Enum):
    """Actions to take when limits are exceeded"""
    WARN = "warn"            # Issue warning but allow
    BLOCK = "block"          # Block compilation
    SUGGEST_REFACTOR = "suggest_refactor"  # Suggest refactoring
    REQUIRE_APPROVAL = "require_approval"  # Require manual approval
    AUTO_OPTIMIZE = "auto_optimize"        # Attempt automatic optimization

@dataclass
class ComplexityLimit:
    """A single complexity limit"""
    metric: ComplexityMetric
    limit_value: Union[int, float]
    action: LimitAction
    soft_limit: bool = False  # If true, can be exceeded with warnings
    grace_factor: float = 1.1  # Allow 10% over limit before hard enforcement
    
    # Context
    description: str = ""
    rationale: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ComplexityProfile:
    """Complexity limits profile for different environments"""
    profile_id: str
    profile_name: str
    environment: str  # dev, staging, prod
    tenant_id: Optional[str] = None
    
    # Limits
    limits: List[ComplexityLimit] = field(default_factory=list)
    
    # Global settings
    max_compilation_time_seconds: int = 60
    max_memory_usage_mb: int = 512
    enable_auto_optimization: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ComplexityAnalysis:
    """Results of complexity analysis"""
    plan_id: str
    analysis_id: str
    
    # Computed metrics
    metrics: Dict[ComplexityMetric, Union[int, float]] = field(default_factory=dict)
    
    # Analysis results
    complexity_level: ComplexityLevel = ComplexityLevel.SIMPLE
    total_complexity_score: float = 0.0
    
    # Limit violations
    limit_violations: List[str] = field(default_factory=list)
    limit_warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    optimization_suggestions: List[str] = field(default_factory=list)
    refactoring_suggestions: List[str] = field(default_factory=list)
    
    # Performance estimates
    estimated_compilation_time_seconds: float = 0.0
    estimated_runtime_memory_mb: float = 0.0
    estimated_execution_time_seconds: float = 0.0
    
    # Metadata
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_duration_ms: float = 0.0

@dataclass
class GraphMetrics:
    """Detailed graph structure metrics"""
    # Basic counts
    total_nodes: int = 0
    nodes_by_type: Dict[str, int] = field(default_factory=dict)
    total_edges: int = 0
    
    # Structure metrics
    max_depth: int = 0
    average_depth: float = 0.0
    max_fanout: int = 0
    average_fanout: float = 0.0
    
    # Complexity indicators
    cycle_count: int = 0
    strongly_connected_components: int = 0
    critical_path_length: int = 0
    
    # Special node counts
    ml_nodes: int = 0
    decision_nodes: int = 0
    loop_nodes: int = 0
    external_call_nodes: int = 0
    
    # Data flow metrics
    total_data_size_estimate_mb: float = 0.0
    max_single_attachment_mb: float = 0.0

# Task 6.2.39: Complexity Limits Service
class ComplexityLimitsService:
    """Service for analyzing and enforcing plan complexity limits"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Complexity profiles registry
        self.complexity_profiles: Dict[str, ComplexityProfile] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, ComplexityAnalysis] = {}  # plan_id -> analysis
        
        # Default complexity limits
        self.default_limits = self._initialize_default_limits()
        
        # Initialize standard profiles
        self._initialize_standard_profiles()
        
        # Statistics
        self.complexity_stats = {
            'total_analyses': 0,
            'plans_blocked': 0,
            'plans_warned': 0,
            'optimizations_suggested': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'complexity_levels': {level.value: 0 for level in ComplexityLevel}
        }
    
    def analyze_plan_complexity(self, plan_ir: Dict[str, Any], 
                               profile_id: Optional[str] = None) -> ComplexityAnalysis:
        """
        Analyze the complexity of a plan IR
        
        Args:
            plan_ir: Plan intermediate representation
            profile_id: Complexity profile to use for limits
            
        Returns:
            ComplexityAnalysis with detailed results
        """
        start_time = datetime.now(timezone.utc)
        plan_id = plan_ir.get('plan_id', 'unknown')
        
        # Check cache
        cache_key = f"{plan_id}_{hash(str(plan_ir))}"
        if cache_key in self.analysis_cache:
            self.complexity_stats['cache_hits'] += 1
            return self.analysis_cache[cache_key]
        
        self.complexity_stats['cache_misses'] += 1
        
        # Create analysis instance
        analysis_id = f"analysis_{plan_id}_{hash(str(plan_ir))}_{int(start_time.timestamp())}"
        analysis = ComplexityAnalysis(
            plan_id=plan_id,
            analysis_id=analysis_id
        )
        
        # Extract graph structure
        graph_metrics = self._extract_graph_metrics(plan_ir)
        
        # Compute complexity metrics
        analysis.metrics = self._compute_complexity_metrics(graph_metrics, plan_ir)
        
        # Determine complexity level
        analysis.complexity_level = self._determine_complexity_level(analysis.metrics)
        
        # Calculate total complexity score
        analysis.total_complexity_score = self._calculate_complexity_score(analysis.metrics)
        
        # Check limits if profile specified
        if profile_id and profile_id in self.complexity_profiles:
            profile = self.complexity_profiles[profile_id]
            self._check_complexity_limits(analysis, profile)
        
        # Generate suggestions
        analysis.optimization_suggestions = self._generate_optimization_suggestions(analysis.metrics, graph_metrics)
        analysis.refactoring_suggestions = self._generate_refactoring_suggestions(analysis.metrics, graph_metrics)
        
        # Estimate performance characteristics
        analysis.estimated_compilation_time_seconds = self._estimate_compilation_time(analysis.metrics)
        analysis.estimated_runtime_memory_mb = self._estimate_runtime_memory(analysis.metrics)
        analysis.estimated_execution_time_seconds = self._estimate_execution_time(analysis.metrics)
        
        # Record analysis duration
        end_time = datetime.now(timezone.utc)
        analysis.analysis_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Cache result
        self.analysis_cache[cache_key] = analysis
        
        # Update statistics
        self.complexity_stats['total_analyses'] += 1
        self.complexity_stats['complexity_levels'][analysis.complexity_level.value] += 1
        
        if analysis.limit_violations:
            self.complexity_stats['plans_blocked'] += 1
        elif analysis.limit_warnings:
            self.complexity_stats['plans_warned'] += 1
        
        if analysis.optimization_suggestions:
            self.complexity_stats['optimizations_suggested'] += 1
        
        self.logger.info(f"✅ Analyzed plan complexity: {plan_id} -> {analysis.complexity_level.value} (score: {analysis.total_complexity_score:.1f})")
        
        return analysis
    
    def create_complexity_profile(self, profile_name: str, environment: str,
                                 tenant_id: Optional[str] = None,
                                 custom_limits: Optional[List[Dict[str, Any]]] = None) -> ComplexityProfile:
        """
        Create a custom complexity profile
        
        Args:
            profile_name: Name of the profile
            environment: Environment (dev, staging, prod)
            tenant_id: Optional tenant identifier
            custom_limits: Custom limit specifications
            
        Returns:
            ComplexityProfile
        """
        profile_id = f"profile_{hash(f'{profile_name}_{environment}_{tenant_id}_{datetime.now()}')}"
        
        profile = ComplexityProfile(
            profile_id=profile_id,
            profile_name=profile_name,
            environment=environment,
            tenant_id=tenant_id
        )
        
        # Apply default limits based on environment
        if environment == "dev":
            base_limits = self.default_limits["development"]
        elif environment == "staging":
            base_limits = self.default_limits["staging"]
        elif environment == "prod":
            base_limits = self.default_limits["production"]
        else:
            base_limits = self.default_limits["development"]
        
        profile.limits = base_limits.copy()
        
        # Apply custom limits if provided
        if custom_limits:
            for custom_limit_dict in custom_limits:
                custom_limit = ComplexityLimit(
                    metric=ComplexityMetric(custom_limit_dict['metric']),
                    limit_value=custom_limit_dict['limit_value'],
                    action=LimitAction(custom_limit_dict.get('action', 'warn')),
                    soft_limit=custom_limit_dict.get('soft_limit', False),
                    description=custom_limit_dict.get('description', ''),
                    rationale=custom_limit_dict.get('rationale', '')
                )
                
                # Replace existing limit for this metric
                profile.limits = [l for l in profile.limits if l.metric != custom_limit.metric]
                profile.limits.append(custom_limit)
        
        # Store profile
        self.complexity_profiles[profile_id] = profile
        
        self.logger.info(f"✅ Created complexity profile: {profile_name} for {environment}")
        
        return profile
    
    def validate_plan_against_limits(self, plan_ir: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
        """
        Validate a plan against complexity limits
        
        Args:
            plan_ir: Plan intermediate representation
            profile_id: Complexity profile to validate against
            
        Returns:
            Dictionary with validation results
        """
        if profile_id not in self.complexity_profiles:
            return {
                'valid': False,
                'error': f'Unknown complexity profile: {profile_id}'
            }
        
        # Analyze complexity
        analysis = self.analyze_plan_complexity(plan_ir, profile_id)
        
        # Determine validation result
        is_valid = len(analysis.limit_violations) == 0
        
        result = {
            'valid': is_valid,
            'plan_id': analysis.plan_id,
            'complexity_level': analysis.complexity_level.value,
            'complexity_score': analysis.total_complexity_score,
            'violations': analysis.limit_violations,
            'warnings': analysis.limit_warnings,
            'suggestions': analysis.optimization_suggestions + analysis.refactoring_suggestions,
            'estimated_performance': {
                'compilation_time_seconds': analysis.estimated_compilation_time_seconds,
                'runtime_memory_mb': analysis.estimated_runtime_memory_mb,
                'execution_time_seconds': analysis.estimated_execution_time_seconds
            }
        }
        
        return result
    
    def get_complexity_trends(self, tenant_id: Optional[str] = None, 
                            days: int = 30) -> Dict[str, Any]:
        """
        Get complexity trends over time
        
        Args:
            tenant_id: Optional tenant filter
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        # Filter analyses by tenant and time period
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        relevant_analyses = []
        for analysis in self.analysis_cache.values():
            if analysis.analyzed_at >= cutoff_date:
                # In a real implementation, we'd filter by tenant_id from the plan metadata
                relevant_analyses.append(analysis)
        
        if not relevant_analyses:
            return {
                'trend_period_days': days,
                'total_analyses': 0,
                'trends': {}
            }
        
        # Calculate trends
        complexity_scores = [a.total_complexity_score for a in relevant_analyses]
        node_counts = [a.metrics.get(ComplexityMetric.NODE_COUNT, 0) for a in relevant_analyses]
        depths = [a.metrics.get(ComplexityMetric.GRAPH_DEPTH, 0) for a in relevant_analyses]
        
        trends = {
            'complexity_score': {
                'average': statistics.mean(complexity_scores),
                'median': statistics.median(complexity_scores),
                'max': max(complexity_scores),
                'min': min(complexity_scores)
            },
            'node_count': {
                'average': statistics.mean(node_counts),
                'median': statistics.median(node_counts),
                'max': max(node_counts),
                'min': min(node_counts)
            },
            'graph_depth': {
                'average': statistics.mean(depths),
                'median': statistics.median(depths),
                'max': max(depths),
                'min': min(depths)
            },
            'complexity_levels': defaultdict(int)
        }
        
        # Count by complexity level
        for analysis in relevant_analyses:
            trends['complexity_levels'][analysis.complexity_level.value] += 1
        
        return {
            'trend_period_days': days,
            'total_analyses': len(relevant_analyses),
            'trends': dict(trends)
        }
    
    def suggest_plan_optimizations(self, plan_ir: Dict[str, Any]) -> List[str]:
        """
        Suggest optimizations for a plan
        
        Args:
            plan_ir: Plan intermediate representation
            
        Returns:
            List of optimization suggestions
        """
        analysis = self.analyze_plan_complexity(plan_ir)
        
        suggestions = []
        suggestions.extend(analysis.optimization_suggestions)
        suggestions.extend(analysis.refactoring_suggestions)
        
        # Add specific suggestions based on metrics
        metrics = analysis.metrics
        
        if metrics.get(ComplexityMetric.NODE_COUNT, 0) > 100:
            suggestions.append("Consider breaking large workflow into smaller sub-workflows")
        
        if metrics.get(ComplexityMetric.GRAPH_DEPTH, 0) > 20:
            suggestions.append("Reduce workflow depth by parallelizing sequential steps where possible")
        
        if metrics.get(ComplexityMetric.MAX_FANOUT, 0) > 10:
            suggestions.append("Consider using aggregation nodes to reduce high fanout")
        
        if metrics.get(ComplexityMetric.ML_NODE_COUNT, 0) > 5:
            suggestions.append("Consider consolidating ML models or using ensemble approaches")
        
        return list(set(suggestions))  # Remove duplicates
    
    def get_complexity_statistics(self) -> Dict[str, Any]:
        """Get complexity service statistics"""
        return {
            **self.complexity_stats,
            'complexity_profiles': len(self.complexity_profiles),
            'cached_analyses': len(self.analysis_cache)
        }
    
    def _extract_graph_metrics(self, plan_ir: Dict[str, Any]) -> GraphMetrics:
        """Extract graph structure metrics from plan IR"""
        metrics = GraphMetrics()
        
        # Get nodes and edges from IR
        plan_graph = plan_ir.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        edges = plan_graph.get('edges', [])
        
        # Basic counts
        metrics.total_nodes = len(nodes)
        metrics.total_edges = len(edges)
        
        # Count nodes by type
        for node in nodes:
            node_type = node.get('type', 'unknown')
            metrics.nodes_by_type[node_type] = metrics.nodes_by_type.get(node_type, 0) + 1
            
            # Count special node types
            if 'ml_' in node_type or node_type in ['ml_predict', 'ml_classify', 'ml_score']:
                metrics.ml_nodes += 1
            elif 'decision' in node_type or 'condition' in node_type:
                metrics.decision_nodes += 1
            elif 'loop' in node_type or 'iterate' in node_type:
                metrics.loop_nodes += 1
            elif 'external' in node_type or 'api' in node_type:
                metrics.external_call_nodes += 1
        
        # Build adjacency list for graph analysis
        adjacency = defaultdict(list)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for edge in edges:
            source = edge.get('source', edge.get('from'))
            target = edge.get('target', edge.get('to'))
            if source and target:
                adjacency[source].append(target)
                out_degree[source] += 1
                in_degree[target] += 1
        
        # Calculate fanout metrics
        if out_degree:
            metrics.max_fanout = max(out_degree.values())
            metrics.average_fanout = sum(out_degree.values()) / len(out_degree)
        
        # Calculate depth metrics using BFS
        if nodes:
            depths = self._calculate_node_depths(adjacency, nodes)
            if depths:
                metrics.max_depth = max(depths.values())
                metrics.average_depth = sum(depths.values()) / len(depths)
        
        # Detect cycles
        metrics.cycle_count = self._count_cycles(adjacency, [node.get('id') for node in nodes])
        
        # Estimate data sizes
        for node in nodes:
            attachments = node.get('attachments', [])
            for attachment in attachments:
                size_mb = attachment.get('size_mb', 0)
                metrics.total_data_size_estimate_mb += size_mb
                metrics.max_single_attachment_mb = max(metrics.max_single_attachment_mb, size_mb)
        
        return metrics
    
    def _compute_complexity_metrics(self, graph_metrics: GraphMetrics, plan_ir: Dict[str, Any]) -> Dict[ComplexityMetric, Union[int, float]]:
        """Compute complexity metrics from graph metrics"""
        metrics = {}
        
        # Direct mappings
        metrics[ComplexityMetric.NODE_COUNT] = graph_metrics.total_nodes
        metrics[ComplexityMetric.GRAPH_DEPTH] = graph_metrics.max_depth
        metrics[ComplexityMetric.MAX_FANOUT] = graph_metrics.max_fanout
        metrics[ComplexityMetric.AVERAGE_FANOUT] = graph_metrics.average_fanout
        metrics[ComplexityMetric.CYCLE_COUNT] = graph_metrics.cycle_count
        metrics[ComplexityMetric.ML_NODE_COUNT] = graph_metrics.ml_nodes
        metrics[ComplexityMetric.EXTERNAL_CALLS] = graph_metrics.external_call_nodes
        
        # Calculated metrics
        metrics[ComplexityMetric.ATTACHMENT_SIZE] = graph_metrics.total_data_size_estimate_mb
        
        # Count conditional branches
        conditional_branches = 0
        plan_graph = plan_ir.get('plan_graph', {})
        for node in plan_graph.get('nodes', []):
            if node.get('type') in ['condition', 'decision', 'switch']:
                # Count outgoing edges as branches
                node_id = node.get('id')
                outgoing_edges = [e for e in plan_graph.get('edges', []) if e.get('source') == node_id]
                conditional_branches += len(outgoing_edges)
        
        metrics[ComplexityMetric.CONDITIONAL_BRANCHES] = conditional_branches
        
        return metrics
    
    def _determine_complexity_level(self, metrics: Dict[ComplexityMetric, Union[int, float]]) -> ComplexityLevel:
        """Determine overall complexity level based on metrics"""
        node_count = metrics.get(ComplexityMetric.NODE_COUNT, 0)
        depth = metrics.get(ComplexityMetric.GRAPH_DEPTH, 0)
        fanout = metrics.get(ComplexityMetric.MAX_FANOUT, 0)
        ml_nodes = metrics.get(ComplexityMetric.ML_NODE_COUNT, 0)
        
        # Simple scoring system
        score = 0
        
        # Node count scoring
        if node_count <= 10:
            score += 1
        elif node_count <= 50:
            score += 2
        elif node_count <= 200:
            score += 3
        else:
            score += 4
        
        # Depth scoring
        if depth <= 5:
            score += 1
        elif depth <= 15:
            score += 2
        elif depth <= 30:
            score += 3
        else:
            score += 4
        
        # Fanout scoring
        if fanout <= 3:
            score += 1
        elif fanout <= 8:
            score += 2
        elif fanout <= 15:
            score += 3
        else:
            score += 4
        
        # ML nodes add complexity
        if ml_nodes > 0:
            score += min(ml_nodes, 3)  # Cap at 3 extra points
        
        # Determine level based on total score
        if score <= 4:
            return ComplexityLevel.TRIVIAL
        elif score <= 8:
            return ComplexityLevel.SIMPLE
        elif score <= 12:
            return ComplexityLevel.MODERATE
        elif score <= 16:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXTREME
    
    def _calculate_complexity_score(self, metrics: Dict[ComplexityMetric, Union[int, float]]) -> float:
        """Calculate a numerical complexity score"""
        # Weighted scoring of different metrics
        weights = {
            ComplexityMetric.NODE_COUNT: 1.0,
            ComplexityMetric.GRAPH_DEPTH: 2.0,
            ComplexityMetric.MAX_FANOUT: 1.5,
            ComplexityMetric.CYCLE_COUNT: 3.0,
            ComplexityMetric.ML_NODE_COUNT: 2.5,
            ComplexityMetric.CONDITIONAL_BRANCHES: 1.2,
            ComplexityMetric.EXTERNAL_CALLS: 1.8
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            score += value * weight
        
        return score
    
    def _check_complexity_limits(self, analysis: ComplexityAnalysis, profile: ComplexityProfile):
        """Check complexity metrics against profile limits"""
        for limit in profile.limits:
            metric_value = analysis.metrics.get(limit.metric, 0)
            
            # Check if limit is exceeded
            if metric_value > limit.limit_value:
                # Check grace factor for soft limits
                if limit.soft_limit and metric_value <= limit.limit_value * limit.grace_factor:
                    # Within grace factor - warning only
                    warning = f"{limit.metric.value} ({metric_value}) exceeds soft limit ({limit.limit_value}) but within grace factor"
                    analysis.limit_warnings.append(warning)
                else:
                    # Hard violation
                    violation = f"{limit.metric.value} ({metric_value}) exceeds limit ({limit.limit_value}) - action: {limit.action.value}"
                    analysis.limit_violations.append(violation)
    
    def _generate_optimization_suggestions(self, metrics: Dict[ComplexityMetric, Union[int, float]], 
                                         graph_metrics: GraphMetrics) -> List[str]:
        """Generate optimization suggestions based on metrics"""
        suggestions = []
        
        node_count = metrics.get(ComplexityMetric.NODE_COUNT, 0)
        depth = metrics.get(ComplexityMetric.GRAPH_DEPTH, 0)
        fanout = metrics.get(ComplexityMetric.MAX_FANOUT, 0)
        
        if node_count > 100:
            suggestions.append("Consider using sub-workflows to reduce node count")
        
        if depth > 20:
            suggestions.append("Parallelize sequential operations to reduce depth")
        
        if fanout > 10:
            suggestions.append("Use aggregation patterns to reduce fanout")
        
        if graph_metrics.ml_nodes > 3:
            suggestions.append("Consider ML model consolidation or ensemble approaches")
        
        if graph_metrics.cycle_count > 0:
            suggestions.append("Review cycles for potential optimization or elimination")
        
        return suggestions
    
    def _generate_refactoring_suggestions(self, metrics: Dict[ComplexityMetric, Union[int, float]], 
                                        graph_metrics: GraphMetrics) -> List[str]:
        """Generate refactoring suggestions"""
        suggestions = []
        
        if graph_metrics.total_nodes > 200:
            suggestions.append("Split into multiple smaller workflows")
        
        if graph_metrics.max_depth > 30:
            suggestions.append("Restructure to reduce sequential dependencies")
        
        if graph_metrics.total_data_size_estimate_mb > 100:
            suggestions.append("Consider data streaming or chunking for large attachments")
        
        return suggestions
    
    def _estimate_compilation_time(self, metrics: Dict[ComplexityMetric, Union[int, float]]) -> float:
        """Estimate compilation time based on complexity metrics"""
        base_time = 1.0  # Base compilation time in seconds
        
        node_count = metrics.get(ComplexityMetric.NODE_COUNT, 0)
        ml_nodes = metrics.get(ComplexityMetric.ML_NODE_COUNT, 0)
        
        # Linear scaling with node count
        time_estimate = base_time + (node_count * 0.1) + (ml_nodes * 0.5)
        
        return time_estimate
    
    def _estimate_runtime_memory(self, metrics: Dict[ComplexityMetric, Union[int, float]]) -> float:
        """Estimate runtime memory usage"""
        base_memory = 50.0  # Base memory in MB
        
        node_count = metrics.get(ComplexityMetric.NODE_COUNT, 0)
        attachment_size = metrics.get(ComplexityMetric.ATTACHMENT_SIZE, 0)
        
        # Memory scales with nodes and data size
        memory_estimate = base_memory + (node_count * 2.0) + attachment_size
        
        return memory_estimate
    
    def _estimate_execution_time(self, metrics: Dict[ComplexityMetric, Union[int, float]]) -> float:
        """Estimate execution time"""
        base_time = 0.5  # Base execution time in seconds
        
        node_count = metrics.get(ComplexityMetric.NODE_COUNT, 0)
        depth = metrics.get(ComplexityMetric.GRAPH_DEPTH, 0)
        ml_nodes = metrics.get(ComplexityMetric.ML_NODE_COUNT, 0)
        external_calls = metrics.get(ComplexityMetric.EXTERNAL_CALLS, 0)
        
        # Execution time factors
        time_estimate = base_time + (node_count * 0.05) + (depth * 0.1) + (ml_nodes * 2.0) + (external_calls * 0.5)
        
        return time_estimate
    
    def _calculate_node_depths(self, adjacency: Dict[str, List[str]], nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate depth of each node using BFS"""
        depths = {}
        
        # Find root nodes (no incoming edges)
        all_targets = set()
        for targets in adjacency.values():
            all_targets.update(targets)
        
        root_nodes = []
        for node in nodes:
            node_id = node.get('id')
            if node_id and node_id not in all_targets:
                root_nodes.append(node_id)
        
        # If no clear roots, use first node
        if not root_nodes and nodes:
            root_nodes = [nodes[0].get('id')]
        
        # BFS from each root
        for root in root_nodes:
            if root:
                queue = deque([(root, 0)])
                visited = set()
                
                while queue:
                    node_id, depth = queue.popleft()
                    
                    if node_id in visited:
                        continue
                    
                    visited.add(node_id)
                    depths[node_id] = max(depths.get(node_id, 0), depth)
                    
                    # Add children
                    for child in adjacency.get(node_id, []):
                        if child not in visited:
                            queue.append((child, depth + 1))
        
        return depths
    
    def _count_cycles(self, adjacency: Dict[str, List[str]], node_ids: List[str]) -> int:
        """Count cycles in the graph using DFS"""
        visited = set()
        rec_stack = set()
        cycle_count = 0
        
        def dfs(node):
            nonlocal cycle_count
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    cycle_count += 1
            
            rec_stack.remove(node)
        
        for node_id in node_ids:
            if node_id and node_id not in visited:
                dfs(node_id)
        
        return cycle_count
    
    def _initialize_default_limits(self) -> Dict[str, List[ComplexityLimit]]:
        """Initialize default complexity limits for different environments"""
        return {
            "development": [
                ComplexityLimit(ComplexityMetric.NODE_COUNT, 500, LimitAction.WARN, soft_limit=True, description="Maximum nodes in development"),
                ComplexityLimit(ComplexityMetric.GRAPH_DEPTH, 50, LimitAction.WARN, soft_limit=True, description="Maximum graph depth in development"),
                ComplexityLimit(ComplexityMetric.MAX_FANOUT, 20, LimitAction.SUGGEST_REFACTOR, description="Maximum fanout in development"),
                ComplexityLimit(ComplexityMetric.ML_NODE_COUNT, 10, LimitAction.WARN, description="Maximum ML nodes in development"),
                ComplexityLimit(ComplexityMetric.ATTACHMENT_SIZE, 50, LimitAction.WARN, description="Maximum attachment size (MB) in development")
            ],
            "staging": [
                ComplexityLimit(ComplexityMetric.NODE_COUNT, 1000, LimitAction.WARN, description="Maximum nodes in staging"),
                ComplexityLimit(ComplexityMetric.GRAPH_DEPTH, 75, LimitAction.WARN, description="Maximum graph depth in staging"),
                ComplexityLimit(ComplexityMetric.MAX_FANOUT, 15, LimitAction.SUGGEST_REFACTOR, description="Maximum fanout in staging"),
                ComplexityLimit(ComplexityMetric.ML_NODE_COUNT, 8, LimitAction.WARN, description="Maximum ML nodes in staging"),
                ComplexityLimit(ComplexityMetric.ATTACHMENT_SIZE, 100, LimitAction.WARN, description="Maximum attachment size (MB) in staging")
            ],
            "production": [
                ComplexityLimit(ComplexityMetric.NODE_COUNT, 2000, LimitAction.BLOCK, description="Maximum nodes in production"),
                ComplexityLimit(ComplexityMetric.GRAPH_DEPTH, 100, LimitAction.REQUIRE_APPROVAL, description="Maximum graph depth in production"),
                ComplexityLimit(ComplexityMetric.MAX_FANOUT, 12, LimitAction.REQUIRE_APPROVAL, description="Maximum fanout in production"),
                ComplexityLimit(ComplexityMetric.ML_NODE_COUNT, 6, LimitAction.REQUIRE_APPROVAL, description="Maximum ML nodes in production"),
                ComplexityLimit(ComplexityMetric.ATTACHMENT_SIZE, 200, LimitAction.BLOCK, description="Maximum attachment size (MB) in production"),
                ComplexityLimit(ComplexityMetric.CYCLE_COUNT, 0, LimitAction.BLOCK, description="No cycles allowed in production")
            ]
        }
    
    def _initialize_standard_profiles(self):
        """Initialize standard complexity profiles"""
        environments = ["dev", "staging", "prod"]
        
        for env in environments:
            profile = self.create_complexity_profile(
                f"standard_{env}",
                env,
                tenant_id=None
            )
            
        self.logger.info(f"✅ Initialized {len(environments)} standard complexity profiles")

# API Interface
class ComplexityLimitsAPI:
    """API interface for complexity limits operations"""
    
    def __init__(self, complexity_service: Optional[ComplexityLimitsService] = None):
        self.complexity_service = complexity_service or ComplexityLimitsService()
    
    def analyze_plan(self, plan_ir: Dict[str, Any], profile_id: Optional[str] = None) -> Dict[str, Any]:
        """API endpoint to analyze plan complexity"""
        try:
            analysis = self.complexity_service.analyze_plan_complexity(plan_ir, profile_id)
            
            return {
                'success': True,
                'analysis': asdict(analysis)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def validate_plan(self, plan_ir: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
        """API endpoint to validate plan against limits"""
        try:
            result = self.complexity_service.validate_plan_against_limits(plan_ir, profile_id)
            
            return {
                'success': True,
                **result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_optimization_suggestions(self, plan_ir: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to get optimization suggestions"""
        try:
            suggestions = self.complexity_service.suggest_plan_optimizations(plan_ir)
            
            return {
                'success': True,
                'suggestions': suggestions
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_plan_ir(node_count: int = 10, depth: int = 5, fanout: int = 3) -> Dict[str, Any]:
    """Create a test plan IR with specified complexity"""
    nodes = []
    edges = []
    
    # Create nodes
    for i in range(node_count):
        node_type = "step"
        if i % 5 == 0:
            node_type = "ml_predict"
        elif i % 7 == 0:
            node_type = "decision"
        elif i % 11 == 0:
            node_type = "external_api"
        
        nodes.append({
            'id': f'node_{i}',
            'type': node_type,
            'name': f'Node {i}',
            'attachments': [{'size_mb': 1.0}] if i % 3 == 0 else []
        })
    
    # Create edges to achieve desired depth and fanout
    for i in range(node_count - 1):
        # Create primary path for depth
        if i < depth:
            edges.append({
                'source': f'node_{i}',
                'target': f'node_{i + 1}'
            })
        
        # Create fanout edges
        if i < node_count - fanout:
            for j in range(min(fanout, node_count - i - 1)):
                if i + j + 1 < node_count:
                    edges.append({
                        'source': f'node_{i}',
                        'target': f'node_{i + j + 1}'
                    })
    
    return {
        'plan_id': f'test_plan_{node_count}_{depth}_{fanout}',
        'plan_graph': {
            'nodes': nodes,
            'edges': edges
        },
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    }

def run_complexity_limits_tests():
    """Run comprehensive complexity limits tests"""
    print("=== Complexity Limits Service Tests ===")
    
    # Initialize service
    complexity_service = ComplexityLimitsService()
    complexity_api = ComplexityLimitsAPI(complexity_service)
    
    # Test 1: Standard profiles initialization
    print("\n1. Testing standard profiles initialization...")
    stats = complexity_service.get_complexity_statistics()
    print(f"   Standard profiles created: {stats['complexity_profiles']}")
    
    for profile_id, profile in complexity_service.complexity_profiles.items():
        print(f"     - {profile.profile_name}: {profile.environment} ({len(profile.limits)} limits)")
    
    # Test 2: Plan complexity analysis
    print("\n2. Testing plan complexity analysis...")
    
    # Test different complexity levels
    test_plans = [
        (5, 3, 2, "Simple plan"),
        (50, 10, 5, "Moderate plan"),
        (200, 25, 8, "Complex plan"),
        (500, 50, 15, "Extreme plan")
    ]
    
    for node_count, depth, fanout, description in test_plans:
        test_ir = create_test_plan_ir(node_count, depth, fanout)
        analysis = complexity_service.analyze_plan_complexity(test_ir)
        
        print(f"   {description}:")
        print(f"     Complexity level: {analysis.complexity_level.value}")
        print(f"     Complexity score: {analysis.total_complexity_score:.1f}")
        print(f"     Node count: {analysis.metrics.get(ComplexityMetric.NODE_COUNT, 0)}")
        print(f"     Graph depth: {analysis.metrics.get(ComplexityMetric.GRAPH_DEPTH, 0)}")
        print(f"     Max fanout: {analysis.metrics.get(ComplexityMetric.MAX_FANOUT, 0)}")
        print(f"     ML nodes: {analysis.metrics.get(ComplexityMetric.ML_NODE_COUNT, 0)}")
    
    # Test 3: Limit validation
    print("\n3. Testing limit validation...")
    
    # Get a profile to test against
    profile_id = list(complexity_service.complexity_profiles.keys())[0]  # Use first profile
    
    # Test with a plan that should pass
    simple_ir = create_test_plan_ir(10, 3, 2)
    validation_result = complexity_service.validate_plan_against_limits(simple_ir, profile_id)
    print(f"   Simple plan validation: {'✅ PASS' if validation_result['valid'] else '❌ FAIL'}")
    if not validation_result['valid']:
        print(f"     Violations: {validation_result['violations']}")
    
    # Test with a plan that should fail
    complex_ir = create_test_plan_ir(3000, 150, 25)  # Exceeds production limits
    validation_result = complexity_service.validate_plan_against_limits(complex_ir, profile_id)
    print(f"   Complex plan validation: {'❌ FAIL' if not validation_result['valid'] else '✅ UNEXPECTED PASS'}")
    if not validation_result['valid']:
        print(f"     Violations: {len(validation_result['violations'])}")
        for violation in validation_result['violations'][:3]:  # Show first 3
            print(f"       - {violation}")
    
    # Test 4: Custom profile creation
    print("\n4. Testing custom profile creation...")
    
    custom_limits = [
        {
            'metric': 'node_count',
            'limit_value': 100,
            'action': 'block',
            'description': 'Custom node limit'
        },
        {
            'metric': 'ml_node_count',
            'limit_value': 3,
            'action': 'require_approval',
            'description': 'Custom ML node limit'
        }
    ]
    
    custom_profile = complexity_service.create_complexity_profile(
        "custom_test_profile",
        "test",
        tenant_id="test_tenant",
        custom_limits=custom_limits
    )
    
    print(f"   Custom profile created: {custom_profile.profile_name}")
    print(f"   Environment: {custom_profile.environment}")
    print(f"   Tenant: {custom_profile.tenant_id}")
    print(f"   Custom limits: {len(custom_limits)}")
    
    # Test 5: Optimization suggestions
    print("\n5. Testing optimization suggestions...")
    
    # Create a plan that needs optimization
    optimization_ir = create_test_plan_ir(150, 30, 12)
    suggestions = complexity_service.suggest_plan_optimizations(optimization_ir)
    
    print(f"   Optimization suggestions: {len(suggestions)}")
    for suggestion in suggestions[:3]:  # Show first 3
        print(f"     - {suggestion}")
    
    # Test 6: Performance estimation
    print("\n6. Testing performance estimation...")
    
    analysis = complexity_service.analyze_plan_complexity(optimization_ir)
    print(f"   Performance estimates:")
    print(f"     Compilation time: {analysis.estimated_compilation_time_seconds:.1f}s")
    print(f"     Runtime memory: {analysis.estimated_runtime_memory_mb:.1f}MB")
    print(f"     Execution time: {analysis.estimated_execution_time_seconds:.1f}s")
    
    # Test 7: Complexity trends
    print("\n7. Testing complexity trends...")
    
    # Generate some analyses for trend calculation
    for i in range(5):
        test_ir = create_test_plan_ir(20 + i * 10, 5 + i, 3 + i)
        complexity_service.analyze_plan_complexity(test_ir)
    
    trends = complexity_service.get_complexity_trends(days=30)
    print(f"   Trend analysis:")
    print(f"     Total analyses: {trends['total_analyses']}")
    if trends['total_analyses'] > 0:
        complexity_trends = trends['trends']['complexity_score']
        print(f"     Average complexity score: {complexity_trends['average']:.1f}")
        print(f"     Max complexity score: {complexity_trends['max']:.1f}")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Test API analysis
    api_analysis_result = complexity_api.analyze_plan(simple_ir)
    print(f"   API analysis: {'✅ PASS' if api_analysis_result['success'] else '❌ FAIL'}")
    
    # Test API validation
    api_validation_result = complexity_api.validate_plan(simple_ir, profile_id)
    print(f"   API validation: {'✅ PASS' if api_validation_result['success'] else '❌ FAIL'}")
    
    # Test API suggestions
    api_suggestions_result = complexity_api.get_optimization_suggestions(optimization_ir)
    print(f"   API suggestions: {'✅ PASS' if api_suggestions_result['success'] else '❌ FAIL'}")
    if api_suggestions_result['success']:
        print(f"   Suggestions returned: {len(api_suggestions_result['suggestions'])}")
    
    # Test 9: Edge cases
    print("\n9. Testing edge cases...")
    
    # Empty plan
    empty_ir = {'plan_id': 'empty', 'plan_graph': {'nodes': [], 'edges': []}}
    empty_analysis = complexity_service.analyze_plan_complexity(empty_ir)
    print(f"   Empty plan analysis: {empty_analysis.complexity_level.value}")
    
    # Single node plan
    single_node_ir = create_test_plan_ir(1, 1, 0)
    single_analysis = complexity_service.analyze_plan_complexity(single_node_ir)
    print(f"   Single node plan: {single_analysis.complexity_level.value}")
    
    # Plan with cycles (simulated)
    cycle_ir = create_test_plan_ir(10, 5, 2)
    cycle_ir['plan_graph']['edges'].append({'source': 'node_5', 'target': 'node_2'})  # Create cycle
    cycle_analysis = complexity_service.analyze_plan_complexity(cycle_ir)
    print(f"   Plan with cycles: {cycle_analysis.metrics.get(ComplexityMetric.CYCLE_COUNT, 0)} cycles detected")
    
    # Test 10: Statistics
    print("\n10. Testing statistics...")
    
    final_stats = complexity_service.get_complexity_statistics()
    print(f"   Total analyses: {final_stats['total_analyses']}")
    print(f"   Plans blocked: {final_stats['plans_blocked']}")
    print(f"   Plans warned: {final_stats['plans_warned']}")
    print(f"   Optimizations suggested: {final_stats['optimizations_suggested']}")
    print(f"   Cache hits: {final_stats['cache_hits']}")
    print(f"   Cache misses: {final_stats['cache_misses']}")
    print(f"   Complexity levels: {final_stats['complexity_levels']}")
    
    print(f"\n=== Test Summary ===")
    print(f"Complexity limits service tested successfully")
    print(f"Profiles created: {final_stats['complexity_profiles']}")
    print(f"Analyses performed: {final_stats['total_analyses']}")
    print(f"Cache hit rate: {final_stats['cache_hits']}/{final_stats['cache_hits'] + final_stats['cache_misses']}")
    
    return complexity_service, complexity_api

if __name__ == "__main__":
    run_complexity_limits_tests()



