"""
Task 6.3.62: Capacity planner (queues → nodes recommendation)
Provide capacity planner with queue analysis and node recommendations
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Resource types for capacity planning"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"

@dataclass
class QueueMetrics:
    """Queue performance metrics"""
    queue_name: str
    avg_depth: float
    max_depth: int
    avg_wait_time_ms: float
    max_wait_time_ms: int
    throughput_per_second: float
    error_rate: float
    timestamp: datetime

@dataclass
class NodeMetrics:
    """Node resource utilization metrics"""
    node_id: str
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    storage_utilization: float
    active_requests: int
    timestamp: datetime

@dataclass
class CapacityRecommendation:
    """Capacity planning recommendation"""
    recommendation_id: str
    queue_name: str
    current_nodes: int
    recommended_nodes: int
    resource_type: ResourceType
    confidence: float
    reasoning: str
    estimated_improvement: Dict[str, float]
    cost_impact: float
    priority: str
    generated_at: datetime

class CapacityPlanner:
    """
    Capacity planning service for queue-based node recommendations
    Task 6.3.62: Predictive scaling with weekly reports
    """
    
    def __init__(self):
        self.queue_metrics_history: Dict[str, List[QueueMetrics]] = {}
        self.node_metrics_history: Dict[str, List[NodeMetrics]] = {}
        self.recommendations: Dict[str, List[CapacityRecommendation]] = {}
        self.planning_window_days = 7
        self.recommendation_threshold = 0.7  # Confidence threshold
    
    def record_queue_metrics(self, metrics: QueueMetrics) -> None:
        """Record queue performance metrics"""
        if metrics.queue_name not in self.queue_metrics_history:
            self.queue_metrics_history[metrics.queue_name] = []
        
        self.queue_metrics_history[metrics.queue_name].append(metrics)
        
        # Keep only recent metrics
        cutoff_time = datetime.utcnow() - timedelta(days=self.planning_window_days)
        self.queue_metrics_history[metrics.queue_name] = [
            m for m in self.queue_metrics_history[metrics.queue_name]
            if m.timestamp > cutoff_time
        ]
    
    def record_node_metrics(self, metrics: NodeMetrics) -> None:
        """Record node resource utilization metrics"""
        if metrics.node_id not in self.node_metrics_history:
            self.node_metrics_history[metrics.node_id] = []
        
        self.node_metrics_history[metrics.node_id].append(metrics)
        
        # Keep only recent metrics
        cutoff_time = datetime.utcnow() - timedelta(days=self.planning_window_days)
        self.node_metrics_history[metrics.node_id] = [
            m for m in self.node_metrics_history[metrics.node_id]
            if m.timestamp > cutoff_time
        ]
    
    def generate_capacity_recommendations(self, queue_name: str) -> List[CapacityRecommendation]:
        """Generate capacity recommendations for a queue"""
        if queue_name not in self.queue_metrics_history:
            logger.warning(f"No metrics found for queue: {queue_name}")
            return []
        
        queue_metrics = self.queue_metrics_history[queue_name]
        if len(queue_metrics) < 10:  # Need sufficient data
            logger.warning(f"Insufficient metrics for queue {queue_name}: {len(queue_metrics)} samples")
            return []
        
        recommendations = []
        
        # Analyze different resource types
        for resource_type in ResourceType:
            recommendation = self._analyze_resource_capacity(queue_name, resource_type, queue_metrics)
            if recommendation and recommendation.confidence >= self.recommendation_threshold:
                recommendations.append(recommendation)
        
        # Store recommendations
        if queue_name not in self.recommendations:
            self.recommendations[queue_name] = []
        
        self.recommendations[queue_name].extend(recommendations)
        
        logger.info(f"Generated {len(recommendations)} capacity recommendations for {queue_name}")
        return recommendations
    
    def _analyze_resource_capacity(
        self,
        queue_name: str,
        resource_type: ResourceType,
        queue_metrics: List[QueueMetrics]
    ) -> Optional[CapacityRecommendation]:
        """Analyze capacity requirements for specific resource type"""
        
        # Calculate queue performance trends
        recent_metrics = queue_metrics[-24:]  # Last 24 samples
        
        avg_depth = statistics.mean(m.avg_depth for m in recent_metrics)
        avg_wait_time = statistics.mean(m.avg_wait_time_ms for m in recent_metrics)
        avg_throughput = statistics.mean(m.throughput_per_second for m in recent_metrics)
        avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
        
        # Get current node utilization
        current_nodes = self._get_current_node_count(queue_name)
        node_utilization = self._get_average_node_utilization(resource_type)
        
        # Determine if scaling is needed
        scaling_decision = self._determine_scaling_need(
            avg_depth, avg_wait_time, avg_throughput, avg_error_rate, node_utilization
        )
        
        if scaling_decision["action"] == "no_change":
            return None
        
        # Calculate recommended nodes
        recommended_nodes = self._calculate_recommended_nodes(
            current_nodes, scaling_decision, resource_type
        )
        
        # Calculate confidence based on data quality and consistency
        confidence = self._calculate_recommendation_confidence(queue_metrics, scaling_decision)
        
        # Estimate improvement
        estimated_improvement = self._estimate_performance_improvement(
            scaling_decision, current_nodes, recommended_nodes
        )
        
        # Calculate cost impact
        cost_impact = self._calculate_cost_impact(current_nodes, recommended_nodes, resource_type)
        
        return CapacityRecommendation(
            recommendation_id=f"{queue_name}_{resource_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            queue_name=queue_name,
            current_nodes=current_nodes,
            recommended_nodes=recommended_nodes,
            resource_type=resource_type,
            confidence=confidence,
            reasoning=scaling_decision["reasoning"],
            estimated_improvement=estimated_improvement,
            cost_impact=cost_impact,
            priority=scaling_decision["priority"],
            generated_at=datetime.utcnow()
        )
    
    def _determine_scaling_need(
        self,
        avg_depth: float,
        avg_wait_time: float,
        avg_throughput: float,
        avg_error_rate: float,
        node_utilization: float
    ) -> Dict[str, Any]:
        """Determine if scaling is needed and in which direction"""
        
        # Define thresholds
        high_depth_threshold = 50
        high_wait_time_threshold = 5000  # 5 seconds
        low_throughput_threshold = 10
        high_error_rate_threshold = 0.05  # 5%
        high_utilization_threshold = 0.8  # 80%
        low_utilization_threshold = 0.3   # 30%
        
        reasons = []
        priority = "medium"
        
        # Check for scale-up conditions
        if avg_depth > high_depth_threshold:
            reasons.append(f"High queue depth: {avg_depth:.1f}")
            priority = "high"
        
        if avg_wait_time > high_wait_time_threshold:
            reasons.append(f"High wait time: {avg_wait_time:.0f}ms")
            priority = "high"
        
        if avg_throughput < low_throughput_threshold:
            reasons.append(f"Low throughput: {avg_throughput:.1f}/s")
        
        if avg_error_rate > high_error_rate_threshold:
            reasons.append(f"High error rate: {avg_error_rate:.2%}")
            priority = "high"
        
        if node_utilization > high_utilization_threshold:
            reasons.append(f"High resource utilization: {node_utilization:.1%}")
        
        # Scale up if any concerning metrics
        if reasons:
            return {
                "action": "scale_up",
                "reasoning": "; ".join(reasons),
                "priority": priority,
                "scale_factor": self._calculate_scale_factor(avg_depth, avg_wait_time, node_utilization)
            }
        
        # Check for scale-down conditions
        if (node_utilization < low_utilization_threshold and 
            avg_depth < 10 and 
            avg_wait_time < 1000 and 
            avg_error_rate < 0.01):
            return {
                "action": "scale_down",
                "reasoning": f"Low utilization ({node_utilization:.1%}) with good performance metrics",
                "priority": "low",
                "scale_factor": 0.8  # Reduce by 20%
            }
        
        return {
            "action": "no_change",
            "reasoning": "Metrics within acceptable ranges",
            "priority": "none",
            "scale_factor": 1.0
        }
    
    def _calculate_scale_factor(self, avg_depth: float, avg_wait_time: float, utilization: float) -> float:
        """Calculate how much to scale based on metrics"""
        base_factor = 1.2  # 20% increase minimum
        
        # Increase factor based on severity
        if avg_depth > 100:
            base_factor += 0.3
        elif avg_depth > 50:
            base_factor += 0.1
        
        if avg_wait_time > 10000:  # 10 seconds
            base_factor += 0.3
        elif avg_wait_time > 5000:  # 5 seconds
            base_factor += 0.1
        
        if utilization > 0.9:
            base_factor += 0.2
        elif utilization > 0.8:
            base_factor += 0.1
        
        return min(base_factor, 2.0)  # Cap at 2x scaling
    
    def _calculate_recommended_nodes(
        self,
        current_nodes: int,
        scaling_decision: Dict[str, Any],
        resource_type: ResourceType
    ) -> int:
        """Calculate recommended number of nodes"""
        if scaling_decision["action"] == "scale_up":
            recommended = int(current_nodes * scaling_decision["scale_factor"])
            return max(recommended, current_nodes + 1)  # At least +1 node
        elif scaling_decision["action"] == "scale_down":
            recommended = int(current_nodes * scaling_decision["scale_factor"])
            return max(recommended, 1)  # At least 1 node
        else:
            return current_nodes
    
    def _calculate_recommendation_confidence(
        self,
        queue_metrics: List[QueueMetrics],
        scaling_decision: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the recommendation"""
        base_confidence = 0.5
        
        # More data points increase confidence
        data_points = len(queue_metrics)
        if data_points > 100:
            base_confidence += 0.3
        elif data_points > 50:
            base_confidence += 0.2
        elif data_points > 20:
            base_confidence += 0.1
        
        # Consistent trends increase confidence
        recent_depths = [m.avg_depth for m in queue_metrics[-10:]]
        if len(set(recent_depths)) < len(recent_depths) * 0.3:  # Low variance
            base_confidence += 0.1
        
        # Priority affects confidence
        if scaling_decision["priority"] == "high":
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _estimate_performance_improvement(
        self,
        scaling_decision: Dict[str, Any],
        current_nodes: int,
        recommended_nodes: int
    ) -> Dict[str, float]:
        """Estimate performance improvement from scaling"""
        if scaling_decision["action"] == "no_change":
            return {}
        
        node_ratio = recommended_nodes / current_nodes if current_nodes > 0 else 1.0
        
        if scaling_decision["action"] == "scale_up":
            return {
                "queue_depth_reduction": min(0.4, (node_ratio - 1) * 0.5),
                "wait_time_reduction": min(0.5, (node_ratio - 1) * 0.6),
                "throughput_increase": min(0.3, (node_ratio - 1) * 0.4),
                "error_rate_reduction": min(0.2, (node_ratio - 1) * 0.3)
            }
        else:  # scale_down
            return {
                "cost_reduction": (1 - node_ratio) * 0.8,
                "resource_efficiency": 0.2
            }
    
    def _calculate_cost_impact(self, current_nodes: int, recommended_nodes: int, resource_type: ResourceType) -> float:
        """Calculate cost impact of scaling recommendation"""
        # Simplified cost calculation (would integrate with actual cost data)
        cost_per_node = {
            ResourceType.CPU: 100,    # $100/month per CPU node
            ResourceType.MEMORY: 80,  # $80/month per memory node
            ResourceType.GPU: 500,    # $500/month per GPU node
            ResourceType.STORAGE: 50  # $50/month per storage node
        }
        
        node_cost = cost_per_node.get(resource_type, 100)
        cost_change = (recommended_nodes - current_nodes) * node_cost
        
        return cost_change
    
    def _get_current_node_count(self, queue_name: str) -> int:
        """Get current number of nodes serving the queue"""
        # Simplified - would integrate with actual cluster state
        return 3  # Default assumption
    
    def _get_average_node_utilization(self, resource_type: ResourceType) -> float:
        """Get average utilization for resource type across all nodes"""
        if not self.node_metrics_history:
            return 0.5  # Default assumption
        
        all_metrics = []
        for node_metrics in self.node_metrics_history.values():
            all_metrics.extend(node_metrics)
        
        if not all_metrics:
            return 0.5
        
        recent_metrics = all_metrics[-50:]  # Last 50 samples
        
        if resource_type == ResourceType.CPU:
            return statistics.mean(m.cpu_utilization for m in recent_metrics)
        elif resource_type == ResourceType.MEMORY:
            return statistics.mean(m.memory_utilization for m in recent_metrics)
        elif resource_type == ResourceType.GPU:
            return statistics.mean(m.gpu_utilization for m in recent_metrics)
        elif resource_type == ResourceType.STORAGE:
            return statistics.mean(m.storage_utilization for m in recent_metrics)
        
        return 0.5
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly capacity planning report"""
        report = {
            "report_date": datetime.utcnow().isoformat(),
            "planning_period": f"Last {self.planning_window_days} days",
            "queues_analyzed": len(self.queue_metrics_history),
            "total_recommendations": sum(len(recs) for recs in self.recommendations.values()),
            "queue_summaries": {},
            "cost_impact_summary": 0.0,
            "priority_breakdown": {"high": 0, "medium": 0, "low": 0}
        }
        
        for queue_name, recommendations in self.recommendations.items():
            if not recommendations:
                continue
            
            recent_recs = [r for r in recommendations if r.generated_at > datetime.utcnow() - timedelta(days=7)]
            
            if recent_recs:
                total_cost_impact = sum(r.cost_impact for r in recent_recs)
                avg_confidence = statistics.mean(r.confidence for r in recent_recs)
                
                report["queue_summaries"][queue_name] = {
                    "recommendations_count": len(recent_recs),
                    "avg_confidence": avg_confidence,
                    "total_cost_impact": total_cost_impact,
                    "resource_types": list(set(r.resource_type.value for r in recent_recs))
                }
                
                report["cost_impact_summary"] += total_cost_impact
                
                for rec in recent_recs:
                    report["priority_breakdown"][rec.priority] += 1
        
        return report
    
    def get_recommendations_for_queue(self, queue_name: str) -> List[CapacityRecommendation]:
        """Get all recommendations for a specific queue"""
        return self.recommendations.get(queue_name, [])

# Global capacity planner instance
capacity_planner = CapacityPlanner()
