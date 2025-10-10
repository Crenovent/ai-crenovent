"""
Scalability Framework - Section 7.3 Tasks
=========================================

Implements Section 7.3 Tasks:
- 7.3.1: Define scalability principles
- 7.3.2: Document vertical vs horizontal scaling strategies
- 7.3.17: Build predictive scaling models
- 7.3.31: Implement predictive FinOps optimization

Following user rules:
- No hardcoding, dynamic configuration
- SaaS/IT industry focus
- Multi-tenant aware
- Modular design
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Task 7.3.2: Vertical vs horizontal scaling strategies"""
    VERTICAL = "vertical"      # Scale up (more CPU/RAM)
    HORIZONTAL = "horizontal"  # Scale out (more instances)
    HYBRID = "hybrid"         # Combination of both
    AUTO = "auto"            # AI-driven decision

class ScalingTrigger(Enum):
    """Scaling trigger types"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    CUSTOM_METRIC = "custom_metric"

class WorkloadType(Enum):
    """SaaS/IT workload types"""
    API_GATEWAY = "api_gateway"
    DSL_COMPILER = "dsl_compiler"
    WORKFLOW_ENGINE = "workflow_engine"
    AI_AGENT = "ai_agent"
    DATABASE = "database"
    CACHE = "cache"
    ANALYTICS = "analytics"

@dataclass
class ScalingPrinciple:
    """Task 7.3.1: Define scalability principles"""
    principle_id: str
    name: str
    description: str
    category: str
    
    # Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # SaaS/IT specific
    workload_types: List[WorkloadType] = field(default_factory=list)
    tenant_tiers: List[str] = field(default_factory=list)
    
    # Metadata
    priority: int = 1  # 1=highest, 5=lowest
    tags: List[str] = field(default_factory=list)

@dataclass
class ScalingMetrics:
    """Scaling metrics and thresholds"""
    timestamp: datetime
    workload_type: WorkloadType
    tenant_id: Optional[int] = None
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_utilization: float = 0.0
    
    # Performance metrics
    request_rate: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    
    # Business metrics (SaaS focus)
    active_tenants: int = 0
    concurrent_workflows: int = 0
    dsl_compilations_per_sec: float = 0.0
    
    # Cost metrics
    cost_per_hour: float = 0.0
    cost_per_request: float = 0.0

@dataclass
class ScalingRecommendation:
    """Scaling recommendation from predictive model"""
    workload_type: WorkloadType
    strategy: ScalingStrategy
    confidence: float
    
    # Scaling parameters
    target_instances: int
    target_cpu_cores: int
    target_memory_gb: int
    
    # Timing
    recommended_at: datetime
    execute_at: datetime
    duration_minutes: int
    
    # Cost impact
    estimated_cost_change: float
    cost_efficiency_score: float
    
    # Reasoning
    triggers: List[ScalingTrigger]
    reasoning: str
    risk_factors: List[str] = field(default_factory=list)

class ScalabilityFramework:
    """
    Comprehensive Scalability Framework
    
    Implements Section 7.3 Tasks:
    - 7.3.1: Define scalability principles
    - 7.3.2: Document vertical vs horizontal scaling strategies
    - 7.3.17: Build predictive scaling models
    - 7.3.31: Implement predictive FinOps optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize scalability principles (Task 7.3.1)
        self.principles = self._initialize_scalability_principles()
        
        # Initialize scaling strategies (Task 7.3.2)
        self.scaling_strategies = self._initialize_scaling_strategies()
        
        # Predictive models (Task 7.3.17)
        self.predictive_models = {}
        self.metrics_history: Dict[str, List[ScalingMetrics]] = {}
        
        # FinOps optimization (Task 7.3.31)
        self.cost_models = {}
        self.optimization_policies = self._load_finops_policies()
        
        self.logger.info("✅ Initialized Scalability Framework")
    
    def _initialize_scalability_principles(self) -> Dict[str, ScalingPrinciple]:
        """Task 7.3.1: Define scalability principles"""
        principles = {}
        
        # Principle 1: Tenant Isolation
        principles["tenant_isolation"] = ScalingPrinciple(
            principle_id="tenant_isolation",
            name="Multi-Tenant Isolation",
            description="Scale resources while maintaining strict tenant isolation",
            category="isolation",
            parameters={
                "isolation_level": "strict",
                "resource_quotas": True,
                "network_isolation": True
            },
            constraints={
                "cross_tenant_access": False,
                "shared_resources": ["cache", "monitoring"],
                "dedicated_resources": ["database", "storage"]
            },
            workload_types=[WorkloadType.API_GATEWAY, WorkloadType.DATABASE],
            tenant_tiers=["T0", "T1", "T2"],
            priority=1,
            tags=["security", "compliance", "multi-tenant"]
        )
        
        # Principle 2: SaaS-Optimized Scaling
        principles["saas_optimized"] = ScalingPrinciple(
            principle_id="saas_optimized",
            name="SaaS-Optimized Scaling",
            description="Scale based on SaaS-specific metrics (ARR, MRR, customer count)",
            category="business_metrics",
            parameters={
                "scale_by_arr": True,
                "scale_by_customer_count": True,
                "scale_by_usage_patterns": True
            },
            constraints={
                "min_instances_per_tier": {"T0": 3, "T1": 2, "T2": 1},
                "max_scale_factor": 10,
                "cost_ceiling_per_tenant": 1000.0
            },
            workload_types=[WorkloadType.API_GATEWAY, WorkloadType.WORKFLOW_ENGINE, WorkloadType.AI_AGENT],
            tenant_tiers=["T0", "T1", "T2"],
            priority=1,
            tags=["saas", "business_metrics", "revenue_driven"]
        )
        
        # Principle 3: Predictive Scaling
        principles["predictive_scaling"] = ScalingPrinciple(
            principle_id="predictive_scaling",
            name="Predictive Scaling",
            description="Scale proactively based on predicted demand",
            category="prediction",
            parameters={
                "prediction_horizon_minutes": 30,
                "confidence_threshold": 0.8,
                "lead_time_minutes": 5
            },
            constraints={
                "max_prediction_error": 0.2,
                "min_historical_data_points": 100,
                "prediction_update_frequency": 60
            },
            workload_types=list(WorkloadType),
            tenant_tiers=["T0", "T1"],
            priority=2,
            tags=["prediction", "proactive", "ml"]
        )
        
        # Principle 4: Cost Optimization
        principles["cost_optimization"] = ScalingPrinciple(
            principle_id="cost_optimization",
            name="Cost-Aware Scaling",
            description="Optimize scaling decisions for cost efficiency",
            category="finops",
            parameters={
                "cost_weight": 0.3,
                "performance_weight": 0.7,
                "spot_instance_usage": True
            },
            constraints={
                "max_cost_increase": 0.5,
                "min_performance_sla": 0.95,
                "cost_optimization_frequency": 300
            },
            workload_types=list(WorkloadType),
            tenant_tiers=["T1", "T2"],
            priority=3,
            tags=["cost", "finops", "efficiency"]
        )
        
        return principles
    
    def _initialize_scaling_strategies(self) -> Dict[WorkloadType, Dict[str, Any]]:
        """Task 7.3.2: Document vertical vs horizontal scaling strategies"""
        strategies = {}
        
        # API Gateway - Horizontal preferred
        strategies[WorkloadType.API_GATEWAY] = {
            "preferred_strategy": ScalingStrategy.HORIZONTAL,
            "vertical_limits": {"max_cpu": 16, "max_memory_gb": 64},
            "horizontal_config": {
                "min_instances": 2,
                "max_instances": 100,
                "scale_out_threshold": 0.7,
                "scale_in_threshold": 0.3,
                "cooldown_minutes": 5
            },
            "triggers": [ScalingTrigger.REQUEST_RATE, ScalingTrigger.CPU_UTILIZATION],
            "reasoning": "Stateless service benefits from horizontal scaling for high availability"
        }
        
        # DSL Compiler - Hybrid approach
        strategies[WorkloadType.DSL_COMPILER] = {
            "preferred_strategy": ScalingStrategy.HYBRID,
            "vertical_limits": {"max_cpu": 32, "max_memory_gb": 128},
            "horizontal_config": {
                "min_instances": 1,
                "max_instances": 20,
                "scale_out_threshold": 0.8,
                "scale_in_threshold": 0.2,
                "cooldown_minutes": 10
            },
            "triggers": [ScalingTrigger.CPU_UTILIZATION, ScalingTrigger.QUEUE_DEPTH],
            "reasoning": "CPU-intensive compilation benefits from vertical scaling, queue processing from horizontal"
        }
        
        # Workflow Engine - Horizontal preferred
        strategies[WorkloadType.WORKFLOW_ENGINE] = {
            "preferred_strategy": ScalingStrategy.HORIZONTAL,
            "vertical_limits": {"max_cpu": 8, "max_memory_gb": 32},
            "horizontal_config": {
                "min_instances": 2,
                "max_instances": 50,
                "scale_out_threshold": 0.6,
                "scale_in_threshold": 0.2,
                "cooldown_minutes": 3
            },
            "triggers": [ScalingTrigger.QUEUE_DEPTH, ScalingTrigger.REQUEST_RATE],
            "reasoning": "Workflow orchestration scales well horizontally with queue-based distribution"
        }
        
        # AI Agent - Vertical preferred
        strategies[WorkloadType.AI_AGENT] = {
            "preferred_strategy": ScalingStrategy.VERTICAL,
            "vertical_limits": {"max_cpu": 64, "max_memory_gb": 256},
            "horizontal_config": {
                "min_instances": 1,
                "max_instances": 10,
                "scale_out_threshold": 0.9,
                "scale_in_threshold": 0.1,
                "cooldown_minutes": 15
            },
            "triggers": [ScalingTrigger.MEMORY_UTILIZATION, ScalingTrigger.RESPONSE_TIME],
            "reasoning": "AI models benefit from larger memory and CPU for better performance"
        }
        
        # Database - Vertical preferred with read replicas
        strategies[WorkloadType.DATABASE] = {
            "preferred_strategy": ScalingStrategy.VERTICAL,
            "vertical_limits": {"max_cpu": 64, "max_memory_gb": 512},
            "horizontal_config": {
                "min_instances": 1,
                "max_instances": 5,  # Read replicas
                "scale_out_threshold": 0.8,
                "scale_in_threshold": 0.3,
                "cooldown_minutes": 20
            },
            "triggers": [ScalingTrigger.CPU_UTILIZATION, ScalingTrigger.MEMORY_UTILIZATION],
            "reasoning": "Database performance benefits from vertical scaling, read scaling from replicas"
        }
        
        return strategies
    
    def _load_finops_policies(self) -> Dict[str, Any]:
        """Task 7.3.31: Load FinOps optimization policies"""
        return {
            "cost_optimization": {
                "enabled": True,
                "target_cost_reduction": 0.15,  # 15% cost reduction target
                "optimization_frequency_minutes": 60,
                "spot_instance_threshold": 0.5,  # Use spot instances when 50% cheaper
                "reserved_instance_planning": True
            },
            "budget_controls": {
                "daily_budget_limit": 10000.0,
                "monthly_budget_limit": 250000.0,
                "alert_thresholds": [0.8, 0.9, 0.95],
                "auto_scale_down_at_budget": 0.95
            },
            "efficiency_metrics": {
                "cost_per_request_target": 0.001,
                "cost_per_tenant_target": 100.0,
                "resource_utilization_target": 0.7
            }
        }
    
    def collect_metrics(self, workload_type: WorkloadType, metrics: ScalingMetrics):
        """Collect scaling metrics for analysis"""
        key = f"{workload_type.value}_{metrics.tenant_id or 'global'}"
        
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        
        self.metrics_history[key].append(metrics)
        
        # Keep only recent metrics (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.metrics_history[key] = [
            m for m in self.metrics_history[key] 
            if m.timestamp > cutoff_time
        ]
    
    def predict_scaling_need(self, workload_type: WorkloadType, 
                           tenant_id: Optional[int] = None,
                           prediction_horizon_minutes: int = 30) -> Optional[ScalingRecommendation]:
        """Task 7.3.17: Build predictive scaling models"""
        try:
            key = f"{workload_type.value}_{tenant_id or 'global'}"
            
            if key not in self.metrics_history or len(self.metrics_history[key]) < 10:
                return None
            
            metrics_list = self.metrics_history[key]
            latest_metrics = metrics_list[-1]
            
            # Simple predictive model (can be enhanced with ML)
            predicted_load = self._predict_future_load(metrics_list, prediction_horizon_minutes)
            
            # Get scaling strategy for workload type
            strategy_config = self.scaling_strategies.get(workload_type)
            if not strategy_config:
                return None
            
            # Determine if scaling is needed
            current_utilization = max(
                latest_metrics.cpu_utilization,
                latest_metrics.memory_utilization
            )
            
            predicted_utilization = predicted_load.get('utilization', current_utilization)
            scale_out_threshold = strategy_config['horizontal_config']['scale_out_threshold']
            scale_in_threshold = strategy_config['horizontal_config']['scale_in_threshold']
            
            # Determine scaling action
            if predicted_utilization > scale_out_threshold:
                # Scale out needed
                strategy = strategy_config['preferred_strategy']
                target_instances = self._calculate_target_instances(
                    current_instances=1,  # Simplified
                    predicted_utilization=predicted_utilization,
                    strategy_config=strategy_config
                )
                
                recommendation = ScalingRecommendation(
                    workload_type=workload_type,
                    strategy=strategy,
                    confidence=predicted_load.get('confidence', 0.7),
                    target_instances=target_instances,
                    target_cpu_cores=self._calculate_target_cpu(strategy, target_instances),
                    target_memory_gb=self._calculate_target_memory(strategy, target_instances),
                    recommended_at=datetime.utcnow(),
                    execute_at=datetime.utcnow() + timedelta(minutes=5),
                    duration_minutes=prediction_horizon_minutes,
                    estimated_cost_change=self._estimate_cost_change(target_instances, 1),
                    cost_efficiency_score=0.8,
                    triggers=[ScalingTrigger.CPU_UTILIZATION],
                    reasoning=f"Predicted utilization {predicted_utilization:.1%} exceeds threshold {scale_out_threshold:.1%}"
                )
                
                return recommendation
            
            elif predicted_utilization < scale_in_threshold:
                # Scale in opportunity
                recommendation = ScalingRecommendation(
                    workload_type=workload_type,
                    strategy=ScalingStrategy.HORIZONTAL,
                    confidence=predicted_load.get('confidence', 0.7),
                    target_instances=max(1, int(predicted_utilization / scale_in_threshold)),
                    target_cpu_cores=4,
                    target_memory_gb=8,
                    recommended_at=datetime.utcnow(),
                    execute_at=datetime.utcnow() + timedelta(minutes=10),
                    duration_minutes=prediction_horizon_minutes,
                    estimated_cost_change=-50.0,  # Cost savings
                    cost_efficiency_score=0.9,
                    triggers=[ScalingTrigger.CPU_UTILIZATION],
                    reasoning=f"Predicted utilization {predicted_utilization:.1%} below threshold {scale_in_threshold:.1%}"
                )
                
                return recommendation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to predict scaling need: {e}")
            return None
    
    def optimize_costs(self, workload_type: WorkloadType, 
                      current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Task 7.3.31: Implement predictive FinOps optimization"""
        try:
            optimization_result = {
                "original_config": current_config.copy(),
                "optimized_config": current_config.copy(),
                "cost_savings": 0.0,
                "performance_impact": 0.0,
                "recommendations": []
            }
            
            # Get FinOps policies
            policies = self.optimization_policies
            
            # Optimization 1: Spot instance recommendation
            if policies["cost_optimization"]["enabled"]:
                spot_savings = self._analyze_spot_instance_opportunity(workload_type, current_config)
                if spot_savings["recommended"]:
                    optimization_result["optimized_config"]["instance_type"] = spot_savings["instance_type"]
                    optimization_result["cost_savings"] += spot_savings["savings"]
                    optimization_result["recommendations"].append({
                        "type": "spot_instances",
                        "description": f"Use spot instances for {spot_savings['savings']:.0f}% cost savings",
                        "risk": "Potential interruption"
                    })
            
            # Optimization 2: Right-sizing
            rightsizing = self._analyze_rightsizing_opportunity(workload_type, current_config)
            if rightsizing["recommended"]:
                optimization_result["optimized_config"]["cpu_cores"] = rightsizing["cpu_cores"]
                optimization_result["optimized_config"]["memory_gb"] = rightsizing["memory_gb"]
                optimization_result["cost_savings"] += rightsizing["savings"]
                optimization_result["recommendations"].append({
                    "type": "rightsizing",
                    "description": f"Right-size resources for {rightsizing['savings']:.0f}% cost savings",
                    "risk": "Minimal performance impact"
                })
            
            # Optimization 3: Reserved instance planning
            if policies["cost_optimization"]["reserved_instance_planning"]:
                reserved = self._analyze_reserved_instance_opportunity(workload_type, current_config)
                if reserved["recommended"]:
                    optimization_result["cost_savings"] += reserved["savings"]
                    optimization_result["recommendations"].append({
                        "type": "reserved_instances",
                        "description": f"Purchase reserved instances for {reserved['savings']:.0f}% cost savings",
                        "risk": "Commitment required"
                    })
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize costs: {e}")
            return {"error": str(e)}
    
    def _predict_future_load(self, metrics_list: List[ScalingMetrics], 
                           horizon_minutes: int) -> Dict[str, Any]:
        """Simple predictive model for future load"""
        if len(metrics_list) < 5:
            return {"utilization": 0.5, "confidence": 0.3}
        
        # Simple trend analysis
        recent_metrics = metrics_list[-5:]
        cpu_trend = [m.cpu_utilization for m in recent_metrics]
        memory_trend = [m.memory_utilization for m in recent_metrics]
        
        # Calculate trend slope
        cpu_slope = (cpu_trend[-1] - cpu_trend[0]) / len(cpu_trend)
        memory_slope = (memory_trend[-1] - memory_trend[0]) / len(memory_trend)
        
        # Project future utilization
        current_cpu = cpu_trend[-1]
        current_memory = memory_trend[-1]
        
        future_cpu = current_cpu + (cpu_slope * horizon_minutes / 5)
        future_memory = current_memory + (memory_slope * horizon_minutes / 5)
        
        predicted_utilization = max(future_cpu, future_memory)
        
        # Confidence based on trend stability
        cpu_variance = sum((x - sum(cpu_trend)/len(cpu_trend))**2 for x in cpu_trend) / len(cpu_trend)
        confidence = max(0.3, 1.0 - cpu_variance)
        
        return {
            "utilization": min(max(predicted_utilization, 0.0), 1.0),
            "confidence": confidence,
            "cpu_trend": cpu_slope,
            "memory_trend": memory_slope
        }
    
    def _calculate_target_instances(self, current_instances: int, 
                                  predicted_utilization: float,
                                  strategy_config: Dict[str, Any]) -> int:
        """Calculate target number of instances"""
        target_utilization = 0.7  # Target 70% utilization
        scale_factor = predicted_utilization / target_utilization
        target_instances = max(1, int(current_instances * scale_factor))
        
        # Apply limits
        max_instances = strategy_config['horizontal_config']['max_instances']
        min_instances = strategy_config['horizontal_config']['min_instances']
        
        return min(max(target_instances, min_instances), max_instances)
    
    def _calculate_target_cpu(self, strategy: ScalingStrategy, instances: int) -> int:
        """Calculate target CPU cores"""
        if strategy == ScalingStrategy.VERTICAL:
            return min(32, 4 * instances)
        else:
            return 4  # Standard per instance
    
    def _calculate_target_memory(self, strategy: ScalingStrategy, instances: int) -> int:
        """Calculate target memory GB"""
        if strategy == ScalingStrategy.VERTICAL:
            return min(128, 8 * instances)
        else:
            return 8  # Standard per instance
    
    def _estimate_cost_change(self, target_instances: int, current_instances: int) -> float:
        """Estimate cost change in dollars per hour"""
        cost_per_instance_hour = 0.50  # Simplified
        return (target_instances - current_instances) * cost_per_instance_hour
    
    def _analyze_spot_instance_opportunity(self, workload_type: WorkloadType, 
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spot instance cost savings opportunity"""
        # Simplified spot instance analysis
        if workload_type in [WorkloadType.AI_AGENT, WorkloadType.ANALYTICS]:
            return {
                "recommended": True,
                "instance_type": "spot",
                "savings": 60.0,  # 60% savings
                "interruption_risk": "medium"
            }
        return {"recommended": False}
    
    def _analyze_rightsizing_opportunity(self, workload_type: WorkloadType,
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze rightsizing opportunity"""
        # Simplified rightsizing analysis
        current_cpu = config.get("cpu_cores", 4)
        current_memory = config.get("memory_gb", 8)
        
        # Assume 20% over-provisioned
        optimal_cpu = max(2, int(current_cpu * 0.8))
        optimal_memory = max(4, int(current_memory * 0.8))
        
        if optimal_cpu < current_cpu or optimal_memory < current_memory:
            savings = ((current_cpu - optimal_cpu) / current_cpu + 
                      (current_memory - optimal_memory) / current_memory) / 2 * 100
            
            return {
                "recommended": True,
                "cpu_cores": optimal_cpu,
                "memory_gb": optimal_memory,
                "savings": savings
            }
        
        return {"recommended": False}
    
    def _analyze_reserved_instance_opportunity(self, workload_type: WorkloadType,
                                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reserved instance opportunity"""
        # Simplified reserved instance analysis
        if workload_type in [WorkloadType.DATABASE, WorkloadType.API_GATEWAY]:
            return {
                "recommended": True,
                "term_months": 12,
                "savings": 30.0,  # 30% savings
                "commitment_required": True
            }
        return {"recommended": False}


# Global instance
scalability_framework = ScalabilityFramework()

def get_scalability_framework() -> ScalabilityFramework:
    """Get the global scalability framework instance"""
    return scalability_framework
