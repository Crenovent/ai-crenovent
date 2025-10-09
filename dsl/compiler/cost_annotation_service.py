"""
Cost Annotations - Task 6.2.62
===============================

Estimated inference cost annotations
- FinOps preview for dashboard feed
- Analyzer for cost estimation
- IR attributes for cost tracking
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CostCategory(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    ML_INFERENCE = "ml_inference"
    EXTERNAL_API = "external_api"
    HUMAN_TIME = "human_time"

@dataclass
class CostEstimate:
    category: CostCategory
    estimated_cost_usd: float
    cost_per_execution: float
    monthly_estimate_usd: float
    cost_basis: str  # What the cost is based on

class CostAnnotationService:
    """Service for annotating plans with cost estimates"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cost_models = self._initialize_cost_models()
    
    def _initialize_cost_models(self) -> Dict[str, Dict[str, float]]:
        """Initialize cost models for different node types"""
        return {
            'ml_node': {
                'base_inference_cost': 0.001,  # $0.001 per inference
                'compute_cost_per_second': 0.0001,
                'memory_cost_per_gb_hour': 0.0005
            },
            'external_api': {
                'api_call_cost': 0.0001,  # $0.0001 per API call
                'data_transfer_cost_per_gb': 0.09
            },
            'data_transform': {
                'compute_cost_per_second': 0.00005,
                'memory_cost_per_gb_hour': 0.0002
            },
            'human_task': {
                'hourly_cost': 50.0,  # $50 per hour for human intervention
                'average_time_minutes': 15
            },
            'storage': {
                'storage_cost_per_gb_month': 0.023,
                'io_cost_per_request': 0.0000004
            }
        }
    
    def estimate_node_cost(self, node: Dict[str, Any]) -> List[CostEstimate]:
        """Estimate cost for a single node"""
        node_type = node.get('type', 'unknown')
        node_id = node.get('id', 'unknown')
        cost_estimates = []
        
        if 'ml_' in node_type or node_type == 'ml_node':
            cost_estimates.extend(self._estimate_ml_node_cost(node))
        elif 'external' in node_type:
            cost_estimates.extend(self._estimate_external_api_cost(node))
        elif 'human' in node_type or 'manual' in node_type:
            cost_estimates.extend(self._estimate_human_task_cost(node))
        elif 'data' in node_type or 'transform' in node_type:
            cost_estimates.extend(self._estimate_compute_cost(node))
        else:
            # Default compute cost
            cost_estimates.extend(self._estimate_default_cost(node))
        
        return cost_estimates
    
    def _estimate_ml_node_cost(self, node: Dict[str, Any]) -> List[CostEstimate]:
        """Estimate cost for ML inference node"""
        estimates = []
        model_config = node.get('model_config', {})
        cost_model = self.cost_models['ml_node']
        
        # Base inference cost
        base_cost = cost_model['base_inference_cost']
        
        # Adjust for model complexity
        model_size = model_config.get('model_size', 'medium')
        if model_size == 'large':
            base_cost *= 5
        elif model_size == 'small':
            base_cost *= 0.5
        
        # Monthly estimate based on expected usage
        expected_executions_per_month = node.get('expected_executions_per_month', 1000)
        monthly_cost = base_cost * expected_executions_per_month
        
        estimates.append(CostEstimate(
            category=CostCategory.ML_INFERENCE,
            estimated_cost_usd=base_cost,
            cost_per_execution=base_cost,
            monthly_estimate_usd=monthly_cost,
            cost_basis=f"ML inference for {model_config.get('model_type', 'unknown')} model"
        ))
        
        return estimates
    
    def _estimate_external_api_cost(self, node: Dict[str, Any]) -> List[CostEstimate]:
        """Estimate cost for external API calls"""
        estimates = []
        cost_model = self.cost_models['external_api']
        
        api_call_cost = cost_model['api_call_cost']
        expected_calls_per_month = node.get('expected_calls_per_month', 500)
        
        estimates.append(CostEstimate(
            category=CostCategory.EXTERNAL_API,
            estimated_cost_usd=api_call_cost,
            cost_per_execution=api_call_cost,
            monthly_estimate_usd=api_call_cost * expected_calls_per_month,
            cost_basis="External API call"
        ))
        
        return estimates
    
    def _estimate_human_task_cost(self, node: Dict[str, Any]) -> List[CostEstimate]:
        """Estimate cost for human tasks"""
        estimates = []
        cost_model = self.cost_models['human_task']
        
        hourly_cost = cost_model['hourly_cost']
        avg_time_minutes = node.get('average_time_minutes', cost_model['average_time_minutes'])
        cost_per_execution = (hourly_cost / 60) * avg_time_minutes
        
        expected_tasks_per_month = node.get('expected_tasks_per_month', 100)
        
        estimates.append(CostEstimate(
            category=CostCategory.HUMAN_TIME,
            estimated_cost_usd=cost_per_execution,
            cost_per_execution=cost_per_execution,
            monthly_estimate_usd=cost_per_execution * expected_tasks_per_month,
            cost_basis=f"Human task ({avg_time_minutes} minutes average)"
        ))
        
        return estimates
    
    def _estimate_compute_cost(self, node: Dict[str, Any]) -> List[CostEstimate]:
        """Estimate compute cost for processing nodes"""
        estimates = []
        cost_model = self.cost_models['data_transform']
        
        compute_cost_per_second = cost_model['compute_cost_per_second']
        expected_runtime_seconds = node.get('expected_runtime_seconds', 10)
        cost_per_execution = compute_cost_per_second * expected_runtime_seconds
        
        expected_executions_per_month = node.get('expected_executions_per_month', 1000)
        
        estimates.append(CostEstimate(
            category=CostCategory.COMPUTE,
            estimated_cost_usd=cost_per_execution,
            cost_per_execution=cost_per_execution,
            monthly_estimate_usd=cost_per_execution * expected_executions_per_month,
            cost_basis=f"Compute time ({expected_runtime_seconds}s)"
        ))
        
        return estimates
    
    def _estimate_default_cost(self, node: Dict[str, Any]) -> List[CostEstimate]:
        """Default cost estimate for unknown node types"""
        return [CostEstimate(
            category=CostCategory.COMPUTE,
            estimated_cost_usd=0.0001,
            cost_per_execution=0.0001,
            monthly_estimate_usd=0.1,
            cost_basis="Default compute cost"
        )]
    
    def annotate_plan_with_costs(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Annotate entire plan with cost estimates"""
        nodes = ir_data.get('nodes', [])
        total_cost_per_execution = 0.0
        total_monthly_cost = 0.0
        cost_breakdown = {category.value: 0.0 for category in CostCategory}
        
        for node in nodes:
            node_costs = self.estimate_node_cost(node)
            
            # Add cost annotations to node
            node['cost_annotations'] = [
                {
                    'category': cost.category.value,
                    'estimated_cost_usd': cost.estimated_cost_usd,
                    'cost_per_execution': cost.cost_per_execution,
                    'monthly_estimate_usd': cost.monthly_estimate_usd,
                    'cost_basis': cost.cost_basis
                }
                for cost in node_costs
            ]
            
            # Aggregate costs
            for cost in node_costs:
                total_cost_per_execution += cost.cost_per_execution
                total_monthly_cost += cost.monthly_estimate_usd
                cost_breakdown[cost.category.value] += cost.monthly_estimate_usd
        
        # Add plan-level cost summary
        if 'metadata' not in ir_data:
            ir_data['metadata'] = {}
        
        ir_data['metadata']['cost_summary'] = {
            'total_cost_per_execution_usd': total_cost_per_execution,
            'total_monthly_estimate_usd': total_monthly_cost,
            'cost_breakdown_monthly': cost_breakdown,
            'cost_optimization_suggestions': self._generate_cost_optimization_suggestions(
                total_monthly_cost, cost_breakdown
            )
        }
        
        return ir_data
    
    def _generate_cost_optimization_suggestions(self, total_monthly_cost: float, 
                                              cost_breakdown: Dict[str, float]) -> List[str]:
        """Generate cost optimization suggestions"""
        suggestions = []
        
        # High total cost
        if total_monthly_cost > 1000:
            suggestions.append("Consider plan modularization to reduce execution frequency")
        
        # High ML inference costs
        if cost_breakdown.get('ml_inference', 0) > total_monthly_cost * 0.5:
            suggestions.append("ML inference costs are high - consider model optimization or caching")
        
        # High human task costs
        if cost_breakdown.get('human_time', 0) > total_monthly_cost * 0.3:
            suggestions.append("High human intervention costs - consider automation opportunities")
        
        # High external API costs
        if cost_breakdown.get('external_api', 0) > total_monthly_cost * 0.4:
            suggestions.append("External API costs are significant - consider caching or bulk operations")
        
        return suggestions
    
    def generate_finops_dashboard_data(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for FinOps dashboard"""
        cost_summary = ir_data.get('metadata', {}).get('cost_summary', {})
        nodes = ir_data.get('nodes', [])
        
        # Cost by node
        node_costs = []
        for node in nodes:
            cost_annotations = node.get('cost_annotations', [])
            total_node_monthly_cost = sum(
                cost.get('monthly_estimate_usd', 0) for cost in cost_annotations
            )
            
            if total_node_monthly_cost > 0:
                node_costs.append({
                    'node_id': node.get('id'),
                    'node_type': node.get('type'),
                    'monthly_cost_usd': total_node_monthly_cost
                })
        
        # Sort by cost (highest first)
        node_costs.sort(key=lambda x: x['monthly_cost_usd'], reverse=True)
        
        return {
            'plan_id': ir_data.get('plan_id', 'unknown'),
            'total_monthly_cost_usd': cost_summary.get('total_monthly_estimate_usd', 0),
            'cost_per_execution_usd': cost_summary.get('total_cost_per_execution_usd', 0),
            'cost_breakdown': cost_summary.get('cost_breakdown_monthly', {}),
            'top_cost_nodes': node_costs[:5],  # Top 5 most expensive nodes
            'optimization_suggestions': cost_summary.get('cost_optimization_suggestions', []),
            'cost_trend': 'stable',  # Would be calculated from historical data
            'generated_at': ir_data.get('metadata', {}).get('generated_at')
        }
