"""
Task 6.3.64: Spot/low-cost pools with auto-evict fallback
Cost optimization service with spot instance management
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class InstanceType(Enum):
    """Instance types for cost optimization"""
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    PREEMPTIBLE = "preemptible"

@dataclass
class CostPool:
    """Cost-optimized compute pool"""
    pool_id: str
    instance_type: InstanceType
    cost_per_hour: float
    reliability_score: float  # 0-1, higher is more reliable
    max_instances: int
    current_instances: int = 0
    eviction_rate: float = 0.1  # Historical eviction rate

@dataclass
class WorkloadAssignment:
    """Workload assignment to cost pool"""
    workload_id: str
    model_id: str
    pool_id: str
    instance_id: str
    assigned_at: datetime
    priority: str = "normal"
    fallback_pools: List[str] = None

class SpotInstanceService:
    """
    Spot instance and cost optimization service
    Task 6.3.64: Cost reduction with auto-evict fallback
    """
    
    def __init__(self):
        self.cost_pools: Dict[str, CostPool] = {}
        self.workload_assignments: Dict[str, WorkloadAssignment] = {}
        self.eviction_history: List[Dict] = []
        self.cost_savings: float = 0.0
        
        # Initialize default pools
        self._initialize_default_pools()
    
    def _initialize_default_pools(self) -> None:
        """Initialize default cost pools"""
        pools = [
            CostPool(
                pool_id="spot_gpu",
                instance_type=InstanceType.SPOT,
                cost_per_hour=0.50,
                reliability_score=0.7,
                max_instances=10
            ),
            CostPool(
                pool_id="on_demand_gpu",
                instance_type=InstanceType.ON_DEMAND,
                cost_per_hour=2.00,
                reliability_score=0.99,
                max_instances=5
            ),
            CostPool(
                pool_id="preemptible_cpu",
                instance_type=InstanceType.PREEMPTIBLE,
                cost_per_hour=0.10,
                reliability_score=0.8,
                max_instances=20
            ),
            CostPool(
                pool_id="reserved_gpu",
                instance_type=InstanceType.RESERVED,
                cost_per_hour=1.20,
                reliability_score=0.99,
                max_instances=3
            )
        ]
        
        for pool in pools:
            self.cost_pools[pool.pool_id] = pool
    
    def register_cost_pool(self, pool: CostPool) -> bool:
        """Register a cost-optimized pool"""
        try:
            self.cost_pools[pool.pool_id] = pool
            logger.info(f"Registered cost pool: {pool.pool_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register cost pool: {e}")
            return False
    
    async def assign_workload(self, workload_id: str, model_id: str, priority: str = "normal") -> Optional[str]:
        """Assign workload to most cost-effective pool"""
        try:
            # Find best pool based on cost and availability
            best_pool = self._select_optimal_pool(priority)
            
            if not best_pool:
                logger.warning(f"No available pools for workload {workload_id}")
                return None
            
            # Create assignment
            instance_id = f"{best_pool.pool_id}_instance_{datetime.utcnow().timestamp()}"
            
            assignment = WorkloadAssignment(
                workload_id=workload_id,
                model_id=model_id,
                pool_id=best_pool.pool_id,
                instance_id=instance_id,
                assigned_at=datetime.utcnow(),
                priority=priority,
                fallback_pools=self._get_fallback_pools(best_pool.pool_id, priority)
            )
            
            # Update pool usage
            best_pool.current_instances += 1
            
            # Store assignment
            self.workload_assignments[workload_id] = assignment
            
            logger.info(f"Assigned workload {workload_id} to pool {best_pool.pool_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to assign workload {workload_id}: {e}")
            return None
    
    def _select_optimal_pool(self, priority: str) -> Optional[CostPool]:
        """Select optimal pool based on cost and availability"""
        available_pools = [
            pool for pool in self.cost_pools.values()
            if pool.current_instances < pool.max_instances
        ]
        
        if not available_pools:
            return None
        
        # For high priority, prefer reliability over cost
        if priority == "high":
            return max(available_pools, key=lambda p: p.reliability_score)
        
        # For normal/low priority, optimize for cost
        # Score = cost_efficiency * reliability
        best_pool = None
        best_score = -1
        
        for pool in available_pools:
            # Cost efficiency (inverse of cost)
            cost_efficiency = 1.0 / pool.cost_per_hour
            
            # Combined score
            score = (cost_efficiency * 0.7) + (pool.reliability_score * 0.3)
            
            if score > best_score:
                best_score = score
                best_pool = pool
        
        return best_pool
    
    def _get_fallback_pools(self, primary_pool_id: str, priority: str) -> List[str]:
        """Get fallback pools for eviction scenarios"""
        fallbacks = []
        
        primary_pool = self.cost_pools[primary_pool_id]
        
        # Find pools with higher reliability
        for pool_id, pool in self.cost_pools.items():
            if (pool_id != primary_pool_id and 
                pool.reliability_score > primary_pool.reliability_score):
                fallbacks.append(pool_id)
        
        # Sort by reliability (highest first)
        fallbacks.sort(key=lambda pid: self.cost_pools[pid].reliability_score, reverse=True)
        
        return fallbacks[:2]  # Limit to 2 fallback options
    
    async def handle_eviction(self, workload_id: str) -> bool:
        """Handle instance eviction with fallback"""
        if workload_id not in self.workload_assignments:
            return False
        
        assignment = self.workload_assignments[workload_id]
        original_pool = self.cost_pools[assignment.pool_id]
        
        try:
            # Record eviction
            eviction_record = {
                "workload_id": workload_id,
                "evicted_pool": assignment.pool_id,
                "evicted_at": datetime.utcnow(),
                "fallback_attempted": False,
                "fallback_successful": False
            }
            
            # Free up original pool
            original_pool.current_instances -= 1
            
            # Try fallback pools
            if assignment.fallback_pools:
                for fallback_pool_id in assignment.fallback_pools:
                    fallback_pool = self.cost_pools[fallback_pool_id]
                    
                    if fallback_pool.current_instances < fallback_pool.max_instances:
                        # Move to fallback pool
                        assignment.pool_id = fallback_pool_id
                        assignment.instance_id = f"{fallback_pool_id}_instance_{datetime.utcnow().timestamp()}"
                        assignment.assigned_at = datetime.utcnow()
                        
                        fallback_pool.current_instances += 1
                        
                        eviction_record["fallback_attempted"] = True
                        eviction_record["fallback_successful"] = True
                        eviction_record["fallback_pool"] = fallback_pool_id
                        
                        logger.info(f"Workload {workload_id} moved to fallback pool {fallback_pool_id}")
                        break
            
            self.eviction_history.append(eviction_record)
            
            # Calculate cost impact
            if eviction_record["fallback_successful"]:
                cost_increase = (
                    self.cost_pools[eviction_record["fallback_pool"]].cost_per_hour -
                    original_pool.cost_per_hour
                )
                logger.info(f"Eviction fallback cost increase: ${cost_increase:.2f}/hour")
            
            return eviction_record["fallback_successful"]
            
        except Exception as e:
            logger.error(f"Failed to handle eviction for workload {workload_id}: {e}")
            return False
    
    def complete_workload(self, workload_id: str) -> bool:
        """Complete workload and free resources"""
        if workload_id not in self.workload_assignments:
            return False
        
        assignment = self.workload_assignments[workload_id]
        pool = self.cost_pools[assignment.pool_id]
        
        # Free pool resources
        pool.current_instances -= 1
        
        # Calculate cost savings
        runtime_hours = (datetime.utcnow() - assignment.assigned_at).total_seconds() / 3600
        actual_cost = pool.cost_per_hour * runtime_hours
        
        # Compare with on-demand cost
        on_demand_cost = 2.00 * runtime_hours  # Baseline on-demand cost
        savings = on_demand_cost - actual_cost
        
        if savings > 0:
            self.cost_savings += savings
        
        # Remove assignment
        del self.workload_assignments[workload_id]
        
        logger.info(f"Completed workload {workload_id}, savings: ${savings:.2f}")
        return True
    
    def get_cost_optimization_report(self) -> Dict:
        """Generate cost optimization report"""
        total_workloads = len(self.workload_assignments) + len(self.eviction_history)
        evictions = len(self.eviction_history)
        successful_fallbacks = len([e for e in self.eviction_history if e["fallback_successful"]])
        
        # Pool utilization
        pool_stats = {}
        for pool_id, pool in self.cost_pools.items():
            utilization = pool.current_instances / pool.max_instances if pool.max_instances > 0 else 0
            pool_stats[pool_id] = {
                "type": pool.instance_type.value,
                "cost_per_hour": pool.cost_per_hour,
                "utilization": utilization,
                "current_instances": pool.current_instances,
                "max_instances": pool.max_instances
            }
        
        return {
            "total_cost_savings": self.cost_savings,
            "total_workloads": total_workloads,
            "eviction_rate": evictions / total_workloads if total_workloads > 0 else 0,
            "fallback_success_rate": successful_fallbacks / evictions if evictions > 0 else 0,
            "active_workloads": len(self.workload_assignments),
            "pool_utilization": pool_stats,
            "cost_efficiency": self.cost_savings / total_workloads if total_workloads > 0 else 0
        }
    
    def simulate_spot_eviction(self, pool_id: str) -> List[str]:
        """Simulate spot instance eviction for testing"""
        if pool_id not in self.cost_pools:
            return []
        
        # Find workloads in this pool
        affected_workloads = [
            wid for wid, assignment in self.workload_assignments.items()
            if assignment.pool_id == pool_id
        ]
        
        # Simulate eviction for all workloads in pool
        evicted_workloads = []
        for workload_id in affected_workloads:
            success = asyncio.create_task(self.handle_eviction(workload_id))
            if success:
                evicted_workloads.append(workload_id)
        
        logger.info(f"Simulated eviction of {len(evicted_workloads)} workloads from pool {pool_id}")
        return evicted_workloads

# Global spot instance service
spot_instance_service = SpotInstanceService()
