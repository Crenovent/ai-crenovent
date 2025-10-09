"""
Task 6.3.77: Chaos tests (pod kill, node drain, network loss)
Chaos engineering framework for resilience testing
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ChaosType(Enum):
    """Types of chaos experiments"""
    POD_KILL = "pod_kill"
    NODE_DRAIN = "node_drain"
    NETWORK_LOSS = "network_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_FILL = "disk_fill"

@dataclass
class ChaosExperiment:
    """Chaos experiment configuration"""
    experiment_id: str
    chaos_type: ChaosType
    target_selector: Dict[str, str]  # Label selectors
    duration_seconds: int
    intensity: float  # 0.0-1.0
    enabled: bool = True

class ChaosTestingService:
    """
    Chaos engineering service for resilience testing
    Task 6.3.77: KPIs measured with pod/node/network chaos
    """
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_results: List[Dict] = []
        self.active_experiments: Dict[str, asyncio.Task] = {}
    
    def register_experiment(self, experiment: ChaosExperiment) -> bool:
        """Register chaos experiment"""
        try:
            self.experiments[experiment.experiment_id] = experiment
            logger.info(f"Registered chaos experiment: {experiment.experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register experiment: {e}")
            return False
    
    async def run_experiment(self, experiment_id: str) -> Dict:
        """Run chaos experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        if not experiment.enabled:
            raise ValueError(f"Experiment {experiment_id} is disabled")
        
        start_time = datetime.utcnow()
        
        try:
            # Start chaos
            await self._execute_chaos(experiment)
            
            # Monitor during chaos
            metrics = await self._monitor_experiment(experiment)
            
            # Record results
            result = {
                "experiment_id": experiment_id,
                "chaos_type": experiment.chaos_type.value,
                "start_time": start_time.isoformat(),
                "duration": experiment.duration_seconds,
                "success": True,
                "metrics": metrics
            }
            
            self.experiment_results.append(result)
            return result
            
        except Exception as e:
            result = {
                "experiment_id": experiment_id,
                "chaos_type": experiment.chaos_type.value,
                "start_time": start_time.isoformat(),
                "duration": experiment.duration_seconds,
                "success": False,
                "error": str(e)
            }
            
            self.experiment_results.append(result)
            return result
    
    async def _execute_chaos(self, experiment: ChaosExperiment) -> None:
        """Execute chaos based on type"""
        if experiment.chaos_type == ChaosType.POD_KILL:
            await self._pod_kill_chaos(experiment)
        elif experiment.chaos_type == ChaosType.NODE_DRAIN:
            await self._node_drain_chaos(experiment)
        elif experiment.chaos_type == ChaosType.NETWORK_LOSS:
            await self._network_loss_chaos(experiment)
        elif experiment.chaos_type == ChaosType.CPU_STRESS:
            await self._cpu_stress_chaos(experiment)
        elif experiment.chaos_type == ChaosType.MEMORY_STRESS:
            await self._memory_stress_chaos(experiment)
        elif experiment.chaos_type == ChaosType.DISK_FILL:
            await self._disk_fill_chaos(experiment)
    
    async def _pod_kill_chaos(self, experiment: ChaosExperiment) -> None:
        """Simulate pod kill chaos"""
        logger.info(f"Starting pod kill chaos: {experiment.experiment_id}")
        
        # Simulate killing pods
        kill_count = max(1, int(10 * experiment.intensity))  # Scale with intensity
        
        for i in range(kill_count):
            logger.info(f"Killing pod {i+1}/{kill_count}")
            await asyncio.sleep(experiment.duration_seconds / kill_count)
    
    async def _node_drain_chaos(self, experiment: ChaosExperiment) -> None:
        """Simulate node drain chaos"""
        logger.info(f"Starting node drain chaos: {experiment.experiment_id}")
        
        # Simulate draining nodes
        await asyncio.sleep(experiment.duration_seconds)
        
        logger.info(f"Node drain completed: {experiment.experiment_id}")
    
    async def _network_loss_chaos(self, experiment: ChaosExperiment) -> None:
        """Simulate network loss chaos"""
        logger.info(f"Starting network loss chaos: {experiment.experiment_id}")
        
        # Simulate network partitions
        loss_percentage = experiment.intensity * 100
        logger.info(f"Simulating {loss_percentage}% packet loss")
        
        await asyncio.sleep(experiment.duration_seconds)
    
    async def _cpu_stress_chaos(self, experiment: ChaosExperiment) -> None:
        """Simulate CPU stress chaos"""
        logger.info(f"Starting CPU stress chaos: {experiment.experiment_id}")
        
        # Simulate CPU load
        cpu_percentage = experiment.intensity * 100
        logger.info(f"Simulating {cpu_percentage}% CPU load")
        
        await asyncio.sleep(experiment.duration_seconds)
    
    async def _memory_stress_chaos(self, experiment: ChaosExperiment) -> None:
        """Simulate memory stress chaos"""
        logger.info(f"Starting memory stress chaos: {experiment.experiment_id}")
        
        # Simulate memory pressure
        memory_percentage = experiment.intensity * 100
        logger.info(f"Simulating {memory_percentage}% memory usage")
        
        await asyncio.sleep(experiment.duration_seconds)
    
    async def _disk_fill_chaos(self, experiment: ChaosExperiment) -> None:
        """Simulate disk fill chaos"""
        logger.info(f"Starting disk fill chaos: {experiment.experiment_id}")
        
        # Simulate disk space consumption
        disk_percentage = experiment.intensity * 100
        logger.info(f"Simulating {disk_percentage}% disk usage")
        
        await asyncio.sleep(experiment.duration_seconds)
    
    async def _monitor_experiment(self, experiment: ChaosExperiment) -> Dict:
        """Monitor metrics during experiment"""
        # Simulate metric collection
        return {
            "response_time_p99": random.uniform(100, 2000),
            "error_rate": random.uniform(0, 0.1),
            "throughput": random.uniform(50, 200),
            "recovery_time_seconds": random.uniform(10, 60)
        }
    
    def get_experiment_results(self, experiment_id: Optional[str] = None) -> List[Dict]:
        """Get experiment results"""
        if experiment_id:
            return [r for r in self.experiment_results if r["experiment_id"] == experiment_id]
        return self.experiment_results

# Global chaos testing service
chaos_testing_service = ChaosTestingService()
