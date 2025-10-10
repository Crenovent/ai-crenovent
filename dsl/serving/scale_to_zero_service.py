"""
Task 6.3.50: Scale-to-zero for infrequent models
Implement scale-to-zero functionality for cost efficiency
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ScaleToZeroConfig:
    """Scale-to-zero configuration"""
    model_id: str
    idle_timeout_minutes: int = 30
    cold_start_timeout_seconds: int = 60
    min_instances: int = 0
    max_instances: int = 10
    enabled: bool = True

@dataclass
class ModelInstance:
    """Model instance state"""
    model_id: str
    instance_id: str
    status: str  # "running", "idle", "scaling_down", "scaled_to_zero"
    last_request_time: datetime
    created_at: datetime
    requests_served: int = 0

class ScaleToZeroService:
    """
    Scale-to-zero service for infrequent models
    Task 6.3.50: Cost efficiency with cold start guard
    """
    
    def __init__(self):
        self.configs: Dict[str, ScaleToZeroConfig] = {}
        self.instances: Dict[str, Dict[str, ModelInstance]] = {}  # model_id -> {instance_id -> instance}
        self.scaling_tasks: Dict[str, asyncio.Task] = {}
    
    def configure_scale_to_zero(self, config: ScaleToZeroConfig) -> bool:
        """Configure scale-to-zero for a model"""
        try:
            self.configs[config.model_id] = config
            
            if config.model_id not in self.instances:
                self.instances[config.model_id] = {}
            
            # Start monitoring task
            if config.enabled and config.model_id not in self.scaling_tasks:
                task = asyncio.create_task(self._monitor_model(config.model_id))
                self.scaling_tasks[config.model_id] = task
            
            logger.info(f"Configured scale-to-zero for model: {config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure scale-to-zero: {e}")
            return False
    
    async def handle_request(self, model_id: str, request_data: Dict) -> Dict:
        """Handle request with scale-to-zero logic"""
        config = self.configs.get(model_id)
        if not config or not config.enabled:
            return await self._process_request_normally(model_id, request_data)
        
        # Check if we have running instances
        running_instances = self._get_running_instances(model_id)
        
        if not running_instances:
            # Scale up from zero
            logger.info(f"Scaling up model {model_id} from zero")
            instance = await self._scale_up_instance(model_id)
            if not instance:
                raise Exception(f"Failed to scale up model {model_id}")
            running_instances = [instance]
        
        # Select instance and process request
        instance = running_instances[0]  # Simple round-robin
        result = await self._process_request_on_instance(instance, request_data)
        
        # Update instance stats
        instance.last_request_time = datetime.utcnow()
        instance.requests_served += 1
        
        return result
    
    async def _monitor_model(self, model_id: str) -> None:
        """Monitor model for scale-to-zero opportunities"""
        config = self.configs[model_id]
        
        while config.enabled:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                idle_threshold = current_time - timedelta(minutes=config.idle_timeout_minutes)
                
                instances = self.instances.get(model_id, {})
                
                for instance_id, instance in list(instances.items()):
                    if instance.status == "running" and instance.last_request_time < idle_threshold:
                        # Scale down idle instance
                        await self._scale_down_instance(model_id, instance_id)
                
            except Exception as e:
                logger.error(f"Monitoring error for model {model_id}: {e}")
                await asyncio.sleep(60)
    
    async def _scale_up_instance(self, model_id: str) -> Optional[ModelInstance]:
        """Scale up a new instance"""
        try:
            config = self.configs[model_id]
            current_instances = len(self.instances.get(model_id, {}))
            
            if current_instances >= config.max_instances:
                logger.warning(f"Max instances reached for model {model_id}")
                return None
            
            # Simulate cold start
            logger.info(f"Cold starting instance for model {model_id}")
            await asyncio.sleep(0.5)  # Simulate cold start time
            
            instance_id = f"{model_id}_instance_{datetime.utcnow().timestamp()}"
            instance = ModelInstance(
                model_id=model_id,
                instance_id=instance_id,
                status="running",
                last_request_time=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            
            if model_id not in self.instances:
                self.instances[model_id] = {}
            
            self.instances[model_id][instance_id] = instance
            
            logger.info(f"Scaled up instance {instance_id} for model {model_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to scale up instance for model {model_id}: {e}")
            return None
    
    async def _scale_down_instance(self, model_id: str, instance_id: str) -> bool:
        """Scale down an instance"""
        try:
            if model_id not in self.instances or instance_id not in self.instances[model_id]:
                return False
            
            instance = self.instances[model_id][instance_id]
            instance.status = "scaling_down"
            
            # Simulate graceful shutdown
            await asyncio.sleep(0.1)
            
            # Remove instance
            del self.instances[model_id][instance_id]
            
            logger.info(f"Scaled down instance {instance_id} for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale down instance {instance_id}: {e}")
            return False
    
    def _get_running_instances(self, model_id: str) -> list:
        """Get running instances for a model"""
        if model_id not in self.instances:
            return []
        
        return [
            instance for instance in self.instances[model_id].values()
            if instance.status == "running"
        ]
    
    async def _process_request_on_instance(self, instance: ModelInstance, request_data: Dict) -> Dict:
        """Process request on specific instance"""
        # Simulate request processing
        await asyncio.sleep(0.1)
        
        return {
            "prediction": 0.75,
            "confidence": 0.85,
            "model_id": instance.model_id,
            "instance_id": instance.instance_id,
            "cold_start": False
        }
    
    async def _process_request_normally(self, model_id: str, request_data: Dict) -> Dict:
        """Process request without scale-to-zero"""
        await asyncio.sleep(0.1)
        
        return {
            "prediction": 0.75,
            "confidence": 0.85,
            "model_id": model_id,
            "scale_to_zero": False
        }
    
    def get_scaling_stats(self, model_id: str) -> Dict:
        """Get scaling statistics for a model"""
        config = self.configs.get(model_id)
        instances = self.instances.get(model_id, {})
        
        running_count = len([i for i in instances.values() if i.status == "running"])
        total_requests = sum(i.requests_served for i in instances.values())
        
        return {
            "model_id": model_id,
            "scale_to_zero_enabled": config.enabled if config else False,
            "current_instances": running_count,
            "total_instances_created": len(instances),
            "total_requests_served": total_requests,
            "idle_timeout_minutes": config.idle_timeout_minutes if config else None
        }
    
    async def force_scale_to_zero(self, model_id: str) -> bool:
        """Force scale all instances to zero"""
        if model_id not in self.instances:
            return True
        
        instances = list(self.instances[model_id].keys())
        for instance_id in instances:
            await self._scale_down_instance(model_id, instance_id)
        
        logger.info(f"Forced scale-to-zero for model {model_id}")
        return True

# Global scale-to-zero service
scale_to_zero_service = ScaleToZeroService()
