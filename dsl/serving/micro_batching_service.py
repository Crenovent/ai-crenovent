"""
Task 6.3.14: Request batching (micro-batching for NN models)
Implement micro-batching for neural network model inference
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    model_id: str
    input_data: Dict[str, Any]
    future: asyncio.Future
    received_at: datetime

@dataclass
class BatchConfig:
    """Batching configuration for a model"""
    model_id: str
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    enabled: bool = True

class MicroBatchingService:
    """
    Micro-batching service for neural network models
    Task 6.3.14: Throughput gains with batching window
    """
    
    def __init__(self):
        self.batch_configs: Dict[str, BatchConfig] = {}
        self.pending_batches: Dict[str, List[BatchRequest]] = defaultdict(list)
        self.batch_processors: Dict[str, asyncio.Task] = {}
    
    def configure_batching(self, config: BatchConfig) -> bool:
        """Configure batching for a model"""
        try:
            self.batch_configs[config.model_id] = config
            
            # Start batch processor if not running
            if config.model_id not in self.batch_processors and config.enabled:
                task = asyncio.create_task(self._batch_processor(config.model_id))
                self.batch_processors[config.model_id] = task
            
            logger.info(f"Configured batching for model: {config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure batching: {e}")
            return False
    
    async def submit_request(self, model_id: str, input_data: Dict[str, Any]) -> Any:
        """Submit request for batched processing"""
        if model_id not in self.batch_configs or not self.batch_configs[model_id].enabled:
            # Process immediately if batching not configured
            return await self._process_single_request(model_id, input_data)
        
        # Create batch request
        request_id = f"{model_id}_{datetime.utcnow().timestamp()}"
        future = asyncio.Future()
        
        batch_request = BatchRequest(
            request_id=request_id,
            model_id=model_id,
            input_data=input_data,
            future=future,
            received_at=datetime.utcnow()
        )
        
        # Add to pending batch
        self.pending_batches[model_id].append(batch_request)
        
        # Wait for result
        return await future
    
    async def _batch_processor(self, model_id: str) -> None:
        """Process batches for a specific model"""
        config = self.batch_configs[model_id]
        
        while config.enabled:
            try:
                # Wait for batch timeout or max size
                await asyncio.sleep(config.batch_timeout_ms / 1000)
                
                # Get pending requests
                pending = self.pending_batches[model_id]
                if not pending:
                    continue
                
                # Create batch
                batch_size = min(len(pending), config.max_batch_size)
                batch = pending[:batch_size]
                self.pending_batches[model_id] = pending[batch_size:]
                
                # Process batch
                await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Batch processor error for {model_id}: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _process_batch(self, batch: List[BatchRequest]) -> None:
        """Process a batch of requests"""
        if not batch:
            return
        
        model_id = batch[0].model_id
        
        try:
            # Prepare batch input
            batch_input = [req.input_data for req in batch]
            
            # Process batch inference
            batch_results = await self._batch_inference(model_id, batch_input)
            
            # Distribute results to futures
            for i, request in enumerate(batch):
                if i < len(batch_results):
                    request.future.set_result(batch_results[i])
                else:
                    request.future.set_exception(
                        ValueError(f"Missing result for request {request.request_id}")
                    )
        
        except Exception as e:
            # Set exception for all requests in batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _batch_inference(self, model_id: str, batch_input: List[Dict[str, Any]]) -> List[Any]:
        """Perform batch inference"""
        # Simulate batch inference with improved throughput
        await asyncio.sleep(0.05)  # Batch processing time
        
        return [
            {
                "prediction": 0.75 + (i * 0.01),  # Slight variation per item
                "confidence": 0.85,
                "model_id": model_id,
                "batch_size": len(batch_input),
                "batch_index": i
            }
            for i in range(len(batch_input))
        ]
    
    async def _process_single_request(self, model_id: str, input_data: Dict[str, Any]) -> Any:
        """Process single request without batching"""
        await asyncio.sleep(0.1)  # Single request processing time
        
        return {
            "prediction": 0.75,
            "confidence": 0.85,
            "model_id": model_id,
            "batched": False
        }
    
    def get_batch_stats(self, model_id: str) -> Dict[str, Any]:
        """Get batching statistics for a model"""
        return {
            "model_id": model_id,
            "pending_requests": len(self.pending_batches[model_id]),
            "batching_enabled": self.batch_configs.get(model_id, BatchConfig(model_id)).enabled,
            "batch_config": self.batch_configs.get(model_id).__dict__ if model_id in self.batch_configs else None
        }

# Global micro-batching service
micro_batching_service = MicroBatchingService()
