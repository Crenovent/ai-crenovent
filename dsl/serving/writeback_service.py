"""
Task 6.3.54: Writeback scores to operational stores
Writeback model scores to operational stores (Postgres/CRM/Billing)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class WritebackDestination(Enum):
    """Supported writeback destinations"""
    POSTGRES = "postgres"
    CRM = "crm"
    BILLING = "billing"
    WEBHOOK = "webhook"

@dataclass
class WritebackConfig:
    """Writeback configuration"""
    destination: WritebackDestination
    table_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    batch_size: int = 100
    retry_attempts: int = 3
    idempotent: bool = True

@dataclass
class ScoreRecord:
    """Score record for writeback"""
    entity_id: str
    model_id: str
    score: float
    confidence: float
    prediction_timestamp: datetime
    tenant_id: str
    metadata: Dict[str, Any] = None

class WritebackService:
    """
    Score writeback service to operational stores
    Task 6.3.54: Closed loop integration
    """
    
    def __init__(self):
        self.writeback_configs: Dict[str, WritebackConfig] = {}
        self.pending_writebacks: Dict[str, List[ScoreRecord]] = {}
    
    def register_writeback_config(self, destination_id: str, config: WritebackConfig) -> bool:
        """Register writeback configuration"""
        try:
            self.writeback_configs[destination_id] = config
            self.pending_writebacks[destination_id] = []
            logger.info(f"Registered writeback config: {destination_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register writeback config: {e}")
            return False
    
    async def writeback_score(self, destination_id: str, score_record: ScoreRecord) -> bool:
        """Writeback single score record"""
        if destination_id not in self.writeback_configs:
            logger.error(f"Writeback destination not found: {destination_id}")
            return False
        
        config = self.writeback_configs[destination_id]
        
        # Add to pending batch
        self.pending_writebacks[destination_id].append(score_record)
        
        # Process batch if size reached
        if len(self.pending_writebacks[destination_id]) >= config.batch_size:
            return await self._process_batch(destination_id)
        
        return True
    
    async def writeback_batch(self, destination_id: str, score_records: List[ScoreRecord]) -> bool:
        """Writeback batch of score records"""
        if destination_id not in self.writeback_configs:
            logger.error(f"Writeback destination not found: {destination_id}")
            return False
        
        config = self.writeback_configs[destination_id]
        
        for attempt in range(config.retry_attempts):
            try:
                if config.destination == WritebackDestination.POSTGRES:
                    success = await self._writeback_postgres(config, score_records)
                elif config.destination == WritebackDestination.CRM:
                    success = await self._writeback_crm(config, score_records)
                elif config.destination == WritebackDestination.BILLING:
                    success = await self._writeback_billing(config, score_records)
                elif config.destination == WritebackDestination.WEBHOOK:
                    success = await self._writeback_webhook(config, score_records)
                else:
                    logger.error(f"Unsupported destination: {config.destination}")
                    return False
                
                if success:
                    logger.info(f"Successfully wrote back {len(score_records)} records to {destination_id}")
                    return True
                
            except Exception as e:
                logger.warning(f"Writeback attempt {attempt + 1} failed: {e}")
                if attempt < config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"All writeback attempts failed for {destination_id}")
        return False
    
    async def _process_batch(self, destination_id: str) -> bool:
        """Process pending batch for destination"""
        records = self.pending_writebacks[destination_id]
        if not records:
            return True
        
        success = await self.writeback_batch(destination_id, records)
        
        if success:
            self.pending_writebacks[destination_id] = []
        
        return success
    
    async def _writeback_postgres(self, config: WritebackConfig, records: List[ScoreRecord]) -> bool:
        """Writeback to PostgreSQL"""
        # Simulate PostgreSQL writeback
        logger.info(f"Writing {len(records)} records to PostgreSQL table: {config.table_name}")
        
        # Build idempotent upsert query
        if config.idempotent:
            # ON CONFLICT DO UPDATE for idempotency
            query = f"""
            INSERT INTO {config.table_name} 
            (entity_id, model_id, score, confidence, prediction_timestamp, tenant_id, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, model_id, tenant_id) 
            DO UPDATE SET 
                score = EXCLUDED.score,
                confidence = EXCLUDED.confidence,
                prediction_timestamp = EXCLUDED.prediction_timestamp,
                metadata = EXCLUDED.metadata
            """
        
        await asyncio.sleep(0.1)  # Simulate database write
        return True
    
    async def _writeback_crm(self, config: WritebackConfig, records: List[ScoreRecord]) -> bool:
        """Writeback to CRM system"""
        logger.info(f"Writing {len(records)} records to CRM: {config.endpoint_url}")
        
        # Simulate CRM API call
        payload = {
            "records": [
                {
                    "entity_id": record.entity_id,
                    "score": record.score,
                    "confidence": record.confidence,
                    "model": record.model_id,
                    "timestamp": record.prediction_timestamp.isoformat()
                }
                for record in records
            ]
        }
        
        await asyncio.sleep(0.2)  # Simulate API call
        return True
    
    async def _writeback_billing(self, config: WritebackConfig, records: List[ScoreRecord]) -> bool:
        """Writeback to billing system"""
        logger.info(f"Writing {len(records)} records to billing system")
        
        # Simulate billing system integration
        for record in records:
            billing_event = {
                "customer_id": record.entity_id,
                "service": "ml_inference",
                "model_id": record.model_id,
                "usage_timestamp": record.prediction_timestamp.isoformat(),
                "tenant_id": record.tenant_id
            }
        
        await asyncio.sleep(0.1)  # Simulate billing API
        return True
    
    async def _writeback_webhook(self, config: WritebackConfig, records: List[ScoreRecord]) -> bool:
        """Writeback via webhook"""
        logger.info(f"Sending {len(records)} records to webhook: {config.endpoint_url}")
        
        # Simulate webhook call
        payload = {
            "scores": [
                {
                    "entity_id": record.entity_id,
                    "model_id": record.model_id,
                    "score": record.score,
                    "confidence": record.confidence,
                    "timestamp": record.prediction_timestamp.isoformat(),
                    "tenant_id": record.tenant_id,
                    "metadata": record.metadata
                }
                for record in records
            ]
        }
        
        await asyncio.sleep(0.15)  # Simulate HTTP request
        return True
    
    async def flush_all_pending(self) -> Dict[str, bool]:
        """Flush all pending writebacks"""
        results = {}
        
        for destination_id in self.pending_writebacks:
            if self.pending_writebacks[destination_id]:
                results[destination_id] = await self._process_batch(destination_id)
            else:
                results[destination_id] = True
        
        return results

# Global writeback service instance
writeback_service = WritebackService()
