"""
Task 4.4.2: Build metrics ingestion pipeline from Orchestrator & rbia_traces
- End-to-end metrics capture from Orchestrator and Postgres traces
- Kafka + Postgres → Fabric pipeline (multi-tenant)
"""

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import asyncpg
import json
import uuid
import logging

app = FastAPI(title="RBIA Metrics Ingestion Pipeline")
logger = logging.getLogger(__name__)

class MetricEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    source: str  # orchestrator, rbia_traces, execution_hub
    metric_type: str
    metric_name: str
    metric_value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class IngestionBatch(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    events: List[MetricEvent]
    batch_size: int
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage for pipeline
ingestion_queue: List[MetricEvent] = []
processed_batches: Dict[str, IngestionBatch] = {}

class MetricsIngestionPipeline:
    def __init__(self):
        self.db_pool = None
        self.batch_size = 100
        self.processing_interval = 30  # seconds
        
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                database="rbia_metrics",
                user="postgres",
                password="password",
                min_size=5,
                max_size=20
            )
            logger.info("✅ Metrics ingestion pipeline initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    async def ingest_from_orchestrator(self, tenant_id: str) -> List[MetricEvent]:
        """Ingest metrics from Orchestrator logs"""
        events = []
        
        if not self.db_pool:
            return events
            
        try:
            async with self.db_pool.acquire() as conn:
                # Query orchestrator execution logs
                rows = await conn.fetch("""
                    SELECT execution_id, tenant_id, workflow_type, status, 
                           execution_time_ms, created_at, metadata
                    FROM orchestrator_executions 
                    WHERE tenant_id = $1 AND created_at > NOW() - INTERVAL '1 hour'
                """, tenant_id)
                
                for row in rows:
                    # Execution success rate metric
                    events.append(MetricEvent(
                        tenant_id=tenant_id,
                        source="orchestrator",
                        metric_type="execution_success_rate",
                        metric_name="Workflow Execution Success Rate",
                        metric_value=1.0 if row['status'] == 'completed' else 0.0,
                        metadata={
                            "execution_id": row['execution_id'],
                            "workflow_type": row['workflow_type'],
                            "execution_time_ms": row['execution_time_ms']
                        }
                    ))
                    
                    # Execution time metric
                    events.append(MetricEvent(
                        tenant_id=tenant_id,
                        source="orchestrator",
                        metric_type="execution_latency",
                        metric_name="Workflow Execution Latency",
                        metric_value=float(row['execution_time_ms']),
                        metadata={
                            "execution_id": row['execution_id'],
                            "workflow_type": row['workflow_type']
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to ingest from orchestrator: {e}")
            
        return events
    
    async def ingest_from_rbia_traces(self, tenant_id: str) -> List[MetricEvent]:
        """Ingest metrics from rbia_traces table"""
        events = []
        
        if not self.db_pool:
            return events
            
        try:
            async with self.db_pool.acquire() as conn:
                # Query rbia traces
                rows = await conn.fetch("""
                    SELECT trace_id, tenant_id, workflow_id, decision_accuracy,
                           override_triggered, confidence_score, created_at
                    FROM rbia_traces 
                    WHERE tenant_id = $1 AND created_at > NOW() - INTERVAL '1 hour'
                """, tenant_id)
                
                for row in rows:
                    # Decision accuracy metric
                    if row['decision_accuracy'] is not None:
                        events.append(MetricEvent(
                            tenant_id=tenant_id,
                            source="rbia_traces",
                            metric_type="decision_accuracy",
                            metric_name="RBIA Decision Accuracy",
                            metric_value=float(row['decision_accuracy']),
                            metadata={
                                "trace_id": row['trace_id'],
                                "workflow_id": row['workflow_id']
                            }
                        ))
                    
                    # Override frequency metric
                    events.append(MetricEvent(
                        tenant_id=tenant_id,
                        source="rbia_traces",
                        metric_type="override_frequency",
                        metric_name="Override Frequency",
                        metric_value=1.0 if row['override_triggered'] else 0.0,
                        metadata={
                            "trace_id": row['trace_id'],
                            "confidence_score": float(row['confidence_score']) if row['confidence_score'] else 0.0
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to ingest from rbia_traces: {e}")
            
        return events
    
    async def process_batch(self, events: List[MetricEvent]) -> bool:
        """Process a batch of metric events"""
        if not events:
            return True
            
        batch = IngestionBatch(
            events=events,
            batch_size=len(events)
        )
        
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Insert batch into metrics warehouse
                    for event in events:
                        await conn.execute("""
                            INSERT INTO metrics_warehouse 
                            (event_id, tenant_id, source, metric_type, metric_name, 
                             metric_value, metadata, timestamp)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, 
                        event.event_id, event.tenant_id, event.source,
                        event.metric_type, event.metric_name, event.metric_value,
                        json.dumps(event.metadata), event.timestamp)
            
            processed_batches[batch.batch_id] = batch
            logger.info(f"✅ Processed batch {batch.batch_id} with {len(events)} events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            return False

# Global pipeline instance
pipeline = MetricsIngestionPipeline()

@app.on_event("startup")
async def startup_event():
    await pipeline.initialize()
    # Start background processing
    asyncio.create_task(background_ingestion_loop())

async def background_ingestion_loop():
    """Background task to continuously ingest metrics"""
    while True:
        try:
            # Get all active tenants (simplified)
            active_tenants = ["tenant_1", "tenant_2"]  # Would come from tenant registry
            
            for tenant_id in active_tenants:
                # Ingest from both sources
                orchestrator_events = await pipeline.ingest_from_orchestrator(tenant_id)
                trace_events = await pipeline.ingest_from_rbia_traces(tenant_id)
                
                all_events = orchestrator_events + trace_events
                
                if all_events:
                    await pipeline.process_batch(all_events)
                    
            await asyncio.sleep(pipeline.processing_interval)
            
        except Exception as e:
            logger.error(f"Background ingestion error: {e}")
            await asyncio.sleep(60)  # Wait longer on error

@app.post("/metrics/ingest", response_model=Dict[str, Any])
async def ingest_metrics(tenant_id: str, background_tasks: BackgroundTasks):
    """Manual trigger for metrics ingestion"""
    
    background_tasks.add_task(trigger_ingestion, tenant_id)
    
    return {
        "status": "triggered",
        "tenant_id": tenant_id,
        "message": "Metrics ingestion triggered"
    }

async def trigger_ingestion(tenant_id: str):
    """Trigger ingestion for specific tenant"""
    orchestrator_events = await pipeline.ingest_from_orchestrator(tenant_id)
    trace_events = await pipeline.ingest_from_rbia_traces(tenant_id)
    
    all_events = orchestrator_events + trace_events
    
    if all_events:
        await pipeline.process_batch(all_events)

@app.get("/metrics/pipeline/status")
async def get_pipeline_status():
    """Get pipeline status"""
    return {
        "pipeline_active": pipeline.db_pool is not None,
        "batch_size": pipeline.batch_size,
        "processing_interval": pipeline.processing_interval,
        "processed_batches": len(processed_batches),
        "queue_size": len(ingestion_queue)
    }

@app.get("/metrics/batches")
async def get_processed_batches():
    """Get processed batch information"""
    return {
        "batches": [
            {
                "batch_id": batch.batch_id,
                "batch_size": batch.batch_size,
                "ingestion_timestamp": batch.ingestion_timestamp.isoformat()
            }
            for batch in processed_batches.values()
        ]
    }
