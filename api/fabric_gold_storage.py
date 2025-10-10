"""
Task 4.4.3: Implement Fabric Gold layer storage for metrics (partitioned by tenant, industry)
- Metrics warehouse with multi-tenant partitions
- Residency enforced storage
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncpg
import json
import uuid
import logging

app = FastAPI(title="RBIA Fabric Gold Layer Storage")
logger = logging.getLogger(__name__)

class MetricRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    industry_code: str
    partition_key: str
    
    # Metric data
    metric_type: str
    metric_name: str
    metric_value: float
    metric_unit: str
    
    # Metadata
    source_system: str
    data_residency: str  # US, EU, APAC
    quality_score: float = 1.0
    
    # Timestamps
    event_timestamp: datetime
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PartitionInfo(BaseModel):
    partition_name: str
    tenant_id: str
    industry_code: str
    data_residency: str
    record_count: int
    size_mb: float
    last_updated: datetime

class FabricGoldStorage:
    def __init__(self):
        self.db_pool = None
        self.partition_cache: Dict[str, PartitionInfo] = {}
        
    async def initialize(self):
        """Initialize Fabric Gold layer storage"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                database="rbia_fabric_gold",
                user="postgres", 
                password="password",
                min_size=10,
                max_size=50
            )
            
            await self._create_partition_tables()
            logger.info("✅ Fabric Gold layer storage initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Fabric Gold storage: {e}")
            return False
    
    async def _create_partition_tables(self):
        """Create partitioned tables for metrics storage"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            # Create main partitioned table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics_gold (
                    record_id UUID PRIMARY KEY,
                    tenant_id VARCHAR(255) NOT NULL,
                    industry_code VARCHAR(50) NOT NULL,
                    partition_key VARCHAR(255) NOT NULL,
                    
                    metric_type VARCHAR(100) NOT NULL,
                    metric_name VARCHAR(255) NOT NULL,
                    metric_value DECIMAL(15,4) NOT NULL,
                    metric_unit VARCHAR(50) NOT NULL,
                    
                    source_system VARCHAR(100) NOT NULL,
                    data_residency VARCHAR(10) NOT NULL,
                    quality_score DECIMAL(3,2) DEFAULT 1.0,
                    
                    event_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    ingestion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    
                    metadata JSONB DEFAULT '{}',
                    
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                ) PARTITION BY HASH (tenant_id, industry_code);
            """)
            
            # Create partition management table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS partition_registry (
                    partition_name VARCHAR(255) PRIMARY KEY,
                    tenant_id VARCHAR(255) NOT NULL,
                    industry_code VARCHAR(50) NOT NULL,
                    data_residency VARCHAR(10) NOT NULL,
                    record_count BIGINT DEFAULT 0,
                    size_mb DECIMAL(10,2) DEFAULT 0.0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
    
    async def _ensure_partition(self, tenant_id: str, industry_code: str, data_residency: str) -> str:
        """Ensure partition exists for tenant/industry combination"""
        partition_key = f"{tenant_id}_{industry_code}_{data_residency}"
        partition_name = f"metrics_gold_{partition_key}"
        
        if partition_name in self.partition_cache:
            return partition_name
            
        if not self.db_pool:
            return partition_name
            
        try:
            async with self.db_pool.acquire() as conn:
                # Check if partition exists
                exists = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = $1
                """, partition_name)
                
                if not exists:
                    # Create partition
                    await conn.execute(f"""
                        CREATE TABLE {partition_name} PARTITION OF metrics_gold
                        FOR VALUES WITH (MODULUS 4, REMAINDER {hash(partition_key) % 4});
                    """)
                    
                    # Register partition
                    await conn.execute("""
                        INSERT INTO partition_registry 
                        (partition_name, tenant_id, industry_code, data_residency)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (partition_name) DO NOTHING
                    """, partition_name, tenant_id, industry_code, data_residency)
                
                # Cache partition info
                self.partition_cache[partition_name] = PartitionInfo(
                    partition_name=partition_name,
                    tenant_id=tenant_id,
                    industry_code=industry_code,
                    data_residency=data_residency,
                    record_count=0,
                    size_mb=0.0,
                    last_updated=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Failed to ensure partition {partition_name}: {e}")
            
        return partition_name
    
    async def store_metric_record(self, record: MetricRecord) -> bool:
        """Store metric record in appropriate partition"""
        try:
            # Ensure partition exists
            partition_name = await self._ensure_partition(
                record.tenant_id, record.industry_code, record.data_residency
            )
            
            if not self.db_pool:
                return False
                
            async with self.db_pool.acquire() as conn:
                # Insert record
                await conn.execute("""
                    INSERT INTO metrics_gold 
                    (record_id, tenant_id, industry_code, partition_key,
                     metric_type, metric_name, metric_value, metric_unit,
                     source_system, data_residency, quality_score,
                     event_timestamp, ingestion_timestamp, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, 
                record.record_id, record.tenant_id, record.industry_code, record.partition_key,
                record.metric_type, record.metric_name, record.metric_value, record.metric_unit,
                record.source_system, record.data_residency, record.quality_score,
                record.event_timestamp, record.ingestion_timestamp, json.dumps(record.metadata))
                
                # Update partition stats
                await conn.execute("""
                    UPDATE partition_registry 
                    SET record_count = record_count + 1, last_updated = NOW()
                    WHERE partition_name = $1
                """, partition_name)
                
            logger.info(f"✅ Stored metric record {record.record_id} in partition {partition_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metric record: {e}")
            return False
    
    async def query_metrics(self, tenant_id: str, industry_code: Optional[str] = None,
                          metric_type: Optional[str] = None, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Query metrics from Gold layer with partitioning"""
        if not self.db_pool:
            return []
            
        try:
            query = """
                SELECT record_id, tenant_id, industry_code, metric_type, metric_name,
                       metric_value, metric_unit, source_system, data_residency,
                       event_timestamp, metadata
                FROM metrics_gold 
                WHERE tenant_id = $1
            """
            params = [tenant_id]
            param_count = 1
            
            if industry_code:
                param_count += 1
                query += f" AND industry_code = ${param_count}"
                params.append(industry_code)
                
            if metric_type:
                param_count += 1
                query += f" AND metric_type = ${param_count}"
                params.append(metric_type)
                
            if start_date:
                param_count += 1
                query += f" AND event_timestamp >= ${param_count}"
                params.append(start_date)
                
            if end_date:
                param_count += 1
                query += f" AND event_timestamp <= ${param_count}"
                params.append(end_date)
                
            query += " ORDER BY event_timestamp DESC LIMIT 1000"
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "record_id": str(row['record_id']),
                        "tenant_id": row['tenant_id'],
                        "industry_code": row['industry_code'],
                        "metric_type": row['metric_type'],
                        "metric_name": row['metric_name'],
                        "metric_value": float(row['metric_value']),
                        "metric_unit": row['metric_unit'],
                        "source_system": row['source_system'],
                        "data_residency": row['data_residency'],
                        "event_timestamp": row['event_timestamp'].isoformat(),
                        "metadata": row['metadata']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to query metrics: {e}")
            return []

# Global storage instance
fabric_storage = FabricGoldStorage()

@app.on_event("startup")
async def startup_event():
    await fabric_storage.initialize()

@app.post("/fabric/metrics/store", response_model=Dict[str, Any])
async def store_metric(record: MetricRecord):
    """Store metric record in Fabric Gold layer"""
    success = await fabric_storage.store_metric_record(record)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store metric record")
        
    return {
        "status": "stored",
        "record_id": record.record_id,
        "partition_key": record.partition_key
    }

@app.get("/fabric/metrics/query")
async def query_metrics(
    tenant_id: str,
    industry_code: Optional[str] = None,
    metric_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Query metrics from Fabric Gold layer"""
    metrics = await fabric_storage.query_metrics(
        tenant_id, industry_code, metric_type, start_date, end_date
    )
    
    return {
        "tenant_id": tenant_id,
        "record_count": len(metrics),
        "metrics": metrics
    }

@app.get("/fabric/partitions")
async def list_partitions():
    """List all partitions in Fabric Gold layer"""
    if not fabric_storage.db_pool:
        raise HTTPException(status_code=503, detail="Storage not initialized")
        
    try:
        async with fabric_storage.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT partition_name, tenant_id, industry_code, data_residency,
                       record_count, size_mb, last_updated
                FROM partition_registry
                ORDER BY last_updated DESC
            """)
            
            return {
                "partitions": [
                    {
                        "partition_name": row['partition_name'],
                        "tenant_id": row['tenant_id'],
                        "industry_code": row['industry_code'],
                        "data_residency": row['data_residency'],
                        "record_count": row['record_count'],
                        "size_mb": float(row['size_mb']),
                        "last_updated": row['last_updated'].isoformat()
                    }
                    for row in rows
                ]
            }
            
    except Exception as e:
        logger.error(f"Failed to list partitions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list partitions")
