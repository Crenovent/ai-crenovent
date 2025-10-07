"""
Dead Letter Queue (DLQ) + Replay Tooling Service
Implements DLQ management, replay functionality, and SLO monitoring for time-to-recovery
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime, timedelta
import asyncio
import json

app = FastAPI(
    title="DLQ + Replay Tooling Service",
    description="Dead Letter Queue management, replay functionality, and SLO monitoring",
    version="1.0.0"
)

class DLQStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    REPLAYED = "replayed"
    FAILED = "failed"
    EXPIRED = "expired"

class ReplayStrategy(str, Enum):
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    SCHEDULED = "scheduled"

class SLOTarget(str, Enum):
    TIME_TO_RECOVERY = "time_to_recovery"
    REPLAY_SUCCESS_RATE = "replay_success_rate"
    DLQ_SIZE = "dlq_size"
    PROCESSING_LATENCY = "processing_latency"

class DLQMessage(BaseModel):
    message_id: str
    tenant_id: str
    original_topic: str
    original_message: Dict[str, Any]
    failure_reason: str
    error_details: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    status: DLQStatus = DLQStatus.PENDING
    replay_strategy: ReplayStrategy = ReplayStrategy.EXPONENTIAL_BACKOFF
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReplayRequest(BaseModel):
    dlq_message_id: str
    replay_strategy: Optional[ReplayStrategy] = None
    max_retries: Optional[int] = None
    delay_seconds: Optional[int] = None
    target_topic: Optional[str] = None

class ReplayBatchRequest(BaseModel):
    tenant_id: str
    topic_filter: Optional[str] = None
    status_filter: Optional[DLQStatus] = None
    replay_strategy: ReplayStrategy = ReplayStrategy.EXPONENTIAL_BACKOFF
    max_messages: int = 100

class SLOMetric(BaseModel):
    metric_id: str
    tenant_id: str
    target_type: SLOTarget
    target_value: float
    actual_value: float
    measurement_time: datetime
    status: str  # met, warning, violated
    details: Dict[str, Any] = Field(default_factory=dict)

class SLOTarget(BaseModel):
    target_id: str
    tenant_id: str
    target_type: SLOTarget
    target_value: float
    warning_threshold: float
    critical_threshold: float
    measurement_window_minutes: int = 60
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage (replace with persistent storage in production)
dlq_messages_db: Dict[str, DLQMessage] = {}
slo_metrics_db: List[SLOMetric] = []
slo_targets_db: Dict[str, SLOTarget] = {}

# Default SLO targets
DEFAULT_SLO_TARGETS = [
    {
        "target_id": "time_to_recovery_default",
        "tenant_id": "default",
        "target_type": SLOTarget.TIME_TO_RECOVERY,
        "target_value": 300.0,  # 5 minutes
        "warning_threshold": 240.0,  # 4 minutes
        "critical_threshold": 180.0,  # 3 minutes
        "measurement_window_minutes": 60
    },
    {
        "target_id": "replay_success_rate_default",
        "tenant_id": "default",
        "target_type": SLOTarget.REPLAY_SUCCESS_RATE,
        "target_value": 0.95,  # 95%
        "warning_threshold": 0.90,  # 90%
        "critical_threshold": 0.85,  # 85%
        "measurement_window_minutes": 60
    },
    {
        "target_id": "dlq_size_default",
        "tenant_id": "default",
        "target_type": SLOTarget.DLQ_SIZE,
        "target_value": 100.0,  # 100 messages
        "warning_threshold": 200.0,  # 200 messages
        "critical_threshold": 500.0,  # 500 messages
        "measurement_window_minutes": 15
    }
]

# Initialize default SLO targets
for target_data in DEFAULT_SLO_TARGETS:
    target = SLOTarget(**target_data)
    slo_targets_db[target.target_id] = target

def calculate_next_retry_time(retry_count: int, strategy: ReplayStrategy, base_delay: int = 60) -> datetime:
    """Calculate next retry time based on strategy"""
    now = datetime.utcnow()
    
    if strategy == ReplayStrategy.IMMEDIATE:
        return now
    elif strategy == ReplayStrategy.LINEAR_BACKOFF:
        delay_seconds = base_delay * (retry_count + 1)
    elif strategy == ReplayStrategy.EXPONENTIAL_BACKOFF:
        delay_seconds = base_delay * (2 ** retry_count)
    elif strategy == ReplayStrategy.SCHEDULED:
        # Schedule for next hour
        delay_seconds = 3600
    else:
        delay_seconds = base_delay
    
    return now + timedelta(seconds=delay_seconds)

async def simulate_message_replay(message: DLQMessage) -> bool:
    """Simulate replaying a message (replace with actual message publishing)"""
    try:
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Simulate success/failure based on retry count
        success_rate = max(0.1, 1.0 - (message.retry_count * 0.2))
        success = asyncio.get_event_loop().run_in_executor(None, lambda: __import__('random').random() < success_rate)
        
        return await success
    except Exception as e:
        print(f"Replay simulation error: {e}")
        return False

@app.post("/dlq/messages")
async def add_dlq_message(message: DLQMessage):
    """Add a message to the Dead Letter Queue"""
    message.message_id = str(uuid.uuid4())
    message.created_at = datetime.utcnow()
    message.status = DLQStatus.PENDING
    message.next_retry = calculate_next_retry_time(message.retry_count, message.replay_strategy)
    
    dlq_messages_db[message.message_id] = message
    
    # Record SLO metric for DLQ size
    await record_slo_metric(
        tenant_id=message.tenant_id,
        target_type=SLOTarget.DLQ_SIZE,
        actual_value=len([m for m in dlq_messages_db.values() if m.tenant_id == message.tenant_id]),
        details={"action": "message_added", "topic": message.original_topic}
    )
    
    return {
        "message": "Message added to DLQ successfully",
        "message_id": message.message_id,
        "next_retry": message.next_retry.isoformat()
    }

@app.get("/dlq/messages", response_model=List[DLQMessage])
async def list_dlq_messages(
    tenant_id: str,
    status: Optional[DLQStatus] = None,
    topic: Optional[str] = None,
    limit: int = 100
):
    """List DLQ messages for a tenant"""
    messages = [m for m in dlq_messages_db.values() if m.tenant_id == tenant_id]
    
    if status:
        messages = [m for m in messages if m.status == status]
    
    if topic:
        messages = [m for m in messages if m.original_topic == topic]
    
    # Sort by creation time (oldest first)
    messages.sort(key=lambda m: m.created_at)
    
    return messages[:limit]

@app.post("/dlq/replay")
async def replay_message(request: ReplayRequest, background_tasks: BackgroundTasks):
    """Replay a single DLQ message"""
    message = dlq_messages_db.get(request.dlq_message_id)
    if not message:
        raise HTTPException(status_code=404, detail="DLQ message not found")
    
    if message.status == DLQStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Message is already being processed")
    
    # Update replay parameters if provided
    if request.replay_strategy:
        message.replay_strategy = request.replay_strategy
    if request.max_retries:
        message.max_retries = request.max_retries
    
    # Start replay in background
    background_tasks.add_task(process_replay, message, request.target_topic)
    
    return {
        "message": "Replay initiated successfully",
        "message_id": request.dlq_message_id,
        "status": "processing"
    }

@app.post("/dlq/replay/batch")
async def replay_batch(request: ReplayBatchRequest, background_tasks: BackgroundTasks):
    """Replay multiple DLQ messages"""
    messages = [m for m in dlq_messages_db.values() 
               if m.tenant_id == request.tenant_id and m.status == DLQStatus.PENDING]
    
    if request.topic_filter:
        messages = [m for m in messages if m.original_topic == request.topic_filter]
    
    if request.status_filter:
        messages = [m for m in messages if m.status == request.status_filter]
    
    # Limit number of messages
    messages = messages[:request.max_messages]
    
    if not messages:
        return {
            "message": "No messages found for replay",
            "replayed_count": 0
        }
    
    # Start batch replay in background
    background_tasks.add_task(process_batch_replay, messages, request.replay_strategy)
    
    return {
        "message": "Batch replay initiated successfully",
        "messages_count": len(messages),
        "status": "processing"
    }

async def process_replay(message: DLQMessage, target_topic: Optional[str] = None):
    """Process a single message replay"""
    message.status = DLQStatus.PROCESSING
    message.last_attempt = datetime.utcnow()
    
    try:
        # Simulate message replay
        success = await simulate_message_replay(message)
        
        if success:
            message.status = DLQStatus.REPLAYED
            message.retry_count += 1
            
            # Record successful replay
            await record_slo_metric(
                tenant_id=message.tenant_id,
                target_type=SLOTarget.REPLAY_SUCCESS_RATE,
                actual_value=1.0,
                details={"message_id": message.message_id, "retry_count": message.retry_count}
            )
        else:
            message.retry_count += 1
            
            if message.retry_count >= message.max_retries:
                message.status = DLQStatus.FAILED
            else:
                message.status = DLQStatus.PENDING
                message.next_retry = calculate_next_retry_time(
                    message.retry_count, 
                    message.replay_strategy
                )
            
            # Record failed replay
            await record_slo_metric(
                tenant_id=message.tenant_id,
                target_type=SLOTarget.REPLAY_SUCCESS_RATE,
                actual_value=0.0,
                details={"message_id": message.message_id, "retry_count": message.retry_count}
            )
    
    except Exception as e:
        message.status = DLQStatus.FAILED
        message.metadata["error"] = str(e)
        
        # Record error
        await record_slo_metric(
            tenant_id=message.tenant_id,
            target_type=SLOTarget.REPLAY_SUCCESS_RATE,
            actual_value=0.0,
            details={"error": str(e), "message_id": message.message_id}
        )

async def process_batch_replay(messages: List[DLQMessage], strategy: ReplayStrategy):
    """Process batch replay of multiple messages"""
    successful_replays = 0
    failed_replays = 0
    
    for message in messages:
        try:
            await process_replay(message)
            if message.status == DLQStatus.REPLAYED:
                successful_replays += 1
            else:
                failed_replays += 1
        except Exception as e:
            failed_replays += 1
            print(f"Batch replay error for message {message.message_id}: {e}")
    
    # Record batch replay metrics
    if messages:
        tenant_id = messages[0].tenant_id
        success_rate = successful_replays / len(messages)
        
        await record_slo_metric(
            tenant_id=tenant_id,
            target_type=SLOTarget.REPLAY_SUCCESS_RATE,
            actual_value=success_rate,
            details={
                "batch_size": len(messages),
                "successful": successful_replays,
                "failed": failed_replays
            }
        )

@app.get("/dlq/stats")
async def get_dlq_stats(tenant_id: str):
    """Get DLQ statistics for a tenant"""
    messages = [m for m in dlq_messages_db.values() if m.tenant_id == tenant_id]
    
    stats = {
        "tenant_id": tenant_id,
        "total_messages": len(messages),
        "pending_messages": len([m for m in messages if m.status == DLQStatus.PENDING]),
        "processing_messages": len([m for m in messages if m.status == DLQStatus.PROCESSING]),
        "replayed_messages": len([m for m in messages if m.status == DLQStatus.REPLAYED]),
        "failed_messages": len([m for m in messages if m.status == DLQStatus.FAILED]),
        "expired_messages": len([m for m in messages if m.status == DLQStatus.EXPIRED]),
        "avg_retry_count": sum(m.retry_count for m in messages) / len(messages) if messages else 0,
        "top_topics": {},
        "generated_at": datetime.utcnow().isoformat()
    }
    
    # Calculate top topics
    topic_counts = {}
    for message in messages:
        topic = message.original_topic
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    stats["top_topics"] = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return stats

@app.post("/slo/targets", response_model=SLOTarget)
async def create_slo_target(target: SLOTarget):
    """Create a new SLO target"""
    target.target_id = str(uuid.uuid4())
    target.created_at = datetime.utcnow()
    slo_targets_db[target.target_id] = target
    return target

@app.get("/slo/targets", response_model=List[SLOTarget])
async def list_slo_targets(tenant_id: str):
    """List SLO targets for a tenant"""
    return [target for target in slo_targets_db.values() 
            if target.tenant_id == tenant_id or target.tenant_id == "default"]

async def record_slo_metric(tenant_id: str, target_type: SLOTarget, actual_value: float, details: Dict[str, Any]):
    """Record an SLO metric"""
    metric = SLOMetric(
        metric_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        target_type=target_type,
        target_value=0.0,  # Will be set based on target
        actual_value=actual_value,
        measurement_time=datetime.utcnow(),
        status="met",
        details=details
    )
    
    # Find matching SLO target
    target = next((t for t in slo_targets_db.values() 
                  if t.tenant_id in [tenant_id, "default"] and t.target_type == target_type), None)
    
    if target:
        metric.target_value = target.target_value
        
        if actual_value >= target.target_value:
            metric.status = "met"
        elif actual_value >= target.warning_threshold:
            metric.status = "warning"
        else:
            metric.status = "violated"
    
    slo_metrics_db.append(metric)

@app.get("/slo/metrics", response_model=List[SLOMetric])
async def get_slo_metrics(
    tenant_id: str,
    target_type: Optional[SLOTarget] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100
):
    """Get SLO metrics for a tenant"""
    metrics = [m for m in slo_metrics_db if m.tenant_id == tenant_id]
    
    if target_type:
        metrics = [m for m in metrics if m.target_type == target_type]
    
    if start_time:
        metrics = [m for m in metrics if m.measurement_time >= start_time]
    
    if end_time:
        metrics = [m for m in metrics if m.measurement_time <= end_time]
    
    # Sort by measurement time (most recent first)
    metrics.sort(key=lambda m: m.measurement_time, reverse=True)
    
    return metrics[:limit]

@app.get("/slo/dashboard")
async def get_slo_dashboard(tenant_id: str):
    """Get SLO dashboard data"""
    # Get recent metrics (last hour)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    
    recent_metrics = [m for m in slo_metrics_db 
                     if m.tenant_id == tenant_id and start_time <= m.measurement_time <= end_time]
    
    # Calculate SLO status for each target type
    dashboard = {}
    for target_type in SLOTarget:
        target_metrics = [m for m in recent_metrics if m.target_type == target_type]
        
        if target_metrics:
            avg_value = sum(m.actual_value for m in target_metrics) / len(target_metrics)
            violation_count = len([m for m in target_metrics if m.status == "violated"])
            warning_count = len([m for m in target_metrics if m.status == "warning"])
            
            dashboard[target_type.value] = {
                "current_value": avg_value,
                "violation_count": violation_count,
                "warning_count": warning_count,
                "total_measurements": len(target_metrics),
                "status": "met" if violation_count == 0 else "warning" if warning_count > violation_count else "violated"
            }
        else:
            dashboard[target_type.value] = {
                "current_value": 0.0,
                "violation_count": 0,
                "warning_count": 0,
                "total_measurements": 0,
                "status": "no_data"
            }
    
    return {
        "tenant_id": tenant_id,
        "period": "1h",
        "slo_status": dashboard,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "dlq-replay-tooling",
        "timestamp": datetime.utcnow().isoformat(),
        "dlq_messages_count": len(dlq_messages_db),
        "slo_metrics_count": len(slo_metrics_db),
        "slo_targets_count": len(slo_targets_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
