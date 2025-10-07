"""
Pub/Sub Event Bus with Schema Registry Service
Implements event bus with Kafka/Redis Streams and schema registry for event validation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import uuid
from datetime import datetime
import asyncio
import json
import jsonschema
from jsonschema import validate, ValidationError

app = FastAPI(
    title="Pub/Sub Event Bus with Schema Registry Service",
    description="Event bus with Kafka/Redis Streams and schema registry for event validation",
    version="1.0.0"
)

class EventType(str, Enum):
    CALENDAR_EVENT_CREATED = "calendar.event.created"
    CALENDAR_EVENT_UPDATED = "calendar.event.updated"
    CALENDAR_EVENT_DELETED = "calendar.event.deleted"
    MEETING_CAPTURED = "letsmeet.meeting.captured"
    TRANSCRIPTION_COMPLETED = "letsmeet.transcription.completed"
    SUMMARY_GENERATED = "letsmeet.summary.generated"
    ACTION_EXTRACTED = "letsmeet.action.extracted"
    CRUXX_ACTION_CREATED = "cruxx.action.created"
    CRUXX_ACTION_UPDATED = "cruxx.action.updated"
    CRUXX_ACTION_COMPLETED = "cruxx.action.completed"
    OPPORTUNITY_UPDATED = "crm.opportunity.updated"
    OPPORTUNITY_CLOSED = "crm.opportunity.closed"
    USER_OVERRIDE = "system.user.override"
    TRUST_DRIFT = "system.trust.drift"
    SLA_VIOLATION = "system.sla.violation"

class SchemaType(str, Enum):
    AVRO = "avro"
    JSON_SCHEMA = "json_schema"
    PROTOBUF = "protobuf"

class EventStatus(str, Enum):
    PENDING = "pending"
    PUBLISHED = "published"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"

class EventSchema(BaseModel):
    schema_id: str
    event_type: EventType
    schema_type: SchemaType
    version: str
    schema_definition: Dict[str, Any]
    tenant_id: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class EventMessage(BaseModel):
    event_id: str
    tenant_id: str
    event_type: EventType
    topic: str
    partition_key: Optional[str] = None
    payload: Dict[str, Any]
    headers: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EventSubscription(BaseModel):
    subscription_id: str
    tenant_id: str
    topic_pattern: str
    event_types: List[EventType]
    consumer_group: str
    handler_url: str
    is_active: bool = True
    retry_policy: Dict[str, Any] = Field(default_factory=dict)
    dead_letter_topic: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EventDelivery(BaseModel):
    delivery_id: str
    event_id: str
    subscription_id: str
    status: EventStatus
    attempt_count: int = 0
    max_attempts: int = 3
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None

# In-memory storage (replace with persistent storage in production)
event_schemas_db: Dict[str, EventSchema] = {}
event_messages_db: Dict[str, EventMessage] = {}
event_subscriptions_db: Dict[str, EventSubscription] = {}
event_deliveries_db: List[EventDelivery] = []

# Default event schemas
DEFAULT_SCHEMAS = [
    {
        "schema_id": "calendar_event_created_v1",
        "event_type": EventType.CALENDAR_EVENT_CREATED,
        "schema_type": SchemaType.JSON_SCHEMA,
        "version": "1.0.0",
        "schema_definition": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "tenant_id": {"type": "string"},
                "user_id": {"type": "string"},
                "title": {"type": "string"},
                "start_time": {"type": "string", "format": "date-time"},
                "end_time": {"type": "string", "format": "date-time"},
                "location": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"}
            },
            "required": ["event_id", "tenant_id", "user_id", "title", "start_time", "end_time"]
        },
        "tenant_id": "default"
    },
    {
        "schema_id": "meeting_captured_v1",
        "event_type": EventType.MEETING_CAPTURED,
        "schema_type": SchemaType.JSON_SCHEMA,
        "version": "1.0.0",
        "schema_definition": {
            "type": "object",
            "properties": {
                "meeting_id": {"type": "string"},
                "tenant_id": {"type": "string"},
                "user_id": {"type": "string"},
                "calendar_event_id": {"type": "string"},
                "audio_url": {"type": "string"},
                "duration_seconds": {"type": "number"},
                "participants": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"}
            },
            "required": ["meeting_id", "tenant_id", "user_id", "calendar_event_id", "audio_url"]
        },
        "tenant_id": "default"
    },
    {
        "schema_id": "cruxx_action_created_v1",
        "event_type": EventType.CRUXX_ACTION_CREATED,
        "schema_type": SchemaType.JSON_SCHEMA,
        "version": "1.0.0",
        "schema_definition": {
            "type": "object",
            "properties": {
                "cruxx_id": {"type": "string"},
                "tenant_id": {"type": "string"},
                "opportunity_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "assignee": {"type": "string"},
                "due_date": {"type": "string", "format": "date-time"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "source_meeting_id": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["cruxx_id", "tenant_id", "opportunity_id", "title", "assignee", "due_date"]
        },
        "tenant_id": "default"
    }
]

# Initialize default schemas
for schema_data in DEFAULT_SCHEMAS:
    schema = EventSchema(**schema_data)
    event_schemas_db[schema.schema_id] = schema

def validate_event_payload(event_type: EventType, payload: Dict[str, Any]) -> bool:
    """Validate event payload against schema"""
    # Find schema for event type
    schema = next((s for s in event_schemas_db.values() 
                  if s.event_type == event_type and s.is_active), None)
    
    if not schema:
        return True  # No schema defined, allow all
    
    try:
        validate(instance=payload, schema=schema.schema_definition)
        return True
    except ValidationError as e:
        print(f"Schema validation error: {e}")
        return False

async def simulate_event_delivery(event: EventMessage, subscription: EventSubscription) -> bool:
    """Simulate event delivery to subscriber"""
    try:
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Simulate success/failure (90% success rate)
        success = asyncio.get_event_loop().run_in_executor(None, lambda: __import__('random').random() < 0.9)
        return await success
    except Exception as e:
        print(f"Event delivery simulation error: {e}")
        return False

@app.post("/events/publish")
async def publish_event(event: EventMessage, background_tasks: BackgroundTasks):
    """Publish an event to the event bus"""
    
    # Validate event payload
    if not validate_event_payload(event.event_type, event.payload):
        raise HTTPException(status_code=400, detail="Event payload validation failed")
    
    event.event_id = str(uuid.uuid4())
    event.timestamp = datetime.utcnow()
    event_messages_db[event.event_id] = event
    
    # Find matching subscriptions
    matching_subscriptions = []
    for subscription in event_subscriptions_db.values():
        if (subscription.is_active and 
            subscription.tenant_id == event.tenant_id and
            event.event_type in subscription.event_types):
            matching_subscriptions.append(subscription)
    
    # Create delivery records and start delivery
    for subscription in matching_subscriptions:
        delivery = EventDelivery(
            delivery_id=str(uuid.uuid4()),
            event_id=event.event_id,
            subscription_id=subscription.subscription_id,
            status=EventStatus.PENDING,
            max_attempts=subscription.retry_policy.get("max_attempts", 3)
        )
        event_deliveries_db.append(delivery)
        
        # Start delivery in background
        background_tasks.add_task(process_event_delivery, delivery, event, subscription)
    
    return {
        "message": "Event published successfully",
        "event_id": event.event_id,
        "subscriptions_notified": len(matching_subscriptions)
    }

async def process_event_delivery(delivery: EventDelivery, event: EventMessage, subscription: EventSubscription):
    """Process event delivery to a subscription"""
    delivery.status = EventStatus.PENDING
    delivery.attempt_count += 1
    delivery.last_attempt = datetime.utcnow()
    
    try:
        # Simulate event delivery
        success = await simulate_event_delivery(event, subscription)
        
        if success:
            delivery.status = EventStatus.DELIVERED
            delivery.delivered_at = datetime.utcnow()
        else:
            if delivery.attempt_count >= delivery.max_attempts:
                delivery.status = EventStatus.DEAD_LETTERED
                delivery.error_message = "Max retry attempts exceeded"
            else:
                delivery.status = EventStatus.FAILED
                delivery.error_message = "Delivery failed"
                # Calculate next retry time (exponential backoff)
                delay_seconds = 2 ** delivery.attempt_count
                delivery.next_retry = datetime.utcnow() + timedelta(seconds=delay_seconds)
    
    except Exception as e:
        delivery.status = EventStatus.FAILED
        delivery.error_message = str(e)
        if delivery.attempt_count >= delivery.max_attempts:
            delivery.status = EventStatus.DEAD_LETTERED

@app.post("/schemas", response_model=EventSchema)
async def create_event_schema(schema: EventSchema):
    """Create a new event schema"""
    schema.schema_id = str(uuid.uuid4())
    schema.created_at = datetime.utcnow()
    schema.updated_at = datetime.utcnow()
    event_schemas_db[schema.schema_id] = schema
    return schema

@app.get("/schemas", response_model=List[EventSchema])
async def list_event_schemas(tenant_id: str):
    """List event schemas for a tenant"""
    return [schema for schema in event_schemas_db.values() 
            if schema.tenant_id == tenant_id or schema.tenant_id == "default"]

@app.get("/schemas/{schema_id}", response_model=EventSchema)
async def get_event_schema(schema_id: str):
    """Get event schema by ID"""
    schema = event_schemas_db.get(schema_id)
    if not schema:
        raise HTTPException(status_code=404, detail="Event schema not found")
    return schema

@app.put("/schemas/{schema_id}", response_model=EventSchema)
async def update_event_schema(schema_id: str, schema_update: Dict[str, Any]):
    """Update an event schema"""
    schema = event_schemas_db.get(schema_id)
    if not schema:
        raise HTTPException(status_code=404, detail="Event schema not found")
    
    for field, value in schema_update.items():
        if hasattr(schema, field):
            setattr(schema, field, value)
    
    schema.updated_at = datetime.utcnow()
    return schema

@app.post("/subscriptions", response_model=EventSubscription)
async def create_subscription(subscription: EventSubscription):
    """Create a new event subscription"""
    subscription.subscription_id = str(uuid.uuid4())
    subscription.created_at = datetime.utcnow()
    event_subscriptions_db[subscription.subscription_id] = subscription
    return subscription

@app.get("/subscriptions", response_model=List[EventSubscription])
async def list_subscriptions(tenant_id: str):
    """List event subscriptions for a tenant"""
    return [sub for sub in event_subscriptions_db.values() if sub.tenant_id == tenant_id]

@app.get("/events/{event_id}", response_model=EventMessage)
async def get_event(event_id: str):
    """Get event by ID"""
    event = event_messages_db.get(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@app.get("/events", response_model=List[EventMessage])
async def list_events(
    tenant_id: str,
    event_type: Optional[EventType] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100
):
    """List events for a tenant"""
    events = [e for e in event_messages_db.values() if e.tenant_id == tenant_id]
    
    if event_type:
        events = [e for e in events if e.event_type == event_type]
    
    if start_time:
        events = [e for e in events if e.timestamp >= start_time]
    
    if end_time:
        events = [e for e in events if e.timestamp <= end_time]
    
    # Sort by timestamp (most recent first)
    events.sort(key=lambda e: e.timestamp, reverse=True)
    
    return events[:limit]

@app.get("/deliveries", response_model=List[EventDelivery])
async def list_deliveries(
    tenant_id: str,
    status: Optional[EventStatus] = None,
    subscription_id: Optional[str] = None,
    limit: int = 100
):
    """List event deliveries for a tenant"""
    deliveries = []
    
    for delivery in event_deliveries_db:
        # Find the associated event
        event = event_messages_db.get(delivery.event_id)
        if event and event.tenant_id == tenant_id:
            if status is None or delivery.status == status:
                if subscription_id is None or delivery.subscription_id == subscription_id:
                    deliveries.append(delivery)
    
    # Sort by last attempt (most recent first)
    deliveries.sort(key=lambda d: d.last_attempt or datetime.min, reverse=True)
    
    return deliveries[:limit]

@app.get("/topics/stats")
async def get_topic_stats(tenant_id: str):
    """Get topic statistics"""
    events = [e for e in event_messages_db.values() if e.tenant_id == tenant_id]
    
    # Group by topic
    topic_stats = {}
    for event in events:
        topic = event.topic
        if topic not in topic_stats:
            topic_stats[topic] = {
                "total_events": 0,
                "event_types": {},
                "avg_payload_size": 0,
                "last_event": None
            }
        
        topic_stats[topic]["total_events"] += 1
        event_type = event.event_type.value
        topic_stats[topic]["event_types"][event_type] = topic_stats[topic]["event_types"].get(event_type, 0) + 1
        
        if not topic_stats[topic]["last_event"] or event.timestamp > topic_stats[topic]["last_event"]:
            topic_stats[topic]["last_event"] = event.timestamp
    
    # Calculate average payload size
    for topic, stats in topic_stats.items():
        topic_events = [e for e in events if e.topic == topic]
        if topic_events:
            total_size = sum(len(json.dumps(e.payload)) for e in topic_events)
            stats["avg_payload_size"] = total_size / len(topic_events)
    
    return {
        "tenant_id": tenant_id,
        "topics": topic_stats,
        "total_topics": len(topic_stats),
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "event-bus-schema-registry",
        "timestamp": datetime.utcnow().isoformat(),
        "schemas_count": len(event_schemas_db),
        "events_count": len(event_messages_db),
        "subscriptions_count": len(event_subscriptions_db),
        "deliveries_count": len(event_deliveries_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
