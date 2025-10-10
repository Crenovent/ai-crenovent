"""
Task 6.3.52: Event triggers (webhooks, Kafka) for scoring
Event-driven scoring system with webhook and message queue support
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Supported event types"""
    WEBHOOK = "webhook"
    KAFKA_MESSAGE = "kafka_message"
    SCHEDULED = "scheduled"
    API_TRIGGER = "api_trigger"

@dataclass
class EventTrigger:
    """Event trigger configuration"""
    trigger_id: str
    event_type: EventType
    model_id: str
    trigger_config: Dict[str, Any]
    debounce_window_ms: int = 1000
    enabled: bool = True

@dataclass
class ScoringEvent:
    """Scoring event data"""
    event_id: str
    trigger_id: str
    event_type: EventType
    payload: Dict[str, Any]
    received_at: datetime
    processed_at: Optional[datetime] = None

class EventTriggerService:
    """
    Event-driven scoring service
    Task 6.3.52: Reactive patterns with debounce window
    """
    
    def __init__(self):
        self.triggers: Dict[str, EventTrigger] = {}
        self.event_handlers: Dict[str, Callable] = {}
        self.event_history: List[ScoringEvent] = []
        self.debounce_tasks: Dict[str, asyncio.Task] = {}
    
    def register_event_trigger(self, trigger: EventTrigger, handler: Callable) -> bool:
        """Register an event trigger with handler"""
        try:
            self.triggers[trigger.trigger_id] = trigger
            self.event_handlers[trigger.trigger_id] = handler
            
            logger.info(f"Registered event trigger: {trigger.trigger_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register event trigger: {e}")
            return False
    
    async def handle_webhook_event(self, trigger_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook event"""
        if trigger_id not in self.triggers:
            raise ValueError(f"Unknown trigger: {trigger_id}")
        
        trigger = self.triggers[trigger_id]
        if trigger.event_type != EventType.WEBHOOK or not trigger.enabled:
            raise ValueError(f"Invalid webhook trigger: {trigger_id}")
        
        # Create scoring event
        event = ScoringEvent(
            event_id=f"webhook_{datetime.utcnow().timestamp()}",
            trigger_id=trigger_id,
            event_type=EventType.WEBHOOK,
            payload=payload,
            received_at=datetime.utcnow()
        )
        
        # Process with debouncing
        result = await self._process_event_with_debounce(event)
        return result
    
    async def handle_kafka_message(self, trigger_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Kafka message event"""
        if trigger_id not in self.triggers:
            raise ValueError(f"Unknown trigger: {trigger_id}")
        
        trigger = self.triggers[trigger_id]
        if trigger.event_type != EventType.KAFKA_MESSAGE or not trigger.enabled:
            raise ValueError(f"Invalid Kafka trigger: {trigger_id}")
        
        # Create scoring event
        event = ScoringEvent(
            event_id=f"kafka_{datetime.utcnow().timestamp()}",
            trigger_id=trigger_id,
            event_type=EventType.KAFKA_MESSAGE,
            payload=message,
            received_at=datetime.utcnow()
        )
        
        # Process with debouncing
        result = await self._process_event_with_debounce(event)
        return result
    
    async def _process_event_with_debounce(self, event: ScoringEvent) -> Dict[str, Any]:
        """Process event with debouncing"""
        trigger = self.triggers[event.trigger_id]
        
        # Cancel existing debounce task if any
        if event.trigger_id in self.debounce_tasks:
            self.debounce_tasks[event.trigger_id].cancel()
        
        # Create new debounce task
        debounce_task = asyncio.create_task(
            self._debounced_process(event, trigger.debounce_window_ms)
        )
        self.debounce_tasks[event.trigger_id] = debounce_task
        
        try:
            result = await debounce_task
            return result
        except asyncio.CancelledError:
            logger.info(f"Event processing cancelled for debouncing: {event.event_id}")
            raise
        finally:
            # Clean up task
            if event.trigger_id in self.debounce_tasks:
                del self.debounce_tasks[event.trigger_id]
    
    async def _debounced_process(self, event: ScoringEvent, debounce_ms: int) -> Dict[str, Any]:
        """Process event after debounce window"""
        # Wait for debounce window
        await asyncio.sleep(debounce_ms / 1000)
        
        # Process the event
        return await self._process_scoring_event(event)
    
    async def _process_scoring_event(self, event: ScoringEvent) -> Dict[str, Any]:
        """Process scoring event"""
        try:
            trigger = self.triggers[event.trigger_id]
            handler = self.event_handlers[event.trigger_id]
            
            # Extract input data from payload
            input_data = self._extract_input_data(event.payload, trigger.trigger_config)
            
            # Call handler
            result = await handler(trigger.model_id, input_data)
            
            # Mark as processed
            event.processed_at = datetime.utcnow()
            self.event_history.append(event)
            
            # Clean up old events
            self._cleanup_event_history()
            
            logger.info(f"Processed scoring event: {event.event_id}")
            
            return {
                "event_id": event.event_id,
                "trigger_id": event.trigger_id,
                "model_id": trigger.model_id,
                "result": result,
                "processed_at": event.processed_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process scoring event {event.event_id}: {e}")
            raise
    
    def _extract_input_data(self, payload: Dict[str, Any], trigger_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract input data from event payload"""
        # Simple extraction based on configuration
        input_mapping = trigger_config.get("input_mapping", {})
        
        if not input_mapping:
            # Use payload as-is if no mapping configured
            return payload
        
        input_data = {}
        for target_field, source_path in input_mapping.items():
            # Simple dot notation support
            value = payload
            for key in source_path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            if value is not None:
                input_data[target_field] = value
        
        return input_data
    
    def _cleanup_event_history(self) -> None:
        """Clean up old events from history"""
        # Keep only last 1000 events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
    
    def create_webhook_trigger(self, model_id: str, trigger_id: str, input_mapping: Dict[str, str]) -> EventTrigger:
        """Create webhook trigger configuration"""
        return EventTrigger(
            trigger_id=trigger_id,
            event_type=EventType.WEBHOOK,
            model_id=model_id,
            trigger_config={
                "input_mapping": input_mapping,
                "webhook_path": f"/webhooks/{trigger_id}"
            }
        )
    
    def create_kafka_trigger(self, model_id: str, trigger_id: str, topic: str, input_mapping: Dict[str, str]) -> EventTrigger:
        """Create Kafka trigger configuration"""
        return EventTrigger(
            trigger_id=trigger_id,
            event_type=EventType.KAFKA_MESSAGE,
            model_id=model_id,
            trigger_config={
                "topic": topic,
                "input_mapping": input_mapping,
                "consumer_group": f"scoring_{trigger_id}"
            }
        )
    
    def get_trigger_stats(self, trigger_id: str) -> Dict[str, Any]:
        """Get statistics for a trigger"""
        if trigger_id not in self.triggers:
            return {}
        
        trigger = self.triggers[trigger_id]
        
        # Count events for this trigger
        trigger_events = [e for e in self.event_history if e.trigger_id == trigger_id]
        processed_events = [e for e in trigger_events if e.processed_at is not None]
        
        return {
            "trigger_id": trigger_id,
            "model_id": trigger.model_id,
            "event_type": trigger.event_type.value,
            "enabled": trigger.enabled,
            "total_events": len(trigger_events),
            "processed_events": len(processed_events),
            "success_rate": len(processed_events) / len(trigger_events) if trigger_events else 0,
            "debounce_window_ms": trigger.debounce_window_ms
        }
    
    async def simulate_kafka_consumer(self, trigger_id: str) -> None:
        """Simulate Kafka consumer for testing"""
        if trigger_id not in self.triggers:
            return
        
        trigger = self.triggers[trigger_id]
        if trigger.event_type != EventType.KAFKA_MESSAGE:
            return
        
        logger.info(f"Starting Kafka consumer simulation for trigger: {trigger_id}")
        
        # Simulate receiving messages
        while trigger.enabled:
            try:
                # Simulate message
                message = {
                    "topic": trigger.trigger_config.get("topic", "default"),
                    "key": f"key_{datetime.utcnow().timestamp()}",
                    "value": {
                        "customer_id": "12345",
                        "event_type": "purchase",
                        "amount": 99.99,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                await self.handle_kafka_message(trigger_id, message)
                await asyncio.sleep(10)  # Wait 10 seconds between messages
                
            except Exception as e:
                logger.error(f"Kafka consumer error for {trigger_id}: {e}")
                await asyncio.sleep(5)

# Global event trigger service
event_trigger_service = EventTriggerService()
