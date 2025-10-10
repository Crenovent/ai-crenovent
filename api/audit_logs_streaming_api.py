"""
Task 8.2-T19: Implement streaming API for real-time audit feeds
WebSocket + SSE for real-time audit monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, AsyncGenerator
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import asyncpg
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of audit streams"""
    WEBSOCKET = "websocket"
    SSE = "sse"


class FilterType(Enum):
    """Stream filter types"""
    TENANT = "tenant"
    EVENT_TYPE = "event_type"
    ACTOR = "actor"
    ACTION = "action"
    RESOURCE_TYPE = "resource_type"
    SEVERITY = "severity"


@dataclass
class StreamFilter:
    """Stream filter configuration"""
    filter_type: FilterType
    values: List[str] = field(default_factory=list)
    exclude: bool = False


@dataclass
class StreamSubscription:
    """Stream subscription configuration"""
    subscription_id: str
    client_id: str
    tenant_id: int
    
    # Filters
    filters: List[StreamFilter] = field(default_factory=list)
    
    # Configuration
    include_metadata: bool = True
    buffer_size: int = 1000
    max_events_per_second: int = 100
    
    # Connection details
    stream_type: StreamType = StreamType.WEBSOCKET
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Statistics
    events_sent: int = 0
    events_filtered: int = 0
    connection_errors: int = 0


@dataclass
class AuditStreamEvent:
    """Audit event for streaming"""
    event_id: str
    tenant_id: int
    event_type: str
    actor_id: str
    action: str
    resource_type: str
    resource_id: str
    
    # Event data
    event_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    event_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stream_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Stream metadata
    subscription_id: Optional[str] = None
    sequence_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "event_data": self.event_data,
            "metadata": self.metadata,
            "event_timestamp": self.event_timestamp.isoformat(),
            "stream_timestamp": self.stream_timestamp.isoformat(),
            "subscription_id": self.subscription_id,
            "sequence_number": self.sequence_number
        }
    
    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format"""
        data = json.dumps(self.to_dict())
        return f"data: {data}\n\n"


class AuditLogsStreamingAPI:
    """
    Audit Logs Streaming API - Task 8.2-T19
    
    Provides real-time audit log streaming via WebSocket and Server-Sent Events
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'max_concurrent_connections': 1000,
            'heartbeat_interval_seconds': 30,
            'connection_timeout_seconds': 300,
            'max_buffer_size': 10000,
            'enable_compression': True,
            'rate_limit_per_client': 1000,  # events per minute
            'cleanup_interval_seconds': 60
        }
        
        # Active connections
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.sse_connections: Dict[str, Any] = {}
        self.subscriptions: Dict[str, StreamSubscription] = {}
        
        # Event buffers
        self.event_buffers: Dict[str, List[AuditStreamEvent]] = {}
        
        # Statistics
        self.streaming_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_events_streamed': 0,
            'events_per_second': 0.0,
            'average_latency_ms': 0.0,
            'connection_errors': 0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    async def initialize(self) -> bool:
        """Initialize streaming API"""
        try:
            await self._create_streaming_tables()
            await self._start_background_tasks()
            self.is_running = True
            self.logger.info("✅ Audit logs streaming API initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize streaming API: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown streaming API"""
        self.is_running = False
        
        # Close all connections
        for websocket in self.websocket_connections.values():
            try:
                await websocket.close()
            except:
                pass
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("🛑 Audit logs streaming API shutdown")
    
    async def handle_websocket_connection(
        self,
        websocket: WebSocket,
        client_id: str,
        tenant_id: int,
        filters: List[Dict[str, Any]] = None
    ):
        """
        Handle WebSocket connection for real-time audit streaming
        
        WebSocket endpoint: /api/v1/audit-logs/stream/ws
        """
        
        try:
            # Accept connection
            await websocket.accept()
            
            # Check connection limits
            if len(self.websocket_connections) >= self.config['max_concurrent_connections']:
                await websocket.close(code=1013, reason="Too many connections")
                return
            
            # Create subscription
            subscription_id = f"ws_{client_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            subscription = StreamSubscription(
                subscription_id=subscription_id,
                client_id=client_id,
                tenant_id=tenant_id,
                stream_type=StreamType.WEBSOCKET,
                filters=self._parse_filters(filters or [])
            )
            
            # Register connection
            self.websocket_connections[subscription_id] = websocket
            self.subscriptions[subscription_id] = subscription
            self.event_buffers[subscription_id] = []
            
            self.streaming_stats['total_connections'] += 1
            self.streaming_stats['active_connections'] += 1
            
            self.logger.info(f"🔌 WebSocket connected: {subscription_id} (tenant: {tenant_id})")
            
            # Send connection confirmation
            await websocket.send_json({
                "type": "connection_established",
                "subscription_id": subscription_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Handle messages
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=self.config['connection_timeout_seconds']
                    )
                    
                    await self._handle_websocket_message(websocket, subscription, message)
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                except WebSocketDisconnect:
                    break
                    
        except Exception as e:
            self.logger.error(f"❌ WebSocket error: {e}")
            
        finally:
            # Cleanup connection
            if subscription_id in self.websocket_connections:
                del self.websocket_connections[subscription_id]
            if subscription_id in self.subscriptions:
                del self.subscriptions[subscription_id]
            if subscription_id in self.event_buffers:
                del self.event_buffers[subscription_id]
            
            self.streaming_stats['active_connections'] -= 1
            
            self.logger.info(f"🔌 WebSocket disconnected: {subscription_id}")
    
    async def handle_sse_connection(
        self,
        client_id: str,
        tenant_id: int,
        filters: List[Dict[str, Any]] = None
    ) -> StreamingResponse:
        """
        Handle Server-Sent Events connection for real-time audit streaming
        
        SSE endpoint: /api/v1/audit-logs/stream/sse
        """
        
        try:
            # Check connection limits
            if len(self.sse_connections) >= self.config['max_concurrent_connections']:
                raise HTTPException(status_code=429, detail="Too many connections")
            
            # Create subscription
            subscription_id = f"sse_{client_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            subscription = StreamSubscription(
                subscription_id=subscription_id,
                client_id=client_id,
                tenant_id=tenant_id,
                stream_type=StreamType.SSE,
                filters=self._parse_filters(filters or [])
            )
            
            # Register connection
            self.subscriptions[subscription_id] = subscription
            self.event_buffers[subscription_id] = []
            
            self.streaming_stats['total_connections'] += 1
            self.streaming_stats['active_connections'] += 1
            
            self.logger.info(f"📡 SSE connected: {subscription_id} (tenant: {tenant_id})")
            
            # Create event generator
            async def event_generator():
                try:
                    # Send connection established event
                    yield f"data: {json.dumps({'type': 'connection_established', 'subscription_id': subscription_id, 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
                    
                    last_heartbeat = datetime.now(timezone.utc)
                    
                    while subscription_id in self.subscriptions:
                        # Check for buffered events
                        if subscription_id in self.event_buffers and self.event_buffers[subscription_id]:
                            events = self.event_buffers[subscription_id].copy()
                            self.event_buffers[subscription_id].clear()
                            
                            for event in events:
                                yield event.to_sse_format()
                        
                        # Send heartbeat if needed
                        now = datetime.now(timezone.utc)
                        if (now - last_heartbeat).total_seconds() >= self.config['heartbeat_interval_seconds']:
                            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': now.isoformat()})}\n\n"
                            last_heartbeat = now
                        
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    self.logger.error(f"❌ SSE generator error: {e}")
                    
                finally:
                    # Cleanup
                    if subscription_id in self.subscriptions:
                        del self.subscriptions[subscription_id]
                    if subscription_id in self.event_buffers:
                        del self.event_buffers[subscription_id]
                    
                    self.streaming_stats['active_connections'] -= 1
                    self.logger.info(f"📡 SSE disconnected: {subscription_id}")
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ SSE connection error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to establish SSE connection: {str(e)}")
    
    async def stream_audit_event(self, audit_event: Dict[str, Any]):
        """
        Stream audit event to all matching subscriptions
        
        This method is called by the audit ingestion pipeline
        """
        
        try:
            # Convert to stream event
            stream_event = AuditStreamEvent(
                event_id=audit_event.get('event_id', ''),
                tenant_id=audit_event.get('tenant_id', 0),
                event_type=audit_event.get('event_type', ''),
                actor_id=audit_event.get('actor_id', ''),
                action=audit_event.get('action', ''),
                resource_type=audit_event.get('resource_type', ''),
                resource_id=audit_event.get('resource_id', ''),
                event_data=audit_event.get('event_data', {}),
                metadata=audit_event.get('metadata', {}),
                event_timestamp=audit_event.get('event_timestamp', datetime.now(timezone.utc))
            )
            
            # Find matching subscriptions
            matching_subscriptions = self._find_matching_subscriptions(stream_event)
            
            # Stream to matching subscriptions
            for subscription_id in matching_subscriptions:
                await self._send_to_subscription(subscription_id, stream_event)
            
            self.streaming_stats['total_events_streamed'] += len(matching_subscriptions)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stream audit event: {e}")
    
    def _parse_filters(self, filters_data: List[Dict[str, Any]]) -> List[StreamFilter]:
        """Parse filter configuration"""
        
        filters = []
        
        for filter_data in filters_data:
            try:
                filter_type = FilterType(filter_data.get('type'))
                values = filter_data.get('values', [])
                exclude = filter_data.get('exclude', False)
                
                filters.append(StreamFilter(
                    filter_type=filter_type,
                    values=values,
                    exclude=exclude
                ))
                
            except (ValueError, KeyError) as e:
                self.logger.warning(f"⚠️ Invalid filter configuration: {filter_data} - {e}")
        
        return filters
    
    def _find_matching_subscriptions(self, event: AuditStreamEvent) -> List[str]:
        """Find subscriptions that match the event"""
        
        matching_subscriptions = []
        
        for subscription_id, subscription in self.subscriptions.items():
            # Check tenant filter
            if subscription.tenant_id != event.tenant_id:
                continue
            
            # Check custom filters
            if self._event_matches_filters(event, subscription.filters):
                matching_subscriptions.append(subscription_id)
        
        return matching_subscriptions
    
    def _event_matches_filters(self, event: AuditStreamEvent, filters: List[StreamFilter]) -> bool:
        """Check if event matches subscription filters"""
        
        if not filters:
            return True  # No filters = match all
        
        for filter_config in filters:
            field_value = None
            
            # Get field value based on filter type
            if filter_config.filter_type == FilterType.TENANT:
                field_value = str(event.tenant_id)
            elif filter_config.filter_type == FilterType.EVENT_TYPE:
                field_value = event.event_type
            elif filter_config.filter_type == FilterType.ACTOR:
                field_value = event.actor_id
            elif filter_config.filter_type == FilterType.ACTION:
                field_value = event.action
            elif filter_config.filter_type == FilterType.RESOURCE_TYPE:
                field_value = event.resource_type
            elif filter_config.filter_type == FilterType.SEVERITY:
                field_value = event.metadata.get('severity', 'medium')
            
            if field_value is None:
                continue
            
            # Check filter match
            is_match = field_value in filter_config.values
            
            # Apply exclude logic
            if filter_config.exclude:
                if is_match:
                    return False  # Excluded
            else:
                if not is_match:
                    return False  # Required but not matched
        
        return True
    
    async def _send_to_subscription(self, subscription_id: str, event: AuditStreamEvent):
        """Send event to specific subscription"""
        
        try:
            subscription = self.subscriptions.get(subscription_id)
            if not subscription:
                return
            
            # Set subscription metadata
            event.subscription_id = subscription_id
            event.sequence_number = subscription.events_sent + 1
            
            # Rate limiting check
            if subscription.events_sent >= subscription.max_events_per_second:
                subscription.events_filtered += 1
                return
            
            if subscription.stream_type == StreamType.WEBSOCKET:
                await self._send_websocket_event(subscription_id, event)
            elif subscription.stream_type == StreamType.SSE:
                await self._send_sse_event(subscription_id, event)
            
            subscription.events_sent += 1
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send event to subscription {subscription_id}: {e}")
            subscription = self.subscriptions.get(subscription_id)
            if subscription:
                subscription.connection_errors += 1
    
    async def _send_websocket_event(self, subscription_id: str, event: AuditStreamEvent):
        """Send event via WebSocket"""
        
        websocket = self.websocket_connections.get(subscription_id)
        if not websocket:
            return
        
        try:
            await websocket.send_json({
                "type": "audit_event",
                "event": event.to_dict()
            })
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket send error: {e}")
            # Remove failed connection
            if subscription_id in self.websocket_connections:
                del self.websocket_connections[subscription_id]
    
    async def _send_sse_event(self, subscription_id: str, event: AuditStreamEvent):
        """Send event via SSE (buffer for generator)"""
        
        if subscription_id in self.event_buffers:
            # Add to buffer (generator will pick it up)
            buffer = self.event_buffers[subscription_id]
            
            # Prevent buffer overflow
            if len(buffer) >= self.config['max_buffer_size']:
                buffer.pop(0)  # Remove oldest event
            
            buffer.append(event)
    
    async def _handle_websocket_message(
        self,
        websocket: WebSocket,
        subscription: StreamSubscription,
        message: Dict[str, Any]
    ):
        """Handle incoming WebSocket message"""
        
        message_type = message.get('type')
        
        if message_type == 'heartbeat':
            subscription.last_heartbeat = datetime.now(timezone.utc)
            await websocket.send_json({
                "type": "heartbeat_ack",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        elif message_type == 'update_filters':
            # Update subscription filters
            new_filters = self._parse_filters(message.get('filters', []))
            subscription.filters = new_filters
            
            await websocket.send_json({
                "type": "filters_updated",
                "filters_count": len(new_filters),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        elif message_type == 'get_stats':
            # Send subscription statistics
            await websocket.send_json({
                "type": "subscription_stats",
                "stats": {
                    "events_sent": subscription.events_sent,
                    "events_filtered": subscription.events_filtered,
                    "connection_errors": subscription.connection_errors,
                    "connected_duration_seconds": (
                        datetime.now(timezone.utc) - subscription.connected_at
                    ).total_seconds()
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_task())
        self.background_tasks.append(cleanup_task)
        
        # Statistics task
        stats_task = asyncio.create_task(self._statistics_task())
        self.background_tasks.append(stats_task)
    
    async def _cleanup_task(self):
        """Background task for cleaning up stale connections"""
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config['cleanup_interval_seconds'])
                
                current_time = datetime.now(timezone.utc)
                stale_subscriptions = []
                
                for subscription_id, subscription in self.subscriptions.items():
                    # Check for stale connections
                    time_since_heartbeat = (current_time - subscription.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.config['connection_timeout_seconds']:
                        stale_subscriptions.append(subscription_id)
                
                # Remove stale subscriptions
                for subscription_id in stale_subscriptions:
                    self.logger.info(f"🧹 Cleaning up stale subscription: {subscription_id}")
                    
                    if subscription_id in self.websocket_connections:
                        try:
                            await self.websocket_connections[subscription_id].close()
                        except:
                            pass
                        del self.websocket_connections[subscription_id]
                    
                    if subscription_id in self.subscriptions:
                        del self.subscriptions[subscription_id]
                    
                    if subscription_id in self.event_buffers:
                        del self.event_buffers[subscription_id]
                    
                    self.streaming_stats['active_connections'] -= 1
                
            except Exception as e:
                self.logger.error(f"❌ Cleanup task error: {e}")
    
    async def _statistics_task(self):
        """Background task for updating statistics"""
        
        last_event_count = 0
        last_update = datetime.now(timezone.utc)
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                current_time = datetime.now(timezone.utc)
                current_event_count = self.streaming_stats['total_events_streamed']
                
                # Calculate events per second
                time_diff = (current_time - last_update).total_seconds()
                if time_diff > 0:
                    events_diff = current_event_count - last_event_count
                    self.streaming_stats['events_per_second'] = events_diff / time_diff
                
                last_event_count = current_event_count
                last_update = current_time
                
                # Update active connections count
                self.streaming_stats['active_connections'] = len(self.subscriptions)
                
            except Exception as e:
                self.logger.error(f"❌ Statistics task error: {e}")
    
    async def _create_streaming_tables(self):
        """Create streaming-related database tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Streaming connections log
        CREATE TABLE IF NOT EXISTS audit_streaming_connections (
            connection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            subscription_id VARCHAR(200) NOT NULL,
            client_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Connection details
            stream_type VARCHAR(20) NOT NULL,
            connection_source VARCHAR(100),
            user_agent TEXT,
            
            -- Statistics
            events_sent INTEGER NOT NULL DEFAULT 0,
            events_filtered INTEGER NOT NULL DEFAULT 0,
            connection_errors INTEGER NOT NULL DEFAULT 0,
            
            -- Timestamps
            connected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            disconnected_at TIMESTAMPTZ,
            last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
            
            -- Configuration
            filters_config JSONB DEFAULT '[]',
            
            -- Constraints
            CONSTRAINT chk_stream_type CHECK (stream_type IN ('websocket', 'sse'))
        );
        
        -- Streaming statistics
        CREATE TABLE IF NOT EXISTS audit_streaming_stats (
            stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            
            -- Time period
            stat_date DATE NOT NULL DEFAULT CURRENT_DATE,
            stat_hour INTEGER NOT NULL DEFAULT EXTRACT(HOUR FROM NOW()),
            
            -- Metrics
            total_connections INTEGER NOT NULL DEFAULT 0,
            peak_concurrent_connections INTEGER NOT NULL DEFAULT 0,
            total_events_streamed INTEGER NOT NULL DEFAULT 0,
            average_events_per_second FLOAT NOT NULL DEFAULT 0,
            
            -- Performance
            average_latency_ms FLOAT,
            connection_errors INTEGER NOT NULL DEFAULT 0,
            
            -- Timestamps
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            CONSTRAINT uq_streaming_stats_date_hour UNIQUE (stat_date, stat_hour)
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_streaming_connections_client ON audit_streaming_connections(client_id, connected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_streaming_connections_tenant ON audit_streaming_connections(tenant_id, connected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_streaming_connections_active ON audit_streaming_connections(connected_at DESC) WHERE disconnected_at IS NULL;
        
        CREATE INDEX IF NOT EXISTS idx_streaming_stats_date ON audit_streaming_stats(stat_date DESC, stat_hour DESC);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Audit streaming tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create streaming tables: {e}")
            raise
    
    def get_streaming_statistics(self) -> Dict[str, Any]:
        """Get streaming API statistics"""
        
        return {
            'total_connections': self.streaming_stats['total_connections'],
            'active_connections': self.streaming_stats['active_connections'],
            'websocket_connections': len(self.websocket_connections),
            'sse_connections': len(self.sse_connections),
            'total_events_streamed': self.streaming_stats['total_events_streamed'],
            'events_per_second': round(self.streaming_stats['events_per_second'], 2),
            'average_latency_ms': round(self.streaming_stats['average_latency_ms'], 2),
            'connection_errors': self.streaming_stats['connection_errors'],
            'max_concurrent_connections': self.config['max_concurrent_connections'],
            'heartbeat_interval_seconds': self.config['heartbeat_interval_seconds'],
            'is_running': self.is_running
        }
    
    async def get_active_subscriptions(self) -> List[Dict[str, Any]]:
        """Get list of active subscriptions"""
        
        subscriptions = []
        
        for subscription_id, subscription in self.subscriptions.items():
            subscriptions.append({
                'subscription_id': subscription_id,
                'client_id': subscription.client_id,
                'tenant_id': subscription.tenant_id,
                'stream_type': subscription.stream_type.value,
                'connected_at': subscription.connected_at.isoformat(),
                'events_sent': subscription.events_sent,
                'events_filtered': subscription.events_filtered,
                'connection_errors': subscription.connection_errors,
                'filters_count': len(subscription.filters),
                'buffer_size': len(self.event_buffers.get(subscription_id, []))
            })
        
        return subscriptions


# Global streaming API instance
audit_logs_streaming_api = AuditLogsStreamingAPI()
