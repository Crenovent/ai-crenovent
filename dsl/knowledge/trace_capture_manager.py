"""
Trace Capture Manager
====================
Implements Tasks 8.1.3-8.1.6: Configure trace capture across all architectural planes

Features:
- Control Plane: Route decisions, policy evaluations, orchestration events
- Execution Plane: Workflow executions, agent calls, operator invocations  
- Data Plane: Data access, transformations, storage operations
- Governance Plane: Policy enforcement, override ledger, evidence generation

Architecture:
- Unified trace schema across all planes
- Async trace collection with minimal performance impact
- Tenant-aware trace partitioning
- Real-time streaming to Knowledge Graph
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import contextlib

logger = logging.getLogger(__name__)

class TracePlane(Enum):
    CONTROL = "control"
    EXECUTION = "execution" 
    DATA = "data"
    GOVERNANCE = "governance"

class TraceEventType(Enum):
    # Control Plane Events
    ROUTE_DECISION = "route_decision"
    POLICY_EVALUATION = "policy_evaluation"
    ORCHESTRATION_START = "orchestration_start"
    ORCHESTRATION_END = "orchestration_end"
    
    # Execution Plane Events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    AGENT_CALL = "agent_call"
    OPERATOR_INVOKE = "operator_invoke"
    
    # Data Plane Events
    DATA_ACCESS = "data_access"
    DATA_TRANSFORM = "data_transform"
    STORAGE_OPERATION = "storage_operation"
    QUERY_EXECUTION = "query_execution"
    
    # Governance Plane Events
    POLICY_ENFORCEMENT = "policy_enforcement"
    OVERRIDE_RECORDED = "override_recorded"
    EVIDENCE_GENERATED = "evidence_generated"
    COMPLIANCE_CHECK = "compliance_check"

@dataclass
class TraceEvent:
    """Unified trace event structure across all planes"""
    
    # Core identifiers
    trace_id: str
    event_id: str
    tenant_id: int
    user_id: Optional[int]
    
    # Event metadata
    plane: TracePlane
    event_type: TraceEventType
    timestamp: datetime
    duration_ms: Optional[int] = None
    
    # Context
    workflow_id: Optional[str] = None
    agent_name: Optional[str] = None
    operator_type: Optional[str] = None
    
    # Event data
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    
    # Governance
    policy_pack_id: Optional[str] = None
    override_id: Optional[str] = None
    evidence_pack_id: Optional[str] = None
    
    # Performance
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    
    # Lineage
    parent_trace_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['plane'] = self.plane.value
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class TraceCaptureManager:
    """
    Manages trace capture across all architectural planes
    
    Features:
    - Task 8.1.3: Control Plane trace capture
    - Task 8.1.4: Execution Plane trace capture  
    - Task 8.1.5: Data Plane trace capture
    - Task 8.1.6: Governance Plane trace capture
    """
    
    def __init__(self, pool_manager, kg_store=None):
        self.pool_manager = pool_manager
        self.kg_store = kg_store
        self.logger = logging.getLogger(__name__)
        
        # Trace buffer for batch processing
        self.trace_buffer: List[TraceEvent] = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        
        # Performance tracking
        self.traces_captured = 0
        self.traces_processed = 0
        
        # Background task for flushing
        self._flush_task = None
        
    async def initialize(self) -> bool:
        """Initialize trace capture manager"""
        try:
            self.logger.info("📊 Initializing Trace Capture Manager...")
            
            # Start background flush task
            self._flush_task = asyncio.create_task(self._periodic_flush())
            
            self.logger.info("✅ Trace Capture Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trace Capture Manager initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown trace capture manager"""
        try:
            # Cancel background task
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            # Flush remaining traces
            await self._flush_traces()
            
            self.logger.info("✅ Trace Capture Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Trace Capture Manager shutdown failed: {e}")
    
    # ============================================================================
    # CONTROL PLANE TRACE CAPTURE (Task 8.1.3)
    # ============================================================================
    
    async def capture_route_decision(self, tenant_id: int, user_id: int, 
                                   input_request: Dict[str, Any], 
                                   route_decision: Dict[str, Any],
                                   duration_ms: int) -> str:
        """Capture routing orchestrator decisions"""
        
        trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.CONTROL,
            event_type=TraceEventType.ROUTE_DECISION,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            input_data=input_request,
            output_data=route_decision
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    async def capture_policy_evaluation(self, tenant_id: int, user_id: int,
                                      policy_pack_id: str, evaluation_result: Dict[str, Any],
                                      duration_ms: int, trace_id: str = None) -> str:
        """Capture policy engine evaluations"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.CONTROL,
            event_type=TraceEventType.POLICY_EVALUATION,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            policy_pack_id=policy_pack_id,
            output_data=evaluation_result
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    @contextlib.asynccontextmanager
    async def capture_orchestration(self, tenant_id: int, user_id: int, 
                                  workflow_id: str, trace_id: str = None):
        """Context manager for orchestration tracing"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
            
        start_time = datetime.now(timezone.utc)
        
        # Start event
        start_event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.CONTROL,
            event_type=TraceEventType.ORCHESTRATION_START,
            timestamp=start_time,
            workflow_id=workflow_id
        )
        
        await self._buffer_trace(start_event)
        
        try:
            yield trace_id
        finally:
            # End event
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            end_event = TraceEvent(
                trace_id=trace_id,
                event_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                plane=TracePlane.CONTROL,
                event_type=TraceEventType.ORCHESTRATION_END,
                timestamp=end_time,
                duration_ms=duration_ms,
                workflow_id=workflow_id
            )
            
            await self._buffer_trace(end_event)
    
    # ============================================================================
    # EXECUTION PLANE TRACE CAPTURE (Task 8.1.4)
    # ============================================================================
    
    @contextlib.asynccontextmanager
    async def capture_workflow_execution(self, tenant_id: int, user_id: int,
                                       workflow_id: str, agent_name: str,
                                       input_data: Dict[str, Any],
                                       trace_id: str = None):
        """Context manager for workflow execution tracing"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
            
        start_time = datetime.now(timezone.utc)
        
        # Start event
        start_event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.EXECUTION,
            event_type=TraceEventType.WORKFLOW_START,
            timestamp=start_time,
            workflow_id=workflow_id,
            agent_name=agent_name,
            input_data=input_data
        )
        
        await self._buffer_trace(start_event)
        
        error_data = None
        output_data = None
        
        try:
            yield trace_id
        except Exception as e:
            error_data = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'error_traceback': str(e.__traceback__)
            }
            raise
        finally:
            # End event
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            end_event = TraceEvent(
                trace_id=trace_id,
                event_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                plane=TracePlane.EXECUTION,
                event_type=TraceEventType.WORKFLOW_END,
                timestamp=end_time,
                duration_ms=duration_ms,
                workflow_id=workflow_id,
                agent_name=agent_name,
                output_data=output_data,
                error_data=error_data
            )
            
            await self._buffer_trace(end_event)
    
    async def capture_agent_call(self, tenant_id: int, user_id: int,
                               agent_name: str, operator_type: str,
                               input_data: Dict[str, Any], output_data: Dict[str, Any],
                               duration_ms: int, trace_id: str = None) -> str:
        """Capture individual agent/operator calls"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.EXECUTION,
            event_type=TraceEventType.AGENT_CALL,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            agent_name=agent_name,
            operator_type=operator_type,
            input_data=input_data,
            output_data=output_data
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    # ============================================================================
    # DATA PLANE TRACE CAPTURE (Task 8.1.5)
    # ============================================================================
    
    async def capture_data_access(self, tenant_id: int, user_id: int,
                                query_type: str, data_source: str,
                                query_sql: str, row_count: int,
                                duration_ms: int, trace_id: str = None) -> str:
        """Capture data access operations"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.DATA,
            event_type=TraceEventType.DATA_ACCESS,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            input_data={
                'query_type': query_type,
                'data_source': data_source,
                'query_sql': query_sql[:500],  # Truncate long queries
                'row_count': row_count
            }
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    async def capture_data_transform(self, tenant_id: int, user_id: int,
                                   transform_type: str, input_schema: Dict[str, Any],
                                   output_schema: Dict[str, Any], record_count: int,
                                   duration_ms: int, trace_id: str = None) -> str:
        """Capture data transformation operations"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.DATA,
            event_type=TraceEventType.DATA_TRANSFORM,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            input_data={
                'transform_type': transform_type,
                'input_schema': input_schema,
                'output_schema': output_schema,
                'record_count': record_count
            }
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    # ============================================================================
    # GOVERNANCE PLANE TRACE CAPTURE (Task 8.1.6)
    # ============================================================================
    
    async def capture_policy_enforcement(self, tenant_id: int, user_id: int,
                                       policy_pack_id: str, enforcement_result: Dict[str, Any],
                                       duration_ms: int, trace_id: str = None) -> str:
        """Capture policy enforcement events"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.GOVERNANCE,
            event_type=TraceEventType.POLICY_ENFORCEMENT,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            policy_pack_id=policy_pack_id,
            output_data=enforcement_result
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    async def capture_override_recorded(self, tenant_id: int, user_id: int,
                                      override_id: str, override_data: Dict[str, Any],
                                      trace_id: str = None) -> str:
        """Capture override ledger entries"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.GOVERNANCE,
            event_type=TraceEventType.OVERRIDE_RECORDED,
            timestamp=datetime.now(timezone.utc),
            override_id=override_id,
            output_data=override_data
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    async def capture_evidence_generated(self, tenant_id: int, user_id: int,
                                       evidence_pack_id: str, evidence_data: Dict[str, Any],
                                       trace_id: str = None) -> str:
        """Capture evidence pack generation"""
        
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            plane=TracePlane.GOVERNANCE,
            event_type=TraceEventType.EVIDENCE_GENERATED,
            timestamp=datetime.now(timezone.utc),
            evidence_pack_id=evidence_pack_id,
            output_data=evidence_data
        )
        
        await self._buffer_trace(event)
        return trace_id
    
    # ============================================================================
    # TRACE PROCESSING & STORAGE
    # ============================================================================
    
    async def _buffer_trace(self, event: TraceEvent):
        """Add trace to buffer for batch processing"""
        self.trace_buffer.append(event)
        self.traces_captured += 1
        
        # Flush if buffer is full
        if len(self.trace_buffer) >= self.buffer_size:
            await self._flush_traces()
    
    async def _flush_traces(self):
        """Flush trace buffer to storage"""
        if not self.trace_buffer:
            return
        
        try:
            # Get traces to flush
            traces_to_flush = self.trace_buffer.copy()
            self.trace_buffer.clear()
            
            # Store in database
            await self._store_traces_in_db(traces_to_flush)
            
            # Send to Knowledge Graph if available
            if self.kg_store:
                await self._send_traces_to_kg(traces_to_flush)
            
            self.traces_processed += len(traces_to_flush)
            
            self.logger.debug(f"📊 Flushed {len(traces_to_flush)} traces to storage")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to flush traces: {e}")
    
    async def _store_traces_in_db(self, traces: List[TraceEvent]):
        """Store traces in database"""
        if not self.pool_manager or not self.pool_manager.postgres_pool:
            return
        
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Insert traces
                for trace in traces:
                    await conn.execute("""
                        INSERT INTO kg_execution_traces (
                            trace_id, event_id, tenant_id, user_id,
                            plane, event_type, timestamp, duration_ms,
                            workflow_id, agent_name, operator_type,
                            input_data, output_data, error_data,
                            policy_pack_id, override_id, evidence_pack_id,
                            parent_trace_id, correlation_id,
                            created_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18, $19, NOW()
                        )
                    """,
                    trace.trace_id, trace.event_id, trace.tenant_id, trace.user_id,
                    trace.plane.value, trace.event_type.value, trace.timestamp, trace.duration_ms,
                    trace.workflow_id, trace.agent_name, trace.operator_type,
                    json.dumps(trace.input_data) if trace.input_data else None,
                    json.dumps(trace.output_data) if trace.output_data else None,
                    json.dumps(trace.error_data) if trace.error_data else None,
                    trace.policy_pack_id, trace.override_id, trace.evidence_pack_id,
                    trace.parent_trace_id, trace.correlation_id
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to store traces in database: {e}")
    
    async def _send_traces_to_kg(self, traces: List[TraceEvent]):
        """Send traces to Knowledge Graph for real-time processing"""
        try:
            for trace in traces:
                await self.kg_store.ingest_trace_event(trace.to_dict())
                
        except Exception as e:
            self.logger.error(f"❌ Failed to send traces to KG: {e}")
    
    async def _periodic_flush(self):
        """Periodic background flush of trace buffer"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_traces()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in periodic flush: {e}")

# Global instance
_trace_capture_manager = None

async def get_trace_capture_manager(pool_manager, kg_store=None):
    """Get global trace capture manager instance"""
    global _trace_capture_manager
    
    if _trace_capture_manager is None:
        _trace_capture_manager = TraceCaptureManager(pool_manager, kg_store)
        await _trace_capture_manager.initialize()
    
    return _trace_capture_manager
