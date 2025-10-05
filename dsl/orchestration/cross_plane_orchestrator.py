"""
Cross-Plane Orchestration API
Task 9.3.3: Build orchestration API for cross-plane calls

Provides standardized control messages and orchestration across:
- Control Plane (routing, policy, orchestration)
- Execution Plane (RBA, RBIA, AALA engines)
- Data Plane (registry, traces, knowledge)
- Governance Plane (policy, evidence, audit)
- UX Plane (dashboards, interfaces)
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib

logger = logging.getLogger(__name__)

class PlaneType(Enum):
    """Architectural planes in the system"""
    CONTROL = "control"
    EXECUTION = "execution"
    DATA = "data"
    GOVERNANCE = "governance"
    UX = "ux"

class MessageType(Enum):
    """Cross-plane message types"""
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    RESPONSE = "response"
    NOTIFICATION = "notification"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class ExecutionStatus(Enum):
    """Cross-plane execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class GovernanceMetadata:
    """Mandatory governance metadata for all cross-plane calls"""
    tenant_id: int
    region_id: str
    policy_pack_id: Optional[str] = None
    evidence_pack_id: Optional[str] = None
    compliance_frameworks: List[str] = None
    sla_tier: str = "T2"  # T0, T1, T2
    
    def __post_init__(self):
        if self.compliance_frameworks is None:
            self.compliance_frameworks = []

@dataclass
class CrossPlaneMessage:
    """Standardized cross-plane message format"""
    message_id: str
    source_plane: PlaneType
    target_plane: PlaneType
    message_type: MessageType
    priority: MessagePriority
    
    # Governance metadata (mandatory)
    governance: GovernanceMetadata
    
    # Message content
    operation: str
    payload: Dict[str, Any]
    
    # Execution context
    correlation_id: str
    parent_message_id: Optional[str] = None
    execution_timeout_ms: int = 30000
    retry_count: int = 0
    max_retries: int = 3
    
    # Timestamps
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    # Response handling
    response_required: bool = True
    callback_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['source_plane'] = self.source_plane.value
        result['target_plane'] = self.target_plane.value
        result['message_type'] = self.message_type.value
        result['priority'] = self.priority.value
        result['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        return result

@dataclass
class CrossPlaneResponse:
    """Standardized cross-plane response format"""
    response_id: str
    original_message_id: str
    source_plane: PlaneType
    target_plane: PlaneType
    
    # Response data
    status: ExecutionStatus
    result: Dict[str, Any]
    error: Optional[str] = None
    
    # Governance
    governance: GovernanceMetadata
    evidence_pack_id: Optional[str] = None
    
    # Timing
    execution_time_ms: int
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['source_plane'] = self.source_plane.value
        result['target_plane'] = self.target_plane.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        return result

class CrossPlaneOrchestrator:
    """
    Cross-Plane Orchestration Engine
    Task 9.3.3: Standardized control messages and orchestration
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Message routing and handlers
        self.plane_handlers = {}
        self.message_queue = {}
        self.response_callbacks = {}
        
        # Governance and policy integration
        self.policy_engine = None
        self.evidence_generator = None
        
        # Performance monitoring
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "responses_sent": 0,
            "responses_received": 0,
            "errors": 0,
            "timeouts": 0
        }
        
        # SLA configurations per tier
        self.sla_configs = {
            "T0": {"timeout_ms": 5000, "max_retries": 5, "priority": MessagePriority.CRITICAL},
            "T1": {"timeout_ms": 15000, "max_retries": 3, "priority": MessagePriority.HIGH},
            "T2": {"timeout_ms": 30000, "max_retries": 2, "priority": MessagePriority.NORMAL}
        }
        
    async def initialize(self) -> bool:
        """Initialize the cross-plane orchestrator"""
        try:
            self.logger.info("🚀 Initializing Cross-Plane Orchestrator...")
            
            # Initialize governance components
            await self._initialize_governance_components()
            
            # Register default plane handlers
            await self._register_default_handlers()
            
            # Create message routing tables
            await self._create_message_tables()
            
            self.logger.info("✅ Cross-Plane Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Cross-Plane Orchestrator: {e}")
            return False
    
    async def send_message(self, 
                         target_plane: PlaneType,
                         operation: str,
                         payload: Dict[str, Any],
                         governance: GovernanceMetadata,
                         source_plane: PlaneType = PlaneType.CONTROL,
                         message_type: MessageType = MessageType.COMMAND,
                         priority: MessagePriority = None,
                         timeout_ms: int = None,
                         correlation_id: str = None) -> CrossPlaneResponse:
        """
        Send a cross-plane message and wait for response
        """
        try:
            # Generate message ID and correlation ID
            message_id = str(uuid.uuid4())
            if correlation_id is None:
                correlation_id = str(uuid.uuid4())
            
            # Apply SLA-based configurations
            sla_config = self.sla_configs.get(governance.sla_tier, self.sla_configs["T2"])
            if priority is None:
                priority = sla_config["priority"]
            if timeout_ms is None:
                timeout_ms = sla_config["timeout_ms"]
            
            # Create message
            message = CrossPlaneMessage(
                message_id=message_id,
                source_plane=source_plane,
                target_plane=target_plane,
                message_type=message_type,
                priority=priority,
                governance=governance,
                operation=operation,
                payload=payload,
                correlation_id=correlation_id,
                execution_timeout_ms=timeout_ms,
                max_retries=sla_config["max_retries"],
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(milliseconds=timeout_ms * 2)
            )
            
            # Validate governance requirements
            validation_result = await self._validate_governance(message)
            if not validation_result["valid"]:
                raise ValueError(f"Governance validation failed: {validation_result['errors']}")
            
            # Route message to target plane
            response = await self._route_message(message)
            
            # Generate evidence pack
            await self._generate_message_evidence(message, response)
            
            # Update metrics
            self.metrics["messages_sent"] += 1
            if response.status == ExecutionStatus.COMPLETED:
                self.metrics["responses_received"] += 1
            else:
                self.metrics["errors"] += 1
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send cross-plane message: {e}")
            self.metrics["errors"] += 1
            
            # Return error response
            return CrossPlaneResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message_id if 'message_id' in locals() else "unknown",
                source_plane=target_plane,
                target_plane=source_plane,
                status=ExecutionStatus.FAILED,
                result={},
                error=str(e),
                governance=governance,
                execution_time_ms=0,
                created_at=datetime.now()
            )
    
    async def send_notification(self,
                              target_plane: PlaneType,
                              event_type: str,
                              event_data: Dict[str, Any],
                              governance: GovernanceMetadata,
                              source_plane: PlaneType = PlaneType.CONTROL) -> bool:
        """
        Send a fire-and-forget notification to a plane
        """
        try:
            message = CrossPlaneMessage(
                message_id=str(uuid.uuid4()),
                source_plane=source_plane,
                target_plane=target_plane,
                message_type=MessageType.NOTIFICATION,
                priority=MessagePriority.NORMAL,
                governance=governance,
                operation=event_type,
                payload=event_data,
                correlation_id=str(uuid.uuid4()),
                created_at=datetime.now(),
                response_required=False
            )
            
            # Route notification (no response expected)
            await self._route_notification(message)
            
            self.metrics["messages_sent"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send notification: {e}")
            self.metrics["errors"] += 1
            return False
    
    async def register_plane_handler(self, 
                                   plane: PlaneType, 
                                   handler_func: callable) -> bool:
        """
        Register a handler function for a specific plane
        """
        try:
            self.plane_handlers[plane] = handler_func
            self.logger.info(f"✅ Registered handler for {plane.value} plane")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register plane handler: {e}")
            return False
    
    async def query_plane(self,
                        target_plane: PlaneType,
                        query_type: str,
                        query_params: Dict[str, Any],
                        governance: GovernanceMetadata,
                        source_plane: PlaneType = PlaneType.CONTROL) -> Dict[str, Any]:
        """
        Query a plane for information
        """
        try:
            response = await self.send_message(
                target_plane=target_plane,
                operation=f"query_{query_type}",
                payload=query_params,
                governance=governance,
                source_plane=source_plane,
                message_type=MessageType.QUERY
            )
            
            if response.status == ExecutionStatus.COMPLETED:
                return response.result
            else:
                raise Exception(f"Query failed: {response.error}")
                
        except Exception as e:
            self.logger.error(f"❌ Plane query failed: {e}")
            raise
    
    async def execute_cross_plane_workflow(self,
                                         workflow_definition: Dict[str, Any],
                                         governance: GovernanceMetadata) -> Dict[str, Any]:
        """
        Execute a multi-plane workflow with orchestration
        """
        try:
            workflow_id = str(uuid.uuid4())
            correlation_id = str(uuid.uuid4())
            
            self.logger.info(f"🚀 Starting cross-plane workflow: {workflow_id}")
            
            # Parse workflow steps
            steps = workflow_definition.get("steps", [])
            results = {}
            
            for step_idx, step in enumerate(steps):
                step_id = f"{workflow_id}_step_{step_idx}"
                
                # Extract step configuration
                target_plane = PlaneType(step["target_plane"])
                operation = step["operation"]
                step_payload = step.get("payload", {})
                
                # Add previous step results to payload if needed
                if step.get("use_previous_results", False) and results:
                    step_payload["previous_results"] = results
                
                # Execute step
                self.logger.info(f"📋 Executing step {step_idx + 1}/{len(steps)}: {operation}")
                
                response = await self.send_message(
                    target_plane=target_plane,
                    operation=operation,
                    payload=step_payload,
                    governance=governance,
                    correlation_id=correlation_id
                )
                
                if response.status != ExecutionStatus.COMPLETED:
                    raise Exception(f"Step {step_idx + 1} failed: {response.error}")
                
                # Store step results
                results[f"step_{step_idx + 1}"] = response.result
                
                # Check for conditional execution
                if step.get("condition") and not self._evaluate_condition(step["condition"], results):
                    self.logger.info(f"⏭️ Skipping remaining steps due to condition")
                    break
            
            self.logger.info(f"✅ Cross-plane workflow completed: {workflow_id}")
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": results,
                "steps_executed": len(results),
                "execution_time_ms": (datetime.now() - datetime.now()).total_seconds() * 1000
            }
            
        except Exception as e:
            self.logger.error(f"❌ Cross-plane workflow failed: {e}")
            return {
                "workflow_id": workflow_id if 'workflow_id' in locals() else "unknown",
                "status": "failed",
                "error": str(e),
                "results": results if 'results' in locals() else {}
            }
    
    async def get_plane_health(self, plane: PlaneType) -> Dict[str, Any]:
        """
        Get health status of a specific plane
        """
        try:
            governance = GovernanceMetadata(
                tenant_id=0,  # System health check
                region_id="system"
            )
            
            response = await self.send_message(
                target_plane=plane,
                operation="health_check",
                payload={},
                governance=governance,
                message_type=MessageType.QUERY,
                timeout_ms=5000
            )
            
            return {
                "plane": plane.value,
                "healthy": response.status == ExecutionStatus.COMPLETED,
                "response_time_ms": response.execution_time_ms,
                "details": response.result
            }
            
        except Exception as e:
            return {
                "plane": plane.value,
                "healthy": False,
                "error": str(e),
                "response_time_ms": -1
            }
    
    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """
        Get orchestrator performance metrics
        """
        return {
            "metrics": self.metrics.copy(),
            "registered_planes": list(self.plane_handlers.keys()),
            "active_correlations": len(self.response_callbacks),
            "timestamp": datetime.now().isoformat()
        }
    
    # Private helper methods
    
    async def _initialize_governance_components(self):
        """Initialize governance and policy components"""
        try:
            # Import governance components
            from dsl.governance.policy_engine import PolicyEngine
            from api.evidence_pack_api import EvidencePackService
            
            self.policy_engine = PolicyEngine(self.pool_manager)
            self.evidence_generator = EvidencePackService(self.pool_manager)
            
        except ImportError as e:
            self.logger.warning(f"Some governance components not available: {e}")
    
    async def _register_default_handlers(self):
        """Register default handlers for each plane"""
        # Default handlers would be registered here
        # For now, we'll use placeholder handlers
        
        async def default_handler(message: CrossPlaneMessage) -> CrossPlaneResponse:
            return CrossPlaneResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                source_plane=message.target_plane,
                target_plane=message.source_plane,
                status=ExecutionStatus.COMPLETED,
                result={"message": "Default handler response"},
                governance=message.governance,
                execution_time_ms=100,
                created_at=datetime.now()
            )
        
        # Register default handlers for all planes
        for plane in PlaneType:
            if plane not in self.plane_handlers:
                self.plane_handlers[plane] = default_handler
    
    async def _create_message_tables(self):
        """Create database tables for message tracking"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        async with self.pool_manager.postgres_pool.acquire() as conn:
            # Cross-plane messages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_plane_messages (
                    message_id UUID PRIMARY KEY,
                    source_plane VARCHAR(50) NOT NULL,
                    target_plane VARCHAR(50) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    operation VARCHAR(255) NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    correlation_id UUID NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    payload JSONB,
                    response JSONB,
                    execution_time_ms INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    
                    INDEX(tenant_id),
                    INDEX(correlation_id),
                    INDEX(created_at)
                )
            """)
    
    async def _validate_governance(self, message: CrossPlaneMessage) -> Dict[str, Any]:
        """Validate governance requirements for cross-plane message"""
        errors = []
        
        # Check mandatory fields
        if not message.governance.tenant_id:
            errors.append("tenant_id is required")
        
        if not message.governance.region_id:
            errors.append("region_id is required")
        
        # Validate SLA tier
        if message.governance.sla_tier not in ["T0", "T1", "T2"]:
            errors.append("Invalid SLA tier")
        
        # Policy validation (if policy engine available)
        if self.policy_engine:
            try:
                policy_result = await self.policy_engine.enforce_policy(
                    message.governance.tenant_id,
                    message.payload,
                    {"operation": message.operation, "target_plane": message.target_plane.value}
                )
                if not policy_result[0]:  # policy_result is (allowed, violations)
                    errors.extend(policy_result[1])
            except Exception as e:
                self.logger.warning(f"Policy validation failed: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _route_message(self, message: CrossPlaneMessage) -> CrossPlaneResponse:
        """Route message to target plane handler"""
        try:
            start_time = datetime.now()
            
            # Get handler for target plane
            handler = self.plane_handlers.get(message.target_plane)
            if not handler:
                raise Exception(f"No handler registered for plane: {message.target_plane.value}")
            
            # Execute handler with timeout
            try:
                response = await asyncio.wait_for(
                    handler(message),
                    timeout=message.execution_timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                self.metrics["timeouts"] += 1
                raise Exception(f"Message execution timed out after {message.execution_timeout_ms}ms")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            response.execution_time_ms = int(execution_time)
            
            # Store message and response
            await self._store_message_trace(message, response)
            
            return response
            
        except Exception as e:
            # Create error response
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            error_response = CrossPlaneResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                source_plane=message.target_plane,
                target_plane=message.source_plane,
                status=ExecutionStatus.FAILED,
                result={},
                error=str(e),
                governance=message.governance,
                execution_time_ms=int(execution_time),
                created_at=datetime.now()
            )
            
            await self._store_message_trace(message, error_response)
            return error_response
    
    async def _route_notification(self, message: CrossPlaneMessage):
        """Route notification message (fire-and-forget)"""
        try:
            handler = self.plane_handlers.get(message.target_plane)
            if handler:
                # Execute handler without waiting for response
                asyncio.create_task(handler(message))
            
            # Store notification
            await self._store_notification_trace(message)
            
        except Exception as e:
            self.logger.error(f"Failed to route notification: {e}")
    
    async def _store_message_trace(self, message: CrossPlaneMessage, response: CrossPlaneResponse):
        """Store message and response for audit trail"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO cross_plane_messages 
                    (message_id, source_plane, target_plane, message_type, operation,
                     tenant_id, correlation_id, status, payload, response, execution_time_ms, completed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, 
                message.message_id,
                message.source_plane.value,
                message.target_plane.value,
                message.message_type.value,
                message.operation,
                message.governance.tenant_id,
                message.correlation_id,
                response.status.value,
                json.dumps(message.payload),
                json.dumps(response.result),
                response.execution_time_ms,
                datetime.now())
                
        except Exception as e:
            self.logger.error(f"Failed to store message trace: {e}")
    
    async def _store_notification_trace(self, message: CrossPlaneMessage):
        """Store notification for audit trail"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO cross_plane_messages 
                    (message_id, source_plane, target_plane, message_type, operation,
                     tenant_id, correlation_id, status, payload)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                message.message_id,
                message.source_plane.value,
                message.target_plane.value,
                message.message_type.value,
                message.operation,
                message.governance.tenant_id,
                message.correlation_id,
                "notification_sent",
                json.dumps(message.payload))
                
        except Exception as e:
            self.logger.error(f"Failed to store notification trace: {e}")
    
    async def _generate_message_evidence(self, message: CrossPlaneMessage, response: CrossPlaneResponse):
        """Generate evidence pack for cross-plane message"""
        if not self.evidence_generator:
            return
        
        try:
            evidence_data = {
                "message_id": message.message_id,
                "correlation_id": message.correlation_id,
                "source_plane": message.source_plane.value,
                "target_plane": message.target_plane.value,
                "operation": message.operation,
                "status": response.status.value,
                "execution_time_ms": response.execution_time_ms,
                "sla_tier": message.governance.sla_tier,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.evidence_generator.generate_evidence_pack(
                pack_type="cross_plane_message",
                workflow_id=message.correlation_id,
                tenant_id=message.governance.tenant_id,
                evidence_data=evidence_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate message evidence: {e}")
    
    def _evaluate_condition(self, condition: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Evaluate workflow step condition"""
        # Simple condition evaluation - can be extended
        condition_type = condition.get("type", "always")
        
        if condition_type == "always":
            return True
        elif condition_type == "never":
            return False
        elif condition_type == "result_equals":
            step_key = condition.get("step")
            expected_value = condition.get("value")
            return results.get(step_key, {}).get("status") == expected_value
        
        return True
