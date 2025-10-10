"""
Task 7.3-T39: Audit Log every registry action
Non-repudiation with append-only log linked to Evidence
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import asyncpg
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding


class AuditActionType(Enum):
    """Types of auditable actions"""
    # Workflow actions
    WORKFLOW_CREATE = "workflow_create"
    WORKFLOW_UPDATE = "workflow_update"
    WORKFLOW_DELETE = "workflow_delete"
    WORKFLOW_PUBLISH = "workflow_publish"
    WORKFLOW_DEPRECATE = "workflow_deprecate"
    WORKFLOW_RETIRE = "workflow_retire"
    
    # Version actions
    VERSION_CREATE = "version_create"
    VERSION_UPDATE = "version_update"
    VERSION_DELETE = "version_delete"
    VERSION_APPROVE = "version_approve"
    VERSION_REJECT = "version_reject"
    
    # Artifact actions
    ARTIFACT_UPLOAD = "artifact_upload"
    ARTIFACT_DOWNLOAD = "artifact_download"
    ARTIFACT_DELETE = "artifact_delete"
    ARTIFACT_VERIFY = "artifact_verify"
    
    # Registry actions
    REGISTRY_SEARCH = "registry_search"
    REGISTRY_BROWSE = "registry_browse"
    REGISTRY_EXPORT = "registry_export"
    REGISTRY_IMPORT = "registry_import"
    
    # Security actions
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    
    # Administrative actions
    POLICY_CREATE = "policy_create"
    POLICY_UPDATE = "policy_update"
    POLICY_DELETE = "policy_delete"
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    
    # Compliance actions
    COMPLIANCE_CHECK = "compliance_check"
    LEGAL_HOLD_APPLY = "legal_hold_apply"
    LEGAL_HOLD_RELEASE = "legal_hold_release"
    DATA_ERASURE = "data_erasure"


class AuditActorType(Enum):
    """Types of actors performing actions"""
    USER = "user"
    SYSTEM = "system"
    API_CLIENT = "api_client"
    SERVICE = "service"
    SCHEDULER = "scheduler"


class AuditStatus(Enum):
    """Status of audited actions"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"


class RiskLevel(Enum):
    """Risk level of audited actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditContext:
    """Context information for audit events"""
    tenant_id: Optional[int] = None
    organization_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    api_version: Optional[str] = None
    client_version: Optional[str] = None
    environment: str = "prod"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "user_agent": self.user_agent,
            "client_ip": self.client_ip,
            "api_version": self.api_version,
            "client_version": self.client_version,
            "environment": self.environment
        }


@dataclass
class AuditEvent:
    """Individual audit event"""
    audit_id: str
    action_type: AuditActionType
    resource_type: str
    resource_id: str
    resource_name: Optional[str] = None
    
    # Actor information
    actor_type: AuditActorType = AuditActorType.USER
    actor_id: str = ""
    actor_name: Optional[str] = None
    
    # Context
    context: AuditContext = field(default_factory=AuditContext)
    
    # Change details
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    change_summary: Optional[str] = None
    change_reason: Optional[str] = None
    
    # Outcome
    status: AuditStatus = AuditStatus.SUCCESS
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # Risk and compliance
    risk_level: RiskLevel = RiskLevel.LOW
    compliance_impact: bool = False
    security_impact: bool = False
    approval_required: bool = False
    evidence_generated: bool = False
    
    # Integrity
    audit_hash: Optional[str] = None
    signature: Optional[str] = None
    
    def complete(self, status: AuditStatus = AuditStatus.SUCCESS, error_message: Optional[str] = None):
        """Mark audit event as completed"""
        self.completed_at = datetime.now(timezone.utc)
        self.status = status
        if error_message:
            self.error_message = error_message
        
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)
    
    def calculate_hash(self) -> str:
        """Calculate hash of audit event for integrity"""
        # Create deterministic string representation
        hash_data = {
            "audit_id": self.audit_id,
            "action_type": self.action_type.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "actor_type": self.actor_type.value,
            "actor_id": self.actor_id,
            "started_at": self.started_at.isoformat(),
            "status": self.status.value,
            "before_state": self.before_state,
            "after_state": self.after_state
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            "audit_id": self.audit_id,
            "action_type": self.action_type.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "actor_type": self.actor_type.value,
            "actor_id": self.actor_id,
            "actor_name": self.actor_name,
            "context": self.context.to_dict(),
            "before_state": self.before_state,
            "after_state": self.after_state,
            "change_summary": self.change_summary,
            "change_reason": self.change_reason,
            "status": self.status.value,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "risk_level": self.risk_level.value,
            "compliance_impact": self.compliance_impact,
            "security_impact": self.security_impact,
            "approval_required": self.approval_required,
            "evidence_generated": self.evidence_generated,
            "audit_hash": self.audit_hash,
            "signature": self.signature
        }


class AuditLogger:
    """
    Comprehensive audit logger for registry actions with non-repudiation
    """
    
    def __init__(self, db_pool: asyncpg.Pool, signing_key: Optional[rsa.RSAPrivateKey] = None):
        self.db_pool = db_pool
        self.signing_key = signing_key or self._generate_signing_key()
        self.public_key = self.signing_key.public_key()
        self._event_buffer: List[AuditEvent] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task = None
        self._start_background_flush()
    
    def _generate_signing_key(self) -> rsa.RSAPrivateKey:
        """Generate RSA signing key for audit integrity"""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
    
    def _start_background_flush(self):
        """Start background task to flush audit events"""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._background_flush())
    
    async def _background_flush(self):
        """Background task to periodically flush audit events"""
        while True:
            try:
                await asyncio.sleep(5)  # Flush every 5 seconds
                await self.flush_events()
            except Exception as e:
                print(f"Error in background audit flush: {e}")
    
    @asynccontextmanager
    async def audit_action(
        self,
        action_type: AuditActionType,
        resource_type: str,
        resource_id: str,
        actor_id: str,
        context: Optional[AuditContext] = None,
        **kwargs
    ):
        """Context manager for auditing actions"""
        
        # Create audit event
        event = AuditEvent(
            audit_id=str(uuid.uuid4()),
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            actor_id=actor_id,
            context=context or AuditContext(),
            **kwargs
        )
        
        try:
            yield event
            event.complete(AuditStatus.SUCCESS)
        except Exception as e:
            event.complete(AuditStatus.FAILURE, str(e))
            raise
        finally:
            await self.log_event(event)
    
    async def log_event(self, event: AuditEvent):
        """Log audit event"""
        # Calculate integrity hash
        event.audit_hash = event.calculate_hash()
        
        # Sign the event if signing key is available
        if self.signing_key:
            event.signature = self._sign_event(event)
        
        # Add to buffer
        async with self._buffer_lock:
            self._event_buffer.append(event)
        
        # Flush immediately for high-risk events
        if event.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            await self.flush_events()
    
    def _sign_event(self, event: AuditEvent) -> str:
        """Sign audit event for non-repudiation"""
        try:
            # Create signature payload
            payload = f"{event.audit_id}:{event.audit_hash}:{event.started_at.isoformat()}"
            
            # Sign with RSA-PSS
            signature = self.signing_key.sign(
                payload.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature.hex()
            
        except Exception as e:
            print(f"Error signing audit event: {e}")
            return ""
    
    async def flush_events(self):
        """Flush buffered audit events to database"""
        if not self._event_buffer:
            return
        
        async with self._buffer_lock:
            events_to_flush = self._event_buffer.copy()
            self._event_buffer.clear()
        
        if not events_to_flush:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Prepare batch insert
                values = []
                for event in events_to_flush:
                    values.append((
                        event.audit_id,
                        event.context.correlation_id,
                        event.context.session_id,
                        event.action_type.value,
                        event.resource_type,
                        event.resource_id,
                        event.resource_name,
                        event.actor_type.value,
                        event.actor_id,
                        event.actor_name,
                        event.context.client_ip,
                        event.context.user_agent,
                        event.context.tenant_id,
                        event.context.organization_id,
                        event.context.environment,
                        event.context.api_version,
                        event.context.client_version,
                        json.dumps(event.before_state) if event.before_state else None,
                        json.dumps(event.after_state) if event.after_state else None,
                        event.change_summary,
                        event.change_reason,
                        event.status.value,
                        event.error_code,
                        event.error_message,
                        event.warnings,
                        event.started_at,
                        event.completed_at,
                        event.duration_ms,
                        event.compliance_impact,
                        event.security_impact,
                        event.risk_level.value,
                        event.approval_required,
                        event.evidence_generated,
                        event.audit_hash
                    ))
                
                # Batch insert
                await conn.executemany("""
                    INSERT INTO registry_audit_log (
                        audit_id, correlation_id, session_id, action_type, resource_type,
                        resource_id, resource_name, actor_type, actor_id, actor_name,
                        actor_ip, user_agent, tenant_id, organization_id, environment,
                        api_version, client_version, before_state, after_state,
                        change_summary, change_reason, status, error_code, error_message,
                        warnings, started_at, completed_at, duration_ms, compliance_impact,
                        security_impact, risk_level, approval_required, evidence_generated,
                        audit_hash
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                             $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29,
                             $30, $31, $32, $33, $34)
                """, values)
                
        except Exception as e:
            print(f"Error flushing audit events: {e}")
            # Re-add events to buffer for retry
            async with self._buffer_lock:
                self._event_buffer.extend(events_to_flush)
    
    async def search_audit_logs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search audit logs with filters"""
        
        query_parts = ["SELECT * FROM registry_audit_log WHERE 1=1"]
        params = []
        param_count = 0
        
        # Time range filter
        if start_time:
            param_count += 1
            query_parts.append(f"AND started_at >= ${param_count}")
            params.append(start_time)
        
        if end_time:
            param_count += 1
            query_parts.append(f"AND started_at <= ${param_count}")
            params.append(end_time)
        
        # Dynamic filters
        if filters:
            for key, value in filters.items():
                if key in ['action_type', 'resource_type', 'actor_type', 'status', 'risk_level']:
                    param_count += 1
                    query_parts.append(f"AND {key} = ${param_count}")
                    params.append(value)
                elif key == 'tenant_id':
                    param_count += 1
                    query_parts.append(f"AND tenant_id = ${param_count}")
                    params.append(value)
                elif key == 'actor_id':
                    param_count += 1
                    query_parts.append(f"AND actor_id = ${param_count}")
                    params.append(value)
                elif key == 'resource_id':
                    param_count += 1
                    query_parts.append(f"AND resource_id = ${param_count}")
                    params.append(value)
        
        # Order and pagination
        query_parts.append("ORDER BY started_at DESC")
        param_count += 1
        query_parts.append(f"LIMIT ${param_count}")
        params.append(limit)
        
        param_count += 1
        query_parts.append(f"OFFSET ${param_count}")
        params.append(offset)
        
        query = " ".join(query_parts)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def get_audit_trail(self, resource_id: str, resource_type: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a resource"""
        return await self.search_audit_logs(
            filters={
                'resource_id': resource_id,
                'resource_type': resource_type
            },
            limit=1000
        )
    
    async def verify_audit_integrity(self, audit_id: str) -> Dict[str, Any]:
        """Verify integrity of an audit event"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM registry_audit_log WHERE audit_id = $1",
                audit_id
            )
            
            if not row:
                return {"valid": False, "error": "Audit event not found"}
            
            # Reconstruct event for hash verification
            event_data = dict(row)
            
            # Calculate expected hash
            hash_data = {
                "audit_id": event_data["audit_id"],
                "action_type": event_data["action_type"],
                "resource_type": event_data["resource_type"],
                "resource_id": event_data["resource_id"],
                "actor_type": event_data["actor_type"],
                "actor_id": event_data["actor_id"],
                "started_at": event_data["started_at"].isoformat(),
                "status": event_data["status"],
                "before_state": json.loads(event_data["before_state"]) if event_data["before_state"] else None,
                "after_state": json.loads(event_data["after_state"]) if event_data["after_state"] else None
            }
            
            hash_string = json.dumps(hash_data, sort_keys=True, default=str)
            expected_hash = hashlib.sha256(hash_string.encode()).hexdigest()
            
            return {
                "valid": expected_hash == event_data["audit_hash"],
                "stored_hash": event_data["audit_hash"],
                "calculated_hash": expected_hash,
                "event_data": event_data
            }
    
    async def generate_compliance_report(
        self,
        tenant_id: int,
        start_date: datetime,
        end_date: datetime,
        compliance_framework: str = "SOX"
    ) -> Dict[str, Any]:
        """Generate compliance audit report"""
        
        # Get compliance-relevant events
        compliance_events = await self.search_audit_logs(
            filters={
                'tenant_id': tenant_id,
                'compliance_impact': True
            },
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )
        
        # Categorize events
        event_categories = {}
        risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for event in compliance_events:
            action_type = event['action_type']
            risk_level = event['risk_level']
            
            if action_type not in event_categories:
                event_categories[action_type] = 0
            event_categories[action_type] += 1
            
            if risk_level in risk_distribution:
                risk_distribution[risk_level] += 1
        
        return {
            "tenant_id": tenant_id,
            "compliance_framework": compliance_framework,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": len(compliance_events),
                "event_categories": event_categories,
                "risk_distribution": risk_distribution
            },
            "events": compliance_events,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def close(self):
        """Clean shutdown of audit logger"""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush_events()


# Audit decorators for easy integration
def audit_action(
    action_type: AuditActionType,
    resource_type: str,
    risk_level: RiskLevel = RiskLevel.LOW
):
    """Decorator to automatically audit function calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract audit logger from args/kwargs
            audit_logger = kwargs.get('audit_logger') or getattr(args[0], 'audit_logger', None)
            
            if not audit_logger:
                # No audit logger available, execute without auditing
                return await func(*args, **kwargs)
            
            # Extract context information
            resource_id = kwargs.get('resource_id', 'unknown')
            actor_id = kwargs.get('actor_id', 'system')
            context = kwargs.get('audit_context', AuditContext())
            
            async with audit_logger.audit_action(
                action_type=action_type,
                resource_type=resource_type,
                resource_id=resource_id,
                actor_id=actor_id,
                context=context,
                risk_level=risk_level
            ) as event:
                # Capture before state if available
                if hasattr(args[0], 'get_current_state'):
                    event.before_state = await args[0].get_current_state(resource_id)
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Capture after state if available
                if hasattr(args[0], 'get_current_state'):
                    event.after_state = await args[0].get_current_state(resource_id)
                
                return result
        
        return wrapper
    return decorator


# Example usage
async def example_audit_usage():
    """Example of audit logger usage"""
    
    # Mock database pool (would be real in production)
    class MockPool:
        async def acquire(self):
            return self
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
        
        async def executemany(self, query, values):
            print(f"Would execute: {query} with {len(values)} values")
        
        async def fetch(self, query, *params):
            return []
        
        async def fetchrow(self, query, *params):
            return None
    
    # Create audit logger
    audit_logger = AuditLogger(MockPool())
    
    # Example 1: Using context manager
    context = AuditContext(
        tenant_id=1300,
        session_id="session-123",
        client_ip="192.168.1.100"
    )
    
    async with audit_logger.audit_action(
        action_type=AuditActionType.WORKFLOW_CREATE,
        resource_type="workflow",
        resource_id="workflow-123",
        actor_id="user-456",
        context=context,
        risk_level=RiskLevel.MEDIUM
    ) as event:
        # Simulate workflow creation
        event.change_summary = "Created new pipeline hygiene workflow"
        event.after_state = {
            "workflow_name": "Pipeline Hygiene",
            "version": "1.0.0",
            "status": "draft"
        }
        print("Workflow created successfully")
    
    # Example 2: Direct event logging
    event = AuditEvent(
        audit_id=str(uuid.uuid4()),
        action_type=AuditActionType.ARTIFACT_DOWNLOAD,
        resource_type="artifact",
        resource_id="artifact-789",
        actor_id="user-456",
        context=context
    )
    
    event.complete(AuditStatus.SUCCESS)
    await audit_logger.log_event(event)
    
    # Flush events
    await audit_logger.flush_events()
    
    print("Audit logging example completed")
    
    # Clean shutdown
    await audit_logger.close()


if __name__ == "__main__":
    asyncio.run(example_audit_usage())
