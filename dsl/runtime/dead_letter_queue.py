# Dead Letter Queue for Failed RBA Workflows
# Tasks 6.3-T19, T20: Dead-letter queue and auto-ticket integration

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FailureReason(Enum):
    RETRY_EXHAUSTED = "retry_exhausted"
    PERMANENT_ERROR = "permanent_error"
    COMPLIANCE_VIOLATION = "compliance_violation"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    OVERRIDE_REJECTED = "override_rejected"

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DeadLetterEntry:
    """Dead letter queue entry for failed workflows"""
    entry_id: str
    tenant_id: int
    workflow_id: str
    execution_id: str
    failure_reason: FailureReason
    error_code: str
    error_message: str
    failure_timestamp: str
    retry_count: int
    original_input: Dict[str, Any]
    execution_context: Dict[str, Any]
    evidence_pack_id: Optional[str]
    escalation_id: Optional[str]
    ticket_id: Optional[str]
    resolution_status: str = "pending"
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None

class DeadLetterQueue:
    """
    Dead letter queue for capturing unresolvable workflow failures
    Tasks 6.3-T19, T20: Dead-letter queue + auto-ticket integration
    """
    
    def __init__(self, db_pool=None, ticket_service=None):
        self.db_pool = db_pool
        self.ticket_service = ticket_service
        self.auto_ticket_enabled = ticket_service is not None
        
        # Industry-specific ticket routing
        self.ticket_routing = {
            'SaaS': {
                'compliance_violation': {'priority': TicketPriority.CRITICAL, 'team': 'compliance'},
                'permanent_error': {'priority': TicketPriority.HIGH, 'team': 'engineering'},
                'retry_exhausted': {'priority': TicketPriority.MEDIUM, 'team': 'ops'}
            },
            'Banking': {
                'compliance_violation': {'priority': TicketPriority.CRITICAL, 'team': 'risk_compliance'},
                'permanent_error': {'priority': TicketPriority.CRITICAL, 'team': 'engineering'},
                'retry_exhausted': {'priority': TicketPriority.HIGH, 'team': 'ops'}
            },
            'Insurance': {
                'compliance_violation': {'priority': TicketPriority.CRITICAL, 'team': 'compliance'},
                'permanent_error': {'priority': TicketPriority.HIGH, 'team': 'engineering'},
                'retry_exhausted': {'priority': TicketPriority.MEDIUM, 'team': 'ops'}
            }
        }
    
    async def add_failed_workflow(
        self,
        tenant_id: int,
        workflow_id: str,
        execution_id: str,
        failure_reason: FailureReason,
        error_code: str,
        error_message: str,
        retry_count: int,
        original_input: Dict[str, Any],
        execution_context: Dict[str, Any],
        evidence_pack_id: Optional[str] = None,
        escalation_id: Optional[str] = None
    ) -> str:
        """Add failed workflow to dead letter queue"""
        
        entry_id = str(uuid.uuid4())
        
        # Create dead letter entry
        entry = DeadLetterEntry(
            entry_id=entry_id,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            execution_id=execution_id,
            failure_reason=failure_reason,
            error_code=error_code,
            error_message=error_message,
            failure_timestamp=datetime.now(timezone.utc).isoformat(),
            retry_count=retry_count,
            original_input=original_input,
            execution_context=execution_context,
            evidence_pack_id=evidence_pack_id,
            escalation_id=escalation_id,
            ticket_id=None
        )
        
        # Store in database
        await self._store_dead_letter_entry(entry)
        
        # Auto-create ticket if enabled
        if self.auto_ticket_enabled:
            ticket_id = await self._create_auto_ticket(entry)
            entry.ticket_id = ticket_id
            await self._update_dead_letter_entry(entry)
        
        logger.info(f"Added workflow to dead letter queue: {entry_id}")
        return entry_id
    
    async def _create_auto_ticket(self, entry: DeadLetterEntry) -> Optional[str]:
        """Create automatic ticket for failed workflow"""
        try:
            industry_code = entry.execution_context.get('industry_code', 'SaaS')
            routing_config = self.ticket_routing.get(industry_code, {}).get(
                entry.failure_reason.value, 
                {'priority': TicketPriority.MEDIUM, 'team': 'ops'}
            )
            
            # Create ticket payload
            ticket_data = {
                'title': f"RBA Workflow Failure: {entry.workflow_id}",
                'description': self._generate_ticket_description(entry),
                'priority': routing_config['priority'].value,
                'assigned_team': routing_config['team'],
                'labels': [
                    f"tenant:{entry.tenant_id}",
                    f"workflow:{entry.workflow_id}",
                    f"error_code:{entry.error_code}",
                    f"failure_reason:{entry.failure_reason.value}",
                    f"industry:{industry_code}"
                ],
                'metadata': {
                    'dead_letter_entry_id': entry.entry_id,
                    'execution_id': entry.execution_id,
                    'evidence_pack_id': entry.evidence_pack_id,
                    'escalation_id': entry.escalation_id
                }
            }
            
            # Create ticket via service
            ticket_id = await self.ticket_service.create_ticket(ticket_data)
            
            logger.info(f"Created auto-ticket {ticket_id} for dead letter entry {entry.entry_id}")
            return ticket_id
            
        except Exception as e:
            logger.error(f"Failed to create auto-ticket for {entry.entry_id}: {e}")
            return None
    
    def _generate_ticket_description(self, entry: DeadLetterEntry) -> str:
        """Generate detailed ticket description"""
        return f"""
**RBA Workflow Failure Report**

**Workflow Details:**
- Workflow ID: {entry.workflow_id}
- Execution ID: {entry.execution_id}
- Tenant ID: {entry.tenant_id}
- Failure Time: {entry.failure_timestamp}

**Error Information:**
- Error Code: {entry.error_code}
- Failure Reason: {entry.failure_reason.value}
- Error Message: {entry.error_message}
- Retry Count: {entry.retry_count}

**Context:**
- Industry: {entry.execution_context.get('industry_code', 'Unknown')}
- User: {entry.execution_context.get('user_id', 'Unknown')}
- Compliance Frameworks: {entry.execution_context.get('compliance_frameworks', [])}

**Evidence & Escalation:**
- Evidence Pack ID: {entry.evidence_pack_id or 'None'}
- Escalation ID: {entry.escalation_id or 'None'}

**Next Steps:**
1. Review execution context and error details
2. Check evidence pack for complete audit trail
3. Determine if workflow needs modification or data correction
4. Update resolution status in dead letter queue

**Dead Letter Entry ID:** {entry.entry_id}
        """.strip()
    
    async def _store_dead_letter_entry(self, entry: DeadLetterEntry) -> None:
        """Store dead letter entry in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO dsl_dead_letter_queue (
                        entry_id, tenant_id, workflow_id, execution_id,
                        failure_reason, error_code, error_message, failure_timestamp,
                        retry_count, original_input, execution_context,
                        evidence_pack_id, escalation_id, ticket_id,
                        resolution_status, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    entry.entry_id,
                    entry.tenant_id,
                    entry.workflow_id,
                    entry.execution_id,
                    entry.failure_reason.value,
                    entry.error_code,
                    entry.error_message,
                    entry.failure_timestamp,
                    entry.retry_count,
                    json.dumps(entry.original_input),
                    json.dumps(entry.execution_context),
                    entry.evidence_pack_id,
                    entry.escalation_id,
                    entry.ticket_id,
                    entry.resolution_status,
                    datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error(f"Failed to store dead letter entry: {e}")
            raise
    
    async def _update_dead_letter_entry(self, entry: DeadLetterEntry) -> None:
        """Update dead letter entry with ticket ID"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE dsl_dead_letter_queue 
                    SET ticket_id = $1, updated_at = $2
                    WHERE entry_id = $3
                """, entry.ticket_id, datetime.now(timezone.utc), entry.entry_id)
                
        except Exception as e:
            logger.error(f"Failed to update dead letter entry: {e}")
    
    async def get_pending_entries(
        self, 
        tenant_id: Optional[int] = None,
        limit: int = 100
    ) -> List[DeadLetterEntry]:
        """Get pending dead letter entries"""
        if not self.db_pool:
            return []
        
        try:
            async with self.db_pool.acquire() as conn:
                if tenant_id:
                    rows = await conn.fetch("""
                        SELECT * FROM dsl_dead_letter_queue
                        WHERE tenant_id = $1 AND resolution_status = 'pending'
                        ORDER BY failure_timestamp DESC
                        LIMIT $2
                    """, tenant_id, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT * FROM dsl_dead_letter_queue
                        WHERE resolution_status = 'pending'
                        ORDER BY failure_timestamp DESC
                        LIMIT $1
                    """, limit)
                
                entries = []
                for row in rows:
                    entry = DeadLetterEntry(
                        entry_id=row['entry_id'],
                        tenant_id=row['tenant_id'],
                        workflow_id=row['workflow_id'],
                        execution_id=row['execution_id'],
                        failure_reason=FailureReason(row['failure_reason']),
                        error_code=row['error_code'],
                        error_message=row['error_message'],
                        failure_timestamp=row['failure_timestamp'],
                        retry_count=row['retry_count'],
                        original_input=json.loads(row['original_input']),
                        execution_context=json.loads(row['execution_context']),
                        evidence_pack_id=row['evidence_pack_id'],
                        escalation_id=row['escalation_id'],
                        ticket_id=row['ticket_id'],
                        resolution_status=row['resolution_status'],
                        resolved_at=row['resolved_at'],
                        resolved_by=row['resolved_by'],
                        resolution_notes=row['resolution_notes']
                    )
                    entries.append(entry)
                
                return entries
                
        except Exception as e:
            logger.error(f"Failed to get pending entries: {e}")
            return []
    
    async def resolve_entry(
        self,
        entry_id: str,
        resolved_by: str,
        resolution_notes: str
    ) -> bool:
        """Mark dead letter entry as resolved"""
        if not self.db_pool:
            return False
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE dsl_dead_letter_queue 
                    SET resolution_status = 'resolved',
                        resolved_at = $1,
                        resolved_by = $2,
                        resolution_notes = $3,
                        updated_at = $1
                    WHERE entry_id = $4
                """, 
                    datetime.now(timezone.utc),
                    resolved_by,
                    resolution_notes,
                    entry_id
                )
                
                return result == "UPDATE 1"
                
        except Exception as e:
            logger.error(f"Failed to resolve entry {entry_id}: {e}")
            return False
    
    async def get_statistics(
        self, 
        tenant_id: Optional[int] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get dead letter queue statistics"""
        if not self.db_pool:
            return {}
        
        try:
            since_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            async with self.db_pool.acquire() as conn:
                if tenant_id:
                    stats_query = """
                        SELECT 
                            COUNT(*) as total_entries,
                            COUNT(CASE WHEN resolution_status = 'pending' THEN 1 END) as pending_entries,
                            COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) as resolved_entries,
                            COUNT(CASE WHEN ticket_id IS NOT NULL THEN 1 END) as entries_with_tickets,
                            failure_reason,
                            COUNT(*) as count_by_reason
                        FROM dsl_dead_letter_queue
                        WHERE tenant_id = $1 AND failure_timestamp >= $2
                        GROUP BY failure_reason
                    """
                    rows = await conn.fetch(stats_query, tenant_id, since_date)
                else:
                    stats_query = """
                        SELECT 
                            COUNT(*) as total_entries,
                            COUNT(CASE WHEN resolution_status = 'pending' THEN 1 END) as pending_entries,
                            COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) as resolved_entries,
                            COUNT(CASE WHEN ticket_id IS NOT NULL THEN 1 END) as entries_with_tickets,
                            failure_reason,
                            COUNT(*) as count_by_reason
                        FROM dsl_dead_letter_queue
                        WHERE failure_timestamp >= $1
                        GROUP BY failure_reason
                    """
                    rows = await conn.fetch(stats_query, since_date)
                
                # Aggregate statistics
                total_entries = sum(row['total_entries'] for row in rows) if rows else 0
                pending_entries = sum(row['pending_entries'] for row in rows) if rows else 0
                resolved_entries = sum(row['resolved_entries'] for row in rows) if rows else 0
                entries_with_tickets = sum(row['entries_with_tickets'] for row in rows) if rows else 0
                
                failure_breakdown = {}
                for row in rows:
                    failure_breakdown[row['failure_reason']] = row['count_by_reason']
                
                return {
                    'total_entries': total_entries,
                    'pending_entries': pending_entries,
                    'resolved_entries': resolved_entries,
                    'entries_with_tickets': entries_with_tickets,
                    'resolution_rate': (resolved_entries / total_entries * 100) if total_entries > 0 else 0,
                    'auto_ticket_rate': (entries_with_tickets / total_entries * 100) if total_entries > 0 else 0,
                    'failure_breakdown': failure_breakdown,
                    'period_days': days_back,
                    'tenant_id': tenant_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

class TicketService:
    """Mock ticket service for Jira/ServiceNow integration"""
    
    async def create_ticket(self, ticket_data: Dict[str, Any]) -> str:
        """Create ticket in external system"""
        # This would integrate with actual Jira/ServiceNow APIs
        ticket_id = f"RBA-{uuid.uuid4().hex[:8].upper()}"
        
        logger.info(f"Created ticket {ticket_id}: {ticket_data['title']}")
        return ticket_id

# Global dead letter queue instance
dead_letter_queue = DeadLetterQueue()
