"""
Task 8.3-T16: Integrate access logs with Evidence Pack schema
Link access control events to evidence packs for compliance traceability
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class AccessEventType(Enum):
    """Types of access events"""
    LOGIN = "login"
    LOGOUT = "logout"
    RESOURCE_ACCESS = "resource_access"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_DENY = "permission_deny"
    ROLE_ASSIGNMENT = "role_assignment"
    ROLE_REMOVAL = "role_removal"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AccessOutcome(Enum):
    """Access attempt outcomes"""
    GRANTED = "granted"
    DENIED = "denied"
    ESCALATED = "escalated"
    DEFERRED = "deferred"


@dataclass
class AccessLogEntry:
    """Individual access log entry"""
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: int = 0
    
    # Access details
    event_type: AccessEventType = AccessEventType.RESOURCE_ACCESS
    outcome: AccessOutcome = AccessOutcome.GRANTED
    
    # Actor information
    actor_id: str = ""
    actor_type: str = "user"  # user, service, system
    actor_role: str = ""
    actor_ip: Optional[str] = None
    
    # Resource information
    resource_type: str = ""
    resource_id: str = ""
    resource_path: Optional[str] = None
    
    # Access context
    requested_permission: str = ""
    granted_permissions: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    
    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Evidence linkage
    evidence_pack_id: Optional[str] = None
    workflow_execution_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Compliance metadata
    compliance_frameworks: List[str] = field(default_factory=list)
    retention_period_days: int = 2555  # 7 years default
    
    # Additional context
    user_agent: Optional[str] = None
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "tenant_id": self.tenant_id,
            "event_type": self.event_type.value,
            "outcome": self.outcome.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "actor_role": self.actor_role,
            "actor_ip": self.actor_ip,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_path": self.resource_path,
            "requested_permission": self.requested_permission,
            "granted_permissions": self.granted_permissions,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "evidence_pack_id": self.evidence_pack_id,
            "workflow_execution_id": self.workflow_execution_id,
            "trace_id": self.trace_id,
            "compliance_frameworks": self.compliance_frameworks,
            "retention_period_days": self.retention_period_days,
            "user_agent": self.user_agent,
            "request_metadata": self.request_metadata
        }


@dataclass
class EvidencePackAccessSummary:
    """Summary of access events for an evidence pack"""
    evidence_pack_id: str
    tenant_id: int
    
    # Access statistics
    total_access_events: int = 0
    granted_access_count: int = 0
    denied_access_count: int = 0
    unique_actors_count: int = 0
    
    # Time range
    first_access: Optional[datetime] = None
    last_access: Optional[datetime] = None
    
    # Access patterns
    access_by_role: Dict[str, int] = field(default_factory=dict)
    access_by_resource_type: Dict[str, int] = field(default_factory=dict)
    access_by_outcome: Dict[str, int] = field(default_factory=dict)
    
    # Compliance
    compliance_frameworks: Set[str] = field(default_factory=set)
    
    # Timestamps
    summary_generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_pack_id": self.evidence_pack_id,
            "tenant_id": self.tenant_id,
            "total_access_events": self.total_access_events,
            "granted_access_count": self.granted_access_count,
            "denied_access_count": self.denied_access_count,
            "unique_actors_count": self.unique_actors_count,
            "first_access": self.first_access.isoformat() if self.first_access else None,
            "last_access": self.last_access.isoformat() if self.last_access else None,
            "access_by_role": self.access_by_role,
            "access_by_resource_type": self.access_by_resource_type,
            "access_by_outcome": self.access_by_outcome,
            "compliance_frameworks": list(self.compliance_frameworks),
            "summary_generated_at": self.summary_generated_at.isoformat()
        }


class EvidencePackAccessIntegrator:
    """
    Evidence Pack Access Integration - Task 8.3-T16
    
    Integrates access control events with evidence pack schema
    Provides compliance traceability for access patterns
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'enable_real_time_integration': True,
            'batch_processing_enabled': True,
            'batch_size': 1000,
            'batch_interval_minutes': 5,
            'retention_policy_days': 2555,  # 7 years
            'enable_access_analytics': True,
            'generate_compliance_reports': True,
            'alert_on_suspicious_patterns': True
        }
        
        # In-memory buffer for batch processing
        self.access_log_buffer: List[AccessLogEntry] = []
        self.buffer_lock = asyncio.Lock()
        
        # Statistics
        self.integration_stats = {
            'total_access_events_processed': 0,
            'evidence_packs_linked': 0,
            'compliance_reports_generated': 0,
            'suspicious_patterns_detected': 0,
            'average_processing_time_ms': 0.0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    async def initialize(self) -> bool:
        """Initialize evidence pack access integrator"""
        try:
            await self._create_access_integration_tables()
            if self.config['batch_processing_enabled']:
                await self._start_background_tasks()
            self.is_running = True
            self.logger.info("✅ Evidence pack access integrator initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize evidence pack access integrator: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown integrator"""
        self.is_running = False
        
        # Process remaining buffer
        if self.access_log_buffer:
            await self._process_access_log_batch(self.access_log_buffer.copy())
            self.access_log_buffer.clear()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("🛑 Evidence pack access integrator shutdown")
    
    async def log_access_event(
        self,
        access_event: AccessLogEntry,
        evidence_pack_id: Optional[str] = None,
        workflow_execution_id: Optional[str] = None
    ) -> bool:
        """
        Log access event and integrate with evidence pack
        
        This is the main entry point for access logging
        """
        
        try:
            # Set evidence pack linkage
            if evidence_pack_id:
                access_event.evidence_pack_id = evidence_pack_id
            if workflow_execution_id:
                access_event.workflow_execution_id = workflow_execution_id
            
            # Real-time processing
            if self.config['enable_real_time_integration']:
                await self._process_access_event_immediately(access_event)
            
            # Add to batch buffer
            if self.config['batch_processing_enabled']:
                async with self.buffer_lock:
                    self.access_log_buffer.append(access_event)
            
            self.integration_stats['total_access_events_processed'] += 1
            
            self.logger.debug(f"📝 Access event logged: {access_event.event_type.value} by {access_event.actor_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log access event: {e}")
            return False
    
    async def link_access_logs_to_evidence_pack(
        self,
        evidence_pack_id: str,
        tenant_id: int,
        time_range_start: datetime,
        time_range_end: datetime
    ) -> EvidencePackAccessSummary:
        """
        Link existing access logs to evidence pack and generate summary
        """
        
        try:
            # Get access logs in time range
            access_logs = await self._get_access_logs_in_range(
                tenant_id, time_range_start, time_range_end
            )
            
            # Update access logs with evidence pack ID
            await self._update_access_logs_evidence_pack_id(
                [log['log_id'] for log in access_logs], evidence_pack_id
            )
            
            # Generate access summary
            summary = await self._generate_access_summary(evidence_pack_id, access_logs)
            
            # Store summary
            await self._store_evidence_pack_access_summary(summary)
            
            self.integration_stats['evidence_packs_linked'] += 1
            
            self.logger.info(f"🔗 Linked {len(access_logs)} access logs to evidence pack {evidence_pack_id}")
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to link access logs to evidence pack: {e}")
            raise
    
    async def generate_compliance_access_report(
        self,
        evidence_pack_id: str,
        compliance_framework: str = "SOX"
    ) -> Dict[str, Any]:
        """
        Generate compliance-specific access report for evidence pack
        """
        
        try:
            # Get evidence pack access summary
            summary = await self._get_evidence_pack_access_summary(evidence_pack_id)
            
            if not summary:
                raise ValueError(f"No access summary found for evidence pack {evidence_pack_id}")
            
            # Get detailed access logs
            access_logs = await self._get_evidence_pack_access_logs(evidence_pack_id)
            
            # Generate compliance report
            report = await self._generate_compliance_report(
                summary, access_logs, compliance_framework
            )
            
            # Store report
            await self._store_compliance_report(evidence_pack_id, compliance_framework, report)
            
            self.integration_stats['compliance_reports_generated'] += 1
            
            self.logger.info(f"📊 Generated {compliance_framework} compliance report for evidence pack {evidence_pack_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate compliance access report: {e}")
            raise
    
    async def detect_suspicious_access_patterns(
        self,
        evidence_pack_id: str,
        tenant_id: int
    ) -> List[Dict[str, Any]]:
        """
        Detect suspicious access patterns in evidence pack
        """
        
        try:
            # Get access logs for evidence pack
            access_logs = await self._get_evidence_pack_access_logs(evidence_pack_id)
            
            suspicious_patterns = []
            
            # Pattern 1: Excessive access by single user
            user_access_counts = {}
            for log in access_logs:
                actor_id = log['actor_id']
                user_access_counts[actor_id] = user_access_counts.get(actor_id, 0) + 1
            
            for actor_id, count in user_access_counts.items():
                if count > 50:  # Threshold for excessive access
                    suspicious_patterns.append({
                        'pattern_type': 'excessive_access',
                        'actor_id': actor_id,
                        'access_count': count,
                        'severity': 'medium',
                        'description': f"User {actor_id} accessed evidence pack {count} times"
                    })
            
            # Pattern 2: Access outside business hours
            for log in access_logs:
                access_time = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                if access_time.hour < 6 or access_time.hour > 22:
                    suspicious_patterns.append({
                        'pattern_type': 'off_hours_access',
                        'actor_id': log['actor_id'],
                        'access_time': access_time.isoformat(),
                        'severity': 'low',
                        'description': f"Off-hours access by {log['actor_id']} at {access_time}"
                    })
            
            # Pattern 3: Multiple failed access attempts
            failed_attempts = [log for log in access_logs if log['outcome'] == 'denied']
            if len(failed_attempts) > 10:
                suspicious_patterns.append({
                    'pattern_type': 'multiple_failed_attempts',
                    'failed_count': len(failed_attempts),
                    'severity': 'high',
                    'description': f"{len(failed_attempts)} failed access attempts detected"
                })
            
            # Store suspicious patterns
            if suspicious_patterns:
                await self._store_suspicious_patterns(evidence_pack_id, suspicious_patterns)
                self.integration_stats['suspicious_patterns_detected'] += len(suspicious_patterns)
            
            self.logger.info(f"🔍 Detected {len(suspicious_patterns)} suspicious patterns in evidence pack {evidence_pack_id}")
            return suspicious_patterns
            
        except Exception as e:
            self.logger.error(f"❌ Failed to detect suspicious access patterns: {e}")
            return []
    
    async def _process_access_event_immediately(self, access_event: AccessLogEntry):
        """Process access event immediately for real-time integration"""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Store access log
            await self._store_access_log(access_event)
            
            # If linked to evidence pack, update summary
            if access_event.evidence_pack_id:
                await self._update_evidence_pack_access_summary(access_event)
            
            # Check for suspicious patterns
            if self.config['alert_on_suspicious_patterns']:
                await self._check_immediate_suspicious_patterns(access_event)
            
            # Update processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            current_avg = self.integration_stats['average_processing_time_ms']
            total_processed = self.integration_stats['total_access_events_processed']
            self.integration_stats['average_processing_time_ms'] = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process access event immediately: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        
        # Batch processor
        batch_task = asyncio.create_task(self._batch_processor())
        self.background_tasks.append(batch_task)
        
        # Compliance report generator
        compliance_task = asyncio.create_task(self._compliance_report_generator())
        self.background_tasks.append(compliance_task)
        
        # Pattern detector
        pattern_task = asyncio.create_task(self._pattern_detector())
        self.background_tasks.append(pattern_task)
    
    async def _batch_processor(self):
        """Background task for batch processing access logs"""
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config['batch_interval_minutes'] * 60)
                
                # Get current buffer
                async with self.buffer_lock:
                    if not self.access_log_buffer:
                        continue
                    
                    batch = self.access_log_buffer.copy()
                    self.access_log_buffer.clear()
                
                # Process batch
                await self._process_access_log_batch(batch)
                
            except Exception as e:
                self.logger.error(f"❌ Batch processor error: {e}")
    
    async def _process_access_log_batch(self, batch: List[AccessLogEntry]):
        """Process batch of access logs"""
        
        if not batch:
            return
        
        try:
            # Store all access logs
            for access_event in batch:
                await self._store_access_log(access_event)
            
            # Update evidence pack summaries
            evidence_pack_events = {}
            for access_event in batch:
                if access_event.evidence_pack_id:
                    if access_event.evidence_pack_id not in evidence_pack_events:
                        evidence_pack_events[access_event.evidence_pack_id] = []
                    evidence_pack_events[access_event.evidence_pack_id].append(access_event)
            
            for evidence_pack_id, events in evidence_pack_events.items():
                await self._batch_update_evidence_pack_summary(evidence_pack_id, events)
            
            self.logger.info(f"📦 Processed batch of {len(batch)} access events")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process access log batch: {e}")
    
    async def _compliance_report_generator(self):
        """Background task for generating compliance reports"""
        
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Get evidence packs that need compliance reports
                evidence_packs = await self._get_evidence_packs_needing_reports()
                
                for evidence_pack in evidence_packs:
                    for framework in evidence_pack['compliance_frameworks']:
                        await self.generate_compliance_access_report(
                            evidence_pack['evidence_pack_id'], framework
                        )
                
            except Exception as e:
                self.logger.error(f"❌ Compliance report generator error: {e}")
    
    async def _pattern_detector(self):
        """Background task for detecting suspicious patterns"""
        
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Get evidence packs for pattern detection
                evidence_packs = await self._get_evidence_packs_for_pattern_detection()
                
                for evidence_pack in evidence_packs:
                    await self.detect_suspicious_access_patterns(
                        evidence_pack['evidence_pack_id'],
                        evidence_pack['tenant_id']
                    )
                
            except Exception as e:
                self.logger.error(f"❌ Pattern detector error: {e}")
    
    # Database operations
    
    async def _create_access_integration_tables(self):
        """Create access integration database tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Access logs with evidence pack integration
        CREATE TABLE IF NOT EXISTS evidence_pack_access_logs (
            log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id INTEGER NOT NULL,
            
            -- Access details
            event_type VARCHAR(50) NOT NULL,
            outcome VARCHAR(20) NOT NULL,
            
            -- Actor information
            actor_id VARCHAR(100) NOT NULL,
            actor_type VARCHAR(20) NOT NULL DEFAULT 'user',
            actor_role VARCHAR(100),
            actor_ip INET,
            
            -- Resource information
            resource_type VARCHAR(50) NOT NULL,
            resource_id VARCHAR(100) NOT NULL,
            resource_path TEXT,
            
            -- Access context
            requested_permission VARCHAR(100),
            granted_permissions TEXT[] DEFAULT ARRAY[],
            session_id VARCHAR(100),
            
            -- Evidence linkage
            evidence_pack_id VARCHAR(100),
            workflow_execution_id VARCHAR(100),
            trace_id VARCHAR(100),
            
            -- Compliance metadata
            compliance_frameworks TEXT[] DEFAULT ARRAY[],
            retention_period_days INTEGER NOT NULL DEFAULT 2555,
            
            -- Additional context
            user_agent TEXT,
            request_metadata JSONB DEFAULT '{}',
            
            -- Timestamps
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_access_event_type CHECK (event_type IN ('login', 'logout', 'resource_access', 'permission_grant', 'permission_deny', 'role_assignment', 'role_removal', 'privilege_escalation', 'unauthorized_access')),
            CONSTRAINT chk_access_outcome CHECK (outcome IN ('granted', 'denied', 'escalated', 'deferred'))
        );
        
        -- Evidence pack access summaries
        CREATE TABLE IF NOT EXISTS evidence_pack_access_summaries (
            summary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_pack_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Access statistics
            total_access_events INTEGER NOT NULL DEFAULT 0,
            granted_access_count INTEGER NOT NULL DEFAULT 0,
            denied_access_count INTEGER NOT NULL DEFAULT 0,
            unique_actors_count INTEGER NOT NULL DEFAULT 0,
            
            -- Time range
            first_access TIMESTAMPTZ,
            last_access TIMESTAMPTZ,
            
            -- Access patterns
            access_by_role JSONB DEFAULT '{}',
            access_by_resource_type JSONB DEFAULT '{}',
            access_by_outcome JSONB DEFAULT '{}',
            
            -- Compliance
            compliance_frameworks TEXT[] DEFAULT ARRAY[],
            
            -- Timestamps
            summary_generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT uq_evidence_pack_access_summary UNIQUE (evidence_pack_id)
        );
        
        -- Compliance access reports
        CREATE TABLE IF NOT EXISTS evidence_pack_compliance_reports (
            report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_pack_id VARCHAR(100) NOT NULL,
            compliance_framework VARCHAR(50) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Report content
            report_data JSONB NOT NULL,
            report_summary TEXT,
            
            -- Status
            report_status VARCHAR(20) NOT NULL DEFAULT 'generated',
            
            -- Timestamps
            generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_compliance_report_status CHECK (report_status IN ('generated', 'reviewed', 'approved', 'archived')),
            CONSTRAINT uq_evidence_pack_compliance_report UNIQUE (evidence_pack_id, compliance_framework)
        );
        
        -- Suspicious access patterns
        CREATE TABLE IF NOT EXISTS evidence_pack_suspicious_patterns (
            pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_pack_id VARCHAR(100) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Pattern details
            pattern_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            description TEXT NOT NULL,
            pattern_data JSONB DEFAULT '{}',
            
            -- Status
            status VARCHAR(20) NOT NULL DEFAULT 'detected',
            investigated_by VARCHAR(100),
            resolution_notes TEXT,
            
            -- Timestamps
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            investigated_at TIMESTAMPTZ,
            resolved_at TIMESTAMPTZ,
            
            -- Constraints
            CONSTRAINT chk_pattern_severity CHECK (severity IN ('low', 'medium', 'high', 'critical')),
            CONSTRAINT chk_pattern_status CHECK (status IN ('detected', 'investigating', 'resolved', 'false_positive'))
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_access_logs_evidence_pack ON evidence_pack_access_logs(evidence_pack_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_access_logs_actor ON evidence_pack_access_logs(actor_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_access_logs_tenant_time ON evidence_pack_access_logs(tenant_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_access_logs_outcome ON evidence_pack_access_logs(outcome, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_access_summaries_evidence_pack ON evidence_pack_access_summaries(evidence_pack_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_access_summaries_tenant ON evidence_pack_access_summaries(tenant_id, updated_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_compliance_reports_evidence_pack ON evidence_pack_compliance_reports(evidence_pack_id, compliance_framework);
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_compliance_reports_tenant ON evidence_pack_compliance_reports(tenant_id, generated_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_suspicious_patterns_evidence_pack ON evidence_pack_suspicious_patterns(evidence_pack_id, detected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_evidence_pack_suspicious_patterns_severity ON evidence_pack_suspicious_patterns(severity, status);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Evidence pack access integration tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create access integration tables: {e}")
            raise
    
    async def _store_access_log(self, access_event: AccessLogEntry):
        """Store access log in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO evidence_pack_access_logs (
                    log_id, tenant_id, event_type, outcome, actor_id, actor_type,
                    actor_role, actor_ip, resource_type, resource_id, resource_path,
                    requested_permission, granted_permissions, session_id,
                    evidence_pack_id, workflow_execution_id, trace_id,
                    compliance_frameworks, retention_period_days, user_agent,
                    request_metadata, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    access_event.log_id,
                    access_event.tenant_id,
                    access_event.event_type.value,
                    access_event.outcome.value,
                    access_event.actor_id,
                    access_event.actor_type,
                    access_event.actor_role,
                    access_event.actor_ip,
                    access_event.resource_type,
                    access_event.resource_id,
                    access_event.resource_path,
                    access_event.requested_permission,
                    access_event.granted_permissions,
                    access_event.session_id,
                    access_event.evidence_pack_id,
                    access_event.workflow_execution_id,
                    access_event.trace_id,
                    access_event.compliance_frameworks,
                    access_event.retention_period_days,
                    access_event.user_agent,
                    json.dumps(access_event.request_metadata),
                    access_event.timestamp
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store access log: {e}")
            raise
    
    # Additional helper methods would go here...
    # (Implementing all the database query methods, summary generation, etc.)
    
    async def _get_access_logs_in_range(
        self, tenant_id: int, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get access logs in time range"""
        # Implementation would query database
        return []
    
    async def _generate_access_summary(
        self, evidence_pack_id: str, access_logs: List[Dict[str, Any]]
    ) -> EvidencePackAccessSummary:
        """Generate access summary from logs"""
        # Implementation would analyze logs and create summary
        return EvidencePackAccessSummary(evidence_pack_id=evidence_pack_id, tenant_id=0)
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        
        return {
            'total_access_events_processed': self.integration_stats['total_access_events_processed'],
            'evidence_packs_linked': self.integration_stats['evidence_packs_linked'],
            'compliance_reports_generated': self.integration_stats['compliance_reports_generated'],
            'suspicious_patterns_detected': self.integration_stats['suspicious_patterns_detected'],
            'average_processing_time_ms': round(self.integration_stats['average_processing_time_ms'], 2),
            'buffer_size': len(self.access_log_buffer),
            'is_running': self.is_running,
            'real_time_integration_enabled': self.config['enable_real_time_integration'],
            'batch_processing_enabled': self.config['batch_processing_enabled']
        }


# Global evidence pack access integrator instance
evidence_pack_access_integrator = EvidencePackAccessIntegrator()
