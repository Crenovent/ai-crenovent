"""
Task 14.1-T36: Build approval notification hooks (Slack, Teams, Email)
Real-time communication service for approval workflows
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ApprovalEventType(Enum):
    """Types of approval events"""
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_APPROVED = "approval_approved"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_ESCALATED = "approval_escalated"
    APPROVAL_EXPIRED = "approval_expired"
    OVERRIDE_REQUESTED = "override_requested"
    OVERRIDE_APPROVED = "override_approved"
    OVERRIDE_REJECTED = "override_rejected"
    SLA_BREACH_WARNING = "sla_breach_warning"
    SLA_BREACH_CRITICAL = "sla_breach_critical"


@dataclass
class NotificationTemplate:
    """Notification template configuration"""
    template_id: str
    event_type: ApprovalEventType
    channel: NotificationChannel
    
    # Template content
    subject_template: str
    body_template: str
    
    # Targeting
    target_roles: List[str] = field(default_factory=list)
    target_users: List[str] = field(default_factory=list)
    
    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Settings
    priority: NotificationPriority = NotificationPriority.NORMAL
    retry_attempts: int = 3
    delay_minutes: int = 0
    
    # Metadata
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NotificationContext:
    """Context data for notification rendering"""
    approval_id: str
    workflow_id: str
    tenant_id: int
    
    # Approval details
    request_type: str
    requester_name: str
    requester_email: str
    approval_reason: str
    business_justification: str
    risk_level: str
    
    # Timing
    created_at: datetime
    expires_at: Optional[datetime] = None
    sla_hours_remaining: Optional[float] = None
    
    # Approvers
    current_approver: Optional[str] = None
    approval_chain: List[str] = field(default_factory=list)
    
    # Links
    approval_url: Optional[str] = None
    workflow_url: Optional[str] = None
    
    # Industry/Compliance
    industry_code: str = "SaaS"
    compliance_frameworks: List[str] = field(default_factory=list)


@dataclass
class NotificationDelivery:
    """Notification delivery record"""
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    notification_id: str = ""
    
    # Delivery details
    channel: NotificationChannel = NotificationChannel.EMAIL
    recipient: str = ""
    
    # Content
    subject: str = ""
    body: str = ""
    
    # Status
    status: str = "pending"  # pending, sent, failed, retrying
    attempts: int = 0
    max_attempts: int = 3
    
    # Timing
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ApprovalNotificationService:
    """
    Approval Notification Service - Task 14.1-T36
    
    Provides real-time notifications for approval workflows via multiple channels:
    - Email notifications for approval requests and status updates
    - Slack/Teams integration for instant messaging
    - Webhook notifications for system integrations
    - SLA breach alerts and escalation notifications
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'enable_notifications': True,
            'default_retry_attempts': 3,
            'sla_warning_threshold_hours': 2,
            'sla_critical_threshold_hours': 0.5,
            'batch_size': 50,
            'delivery_timeout_seconds': 30
        }
        
        # Notification templates
        self.templates: Dict[str, NotificationTemplate] = {}
        
        # Channel configurations
        self.channel_configs = {
            NotificationChannel.EMAIL: {
                'enabled': True,
                'smtp_server': 'smtp.office365.com',
                'smtp_port': 587,
                'use_tls': True
            },
            NotificationChannel.SLACK: {
                'enabled': True,
                'webhook_url': None,  # Set from environment
                'default_channel': '#approvals'
            },
            NotificationChannel.TEAMS: {
                'enabled': True,
                'webhook_url': None,  # Set from environment
                'card_template': 'adaptive_card'
            }
        }
        
        # Statistics
        self.notification_stats = {
            'total_notifications_sent': 0,
            'notifications_by_channel': {},
            'notifications_by_event': {},
            'delivery_success_rate': 0.0,
            'average_delivery_time_ms': 0.0,
            'sla_breach_alerts_sent': 0
        }
        
        # Initialize default templates
        self._initialize_default_templates()
    
    async def initialize(self) -> bool:
        """Initialize notification service"""
        try:
            await self._create_notification_tables()
            await self._load_notification_templates()
            self.logger.info("✅ Approval notification service initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize notification service: {e}")
            return False
    
    async def send_approval_notification(
        self,
        event_type: ApprovalEventType,
        context: NotificationContext,
        channels: List[NotificationChannel] = None,
        recipients: List[str] = None
    ) -> Dict[str, Any]:
        """
        Send approval notification for specified event
        
        This is the main entry point for approval notifications
        """
        
        if not self.config['enable_notifications']:
            return {'status': 'disabled', 'notifications_sent': 0}
        
        try:
            # Get applicable templates
            templates = self._get_applicable_templates(event_type, channels)
            
            if not templates:
                self.logger.warning(f"No templates found for event {event_type.value}")
                return {'status': 'no_templates', 'notifications_sent': 0}
            
            # Generate notifications
            notifications = []
            for template in templates:
                # Determine recipients
                template_recipients = recipients or await self._get_template_recipients(
                    template, context
                )
                
                for recipient in template_recipients:
                    notification = await self._create_notification(
                        template, context, recipient
                    )
                    if notification:
                        notifications.append(notification)
            
            # Send notifications
            delivery_results = []
            for notification in notifications:
                result = await self._send_notification(notification)
                delivery_results.append(result)
            
            # Update statistics
            self._update_notification_stats(event_type, delivery_results)
            
            # Store notification records
            if self.db_pool:
                await self._store_notification_records(notifications, delivery_results)
            
            successful_deliveries = len([r for r in delivery_results if r['status'] == 'sent'])
            
            self.logger.info(f"✅ Sent {successful_deliveries}/{len(notifications)} notifications for {event_type.value}")
            
            return {
                'status': 'success',
                'notifications_sent': successful_deliveries,
                'total_notifications': len(notifications),
                'delivery_results': delivery_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send approval notification: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def send_sla_breach_alert(
        self,
        approval_id: str,
        context: NotificationContext,
        severity: str = "warning"
    ) -> Dict[str, Any]:
        """Send SLA breach alert notification"""
        
        event_type = (
            ApprovalEventType.SLA_BREACH_CRITICAL 
            if severity == "critical" 
            else ApprovalEventType.SLA_BREACH_WARNING
        )
        
        # Add SLA context
        sla_context = NotificationContext(
            **context.__dict__,
            sla_breach_severity=severity,
            sla_hours_remaining=context.sla_hours_remaining or 0
        )
        
        return await self.send_approval_notification(
            event_type, sla_context, 
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
        )
    
    async def send_bulk_notifications(
        self,
        notifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send multiple notifications in batch"""
        
        try:
            results = []
            
            # Process in batches
            batch_size = self.config['batch_size']
            for i in range(0, len(notifications), batch_size):
                batch = notifications[i:i + batch_size]
                
                # Send batch concurrently
                batch_tasks = []
                for notification_data in batch:
                    task = self.send_approval_notification(
                        ApprovalEventType(notification_data['event_type']),
                        NotificationContext(**notification_data['context']),
                        notification_data.get('channels'),
                        notification_data.get('recipients')
                    )
                    batch_tasks.append(task)
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
            
            successful_batches = len([r for r in results if isinstance(r, dict) and r.get('status') == 'success'])
            
            return {
                'status': 'completed',
                'total_notifications': len(notifications),
                'successful_batches': successful_batches,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send bulk notifications: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _initialize_default_templates(self):
        """Initialize default notification templates"""
        
        # Approval request template
        approval_request_template = NotificationTemplate(
            template_id="approval_request_email",
            event_type=ApprovalEventType.APPROVAL_REQUESTED,
            channel=NotificationChannel.EMAIL,
            subject_template="🔔 Approval Required: {request_type} for {workflow_id}",
            body_template="""
            Dear {current_approver},
            
            A new approval request requires your attention:
            
            📋 Request Details:
            • Type: {request_type}
            • Workflow: {workflow_id}
            • Requested by: {requester_name} ({requester_email})
            • Risk Level: {risk_level}
            • Industry: {industry_code}
            
            📝 Justification:
            {business_justification}
            
            ⏰ SLA Information:
            • Created: {created_at}
            • Expires: {expires_at}
            • Time Remaining: {sla_hours_remaining} hours
            
            🔗 Actions:
            • Review and Approve: {approval_url}
            • View Workflow: {workflow_url}
            
            📊 Compliance Frameworks: {compliance_frameworks}
            
            Please review and take action before the SLA expires.
            
            Best regards,
            RevAI Pro Governance System
            """,
            target_roles=["approver", "compliance_officer"],
            priority=NotificationPriority.HIGH
        )
        
        # Slack approval request template
        slack_approval_template = NotificationTemplate(
            template_id="approval_request_slack",
            event_type=ApprovalEventType.APPROVAL_REQUESTED,
            channel=NotificationChannel.SLACK,
            subject_template="Approval Required",
            body_template="""
            🔔 *Approval Required*
            
            *Request:* {request_type}
            *Workflow:* {workflow_id}
            *Requester:* {requester_name}
            *Risk:* {risk_level}
            *SLA:* {sla_hours_remaining}h remaining
            
            *Justification:* {business_justification}
            
            <{approval_url}|Review & Approve> | <{workflow_url}|View Workflow>
            """,
            target_roles=["approver"],
            priority=NotificationPriority.HIGH
        )
        
        # SLA breach warning template
        sla_warning_template = NotificationTemplate(
            template_id="sla_breach_warning",
            event_type=ApprovalEventType.SLA_BREACH_WARNING,
            channel=NotificationChannel.EMAIL,
            subject_template="⚠️ SLA Warning: Approval {approval_id} expires soon",
            body_template="""
            ⚠️ SLA Breach Warning
            
            The following approval request is approaching its SLA deadline:
            
            • Approval ID: {approval_id}
            • Workflow: {workflow_id}
            • Time Remaining: {sla_hours_remaining} hours
            • Current Approver: {current_approver}
            
            Please take immediate action to avoid SLA breach.
            
            <{approval_url}|Take Action Now>
            """,
            target_roles=["approver", "manager"],
            priority=NotificationPriority.URGENT
        )
        
        # Store templates
        self.templates = {
            template.template_id: template for template in [
                approval_request_template, slack_approval_template, sla_warning_template
            ]
        }
    
    def _get_applicable_templates(
        self, event_type: ApprovalEventType, channels: List[NotificationChannel] = None
    ) -> List[NotificationTemplate]:
        """Get templates applicable for event and channels"""
        
        applicable = []
        
        for template in self.templates.values():
            if not template.is_active:
                continue
            
            if template.event_type != event_type:
                continue
            
            if channels and template.channel not in channels:
                continue
            
            applicable.append(template)
        
        return applicable
    
    async def _get_template_recipients(
        self, template: NotificationTemplate, context: NotificationContext
    ) -> List[str]:
        """Get recipients for template based on roles and context"""
        
        recipients = []
        
        # Add explicit recipients
        recipients.extend(template.target_users)
        
        # Add role-based recipients
        if template.target_roles:
            # In production, this would query user roles from database
            # For now, use context information
            if "approver" in template.target_roles and context.current_approver:
                recipients.append(context.current_approver)
            
            if "requester" in template.target_roles:
                recipients.append(context.requester_email)
        
        return list(set(recipients))  # Remove duplicates
    
    async def _create_notification(
        self, template: NotificationTemplate, context: NotificationContext, recipient: str
    ) -> Optional[NotificationDelivery]:
        """Create notification from template and context"""
        
        try:
            # Render subject and body
            subject = self._render_template(template.subject_template, context)
            body = self._render_template(template.body_template, context)
            
            # Create delivery record
            delivery = NotificationDelivery(
                notification_id=str(uuid.uuid4()),
                channel=template.channel,
                recipient=recipient,
                subject=subject,
                body=body,
                max_attempts=template.retry_attempts,
                scheduled_at=datetime.now(timezone.utc) + timedelta(minutes=template.delay_minutes),
                metadata={
                    'template_id': template.template_id,
                    'event_type': template.event_type.value,
                    'priority': template.priority.value,
                    'approval_id': context.approval_id,
                    'tenant_id': context.tenant_id
                }
            )
            
            return delivery
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create notification: {e}")
            return None
    
    def _render_template(self, template: str, context: NotificationContext) -> str:
        """Render template with context data"""
        
        try:
            # Convert context to dict for formatting
            context_dict = {
                'approval_id': context.approval_id,
                'workflow_id': context.workflow_id,
                'tenant_id': context.tenant_id,
                'request_type': context.request_type,
                'requester_name': context.requester_name,
                'requester_email': context.requester_email,
                'approval_reason': context.approval_reason,
                'business_justification': context.business_justification,
                'risk_level': context.risk_level,
                'created_at': context.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'expires_at': context.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC') if context.expires_at else 'N/A',
                'sla_hours_remaining': f"{context.sla_hours_remaining:.1f}" if context.sla_hours_remaining else 'N/A',
                'current_approver': context.current_approver or 'Pending Assignment',
                'approval_chain': ', '.join(context.approval_chain),
                'approval_url': context.approval_url or '#',
                'workflow_url': context.workflow_url or '#',
                'industry_code': context.industry_code,
                'compliance_frameworks': ', '.join(context.compliance_frameworks) or 'None'
            }
            
            return template.format(**context_dict)
            
        except Exception as e:
            self.logger.error(f"❌ Template rendering error: {e}")
            return template  # Return unrendered template as fallback
    
    async def _send_notification(self, delivery: NotificationDelivery) -> Dict[str, Any]:
        """Send individual notification"""
        
        try:
            delivery.attempts += 1
            
            if delivery.channel == NotificationChannel.EMAIL:
                result = await self._send_email(delivery)
            elif delivery.channel == NotificationChannel.SLACK:
                result = await self._send_slack(delivery)
            elif delivery.channel == NotificationChannel.TEAMS:
                result = await self._send_teams(delivery)
            else:
                result = {'status': 'unsupported_channel'}
            
            if result.get('status') == 'sent':
                delivery.status = 'sent'
                delivery.sent_at = datetime.now(timezone.utc)
            else:
                delivery.status = 'failed'
                delivery.error_message = result.get('error', 'Unknown error')
                
                # Schedule retry if attempts remaining
                if delivery.attempts < delivery.max_attempts:
                    delivery.status = 'retrying'
                    delivery.retry_at = datetime.now(timezone.utc) + timedelta(minutes=2 ** delivery.attempts)
            
            return {
                'delivery_id': delivery.delivery_id,
                'status': delivery.status,
                'channel': delivery.channel.value,
                'recipient': delivery.recipient,
                'attempts': delivery.attempts,
                'sent_at': delivery.sent_at.isoformat() if delivery.sent_at else None,
                'error': delivery.error_message
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send notification: {e}")
            return {
                'delivery_id': delivery.delivery_id,
                'status': 'error',
                'error': str(e)
            }
    
    async def _send_email(self, delivery: NotificationDelivery) -> Dict[str, Any]:
        """Send email notification (placeholder implementation)"""
        
        # In production, this would use SMTP or email service
        self.logger.info(f"📧 Email notification sent to {delivery.recipient}: {delivery.subject}")
        
        return {
            'status': 'sent',
            'delivery_time_ms': 150,
            'message_id': f"email_{delivery.delivery_id}"
        }
    
    async def _send_slack(self, delivery: NotificationDelivery) -> Dict[str, Any]:
        """Send Slack notification (placeholder implementation)"""
        
        # In production, this would use Slack webhook or API
        self.logger.info(f"💬 Slack notification sent: {delivery.subject}")
        
        return {
            'status': 'sent',
            'delivery_time_ms': 200,
            'message_id': f"slack_{delivery.delivery_id}"
        }
    
    async def _send_teams(self, delivery: NotificationDelivery) -> Dict[str, Any]:
        """Send Teams notification (placeholder implementation)"""
        
        # In production, this would use Teams webhook or API
        self.logger.info(f"👥 Teams notification sent: {delivery.subject}")
        
        return {
            'status': 'sent',
            'delivery_time_ms': 180,
            'message_id': f"teams_{delivery.delivery_id}"
        }
    
    def _update_notification_stats(
        self, event_type: ApprovalEventType, delivery_results: List[Dict[str, Any]]
    ):
        """Update notification statistics"""
        
        successful_deliveries = len([r for r in delivery_results if r['status'] == 'sent'])
        
        self.notification_stats['total_notifications_sent'] += successful_deliveries
        
        # Update by event type
        if event_type.value not in self.notification_stats['notifications_by_event']:
            self.notification_stats['notifications_by_event'][event_type.value] = 0
        self.notification_stats['notifications_by_event'][event_type.value] += successful_deliveries
        
        # Update by channel
        for result in delivery_results:
            if result['status'] == 'sent':
                channel = result['channel']
                if channel not in self.notification_stats['notifications_by_channel']:
                    self.notification_stats['notifications_by_channel'][channel] = 0
                self.notification_stats['notifications_by_channel'][channel] += 1
        
        # Update success rate
        total_attempts = len(delivery_results)
        if total_attempts > 0:
            current_total = self.notification_stats.get('total_attempts', 0)
            current_successful = self.notification_stats.get('total_successful', 0)
            
            new_total = current_total + total_attempts
            new_successful = current_successful + successful_deliveries
            
            self.notification_stats['delivery_success_rate'] = (new_successful / new_total) * 100
            self.notification_stats['total_attempts'] = new_total
            self.notification_stats['total_successful'] = new_successful
    
    async def _create_notification_tables(self):
        """Create notification tracking tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- Notification delivery tracking
        CREATE TABLE IF NOT EXISTS approval_notifications (
            notification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            approval_id UUID NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Notification details
            event_type VARCHAR(50) NOT NULL,
            channel VARCHAR(20) NOT NULL,
            recipient VARCHAR(255) NOT NULL,
            
            -- Content
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            
            -- Delivery status
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            
            -- Timing
            scheduled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            sent_at TIMESTAMPTZ,
            
            -- Error handling
            error_message TEXT,
            retry_at TIMESTAMPTZ,
            
            -- Metadata
            metadata JSONB DEFAULT '{}',
            
            -- Timestamps
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT chk_notification_status CHECK (status IN ('pending', 'sent', 'failed', 'retrying')),
            CONSTRAINT chk_notification_channel CHECK (channel IN ('email', 'slack', 'teams', 'webhook', 'sms', 'in_app'))
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_approval_notifications_approval ON approval_notifications(approval_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_approval_notifications_tenant ON approval_notifications(tenant_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_approval_notifications_status ON approval_notifications(status, scheduled_at);
        CREATE INDEX IF NOT EXISTS idx_approval_notifications_retry ON approval_notifications(retry_at) WHERE status = 'retrying';
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ Notification tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create notification tables: {e}")
            raise
    
    async def _load_notification_templates(self):
        """Load custom notification templates from database"""
        # In production, this would load custom templates from database
        # For now, we use the default templates initialized in memory
        pass
    
    async def _store_notification_records(
        self, notifications: List[NotificationDelivery], results: List[Dict[str, Any]]
    ):
        """Store notification records in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                for notification, result in zip(notifications, results):
                    await conn.execute("""
                        INSERT INTO approval_notifications (
                            notification_id, approval_id, tenant_id, event_type,
                            channel, recipient, subject, body, status, attempts,
                            max_attempts, scheduled_at, sent_at, error_message,
                            retry_at, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """,
                    notification.notification_id,
                    notification.metadata.get('approval_id'),
                    notification.metadata.get('tenant_id'),
                    notification.metadata.get('event_type'),
                    notification.channel.value,
                    notification.recipient,
                    notification.subject,
                    notification.body,
                    notification.status,
                    notification.attempts,
                    notification.max_attempts,
                    notification.scheduled_at,
                    notification.sent_at,
                    notification.error_message,
                    notification.retry_at,
                    json.dumps(notification.metadata))
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to store notification records: {e}")
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification service statistics"""
        
        return {
            'total_notifications_sent': self.notification_stats['total_notifications_sent'],
            'delivery_success_rate': round(self.notification_stats.get('delivery_success_rate', 0), 2),
            'notifications_by_channel': self.notification_stats['notifications_by_channel'],
            'notifications_by_event': self.notification_stats['notifications_by_event'],
            'sla_breach_alerts_sent': self.notification_stats['sla_breach_alerts_sent'],
            'active_templates': len([t for t in self.templates.values() if t.is_active]),
            'supported_channels': [c.value for c in NotificationChannel],
            'supported_events': [e.value for e in ApprovalEventType],
            'notifications_enabled': self.config['enable_notifications']
        }


# Global notification service instance
approval_notification_service = ApprovalNotificationService()
