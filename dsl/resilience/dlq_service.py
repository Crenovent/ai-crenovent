"""
Task 9.5.4: Configure DLQs per plane - Error handling
Dead Letter Queue service for handling failed messages across all architectural planes.

Features:
- Plane-specific DLQ configuration and management
- Automatic retry with exponential backoff
- Message quarantine and approval workflows
- Multi-tenant message isolation
- DYNAMIC configuration (no hardcoding)
- Comprehensive monitoring and alerting
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib

logger = logging.getLogger(__name__)

class PlaneType(Enum):
    """Architectural planes"""
    CONTROL = "control"
    EXECUTION = "execution"
    DATA = "data"
    GOVERNANCE = "governance"
    UX = "ux"

class MessageStatus(Enum):
    """DLQ message status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    RETRY_SCHEDULED = "retry_scheduled"
    QUARANTINED = "quarantined"
    APPROVED_FOR_RETRY = "approved_for_retry"
    PERMANENTLY_FAILED = "permanently_failed"
    RESOLVED = "resolved"

class RetryStrategy(Enum):
    """Retry strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    MANUAL_ONLY = "manual_only"

@dataclass
class DLQMessage:
    """Dead letter queue message"""
    message_id: str
    tenant_id: int
    plane: PlaneType
    service_name: str
    operation: str
    
    # Message content
    original_payload: Dict[str, Any]
    error_details: Dict[str, Any]
    
    # Retry information
    retry_count: int = 0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    next_retry_at: Optional[datetime] = None
    
    # Status and timestamps
    status: MessageStatus = MessageStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_retry_at: Optional[datetime] = None
    quarantined_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Governance and approval
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approval_reason: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "tenant_id": self.tenant_id,
            "plane": self.plane.value,
            "service_name": self.service_name,
            "operation": self.operation,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_strategy": self.retry_strategy.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "quarantined_at": self.quarantined_at.isoformat() if self.quarantined_at else None,
            "requires_approval": self.requires_approval,
            "approved_by": self.approved_by,
            "error_summary": self.error_details.get("error_message", "Unknown error"),
            "tags": self.tags,
            "metadata": self.metadata
        }

@dataclass
class PlaneConfig:
    """Configuration for plane-specific DLQ"""
    plane: PlaneType
    
    # DYNAMIC configuration - no hardcoding
    max_retries: int = field(default_factory=lambda: int(os.getenv("DLQ_MAX_RETRIES", "3")))
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay_seconds: int = field(default_factory=lambda: int(os.getenv("DLQ_BASE_DELAY", "60")))
    max_delay_seconds: int = field(default_factory=lambda: int(os.getenv("DLQ_MAX_DELAY", "3600")))
    
    # Quarantine thresholds
    quarantine_after_retries: int = field(default_factory=lambda: int(os.getenv("DLQ_QUARANTINE_THRESHOLD", "5")))
    auto_quarantine_enabled: bool = field(default_factory=lambda: os.getenv("DLQ_AUTO_QUARANTINE", "true").lower() == "true")
    
    # Approval requirements
    approval_required_for_critical: bool = True
    approval_required_after_quarantine: bool = True
    
    # Retention and cleanup
    retention_days: int = field(default_factory=lambda: int(os.getenv("DLQ_RETENTION_DAYS", "30")))
    cleanup_enabled: bool = field(default_factory=lambda: os.getenv("DLQ_CLEANUP_ENABLED", "true").lower() == "true")

class DLQService:
    """
    Task 9.5.4: Dead Letter Queue Service - Configure DLQs per plane
    Comprehensive error handling and message recovery across all planes
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage (would be database/message queue in production)
        self.messages: Dict[str, DLQMessage] = {}
        self.plane_configs: Dict[PlaneType, PlaneConfig] = {}
        
        # Initialize plane configurations
        for plane in PlaneType:
            self.plane_configs[plane] = PlaneConfig(plane=plane)
        
        # Dynamic service configuration
        self.service_config = {
            "processing_interval_seconds": int(os.getenv("DLQ_PROCESSING_INTERVAL", "30")),
            "batch_size": int(os.getenv("DLQ_BATCH_SIZE", "10")),
            "monitoring_enabled": os.getenv("DLQ_MONITORING", "true").lower() == "true",
            "notifications_enabled": os.getenv("DLQ_NOTIFICATIONS", "true").lower() == "true",
            "auto_retry_enabled": os.getenv("DLQ_AUTO_RETRY", "true").lower() == "true"
        }
        
        # Metrics and monitoring
        self.metrics = {
            "messages_queued": 0,
            "messages_retried": 0,
            "messages_quarantined": 0,
            "messages_resolved": 0,
            "messages_permanently_failed": 0,
            "processing_runs": 0,
            "processing_errors": 0
        }
        
        # Plane-specific metrics
        self.plane_metrics = {
            plane: {
                "queued": 0,
                "processing": 0,
                "quarantined": 0,
                "resolved": 0,
                "failed": 0
            } for plane in PlaneType
        }
        
        logger.info("💀 DLQ Service initialized for all planes")
    
    async def queue_message(self, message: DLQMessage) -> bool:
        """Queue a failed message for retry processing"""
        try:
            # Validate message
            if not message.message_id or not message.tenant_id:
                logger.error(f"Invalid message: missing required fields")
                return False
            
            # Get plane configuration
            plane_config = self.plane_configs.get(message.plane)
            if not plane_config:
                logger.error(f"No configuration found for plane {message.plane}")
                return False
            
            # Apply plane-specific configuration
            message.max_retries = plane_config.max_retries
            message.retry_strategy = plane_config.retry_strategy
            
            # Calculate initial retry time
            if message.retry_count == 0:
                message.next_retry_at = self._calculate_next_retry_time(message, plane_config)
            
            # Store message
            self.messages[message.message_id] = message
            
            # Update metrics
            self.metrics["messages_queued"] += 1
            self.plane_metrics[message.plane]["queued"] += 1
            
            logger.info(f"📥 Queued message {message.message_id} for {message.plane.value} plane")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to queue message {message.message_id}: {e}")
            return False
    
    def _calculate_next_retry_time(self, message: DLQMessage, config: PlaneConfig) -> datetime:
        """Calculate next retry time based on strategy - DYNAMIC calculation"""
        base_delay = config.base_delay_seconds
        max_delay = config.max_delay_seconds
        
        if config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            # Exponential backoff: base_delay * (2 ^ retry_count)
            delay = min(base_delay * (2 ** message.retry_count), max_delay)
        elif config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            # Linear backoff: base_delay * (retry_count + 1)
            delay = min(base_delay * (message.retry_count + 1), max_delay)
        elif config.retry_strategy == RetryStrategy.FIXED_DELAY:
            # Fixed delay
            delay = base_delay
        elif config.retry_strategy == RetryStrategy.IMMEDIATE:
            # Immediate retry
            delay = 0
        else:
            # Manual only - set far future
            delay = max_delay * 24  # 24 hours if max_delay is 1 hour
        
        return datetime.utcnow() + timedelta(seconds=delay)
    
    async def process_dlq_messages(self):
        """Process DLQ messages for retry"""
        try:
            current_time = datetime.utcnow()
            processed_count = 0
            batch_size = self.service_config["batch_size"]
            
            # Get messages ready for retry
            ready_messages = [
                msg for msg in self.messages.values()
                if (msg.status == MessageStatus.QUEUED and 
                    msg.next_retry_at and 
                    msg.next_retry_at <= current_time and
                    msg.retry_count < msg.max_retries)
            ]
            
            # Process in batches
            for i in range(0, len(ready_messages), batch_size):
                batch = ready_messages[i:i + batch_size]
                
                for message in batch:
                    success = await self._retry_message(message)
                    processed_count += 1
                    
                    if processed_count >= batch_size:
                        break
            
            self.metrics["processing_runs"] += 1
            
            if processed_count > 0:
                logger.info(f"🔄 Processed {processed_count} DLQ messages")
            
        except Exception as e:
            logger.error(f"❌ DLQ processing error: {e}")
            self.metrics["processing_errors"] += 1
    
    async def _retry_message(self, message: DLQMessage) -> bool:
        """Retry a specific message"""
        try:
            message.status = MessageStatus.PROCESSING
            message.last_retry_at = datetime.utcnow()
            message.retry_count += 1
            
            # Simulate message processing (would integrate with actual services)
            success = await self._simulate_message_processing(message)
            
            if success:
                # Message processed successfully
                message.status = MessageStatus.RESOLVED
                message.resolved_at = datetime.utcnow()
                
                self.metrics["messages_resolved"] += 1
                self.plane_metrics[message.plane]["resolved"] += 1
                
                logger.info(f"✅ Message {message.message_id} resolved after {message.retry_count} retries")
                
            else:
                # Message failed again
                plane_config = self.plane_configs[message.plane]
                
                if message.retry_count >= message.max_retries:
                    # Exceeded max retries - quarantine or fail permanently
                    if (plane_config.auto_quarantine_enabled and 
                        message.retry_count >= plane_config.quarantine_after_retries):
                        
                        message.status = MessageStatus.QUARANTINED
                        message.quarantined_at = datetime.utcnow()
                        message.requires_approval = plane_config.approval_required_after_quarantine
                        
                        self.metrics["messages_quarantined"] += 1
                        self.plane_metrics[message.plane]["quarantined"] += 1
                        
                        logger.warning(f"⚠️ Message {message.message_id} quarantined after {message.retry_count} retries")
                        
                    else:
                        message.status = MessageStatus.PERMANENTLY_FAILED
                        
                        self.metrics["messages_permanently_failed"] += 1
                        self.plane_metrics[message.plane]["failed"] += 1
                        
                        logger.error(f"💀 Message {message.message_id} permanently failed")
                
                else:
                    # Schedule next retry
                    message.status = MessageStatus.RETRY_SCHEDULED
                    message.next_retry_at = self._calculate_next_retry_time(message, plane_config)
                    
                    logger.info(f"🔄 Message {message.message_id} scheduled for retry {message.retry_count + 1}")
            
            self.metrics["messages_retried"] += 1
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to retry message {message.message_id}: {e}")
            message.status = MessageStatus.QUEUED  # Reset to queued for next attempt
            return False
    
    async def _simulate_message_processing(self, message: DLQMessage) -> bool:
        """Simulate message processing (placeholder for actual service integration)"""
        # In real implementation, this would call the actual service
        # For now, simulate success/failure based on retry count
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Simulate increasing success probability with retries
        success_probability = min(0.3 + (message.retry_count * 0.2), 0.9)
        
        import random
        return random.random() < success_probability
    
    async def approve_quarantined_message(self, message_id: str, approved_by: str, 
                                        reason: str = "") -> bool:
        """Approve a quarantined message for retry"""
        try:
            message = self.messages.get(message_id)
            if not message:
                logger.error(f"Message {message_id} not found")
                return False
            
            if message.status != MessageStatus.QUARANTINED:
                logger.error(f"Message {message_id} is not quarantined (status: {message.status})")
                return False
            
            # Approve message
            message.status = MessageStatus.APPROVED_FOR_RETRY
            message.approved_by = approved_by
            message.approval_reason = reason
            message.retry_count = 0  # Reset retry count
            
            # Calculate next retry time
            plane_config = self.plane_configs[message.plane]
            message.next_retry_at = self._calculate_next_retry_time(message, plane_config)
            
            logger.info(f"✅ Message {message_id} approved for retry by {approved_by}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to approve message {message_id}: {e}")
            return False
    
    async def start_dlq_processing(self):
        """Start background DLQ processing"""
        async def dlq_processor():
            while True:
                try:
                    await asyncio.sleep(self.service_config["processing_interval_seconds"])
                    
                    if self.service_config["auto_retry_enabled"]:
                        await self.process_dlq_messages()
                    
                except Exception as e:
                    logger.error(f"DLQ processor error: {e}")
        
        # Start processing task
        asyncio.create_task(dlq_processor())
        logger.info("🔄 Started DLQ processing for all planes")
    
    def get_dlq_status(self) -> Dict[str, Any]:
        """Get comprehensive DLQ service status"""
        # Calculate current statistics
        current_stats = {
            "total_messages": len(self.messages),
            "by_status": {},
            "by_plane": {}
        }
        
        # Count by status
        for message in self.messages.values():
            status = message.status.value
            current_stats["by_status"][status] = current_stats["by_status"].get(status, 0) + 1
            
            plane = message.plane.value
            if plane not in current_stats["by_plane"]:
                current_stats["by_plane"][plane] = {"total": 0, "by_status": {}}
            
            current_stats["by_plane"][plane]["total"] += 1
            current_stats["by_plane"][plane]["by_status"][status] = (
                current_stats["by_plane"][plane]["by_status"].get(status, 0) + 1
            )
        
        return {
            "service_status": "active",
            "configuration": self.service_config.copy(),
            "metrics": self.metrics.copy(),
            "plane_metrics": {plane.value: metrics for plane, metrics in self.plane_metrics.items()},
            "current_statistics": current_stats,
            "plane_configurations": {
                plane.value: {
                    "max_retries": config.max_retries,
                    "retry_strategy": config.retry_strategy.value,
                    "base_delay_seconds": config.base_delay_seconds,
                    "max_delay_seconds": config.max_delay_seconds,
                    "quarantine_after_retries": config.quarantine_after_retries,
                    "auto_quarantine_enabled": config.auto_quarantine_enabled,
                    "retention_days": config.retention_days
                } for plane, config in self.plane_configs.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_messages_by_plane(self, plane: PlaneType, status: Optional[MessageStatus] = None) -> List[Dict[str, Any]]:
        """Get messages for a specific plane"""
        messages = [
            msg for msg in self.messages.values()
            if msg.plane == plane and (status is None or msg.status == status)
        ]
        
        # Sort by created_at descending
        messages.sort(key=lambda m: m.created_at, reverse=True)
        
        return [msg.to_dict() for msg in messages]
    
    async def cleanup_old_messages(self):
        """Clean up old resolved/failed messages based on retention policy"""
        try:
            current_time = datetime.utcnow()
            cleanup_count = 0
            
            for plane, config in self.plane_configs.items():
                if not config.cleanup_enabled:
                    continue
                
                retention_cutoff = current_time - timedelta(days=config.retention_days)
                
                # Find old messages to cleanup
                messages_to_cleanup = [
                    msg_id for msg_id, msg in self.messages.items()
                    if (msg.plane == plane and 
                        msg.status in [MessageStatus.RESOLVED, MessageStatus.PERMANENTLY_FAILED] and
                        msg.created_at < retention_cutoff)
                ]
                
                # Remove old messages
                for msg_id in messages_to_cleanup:
                    del self.messages[msg_id]
                    cleanup_count += 1
            
            if cleanup_count > 0:
                logger.info(f"🧹 Cleaned up {cleanup_count} old DLQ messages")
            
        except Exception as e:
            logger.error(f"❌ DLQ cleanup error: {e}")

# Global DLQ service instance
dlq_service = DLQService()
