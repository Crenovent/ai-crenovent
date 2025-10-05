"""
RBA Webhook Receiver - Task 10.1-T11
Trusted ingress for vendor webhooks with signature verification and replay protection.
Normalizes external events to internal event bus with governance-first design.
"""

import asyncio
import json
import hmac
import hashlib
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import base64

logger = logging.getLogger(__name__)

class WebhookStatus(Enum):
    """Webhook processing status"""
    RECEIVED = "received"
    VERIFIED = "verified"
    PROCESSED = "processed"
    FAILED = "failed"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"

class WebhookSource(Enum):
    """Supported webhook sources"""
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    STRIPE = "stripe"
    ZUORA = "zuora"
    DOCUSIGN = "docusign"
    NETSUITE = "netsuite"
    CUSTOM = "custom"

@dataclass
class WebhookEvent:
    """Webhook event data structure"""
    event_id: str
    source: WebhookSource
    event_type: str
    tenant_id: str
    
    # Raw webhook data
    headers: Dict[str, str]
    payload: Dict[str, Any]
    signature: Optional[str] = None
    
    # Processing metadata
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    status: WebhookStatus = WebhookStatus.RECEIVED
    
    # Verification results
    signature_valid: bool = False
    replay_check_passed: bool = False
    
    # Normalization results
    normalized_event: Optional[Dict[str, Any]] = None
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "event_id": self.event_id,
            "source": self.source.value,
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "received_at": self.received_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "status": self.status.value,
            "signature_valid": self.signature_valid,
            "replay_check_passed": self.replay_check_passed,
            "error_message": self.error_message,
            "retry_count": self.retry_count
        }

@dataclass
class WebhookConfig:
    """Webhook configuration for a source"""
    source: WebhookSource
    tenant_id: str
    
    # Security configuration
    secret_key: str
    signature_header: str = "X-Signature"
    timestamp_header: str = "X-Timestamp"
    
    # Replay protection
    replay_window_seconds: int = 300  # 5 minutes
    
    # Processing configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60
    
    # Event filtering
    allowed_event_types: List[str] = field(default_factory=list)
    blocked_event_types: List[str] = field(default_factory=list)

class WebhookReceiver:
    """
    Trusted webhook receiver with signature verification and replay protection
    Provides secure ingress for external system events
    """
    
    def __init__(self, event_bus=None, evidence_service=None):
        self.logger = logging.getLogger(__name__)
        
        # Core services
        self.event_bus = event_bus
        self.evidence_service = evidence_service
        
        # Configuration storage
        self.webhook_configs: Dict[str, WebhookConfig] = {}
        
        # Replay protection - store processed event signatures
        self.processed_signatures: Dict[str, datetime] = {}
        
        # Delivery attempt tracking
        self.delivery_attempts: Dict[str, List[Dict[str, Any]]] = {}
        
        # Metrics
        self.metrics = {
            "webhooks_received": 0,
            "webhooks_verified": 0,
            "webhooks_processed": 0,
            "webhooks_rejected": 0,
            "signature_failures": 0,
            "replay_attacks_blocked": 0,
            "processing_errors": 0
        }
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        self.logger.info("🎣 Webhook Receiver initialized with signature verification and replay protection")
    
    def _initialize_default_configs(self):
        """Initialize default webhook configurations for common sources"""
        
        # Salesforce webhook configuration
        self.webhook_configs["salesforce_default"] = WebhookConfig(
            source=WebhookSource.SALESFORCE,
            tenant_id="default",
            secret_key="salesforce_webhook_secret",
            signature_header="X-Salesforce-Signature",
            timestamp_header="X-Salesforce-Timestamp",
            allowed_event_types=["opportunity_updated", "account_created", "contact_updated"]
        )
        
        # Stripe webhook configuration
        self.webhook_configs["stripe_default"] = WebhookConfig(
            source=WebhookSource.STRIPE,
            tenant_id="default",
            secret_key="stripe_webhook_secret",
            signature_header="Stripe-Signature",
            timestamp_header="X-Stripe-Timestamp",
            allowed_event_types=["invoice.payment_succeeded", "subscription.updated", "customer.created"]
        )
        
        # HubSpot webhook configuration
        self.webhook_configs["hubspot_default"] = WebhookConfig(
            source=WebhookSource.HUBSPOT,
            tenant_id="default",
            secret_key="hubspot_webhook_secret",
            signature_header="X-HubSpot-Signature",
            timestamp_header="X-HubSpot-Timestamp",
            allowed_event_types=["contact.propertyChange", "deal.propertyChange", "company.creation"]
        )
    
    async def receive_webhook(self, source: str, tenant_id: str, headers: Dict[str, str], 
                            payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for webhook reception
        Handles verification, replay protection, and event normalization
        """
        event_id = str(uuid.uuid4())
        self.metrics["webhooks_received"] += 1
        
        try:
            # Create webhook event
            webhook_event = WebhookEvent(
                event_id=event_id,
                source=WebhookSource(source.lower()),
                event_type=payload.get("type", "unknown"),
                tenant_id=tenant_id,
                headers=headers,
                payload=payload,
                signature=headers.get("X-Signature") or headers.get("Stripe-Signature") or headers.get("X-Salesforce-Signature")
            )
            
            self.logger.info(f"🎣 Received webhook: {source}/{webhook_event.event_type} for tenant {tenant_id}")
            
            # Step 1: Get webhook configuration
            config = self._get_webhook_config(source, tenant_id)
            if not config:
                webhook_event.status = WebhookStatus.REJECTED
                webhook_event.error_message = f"No configuration found for {source}/{tenant_id}"
                await self._log_webhook_event(webhook_event)
                
                return {
                    "success": False,
                    "error": "Configuration not found",
                    "event_id": event_id
                }
            
            # Step 2: Signature verification
            if not await self._verify_signature(webhook_event, config):
                webhook_event.status = WebhookStatus.REJECTED
                webhook_event.error_message = "Signature verification failed"
                self.metrics["signature_failures"] += 1
                await self._log_webhook_event(webhook_event)
                
                return {
                    "success": False,
                    "error": "Signature verification failed",
                    "event_id": event_id
                }
            
            webhook_event.signature_valid = True
            webhook_event.status = WebhookStatus.VERIFIED
            self.metrics["webhooks_verified"] += 1
            
            # Step 3: Replay protection
            if not await self._check_replay_protection(webhook_event, config):
                webhook_event.status = WebhookStatus.DUPLICATE
                webhook_event.error_message = "Duplicate/replay event detected"
                self.metrics["replay_attacks_blocked"] += 1
                await self._log_webhook_event(webhook_event)
                
                return {
                    "success": False,
                    "error": "Duplicate event",
                    "event_id": event_id
                }
            
            webhook_event.replay_check_passed = True
            
            # Step 4: Event type filtering
            if not self._is_event_allowed(webhook_event, config):
                webhook_event.status = WebhookStatus.REJECTED
                webhook_event.error_message = f"Event type {webhook_event.event_type} not allowed"
                await self._log_webhook_event(webhook_event)
                
                return {
                    "success": False,
                    "error": "Event type not allowed",
                    "event_id": event_id
                }
            
            # Step 5: Event normalization
            normalized_event = await self._normalize_event(webhook_event, config)
            webhook_event.normalized_event = normalized_event
            
            # Step 6: Evidence capture
            await self._capture_webhook_evidence(webhook_event)
            
            # Step 7: Publish to event bus
            if self.event_bus:
                await self.event_bus.publish(normalized_event)
            
            # Step 8: Record delivery attempt
            await self._record_delivery_attempt(webhook_event, True)
            
            webhook_event.status = WebhookStatus.PROCESSED
            webhook_event.processed_at = datetime.now(timezone.utc)
            self.metrics["webhooks_processed"] += 1
            
            await self._log_webhook_event(webhook_event)
            
            self.logger.info(f"✅ Webhook processed successfully: {event_id}")
            
            return {
                "success": True,
                "event_id": event_id,
                "normalized_event": normalized_event,
                "status": webhook_event.status.value
            }
            
        except Exception as e:
            self.metrics["processing_errors"] += 1
            self.logger.error(f"❌ Webhook processing error: {e}")
            
            # Record failed delivery attempt
            if 'webhook_event' in locals():
                webhook_event.status = WebhookStatus.FAILED
                webhook_event.error_message = str(e)
                await self._record_delivery_attempt(webhook_event, False)
                await self._log_webhook_event(webhook_event)
            
            return {
                "success": False,
                "error": str(e),
                "event_id": event_id
            }
    
    def _get_webhook_config(self, source: str, tenant_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration for source/tenant combination"""
        # Try tenant-specific config first
        config_key = f"{source}_{tenant_id}"
        if config_key in self.webhook_configs:
            return self.webhook_configs[config_key]
        
        # Fall back to default config
        default_key = f"{source}_default"
        if default_key in self.webhook_configs:
            config = self.webhook_configs[default_key]
            # Create tenant-specific copy
            tenant_config = WebhookConfig(
                source=config.source,
                tenant_id=tenant_id,
                secret_key=config.secret_key,
                signature_header=config.signature_header,
                timestamp_header=config.timestamp_header,
                replay_window_seconds=config.replay_window_seconds,
                max_retries=config.max_retries,
                retry_delay_seconds=config.retry_delay_seconds,
                allowed_event_types=config.allowed_event_types.copy(),
                blocked_event_types=config.blocked_event_types.copy()
            )
            return tenant_config
        
        return None
    
    async def _verify_signature(self, webhook_event: WebhookEvent, config: WebhookConfig) -> bool:
        """Verify webhook signature using HMAC"""
        try:
            if not webhook_event.signature:
                self.logger.warning(f"⚠️ No signature provided for webhook {webhook_event.event_id}")
                return False
            
            # Get the raw payload for signature verification
            payload_string = json.dumps(webhook_event.payload, separators=(',', ':'), sort_keys=True)
            
            # Handle different signature formats
            if webhook_event.source == WebhookSource.STRIPE:
                return self._verify_stripe_signature(webhook_event.signature, payload_string, config.secret_key)
            elif webhook_event.source == WebhookSource.SALESFORCE:
                return self._verify_salesforce_signature(webhook_event.signature, payload_string, config.secret_key)
            elif webhook_event.source == WebhookSource.HUBSPOT:
                return self._verify_hubspot_signature(webhook_event.signature, payload_string, config.secret_key)
            else:
                # Generic HMAC-SHA256 verification
                return self._verify_generic_signature(webhook_event.signature, payload_string, config.secret_key)
                
        except Exception as e:
            self.logger.error(f"❌ Signature verification error: {e}")
            return False
    
    def _verify_stripe_signature(self, signature: str, payload: str, secret: str) -> bool:
        """Verify Stripe webhook signature"""
        try:
            # Stripe signature format: t=timestamp,v1=signature
            sig_parts = {}
            for part in signature.split(','):
                key, value = part.split('=', 1)
                sig_parts[key] = value
            
            timestamp = sig_parts.get('t')
            signature_hash = sig_parts.get('v1')
            
            if not timestamp or not signature_hash:
                return False
            
            # Create expected signature
            signed_payload = f"{timestamp}.{payload}"
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                signed_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature_hash, expected_signature)
            
        except Exception as e:
            self.logger.error(f"❌ Stripe signature verification error: {e}")
            return False
    
    def _verify_salesforce_signature(self, signature: str, payload: str, secret: str) -> bool:
        """Verify Salesforce webhook signature"""
        try:
            # Salesforce uses base64-encoded HMAC-SHA256
            expected_signature = base64.b64encode(
                hmac.new(
                    secret.encode('utf-8'),
                    payload.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"❌ Salesforce signature verification error: {e}")
            return False
    
    def _verify_hubspot_signature(self, signature: str, payload: str, secret: str) -> bool:
        """Verify HubSpot webhook signature"""
        try:
            # HubSpot uses SHA256 hash
            expected_signature = hashlib.sha256(
                (secret + payload).encode('utf-8')
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"❌ HubSpot signature verification error: {e}")
            return False
    
    def _verify_generic_signature(self, signature: str, payload: str, secret: str) -> bool:
        """Verify generic HMAC-SHA256 signature"""
        try:
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"❌ Generic signature verification error: {e}")
            return False
    
    async def _check_replay_protection(self, webhook_event: WebhookEvent, config: WebhookConfig) -> bool:
        """Check for replay attacks using event signatures and timestamps"""
        try:
            # Create unique signature for this event
            event_signature = self._create_event_signature(webhook_event)
            
            # Check if we've seen this exact event before
            if event_signature in self.processed_signatures:
                last_seen = self.processed_signatures[event_signature]
                self.logger.warning(f"🔄 Duplicate event detected: {webhook_event.event_id} (last seen: {last_seen})")
                return False
            
            # Check timestamp if available
            timestamp_header = webhook_event.headers.get(config.timestamp_header)
            if timestamp_header:
                try:
                    event_timestamp = datetime.fromtimestamp(int(timestamp_header), tz=timezone.utc)
                    now = datetime.now(timezone.utc)
                    
                    # Check if event is too old
                    if (now - event_timestamp).total_seconds() > config.replay_window_seconds:
                        self.logger.warning(f"⏰ Event too old: {webhook_event.event_id} ({event_timestamp})")
                        return False
                    
                    # Check if event is from the future (clock skew tolerance: 5 minutes)
                    if (event_timestamp - now).total_seconds() > 300:
                        self.logger.warning(f"🔮 Event from future: {webhook_event.event_id} ({event_timestamp})")
                        return False
                        
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"⚠️ Invalid timestamp in webhook: {timestamp_header}")
            
            # Record this event signature
            self.processed_signatures[event_signature] = datetime.now(timezone.utc)
            
            # Clean up old signatures (keep only within replay window)
            self._cleanup_old_signatures(config.replay_window_seconds)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Replay protection error: {e}")
            return False
    
    def _create_event_signature(self, webhook_event: WebhookEvent) -> str:
        """Create unique signature for event deduplication"""
        # Create signature from key event properties
        signature_data = {
            "source": webhook_event.source.value,
            "event_type": webhook_event.event_type,
            "tenant_id": webhook_event.tenant_id,
            "payload_hash": hashlib.sha256(
                json.dumps(webhook_event.payload, sort_keys=True).encode('utf-8')
            ).hexdigest()
        }
        
        signature_string = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_string.encode('utf-8')).hexdigest()
    
    def _cleanup_old_signatures(self, window_seconds: int):
        """Clean up old event signatures outside replay window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        
        # Remove old signatures
        old_signatures = [
            sig for sig, timestamp in self.processed_signatures.items()
            if timestamp < cutoff_time
        ]
        
        for sig in old_signatures:
            del self.processed_signatures[sig]
        
        if old_signatures:
            self.logger.debug(f"🧹 Cleaned up {len(old_signatures)} old event signatures")
    
    def _is_event_allowed(self, webhook_event: WebhookEvent, config: WebhookConfig) -> bool:
        """Check if event type is allowed"""
        # Check blocked events first
        if webhook_event.event_type in config.blocked_event_types:
            return False
        
        # If allowed list is empty, allow all (except blocked)
        if not config.allowed_event_types:
            return True
        
        # Check if event type is in allowed list
        return webhook_event.event_type in config.allowed_event_types
    
    async def _normalize_event(self, webhook_event: WebhookEvent, config: WebhookConfig) -> Dict[str, Any]:
        """Normalize webhook event to canonical internal format"""
        try:
            # Base normalized event structure
            normalized = {
                "event_id": webhook_event.event_id,
                "source": webhook_event.source.value,
                "event_type": webhook_event.event_type,
                "tenant_id": webhook_event.tenant_id,
                "timestamp": webhook_event.received_at.isoformat(),
                "version": "1.0",
                "data": {}
            }
            
            # Source-specific normalization
            if webhook_event.source == WebhookSource.SALESFORCE:
                normalized["data"] = self._normalize_salesforce_event(webhook_event.payload)
            elif webhook_event.source == WebhookSource.STRIPE:
                normalized["data"] = self._normalize_stripe_event(webhook_event.payload)
            elif webhook_event.source == WebhookSource.HUBSPOT:
                normalized["data"] = self._normalize_hubspot_event(webhook_event.payload)
            else:
                # Generic normalization
                normalized["data"] = webhook_event.payload
            
            # Add governance metadata
            normalized["governance"] = {
                "signature_verified": webhook_event.signature_valid,
                "replay_protected": webhook_event.replay_check_passed,
                "processing_time": datetime.now(timezone.utc).isoformat()
            }
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"❌ Event normalization error: {e}")
            # Return basic normalized event on error
            return {
                "event_id": webhook_event.event_id,
                "source": webhook_event.source.value,
                "event_type": webhook_event.event_type,
                "tenant_id": webhook_event.tenant_id,
                "timestamp": webhook_event.received_at.isoformat(),
                "data": webhook_event.payload,
                "error": f"Normalization failed: {str(e)}"
            }
    
    def _normalize_salesforce_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Salesforce webhook payload"""
        return {
            "object_type": payload.get("sobject", {}).get("type"),
            "object_id": payload.get("sobject", {}).get("Id"),
            "action": payload.get("event", {}).get("type"),
            "fields_changed": payload.get("sobject", {}).get("fieldsChanged", []),
            "raw_data": payload
        }
    
    def _normalize_stripe_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Stripe webhook payload"""
        return {
            "object_type": payload.get("data", {}).get("object", {}).get("object"),
            "object_id": payload.get("data", {}).get("object", {}).get("id"),
            "action": payload.get("type"),
            "customer_id": payload.get("data", {}).get("object", {}).get("customer"),
            "raw_data": payload
        }
    
    def _normalize_hubspot_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize HubSpot webhook payload"""
        return {
            "object_type": payload.get("objectType"),
            "object_id": payload.get("objectId"),
            "action": payload.get("subscriptionType"),
            "portal_id": payload.get("portalId"),
            "raw_data": payload
        }
    
    async def _capture_webhook_evidence(self, webhook_event: WebhookEvent):
        """Capture evidence of webhook processing"""
        if self.evidence_service:
            try:
                await self.evidence_service.capture_evidence({
                    "evidence_id": str(uuid.uuid4()),
                    "type": "webhook_received",
                    "webhook_event": webhook_event.to_dict(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                self.logger.error(f"❌ Failed to capture webhook evidence: {e}")
    
    async def _record_delivery_attempt(self, webhook_event: WebhookEvent, success: bool):
        """Record webhook delivery attempt"""
        attempt = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "status": webhook_event.status.value,
            "error": webhook_event.error_message if not success else None
        }
        
        if webhook_event.event_id not in self.delivery_attempts:
            self.delivery_attempts[webhook_event.event_id] = []
        
        self.delivery_attempts[webhook_event.event_id].append(attempt)
        
        # Keep only recent attempts (last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.delivery_attempts[webhook_event.event_id] = [
            attempt for attempt in self.delivery_attempts[webhook_event.event_id]
            if datetime.fromisoformat(attempt["timestamp"].replace('Z', '+00:00')) > cutoff_time
        ]
    
    async def _log_webhook_event(self, webhook_event: WebhookEvent):
        """Log webhook event for audit and monitoring"""
        log_level = logging.INFO if webhook_event.status == WebhookStatus.PROCESSED else logging.WARNING
        
        self.logger.log(
            log_level,
            f"🎣 Webhook {webhook_event.status.value}: {webhook_event.event_id} "
            f"({webhook_event.source.value}/{webhook_event.event_type})"
        )
    
    def add_webhook_config(self, config: WebhookConfig):
        """Add webhook configuration for a source/tenant"""
        config_key = f"{config.source.value}_{config.tenant_id}"
        self.webhook_configs[config_key] = config
        self.logger.info(f"➕ Added webhook config: {config_key}")
    
    def get_webhook_metrics(self) -> Dict[str, Any]:
        """Get webhook processing metrics"""
        return {
            "metrics": self.metrics.copy(),
            "active_configs": len(self.webhook_configs),
            "processed_signatures": len(self.processed_signatures),
            "delivery_attempts": len(self.delivery_attempts),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform webhook receiver health check"""
        try:
            return {
                "status": "healthy",
                "metrics": self.metrics.copy(),
                "configurations": len(self.webhook_configs),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global webhook receiver instance
webhook_receiver = WebhookReceiver()
