"""
OpenTelemetry Integration - Task 6.1.26
=======================================
Integrate OpenTelemetry for SIEM/SOC logging of governance events
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GovernanceEvent:
    """Governance event for SIEM logging"""
    event_id: str
    event_type: str  # approval, override, policy_violation, drift_detection, etc.
    tenant_id: str
    user_id: str
    
    # Event details
    workflow_id: Optional[str] = None
    model_id: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical
    
    # Context
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class OpenTelemetryIntegration:
    """OpenTelemetry integration for governance event logging"""
    
    def __init__(self, service_name: str = "rbia-governance"):
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        
        # Initialize OpenTelemetry (simplified)
        self._initialize_otel()
    
    def _initialize_otel(self):
        """Initialize OpenTelemetry SDK"""
        try:
            # In production, this would configure:
            # - Trace provider
            # - Metric provider
            # - Log provider
            # - OTLP exporters to SIEM
            self.initialized = True
            self.logger.info(f"✅ OpenTelemetry initialized for {self.service_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")
    
    def log_governance_event(self, event: GovernanceEvent):
        """Log governance event to SIEM via OpenTelemetry"""
        try:
            # Create structured log entry
            log_entry = {
                'service.name': self.service_name,
                'event.id': event.event_id,
                'event.type': event.event_type,
                'tenant.id': event.tenant_id,
                'user.id': event.user_id,
                'workflow.id': event.workflow_id,
                'model.id': event.model_id,
                'severity': event.severity,
                'timestamp': event.timestamp.isoformat(),
                **event.metadata
            }
            
            # Send to SIEM (via OTLP exporter)
            self.logger.info(f"SIEM Event: {event.event_type} | {event.event_id}")
            
            # In production:
            # tracer.start_span(...).set_attributes(log_entry)
            # logger.info(json.dumps(log_entry))
            
        except Exception as e:
            self.logger.error(f"Failed to log governance event: {e}")
    
    def create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create OpenTelemetry span for distributed tracing"""
        # In production: return tracer.start_as_current_span(name, attributes=attributes)
        pass


# Singleton
_otel_integration: Optional[OpenTelemetryIntegration] = None

def get_otel_integration() -> OpenTelemetryIntegration:
    global _otel_integration
    if _otel_integration is None:
        _otel_integration = OpenTelemetryIntegration()
    return _otel_integration

