"""
Enhanced Observability Framework
===============================

Implements comprehensive observability for RBA workflows as specified in:
- Tasks 6.7-T01 to 6.7-T50: Observability Hooks
- Tasks 20.1-T01 to 20.1-T70: Execution Metrics & Monitoring

Key Features:
- OpenTelemetry tracing with governance context
- Multi-tenant metrics with proper labeling
- Structured logging with PII redaction
- Evidence-linked observability for audit trails
- Real-time SLA monitoring and alerting
- Compliance dashboards per industry
"""

from .otel_tracer import RBATracer, create_tracer, get_tracer, TraceContext
from .metrics_collector import MetricsCollector, RBAMetrics, get_metrics_collector

__all__ = [
    'RBATracer',
    'create_tracer',
    'get_tracer',
    'TraceContext',
    'MetricsCollector', 
    'RBAMetrics',
    'get_metrics_collector'
]
