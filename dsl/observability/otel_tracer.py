#!/usr/bin/env python3
"""
OpenTelemetry Tracer for RBA Workflows
======================================

Implements Tasks 6.7-T02 to 6.7-T06:
- OpenTelemetry tracing in Orchestrator, Compiler, Runtime
- Trace schema with workflow, tenant, evidence links
- Governance-aware tracing with policy context

Features:
- End-to-end workflow tracing with step-level granularity
- Tenant-aware trace context propagation
- Evidence pack linkage for audit trails
- Policy compliance tracking in traces
- Industry overlay trace enrichment
- Performance metrics integration
"""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.propagate import inject, extract
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Fallback implementations
    class MockSpan:
        def set_attribute(self, key, value): pass
        def set_status(self, status): pass
        def record_exception(self, exception): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class MockTracer:
        def start_span(self, name, **kwargs): return MockSpan()

logger = logging.getLogger(__name__)

@dataclass
class TraceContext:
    """Enhanced trace context with governance metadata"""
    trace_id: str
    span_id: str
    tenant_id: int
    user_id: str
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    industry_code: str = 'SaaS'
    compliance_frameworks: List[str] = None
    evidence_pack_id: Optional[str] = None
    policy_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.compliance_frameworks is None:
            self.compliance_frameworks = ['SOX_SAAS', 'GDPR_SAAS']
        if self.policy_context is None:
            self.policy_context = {}

class RBATracer:
    """
    Enhanced OpenTelemetry tracer for RBA workflows
    
    Implements governance-aware tracing with:
    - Multi-tenant context propagation
    - Evidence pack linkage
    - Policy compliance tracking
    - Industry overlay enrichment
    - Performance metrics integration
    """
    
    def __init__(self, service_name: str = "rba-service", 
                 otlp_endpoint: Optional[str] = None,
                 enable_console_export: bool = True):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.enable_console_export = enable_console_export
        
        if OTEL_AVAILABLE:
            self._setup_tracer()
        else:
            logger.warning("⚠️ OpenTelemetry not available, using mock tracer")
            self.tracer = MockTracer()
    
    def _setup_tracer(self):
        """Setup OpenTelemetry tracer with proper configuration"""
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "service.namespace": "rba",
                "deployment.environment": "production"
            })
            
            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Add OTLP exporter if endpoint provided
            if self.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            
            # Add console exporter for development
            if self.enable_console_export:
                from opentelemetry.exporter.console import ConsoleSpanExporter
                console_exporter = ConsoleSpanExporter()
                tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(self.service_name)
            
            logger.info(f"✅ OpenTelemetry tracer initialized for {self.service_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup OpenTelemetry tracer: {e}")
            self.tracer = MockTracer()
    
    @asynccontextmanager
    async def trace_workflow_execution(self, 
                                     workflow_id: str,
                                     execution_id: str,
                                     trace_context: TraceContext,
                                     operation_name: str = "workflow_execution"):
        """
        Trace complete workflow execution with governance context (Task 6.7-T02)
        """
        span_name = f"rba.workflow.{operation_name}"
        
        with self.tracer.start_span(span_name) as span:
            try:
                # Set standard attributes
                span.set_attribute("rba.workflow.id", workflow_id)
                span.set_attribute("rba.execution.id", execution_id)
                span.set_attribute("rba.tenant.id", str(trace_context.tenant_id))
                span.set_attribute("rba.user.id", trace_context.user_id)
                span.set_attribute("rba.industry.code", trace_context.industry_code)
                
                # Set compliance framework attributes
                for i, framework in enumerate(trace_context.compliance_frameworks):
                    span.set_attribute(f"rba.compliance.framework.{i}", framework)
                
                # Set evidence pack linkage (Task 6.7-T06)
                if trace_context.evidence_pack_id:
                    span.set_attribute("rba.evidence.pack_id", trace_context.evidence_pack_id)
                
                # Set policy context attributes
                if trace_context.policy_context:
                    for key, value in trace_context.policy_context.items():
                        span.set_attribute(f"rba.policy.{key}", str(value))
                
                # Add trace metadata
                span.set_attribute("rba.trace.timestamp", datetime.utcnow().isoformat())
                span.set_attribute("rba.trace.service", self.service_name)
                
                yield span
                
                # Mark as successful
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                # Record exception and mark as error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @asynccontextmanager
    async def trace_compilation_step(self, 
                                   step_name: str,
                                   trace_context: TraceContext,
                                   step_metadata: Dict[str, Any] = None):
        """
        Trace compilation steps with governance context (Task 6.7-T03)
        """
        span_name = f"rba.compiler.{step_name}"
        
        with self.tracer.start_span(span_name) as span:
            try:
                # Set compilation-specific attributes
                span.set_attribute("rba.compiler.step", step_name)
                span.set_attribute("rba.tenant.id", str(trace_context.tenant_id))
                span.set_attribute("rba.workflow.id", trace_context.workflow_id or "unknown")
                
                # Add step metadata
                if step_metadata:
                    for key, value in step_metadata.items():
                        span.set_attribute(f"rba.compiler.{key}", str(value))
                
                # Add governance context
                span.set_attribute("rba.governance.active", "true")
                span.set_attribute("rba.compliance.required", "true")
                
                yield span
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @asynccontextmanager
    async def trace_runtime_step(self, 
                               step_name: str,
                               step_id: str,
                               trace_context: TraceContext,
                               step_input: Dict[str, Any] = None):
        """
        Trace runtime execution steps with step-level granularity (Task 6.7-T04)
        """
        span_name = f"rba.runtime.{step_name}"
        
        with self.tracer.start_span(span_name) as span:
            try:
                # Set runtime-specific attributes
                span.set_attribute("rba.runtime.step", step_name)
                span.set_attribute("rba.runtime.step_id", step_id)
                span.set_attribute("rba.tenant.id", str(trace_context.tenant_id))
                span.set_attribute("rba.execution.id", trace_context.execution_id or "unknown")
                
                # Add input metadata (sanitized)
                if step_input:
                    input_size = len(json.dumps(step_input))
                    span.set_attribute("rba.runtime.input_size_bytes", input_size)
                    span.set_attribute("rba.runtime.input_keys", list(step_input.keys()))
                
                # Add performance tracking
                start_time = datetime.utcnow()
                span.set_attribute("rba.runtime.start_time", start_time.isoformat())
                
                yield span
                
                # Calculate execution time
                end_time = datetime.utcnow()
                execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
                span.set_attribute("rba.runtime.execution_time_ms", execution_time_ms)
                span.set_attribute("rba.runtime.end_time", end_time.isoformat())
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @asynccontextmanager
    async def trace_governance_check(self, 
                                   check_type: str,
                                   trace_context: TraceContext,
                                   policy_data: Dict[str, Any] = None):
        """
        Trace governance and policy checks with compliance context
        """
        span_name = f"rba.governance.{check_type}"
        
        with self.tracer.start_span(span_name) as span:
            try:
                # Set governance-specific attributes
                span.set_attribute("rba.governance.check_type", check_type)
                span.set_attribute("rba.tenant.id", str(trace_context.tenant_id))
                span.set_attribute("rba.governance.timestamp", datetime.utcnow().isoformat())
                
                # Add policy data
                if policy_data:
                    span.set_attribute("rba.governance.policies_count", len(policy_data.get('policies', [])))
                    span.set_attribute("rba.governance.compliance_status", policy_data.get('status', 'unknown'))
                
                # Add compliance framework context
                for framework in trace_context.compliance_frameworks:
                    span.set_attribute(f"rba.governance.framework.{framework}", "active")
                
                yield span
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def create_trace_context(self, 
                           tenant_id: int,
                           user_id: str,
                           workflow_id: Optional[str] = None,
                           execution_id: Optional[str] = None,
                           **kwargs) -> TraceContext:
        """
        Create trace context with governance metadata
        """
        # Generate trace IDs if not provided
        trace_id = kwargs.get('trace_id', str(uuid.uuid4()))
        span_id = kwargs.get('span_id', str(uuid.uuid4()))
        
        return TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workflow_id=workflow_id,
            execution_id=execution_id,
            industry_code=kwargs.get('industry_code', 'SaaS'),
            compliance_frameworks=kwargs.get('compliance_frameworks', ['SOX_SAAS', 'GDPR_SAAS']),
            evidence_pack_id=kwargs.get('evidence_pack_id'),
            policy_context=kwargs.get('policy_context', {})
        )
    
    def inject_trace_context(self, headers: Dict[str, str], trace_context: TraceContext):
        """
        Inject trace context into HTTP headers for distributed tracing
        """
        if OTEL_AVAILABLE:
            # Add custom headers for governance context
            headers['x-rba-tenant-id'] = str(trace_context.tenant_id)
            headers['x-rba-user-id'] = trace_context.user_id
            headers['x-rba-industry-code'] = trace_context.industry_code
            
            if trace_context.workflow_id:
                headers['x-rba-workflow-id'] = trace_context.workflow_id
            if trace_context.execution_id:
                headers['x-rba-execution-id'] = trace_context.execution_id
            if trace_context.evidence_pack_id:
                headers['x-rba-evidence-pack-id'] = trace_context.evidence_pack_id
            
            # Inject OpenTelemetry trace context
            inject(headers)
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """
        Extract trace context from HTTP headers
        """
        try:
            if OTEL_AVAILABLE:
                # Extract OpenTelemetry context
                context = extract(headers)
                
                # Extract custom governance headers
                tenant_id = int(headers.get('x-rba-tenant-id', 0))
                user_id = headers.get('x-rba-user-id', 'unknown')
                
                if tenant_id and user_id != 'unknown':
                    return TraceContext(
                        trace_id=str(uuid.uuid4()),  # Will be overridden by OTEL
                        span_id=str(uuid.uuid4()),   # Will be overridden by OTEL
                        tenant_id=tenant_id,
                        user_id=user_id,
                        workflow_id=headers.get('x-rba-workflow-id'),
                        execution_id=headers.get('x-rba-execution-id'),
                        industry_code=headers.get('x-rba-industry-code', 'SaaS'),
                        evidence_pack_id=headers.get('x-rba-evidence-pack-id')
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract trace context: {e}")
            return None

# Global tracer instance
_global_tracer: Optional[RBATracer] = None

def create_tracer(service_name: str = "rba-service", 
                 otlp_endpoint: Optional[str] = None,
                 enable_console_export: bool = True) -> RBATracer:
    """
    Create or get global RBA tracer instance
    """
    global _global_tracer
    
    if _global_tracer is None:
        _global_tracer = RBATracer(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
            enable_console_export=enable_console_export
        )
    
    return _global_tracer

def get_tracer() -> RBATracer:
    """
    Get the global tracer instance
    """
    global _global_tracer
    
    if _global_tracer is None:
        _global_tracer = create_tracer()
    
    return _global_tracer
