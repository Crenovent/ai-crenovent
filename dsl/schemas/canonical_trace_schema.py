# Canonical RBA Trace Schema
# Tasks 7.1-T01 to T15: Canonical trace schema, governance principles, versioning

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TraceEventType(Enum):
    """Types of trace events"""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_FAILED = "workflow_failed"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    STEP_FAILED = "step_failed"
    STEP_RETRY = "step_retry"
    GOVERNANCE_CHECK = "governance_check"
    POLICY_VIOLATION = "policy_violation"
    OVERRIDE_APPLIED = "override_applied"
    EVIDENCE_GENERATED = "evidence_generated"
    ESCALATION_TRIGGERED = "escalation_triggered"

class ExecutionStatus(Enum):
    """Execution status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class ActorType(Enum):
    """Types of actors in RBA execution"""
    USER = "user"
    SYSTEM = "system"
    SERVICE = "service"
    ML_MODEL = "ml_model"
    AGENT = "agent"
    SCHEDULER = "scheduler"

@dataclass
class TraceActor:
    """Actor information in trace"""
    actor_id: str
    actor_type: ActorType
    actor_name: str  # Anonymized for cross-tenant safety
    tenant_id: int
    session_id: Optional[str] = None
    ip_address: Optional[str] = None  # Hashed for privacy
    user_agent: Optional[str] = None

@dataclass
class TraceContext:
    """Execution context information"""
    tenant_id: int
    user_id: str
    session_id: str
    correlation_id: str
    parent_execution_id: Optional[str] = None
    industry_code: str = "SaaS"
    data_residency: str = "US"
    compliance_frameworks: List[str] = field(default_factory=list)
    feature_flags: Dict[str, bool] = field(default_factory=dict)

@dataclass
class TraceInput:
    """Input data for trace"""
    input_data: Dict[str, Any]
    input_hash: str
    input_schema_version: str
    sensitive_fields: List[str] = field(default_factory=list)
    pii_redacted: bool = False

@dataclass
class TraceOutput:
    """Output data for trace"""
    output_data: Dict[str, Any]
    output_hash: str
    output_schema_version: str
    sensitive_fields: List[str] = field(default_factory=list)
    pii_redacted: bool = False

@dataclass
class TraceError:
    """Error information in trace"""
    error_code: str
    error_message: str
    error_type: str
    stack_trace: Optional[str] = None
    retry_count: int = 0
    is_retryable: bool = False
    root_cause: Optional[str] = None

@dataclass
class GovernanceEvent:
    """Governance-related event in trace"""
    event_id: str
    event_type: str
    policy_id: str
    policy_version: str
    decision: str  # allowed, denied, warning
    reason: str
    evidence_refs: List[str] = field(default_factory=list)
    override_id: Optional[str] = None
    approver_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class StepTrace:
    """Individual step execution trace"""
    step_id: str
    step_name: str
    step_type: str
    execution_id: str
    parent_step_id: Optional[str] = None
    
    # Execution details
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_ms: int = 0
    
    # Data
    inputs: Optional[TraceInput] = None
    outputs: Optional[TraceOutput] = None
    error: Optional[TraceError] = None
    
    # Governance
    governance_events: List[GovernanceEvent] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    
    # Performance
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    retry_count: int = 0
    timeout_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class WorkflowTrace:
    """Complete workflow execution trace"""
    # Core identifiers
    trace_id: str
    workflow_id: str
    execution_id: str
    workflow_version: str
    
    # Context
    context: TraceContext
    actor: TraceActor
    
    # Execution details
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_ms: int = 0
    
    # Data
    inputs: Optional[TraceInput] = None
    outputs: Optional[TraceOutput] = None
    error: Optional[TraceError] = None
    
    # Steps
    steps: List[StepTrace] = field(default_factory=list)
    
    # Governance
    governance_events: List[GovernanceEvent] = field(default_factory=list)
    policy_pack_version: str = "default_v1.0"
    compliance_score: float = 1.0
    trust_score: float = 1.0
    
    # Evidence and audit
    evidence_pack_id: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    override_ledger_refs: List[str] = field(default_factory=list)
    
    # Performance and resources
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    sla_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert trace to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTrace':
        """Create trace from dictionary"""
        # Handle enum conversions
        if 'status' in data:
            data['status'] = ExecutionStatus(data['status'])
        
        if 'actor' in data and 'actor_type' in data['actor']:
            data['actor']['actor_type'] = ActorType(data['actor']['actor_type'])
        
        # Handle nested objects
        if 'context' in data:
            data['context'] = TraceContext(**data['context'])
        
        if 'actor' in data:
            data['actor'] = TraceActor(**data['actor'])
        
        if 'inputs' in data and data['inputs']:
            data['inputs'] = TraceInput(**data['inputs'])
        
        if 'outputs' in data and data['outputs']:
            data['outputs'] = TraceOutput(**data['outputs'])
        
        if 'error' in data and data['error']:
            data['error'] = TraceError(**data['error'])
        
        # Handle steps
        if 'steps' in data:
            steps = []
            for step_data in data['steps']:
                if 'status' in step_data:
                    step_data['status'] = ExecutionStatus(step_data['status'])
                
                # Handle nested objects in steps
                if 'inputs' in step_data and step_data['inputs']:
                    step_data['inputs'] = TraceInput(**step_data['inputs'])
                
                if 'outputs' in step_data and step_data['outputs']:
                    step_data['outputs'] = TraceOutput(**step_data['outputs'])
                
                if 'error' in step_data and step_data['error']:
                    step_data['error'] = TraceError(**step_data['error'])
                
                # Handle governance events
                if 'governance_events' in step_data:
                    governance_events = []
                    for event_data in step_data['governance_events']:
                        governance_events.append(GovernanceEvent(**event_data))
                    step_data['governance_events'] = governance_events
                
                steps.append(StepTrace(**step_data))
            data['steps'] = steps
        
        # Handle governance events
        if 'governance_events' in data:
            governance_events = []
            for event_data in data['governance_events']:
                governance_events.append(GovernanceEvent(**event_data))
            data['governance_events'] = governance_events
        
        return cls(**data)

class TraceSchemaValidator:
    """
    Validator for canonical RBA trace schema
    Tasks 7.1-T03, T04: Schema validation and governance
    """
    
    def __init__(self):
        self.schema_version = "1.0.0"
        self.required_fields = [
            'trace_id', 'workflow_id', 'execution_id', 'workflow_version',
            'context', 'actor', 'schema_version'
        ]
        self.governance_required_fields = [
            'policy_pack_version', 'compliance_score', 'trust_score'
        ]
    
    def validate_trace(self, trace: WorkflowTrace) -> Dict[str, Any]:
        """Validate trace against canonical schema"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'governance_compliant': True
        }
        
        # Check required fields
        trace_dict = trace.to_dict()
        for field in self.required_fields:
            if field not in trace_dict or trace_dict[field] is None:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['valid'] = False
        
        # Check governance fields
        for field in self.governance_required_fields:
            if field not in trace_dict or trace_dict[field] is None:
                validation_result['warnings'].append(f"Missing governance field: {field}")
                validation_result['governance_compliant'] = False
        
        # Validate schema version
        if trace.schema_version != self.schema_version:
            validation_result['warnings'].append(
                f"Schema version mismatch: expected {self.schema_version}, got {trace.schema_version}"
            )
        
        # Validate tenant context
        if not trace.context.tenant_id:
            validation_result['errors'].append("Missing tenant_id in context")
            validation_result['valid'] = False
        
        # Validate execution consistency
        if trace.status == ExecutionStatus.COMPLETED and not trace.completed_at:
            validation_result['errors'].append("Completed workflow missing completed_at timestamp")
            validation_result['valid'] = False
        
        if trace.status == ExecutionStatus.FAILED and not trace.error:
            validation_result['errors'].append("Failed workflow missing error information")
            validation_result['valid'] = False
        
        # Validate steps
        for i, step in enumerate(trace.steps):
            step_errors = self._validate_step(step, i)
            validation_result['errors'].extend(step_errors)
            if step_errors:
                validation_result['valid'] = False
        
        # Validate governance events
        for i, event in enumerate(trace.governance_events):
            event_errors = self._validate_governance_event(event, i)
            validation_result['errors'].extend(event_errors)
            if event_errors:
                validation_result['valid'] = False
        
        return validation_result
    
    def _validate_step(self, step: StepTrace, step_index: int) -> List[str]:
        """Validate individual step"""
        errors = []
        
        if not step.step_id:
            errors.append(f"Step {step_index}: Missing step_id")
        
        if not step.step_type:
            errors.append(f"Step {step_index}: Missing step_type")
        
        if step.status == ExecutionStatus.COMPLETED and not step.completed_at:
            errors.append(f"Step {step_index}: Completed step missing completed_at")
        
        if step.status == ExecutionStatus.FAILED and not step.error:
            errors.append(f"Step {step_index}: Failed step missing error information")
        
        return errors
    
    def _validate_governance_event(self, event: GovernanceEvent, event_index: int) -> List[str]:
        """Validate governance event"""
        errors = []
        
        if not event.event_id:
            errors.append(f"Governance event {event_index}: Missing event_id")
        
        if not event.policy_id:
            errors.append(f"Governance event {event_index}: Missing policy_id")
        
        if not event.decision:
            errors.append(f"Governance event {event_index}: Missing decision")
        
        return errors

class TraceBuilder:
    """
    Builder for creating canonical RBA traces
    Tasks 7.1-T05, T06: Trace creation and workflow integration
    """
    
    def __init__(self):
        self.validator = TraceSchemaValidator()
    
    def create_workflow_trace(
        self,
        workflow_id: str,
        execution_id: str,
        workflow_version: str,
        tenant_id: int,
        user_id: str,
        actor_name: str = "system"
    ) -> WorkflowTrace:
        """Create new workflow trace"""
        
        trace_id = str(uuid.uuid4())
        
        context = TraceContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4())
        )
        
        actor = TraceActor(
            actor_id=user_id,
            actor_type=ActorType.USER if user_id != "system" else ActorType.SYSTEM,
            actor_name=actor_name,
            tenant_id=tenant_id
        )
        
        trace = WorkflowTrace(
            trace_id=trace_id,
            workflow_id=workflow_id,
            execution_id=execution_id,
            workflow_version=workflow_version,
            context=context,
            actor=actor
        )
        
        return trace
    
    def add_step_trace(
        self,
        workflow_trace: WorkflowTrace,
        step_id: str,
        step_name: str,
        step_type: str
    ) -> StepTrace:
        """Add step trace to workflow"""
        
        step_trace = StepTrace(
            step_id=step_id,
            step_name=step_name,
            step_type=step_type,
            execution_id=workflow_trace.execution_id
        )
        
        workflow_trace.steps.append(step_trace)
        return step_trace
    
    def start_workflow(self, trace: WorkflowTrace) -> None:
        """Mark workflow as started"""
        trace.status = ExecutionStatus.RUNNING
        trace.started_at = datetime.now(timezone.utc).isoformat()
    
    def complete_workflow(
        self,
        trace: WorkflowTrace,
        outputs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark workflow as completed"""
        trace.status = ExecutionStatus.COMPLETED
        trace.completed_at = datetime.now(timezone.utc).isoformat()
        
        if trace.started_at:
            start_time = datetime.fromisoformat(trace.started_at.replace('Z', '+00:00'))
            end_time = datetime.now(timezone.utc)
            trace.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        if outputs:
            trace.outputs = TraceOutput(
                output_data=outputs,
                output_hash=self._calculate_hash(outputs),
                output_schema_version="1.0.0"
            )
    
    def fail_workflow(
        self,
        trace: WorkflowTrace,
        error_code: str,
        error_message: str,
        error_type: str = "execution_error"
    ) -> None:
        """Mark workflow as failed"""
        trace.status = ExecutionStatus.FAILED
        trace.completed_at = datetime.now(timezone.utc).isoformat()
        
        if trace.started_at:
            start_time = datetime.fromisoformat(trace.started_at.replace('Z', '+00:00'))
            end_time = datetime.now(timezone.utc)
            trace.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        trace.error = TraceError(
            error_code=error_code,
            error_message=error_message,
            error_type=error_type
        )
    
    def start_step(self, step_trace: StepTrace) -> None:
        """Mark step as started"""
        step_trace.status = ExecutionStatus.RUNNING
        step_trace.started_at = datetime.now(timezone.utc).isoformat()
    
    def complete_step(
        self,
        step_trace: StepTrace,
        outputs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark step as completed"""
        step_trace.status = ExecutionStatus.COMPLETED
        step_trace.completed_at = datetime.now(timezone.utc).isoformat()
        
        if step_trace.started_at:
            start_time = datetime.fromisoformat(step_trace.started_at.replace('Z', '+00:00'))
            end_time = datetime.now(timezone.utc)
            step_trace.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        if outputs:
            step_trace.outputs = TraceOutput(
                output_data=outputs,
                output_hash=self._calculate_hash(outputs),
                output_schema_version="1.0.0"
            )
    
    def fail_step(
        self,
        step_trace: StepTrace,
        error_code: str,
        error_message: str,
        error_type: str = "step_error"
    ) -> None:
        """Mark step as failed"""
        step_trace.status = ExecutionStatus.FAILED
        step_trace.completed_at = datetime.now(timezone.utc).isoformat()
        
        if step_trace.started_at:
            start_time = datetime.fromisoformat(step_trace.started_at.replace('Z', '+00:00'))
            end_time = datetime.now(timezone.utc)
            step_trace.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        step_trace.error = TraceError(
            error_code=error_code,
            error_message=error_message,
            error_type=error_type
        )
    
    def add_governance_event(
        self,
        trace: WorkflowTrace,
        event_type: str,
        policy_id: str,
        policy_version: str,
        decision: str,
        reason: str,
        step_id: Optional[str] = None
    ) -> GovernanceEvent:
        """Add governance event to trace"""
        
        event = GovernanceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            policy_id=policy_id,
            policy_version=policy_version,
            decision=decision,
            reason=reason
        )
        
        if step_id:
            # Add to specific step
            for step in trace.steps:
                if step.step_id == step_id:
                    step.governance_events.append(event)
                    break
        else:
            # Add to workflow level
            trace.governance_events.append(event)
        
        return event
    
    def validate_and_finalize(self, trace: WorkflowTrace) -> Dict[str, Any]:
        """Validate trace and prepare for storage"""
        
        # Update timestamps
        trace.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Validate trace
        validation_result = self.validator.validate_trace(trace)
        
        if validation_result['valid']:
            logger.info(f"Trace validated successfully: {trace.trace_id}")
        else:
            logger.error(f"Trace validation failed: {trace.trace_id} - {validation_result['errors']}")
        
        return {
            'trace': trace,
            'validation': validation_result,
            'serialized': trace.to_dict()
        }
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash for data"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

class TraceAggregator:
    """
    Aggregator for trace analytics and KG preparation
    Tasks 7.1-T07, T08: Trace aggregation and analytics
    """
    
    def __init__(self):
        self.aggregation_rules = {
            'workflow_success_rate': self._calculate_success_rate,
            'average_execution_time': self._calculate_avg_execution_time,
            'step_performance': self._calculate_step_performance,
            'governance_compliance': self._calculate_governance_compliance,
            'error_patterns': self._analyze_error_patterns
        }
    
    def aggregate_traces(
        self,
        traces: List[WorkflowTrace],
        aggregation_type: str = 'workflow_summary'
    ) -> Dict[str, Any]:
        """Aggregate multiple traces for analytics"""
        
        if not traces:
            return {'error': 'No traces provided'}
        
        aggregation = {
            'total_traces': len(traces),
            'aggregation_type': aggregation_type,
            'time_range': {
                'start': min(trace.started_at for trace in traces if trace.started_at),
                'end': max(trace.completed_at for trace in traces if trace.completed_at)
            },
            'tenant_breakdown': self._breakdown_by_tenant(traces),
            'workflow_breakdown': self._breakdown_by_workflow(traces),
            'metrics': {}
        }
        
        # Apply aggregation rules
        for metric_name, aggregation_func in self.aggregation_rules.items():
            try:
                aggregation['metrics'][metric_name] = aggregation_func(traces)
            except Exception as e:
                logger.error(f"Failed to calculate {metric_name}: {e}")
                aggregation['metrics'][metric_name] = None
        
        return aggregation
    
    def _calculate_success_rate(self, traces: List[WorkflowTrace]) -> float:
        """Calculate workflow success rate"""
        if not traces:
            return 0.0
        
        successful = sum(1 for trace in traces if trace.status == ExecutionStatus.COMPLETED)
        return (successful / len(traces)) * 100
    
    def _calculate_avg_execution_time(self, traces: List[WorkflowTrace]) -> float:
        """Calculate average execution time"""
        execution_times = [trace.execution_time_ms for trace in traces if trace.execution_time_ms > 0]
        
        if not execution_times:
            return 0.0
        
        return sum(execution_times) / len(execution_times)
    
    def _calculate_step_performance(self, traces: List[WorkflowTrace]) -> Dict[str, Any]:
        """Calculate step-level performance metrics"""
        step_metrics = {}
        
        for trace in traces:
            for step in trace.steps:
                step_type = step.step_type
                
                if step_type not in step_metrics:
                    step_metrics[step_type] = {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'total_time_ms': 0,
                        'error_count': 0
                    }
                
                step_metrics[step_type]['total_executions'] += 1
                
                if step.status == ExecutionStatus.COMPLETED:
                    step_metrics[step_type]['successful_executions'] += 1
                elif step.status == ExecutionStatus.FAILED:
                    step_metrics[step_type]['error_count'] += 1
                
                if step.execution_time_ms > 0:
                    step_metrics[step_type]['total_time_ms'] += step.execution_time_ms
        
        # Calculate derived metrics
        for step_type, metrics in step_metrics.items():
            if metrics['total_executions'] > 0:
                metrics['success_rate'] = (metrics['successful_executions'] / metrics['total_executions']) * 100
                metrics['error_rate'] = (metrics['error_count'] / metrics['total_executions']) * 100
                metrics['avg_execution_time_ms'] = metrics['total_time_ms'] / metrics['total_executions']
        
        return step_metrics
    
    def _calculate_governance_compliance(self, traces: List[WorkflowTrace]) -> Dict[str, Any]:
        """Calculate governance compliance metrics"""
        total_events = 0
        allowed_events = 0
        denied_events = 0
        warning_events = 0
        
        for trace in traces:
            for event in trace.governance_events:
                total_events += 1
                if event.decision == 'allowed':
                    allowed_events += 1
                elif event.decision == 'denied':
                    denied_events += 1
                elif event.decision == 'warning':
                    warning_events += 1
            
            for step in trace.steps:
                for event in step.governance_events:
                    total_events += 1
                    if event.decision == 'allowed':
                        allowed_events += 1
                    elif event.decision == 'denied':
                        denied_events += 1
                    elif event.decision == 'warning':
                        warning_events += 1
        
        if total_events == 0:
            return {'compliance_rate': 100.0, 'total_events': 0}
        
        return {
            'total_events': total_events,
            'allowed_events': allowed_events,
            'denied_events': denied_events,
            'warning_events': warning_events,
            'compliance_rate': (allowed_events / total_events) * 100,
            'denial_rate': (denied_events / total_events) * 100,
            'warning_rate': (warning_events / total_events) * 100
        }
    
    def _analyze_error_patterns(self, traces: List[WorkflowTrace]) -> Dict[str, Any]:
        """Analyze error patterns across traces"""
        error_patterns = {}
        
        for trace in traces:
            if trace.error:
                error_type = trace.error.error_type
                if error_type not in error_patterns:
                    error_patterns[error_type] = {
                        'count': 0,
                        'error_codes': {},
                        'retryable_count': 0
                    }
                
                error_patterns[error_type]['count'] += 1
                
                error_code = trace.error.error_code
                if error_code not in error_patterns[error_type]['error_codes']:
                    error_patterns[error_type]['error_codes'][error_code] = 0
                error_patterns[error_type]['error_codes'][error_code] += 1
                
                if trace.error.is_retryable:
                    error_patterns[error_type]['retryable_count'] += 1
            
            # Analyze step errors
            for step in trace.steps:
                if step.error:
                    error_type = f"step_{step.error.error_type}"
                    if error_type not in error_patterns:
                        error_patterns[error_type] = {
                            'count': 0,
                            'error_codes': {},
                            'retryable_count': 0
                        }
                    
                    error_patterns[error_type]['count'] += 1
                    
                    error_code = step.error.error_code
                    if error_code not in error_patterns[error_type]['error_codes']:
                        error_patterns[error_type]['error_codes'][error_code] = 0
                    error_patterns[error_type]['error_codes'][error_code] += 1
                    
                    if step.error.is_retryable:
                        error_patterns[error_type]['retryable_count'] += 1
        
        return error_patterns
    
    def _breakdown_by_tenant(self, traces: List[WorkflowTrace]) -> Dict[str, int]:
        """Break down traces by tenant"""
        tenant_counts = {}
        
        for trace in traces:
            tenant_id = str(trace.context.tenant_id)
            if tenant_id not in tenant_counts:
                tenant_counts[tenant_id] = 0
            tenant_counts[tenant_id] += 1
        
        return tenant_counts
    
    def _breakdown_by_workflow(self, traces: List[WorkflowTrace]) -> Dict[str, int]:
        """Break down traces by workflow"""
        workflow_counts = {}
        
        for trace in traces:
            workflow_id = trace.workflow_id
            if workflow_id not in workflow_counts:
                workflow_counts[workflow_id] = 0
            workflow_counts[workflow_id] += 1
        
        return workflow_counts

# Global instances
trace_builder = TraceBuilder()
trace_aggregator = TraceAggregator()
trace_validator = TraceSchemaValidator()
